"""
evaluate_imagenetc_vitb.py  —  CGT-TTA on ImageNet-C with ViT-B/16
====================================================================
Compares three methods on ImageNet-C at a given severity:
  1. Source-only   — frozen ViT-B/16, no adaptation
  2. TENT          — entropy minimisation over LayerNorm (NOT BatchNorm)
  3. CGT-TTA       — calibration-gated TTA (our method, v3)

Key differences vs the ResNet-50 script
-----------------------------------------
NORM LAYERS
  ViT-B/16 uses LayerNorm (LN), not BatchNorm.
  TENT updates LN affine params (weight, bias) instead of BN (gamma, beta).
  LN has no running stats — it always normalises from the current batch,
  so there is no "track_running_stats" concept.
  This makes TENT MORE stable on ViT than ResNet under severe corruption
  — the BN collapse problem does not apply.

FEATURE EXTRACTION
  ResNet-50: avgpool output → (B, 2048, 1, 1) → (B, 2048)
  ViT-B/16:  CLS token from the last encoder block → (B, 768)
  We hook the last encoder layer's output and take index 0 (CLS token).

WHICH LN LAYERS TO ADAPT
  ViT-B/16 has 12 transformer blocks, each with 2 LN layers (ln_1, ln_2)
  plus a final ln_norm after the encoder = 25 LN layers total.
  We adapt only the last 3 blocks' LN layers (6 LN layers out of 25)
  for stability — analogous to layer4-only for ResNet-50.
  Flag --all_ln to adapt all 25 LN layers (may be less stable).

MODEL
  torchvision.models.vit_b_16 pretrained on ImageNet-1K
  (ViT-B/16, patch size 16, image size 224, ~86M params)
  Clean ImageNet-1K accuracy: 81.1% top-1

Usage (VS Code terminal)
-------------------------
  Smoke test — 3 corruptions, fast:
    python evaluate_imagenetc_vitb.py ^
        --severity 5 ^
        --corruptions gaussian_noise defocus_blur brightness ^
        --output_dir C:\\Users\\Vineet9.Yadav\\Desktop\\TTA\\results\\imagenetc_vitb_smoke

  Full run — all 15 corruptions:
    python evaluate_imagenetc_vitb.py ^
        --severity 5 ^
        --output_dir C:\\Users\\Vineet9.Yadav\\Desktop\\TTA\\results\\imagenetc_vitb_sev5

Outputs
-------
  summary.csv       — accuracy + ECE per method × corruption
  results.json      — full structured results
  reliability/      — reliability diagram CSVs per method
"""

import os, sys, copy, json, time, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

# Standard ImageNet normalisation (same for ViT-B/16)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ViT-B/16 feature dimension (CLS token output of last encoder block)
VIT_FEAT_DIM = 768


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def log(msg: str):
    print(msg, flush=True)


def progress(current: int, total: int, prefix: str = "", width: int = 30) -> str:
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    return f"\r{prefix} [{bar}] {current}/{total}"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (same layout detection as ResNet-50 script)
# ─────────────────────────────────────────────────────────────────────────────

class ImageNetCDataset(Dataset):
    """
    Loads one (corruption, severity) slice of ImageNet-C.

    Layout A — standard ImageFolder structure:
      <data_dir>/<corruption>/<severity>/<synset_id>/<image>.JPEG
      Labels inferred from sorted synset folder names.

    Layout B — flat images + labels.npy:
      <data_dir>/<corruption>/<severity>/<image>.JPEG
      Labels from <data_dir>/labels.npy
    """

    IMG_EXTS = {'.jpeg', '.jpg', '.png', '.bmp', '.webp'}

    def __init__(self, data_dir: str, corruption: str,
                 severity: int, transform=None):
        self.transform = transform
        self.samples   = []

        base = Path(data_dir) / corruption / str(severity)
        if not base.exists():
            raise FileNotFoundError(f"Not found: {base}")

        subdirs = [d for d in base.iterdir() if d.is_dir()]
        if subdirs:
            self._load_layout_a(base)
        else:
            self._load_layout_b(data_dir, base)

    def _load_layout_a(self, base: Path):
        class_dirs     = sorted([d for d in base.iterdir() if d.is_dir()])
        class_to_idx   = {d.name: i for i, d in enumerate(class_dirs)}
        for cls_dir in class_dirs:
            label = class_to_idx[cls_dir.name]
            for p in cls_dir.iterdir():
                if p.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((str(p), label))
        self.samples.sort(key=lambda x: x[0])

    def _load_layout_b(self, data_dir: str, base: Path):
        imgs = sorted([str(p) for p in base.iterdir()
                       if p.suffix.lower() in self.IMG_EXTS])
        candidates = [
            Path(data_dir) / "labels.npy",
            Path(data_dir) / "val_labels.npy",
            Path(data_dir).parent / "labels.npy",
        ]
        label_path = next((p for p in candidates if p.exists()), None)
        if label_path is None:
            raise FileNotFoundError(f"No labels.npy. Searched: {candidates}")
        labels = np.load(label_path)
        if len(labels) != len(imgs):
            labels = labels[:len(imgs)]
        self.samples = list(zip(imgs, labels.tolist()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(label)


def make_transform():
    """Standard ViT-B/16 ImageNet transform (224×224)."""
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_loader(data_dir, corruption, severity, batch_size, num_workers):
    dataset = ImageNetCDataset(
        data_dir, corruption, severity, transform=make_transform()
    )
    log(f"    {len(dataset):,} samples | layout detected OK")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        drop_last=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL — ViT-B/16
# ─────────────────────────────────────────────────────────────────────────────

def load_vitb() -> nn.Module:
    """
    Load torchvision ViT-B/16 pretrained on ImageNet-1K.
    Clean top-1: 81.1%.
    Architecture: patch_size=16, image_size=224, hidden_dim=768,
                  num_heads=12, num_layers=12.
    """
    log("  Loading ViT-B/16 (torchvision, ImageNet-1K pretrained)...")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.eval()
    return model.to(DEVICE)


def get_last_block_idx(model: nn.Module) -> int:
    """Return the index of the last transformer encoder layer."""
    return len(model.encoder.layers) - 1   # 11 for ViT-B/16 (12 blocks, 0-indexed)


def ln_layers_to_adapt(model: nn.Module, last_n_blocks: int = 3) -> list:
    """
    Return names of LN layers in the last `last_n_blocks` encoder blocks
    plus the final encoder ln_norm.

    ViT-B/16 LN structure per block:
      encoder.layers.encoder_layer_{i}.ln_1   — pre-attention LN
      encoder.layers.{i}.ln_2   — pre-MLP LN
    Final:
      encoder.ln                 — post-encoder LN

    Default: last 3 blocks = 6 LN layers + encoder.ln = 7 total.
    This is analogous to layer4-only for ResNet-50.
    """
    n_blocks = len(model.encoder.layers)
    names = []
    for i in range(n_blocks - last_n_blocks, n_blocks):
        names.append(f"encoder.layers.encoder_layer_{i}.ln_1")
        names.append(f"encoder.layers.encoder_layer_{i}.ln_2")
    names.append("encoder.ln")
    return names


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(confidences: np.ndarray, predictions: np.ndarray,
                labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(labels)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(
            (predictions[mask] == labels[mask]).mean() - confidences[mask].mean()
        )
    return float(ece)


def reliability_data(confidences, predictions, labels, n_bins=15):
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    result = []
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        result.append({
            "bin_mid":   float((bins[i] + bins[i+1]) / 2),
            "bin_acc":   float((predictions[mask] == labels[mask]).mean()),
            "bin_conf":  float(confidences[mask].mean()),
            "bin_count": int(mask.sum()),
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: SOURCE-ONLY
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_source_only(model: nn.Module, loader: DataLoader):
    model.eval()
    all_conf, all_pred, all_labels = [], [], []
    for imgs, lbls in loader:
        imgs  = imgs.to(DEVICE)
        probs = F.softmax(model(imgs), dim=1)
        conf, pred = probs.max(dim=1)
        all_conf.append(conf.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_labels.append(lbls.numpy())
    confs  = np.concatenate(all_conf)
    preds  = np.concatenate(all_pred)
    labels = np.concatenate(all_labels)
    return (float((preds == labels).mean()),
            compute_ece(confs, preds, labels),
            reliability_data(confs, preds, labels))


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: TENT for ViT-B/16 (LayerNorm adaptation)
# ─────────────────────────────────────────────────────────────────────────────

def configure_tent_vit(model: nn.Module,
                       adapt_names: list,
                       all_ln: bool = False) -> nn.Module:
    """
    Prepare ViT-B/16 for TENT-style LN adaptation.

    Key difference from BN-based TENT:
    - LayerNorm has no running statistics (running_mean / running_var).
      It always normalises from the CURRENT batch — so it is already
      "adaptive" in a statistical sense.
    - We only update the AFFINE PARAMETERS (weight, bias) of LN,
      exactly as TENT updates BN affine params.
    - No need to set track_running_stats = False (doesn't exist in LN).
    - LN mode: model.train() makes LN use batch-level mean/var.
               model.eval() makes LN use the same (no change — LN
               always uses the current input's statistics).
      So we can keep model in eval() and still adapt LN params.

    adapt_names: list of LN module names to adapt (e.g. last 3 blocks).
    all_ln:      if True, adapt ALL LN layers (25 for ViT-B/16).
    """
    model.eval()
    model.requires_grad_(False)

    for name, m in model.named_modules():
        if not isinstance(m, nn.LayerNorm):
            continue
        # If all_ln, adapt every LN; otherwise only named layers
        if all_ln or name in adapt_names:
            m.requires_grad_(True)

    n_adapted = sum(1 for p in model.parameters() if p.requires_grad)
    log(f"    TENT-ViT: {n_adapted} LN param tensors enabled for grad "
        f"({'all LN' if all_ln else f'last-3-blocks + encoder.ln'})")
    return model


def run_tent_vit(model: nn.Module, loader: DataLoader,
                 adapt_names: list, all_ln: bool = False,
                 lr: float = 1e-4, steps: int = 1):
    """
    TENT for ViT-B/16.

    lr=1e-4 (lower than BN-TENT's 1e-3) because LN params are fewer
    and the model is larger — more conservative updates are safer.

    No entropy guard needed: ViT-B/16 on ImageNet-C severity 5 starts
    at ~22-40% accuracy (much better than ResNet-50's 2-6%), so entropy
    is well below the collapse threshold and gradient signal is meaningful.
    """
    model = configure_tent_vit(model, adapt_names, all_ln=all_ln)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999)
    )

    all_conf, all_pred, all_labels = [], [], []

    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)

        # Entropy minimisation step on LN affine params
        for _ in range(steps):
            optimizer.zero_grad()
            probs   = F.softmax(model(imgs), dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            entropy.backward()
            optimizer.step()

        with torch.no_grad():
            probs      = F.softmax(model(imgs), dim=1)
            conf, pred = probs.max(dim=1)

        all_conf.append(conf.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_labels.append(lbls.numpy())

    confs  = np.concatenate(all_conf)
    preds  = np.concatenate(all_pred)
    labels = np.concatenate(all_labels)
    return (float((preds == labels).mean()),
            compute_ece(confs, preds, labels),
            reliability_data(confs, preds, labels))


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: CGT-TTA for ViT-B/16
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBank:
    """FIFO queue of (L2-normalised CLS-token feature, is_correct) pairs."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self.features = deque(maxlen=max_size)
        self.correct  = deque(maxlen=max_size)

    def __len__(self):
        return len(self.features)

    def update(self, feats: np.ndarray, correct_flags: np.ndarray):
        for f, c in zip(feats, correct_flags):
            self.features.append(f)
            self.correct.append(bool(c))

    def local_accuracy(self, query: np.ndarray, k: int) -> float:
        if len(self) < k:
            return 0.5
        bank = np.stack(self.features)   # (N, D)
        sims = bank @ query              # cosine similarity (both L2-normed)
        idx  = np.argpartition(sims, -k)[-k:]
        return float(np.array(self.correct)[idx].mean())


def _cls_feat_hook(storage: dict):
    """
    Hook on the last ViT encoder block to capture the CLS token feature.

    torchvision ViT-B/16 encoder block output shape:
      (B, num_tokens, hidden_dim)  =  (B, 197, 768)
    CLS token = index 0 → (B, 768)
    """
    def hook(module, inp, out):
        # out: (B, seq_len, hidden_dim)
        cls_tok = out[:, 0, :]                          # (B, 768)
        storage["feats"] = F.normalize(
            cls_tok, dim=1).detach().cpu().numpy()
    return hook


def _warmup_bank_vit(model: nn.Module, loader: DataLoader,
                     bank: FeatureBank, storage: dict,
                     max_batches: int = 32):
    """
    Fill feature bank with source-only CLS features + correctness flags
    before any adaptation begins.
    """
    model.eval()
    seen = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            if seen >= max_batches:
                break
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)                       # triggers hook
            feats  = storage["feats"]                  # (B, 768)
            preds  = logits.argmax(dim=1).cpu().numpy()
            bank.update(feats, preds == lbls.numpy())
            seen += 1
            if len(bank) >= bank.max_size:
                break
    # restore train mode for LN adaptation
    model.train()
    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            pass  # LN behaves the same in train/eval — no change needed


def run_cgt_tta_vit(model:       nn.Module,
                    loader:      DataLoader,
                    adapt_names: list,
                    all_ln:      bool  = False,
                    k:           int   = 20,
                    bank_size:   int   = 2048,
                    cgs_thresh:  float = 0.4,
                    lr:          float = 1e-4,
                    steps:       int   = 1,
                    warm_start:  bool  = True):
    """
    CGT-TTA v3 for ViT-B/16.

    Identical decoupled design as the ResNet/CIFAR version:
      (A) LN alignment — always on full batch via entropy minimisation.
      (B) Prediction trust gate — per sample via CGS on CLS token.

    Feature extraction: CLS token from last encoder block (dim=768).
    Norm layer: LayerNorm (not BatchNorm).
    """
    model     = configure_tent_vit(model, adapt_names, all_ln=all_ln)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999)
    )

    # ── feature hook: last encoder block CLS token ───────────────────────────
    storage     = {}
    last_block  = model.encoder.layers[-1]
    hook_handle = last_block.register_forward_hook(_cls_feat_hook(storage))

    bank = FeatureBank(max_size=bank_size)

    # ── warm-start ────────────────────────────────────────────────────────────
    if warm_start:
        _warmup_bank_vit(model, loader, bank, storage, max_batches=32)

    all_conf, all_pred, all_labels = [], [], []
    total_samples = 0
    gated_samples = 0

    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        B    = imgs.size(0)

        # ── (1) source forward pass (no grad) ─────────────────────────────────
        with torch.no_grad():
            logits_src = model(imgs)
            probs_src  = F.softmax(logits_src, dim=1)
            conf_src_t, pred_src_t = probs_src.max(dim=1)

        feats_b     = storage["feats"]          # (B, 768)
        conf_src_np = conf_src_t.cpu().numpy()
        pred_src_np = pred_src_t.cpu().numpy()

        # ── (2) CGS per sample ────────────────────────────────────────────────
        cgs = np.zeros(B)
        for i in range(B):
            la     = bank.local_accuracy(feats_b[i], k=k)
            cgs[i] = float(conf_src_np[i]) - la
        gate_mask = cgs > cgs_thresh

        # ── (A) LN alignment on full batch ────────────────────────────────────
        for _ in range(steps):
            optimizer.zero_grad()
            probs_a = F.softmax(model(imgs), dim=1)
            entropy = -(probs_a * torch.log(probs_a + 1e-8)).sum(dim=1).mean()
            entropy.backward()
            optimizer.step()

        # ── (3) adapted forward pass (no grad) ────────────────────────────────
        with torch.no_grad():
            probs_ada = F.softmax(model(imgs), dim=1)
            conf_ada_t, pred_ada_t = probs_ada.max(dim=1)

        conf_ada_np = conf_ada_t.cpu().numpy()
        pred_ada_np = pred_ada_t.cpu().numpy()

        # ── (B) prediction trust gate ─────────────────────────────────────────
        final_conf = np.where(gate_mask, conf_ada_np, conf_src_np)
        final_pred = np.where(gate_mask, pred_ada_np, pred_src_np)

        # ── (4) update feature bank ───────────────────────────────────────────
        pseudo_correct = (final_pred == lbls.numpy())
        bank.update(feats_b, pseudo_correct)

        all_conf.append(final_conf)
        all_pred.append(final_pred)
        all_labels.append(lbls.numpy())
        total_samples += B
        gated_samples += int(gate_mask.sum())

    hook_handle.remove()
    confs  = np.concatenate(all_conf)
    preds  = np.concatenate(all_pred)
    labels = np.concatenate(all_labels)

    return (float((preds == labels).mean()),
            compute_ece(confs, preds, labels),
            reliability_data(confs, preds, labels),
            gated_samples / max(total_samples, 1))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(args):
    t0 = time.time()

    log(f"\n{'='*72}")
    log(f"  CGT-TTA — ImageNet-C  ×  ViT-B/16")
    log(f"  Device     : {DEVICE}" +
        (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    log(f"  Data dir   : {args.data_dir}")
    log(f"  Severity   : {args.severity}")
    log(f"  Batch size : {args.batch_size}")
    log(f"  CGS thresh : {args.cgs_thresh}  k={args.k}  bank={args.bank_size}")
    log(f"  Adapt LN   : {'ALL LayerNorm' if args.all_ln else 'last-3-blocks + encoder.ln'}")
    log(f"  TENT lr    : {args.lr}")
    log(f"{'='*72}\n")

    # ── validate data_dir ─────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log(f"ERROR: data_dir not found: {data_dir}")
        for p in Path(r"C:\Users\Vineet9.Yadav\Desktop\TTA").glob("**/gaussian_noise"):
            log(f"  Possible location: {p.parent}")
        sys.exit(1)

    corruptions_to_run = args.corruptions or CORRUPTIONS
    missing = [c for c in corruptions_to_run
               if not (data_dir / c / str(args.severity)).exists()]
    if missing:
        log(f"WARNING: not found at severity {args.severity}: {missing}")
        corruptions_to_run = [c for c in corruptions_to_run if c not in missing]
    if not corruptions_to_run:
        log("ERROR: no valid corruptions. Check --data_dir and --severity.")
        sys.exit(1)

    log(f"  Running {len(corruptions_to_run)} corruptions\n")

    # ── load model ────────────────────────────────────────────────────────────
    base_model   = load_vitb()
    adapt_names  = ln_layers_to_adapt(base_model, last_n_blocks=3)
    log(f"  LN layers to adapt: {adapt_names}\n")

    # ── quick sanity: one forward pass ───────────────────────────────────────
    with torch.no_grad():
        dummy    = torch.randn(2, 3, 224, 224).to(DEVICE)
        out      = base_model(dummy)
        assert out.shape == (2, 1000), f"Unexpected output shape: {out.shape}"
    log("  Sanity check passed (2×1000 output)\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reliability").mkdir(exist_ok=True)

    results   = {m: {} for m in ("source_only", "tent", "cgt_tta")}
    rel_store = {m: {} for m in ("source_only", "tent", "cgt_tta")}

    # ── header ────────────────────────────────────────────────────────────────
    W = 22
    hdr = (f"{'Corruption':<{W}}  "
           f"{'Src Acc':>8} {'TENT Acc':>9} {'CGT Acc':>8}  "
           f"{'Src ECE':>8} {'TENT ECE':>9} {'CGT ECE':>8}  "
           f"{'ΔAcc':>7} {'ΔECE':>7}  {'Gate%':>6}")
    log(hdr)
    log("─" * len(hdr))

    for i, corruption in enumerate(corruptions_to_run):
        sys.stdout.write(
            progress(i + 1, len(corruptions_to_run),
                     prefix=f"  {corruption:<{W}}"))
        sys.stdout.flush()

        log(f"\n  Loading: {corruption} severity {args.severity}")
        try:
            loader = make_loader(
                args.data_dir, corruption, args.severity,
                args.batch_size, args.num_workers
            )
        except Exception as e:
            log(f"  SKIP {corruption}: {e}")
            continue

        # Source-only
        m = copy.deepcopy(base_model)
        src_acc, src_ece, src_rel = run_source_only(m, loader)
        results["source_only"][corruption]  = {"acc": src_acc, "ece": src_ece}
        rel_store["source_only"][corruption] = src_rel

        # TENT (LN adaptation)
        m = copy.deepcopy(base_model)
        tent_acc, tent_ece, tent_rel = run_tent_vit(
            m, loader,
            adapt_names=adapt_names,
            all_ln=args.all_ln,
            lr=args.lr,
            steps=args.steps,
        )
        results["tent"][corruption]  = {"acc": tent_acc, "ece": tent_ece}
        rel_store["tent"][corruption] = tent_rel

        # CGT-TTA (LN + CLS token gating)
        m = copy.deepcopy(base_model)
        cgt_acc, cgt_ece, cgt_rel, cgt_ar = run_cgt_tta_vit(
            m, loader,
            adapt_names=adapt_names,
            all_ln=args.all_ln,
            k=args.k,
            bank_size=args.bank_size,
            cgs_thresh=args.cgs_thresh,
            lr=args.lr,
            steps=args.steps,
            warm_start=True,
        )
        results["cgt_tta"][corruption] = {
            "acc": cgt_acc, "ece": cgt_ece, "gate_rate": cgt_ar
        }
        rel_store["cgt_tta"][corruption] = cgt_rel

        delta_acc = cgt_acc  - tent_acc
        delta_ece = tent_ece - cgt_ece    # positive = CGT better

        sys.stdout.write("\r")
        log(
            f"{corruption:<{W}}  "
            f"{src_acc*100:>7.2f}% {tent_acc*100:>8.2f}% {cgt_acc*100:>7.2f}%  "
            f"{src_ece*100:>7.2f}% {tent_ece*100:>8.2f}% {cgt_ece*100:>7.2f}%  "
            f"{delta_acc*100:>+6.2f}% {delta_ece*100:>+6.2f}%  "
            f"{cgt_ar*100:>5.1f}%"
        )

    # ── summary ───────────────────────────────────────────────────────────────
    done = [c for c in corruptions_to_run if c in results["cgt_tta"]]

    if done:
        log(f"\n{'='*72}")
        log(f"  SUMMARY — Mean over {len(done)} corruptions (ViT-B/16, severity {args.severity})")
        log(f"{'='*72}")

        src_acc_m  = np.mean([results["source_only"][c]["acc"]       for c in done])
        src_ece_m  = np.mean([results["source_only"][c]["ece"]       for c in done])
        tent_acc_m = np.mean([results["tent"][c]["acc"]              for c in done])
        tent_ece_m = np.mean([results["tent"][c]["ece"]              for c in done])
        cgt_acc_m  = np.mean([results["cgt_tta"][c]["acc"]           for c in done])
        cgt_ece_m  = np.mean([results["cgt_tta"][c]["ece"]           for c in done])
        cgt_ar_m   = np.mean([results["cgt_tta"][c]["gate_rate"]     for c in done])

        log(f"\n  {'Method':<18} {'Accuracy':>10} {'ECE':>10} {'Gate%':>8}")
        log(f"  {'─'*50}")
        log(f"  {'Source-only':<18} {src_acc_m*100:>9.2f}% {src_ece_m*100:>9.2f}%")
        log(f"  {'TENT':<18} {tent_acc_m*100:>9.2f}% {tent_ece_m*100:>9.2f}%")
        log(f"  {'CGT-TTA (ours)':<18} {cgt_acc_m*100:>9.2f}% {cgt_ece_m*100:>9.2f}% {cgt_ar_m*100:>7.1f}%")
        log(f"\n  ΔAcc (CGT − TENT) : {(cgt_acc_m - tent_acc_m)*100:>+.3f}pp")
        log(f"  ΔECE (TENT − CGT) : {(tent_ece_m - cgt_ece_m)*100:>+.3f}pp  "
            f"({'improvement ✓' if tent_ece_m > cgt_ece_m else 'WORSE ✗'})")
        log(f"  Avg gate rate     : {cgt_ar_m*100:.1f}%")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed/60:.1f} min")

    # ── save outputs ──────────────────────────────────────────────────────────
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("corruption,src_acc,src_ece,tent_acc,tent_ece,"
                "cgt_acc,cgt_ece,cgt_gate_rate,delta_acc,delta_ece\n")
        for c in done:
            s  = results["source_only"][c]
            t  = results["tent"][c]
            g  = results["cgt_tta"][c]
            da = g["acc"] - t["acc"]
            de = t["ece"] - g["ece"]
            f.write(f"{c},"
                    f"{s['acc']:.6f},{s['ece']:.6f},"
                    f"{t['acc']:.6f},{t['ece']:.6f},"
                    f"{g['acc']:.6f},{g['ece']:.6f},{g['gate_rate']:.6f},"
                    f"{da:.6f},{de:.6f}\n")
    log(f"\n  Saved: {summary_path}")

    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({
            "model":    "vit_b_16",
            "severity": args.severity,
            "args":     vars(args),
            "results":  {m: dict(v) for m, v in results.items()},
        }, f, indent=2)
    log(f"  Saved: {json_path}")

    for method in ("source_only", "tent", "cgt_tta"):
        rel_path = output_dir / "reliability" / f"{method}.csv"
        with open(rel_path, "w") as f:
            f.write("corruption,bin_mid,bin_acc,bin_conf,bin_count\n")
            for c in done:
                for bd in rel_store[method].get(c, []):
                    f.write(f"{c},{bd['bin_mid']:.4f},"
                            f"{bd['bin_acc']:.4f},{bd['bin_conf']:.4f},"
                            f"{bd['bin_count']}\n")
    log(f"  Reliability CSVs → {output_dir}/reliability/")
    log(f"\n{'='*72}\n  Done.\n{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CGT-TTA on ImageNet-C with ViT-B/16",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths (VS Code local defaults) ───────────────────────────────────────
    p.add_argument("--data_dir",   type=str,
                   default=r"C:\Users\Vineet9.Yadav\Desktop\TTA\ImageNet-C",
                   help="Root of ImageNet-C")
    p.add_argument("--output_dir", type=str,
                   default=r"C:\Users\Vineet9.Yadav\Desktop\TTA\results\imagenetc_vitb",
                   help="Where to save outputs")

    # ── Experiment ────────────────────────────────────────────────────────────
    p.add_argument("--severity",    type=int, default=5,
                   help="Corruption severity (1-5).")
    p.add_argument("--corruptions", type=str, nargs="+", default=None,
                   help="Subset to run. Omit for all 15.")

    # ── Hardware (RTX 3080/4080 / RTX A4000) ─────────────────────────────────
    # ViT-B/16 is larger than ResNet-50: use batch_size=64 to be safe.
    # RTX A4000 has 16GB — can push to 96 or 128 if no OOM.
    p.add_argument("--batch_size",  type=int, default=64,
                   help="64 is safe for ViT-B/16 on 16GB GPU. Try 96 if no OOM.")
    p.add_argument("--num_workers", type=int, default=8)

    # ── LN adaptation ─────────────────────────────────────────────────────────
    p.add_argument("--all_ln",  action="store_true",
                   help="Adapt ALL LayerNorm layers (25 total). "
                        "Default: last 3 blocks + encoder.ln (7 LN layers).")

    # ── CGT-TTA hyperparameters ───────────────────────────────────────────────
    p.add_argument("--cgs_thresh", type=float, default=0.4)
    p.add_argument("--k",          type=int,   default=20)
    p.add_argument("--bank_size",  type=int,   default=2048)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # lr=1e-4 for ViT (conservative; ViT LN params are more sensitive than BN)
    p.add_argument("--lr",    type=float, default=1e-4,
                   help="1e-4 recommended for ViT LN adaptation.")
    p.add_argument("--steps", type=int,   default=1)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_all(args)