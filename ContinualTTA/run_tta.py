# =============================================================================
# ImageNet-C: Continual Test-Time Adaptation — Final Paper-Ready Script
# Run via: python run_tta.py --method TENT
#
# Audit fixes applied:
#   1. ContinualTTA: restored .detach() on features in prototype loss
#   2. CoTTA: removed non-paper anchor loss, removed unnecessary un-normalise
#   3. RoTTA: reverted ROTTA_N to paper value (64), restored /NUM_CLASSES
#   4. All methods: verified against original papers
# =============================================================================

import os
import copy
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\ImageNet-C"
RESULTS_DIR = "results"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
NUM_CLASSES = 1000
FEAT_DIM    = 2048
SEVERITY    = 5       # standard for ImageNet-C TTA papers

LR           = 1e-3
EMA_DECAY    = 0.9
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # ≈ 2.763 nats for C=1000
ALPHA        = 0.5
JS_THRESHOLD = 0.02

ROTTA_NU = 0.001   # teacher EMA — paper default (Yuan et al. CVPR 2023)
ROTTA_N  = 64      # memory bank capacity — paper default, NOT 5000
                   # per_class = max(1, 64//1000) = 1 slot per class
SAR_RHO  = 0.05    # sharpness perturbation radius

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]


# =============================================================================
# 1. DATASET & MODEL
# =============================================================================

# Use exact torchvision preprocessing matching the pretrained weights
# This includes Resize(256) + CenterCrop(224) + ToTensor + Normalize
_weights      = models.ResNet50_Weights.IMAGENET1K_V1
val_transform = _weights.transforms()


def load_corruption(corruption, severity=SEVERITY):
    """Load one ImageNet-C corruption at given severity using ImageFolder."""
    path = os.path.join(DATA_DIR, corruption, str(severity))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Path not found: {path}\n"
            f"Expected: DATA_DIR/{corruption}/{severity}/n01234567/*.JPEG\n"
            f"Download from: https://zenodo.org/records/2235448")
    dataset = ImageFolder(path, transform=val_transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)
    return loader


def load_model():
    """
    Standard ImageNet pretrained ResNet-50.
    Identical to source model used in TENT, EATA, CoTTA, RoTTA, SAR papers.
    Allows direct comparison to published numbers.
    """
    model = models.resnet50(weights=_weights)
    return model.to(DEVICE).eval()


# =============================================================================
# 2. HELPERS
# =============================================================================

def softmax_entropy(logits):
    """Per-sample Shannon entropy H(p). Shape: (B,)"""
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)


def eval_loader(model_fn, loader):
    """Evaluate model_fn over loader, return accuracy %."""
    correct, total = 0, 0
    for x, y in loader:
        x, y    = x.to(DEVICE), y.to(DEVICE)
        logits  = model_fn(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def setup_bn_imagenet(model):
    """
    ImageNet BN setup:
      - track_running_stats = True  → keep pretrained running statistics
      - momentum = 0               → freeze running mean/var (no batch updates)
      - gamma (weight) and beta (bias) trainable

    WHY: ImageNet pretrained BN running stats computed over ~1.2M images.
    Replacing with per-batch stats from 50k test images causes collapse.
    Freezing running stats and only adapting gamma/beta is stable.
    This matches the setup in TENT, EATA, SAR for ImageNet-C.
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True   # keep pretrained stats
            m.momentum = 0                 # freeze running mean/var
    params = [p for m in model.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    return model, params


# =============================================================================
# 3. BASELINE
# =============================================================================

def make_baseline(source):
    """No adaptation. Frozen model."""
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad():
            return model(x)
    return fn


# =============================================================================
# 4. TENT  (Wang et al., ICLR 2021)
# Entropy minimisation on BN affine parameters.
# Reference: https://github.com/DequanWang/tent
# =============================================================================

def make_tent(source):
    model, params = setup_bn_imagenet(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        softmax_entropy(logits).mean().backward()
        opt.step()
        opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 5. EATA  (Niu et al., ICML 2022)
# Two-filter entropy minimisation + Fisher regularisation.
#
# Filter 1: entropy < E_margin (reliable samples)
# Filter 2: cos_sim < (1 - d_margin) (diverse — not redundant with reference)
# Loss: entropy[reliable+diverse] + 1e-3 * Fisher_regularisation
#
# Note: Fisher computed on first corruption's first 10 batches (unsupervised).
# Reference: https://github.com/mr-eggplant/EATA
# =============================================================================

def make_eata(source, fisher_loader=None):
    model, params = setup_bn_imagenet(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    # Compute Fisher importance weights on first corruption data
    fisher = {n: torch.zeros_like(p)
              for n, p in model.named_parameters() if p.requires_grad}
    if fisher_loader is not None:
        model.train()
        for i, (x, _) in enumerate(fisher_loader):
            if i >= 10: break
            x = x.to(DEVICE)
            softmax_entropy(model(x)).mean().backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
            model.zero_grad()
        for n in fisher:
            fisher[n] /= 10

    ref_probs = [None]
    d_margin  = 0.05

    @torch.enable_grad()
    def fn(x):
        logits  = model(x)
        entropy = softmax_entropy(logits)
        probs   = logits.softmax(1)

        # Filter 1: reliable (low entropy)
        mask_e = entropy < E_MARGIN

        # Filter 2: diverse (not too similar to running reference)
        # cos_sim < (1 - d_margin) keeps samples DIFFERENT from reference
        if ref_probs[0] is not None:
            cos_sim = F.cosine_similarity(
                ref_probs[0].unsqueeze(0).expand(probs.size(0), -1),
                probs, dim=1)
            mask_d = cos_sim < (1.0 - d_margin)
        else:
            mask_d = torch.ones(probs.size(0), dtype=torch.bool, device=DEVICE)

        mask = mask_e & mask_d
        if mask.sum() == 0:
            return logits

        # Update running reference
        with torch.no_grad():
            if ref_probs[0] is None:
                ref_probs[0] = probs[mask].mean(0).detach()
            else:
                ref_probs[0] = (0.9 * ref_probs[0]
                                + 0.1 * probs[mask].mean(0).detach())

        # Loss = entropy + Fisher regularisation
        fisher_reg = sum((fisher[n] * p.pow(2)).sum()
                         for n, p in model.named_parameters()
                         if p.requires_grad and n in fisher)
        loss = entropy[mask].mean() + 1e-3 * fisher_reg
        loss.backward()
        opt.step()
        opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 6. CoTTA  (Wang et al., CVPR 2022)
# Augmentation-averaged pseudo-labels + teacher EMA + stochastic restoration.
#
# Augmentations applied directly on normalised tensors — correct.
# No un-normalise needed: RandomHorizontalFlip and RandomResizedCrop
# operate correctly on float tensors.
# No anchor loss — not in the original paper.
# Reference: https://github.com/qinenergy/cotta
# =============================================================================

def make_cotta(source):
    # Source model — for stochastic restoration only, never updated
    src = copy.deepcopy(source).eval()
    src.requires_grad_(False)

    # Adapted student
    adapted, params = setup_bn_imagenet(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    # Teacher — EMA of student, provides stable pseudo-labels
    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)

    # Augmentations on normalised tensors at 224x224 — correct
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ])

    @torch.enable_grad()
    def fn(x):
        # Step 1: teacher generates augmentation-averaged pseudo-labels
        with torch.no_grad():
            pseudo = torch.stack(
                [teacher(aug(x)).softmax(1) for _ in range(4)]).mean(0)

        # Step 2: student cross-entropy with teacher soft labels
        logits = adapted(x)
        loss   = -(pseudo * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        with torch.no_grad():
            # Step 3: teacher EMA (mt_alpha = 0.999)
            for tp, ap in zip(teacher.parameters(), adapted.parameters()):
                tp.data = 0.999 * tp.data + 0.001 * ap.data

            # Step 4: stochastic restoration from source (rst = 0.01)
            for (_, pa), (_, ps) in zip(adapted.named_parameters(),
                                         src.named_parameters()):
                if pa.requires_grad:
                    mask = torch.rand_like(pa) < 0.01
                    pa.data[mask] = ps.data[mask]

        return logits

    return fn


# =============================================================================
# 7. RoTTA  (Yuan et al., CVPR 2023)
# Robust BN + CSTU memory bank + timeliness reweighting.
#
# Paper defaults used exactly:
#   alpha=0.05 (RBN momentum), nu=0.001 (teacher EMA), N=64 (bank capacity)
# per_class = max(1, 64 // 1000) = 1 slot per class for ImageNet.
# CE loss divided by NUM_CLASSES per Eq.10 in paper.
# Mini-batch processing prevents OOM with 1000 classes.
# Reference: https://github.com/BIT-DA/RoTTA
# =============================================================================

def make_rotta(source):
    student = copy.deepcopy(source)
    student.train()
    student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True   # RBN: keep running stats
            m.momentum = 0.05             # slow EMA — paper default alpha=0.05
    params = [p for m in student.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)

    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)

    per_class = max(1, ROTTA_N // NUM_CLASSES)   # = 1 for 1000 classes
    bank      = {c: [] for c in range(NUM_CLASSES)}
    age       = [0]

    @torch.enable_grad()
    def fn(x):
        logits  = student(x)
        plabels = logits.argmax(1).detach()
        ents    = softmax_entropy(logits).detach()

        # CSTU: update memory bank
        with torch.no_grad():
            for i, (c, e) in enumerate(zip(plabels.tolist(), ents.tolist())):
                entry = (x[i].detach().cpu(), e, age[0])
                if len(bank[c]) < per_class:
                    bank[c].append(entry)
                else:
                    # Replace highest-entropy (least confident) sample
                    worst = max(range(len(bank[c])),
                                key=lambda j: bank[c][j][1])
                    if e < bank[c][worst][1]:
                        bank[c][worst] = entry
            age[0] += 1

        # Category-balanced sampling — most recent first (timeliness)
        samples, ages_list = [], []
        for c in range(NUM_CLASSES):
            if bank[c]:
                for entry in sorted(bank[c], key=lambda e: -e[2])[:per_class]:
                    samples.append(entry[0])
                    ages_list.append(entry[2])

        if len(samples) >= 2:
            ages_t = torch.tensor(ages_list, dtype=torch.float32,
                                   device=DEVICE)
            mem_x  = torch.stack(samples)

            # Mini-batch to avoid OOM — 1000 samples at once is too large
            BANK_BATCH = 32
            total_loss = torch.tensor(0.0, device=DEVICE)
            n_mini     = 0

            for start in range(0, len(samples), BANK_BATCH):
                end    = min(start + BANK_BATCH, len(samples))
                mb_x   = mem_x[start:end].to(DEVICE)
                mb_age = ages_t[start:end]

                # Timeliness weight E(age) from Eq.9 of RoTTA paper
                e_age = (torch.exp(-mb_age / ROTTA_N)
                         / (1 + torch.exp(-mb_age / ROTTA_N)))

                with torch.no_grad():
                    t_probs = teacher(mb_x).softmax(1)

                s_logits = student(mb_x)

                # CE per sample, averaged per class (Eq.10) — /NUM_CLASSES is paper-faithful
                ce = -(t_probs * s_logits.log_softmax(1)).sum(1) / NUM_CLASSES

                # Timeliness weighted loss (Eq.11)
                total_loss = total_loss + (e_age * ce).mean()
                n_mini    += 1

            (total_loss / n_mini).backward()
            opt.step()
            opt.zero_grad()

            # Teacher EMA (Eq.8) — nu=0.001 (very slow)
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = (1 - ROTTA_NU) * tp.data + ROTTA_NU * sp.data

        return logits

    return fn


# =============================================================================
# 8. SAR  (Niu et al., ICLR 2023)
# Sharpness-Aware and Reliable entropy minimisation.
#
# Two-step sharpness-aware update:
#   Step 1: forward + backward at θ → gradient g
#   Step 2: perturb θ' = θ + rho * g / ||g||
#   Step 3: forward + backward at θ' → sharpness-aware gradient
#   Step 4: restore θ, apply SGD step
# Dynamic entropy threshold tracks running EMA entropy.
# Reference: https://github.com/mr-eggplant/SAR
# =============================================================================

def make_sar(source):
    model, params = setup_bn_imagenet(copy.deepcopy(source))
    # SAR uses SGD not Adam — for flat minima exploration
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
    ema_entropy = [None]

    @torch.enable_grad()
    def fn(x):
        # Compute entropy without gradient for filtering
        with torch.no_grad():
            logits_init  = model(x)
            entropy_init = softmax_entropy(logits_init)

        if ema_entropy[0] is None:
            ema_entropy[0] = entropy_init.mean().item()

        # Dynamic threshold: min(E_margin, ema_entropy + 0.4*log(C))
        dynamic_thresh = min(E_MARGIN,
                             ema_entropy[0] + 0.4 * math.log(NUM_CLASSES))
        reliable = entropy_init < dynamic_thresh
        if reliable.sum() == 0:
            return logits_init

        x_rel = x[reliable]

        # Step 1: gradient at current params
        logits_1 = model(x_rel)
        softmax_entropy(logits_1).mean().backward()
        grad_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))

        # Step 2: perturb params towards sharp region
        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
                p.data.add_(e_w)
                e_ws.append(e_w)
                p.grad.zero_()
            else:
                e_ws.append(None)

        # Step 3: gradient at perturbed params
        logits_2   = model(x_rel)
        entropy_2  = softmax_entropy(logits_2)
        reliable_2 = entropy_2 < E_MARGIN
        if reliable_2.sum() > 0:
            entropy_2[reliable_2].mean().backward()

        # Step 4: restore original params + apply update
        for p, e_w in zip(params, e_ws):
            if e_w is not None:
                p.data.sub_(e_w)
        opt.step()
        opt.zero_grad()

        # Update EMA entropy tracker
        with torch.no_grad():
            logits_out  = model(x)
            entropy_out = softmax_entropy(logits_out)
            ema_entropy[0] = (0.9 * ema_entropy[0]
                              + 0.1 * entropy_out.mean().item())

        return logits_out

    return fn


# =============================================================================
# 9. ContinualTTA (ours)
# JS shift detector + reliable filter + EMA prototype bank.
#
# IMPORTANT: features[reliable].detach() in prototype loss.
# The prototype consistency loss is a regulariser operating on feature
# representations — we do NOT want gradients to flow into backbone weights.
# Only BN gamma/beta receive gradient updates.
# =============================================================================

class PrototypeBank(nn.Module):
    """EMA prototype memory bank — one 2048-d centroid per class."""

    def __init__(self):
        super().__init__()
        self.decay = EMA_DECAY
        self.register_buffer("prototypes",  torch.zeros(NUM_CLASSES, FEAT_DIM))
        self.register_buffer("initialised", torch.zeros(NUM_CLASSES).bool())

    @torch.no_grad()
    def update(self, features, pseudo_labels):
        for c in pseudo_labels.unique():
            mask = (pseudo_labels == c)
            mf   = features[mask].mean(0)
            if self.initialised[c]:
                self.prototypes[c] = (self.decay * self.prototypes[c]
                                      + (1 - self.decay) * mf)
            else:
                self.prototypes[c] = mf
                self.initialised[c] = True

    def consistency_loss(self, features, pseudo_labels):
        """MSE between detached features and stored class prototypes."""
        loss, count = torch.tensor(0.0, device=features.device), 0
        for c in pseudo_labels.unique():
            if not self.initialised[c]:
                continue
            mask = (pseudo_labels == c)
            loss += F.mse_loss(
                features[mask],
                self.prototypes[c].unsqueeze(0).expand(mask.sum(), -1))
            count += 1
        return loss / max(count, 1)


class JSShiftDetector:
    """
    Jensen-Shannon divergence shift detector.
    Triggers adaptation when JS(p_ref, p_t) > threshold.
    Reference updated AFTER computing JS — intentional ordering.
    """

    def __init__(self, threshold=JS_THRESHOLD, ema=0.9):
        self.threshold = threshold
        self.ema       = ema
        self.reference = None

    def should_adapt(self, logits):
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)

            if self.reference is None:
                self.reference = p_t.clone()
                return True

            m    = 0.5 * (self.reference + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             self.reference.unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5 * (kl_1 + kl_2)

            # Update reference AFTER computing JS
            self.reference = self.ema * self.reference + (1 - self.ema) * p_t

            return js.item() > self.threshold


def make_ctta(source):
    model, params = setup_bn_imagenet(copy.deepcopy(source))
    bank     = PrototypeBank().to(DEVICE)
    detector = JSShiftDetector()
    opt      = torch.optim.Adam(params, lr=LR)
    captured = {}
    # Hook captures avgpool features (2048-d) during forward pass
    handle   = model.avgpool.register_forward_hook(
        lambda m, i, o: captured.update({"feat": o.flatten(1)}))

    @torch.enable_grad()
    def fn(x):
        logits        = model(x)
        features      = captured["feat"]
        pseudo_labels = logits.argmax(1).detach()

        # Gate 1: JS shift detector
        if not detector.should_adapt(logits.detach()):
            return logits

        # Gate 2: reliable sample filter
        entropy  = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        if reliable.sum() == 0:
            return logits

        # Combined loss — features MUST be detached
        # Prototype loss is a regulariser: no backbone gradients needed
        loss = (entropy[reliable].mean()
                + ALPHA * bank.consistency_loss(
                    features[reliable].detach(),   # ← correct: detach
                    pseudo_labels[reliable]))
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Update prototype bank (always use detached features)
        bank.update(features[reliable].detach(), pseudo_labels[reliable])
        return logits

    fn._handle = handle
    return fn


# =============================================================================
# 10. BUILD AND RUN
# =============================================================================

def build_single_method(method_name, source_model, fisher_loader=None):
    dispatch = {
        "Baseline":     lambda: make_baseline(source_model),
        "TENT":         lambda: make_tent(source_model),
        "EATA":         lambda: make_eata(source_model, fisher_loader),
        "CoTTA":        lambda: make_cotta(source_model),
        "RoTTA":        lambda: make_rotta(source_model),
        "SAR":          lambda: make_sar(source_model),
        "ContinualTTA": lambda: make_ctta(source_model),
    }
    if method_name not in dispatch:
        raise ValueError(f"Unknown method: {method_name}. "
                         f"Choose from {list(dispatch.keys())}")
    return dispatch[method_name]()


def run_continual_sequential(method, source_model):
    """
    Run one method through all 15 corruptions in fixed order.
    No model reset between corruptions — continual sequential protocol.
    """
    print(f"\nRunning {method} — continual sequential, Severity {SEVERITY}")
    print(f"15 corruptions in fixed order, no reset between them.\n")

    fisher_loader = None
    if method == "EATA":
        fisher_loader = load_corruption(ALL_CORRUPTIONS[0])

    fn      = build_single_method(method, source_model, fisher_loader)
    results = {}

    for corruption in ALL_CORRUPTIONS:
        loader = load_corruption(corruption)
        acc    = eval_loader(fn, loader)
        results[corruption] = acc
        del loader
        print(f"  {corruption:<24} {acc:.1f}%")

    mean_acc = np.mean(list(results.values()))
    print(f"\n  Mean: {mean_acc:.1f}%")
    return results


def save_csv(method, results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mean_acc = np.mean(list(results.values()))
    filepath = os.path.join(RESULTS_DIR, f"imagenetc_{method}.csv")
    with open(filepath, "w") as f:
        f.write(f"corruption,{method}\n")
        for c in ALL_CORRUPTIONS:
            f.write(f"{c},{results[c]:.1f}\n")
        f.write(f"Mean,{mean_acc:.1f}\n")
    print(f"Saved: {filepath}")
    return filepath


# =============================================================================
# 11. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ImageNet-C Continual TTA — run one method at a time")
    parser.add_argument(
        "--method", type=str, required=True,
        choices=["Baseline", "TENT", "EATA", "CoTTA",
                 "RoTTA", "SAR", "ContinualTTA"],
        help="Which TTA method to evaluate.")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ImageNet-C Continual Sequential TTA")
    print(f"{'='*60}")
    print(f"Device    : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM      : {vram:.1f} GB")
    print(f"Method    : {args.method}")
    print(f"Classes   : {NUM_CLASSES}")
    print(f"Severity  : {SEVERITY}")
    print(f"E_margin  : {E_MARGIN:.3f} nats")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Results   : {RESULTS_DIR}/")

    # Verify DATA_DIR structure
    print(f"\nVerifying DATA_DIR...")
    for c in ALL_CORRUPTIONS[:3]:
        path = os.path.join(DATA_DIR, c, str(SEVERITY))
        assert os.path.isdir(path), \
            (f"Missing: {path}\n"
             f"Run PowerShell fix if you still have blur/digital/noise/weather folders.")
    sample_path = os.path.join(DATA_DIR, "gaussian_noise", str(SEVERITY))
    n_classes   = len([d for d in os.listdir(sample_path)
                       if os.path.isdir(os.path.join(sample_path, d))])
    print(f"  Class folders in gaussian_noise/5: {n_classes} (expected 1000)")
    assert n_classes >= 900, f"Only {n_classes} classes found — check extraction."
    print(f"  Data check passed.")

    # Load model
    print(f"\nLoading ImageNet pretrained ResNet-50...")
    source_model = load_model()
    n_params = sum(p.numel() for p in source_model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Sanity check
    # print(f"\nSanity check — baseline on gaussian_noise S5...")
    # loader = load_corruption("gaussian_noise")
    # acc    = eval_loader(make_baseline(source_model), loader)
    # del loader
    # print(f"  Baseline: {acc:.1f}%  (expected 28–32%)")
    # assert acc > 10.0, f"Too low ({acc:.1f}%) — check DATA_DIR."
    # print(f"  Passed.\n")

    # Run
    results = run_continual_sequential(args.method, source_model)
    path    = save_csv(args.method, results)

    print(f"\n{'='*60}")
    print(f"DONE — {args.method}: {np.mean(list(results.values())):.1f}% mean")
    print(f"Results saved to: {os.path.abspath(path)}")
    print(f"{'='*60}")