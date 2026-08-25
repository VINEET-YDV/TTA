# """
# evaluate_imagenet_c.py  —  CGT-TTA on ImageNet-C (Severity 5)
# ==============================================================
# Compares three methods on ImageNet-C at severity 5:
#   1. Source-only   — frozen ResNet-50, no adaptation
#   2. TENT          — entropy minimisation over BN layers
#   3. CGT-TTA       — calibration-gated TTA (our method, v3)

# Supports two common ImageNet-C folder layouts on Kaggle:

#   Layout A (standard):
#     <data_dir>/<corruption>/<severity>/<class_id>/<image>.JPEG

#   Layout B (flat per-corruption):
#     <data_dir>/<corruption>/<severity>/<image>.JPEG
#     <data_dir>/labels.npy   (or val_labels.npy)

# Auto-detects which layout is present.

# Usage (Kaggle notebook cell)
# -----------------------------
#   !python evaluate_imagenet_c.py \
#       --data_dir   /kaggle/input/imagenet-c/ImageNet-C \
#       --output_dir /kaggle/working/results_imagenetc \
#       --severity   5 \
#       --batch_size 64 \
#       --cgs_thresh 0.4 \
#       --k          20 \
#       --bank_size  2048 \
#       --num_workers 4

# To run only a subset of corruptions (faster for testing):
#   --corruptions gaussian_noise shot_noise impulse_noise

# Outputs
# -------
#   summary.csv          — accuracy + ECE per method × corruption
#   results.json         — full structured results
#   reliability/         — reliability diagram CSVs
# """

# import os, sys, copy, json, time, argparse
# import numpy as np
# from pathlib import Path
# from collections import defaultdict, deque

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image

# # ─────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────────────────────────

# CORRUPTIONS = [
#     "gaussian_noise", "shot_noise",    "impulse_noise",
#     "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
#     "snow",           "frost",         "fog",           "brightness",
#     "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
# ]

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD  = (0.229, 0.224, 0.225)
# DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ─────────────────────────────────────────────────────────────────────────────
# # UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────

# def log(msg):
#     print(msg, flush=True)

# def progress(current, total, prefix="", width=30):
#     filled = int(width * current / total)
#     bar = "█" * filled + "░" * (width - filled)
#     return f"\r{prefix} [{bar}] {current}/{total}"


# # ─────────────────────────────────────────────────────────────────────────────
# # DATASET — auto-detects Layout A (ImageFolder) or Layout B (flat + labels)
# # ─────────────────────────────────────────────────────────────────────────────

# class ImageNetCDataset(Dataset):
#     """
#     Loads one (corruption, severity) slice of ImageNet-C.

#     Layout A — standard ImageNet folder structure:
#       <data_dir>/<corruption>/<severity>/<synset_id>/<image>.JPEG
#       Labels inferred from folder names mapped to ImageNet class indices.

#     Layout B — flat images + separate label file:
#       <data_dir>/<corruption>/<severity>/<image>.JPEG
#       Labels from <data_dir>/labels.npy or val_labels.npy (50000,)
#     """

#     def __init__(self, data_dir: str, corruption: str,
#                  severity: int, transform=None):
#         self.transform = transform
#         self.samples   = []   # list of (path, label)

#         base = Path(data_dir) / corruption / str(severity)
#         if not base.exists():
#             raise FileNotFoundError(f"Path not found: {base}")

#         # ── Detect layout ────────────────────────────────────────────────────
#         subdirs = [d for d in base.iterdir() if d.is_dir()]

#         if subdirs:
#             # Layout A: subfolders = class synset ids
#             log(f"    Layout A detected (ImageFolder): {base}")
#             self._load_layout_a(base)
#         else:
#             # Layout B: flat images + labels.npy
#             log(f"    Layout B detected (flat + labels): {base}")
#             self._load_layout_b(data_dir, base)

#     def _load_layout_a(self, base: Path):
#         """
#         ImageNet-C standard layout.
#         Maps synset folder names to class indices 0–999 using
#         sorted order (same as torchvision.datasets.ImageFolder).
#         """
#         class_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
#         class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

#         IMG_EXTS = {'.jpeg', '.jpg', '.png', '.bmp', '.webp'}
#         for cls_dir in class_dirs:
#             label = class_to_idx[cls_dir.name]
#             for img_path in cls_dir.iterdir():
#                 if img_path.suffix.lower() in IMG_EXTS:
#                     self.samples.append((str(img_path), label))

#         self.samples.sort(key=lambda x: x[0])   # deterministic order

#     def _load_layout_b(self, data_dir: str, base: Path):
#         """
#         Flat layout — images sorted alphabetically, labels from .npy file.
#         Looks for labels in several common locations.
#         """
#         IMG_EXTS = {'.jpeg', '.jpg', '.png', '.bmp', '.webp'}
#         imgs = sorted([
#             str(p) for p in base.iterdir()
#             if p.suffix.lower() in IMG_EXTS
#         ])

#         # find labels file
#         label_candidates = [
#             Path(data_dir) / "labels.npy",
#             Path(data_dir) / "val_labels.npy",
#             Path(data_dir).parent / "labels.npy",
#         ]
#         label_path = next((p for p in label_candidates if p.exists()), None)
#         if label_path is None:
#             raise FileNotFoundError(
#                 f"No labels.npy found. Searched: {label_candidates}"
#             )

#         labels = np.load(label_path)   # (50000,) for full ImageNet-C
#         if len(labels) != len(imgs):
#             # try to match by taking first len(imgs) labels
#             log(f"    Warning: {len(labels)} labels, {len(imgs)} images — using first {len(imgs)}")
#             labels = labels[:len(imgs)]

#         self.samples = list(zip(imgs, labels.tolist()))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         img = Image.open(path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)
#         return img, int(label)


# def make_transform():
#     return transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
#     ])


# def make_loader(data_dir, corruption, severity, batch_size, num_workers):
#     dataset = ImageNetCDataset(
#         data_dir, corruption, severity, transform=make_transform()
#     )
#     log(f"    Loaded {len(dataset):,} samples")
#     return DataLoader(
#         dataset, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers, pin_memory=True
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # MODEL
# # ─────────────────────────────────────────────────────────────────────────────

# def load_model() -> nn.Module:
#     """Load torchvision ResNet-50 pretrained on ImageNet."""
#     log("  Loading ResNet-50 (torchvision, ImageNet pretrained)...")
#     model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#     model.eval()
#     return model.to(DEVICE)


# # ─────────────────────────────────────────────────────────────────────────────
# # CALIBRATION
# # ─────────────────────────────────────────────────────────────────────────────

# def compute_ece(confidences, predictions, labels, n_bins=15):
#     bins = np.linspace(0.0, 1.0, n_bins + 1)
#     ece, n = 0.0, len(labels)
#     for i in range(n_bins):
#         mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
#         if mask.sum() == 0:
#             continue
#         ece += (mask.sum() / n) * abs(
#             (predictions[mask] == labels[mask]).mean() - confidences[mask].mean()
#         )
#     return float(ece)


# def reliability_data(confidences, predictions, labels, n_bins=15):
#     bins = np.linspace(0.0, 1.0, n_bins + 1)
#     result = []
#     for i in range(n_bins):
#         mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
#         if mask.sum() == 0:
#             continue
#         result.append({
#             "bin_mid":   float((bins[i] + bins[i+1]) / 2),
#             "bin_acc":   float((predictions[mask] == labels[mask]).mean()),
#             "bin_conf":  float(confidences[mask].mean()),
#             "bin_count": int(mask.sum()),
#         })
#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # METHOD 1: SOURCE-ONLY
# # ─────────────────────────────────────────────────────────────────────────────

# @torch.no_grad()
# def run_source_only(model, loader):
#     model.eval()
#     all_conf, all_pred, all_labels = [], [], []
#     for imgs, lbls in loader:
#         imgs  = imgs.to(DEVICE)
#         probs = F.softmax(model(imgs), dim=1)
#         conf, pred = probs.max(dim=1)
#         all_conf.append(conf.cpu().numpy())
#         all_pred.append(pred.cpu().numpy())
#         all_labels.append(lbls.numpy())
#     confs  = np.concatenate(all_conf)
#     preds  = np.concatenate(all_pred)
#     labels = np.concatenate(all_labels)
#     return (float((preds == labels).mean()),
#             compute_ece(confs, preds, labels),
#             reliability_data(confs, preds, labels))


# # ─────────────────────────────────────────────────────────────────────────────
# # METHOD 2: TENT
# # ─────────────────────────────────────────────────────────────────────────────

# def configure_tent(model):
#     model.train()
#     model.requires_grad_(False)
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
#             m.requires_grad_(True)
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var  = None
#     return model


# def run_tent(model, loader, lr=1e-3, steps=1):
#     model = configure_tent(model)
#     optimizer = torch.optim.Adam(
#         [p for p in model.parameters() if p.requires_grad],
#         lr=lr, betas=(0.9, 0.999)
#     )
#     all_conf, all_pred, all_labels = [], [], []
#     for imgs, lbls in loader:
#         imgs = imgs.to(DEVICE)
#         for _ in range(steps):
#             optimizer.zero_grad()
#             probs   = F.softmax(model(imgs), dim=1)
#             entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
#             entropy.backward()
#             optimizer.step()
#         with torch.no_grad():
#             probs      = F.softmax(model(imgs), dim=1)
#             conf, pred = probs.max(dim=1)
#         all_conf.append(conf.cpu().numpy())
#         all_pred.append(pred.cpu().numpy())
#         all_labels.append(lbls.numpy())
#     confs  = np.concatenate(all_conf)
#     preds  = np.concatenate(all_pred)
#     labels = np.concatenate(all_labels)
#     return (float((preds == labels).mean()),
#             compute_ece(confs, preds, labels),
#             reliability_data(confs, preds, labels))


# # ─────────────────────────────────────────────────────────────────────────────
# # METHOD 3: CGT-TTA  (v3 — decoupled design)
# # ─────────────────────────────────────────────────────────────────────────────

# class FeatureBank:
#     def __init__(self, max_size=2048):
#         self.max_size = max_size
#         self.features = deque(maxlen=max_size)
#         self.correct  = deque(maxlen=max_size)

#     def __len__(self):
#         return len(self.features)

#     def update(self, feats, correct_flags):
#         for f, c in zip(feats, correct_flags):
#             self.features.append(f)
#             self.correct.append(bool(c))

#     def local_accuracy(self, query, k):
#         if len(self) < k:
#             return 0.5
#         bank = np.stack(self.features)
#         sims = bank @ query
#         idx  = np.argpartition(sims, -k)[-k:]
#         return float(np.array(self.correct)[idx].mean())


# def _feat_hook(storage):
#     def hook(module, inp, out):
#         # ResNet-50 avgpool: (B, 2048, 1, 1) → (B, 2048)
#         f = out.squeeze(-1).squeeze(-1)
#         storage["feats"] = F.normalize(f, dim=1).detach().cpu().numpy()
#     return hook


# def _warmup_bank(model, loader, bank, storage, max_batches=32):
#     model.eval()
#     seen = 0
#     with torch.no_grad():
#         for imgs, lbls in loader:
#             if seen >= max_batches:
#                 break
#             imgs   = imgs.to(DEVICE)
#             logits = model(imgs)
#             feats  = storage["feats"]
#             preds  = logits.argmax(dim=1).cpu().numpy()
#             bank.update(feats, preds == lbls.numpy())
#             seen += 1
#             if len(bank) >= bank.max_size:
#                 break
#     model.train()
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var  = None


# def run_cgt_tta(model, loader, k=20, bank_size=2048,
#                 cgs_thresh=0.4, lr=1e-3, steps=1, warm_start=True):
#     """
#     CGT-TTA v3 — decoupled BN alignment + per-sample prediction trust gate.
#     Identical logic to the CIFAR-10-C version; feature dim is 2048 for ResNet-50.
#     """
#     model     = configure_tent(model)
#     optimizer = torch.optim.Adam(
#         [p for p in model.parameters() if p.requires_grad],
#         lr=lr, betas=(0.9, 0.999)
#     )

#     storage     = {}
#     hook_handle = model.avgpool.register_forward_hook(_feat_hook(storage))
#     bank        = FeatureBank(max_size=bank_size)

#     if warm_start:
#         _warmup_bank(model, loader, bank, storage, max_batches=32)

#     all_conf, all_pred, all_labels = [], [], []
#     total_samples = 0
#     gated_samples = 0

#     for imgs, lbls in loader:
#         imgs = imgs.to(DEVICE)
#         B    = imgs.size(0)

#         # (1) source forward pass
#         with torch.no_grad():
#             logits_src = model(imgs)
#             probs_src  = F.softmax(logits_src, dim=1)
#             conf_src_t, pred_src_t = probs_src.max(dim=1)

#         feats_b     = storage["feats"]
#         conf_src_np = conf_src_t.cpu().numpy()
#         pred_src_np = pred_src_t.cpu().numpy()

#         # (2) CGS per sample
#         cgs = np.zeros(B)
#         for i in range(B):
#             la     = bank.local_accuracy(feats_b[i], k=k)
#             cgs[i] = float(conf_src_np[i]) - la
#         gate_mask = cgs > cgs_thresh

#         # (3) BN alignment — always on full batch
#         for _ in range(steps):
#             optimizer.zero_grad()
#             probs_a = F.softmax(model(imgs), dim=1)
#             entropy = -(probs_a * torch.log(probs_a + 1e-8)).sum(dim=1).mean()
#             entropy.backward()
#             optimizer.step()

#         # (4) adapted forward pass
#         with torch.no_grad():
#             probs_ada = F.softmax(model(imgs), dim=1)
#             conf_ada_t, pred_ada_t = probs_ada.max(dim=1)

#         conf_ada_np = conf_ada_t.cpu().numpy()
#         pred_ada_np = pred_ada_t.cpu().numpy()

#         # (5) prediction trust gate
#         final_conf = np.where(gate_mask, conf_ada_np, conf_src_np)
#         final_pred = np.where(gate_mask, pred_ada_np, pred_src_np)

#         # (6) update bank
#         bank.update(feats_b, final_pred == lbls.numpy())

#         all_conf.append(final_conf)
#         all_pred.append(final_pred)
#         all_labels.append(lbls.numpy())
#         total_samples += B
#         gated_samples += int(gate_mask.sum())

#     hook_handle.remove()
#     confs  = np.concatenate(all_conf)
#     preds  = np.concatenate(all_pred)
#     labels = np.concatenate(all_labels)

#     return (float((preds == labels).mean()),
#             compute_ece(confs, preds, labels),
#             reliability_data(confs, preds, labels),
#             gated_samples / max(total_samples, 1))


# # ─────────────────────────────────────────────────────────────────────────────
# # MAIN EVALUATION LOOP
# # ─────────────────────────────────────────────────────────────────────────────

# def evaluate_all(args):
#     t0 = time.time()

#     log(f"\n{'='*72}")
#     log(f"  CGT-TTA — ImageNet-C Evaluation")
#     log(f"  Device     : {DEVICE}" +
#         (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
#     log(f"  Data dir   : {args.data_dir}")
#     log(f"  Severity   : {args.severity}")
#     log(f"  Batch size : {args.batch_size}")
#     log(f"  CGS thresh : {args.cgs_thresh}  k={args.k}  bank={args.bank_size}")
#     log(f"{'='*72}\n")

#     # ── validate data directory ───────────────────────────────────────────────
#     data_dir = Path(args.data_dir)
#     if not data_dir.exists():
#         log(f"ERROR: data_dir not found: {data_dir}")
#         log("Common Kaggle paths to check:")
#         log("  /kaggle/input/imagenet-c/ImageNet-C")
#         log("  /kaggle/input/imagenet-c/imagenet-c")
#         # try to auto-find
#         for p in Path("/kaggle/input").glob("**/gaussian_noise"):
#             log(f"  Found corruption folder at: {p.parent}")
#         sys.exit(1)

#     # confirm at least one corruption folder exists
#     corruptions_to_run = args.corruptions or CORRUPTIONS
#     missing = [c for c in corruptions_to_run
#                if not (data_dir / c / str(args.severity)).exists()]
#     if missing:
#         log(f"WARNING: these corruptions not found at severity {args.severity}: {missing}")
#         corruptions_to_run = [c for c in corruptions_to_run if c not in missing]
#         if not corruptions_to_run:
#             log("ERROR: no valid corruptions found. Check --data_dir and --severity.")
#             sys.exit(1)

#     log(f"  Running {len(corruptions_to_run)} corruptions: {corruptions_to_run}\n")

#     base_model = load_model()
#     log(f"  ResNet-50 loaded.\n")

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
#     (output_dir / "reliability").mkdir(exist_ok=True)

#     results   = defaultdict(dict)
#     rel_store = defaultdict(dict)

#     # ── header ────────────────────────────────────────────────────────────────
#     W = 22
#     hdr = (f"{'Corruption':<{W}}  "
#            f"{'Src Acc':>8} {'TENT Acc':>9} {'CGT Acc':>8}  "
#            f"{'Src ECE':>8} {'TENT ECE':>9} {'CGT ECE':>8}  "
#            f"{'ΔAcc':>7} {'ΔECE':>7}  {'Gate%':>6}")
#     log(hdr)
#     log("─" * len(hdr))

#     for i, corruption in enumerate(corruptions_to_run):
#         sys.stdout.write(progress(i+1, len(corruptions_to_run),
#                                   prefix=f"  {corruption:<{W}}"))
#         sys.stdout.flush()

#         log(f"\n  Loading: {corruption} severity {args.severity}")
#         try:
#             loader = make_loader(
#                 args.data_dir, corruption, args.severity,
#                 args.batch_size, args.num_workers
#             )
#         except Exception as e:
#             log(f"  SKIP {corruption}: {e}")
#             continue

#         # Source-only
#         m = copy.deepcopy(base_model)
#         src_acc, src_ece, src_rel = run_source_only(m, loader)
#         results["source_only"][corruption] = {"acc": src_acc, "ece": src_ece}
#         rel_store["source_only"][corruption] = src_rel

#         # TENT
#         m = copy.deepcopy(base_model)
#         tent_acc, tent_ece, tent_rel = run_tent(
#             m, loader, lr=args.lr, steps=args.steps
#         )
#         results["tent"][corruption] = {"acc": tent_acc, "ece": tent_ece}
#         rel_store["tent"][corruption] = tent_rel

#         # CGT-TTA
#         m = copy.deepcopy(base_model)
#         cgt_acc, cgt_ece, cgt_rel, cgt_ar = run_cgt_tta(
#             m, loader,
#             k=args.k, bank_size=args.bank_size,
#             cgs_thresh=args.cgs_thresh,
#             lr=args.lr, steps=args.steps,
#             warm_start=True,
#         )
#         results["cgt_tta"][corruption] = {
#             "acc": cgt_acc, "ece": cgt_ece, "gate_rate": cgt_ar
#         }
#         rel_store["cgt_tta"][corruption] = cgt_rel

#         delta_acc = cgt_acc - tent_acc
#         delta_ece = tent_ece - cgt_ece   # positive = CGT better

#         sys.stdout.write("\r")
#         log(
#             f"{corruption:<{W}}  "
#             f"{src_acc*100:>7.2f}% {tent_acc*100:>8.2f}% {cgt_acc*100:>7.2f}%  "
#             f"{src_ece*100:>7.2f}% {tent_ece*100:>8.2f}% {cgt_ece*100:>7.2f}%  "
#             f"{delta_acc*100:>+6.2f}% {delta_ece*100:>+6.2f}%  "
#             f"{cgt_ar*100:>5.1f}%"
#         )

#     # ── summary ───────────────────────────────────────────────────────────────
#     done = [c for c in corruptions_to_run if c in results["cgt_tta"]]

#     if done:
#         log(f"\n{'='*72}")
#         log(f"  SUMMARY — Mean over {len(done)} corruptions (severity {args.severity})")
#         log(f"{'='*72}")

#         def mean_r(method, key):
#             return np.mean([results[method][c][key] for c in done])

#         src_acc_m   = mean_r("source_only", "acc")
#         src_ece_m   = mean_r("source_only", "ece")
#         tent_acc_m  = mean_r("tent",        "acc")
#         tent_ece_m  = mean_r("tent",        "ece")
#         cgt_acc_m   = mean_r("cgt_tta",     "acc")
#         cgt_ece_m   = mean_r("cgt_tta",     "ece")
#         cgt_ar_m    = mean_r("cgt_tta",     "gate_rate")

#         log(f"\n  {'Method':<18} {'Accuracy':>10} {'ECE':>10} {'Gate%':>8}")
#         log(f"  {'─'*50}")
#         log(f"  {'Source-only':<18} {src_acc_m*100:>9.2f}% {src_ece_m*100:>9.2f}%")
#         log(f"  {'TENT':<18} {tent_acc_m*100:>9.2f}% {tent_ece_m*100:>9.2f}%")
#         log(f"  {'CGT-TTA (ours)':<18} {cgt_acc_m*100:>9.2f}% {cgt_ece_m*100:>9.2f}% {cgt_ar_m*100:>7.1f}%")
#         log(f"\n  ΔAcc (CGT − TENT) : {(cgt_acc_m - tent_acc_m)*100:>+.3f}pp")
#         log(f"  ΔECE (TENT − CGT) : {(tent_ece_m - cgt_ece_m)*100:>+.3f}pp  "
#             f"({'improvement' if tent_ece_m > cgt_ece_m else 'WORSE'})")
#         log(f"  Avg gate rate     : {cgt_ar_m*100:.1f}%")

#     elapsed = time.time() - t0
#     log(f"\n  Total time: {elapsed/60:.1f} min")

#     # ── save summary CSV ──────────────────────────────────────────────────────
#     summary_path = output_dir / "summary.csv"
#     with open(summary_path, "w") as f:
#         f.write("corruption,src_acc,src_ece,tent_acc,tent_ece,"
#                 "cgt_acc,cgt_ece,cgt_gate_rate,"
#                 "delta_acc,delta_ece\n")
#         for c in done:
#             s  = results["source_only"][c]
#             t  = results["tent"][c]
#             g  = results["cgt_tta"][c]
#             da = g["acc"] - t["acc"]
#             de = t["ece"] - g["ece"]
#             f.write(f"{c},"
#                     f"{s['acc']:.6f},{s['ece']:.6f},"
#                     f"{t['acc']:.6f},{t['ece']:.6f},"
#                     f"{g['acc']:.6f},{g['ece']:.6f},{g['gate_rate']:.6f},"
#                     f"{da:.6f},{de:.6f}\n")
#     log(f"\n  Saved: {summary_path}")

#     # ── save JSON ─────────────────────────────────────────────────────────────
#     json_path = output_dir / "results.json"
#     with open(json_path, "w") as f:
#         json.dump({
#             "args": vars(args),
#             "severity": args.severity,
#             "results": {m: dict(v) for m, v in results.items()},
#         }, f, indent=2)
#     log(f"  Saved: {json_path}")

#     # ── save reliability data ─────────────────────────────────────────────────
#     for method in ("source_only", "tent", "cgt_tta"):
#         rel_path = output_dir / "reliability" / f"{method}.csv"
#         with open(rel_path, "w") as f:
#             f.write("corruption,bin_mid,bin_acc,bin_conf,bin_count\n")
#             for c in done:
#                 for bd in rel_store[method].get(c, []):
#                     f.write(f"{c},{bd['bin_mid']:.4f},"
#                             f"{bd['bin_acc']:.4f},{bd['bin_conf']:.4f},"
#                             f"{bd['bin_count']}\n")
#     log(f"  Reliability CSVs → {output_dir}/reliability/")

#     log(f"\n{'='*72}\n  Done.\n{'='*72}\n")


# # ─────────────────────────────────────────────────────────────────────────────
# # ARGUMENT PARSING
# # ─────────────────────────────────────────────────────────────────────────────

# def parse_args():
#     p = argparse.ArgumentParser(
#         description="CGT-TTA on ImageNet-C",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     # ── Paths — local VS Code defaults ────────────────────────────────────
#     # data_dir:   folder containing gaussian_noise/, shot_noise/, etc.
#     # output_dir: where results are written
#     # Override via CLI if your folder is elsewhere.
#     p.add_argument("--data_dir",   type=str,
#                    default=r"C:\Users\Vineet9.Yadav\Desktop\TTA\ImageNet-C",
#                    help="Root of ImageNet-C: <data_dir>/<corruption>/<severity>/...")
#     p.add_argument("--output_dir", type=str,
#                    default=r"C:\Users\Vineet9.Yadav\Desktop\TTA\results\imagenetc",
#                    help="Where to save summary.csv, results.json, reliability/")

#     # ── Experiment ────────────────────────────────────────────────────────
#     p.add_argument("--severity",    type=int, default=5,
#                    help="Corruption severity (1-5). 5 = hardest.")
#     p.add_argument("--corruptions", type=str, nargs="+", default=None,
#                    help="Subset of corruptions. Omit to run all 15. "
#                         "Example: --corruptions gaussian_noise impulse_noise")

#     # ── Hardware — tuned for RTX 3080/4080 (10-16 GB VRAM) ───────────────
#     # 128 uses ~8-9 GB VRAM with ResNet-50. Drop to 64 if OOM.
#     p.add_argument("--batch_size",  type=int, default=128,
#                    help="128 for RTX 3080/4080. Use 64 if OOM.")
#     p.add_argument("--num_workers", type=int, default=8,
#                    help="8 for local NVMe. Drop to 4 on HDD.")

#     # ── CGT-TTA hyperparameters ───────────────────────────────────────────
#     p.add_argument("--cgs_thresh", type=float, default=0.4,
#                    help="CGS threshold (from CIFAR-10-C ablation).")
#     p.add_argument("--k",          type=int,   default=20,
#                    help="k for k-NN local accuracy estimation.")
#     p.add_argument("--bank_size",  type=int,   default=2048,
#                    help="Max FIFO feature bank size.")

#     # ── Optimiser ─────────────────────────────────────────────────────────
#     p.add_argument("--lr",    type=float, default=1e-3)
#     p.add_argument("--steps", type=int,   default=1,
#                    help="Gradient steps per batch.")

#     return p.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     evaluate_all(args)










"""
evaluate_imagenet_c.py  —  CGT-TTA on ImageNet-C (Severity 5)
==============================================================
Compares three methods on ImageNet-C at severity 5:
  1. Source-only   — frozen ResNet-50, no adaptation
  2. TENT          — entropy minimisation over BN layers
  3. CGT-TTA       — calibration-gated TTA (our method, v3)

Supports two common ImageNet-C folder layouts on Kaggle:

  Layout A (standard):
    <data_dir>/<corruption>/<severity>/<class_id>/<image>.JPEG

  Layout B (flat per-corruption):
    <data_dir>/<corruption>/<severity>/<image>.JPEG
    <data_dir>/labels.npy   (or val_labels.npy)

Auto-detects which layout is present.

Usage (Kaggle notebook cell)
-----------------------------
  !python evaluate_imagenet_c.py \
      --data_dir   /kaggle/input/imagenet-c/ImageNet-C \
      --output_dir /kaggle/working/results_imagenetc \
      --severity   5 \
      --batch_size 64 \
      --cgs_thresh 0.4 \
      --k          20 \
      --bank_size  2048 \
      --num_workers 4

To run only a subset of corruptions (faster for testing):
  --corruptions gaussian_noise shot_noise impulse_noise

Outputs
-------
  summary.csv          — accuracy + ECE per method × corruption
  results.json         — full structured results
  reliability/         — reliability diagram CSVs
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)

def progress(current, total, prefix="", width=30):
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r{prefix} [{bar}] {current}/{total}"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET — auto-detects Layout A (ImageFolder) or Layout B (flat + labels)
# ─────────────────────────────────────────────────────────────────────────────

class ImageNetCDataset(Dataset):
    """
    Loads one (corruption, severity) slice of ImageNet-C.

    Layout A — standard ImageNet folder structure:
      <data_dir>/<corruption>/<severity>/<synset_id>/<image>.JPEG
      Labels inferred from folder names mapped to ImageNet class indices.

    Layout B — flat images + separate label file:
      <data_dir>/<corruption>/<severity>/<image>.JPEG
      Labels from <data_dir>/labels.npy or val_labels.npy (50000,)
    """

    def __init__(self, data_dir: str, corruption: str,
                 severity: int, transform=None):
        self.transform = transform
        self.samples   = []   # list of (path, label)

        base = Path(data_dir) / corruption / str(severity)
        if not base.exists():
            raise FileNotFoundError(f"Path not found: {base}")

        # ── Detect layout ────────────────────────────────────────────────────
        subdirs = [d for d in base.iterdir() if d.is_dir()]

        if subdirs:
            # Layout A: subfolders = class synset ids
            log(f"    Layout A detected (ImageFolder): {base}")
            self._load_layout_a(base)
        else:
            # Layout B: flat images + labels.npy
            log(f"    Layout B detected (flat + labels): {base}")
            self._load_layout_b(data_dir, base)

    def _load_layout_a(self, base: Path):
        """
        ImageNet-C standard layout.
        Maps synset folder names to class indices 0–999 using
        sorted order (same as torchvision.datasets.ImageFolder).
        """
        class_dirs = sorted([d for d in base.iterdir() if d.is_dir()])
        class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        IMG_EXTS = {'.jpeg', '.jpg', '.png', '.bmp', '.webp'}
        for cls_dir in class_dirs:
            label = class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(img_path), label))

        self.samples.sort(key=lambda x: x[0])   # deterministic order

    def _load_layout_b(self, data_dir: str, base: Path):
        """
        Flat layout — images sorted alphabetically, labels from .npy file.
        Looks for labels in several common locations.
        """
        IMG_EXTS = {'.jpeg', '.jpg', '.png', '.bmp', '.webp'}
        imgs = sorted([
            str(p) for p in base.iterdir()
            if p.suffix.lower() in IMG_EXTS
        ])

        # find labels file
        label_candidates = [
            Path(data_dir) / "labels.npy",
            Path(data_dir) / "val_labels.npy",
            Path(data_dir).parent / "labels.npy",
        ]
        label_path = next((p for p in label_candidates if p.exists()), None)
        if label_path is None:
            raise FileNotFoundError(
                f"No labels.npy found. Searched: {label_candidates}"
            )

        labels = np.load(label_path)   # (50000,) for full ImageNet-C
        if len(labels) != len(imgs):
            # try to match by taking first len(imgs) labels
            log(f"    Warning: {len(labels)} labels, {len(imgs)} images — using first {len(imgs)}")
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
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_loader(data_dir, corruption, severity, batch_size, num_workers):
    dataset = ImageNetCDataset(
        data_dir, corruption, severity, transform=make_transform()
    )
    log(f"    Loaded {len(dataset):,} samples")
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model() -> nn.Module:
    """Load torchvision ResNet-50 pretrained on ImageNet."""
    log("  Loading ResNet-50 (torchvision, ImageNet pretrained)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(confidences, predictions, labels, n_bins=15):
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
    bins = np.linspace(0.0, 1.0, n_bins + 1)
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
def run_source_only(model, loader):
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
# METHOD 2: TENT
# ─────────────────────────────────────────────────────────────────────────────

def configure_tent(model, layer4_only=True):
    """
    Prepare model for TENT-style BN adaptation.

    layer4_only=True  (default for ImageNet-C):
        Adapt only the 10 BN layers in ResNet-50's layer4.
        This is much more stable than adapting all 53 BN layers
        because fewer parameters → smaller effective update per batch.
        Sufficient to align high-level feature statistics to the shift.

    layer4_only=False:
        Adapt all BN layers (original TENT). Use for CIFAR-scale models
        where there are far fewer BN layers and classes.
    """
    model.train()
    model.requires_grad_(False)

    for name, m in model.named_modules():
        is_bn = isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
        if not is_bn:
            continue
        # layer4_only: only enable grad for layer4 BN layers
        if layer4_only and 'layer4' not in name:
            # still set to eval-like behavior (use batch stats)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var  = None
            continue
        m.requires_grad_(True)
        m.track_running_stats = False
        m.running_mean = None
        m.running_var  = None

    return model


def run_tent(model, loader, lr=1e-3, steps=1, entropy_thresh=0.95):
    """
    TENT with entropy guard to prevent collapse on ImageNet-C.

    At high severity, ResNet-50 outputs near-uniform distributions
    (entropy ≈ log(1000) = 6.9 nats). TENT's entropy-minimisation
    gradient becomes near-zero and can push BN params to a degenerate
    fixed point where the model predicts a single class for everything.

    Fix: skip the BN update for any batch whose mean entropy exceeds
    entropy_thresh * log(num_classes). This lets TENT adapt when there
    is a meaningful signal while preventing collapse on hopeless batches.
    """
    model = configure_tent(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999)
    )
    all_conf, all_pred, all_labels = [], [], []
    skipped_batches = 0
    total_batches   = 0

    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        total_batches += 1

        # ── entropy guard ─────────────────────────────────────────────────
        with torch.no_grad():
            probs_check = F.softmax(model(imgs), dim=1)
            H = -(probs_check * torch.log(probs_check + 1e-8)).sum(dim=1).mean()
            num_classes  = probs_check.size(1)
            H_max        = np.log(num_classes)   # maximum possible entropy
            H_ratio      = H.item() / H_max

        if H_ratio > entropy_thresh:
            # predictions are near-uniform → skip BN update, use source pred
            conf, pred = probs_check.max(dim=1)
            skipped_batches += 1
        else:
            # entropy is meaningful → run TENT update
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

    if skipped_batches > 0:
        skip_pct = 100 * skipped_batches / total_batches
        log(f"    TENT: skipped {skipped_batches}/{total_batches} batches "
            f"({skip_pct:.0f}%) — entropy above threshold")

    confs  = np.concatenate(all_conf)
    preds  = np.concatenate(all_pred)
    labels = np.concatenate(all_labels)
    return (float((preds == labels).mean()),
            compute_ece(confs, preds, labels),
            reliability_data(confs, preds, labels))


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: CGT-TTA  (v3 — decoupled design)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureBank:
    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.features = deque(maxlen=max_size)
        self.correct  = deque(maxlen=max_size)

    def __len__(self):
        return len(self.features)

    def update(self, feats, correct_flags):
        for f, c in zip(feats, correct_flags):
            self.features.append(f)
            self.correct.append(bool(c))

    def local_accuracy(self, query, k):
        if len(self) < k:
            return 0.5
        bank = np.stack(self.features)
        sims = bank @ query
        idx  = np.argpartition(sims, -k)[-k:]
        return float(np.array(self.correct)[idx].mean())


def _feat_hook(storage):
    def hook(module, inp, out):
        # ResNet-50 avgpool: (B, 2048, 1, 1) → (B, 2048)
        f = out.squeeze(-1).squeeze(-1)
        storage["feats"] = F.normalize(f, dim=1).detach().cpu().numpy()
    return hook


def _warmup_bank(model, loader, bank, storage, max_batches=32):
    model.eval()
    seen = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            if seen >= max_batches:
                break
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            feats  = storage["feats"]
            preds  = logits.argmax(dim=1).cpu().numpy()
            bank.update(feats, preds == lbls.numpy())
            seen += 1
            if len(bank) >= bank.max_size:
                break
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var  = None


def run_cgt_tta(model, loader, k=20, bank_size=2048,
                cgs_thresh=0.4, lr=1e-3, steps=1, warm_start=True,
                entropy_thresh=0.95):
    """
    CGT-TTA v3 — decoupled BN alignment + per-sample prediction trust gate.
    Identical logic to the CIFAR-10-C version; feature dim is 2048 for ResNet-50.
    """
    model     = configure_tent(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999)
    )

    storage     = {}
    hook_handle = model.avgpool.register_forward_hook(_feat_hook(storage))
    bank        = FeatureBank(max_size=bank_size)

    if warm_start:
        _warmup_bank(model, loader, bank, storage, max_batches=32)

    all_conf, all_pred, all_labels = [], [], []
    total_samples = 0
    gated_samples = 0

    for imgs, lbls in loader:
        imgs = imgs.to(DEVICE)
        B    = imgs.size(0)

        # (1) source forward pass
        with torch.no_grad():
            logits_src = model(imgs)
            probs_src  = F.softmax(logits_src, dim=1)
            conf_src_t, pred_src_t = probs_src.max(dim=1)

        feats_b     = storage["feats"]
        conf_src_np = conf_src_t.cpu().numpy()
        pred_src_np = pred_src_t.cpu().numpy()

        # (2) CGS per sample
        cgs = np.zeros(B)
        for i in range(B):
            la     = bank.local_accuracy(feats_b[i], k=k)
            cgs[i] = float(conf_src_np[i]) - la
        gate_mask = cgs > cgs_thresh

        # (3) BN alignment — always on full batch (with entropy guard)
        # Skip update if batch entropy is near-maximum (near-uniform predictions)
        # to prevent the same collapse that affects vanilla TENT.
        with torch.no_grad():
            H_check = -(probs_src * torch.log(probs_src + 1e-8)).sum(dim=1).mean()
            H_ratio = H_check.item() / np.log(probs_src.size(1))
        if H_ratio <= entropy_thresh:
            for _ in range(steps):
                optimizer.zero_grad()
                probs_a = F.softmax(model(imgs), dim=1)
                entropy = -(probs_a * torch.log(probs_a + 1e-8)).sum(dim=1).mean()
                entropy.backward()
                optimizer.step()

        # (4) adapted forward pass
        with torch.no_grad():
            probs_ada = F.softmax(model(imgs), dim=1)
            conf_ada_t, pred_ada_t = probs_ada.max(dim=1)

        conf_ada_np = conf_ada_t.cpu().numpy()
        pred_ada_np = pred_ada_t.cpu().numpy()

        # (5) prediction trust gate
        final_conf = np.where(gate_mask, conf_ada_np, conf_src_np)
        final_pred = np.where(gate_mask, pred_ada_np, pred_src_np)

        # (6) update bank
        bank.update(feats_b, final_pred == lbls.numpy())

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
# MAIN EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(args):
    t0 = time.time()

    log(f"\n{'='*72}")
    log(f"  CGT-TTA — ImageNet-C Evaluation")
    log(f"  Device     : {DEVICE}" +
        (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    log(f"  Data dir   : {args.data_dir}")
    log(f"  Severity   : {args.severity}")
    log(f"  Batch size : {args.batch_size}")
    log(f"  CGS thresh : {args.cgs_thresh}  k={args.k}  bank={args.bank_size}")
    log(f"{'='*72}\n")

    # ── validate data directory ───────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log(f"ERROR: data_dir not found: {data_dir}")
        log("Common Kaggle paths to check:")
        log("  /kaggle/input/imagenet-c/ImageNet-C")
        log("  /kaggle/input/imagenet-c/imagenet-c")
        # try to auto-find
        for p in Path("/kaggle/input").glob("**/gaussian_noise"):
            log(f"  Found corruption folder at: {p.parent}")
        sys.exit(1)

    # confirm at least one corruption folder exists
    corruptions_to_run = args.corruptions or CORRUPTIONS
    missing = [c for c in corruptions_to_run
               if not (data_dir / c / str(args.severity)).exists()]
    if missing:
        log(f"WARNING: these corruptions not found at severity {args.severity}: {missing}")
        corruptions_to_run = [c for c in corruptions_to_run if c not in missing]
        if not corruptions_to_run:
            log("ERROR: no valid corruptions found. Check --data_dir and --severity.")
            sys.exit(1)

    log(f"  Running {len(corruptions_to_run)} corruptions: {corruptions_to_run}\n")

    base_model = load_model()
    log(f"  ResNet-50 loaded.\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reliability").mkdir(exist_ok=True)

    results   = defaultdict(dict)
    rel_store = defaultdict(dict)

    # ── header ────────────────────────────────────────────────────────────────
    W = 22
    hdr = (f"{'Corruption':<{W}}  "
           f"{'Src Acc':>8} {'TENT Acc':>9} {'CGT Acc':>8}  "
           f"{'Src ECE':>8} {'TENT ECE':>9} {'CGT ECE':>8}  "
           f"{'ΔAcc':>7} {'ΔECE':>7}  {'Gate%':>6}")
    log(hdr)
    log("─" * len(hdr))

    for i, corruption in enumerate(corruptions_to_run):
        sys.stdout.write(progress(i+1, len(corruptions_to_run),
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
        results["source_only"][corruption] = {"acc": src_acc, "ece": src_ece}
        rel_store["source_only"][corruption] = src_rel

        # TENT
        m = copy.deepcopy(base_model)
        tent_acc, tent_ece, tent_rel = run_tent(
            m, loader, lr=args.lr, steps=args.steps,
            entropy_thresh=args.entropy_thresh
        )
        results["tent"][corruption] = {"acc": tent_acc, "ece": tent_ece}
        rel_store["tent"][corruption] = tent_rel

        # CGT-TTA
        m = copy.deepcopy(base_model)
        cgt_acc, cgt_ece, cgt_rel, cgt_ar = run_cgt_tta(
            m, loader,
            k=args.k, bank_size=args.bank_size,
            cgs_thresh=args.cgs_thresh,
            lr=args.lr, steps=args.steps,
            warm_start=True,
            entropy_thresh=args.entropy_thresh,
        )
        results["cgt_tta"][corruption] = {
            "acc": cgt_acc, "ece": cgt_ece, "gate_rate": cgt_ar
        }
        rel_store["cgt_tta"][corruption] = cgt_rel

        delta_acc = cgt_acc - tent_acc
        delta_ece = tent_ece - cgt_ece   # positive = CGT better

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
        log(f"  SUMMARY — Mean over {len(done)} corruptions (severity {args.severity})")
        log(f"{'='*72}")

        def mean_r(method, key):
            return np.mean([results[method][c][key] for c in done])

        src_acc_m   = mean_r("source_only", "acc")
        src_ece_m   = mean_r("source_only", "ece")
        tent_acc_m  = mean_r("tent",        "acc")
        tent_ece_m  = mean_r("tent",        "ece")
        cgt_acc_m   = mean_r("cgt_tta",     "acc")
        cgt_ece_m   = mean_r("cgt_tta",     "ece")
        cgt_ar_m    = mean_r("cgt_tta",     "gate_rate")

        log(f"\n  {'Method':<18} {'Accuracy':>10} {'ECE':>10} {'Gate%':>8}")
        log(f"  {'─'*50}")
        log(f"  {'Source-only':<18} {src_acc_m*100:>9.2f}% {src_ece_m*100:>9.2f}%")
        log(f"  {'TENT':<18} {tent_acc_m*100:>9.2f}% {tent_ece_m*100:>9.2f}%")
        log(f"  {'CGT-TTA (ours)':<18} {cgt_acc_m*100:>9.2f}% {cgt_ece_m*100:>9.2f}% {cgt_ar_m*100:>7.1f}%")
        log(f"\n  ΔAcc (CGT − TENT) : {(cgt_acc_m - tent_acc_m)*100:>+.3f}pp")
        log(f"  ΔECE (TENT − CGT) : {(tent_ece_m - cgt_ece_m)*100:>+.3f}pp  "
            f"({'improvement' if tent_ece_m > cgt_ece_m else 'WORSE'})")
        log(f"  Avg gate rate     : {cgt_ar_m*100:.1f}%")

    elapsed = time.time() - t0
    log(f"\n  Total time: {elapsed/60:.1f} min")

    # ── save summary CSV ──────────────────────────────────────────────────────
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w") as f:
        f.write("corruption,src_acc,src_ece,tent_acc,tent_ece,"
                "cgt_acc,cgt_ece,cgt_gate_rate,"
                "delta_acc,delta_ece\n")
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

    # ── save JSON ─────────────────────────────────────────────────────────────
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({
            "args": vars(args),
            "severity": args.severity,
            "results": {m: dict(v) for m, v in results.items()},
        }, f, indent=2)
    log(f"  Saved: {json_path}")

    # ── save reliability data ─────────────────────────────────────────────────
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
        description="CGT-TTA on ImageNet-C",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Paths — local VS Code defaults ────────────────────────────────────
    # data_dir:   folder containing gaussian_noise/, shot_noise/, etc.
    # output_dir: where results are written
    # Override via CLI if your folder is elsewhere.
    p.add_argument("--data_dir",   type=str,
                   default=r"C:\Users\Vineet9.Yadav\Desktop\TTA\ImageNet-C",
                   help="Root of ImageNet-C: <data_dir>/<corruption>/<severity>/...")
    p.add_argument("--output_dir", type=str,
                   default=r"C:\Users\Vineet9.Yadav\Desktop\TTA\results\imagenetc",
                   help="Where to save summary.csv, results.json, reliability/")

    # ── Experiment ────────────────────────────────────────────────────────
    p.add_argument("--severity",    type=int, default=3,
                   help="Corruption severity (1-5). 3 recommended for ImageNet-C.")
    p.add_argument("--corruptions", type=str, nargs="+", default=None,
                   help="Subset of corruptions. Omit to run all 15. "
                        "Example: --corruptions gaussian_noise impulse_noise")

    # ── Hardware — tuned for RTX 3080/4080 (10-16 GB VRAM) ───────────────
    # 128 uses ~8-9 GB VRAM with ResNet-50. Drop to 64 if OOM.
    p.add_argument("--batch_size",  type=int, default=128,
                   help="128 for RTX 3080/4080. Use 64 if OOM.")
    p.add_argument("--num_workers", type=int, default=8,
                   help="8 for local NVMe. Drop to 4 on HDD.")

    # ── CGT-TTA hyperparameters ───────────────────────────────────────────
    p.add_argument("--cgs_thresh", type=float, default=0.4,
                   help="CGS threshold (from CIFAR-10-C ablation).")
    p.add_argument("--k",          type=int,   default=20,
                   help="k for k-NN local accuracy estimation.")
    p.add_argument("--bank_size",  type=int,   default=2048,
                   help="Max FIFO feature bank size.")

    # ── Optimiser ─────────────────────────────────────────────────────────
    p.add_argument("--lr",    type=float, default=1e-3,
                   help="Learning rate for BN adaptation.")
    p.add_argument("--steps", type=int,   default=1,
                   help="Gradient steps per batch.")
    p.add_argument("--entropy_thresh", type=float, default=0.95,
                   help=("Skip BN update if mean batch entropy > thresh * log(num_classes). "
                         "Prevents TENT collapse when predictions are near-uniform. "
                         "0.95 skips when entropy > 95%% of max. Set 1.0 to disable."))

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_all(args)