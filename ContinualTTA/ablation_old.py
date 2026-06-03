# # =============================================================================
# # ContinualTTA — Component Ablation Study (WACV 2027)
# #
# # Run from terminal:
# #   python ablation.py --dataset cifar10c --setting b
# #   python ablation.py --dataset cifar10c --setting a
# #   python ablation.py --dataset imagenetc
# #
# # Ablations (each removes exactly one component from full ContinualTTA):
# #   A0  Full ContinualTTA          — all components (reference)
# #   A1  No JS Detector             — always adapt (remove gating)
# #   A2  No Reliable Filter         — use all samples regardless of entropy
# #   A3  No Prototype Bank          — entropy loss only, no consistency term
# #   A4  No Prototype + No Filter   — plain entropy minimisation (≈ TENT)
# #   A5  KL Detector                — replace JS with KL divergence
# #   A6  Entropy Detector           — replace JS with entropy threshold
# #   A7  EMA Decay β=0.5            — weaker prototype memory
# #   A8  EMA Decay β=0.99           — stronger prototype memory
# #   A9  JS Threshold τ=0.01        — more aggressive adaptation
# #   A10 JS Threshold τ=0.05        — more conservative adaptation
# #
# # All ablations:
# #   - Use identical BN-only adaptation (setup_bn / setup_bn_imagenet)
# #   - Use identical optimiser (Adam, lr=1e-3)
# #   - Differ in EXACTLY ONE component from A0
# #   - Save individual CSVs — merge later with merge_ablations()
# #
# # Output:
# #   results/ablations/A0_Full.csv
# #   results/ablations/A1_NoDetector.csv
# #   ...
# #   results/ablations/ablation_table.tex  (generated after all runs)
# # =============================================================================

# import os
# import copy
# import math
# import argparse
# import platform
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from torchvision.datasets import ImageFolder
# from PIL import Image

# # =============================================================================
# # CONFIG — update paths
# # =============================================================================

# CIFAR_MODEL_PATH = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
# CIFAR_DATA_DIR   = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
# IMAGENET_DATA_DIR = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\ImageNet-C"

# RESULTS_DIR  = os.path.join("results", "ablations")
# NUM_WORKERS  = 0 if platform.system() == "Windows" else 2
# DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# # CIFAR-10-C settings
# CIFAR_BATCH_SIZE  = 32
# CIFAR_CLASSES     = 10
# CIFAR_SEVERITIES  = [1, 2, 3, 4, 5]
# CIFAR_SEED        = 42

# # ImageNet-C settings
# IMAGENET_BATCH_SIZE = 64
# IMAGENET_CLASSES    = 1000
# IMAGENET_SEVERITY   = 5

# # Shared hyperparameters (identical across all ablations)
# LR           = 1e-3
# ALPHA        = 0.5           # prototype loss weight
# JS_THRESHOLD = 0.02          # default JS threshold
# EMA_DECAY    = 0.9           # default prototype EMA decay
# E_MARGIN_FACTOR = 0.4        # entropy margin = factor * log(C)
# SAR_RHO      = 0.05

# ALL_CORRUPTIONS = [
#     "gaussian_noise", "shot_noise",    "impulse_noise",
#     "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
#     "snow",           "frost",         "fog",           "brightness",
#     "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
# ]

# # =============================================================================
# # ABLATION REGISTRY
# # Each entry: (id, name, description, kwargs_override)
# # kwargs_override modifies only the specific component being ablated.
# # =============================================================================

# ABLATIONS = [
#     # id   name                  description                           kwargs
#     ("A0",  "Full",              "All components (reference)",          {}),
#     ("A1",  "NoDetector",        "Always adapt — no JS gating",         {"use_detector": False}),
#     ("A2",  "NoFilter",          "No entropy filter — all samples",     {"use_filter": False}),
#     ("A3",  "NoPrototype",       "No prototype bank — entropy only",    {"use_prototype": False}),
#     ("A4",  "NoProtoNoFilter",   "No prototype + no filter (≈TENT)",    {"use_prototype": False,
#                                                                           "use_filter": False}),
#     ("A5",  "KLDetector",        "KL divergence detector instead of JS",{"detector_type": "kl"}),
#     ("A6",  "EntropyDetector",   "Entropy threshold detector",          {"detector_type": "entropy"}),
#     ("A7",  "WeakMemory",        "Weaker EMA decay β=0.5",              {"ema_decay": 0.5}),
#     ("A8",  "StrongMemory",      "Stronger EMA decay β=0.99",           {"ema_decay": 0.99}),
#     ("A9",  "AggressiveGating",  "More aggressive JS threshold τ=0.01", {"js_threshold": 0.01}),
#     ("A10", "ConservativeGating","More conservative JS threshold τ=0.05",{"js_threshold": 0.05}),
#     ("A11", "ConservativeGating","More conservative JS threshold τ=0.1",{"js_threshold": 0.1}),
#     ("A12", "ConservativeGating","More conservative JS threshold τ=0.06",{"js_threshold": 0.06}),
#     ("A13", "ConservativeGating","More conservative JS threshold τ=0.04",{"js_threshold": 0.04}),
#     ("A14", "ConservativeGating","More conservative JS threshold τ=0.03",{"js_threshold": 0.03}),
# ]


# # =============================================================================
# # 1. DATASETS
# # =============================================================================

# class CIFAR10C_Dataset(Dataset):
#     def __init__(self, corruption, severity, data_dir):
#         data        = np.load(os.path.join(data_dir, f"{corruption}.npy"), mmap_mode='r')
#         labels      = np.load(os.path.join(data_dir, "labels.npy"),        mmap_mode='r')
#         start       = (severity - 1) * 10000
#         self.images = data[start:start + 10000]
#         self.labels = labels[start:start + 10000]
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         ])
#     def __len__(self): return len(self.labels)
#     def __getitem__(self, idx):
#         return self.transform(Image.fromarray(self.images[idx])), int(self.labels[idx])


# def cifar_loader_single(corruption, severity):
#     return DataLoader(CIFAR10C_Dataset(corruption, severity, CIFAR_DATA_DIR),
#                       batch_size=CIFAR_BATCH_SIZE, shuffle=False,
#                       num_workers=NUM_WORKERS, pin_memory=True)


# def cifar_loader_mixed(severity, seed=CIFAR_SEED):
#     combined = ConcatDataset(
#         [CIFAR10C_Dataset(c, severity, CIFAR_DATA_DIR) for c in ALL_CORRUPTIONS])
#     g = torch.Generator(); g.manual_seed(seed)
#     indices = torch.randperm(len(combined), generator=g).tolist()
#     subset  = torch.utils.data.Subset(combined, indices)
#     return DataLoader(subset, batch_size=CIFAR_BATCH_SIZE, shuffle=False,
#                       num_workers=NUM_WORKERS, pin_memory=True)


# def imagenet_loader(corruption):
#     path = os.path.join(IMAGENET_DATA_DIR, corruption, str(IMAGENET_SEVERITY))
#     dataset = ImageFolder(path, transform=models.ResNet50_Weights.IMAGENET1K_V1.transforms())
#     return DataLoader(dataset, batch_size=IMAGENET_BATCH_SIZE, shuffle=False,
#                       num_workers=NUM_WORKERS, pin_memory=True)


# def load_cifar_model():
#     model = models.resnet50(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, CIFAR_CLASSES)
#     model.load_state_dict(torch.load(CIFAR_MODEL_PATH, map_location=DEVICE))
#     return model.to(DEVICE).eval()


# def load_imagenet_model():
#     model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#     return model.to(DEVICE).eval()


# # =============================================================================
# # 2. HELPERS
# # =============================================================================

# def softmax_entropy(logits):
#     p = logits.softmax(1)
#     return -(p * p.log()).sum(1)


# def eval_loader(model_fn, loader):
#     correct, total = 0, 0
#     for x, y in loader:
#         x, y    = x.to(DEVICE), y.to(DEVICE)
#         logits  = model_fn(x)
#         correct += (logits.argmax(1) == y).sum().item()
#         total   += y.size(0)
#     return 100.0 * correct / total


# def setup_bn_cifar(model):
#     """CIFAR: per-batch BN statistics, only gamma/beta trainable."""
#     model.train(); model.requires_grad_(False)
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#             m.requires_grad_(True)
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var  = None
#     params = [p for m in model.modules()
#               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
#               for p in m.parameters() if p.requires_grad]
#     return model, params


# def setup_bn_imagenet(model):
#     """ImageNet: keep pretrained BN stats frozen, only gamma/beta trainable."""
#     model.train(); model.requires_grad_(False)
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#             m.requires_grad_(True)
#             m.track_running_stats = True
#             m.momentum = 0
#     params = [p for m in model.modules()
#               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
#               for p in m.parameters() if p.requires_grad]
#     return model, params


# # =============================================================================
# # 3. ABLATION-AWARE COMPONENTS
# # Each component checks kwargs to decide its behaviour.
# # This keeps all logic in one place — easy to audit and modify.
# # =============================================================================

# class ShiftDetector:
#     """
#     Unified shift detector supporting multiple divergence types.
#     Controlled by detector_type: 'js' | 'kl' | 'entropy' | 'none'
#     """
#     def __init__(self, detector_type="js", threshold=JS_THRESHOLD, ema=0.9):
#         self.detector_type = detector_type
#         self.threshold     = threshold
#         self.ema           = ema
#         self.reference     = None
#         self.ema_entropy   = None    # used for entropy detector only

#     def should_adapt(self, logits):
#         with torch.no_grad():
#             p_t = logits.softmax(1).mean(0)

#             # No detector — always adapt
#             if self.detector_type == "none":
#                 return True

#             # Entropy detector — adapt when batch entropy exceeds threshold
#             if self.detector_type == "entropy":
#                 h = -(p_t * p_t.log().clamp(min=-1e9)).sum()
#                 if self.ema_entropy is None:
#                     self.ema_entropy = h.item()
#                 adapt = h.item() > self.threshold * math.log(logits.size(1))
#                 self.ema_entropy = 0.9 * self.ema_entropy + 0.1 * h.item()
#                 return adapt

#             # JS or KL detector — need reference distribution
#             if self.reference is None:
#                 self.reference = p_t.clone()
#                 return True

#             p_ref = self.reference
#             m     = 0.5 * (p_ref + p_t)

#             if self.detector_type == "js":
#                 kl_1 = F.kl_div(m.log().unsqueeze(0),
#                                  p_ref.unsqueeze(0), reduction="batchmean")
#                 kl_2 = F.kl_div(m.log().unsqueeze(0),
#                                  p_t.unsqueeze(0),   reduction="batchmean")
#                 divergence = 0.5 * (kl_1 + kl_2)

#             elif self.detector_type == "kl":
#                 # KL(p_ref || p_t) — asymmetric, unbounded
#                 divergence = F.kl_div(p_t.log().unsqueeze(0),
#                                       p_ref.unsqueeze(0), reduction="batchmean")

#             # Update reference AFTER computing divergence
#             self.reference = self.ema * self.reference + (1 - self.ema) * p_t
#             return divergence.item() > self.threshold


# class PrototypeBankModule(nn.Module):
#     """
#     EMA prototype memory bank.
#     Controlled by ema_decay kwarg — supports ablations A7 and A8.
#     """
#     def __init__(self, num_classes, feat_dim, ema_decay=EMA_DECAY):
#         super().__init__()
#         self.decay = ema_decay
#         self.register_buffer("prototypes",  torch.zeros(num_classes, feat_dim))
#         self.register_buffer("initialised", torch.zeros(num_classes).bool())

#     @torch.no_grad()
#     def update(self, features, pseudo_labels):
#         for c in pseudo_labels.unique():
#             mask = (pseudo_labels == c)
#             mf   = features[mask].mean(0)
#             if self.initialised[c]:
#                 self.prototypes[c] = self.decay * self.prototypes[c] + (1-self.decay) * mf
#             else:
#                 self.prototypes[c] = mf
#                 self.initialised[c] = True

#     def consistency_loss(self, features, pseudo_labels):
#         loss, count = torch.tensor(0.0, device=features.device), 0
#         for c in pseudo_labels.unique():
#             if not self.initialised[c]: continue
#             mask  = (pseudo_labels == c)
#             loss += F.mse_loss(features[mask],
#                                self.prototypes[c].unsqueeze(0).expand(mask.sum(), -1))
#             count += 1
#         return loss / max(count, 1)


# # =============================================================================
# # 4. ABLATION VARIANT FACTORY
# # Single function that builds any ablation by overriding specific kwargs.
# # This is the heart of the ablation framework — clean and auditable.
# # =============================================================================

# def make_ablation_variant(source, setup_bn_fn, num_classes, feat_dim, **kwargs):
#     """
#     Build a ContinualTTA variant with specific components enabled/disabled.

#     kwargs (all optional, fall back to defaults):
#         use_detector   (bool)  : gate adaptation with shift detector [default: True]
#         use_filter     (bool)  : apply entropy-based reliable filter  [default: True]
#         use_prototype  (bool)  : use prototype consistency loss       [default: True]
#         detector_type  (str)   : 'js' | 'kl' | 'entropy' | 'none'   [default: 'js']
#         js_threshold   (float) : JS / KL detection threshold         [default: 0.02]
#         ema_decay      (float) : prototype EMA decay β               [default: 0.9]
#     """
#     # Parse kwargs with defaults
#     use_detector  = kwargs.get("use_detector",  True)
#     use_filter    = kwargs.get("use_filter",    True)
#     use_prototype = kwargs.get("use_prototype", True)
#     detector_type = kwargs.get("detector_type", "js") if use_detector else "none"
#     js_threshold  = kwargs.get("js_threshold",  JS_THRESHOLD)
#     ema_decay     = kwargs.get("ema_decay",      EMA_DECAY)

#     e_margin = E_MARGIN_FACTOR * math.log(num_classes)

#     # Build components
#     model, params = setup_bn_fn(copy.deepcopy(source))
#     opt      = torch.optim.Adam(params, lr=LR)
#     detector = ShiftDetector(detector_type=detector_type,
#                              threshold=js_threshold)
#     bank     = PrototypeBankModule(num_classes, feat_dim,
#                                    ema_decay=ema_decay).to(DEVICE)
#     captured = {}
#     handle   = model.avgpool.register_forward_hook(
#         lambda m, i, o: captured.update({"feat": o.flatten(1)}))

#     @torch.enable_grad()
#     def fn(x):
#         logits        = model(x)
#         features      = captured["feat"]
#         pseudo_labels = logits.argmax(1).detach()

#         # ── Gate 1: shift detector ────────────────────────────────────────────
#         if not detector.should_adapt(logits.detach()):
#             return logits

#         # ── Gate 2: reliable sample filter ───────────────────────────────────
#         if use_filter:
#             entropy  = softmax_entropy(logits)
#             reliable = entropy < e_margin
#             if reliable.sum() == 0:
#                 return logits
#         else:
#             # No filter — use all samples
#             entropy  = softmax_entropy(logits)
#             reliable = torch.ones(x.size(0), dtype=torch.bool, device=DEVICE)

#         # ── Loss computation ──────────────────────────────────────────────────
#         loss = entropy[reliable].mean()

#         if use_prototype:
#             proto_loss = bank.consistency_loss(
#                 features[reliable].detach(),
#                 pseudo_labels[reliable])
#             loss = loss + ALPHA * proto_loss

#         # ── Parameter update (BN gamma/beta only) ────────────────────────────
#         loss.backward()
#         opt.step()
#         opt.zero_grad()

#         # ── Prototype bank update ─────────────────────────────────────────────
#         if use_prototype:
#             bank.update(features[reliable].detach(), pseudo_labels[reliable])

#         return logits

#     fn._handle = handle
#     return fn


# # =============================================================================
# # 5. EVALUATION LOOPS
# # =============================================================================

# def eval_cifar_setting_b(source_model, fn):
#     """Setting B: continual sequential, S1-S5 averaged."""
#     all_sev = {}
#     for severity in CIFAR_SEVERITIES:
#         results = {}
#         # Fresh function per severity (reset state)
#         for corruption in ALL_CORRUPTIONS:
#             loader = cifar_loader_single(corruption, severity)
#             results[corruption] = eval_loader(fn, loader)
#             del loader
#             torch.cuda.empty_cache()
#         all_sev[severity] = results

#     # Average over severities
#     averaged = {}
#     for corruption in ALL_CORRUPTIONS:
#         averaged[corruption] = np.mean(
#             [all_sev[s][corruption] for s in CIFAR_SEVERITIES])
#     return averaged


# def eval_cifar_setting_a(source_model, setup_bn_fn, num_classes, feat_dim, **kwargs):
#     """Setting A: mixed i.i.d. per severity, fresh model each severity."""
#     sev_results = {}
#     for severity in CIFAR_SEVERITIES:
#         fn = make_ablation_variant(source_model, setup_bn_fn,
#                                    num_classes, feat_dim, **kwargs)
#         loader = cifar_loader_mixed(severity)
#         acc    = eval_loader(fn, loader)
#         sev_results[severity] = acc
#         del loader; torch.cuda.empty_cache()
#     return np.mean(list(sev_results.values())), sev_results


# def eval_imagenet_sequential(source_model, fn):
#     """ImageNet-C: continual sequential, severity 5."""
#     results = {}
#     for corruption in ALL_CORRUPTIONS:
#         loader = imagenet_loader(corruption)
#         results[corruption] = eval_loader(fn, loader)
#         del loader; torch.cuda.empty_cache()
#     return results


# # =============================================================================
# # 6. MAIN RUNNER — runs one ablation at a time
# # =============================================================================

# def run_ablation(ablation_id, dataset, setting, source_model,
#                  setup_bn_fn, num_classes, feat_dim):
#     """Run one ablation variant and save results to CSV."""

#     # Find ablation config
#     config = next((a for a in ABLATIONS if a[0] == ablation_id), None)
#     if config is None:
#         raise ValueError(f"Unknown ablation ID: {ablation_id}. "
#                          f"Choose from {[a[0] for a in ABLATIONS]}")

#     abl_id, abl_name, abl_desc, abl_kwargs = config
#     print(f"\n{'='*60}")
#     print(f"Ablation {abl_id}: {abl_name}")
#     print(f"Description: {abl_desc}")
#     print(f"Kwargs: {abl_kwargs if abl_kwargs else 'none (full method)'}")
#     print(f"Dataset: {dataset.upper()}  |  Setting: {setting.upper()}")
#     print(f"{'='*60}\n")

#     # Build fresh ablation function
#     fn = make_ablation_variant(source_model, setup_bn_fn,
#                                num_classes, feat_dim, **abl_kwargs)

#     # Run evaluation
#     if dataset == "cifar10c":
#         if setting == "b":
#             results = eval_cifar_setting_b(source_model, fn)
#         else:
#             mean_a, sev_results = eval_cifar_setting_a(
#                 source_model, setup_bn_fn, num_classes, feat_dim, **abl_kwargs)
#             results = {f"S{s}": sev_results[s] for s in CIFAR_SEVERITIES}
#             results["Mean"] = mean_a
#     elif dataset == "imagenetc":
#         results = eval_imagenet_sequential(source_model, fn)
#     else:
#         raise ValueError(f"Unknown dataset: {dataset}")

#     # Print results
#     print(f"\nResults for {abl_id} — {abl_name}:")
#     for k, v in results.items():
#         if k != "Mean":
#             print(f"  {k:<24} {v:.1f}%")
#     mean_acc = np.mean([v for k, v in results.items()
#                         if k not in ("Mean",)])
#     print(f"  {'Mean':<24} {mean_acc:.1f}%")

#     # Save CSV
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     csv_path = os.path.join(RESULTS_DIR, f"{abl_id}_{abl_name}.csv")
#     with open(csv_path, "w") as f:
#         f.write(f"corruption,{abl_id}_{abl_name}\n")
#         for k, v in results.items():
#             if k != "Mean":
#                 f.write(f"{k},{v:.1f}\n")
#         f.write(f"Mean,{mean_acc:.1f}\n")
#     print(f"\nSaved: {csv_path}")

#     return results, mean_acc


# # =============================================================================
# # 7. MERGE AND LATEX — run after all ablations complete
# # =============================================================================

# def merge_ablations_and_latex(dataset="cifar10c", setting="b"):
#     """
#     Merge all individual CSVs into one table and generate LaTeX.
#     Run this after all ablations have finished.

#     Usage:
#         python ablation.py --merge --dataset cifar10c --setting b
#     """
#     print("\nMerging ablation results...")
#     all_results = {}    # ablation_name -> {corruption -> acc}
#     all_means   = {}    # ablation_name -> mean_acc

#     for abl_id, abl_name, abl_desc, _ in ABLATIONS:
#         csv_path = os.path.join(RESULTS_DIR, f"{abl_id}_{abl_name}.csv")
#         if not os.path.isfile(csv_path):
#             print(f"  MISSING: {csv_path} — skipping")
#             continue

#         results = {}
#         with open(csv_path) as f:
#             lines = f.readlines()
#         for line in lines[1:]:   # skip header
#             parts = line.strip().split(",")
#             if parts[0] == "Mean":
#                 all_means[f"{abl_id}_{abl_name}"] = float(parts[1])
#             else:
#                 results[parts[0]] = float(parts[1])
#         all_results[f"{abl_id}_{abl_name}"] = results

#     if not all_results:
#         print("No results found. Run ablations first.")
#         return

#     # ── Print merged table ────────────────────────────────────────────────────
#     col = 12
#     names = list(all_results.keys())
#     header = f"{'Corruption':<24}" + "".join(f"{n[:10]:>{col}}" for n in names)
#     print(f"\n{'═'*len(header)}")
#     print("ABLATION STUDY — merged results")
#     print(f"{'═'*len(header)}")
#     print(header)
#     print("─" * len(header))
#     for corruption in ALL_CORRUPTIONS:
#         row = f"{corruption:<24}"
#         for name in names:
#             val  = all_results[name].get(corruption, float('nan'))
#             row += f"{val:.1f}%".rjust(col)
#         print(row)
#     print("─" * len(header))
#     mean_row = f"{'Mean':<24}"
#     for name in names:
#         mean_row += f"{all_means.get(name, float('nan')):.1f}%".rjust(col)
#     print(mean_row)
#     print(f"{'═'*len(header)}")

#     # ── Generate LaTeX ────────────────────────────────────────────────────────
#     corr_names = {
#         "gaussian_noise": "Gauss. Noise", "shot_noise": "Shot Noise",
#         "impulse_noise": "Impulse",       "defocus_blur": "Defocus",
#         "glass_blur": "Glass",            "motion_blur": "Motion",
#         "zoom_blur": "Zoom",              "snow": "Snow",
#         "frost": "Frost",                 "fog": "Fog",
#         "brightness": "Brightness",       "contrast": "Contrast",
#         "elastic_transform": "Elastic",   "pixelate": "Pixelate",
#         "jpeg_compression": "JPEG",
#     }

#     # Column headers: short ablation names for table
#     col_headers = {
#         "A0_Full":              r"\textbf{Full}",
#         "A1_NoDetector":        r"w/o Detector",
#         "A2_NoFilter":          r"w/o Filter",
#         "A3_NoPrototype":       r"w/o Proto.",
#         "A4_NoProtoNoFilter":   r"w/o Proto.+Filt.",
#         "A5_KLDetector":        r"KL Det.",
#         "A6_EntropyDetector":   r"Entr. Det.",
#         "A7_WeakMemory":        r"$\beta{=}0.5$",
#         "A8_StrongMemory":      r"$\beta{=}0.99$",
#         "A9_AggressiveGating":  r"$\tau{=}0.01$",
#         "A10_ConservativeGating": r"$\tau{=}0.05$",
#     }

#     lines = []
#     lines.append(r"\begin{table*}[t]")
#     lines.append(r"\centering")
#     lines.append(r"\caption{Component ablation of \textsc{ContinualTTA} on CIFAR-10-C "
#                  r"continual sequential shift (Setting~B), S1--S5 averaged. "
#                  r"Each column removes or modifies exactly one component from the "
#                  r"full method (leftmost column). "
#                  r"\textbf{Bold} = best per row.}")
#     lines.append(r"\label{tab:ablation}")
#     lines.append(r"\resizebox{\textwidth}{!}{%")
#     n_cols = len(names)
#     lines.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
#     lines.append(r"\toprule")

#     # Header row
#     header_tex = "Corruption"
#     for name in names:
#         header_tex += " & " + col_headers.get(name, name)
#     lines.append(header_tex + r" \\")
#     lines.append(r"\midrule")

#     # Per-corruption rows
#     for corruption in ALL_CORRUPTIONS:
#         vals = [all_results[name].get(corruption, float('nan'))
#                 for name in names]
#         best = max(v for v in vals if not math.isnan(v))
#         row  = corr_names.get(corruption, corruption)
#         for val in vals:
#             if math.isnan(val):
#                 row += " & ---"
#             elif abs(val - best) < 0.05:
#                 row += f" & \\textbf{{{val:.1f}}}"
#             else:
#                 row += f" & {val:.1f}"
#         lines.append(row + r" \\")

#     lines.append(r"\midrule")

#     # Mean row
#     mean_vals = [all_means.get(name, float('nan')) for name in names]
#     best_mean = max(v for v in mean_vals if not math.isnan(v))
#     mean_row_tex = r"\textbf{Mean}"
#     for val in mean_vals:
#         if math.isnan(val):
#             mean_row_tex += " & ---"
#         elif abs(val - best_mean) < 0.05:
#             mean_row_tex += f" & \\textbf{{{val:.1f}}}"
#         else:
#             mean_row_tex += f" & {val:.1f}"
#     lines.append(mean_row_tex + r" \\")
#     lines.append(r"\bottomrule")
#     lines.append(r"\end{tabular}}")
#     lines.append(r"\end{table*}")

#     latex_str = "\n".join(lines)

#     # Save LaTeX
#     tex_path = os.path.join(RESULTS_DIR, "ablation_table.tex")
#     with open(tex_path, "w") as f:
#         f.write(latex_str)
#     print(f"\nLaTeX table saved: {tex_path}")
#     print("\n" + "="*60)
#     print("Paste into Overleaf:")
#     print("="*60)
#     print(latex_str)


# # =============================================================================
# # 8. MAIN
# # =============================================================================

# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(
#         description="ContinualTTA Component Ablation Study",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Run full method on CIFAR-10-C Setting B:
#   python ablation.py --ablation A0 --dataset cifar10c --setting b

#   # Run no-detector ablation:
#   python ablation.py --ablation A1 --dataset cifar10c --setting b

#   # Run on ImageNet-C:
#   python ablation.py --ablation A0 --dataset imagenetc

#   # After all ablations done, merge and generate LaTeX:
#   python ablation.py --merge --dataset cifar10c --setting b

#   # List all available ablations:
#   python ablation.py --list
#         """)

#     parser.add_argument("--ablation", type=str, default=None,
#                         choices=[a[0] for a in ABLATIONS],
#                         help="Which ablation to run (e.g. A0, A1, A2...)")
#     parser.add_argument("--dataset",  type=str, default="cifar10c",
#                         choices=["cifar10c", "imagenetc"],
#                         help="Dataset to evaluate on")
#     parser.add_argument("--setting",  type=str, default="b",
#                         choices=["a", "b"],
#                         help="Evaluation protocol (cifar10c only): a=i.i.d., b=sequential")
#     parser.add_argument("--merge",    action="store_true",
#                         help="Merge all completed CSVs and generate LaTeX table")
#     parser.add_argument("--list",     action="store_true",
#                         help="List all available ablation variants")
#     args = parser.parse_args()

#     # List mode
#     if args.list:
#         print(f"\n{'ID':<6} {'Name':<22} {'Description'}")
#         print("─" * 70)
#         for abl_id, name, desc, kwargs in ABLATIONS:
#             kw_str = str(kwargs) if kwargs else "(full method)"
#             print(f"{abl_id:<6} {name:<22} {desc}")
#             print(f"{'':6} {'':22} kwargs: {kw_str}")
#         print()
#         exit(0)

#     # Merge mode
#     if args.merge:
#         merge_ablations_and_latex(args.dataset, args.setting)
#         exit(0)

#     # Run mode — requires --ablation
#     if args.ablation is None:
#         parser.error("--ablation is required unless using --merge or --list")

#     # Setup
#     print(f"\nDevice     : {DEVICE}")
#     if torch.cuda.is_available():
#         print(f"GPU        : {torch.cuda.get_device_name(0)}")
#     print(f"Ablation   : {args.ablation}")
#     print(f"Dataset    : {args.dataset}")
#     if args.dataset == "cifar10c":
#         print(f"Setting    : {args.setting.upper()}")

#     # Load model and configure for dataset
#     if args.dataset == "cifar10c":
#         source_model = load_cifar_model()
#         setup_bn_fn  = setup_bn_cifar
#         num_classes  = CIFAR_CLASSES
#         feat_dim     = 2048
#     else:
#         source_model = load_imagenet_model()
#         setup_bn_fn  = setup_bn_imagenet
#         num_classes  = IMAGENET_CLASSES
#         feat_dim     = 2048

#     print(f"Parameters : {sum(p.numel() for p in source_model.parameters()):,}")

#     # Run ablation
#     results, mean_acc = run_ablation(
#         ablation_id  = args.ablation,
#         dataset      = args.dataset,
#         setting      = args.setting,
#         source_model = source_model,
#         setup_bn_fn  = setup_bn_fn,
#         num_classes  = num_classes,
#         feat_dim     = feat_dim,
#     )

#     print(f"\n{'='*60}")
#     print(f"DONE — {args.ablation}: {mean_acc:.1f}% mean accuracy")
#     print(f"Results: {os.path.abspath(RESULTS_DIR)}/")
#     print(f"{'='*60}")



# =============================================================================
# ContinualTTA — Component Ablation Study (WACV 2027)
#
# Run from terminal:
#   python ablation.py --dataset cifar10c --setting b
#   python ablation.py --dataset cifar10c --setting a
#   python ablation.py --dataset imagenetc
#
# Ablations (each removes exactly one component from full ContinualTTA):
#   A0  Full ContinualTTA          — all components (reference)
#   A1  No JS Detector             — always adapt (remove gating)
#   A2  No Reliable Filter         — use all samples regardless of entropy
#   A3  No Prototype Bank          — entropy loss only, no consistency term
#   A4  No Prototype + No Filter   — plain entropy minimisation (≈ TENT)
#   A5  KL Detector                — replace JS with KL divergence
#   A6  Entropy Detector           — replace JS with entropy threshold
#   A7  EMA Decay β=0.5            — weaker prototype memory
#   A8  EMA Decay β=0.99           — stronger prototype memory
#   A9  JS Threshold τ=0.01        — more aggressive adaptation
#   A10 JS Threshold τ=0.05        — more conservative adaptation
#
# All ablations:
#   - Use identical BN-only adaptation (setup_bn / setup_bn_imagenet)
#   - Use identical optimiser (Adam, lr=1e-3)
#   - Differ in EXACTLY ONE component from A0
#   - Save individual CSVs — merge later with merge_ablations()
#
# Output:
#   results/ablations/A0_Full.csv
#   results/ablations/A1_NoDetector.csv
#   ...
#   results/ablations/ablation_table.tex  (generated after all runs)
# =============================================================================

import os
import copy
import math
import argparse
import platform
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from PIL import Image

# =============================================================================
# CONFIG — update paths
# =============================================================================

CIFAR_MODEL_PATH = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
CIFAR_DATA_DIR   = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
IMAGENET_DATA_DIR = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\ImageNet-C"

RESULTS_DIR  = os.path.join("results", "ablations")
NUM_WORKERS  = 0 if platform.system() == "Windows" else 2
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10-C settings
CIFAR_BATCH_SIZE  = 32
CIFAR_CLASSES     = 10
CIFAR_SEVERITIES  = [1, 2, 3, 4, 5]
CIFAR_SEED        = 42

# ImageNet-C settings
IMAGENET_BATCH_SIZE = 64
IMAGENET_CLASSES    = 1000
IMAGENET_SEVERITY   = 5

# Shared hyperparameters (identical across all ablations)
LR           = 1e-3
ALPHA        = 0.5           # prototype loss weight
JS_THRESHOLD = 0.04          # optimal from threshold sensitivity study (τ=0.04 → 86.4%)
EMA_DECAY    = 0.9           # default prototype EMA decay
E_MARGIN_FACTOR = 0.4        # entropy margin = factor * log(C)
SAR_RHO      = 0.05

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

# =============================================================================
# ABLATION REGISTRY
# Each entry: (id, name, description, kwargs_override)
# kwargs_override modifies only the specific component being ablated.
# =============================================================================

ABLATIONS = [
    # id   name                  description                           kwargs
    ("A0",  "Full",              "All components (reference)",          {}),
    ("A1",  "NoDetector",        "Always adapt — no JS gating",         {"use_detector": False}),
    ("A2",  "NoFilter",          "No entropy filter — all samples",     {"use_filter": False}),
    ("A3",  "NoPrototype",       "No prototype bank — entropy only",    {"use_prototype": False}),
    ("A4",  "NoProtoNoFilter",   "No prototype + no filter (≈TENT)",    {"use_prototype": False,
                                                                          "use_filter": False}),
    ("A5",  "KLDetector",        "KL divergence detector instead of JS",{"detector_type": "kl"}),
    ("A6",  "EntropyDetector",   "Entropy threshold detector",          {"detector_type": "entropy"}),
    ("A7",  "WeakMemory",        "Weaker EMA decay β=0.5",              {"ema_decay": 0.5}),
    ("A8",  "StrongMemory",      "Stronger EMA decay β=0.99",           {"ema_decay": 0.99}),
    ("A9",  "AggressiveGating",  "More aggressive JS threshold τ=0.01", {"js_threshold": 0.01}),
    ("A10", "ConservativeGating","More conservative JS threshold τ=0.05",{"js_threshold": 0.05}),
]


# =============================================================================
# 1. DATASETS
# =============================================================================

class CIFAR10C_Dataset(Dataset):
    def __init__(self, corruption, severity, data_dir):
        data        = np.load(os.path.join(data_dir, f"{corruption}.npy"), mmap_mode='r')
        labels      = np.load(os.path.join(data_dir, "labels.npy"),        mmap_mode='r')
        start       = (severity - 1) * 10000
        self.images = data[start:start + 10000]
        self.labels = labels[start:start + 10000]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.transform(Image.fromarray(self.images[idx])), int(self.labels[idx])


def cifar_loader_single(corruption, severity):
    return DataLoader(CIFAR10C_Dataset(corruption, severity, CIFAR_DATA_DIR),
                      batch_size=CIFAR_BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def cifar_loader_mixed(severity, seed=CIFAR_SEED):
    combined = ConcatDataset(
        [CIFAR10C_Dataset(c, severity, CIFAR_DATA_DIR) for c in ALL_CORRUPTIONS])
    g = torch.Generator(); g.manual_seed(seed)
    indices = torch.randperm(len(combined), generator=g).tolist()
    subset  = torch.utils.data.Subset(combined, indices)
    return DataLoader(subset, batch_size=CIFAR_BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def imagenet_loader(corruption):
    path = os.path.join(IMAGENET_DATA_DIR, corruption, str(IMAGENET_SEVERITY))
    dataset = ImageFolder(path, transform=models.ResNet50_Weights.IMAGENET1K_V1.transforms())
    return DataLoader(dataset, batch_size=IMAGENET_BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def load_cifar_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, CIFAR_CLASSES)
    model.load_state_dict(torch.load(CIFAR_MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def load_imagenet_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return model.to(DEVICE).eval()


# =============================================================================
# 2. HELPERS
# =============================================================================

def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)


def eval_loader(model_fn, loader):
    correct, total = 0, 0
    for x, y in loader:
        x, y    = x.to(DEVICE), y.to(DEVICE)
        logits  = model_fn(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def setup_bn_cifar(model):
    """CIFAR: per-batch BN statistics, only gamma/beta trainable."""
    model.train(); model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var  = None
    params = [p for m in model.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    return model, params


def setup_bn_imagenet(model):
    """ImageNet: keep pretrained BN stats frozen, only gamma/beta trainable."""
    model.train(); model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True
            m.momentum = 0
    params = [p for m in model.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    return model, params


# =============================================================================
# 3. ABLATION-AWARE COMPONENTS
# Each component checks kwargs to decide its behaviour.
# This keeps all logic in one place — easy to audit and modify.
# =============================================================================

class ShiftDetector:
    """
    Unified shift detector supporting multiple divergence types.
    Controlled by detector_type: 'js' | 'kl' | 'entropy' | 'none'
    """
    def __init__(self, detector_type="js", threshold=JS_THRESHOLD, ema=0.9):
        self.detector_type = detector_type
        self.threshold     = threshold
        self.ema           = ema
        self.reference     = None
        self.ema_entropy   = None    # used for entropy detector only

    def should_adapt(self, logits):
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)

            # No detector — always adapt
            if self.detector_type == "none":
                return True

            # Entropy detector — adapt when batch entropy exceeds threshold
            if self.detector_type == "entropy":
                h = -(p_t * p_t.log().clamp(min=-1e9)).sum()
                if self.ema_entropy is None:
                    self.ema_entropy = h.item()
                adapt = h.item() > self.threshold * math.log(logits.size(1))
                self.ema_entropy = 0.9 * self.ema_entropy + 0.1 * h.item()
                return adapt

            # JS or KL detector — need reference distribution
            if self.reference is None:
                self.reference = p_t.clone()
                return True

            p_ref = self.reference
            m     = 0.5 * (p_ref + p_t)

            if self.detector_type == "js":
                kl_1 = F.kl_div(m.log().unsqueeze(0),
                                 p_ref.unsqueeze(0), reduction="batchmean")
                kl_2 = F.kl_div(m.log().unsqueeze(0),
                                 p_t.unsqueeze(0),   reduction="batchmean")
                divergence = 0.5 * (kl_1 + kl_2)

            elif self.detector_type == "kl":
                # KL(p_ref || p_t) — asymmetric, unbounded
                divergence = F.kl_div(p_t.log().unsqueeze(0),
                                      p_ref.unsqueeze(0), reduction="batchmean")

            # Update reference AFTER computing divergence
            self.reference = self.ema * self.reference + (1 - self.ema) * p_t
            return divergence.item() > self.threshold


class PrototypeBankModule(nn.Module):
    """
    EMA prototype memory bank.
    Controlled by ema_decay kwarg — supports ablations A7 and A8.
    """
    def __init__(self, num_classes, feat_dim, ema_decay=EMA_DECAY):
        super().__init__()
        self.decay = ema_decay
        self.register_buffer("prototypes",  torch.zeros(num_classes, feat_dim))
        self.register_buffer("initialised", torch.zeros(num_classes).bool())

    @torch.no_grad()
    def update(self, features, pseudo_labels):
        for c in pseudo_labels.unique():
            mask = (pseudo_labels == c)
            mf   = features[mask].mean(0)
            if self.initialised[c]:
                self.prototypes[c] = self.decay * self.prototypes[c] + (1-self.decay) * mf
            else:
                self.prototypes[c] = mf
                self.initialised[c] = True

    def consistency_loss(self, features, pseudo_labels):
        """
        Prototype consistency loss.

        features must NOT be detached — gradient must flow through them
        back to BN gamma/beta.

        The prototype is the fixed target so it IS detached.
        MSE(features, prototype.detach()) means:
          - features are the prediction  → gradient flows to BN
          - prototype is the target      → treated as a constant

        Original bug was MSE(features.detach(), prototype) which gave
        zero gradient to BN params because the graph was cut at features.
        """
        loss, count = torch.tensor(0.0, device=features.device), 0
        for c in pseudo_labels.unique():
            if not self.initialised[c]: continue
            mask   = (pseudo_labels == c)
            target = self.prototypes[c].unsqueeze(0).expand(mask.sum(), -1).detach()
            loss  += F.mse_loss(features[mask], target)
            count += 1
        return loss / max(count, 1)


# =============================================================================
# 4. ABLATION VARIANT FACTORY
# Single function that builds any ablation by overriding specific kwargs.
# This is the heart of the ablation framework — clean and auditable.
# =============================================================================

def make_ablation_variant(source, setup_bn_fn, num_classes, feat_dim, **kwargs):
    """
    Build a ContinualTTA variant with specific components enabled/disabled.

    kwargs (all optional, fall back to defaults):
        use_detector   (bool)  : gate adaptation with shift detector [default: True]
        use_filter     (bool)  : apply entropy-based reliable filter  [default: True]
        use_prototype  (bool)  : use prototype consistency loss       [default: True]
        detector_type  (str)   : 'js' | 'kl' | 'entropy' | 'none'   [default: 'js']
        js_threshold   (float) : JS / KL detection threshold         [default: 0.02]
        ema_decay      (float) : prototype EMA decay β               [default: 0.9]
    """
    # Parse kwargs with defaults
    use_detector  = kwargs.get("use_detector",  True)
    use_filter    = kwargs.get("use_filter",    True)
    use_prototype = kwargs.get("use_prototype", True)
    detector_type = kwargs.get("detector_type", "js") if use_detector else "none"
    js_threshold  = kwargs.get("js_threshold",  JS_THRESHOLD)
    ema_decay     = kwargs.get("ema_decay",      EMA_DECAY)

    e_margin = E_MARGIN_FACTOR * math.log(num_classes)

    # Build components
    model, params = setup_bn_fn(copy.deepcopy(source))
    opt      = torch.optim.Adam(params, lr=LR)
    detector = ShiftDetector(detector_type=detector_type,
                             threshold=js_threshold)
    bank     = PrototypeBankModule(num_classes, feat_dim,
                                   ema_decay=ema_decay).to(DEVICE)
    captured = {}
    handle   = model.avgpool.register_forward_hook(
        lambda m, i, o: captured.update({"feat": o.flatten(1)}))

    @torch.enable_grad()
    def fn(x):
        logits        = model(x)
        features      = captured["feat"]
        pseudo_labels = logits.argmax(1).detach()

        # ── Gate 1: shift detector ────────────────────────────────────────────
        if not detector.should_adapt(logits.detach()):
            return logits

        # ── Gate 2: reliable sample filter ───────────────────────────────────
        if use_filter:
            entropy  = softmax_entropy(logits)
            reliable = entropy < e_margin
            if reliable.sum() == 0:
                return logits
        else:
            # No filter — use all samples
            entropy  = softmax_entropy(logits)
            reliable = torch.ones(x.size(0), dtype=torch.bool, device=DEVICE)

        # ── Loss computation ──────────────────────────────────────────────────
        loss = entropy[reliable].mean()

        if use_prototype:
            proto_loss = bank.consistency_loss(
                features[reliable],          # NOT detached — gradient flows to BN
                pseudo_labels[reliable])
            loss = loss + ALPHA * proto_loss

        # ── Parameter update (BN gamma/beta only) ────────────────────────────
        loss.backward()
        opt.step()
        opt.zero_grad()

        # ── Prototype bank update ─────────────────────────────────────────────
        if use_prototype:
            bank.update(features[reliable].detach(), pseudo_labels[reliable])

        return logits

    fn._handle = handle
    return fn


# =============================================================================
# 5. EVALUATION LOOPS
# =============================================================================

def eval_cifar_setting_b(source_model, fn):
    """Setting B: continual sequential, S1-S5 averaged."""
    all_sev = {}
    for severity in CIFAR_SEVERITIES:
        results = {}
        # Fresh function per severity (reset state)
        for corruption in ALL_CORRUPTIONS:
            loader = cifar_loader_single(corruption, severity)
            results[corruption] = eval_loader(fn, loader)
            del loader
            torch.cuda.empty_cache()
        all_sev[severity] = results

    # Average over severities
    averaged = {}
    for corruption in ALL_CORRUPTIONS:
        averaged[corruption] = np.mean(
            [all_sev[s][corruption] for s in CIFAR_SEVERITIES])
    return averaged


def eval_cifar_setting_a(source_model, setup_bn_fn, num_classes, feat_dim, **kwargs):
    """Setting A: mixed i.i.d. per severity, fresh model each severity."""
    sev_results = {}
    for severity in CIFAR_SEVERITIES:
        fn = make_ablation_variant(source_model, setup_bn_fn,
                                   num_classes, feat_dim, **kwargs)
        loader = cifar_loader_mixed(severity)
        acc    = eval_loader(fn, loader)
        sev_results[severity] = acc
        del loader; torch.cuda.empty_cache()
    return np.mean(list(sev_results.values())), sev_results


def eval_imagenet_sequential(source_model, fn):
    """ImageNet-C: continual sequential, severity 5."""
    results = {}
    for corruption in ALL_CORRUPTIONS:
        loader = imagenet_loader(corruption)
        results[corruption] = eval_loader(fn, loader)
        del loader; torch.cuda.empty_cache()
    return results


# =============================================================================
# 6. MAIN RUNNER — runs one ablation at a time
# =============================================================================

def run_ablation(ablation_id, dataset, setting, source_model,
                 setup_bn_fn, num_classes, feat_dim):
    """Run one ablation variant and save results to CSV."""

    # Find ablation config
    config = next((a for a in ABLATIONS if a[0] == ablation_id), None)
    if config is None:
        raise ValueError(f"Unknown ablation ID: {ablation_id}. "
                         f"Choose from {[a[0] for a in ABLATIONS]}")

    abl_id, abl_name, abl_desc, abl_kwargs = config
    print(f"\n{'='*60}")
    print(f"Ablation {abl_id}: {abl_name}")
    print(f"Description: {abl_desc}")
    print(f"Kwargs: {abl_kwargs if abl_kwargs else 'none (full method)'}")
    print(f"Dataset: {dataset.upper()}  |  Setting: {setting.upper()}")
    print(f"{'='*60}\n")

    # Build fresh ablation function
    fn = make_ablation_variant(source_model, setup_bn_fn,
                               num_classes, feat_dim, **abl_kwargs)

    # Run evaluation
    if dataset == "cifar10c":
        if setting == "b":
            results = eval_cifar_setting_b(source_model, fn)
        else:
            mean_a, sev_results = eval_cifar_setting_a(
                source_model, setup_bn_fn, num_classes, feat_dim, **abl_kwargs)
            results = {f"S{s}": sev_results[s] for s in CIFAR_SEVERITIES}
            results["Mean"] = mean_a
    elif dataset == "imagenetc":
        results = eval_imagenet_sequential(source_model, fn)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Print results
    print(f"\nResults for {abl_id} — {abl_name}:")
    for k, v in results.items():
        if k != "Mean":
            print(f"  {k:<24} {v:.1f}%")
    mean_acc = np.mean([v for k, v in results.items()
                        if k not in ("Mean",)])
    print(f"  {'Mean':<24} {mean_acc:.1f}%")

    # Save CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{abl_id}_{abl_name}.csv")
    with open(csv_path, "w") as f:
        f.write(f"corruption,{abl_id}_{abl_name}\n")
        for k, v in results.items():
            if k != "Mean":
                f.write(f"{k},{v:.1f}\n")
        f.write(f"Mean,{mean_acc:.1f}\n")
    print(f"\nSaved: {csv_path}")

    return results, mean_acc


# =============================================================================
# 7. MERGE AND LATEX — run after all ablations complete
# =============================================================================

def merge_ablations_and_latex(dataset="cifar10c", setting="b"):
    """
    Merge all individual CSVs into one table and generate LaTeX.
    Run this after all ablations have finished.

    Usage:
        python ablation.py --merge --dataset cifar10c --setting b
    """
    print("\nMerging ablation results...")
    all_results = {}    # ablation_name -> {corruption -> acc}
    all_means   = {}    # ablation_name -> mean_acc

    for abl_id, abl_name, abl_desc, _ in ABLATIONS:
        csv_path = os.path.join(RESULTS_DIR, f"{abl_id}_{abl_name}.csv")
        if not os.path.isfile(csv_path):
            print(f"  MISSING: {csv_path} — skipping")
            continue

        results = {}
        with open(csv_path) as f:
            lines = f.readlines()
        for line in lines[1:]:   # skip header
            parts = line.strip().split(",")
            if parts[0] == "Mean":
                all_means[f"{abl_id}_{abl_name}"] = float(parts[1])
            else:
                results[parts[0]] = float(parts[1])
        all_results[f"{abl_id}_{abl_name}"] = results

    if not all_results:
        print("No results found. Run ablations first.")
        return

    # ── Print merged table ────────────────────────────────────────────────────
    col = 12
    names = list(all_results.keys())
    header = f"{'Corruption':<24}" + "".join(f"{n[:10]:>{col}}" for n in names)
    print(f"\n{'═'*len(header)}")
    print("ABLATION STUDY — merged results")
    print(f"{'═'*len(header)}")
    print(header)
    print("─" * len(header))
    for corruption in ALL_CORRUPTIONS:
        row = f"{corruption:<24}"
        for name in names:
            val  = all_results[name].get(corruption, float('nan'))
            row += f"{val:.1f}%".rjust(col)
        print(row)
    print("─" * len(header))
    mean_row = f"{'Mean':<24}"
    for name in names:
        mean_row += f"{all_means.get(name, float('nan')):.1f}%".rjust(col)
    print(mean_row)
    print(f"{'═'*len(header)}")

    # ── Generate LaTeX ────────────────────────────────────────────────────────
    corr_names = {
        "gaussian_noise": "Gauss. Noise", "shot_noise": "Shot Noise",
        "impulse_noise": "Impulse",       "defocus_blur": "Defocus",
        "glass_blur": "Glass",            "motion_blur": "Motion",
        "zoom_blur": "Zoom",              "snow": "Snow",
        "frost": "Frost",                 "fog": "Fog",
        "brightness": "Brightness",       "contrast": "Contrast",
        "elastic_transform": "Elastic",   "pixelate": "Pixelate",
        "jpeg_compression": "JPEG",
    }

    # Column headers: short ablation names for table
    col_headers = {
        "A0_Full":              r"\textbf{Full}",
        "A1_NoDetector":        r"w/o Detector",
        "A2_NoFilter":          r"w/o Filter",
        "A3_NoPrototype":       r"w/o Proto.",
        "A4_NoProtoNoFilter":   r"w/o Proto.+Filt.",
        "A5_KLDetector":        r"KL Det.",
        "A6_EntropyDetector":   r"Entr. Det.",
        "A7_WeakMemory":        r"$\beta{=}0.5$",
        "A8_StrongMemory":      r"$\beta{=}0.99$",
        "A9_AggressiveGating":  r"$\tau{=}0.01$",
        "A10_ConservativeGating": r"$\tau{=}0.05$",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Component ablation of \textsc{ContinualTTA} on CIFAR-10-C "
                 r"continual sequential shift (Setting~B), S1--S5 averaged. "
                 r"Each column removes or modifies exactly one component from the "
                 r"full method (leftmost column). "
                 r"\textbf{Bold} = best per row.}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    n_cols = len(names)
    lines.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
    lines.append(r"\toprule")

    # Header row
    header_tex = "Corruption"
    for name in names:
        header_tex += " & " + col_headers.get(name, name)
    lines.append(header_tex + r" \\")
    lines.append(r"\midrule")

    # Per-corruption rows
    for corruption in ALL_CORRUPTIONS:
        vals = [all_results[name].get(corruption, float('nan'))
                for name in names]
        best = max(v for v in vals if not math.isnan(v))
        row  = corr_names.get(corruption, corruption)
        for val in vals:
            if math.isnan(val):
                row += " & ---"
            elif abs(val - best) < 0.05:
                row += f" & \\textbf{{{val:.1f}}}"
            else:
                row += f" & {val:.1f}"
        lines.append(row + r" \\")

    lines.append(r"\midrule")

    # Mean row
    mean_vals = [all_means.get(name, float('nan')) for name in names]
    best_mean = max(v for v in mean_vals if not math.isnan(v))
    mean_row_tex = r"\textbf{Mean}"
    for val in mean_vals:
        if math.isnan(val):
            mean_row_tex += " & ---"
        elif abs(val - best_mean) < 0.05:
            mean_row_tex += f" & \\textbf{{{val:.1f}}}"
        else:
            mean_row_tex += f" & {val:.1f}"
    lines.append(mean_row_tex + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex_str = "\n".join(lines)

    # Save LaTeX
    tex_path = os.path.join(RESULTS_DIR, "ablation_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"\nLaTeX table saved: {tex_path}")
    print("\n" + "="*60)
    print("Paste into Overleaf:")
    print("="*60)
    print(latex_str)


# =============================================================================
# 8. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ContinualTTA Component Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full method on CIFAR-10-C Setting B:
  python ablation.py --ablation A0 --dataset cifar10c --setting b

  # Run no-detector ablation:
  python ablation.py --ablation A1 --dataset cifar10c --setting b

  # Run on ImageNet-C:
  python ablation.py --ablation A0 --dataset imagenetc

  # After all ablations done, merge and generate LaTeX:
  python ablation.py --merge --dataset cifar10c --setting b

  # List all available ablations:
  python ablation.py --list
        """)

    parser.add_argument("--ablation", type=str, default=None,
                        choices=[a[0] for a in ABLATIONS],
                        help="Which ablation to run (e.g. A0, A1, A2...)")
    parser.add_argument("--dataset",  type=str, default="cifar10c",
                        choices=["cifar10c", "imagenetc"],
                        help="Dataset to evaluate on")
    parser.add_argument("--setting",  type=str, default="b",
                        choices=["a", "b"],
                        help="Evaluation protocol (cifar10c only): a=i.i.d., b=sequential")
    parser.add_argument("--merge",    action="store_true",
                        help="Merge all completed CSVs and generate LaTeX table")
    parser.add_argument("--list",     action="store_true",
                        help="List all available ablation variants")
    args = parser.parse_args()

    # List mode
    if args.list:
        print(f"\n{'ID':<6} {'Name':<22} {'Description'}")
        print("─" * 70)
        for abl_id, name, desc, kwargs in ABLATIONS:
            kw_str = str(kwargs) if kwargs else "(full method)"
            print(f"{abl_id:<6} {name:<22} {desc}")
            print(f"{'':6} {'':22} kwargs: {kw_str}")
        print()
        exit(0)

    # Merge mode
    if args.merge:
        merge_ablations_and_latex(args.dataset, args.setting)
        exit(0)

    # Run mode — requires --ablation
    if args.ablation is None:
        parser.error("--ablation is required unless using --merge or --list")

    # Setup
    print(f"\nDevice     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Ablation   : {args.ablation}")
    print(f"Dataset    : {args.dataset}")
    if args.dataset == "cifar10c":
        print(f"Setting    : {args.setting.upper()}")

    # Load model and configure for dataset
    if args.dataset == "cifar10c":
        source_model = load_cifar_model()
        setup_bn_fn  = setup_bn_cifar
        num_classes  = CIFAR_CLASSES
        feat_dim     = 2048
    else:
        source_model = load_imagenet_model()
        setup_bn_fn  = setup_bn_imagenet
        num_classes  = IMAGENET_CLASSES
        feat_dim     = 2048

    print(f"Parameters : {sum(p.numel() for p in source_model.parameters()):,}")

    # Run ablation
    results, mean_acc = run_ablation(
        ablation_id  = args.ablation,
        dataset      = args.dataset,
        setting      = args.setting,
        source_model = source_model,
        setup_bn_fn  = setup_bn_fn,
        num_classes  = num_classes,
        feat_dim     = feat_dim,
    )

    print(f"\n{'='*60}")
    print(f"DONE — {args.ablation}: {mean_acc:.1f}% mean accuracy")
    print(f"Results: {os.path.abspath(RESULTS_DIR)}/")
    print(f"{'='*60}")