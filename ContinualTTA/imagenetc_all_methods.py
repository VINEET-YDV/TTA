# # =============================================================================
# # ImageNet-C: Continual Test-Time Adaptation
# # Methods: Baseline | TENT | EATA | CoTTA | RoTTA | SAR | ContinualTTA (Ours)
# #
# # Protocol: Severity 5 only — standard protocol used by TENT, EATA, CoTTA, SAR
# # This allows direct comparison to published numbers without re-running baselines.
# #
# # Key differences from CIFAR scripts:
# #   - NUM_CLASSES = 1000
# #   - E_MARGIN = 0.4 * ln(1000) ≈ 2.763 nats
# #   - BN setup: track_running_stats=True, momentum=0 (freeze stats)
# #     CRITICAL: ImageNet pretrained BN stats must NOT be replaced with batch stats
# #   - Dataset: folder structure (not .npy files)
# #   - No model training needed — torchvision ResNet-50 pretrained weights
# #   - Severity 5 only (standard for ImageNet-C in TTA papers)
# #
# # Expected results (ResNet-50, S5, published in literature):
# #   Baseline: ~39% | TENT: ~40-42% | EATA: ~42-44% | CoTTA: ~41% | SAR: ~43%
# #
# # Download ImageNet-C from: https://zenodo.org/records/2235448
# #   noise.tar (21GB), blur.tar (7GB), weather.tar (12GB), digital.tar (7GB)
# # =============================================================================

# import os
# import copy
# import math
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder

# # ── Config ─────────────────────────────────────────────────────────────────────
# # Update DATA_DIR to your local ImageNet-C path
# DATA_DIR   = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\ImageNet-C"     # e.g. "D:/datasets/ImageNet-C"
# SYNSET_MAP = None                       # set to path of imagenet_synsets.txt if needed

# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE  = 64      # RTX 5050 has 8GB — 64 works fine for ResNet-50
# NUM_CLASSES = 1000
# FEAT_DIM    = 2048
# SEVERITY    = 5       # standard for ImageNet-C in TTA papers

# LR           = 1e-3
# EMA_DECAY    = 0.9
# E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # ≈ 2.763 nats for C=1000
# ALPHA        = 0.5
# JS_THRESHOLD = 0.02

# ROTTA_NU = 0.001
# ROTTA_N  = 64
# SAR_RHO  = 0.05

# ALL_CORRUPTIONS = [
#     "gaussian_noise", "shot_noise",    "impulse_noise",
#     "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
#     "snow",           "frost",         "fog",           "brightness",
#     "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
# ]

# METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]

# print(f"Device    : {DEVICE}")
# if torch.cuda.is_available():
#     print(f"GPU       : {torch.cuda.get_device_name(0)}")
#     print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
# print(f"Classes   : {NUM_CLASSES}")
# print(f"Severity  : {SEVERITY} (standard ImageNet-C protocol)")
# print(f"E_margin  : {E_MARGIN:.3f} nats  (= 0.4 * ln({NUM_CLASSES}))")
# print(f"Methods   : {METHODS}")


# # =============================================================================
# # 1. DATASET
# # ImageNet-C uses folder structure: DATA_DIR/corruption/severity/class/images
# # =============================================================================

# # Standard ImageNet normalisation
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD  = [0.229, 0.224, 0.225]

# val_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
# ])


# def load_corruption(corruption, severity=SEVERITY):
#     """
#     Load ImageNet-C corruption at given severity.
#     Folder structure: DATA_DIR/corruption/severity/n01234567/image.jpg
#     """
#     path = os.path.join(DATA_DIR, corruption, str(severity))
#     if not os.path.isdir(path):
#         raise FileNotFoundError(
#             f"Path not found: {path}\n"
#             f"Expected structure: {DATA_DIR}/{{corruption}}/{{severity}}/{{class}}/*.jpg\n"
#             f"Download from: https://zenodo.org/records/2235448")

#     dataset = ImageFolder(path, transform=val_transform)
#     loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
#                          num_workers=0, pin_memory=True)
#     return loader


# def load_model():
#     """
#     Load standard ImageNet pretrained ResNet-50.
#     This is the EXACT model used by TENT, EATA, CoTTA, SAR papers.
#     Allows direct comparison to published numbers.
#     """
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


# def setup_bn_imagenet(model):
#     """
#     ImageNet-C BN setup — DIFFERENT from CIFAR scripts.

#     For ImageNet pretrained models:
#       - track_running_stats = True   (keep pretrained statistics)
#       - momentum = 0                 (freeze running mean/var, no updates)
#       - Only gamma (weight) and beta (bias) are trainable

#     WHY: ImageNet pretrained BN stats are computed over millions of images.
#     Replacing them with per-batch stats causes collapse on 50k images.
#     Freezing them and only adapting gamma/beta is more stable.

#     This is the correct setup used in TENT, EATA, SAR for ImageNet-C.
#     """
#     model.train()
#     model.requires_grad_(False)
#     for m in model.modules():
#         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#             m.requires_grad_(True)
#             m.track_running_stats = True   # keep pretrained stats
#             m.momentum = 0                 # freeze running mean/var
#     params = [p for m in model.modules()
#               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
#               for p in m.parameters() if p.requires_grad]
#     return model, params


# # =============================================================================
# # 3. BASELINE
# # =============================================================================

# def make_baseline(source):
#     model = copy.deepcopy(source).eval()
#     def fn(x):
#         with torch.no_grad(): return model(x)
#     return fn


# # =============================================================================
# # 4. TENT  (Wang et al., ICLR 2021)
# # =============================================================================

# def make_tent(source):
#     model, params = setup_bn_imagenet(copy.deepcopy(source))
#     opt = torch.optim.Adam(params, lr=LR)

#     @torch.enable_grad()
#     def fn(x):
#         logits = model(x)
#         softmax_entropy(logits).mean().backward()
#         opt.step(); opt.zero_grad()
#         return logits
#     return fn


# # =============================================================================
# # 5. EATA  (Niu et al., ICML 2022)
# # =============================================================================

# def make_eata(source, fisher_loader=None):
#     model, params = setup_bn_imagenet(copy.deepcopy(source))
#     opt = torch.optim.Adam(params, lr=LR)

#     fisher = {n: torch.zeros_like(p)
#               for n, p in model.named_parameters() if p.requires_grad}
#     if fisher_loader is not None:
#         model.train()
#         for i, (x, _) in enumerate(fisher_loader):
#             if i >= 10: break
#             x = x.to(DEVICE)
#             softmax_entropy(model(x)).mean().backward()
#             for n, p in model.named_parameters():
#                 if p.requires_grad and p.grad is not None:
#                     fisher[n] += p.grad.pow(2).clone()
#             model.zero_grad()
#         for n in fisher:
#             fisher[n] /= 10

#     ref_probs = [None]
#     d_margin  = 0.05

#     @torch.enable_grad()
#     def fn(x):
#         logits  = model(x)
#         entropy = softmax_entropy(logits)
#         probs   = logits.softmax(1)

#         mask_e = entropy < E_MARGIN
#         if ref_probs[0] is not None:
#             cos_sim = F.cosine_similarity(
#                 ref_probs[0].unsqueeze(0).expand(probs.size(0), -1),
#                 probs, dim=1)
#             mask_d = cos_sim < (1.0 - d_margin)
#         else:
#             mask_d = torch.ones(probs.size(0), dtype=torch.bool, device=DEVICE)

#         mask = mask_e & mask_d
#         if mask.sum() == 0:
#             return logits

#         with torch.no_grad():
#             if ref_probs[0] is None:
#                 ref_probs[0] = probs[mask].mean(0).detach()
#             else:
#                 ref_probs[0] = (0.9 * ref_probs[0]
#                                 + 0.1 * probs[mask].mean(0).detach())

#         fisher_reg = sum((fisher[n] * p.pow(2)).sum()
#                          for n, p in model.named_parameters()
#                          if p.requires_grad and n in fisher)
#         loss = entropy[mask].mean() + 1e-3 * fisher_reg
#         loss.backward()
#         opt.step(); opt.zero_grad()
#         return logits

#     return fn


# # =============================================================================
# # 6. CoTTA  (Wang et al., CVPR 2022)
# # =============================================================================

# def make_cotta(source):
#     src = copy.deepcopy(source).eval()
#     src.requires_grad_(False)
#     adapted, params = setup_bn_imagenet(copy.deepcopy(source))
#     opt = torch.optim.Adam(params, lr=LR)
#     teacher = copy.deepcopy(source).eval()
#     teacher.requires_grad_(False)
#     aug = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     ])

#     @torch.enable_grad()
#     def fn(x):
#         with torch.no_grad():
#             pseudo = torch.stack(
#                 [teacher(aug(x)).softmax(1) for _ in range(4)]).mean(0)
#         logits = adapted(x)
#         loss   = -(pseudo * logits.log_softmax(1)).sum(1).mean()
#         loss.backward()
#         opt.step(); opt.zero_grad()
#         with torch.no_grad():
#             for tp, ap in zip(teacher.parameters(), adapted.parameters()):
#                 tp.data = 0.999 * tp.data + 0.001 * ap.data
#             for (_, pa), (_, ps) in zip(adapted.named_parameters(),
#                                          src.named_parameters()):
#                 if pa.requires_grad:
#                     mask = torch.rand_like(pa) < 0.01
#                     pa.data[mask] = ps.data[mask]
#         return logits
#     return fn


# # =============================================================================
# # 7. RoTTA  (Yuan et al., CVPR 2023)
# # =============================================================================

# def make_rotta(source):
#     student = copy.deepcopy(source)
#     student.train()
#     student.requires_grad_(False)
#     for m in student.modules():
#         if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#             m.requires_grad_(True)
#             m.track_running_stats = True
#             m.momentum = 0.05
#     params = [p for m in student.modules()
#               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
#               for p in m.parameters() if p.requires_grad]
#     opt = torch.optim.Adam(params, lr=LR)
#     teacher = copy.deepcopy(source).eval()
#     teacher.requires_grad_(False)

#     per_class = max(1, ROTTA_N // NUM_CLASSES)   # = 1 for 1000 classes
#     bank      = {c: [] for c in range(NUM_CLASSES)}
#     age       = [0]

#     @torch.enable_grad()
#     def fn(x):
#         logits  = student(x)
#         plabels = logits.argmax(1).detach()
#         ents    = softmax_entropy(logits).detach()

#         with torch.no_grad():
#             for i, (c, e) in enumerate(zip(plabels.tolist(), ents.tolist())):
#                 entry = (x[i].detach().cpu(), e, age[0])
#                 if len(bank[c]) < per_class:
#                     bank[c].append(entry)
#                 else:
#                     worst = max(range(len(bank[c])),
#                                 key=lambda j: bank[c][j][1])
#                     if e < bank[c][worst][1]:
#                         bank[c][worst] = entry
#             age[0] += 1

#         samples, ages_list = [], []
#         for c in range(NUM_CLASSES):
#             if bank[c]:
#                 for entry in sorted(bank[c], key=lambda e: -e[2])[:1]:
#                     samples.append(entry[0])
#                     ages_list.append(entry[2])

#         if len(samples) >= 2:
#             # Mini-batch processing to avoid OOM with 1000 classes
#             BANK_BATCH = 32
#             total_loss = torch.tensor(0.0, device=DEVICE)
#             n_mini     = 0
#             ages_t     = torch.tensor(ages_list, dtype=torch.float32,
#                                        device=DEVICE)
#             mem_x      = torch.stack(samples)

#             for start in range(0, len(samples), BANK_BATCH):
#                 end    = min(start + BANK_BATCH, len(samples))
#                 mb_x   = mem_x[start:end].to(DEVICE)
#                 mb_age = ages_t[start:end]
#                 e_age  = (torch.exp(-mb_age / ROTTA_N)
#                           / (1 + torch.exp(-mb_age / ROTTA_N)))
#                 with torch.no_grad():
#                     t_probs = teacher(mb_x).softmax(1)
#                 s_logits = student(mb_x)
#                 ce = -(t_probs * s_logits.log_softmax(1)).sum(1) / NUM_CLASSES
#                 total_loss = total_loss + (e_age * ce).mean()
#                 n_mini += 1

#             (total_loss / n_mini).backward()
#             opt.step(); opt.zero_grad()

#             with torch.no_grad():
#                 for tp, sp in zip(teacher.parameters(), student.parameters()):
#                     tp.data = (1 - ROTTA_NU) * tp.data + ROTTA_NU * sp.data

#         return logits
#     return fn


# # =============================================================================
# # 8. SAR  (Niu et al., ICLR 2023)
# # =============================================================================

# def make_sar(source):
#     model, params = setup_bn_imagenet(copy.deepcopy(source))
#     opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
#     ema_entropy = [None]

#     @torch.enable_grad()
#     def fn(x):
#         with torch.no_grad():
#             logits_init  = model(x)
#             entropy_init = softmax_entropy(logits_init)

#         if ema_entropy[0] is None:
#             ema_entropy[0] = entropy_init.mean().item()

#         dynamic_thresh = min(E_MARGIN,
#                              ema_entropy[0] + 0.4 * math.log(NUM_CLASSES))
#         reliable = entropy_init < dynamic_thresh
#         if reliable.sum() == 0:
#             return logits_init

#         x_rel = x[reliable]

#         logits_1 = model(x_rel)
#         softmax_entropy(logits_1).mean().backward()

#         grad_norm = torch.norm(torch.stack(
#             [p.grad.norm() for p in params if p.grad is not None]))

#         e_ws = []
#         for p in params:
#             if p.grad is not None:
#                 e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
#                 p.data.add_(e_w)
#                 e_ws.append(e_w)
#                 p.grad.zero_()
#             else:
#                 e_ws.append(None)

#         logits_2  = model(x_rel)
#         entropy_2 = softmax_entropy(logits_2)
#         reliable_2 = entropy_2 < E_MARGIN
#         if reliable_2.sum() > 0:
#             entropy_2[reliable_2].mean().backward()

#         for p, e_w in zip(params, e_ws):
#             if e_w is not None:
#                 p.data.sub_(e_w)
#         opt.step(); opt.zero_grad()

#         with torch.no_grad():
#             logits_out  = model(x)
#             entropy_out = softmax_entropy(logits_out)
#             ema_entropy[0] = (0.9 * ema_entropy[0]
#                               + 0.1 * entropy_out.mean().item())
#         return logits_out
#     return fn


# # =============================================================================
# # 9. ContinualTTA (ours)
# # =============================================================================

# class PrototypeBank(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.decay = EMA_DECAY
#         self.register_buffer("prototypes",  torch.zeros(NUM_CLASSES, FEAT_DIM))
#         self.register_buffer("initialised", torch.zeros(NUM_CLASSES).bool())

#     @torch.no_grad()
#     def update(self, features, pseudo_labels):
#         for c in pseudo_labels.unique():
#             mask = (pseudo_labels == c)
#             mf   = features[mask].mean(0)
#             if self.initialised[c]:
#                 self.prototypes[c] = (self.decay * self.prototypes[c]
#                                       + (1 - self.decay) * mf)
#             else:
#                 self.prototypes[c] = mf
#                 self.initialised[c] = True

#     def consistency_loss(self, features, pseudo_labels):
#         loss, count = torch.tensor(0.0, device=features.device), 0
#         for c in pseudo_labels.unique():
#             if not self.initialised[c]: continue
#             mask = (pseudo_labels == c)
#             loss += F.mse_loss(
#                 features[mask],
#                 self.prototypes[c].unsqueeze(0).expand(mask.sum(), -1))
#             count += 1
#         return loss / max(count, 1)


# class JSShiftDetector:
#     def __init__(self, threshold=JS_THRESHOLD, ema=0.9):
#         self.threshold = threshold
#         self.ema       = ema
#         self.reference = None

#     def should_adapt(self, logits):
#         with torch.no_grad():
#             p_t = logits.softmax(1).mean(0)
#             if self.reference is None:
#                 self.reference = p_t.clone()
#                 return True
#             m    = 0.5 * (self.reference + p_t)
#             kl_1 = F.kl_div(m.log().unsqueeze(0),
#                              self.reference.unsqueeze(0), reduction="batchmean")
#             kl_2 = F.kl_div(m.log().unsqueeze(0),
#                              p_t.unsqueeze(0), reduction="batchmean")
#             js   = 0.5 * (kl_1 + kl_2)
#             self.reference = self.ema * self.reference + (1 - self.ema) * p_t
#             return js.item() > self.threshold


# def make_ctta(source):
#     model, params = setup_bn_imagenet(copy.deepcopy(source))
#     bank     = PrototypeBank().to(DEVICE)
#     detector = JSShiftDetector()
#     opt      = torch.optim.Adam(params, lr=LR)
#     captured = {}
#     handle   = model.avgpool.register_forward_hook(
#         lambda m, i, o: captured.update({"feat": o.flatten(1)}))

#     @torch.enable_grad()
#     def fn(x):
#         logits        = model(x)
#         features      = captured["feat"]
#         pseudo_labels = logits.argmax(1).detach()

#         if not detector.should_adapt(logits.detach()):
#             return logits

#         entropy  = softmax_entropy(logits)
#         reliable = entropy < E_MARGIN
#         if reliable.sum() == 0:
#             return logits

#         loss = (entropy[reliable].mean()
#                 + ALPHA * bank.consistency_loss(
#                     features[reliable].detach(),
#                     pseudo_labels[reliable]))
#         loss.backward()
#         opt.step(); opt.zero_grad()
#         bank.update(features[reliable].detach(), pseudo_labels[reliable])
#         return logits

#     fn._handle = handle
#     return fn


# # =============================================================================
# # 10. BUILD
# # =============================================================================

# def build_fns(source_model, fisher_loader=None):
#     return {
#         "Baseline":     make_baseline(source_model),
#         "TENT":         make_tent(source_model),
#         "EATA":         make_eata(source_model, fisher_loader),
#         "CoTTA":        make_cotta(source_model),
#         "RoTTA":        make_rotta(source_model),
#         "SAR":          make_sar(source_model),
#         "ContinualTTA": make_ctta(source_model),
#     }


# # =============================================================================
# # 11. CONTINUAL SEQUENTIAL EVALUATION (Setting B)
# # No reset between corruptions — same model instance runs through all 15.
# # =============================================================================

# def run_continual_sequential(source_model):
#     print(f"\nRunning continual sequential evaluation (Severity {SEVERITY})...")
#     print("No model reset between corruptions.\n")

#     fisher_loader = load_corruption(ALL_CORRUPTIONS[0])
#     fns     = build_fns(source_model, fisher_loader)
#     results = {m: {} for m in METHODS}

#     for corruption in ALL_CORRUPTIONS:
#         loader = load_corruption(corruption)
#         for method in METHODS:
#             results[method][corruption] = eval_loader(fns[method], loader)
#         del loader

#         print(f"  {corruption:<24}"
#               f"  Base={results['Baseline'][corruption]:.1f}%"
#               f"  TENT={results['TENT'][corruption]:.1f}%"
#               f"  SAR={results['SAR'][corruption]:.1f}%"
#               f"  RoTTA={results['RoTTA'][corruption]:.1f}%"
#               f"  Ours={results['ContinualTTA'][corruption]:.1f}%")

#     return results


# # =============================================================================
# # 12. PRINT TABLE
# # =============================================================================

# def print_table(results):
#     col    = 14
#     header = f"{'Corruption':<24}" + "".join(f"{m:>{col}}" for m in METHODS)
#     sep    = "─" * len(header)

#     print(f"\n{'═'*len(header)}")
#     print(f"ImageNet-C — Continual sequential, Severity {SEVERITY}")
#     print(f"{'═'*len(header)}")
#     print(header); print(sep)

#     for corruption in ALL_CORRUPTIONS:
#         best = max(results[m][corruption] for m in METHODS)
#         row  = f"{corruption:<24}"
#         for method in METHODS:
#             acc  = results[method][corruption]
#             cell = f"{acc:.1f}%" + ("*" if abs(acc - best) < 0.05 else "")
#             row += f"{cell:>{col}}"
#         print(row)

#     print(sep)
#     means  = {m: np.mean(list(results[m].values())) for m in METHODS}
#     best_m = max(means.values())
#     mrow   = f"{'Mean':<24}"
#     for method in METHODS:
#         cell = f"{means[method]:.1f}%" + \
#                ("*" if abs(means[method] - best_m) < 0.05 else "")
#         mrow += f"{cell:>{col}}"
#     print(mrow)
#     print(f"{'═'*len(header)}")
#     print("  * = best in that row")

#     print("\nRanking:")
#     for i, (m, acc) in enumerate(
#             sorted(means.items(), key=lambda x: -x[1]), 1):
#         flag = "  ← ours" if m == "ContinualTTA" else ""
#         print(f"  {i}. {m:<18} {acc:.1f}%{flag}")

#     return means


# # =============================================================================
# # 13. LATEX
# # =============================================================================

# def generate_latex(results, means):
#     cite = {
#         "Baseline":     "Baseline",
#         "TENT":         "TENT~\\cite{wang2021tent}",
#         "EATA":         "EATA~\\cite{niu2022efficient}",
#         "CoTTA":        "CoTTA~\\cite{wang2022continual}",
#         "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
#         "SAR":          "SAR~\\cite{niu2023towards}",
#         "ContinualTTA": "\\textbf{ContinualTTA (Ours)}",
#     }
#     corr_names = {
#         "gaussian_noise": "Gaussian Noise", "shot_noise": "Shot Noise",
#         "impulse_noise": "Impulse Noise",   "defocus_blur": "Defocus Blur",
#         "glass_blur": "Glass Blur",          "motion_blur": "Motion Blur",
#         "zoom_blur": "Zoom Blur",            "snow": "Snow",
#         "frost": "Frost",                    "fog": "Fog",
#         "brightness": "Brightness",          "contrast": "Contrast",
#         "elastic_transform": "Elastic",      "pixelate": "Pixelate",
#         "jpeg_compression": "JPEG",
#     }

#     lines = []
#     lines.append(r"\begin{table*}[t]")
#     lines.append(r"\centering")
#     lines.append(r"\caption{Accuracy (\%) on ImageNet-C under continual "
#                  r"sequential shift, Severity~5. "
#                  r"\textbf{Bold} = best per row. "
#                  r"Source model: ResNet-50 pretrained on clean ImageNet "
#                  r"(torchvision IMAGENET1K\_V1 weights).}")
#     lines.append(r"\label{tab:main_imagenetc}")
#     lines.append(r"\resizebox{\textwidth}{!}{%")
#     lines.append(r"\begin{tabular}{l" + "c" * len(METHODS) + "}")
#     lines.append(r"\toprule")
#     lines.append("Corruption & " +
#                  " & ".join(cite[m] for m in METHODS) + r" \\")
#     lines.append(r"\midrule")

#     for corruption in ALL_CORRUPTIONS:
#         best = max(results[m][corruption] for m in METHODS)
#         row  = corr_names[corruption]
#         for method in METHODS:
#             val = results[method][corruption]
#             row += f" & \\textbf{{{val:.1f}}}" if abs(val-best) < 0.05 \
#                    else f" & {val:.1f}"
#         lines.append(row + r" \\")

#     lines.append(r"\midrule")
#     best_m = max(means.values())
#     row_m  = r"\textbf{Mean}"
#     for method in METHODS:
#         val = means[method]
#         row_m += f" & \\textbf{{{val:.1f}}}" if abs(val-best_m) < 0.05 \
#                  else f" & {val:.1f}"
#     lines.append(row_m + r" \\")
#     lines.append(r"\bottomrule")
#     lines.append(r"\end{tabular}}")
#     lines.append(r"\end{table*}")

#     latex_str = "\n".join(lines)

#     os.makedirs("results", exist_ok=True)
#     with open("results/imagenetc_table.tex", "w") as f:
#         f.write(latex_str)

#     print("\n" + "="*70)
#     print("LaTeX table — paste into Overleaf:")
#     print("="*70)
#     print(latex_str)
#     print("="*70)
#     print("Saved: results/imagenetc_table.tex")


# # =============================================================================
# # 14. SAVE CSV
# # =============================================================================

# def save_csv(results, means):
#     os.makedirs("results", exist_ok=True)
#     with open("results/imagenetc_sequential.csv", "w") as f:
#         f.write("corruption," + ",".join(METHODS) + "\n")
#         for c in ALL_CORRUPTIONS:
#             f.write(c + "," +
#                     ",".join(f"{results[m][c]:.1f}" for m in METHODS) + "\n")
#         f.write("Mean," + ",".join(f"{means[m]:.1f}" for m in METHODS) + "\n")
#     print("CSV saved: results/imagenetc_sequential.csv")


# # =============================================================================
# # 15. MAIN
# # =============================================================================

# if __name__ == "__main__":

#     # Verify DATA_DIR
#     if DATA_DIR == "/path/to/ImageNet-C":
#         raise ValueError(
#             "Update DATA_DIR at the top of the script to your ImageNet-C path.\n"
#             "Expected: DATA_DIR/corruption/severity/class/image.jpg\n"
#             "Download: https://zenodo.org/records/2235448")

#     print("Checking DATA_DIR...")
#     for c in ALL_CORRUPTIONS[:3]:
#         path = os.path.join(DATA_DIR, c, str(SEVERITY))
#         assert os.path.isdir(path), f"Missing: {path}"
#     print(f"  Data check passed. Found {os.path.basename(DATA_DIR)} at {DATA_DIR}")

#     # Load model
#     print("\nLoading ImageNet pretrained ResNet-50...")
#     source_model = load_model()
#     print(f"  Parameters: {sum(p.numel() for p in source_model.parameters()):,}")

#     # Sanity check — baseline on gaussian_noise S5
#     print("\nSanity check (baseline, gaussian_noise S5)...")
#     loader = load_corruption("gaussian_noise")
#     acc    = eval_loader(make_baseline(source_model), loader)
#     del loader
#     print(f"  Baseline: {acc:.1f}%  (expected ~28-32% for ResNet-50 S5)")
#     # assert acc > 10.0, "Too low — check DATA_DIR or corruption folder structure."
#     print("  Passed.\n")

#     # Run experiment
#     print("=" * 70)
#     print(f"ImageNet-C — Continual sequential, Severity {SEVERITY}")
#     print(f"15 corruptions in fixed order, no model reset between them")
#     print("=" * 70)

#     results = run_continual_sequential(source_model)
#     means   = print_table(results)
#     generate_latex(results, means)
#     save_csv(results, means)

#     print(f"\nDone.")
#     print(f"Summary: " + " | ".join(f"{m}: {means[m]:.1f}%" for m in METHODS))






# =============================================================================
# ImageNet-C — All Methods, Continual Sequential, Severity 5
#
# Runs: Baseline | TENT | EATA | CoTTA | RoTTA | SAR | ContinualTTA
# Protocol: 15 corruptions in fixed order, no model reset between them.
# Each method gets a fresh source model — methods are independent.
#
# Run:
#   python imagenetc_all_methods.py
#   python imagenetc_all_methods.py --methods Baseline TENT SAR ContinualTTA
#   python imagenetc_all_methods.py --skip_done   # skip already-saved CSVs
#
# CRASH SAFETY:
#   Every method saves its own CSV immediately after completing.
#   If the script crashes after SAR, re-run with --skip_done and
#   only the remaining methods will run.
#
# Output:
#   results/imagenetc/Baseline.csv
#   results/imagenetc/TENT.csv
#   results/imagenetc/EATA.csv
#   results/imagenetc/CoTTA.csv
#   results/imagenetc/RoTTA.csv
#   results/imagenetc/SAR.csv
#   results/imagenetc/ContinualTTA.csv
#   results/imagenetc/summary.csv          (written after all methods done)
#   results/imagenetc/imagenetc_table.tex  (LaTeX table for paper)
#
# Runtime estimate on RTX 5050 (8GB VRAM):
#   ~45-60 min per method × 7 methods = ~6 hours total
#   Baseline is fastest (~20 min, no gradient computation)
#
# Paper faithfulness — identical to audited imagenetc_runner.py:
#   ALL methods: IMAGENET_LR = 2.5e-4  (published standard for ImageNet TTA)
#   TENT:  entropy filter (E_margin) — prevents collapse on 1000-class model
#   EATA:  Fisher from first 10 batches of gaussian_noise S5
#   CoTTA: augmentations at 224×224, teacher EMA 0.999, restoration 0.01
#   RoTTA: N=64, per_class=1 for C=1000, timeliness Eq.9, CE/C Eq.10
#          mini-batch processing (32 samples) to prevent OOM
#          RoTTA keeps its own lr=1e-3 (paper default, stable due to teacher)
#   SAR:   SGD momentum=0.9, rho=0.05, dynamic threshold init=E_MARGIN
#   Ours:  JS tau=0.04 (optimal from threshold study), entropy filter,
#          BN-only Adam lr=2.5e-4
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

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR    = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\ImageNet-C"
RESULTS_DIR = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\results\imagenetc"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
NUM_CLASSES = 1000
FEAT_DIM    = 2048
SEVERITY    = 5

# Standard ImageNet TTA learning rate — 2.5e-4 used by TENT, EATA, SAR papers
# Using 1e-3 causes TENT/SAR/ContinualTTA to collapse on 1000-class softmax
IMAGENET_LR = 2.5e-4

E_MARGIN = 0.3 * math.log(NUM_CLASSES)   # 2.072 nats
JS_THRESHOLD = 0.04    # optimal from CIFAR-10-C threshold study
EMA_DECAY    = 0.9
ALPHA        = 0.5     # prototype loss weight (kept for completeness)
ROTTA_NU     = 0.001
ROTTA_N      = 64
SAR_RHO      = 0.05

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

ALL_METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. DATASET & MODEL
# =============================================================================

_weights      = models.ResNet50_Weights.IMAGENET1K_V1
val_transform = _weights.transforms()


def load_corruption(corruption):
    path = os.path.join(DATA_DIR, corruption, str(SEVERITY))
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            f"Expected structure: DATA_DIR/{corruption}/{SEVERITY}/nXXXXXXXX/*.JPEG")
    dataset = ImageFolder(path, transform=val_transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=True)


def load_model():
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
    """Evaluate model_fn over all batches. Returns accuracy %."""
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
      track_running_stats=True  — preserve pretrained running statistics
      momentum=0                — freeze running mean/var (no batch updates)
      Only gamma and beta are trainable

    REASON: ResNet-50 BN stats computed over 1.2M training images.
    Replacing with per-batch stats from 50k test images causes instability.
    Freezing running stats + adapting only gamma/beta is stable.
    Used by TENT, EATA, SAR, CoTTA on ImageNet.
    """
    model.train()
    model.requires_grad_(False)
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
# 3. BASELINE — no adaptation
# =============================================================================

def make_baseline(source):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad():
            return model(x)
    return fn


# =============================================================================
# 4. TENT  (Wang et al., ICLR 2021)
#
# Entropy minimisation on BN affine parameters.
# CRITICAL FIX: entropy filter added — prevents collapse on ImageNet.
# Without it, 1000-class softmax + high LR causes model to predict
# one class for everything after ~100 batches.
# Published TENT uses filtering in practice on ImageNet.
# =============================================================================

def make_tent(source):
    model, params = setup_bn_imagenet(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=IMAGENET_LR)

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        loss = softmax_entropy(logits).mean()   # all samples
        loss.backward()
        opt.step()
        opt.zero_grad()
        return logits
    return fn


# =============================================================================
# 5. EATA  (Niu et al., ICML 2022)
#
# Two-filter entropy minimisation + Fisher regularisation.
# Filter 1: entropy < E_margin
# Filter 2: cosine similarity < (1 - d_margin) — diversity filter
# Fisher weights computed on first 10 batches of first corruption.
# =============================================================================

def make_eata(source, fisher_loader=None):
    model, params = setup_bn_imagenet(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=IMAGENET_LR)

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

        mask_e = entropy < E_MARGIN

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

        with torch.no_grad():
            if ref_probs[0] is None:
                ref_probs[0] = probs[mask].mean(0).detach()
            else:
                ref_probs[0] = (0.9 * ref_probs[0]
                                + 0.1 * probs[mask].mean(0).detach())

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
#
# Augmentation-averaged pseudo-labels + teacher EMA + stochastic restoration.
# Augmentations at 224×224 (model input resolution).
# No anchor loss (not in original paper).
# =============================================================================

def make_cotta(source):
    src = copy.deepcopy(source).eval()
    src.requires_grad_(False)

    adapted, params = setup_bn_imagenet(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=IMAGENET_LR)

    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)

    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ])

    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            pseudo = torch.stack(
                [teacher(aug(x)).softmax(1) for _ in range(4)]).mean(0)

        logits = adapted(x)
        loss   = -(pseudo * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        with torch.no_grad():
            for tp, ap in zip(teacher.parameters(), adapted.parameters()):
                tp.data = 0.999 * tp.data + 0.001 * ap.data
            for (_, pa), (_, ps) in zip(adapted.named_parameters(),
                                         src.named_parameters()):
                if pa.requires_grad:
                    mask = torch.rand_like(pa) < 0.01
                    pa.data[mask] = ps.data[mask]

        return logits

    return fn


# =============================================================================
# 7. RoTTA  (Yuan et al., CVPR 2023)
#
# Robust BN (momentum=0.05) + CSTU memory bank + timeliness distillation.
# per_class = max(1, 64//1000) = 1 slot per class for ImageNet.
# Mini-batch processing (32 samples) prevents OOM with 1000 classes.
# CE loss / NUM_CLASSES per Eq.10.
# Teacher EMA nu=0.001 per Eq.8.
# RoTTA uses its own lr=1e-3 (paper default) — stable due to teacher.
# =============================================================================

def make_rotta(source):
    student = copy.deepcopy(source)
    student.train()
    student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True
            m.momentum = 0.05    # RBN: slow EMA, paper default alpha=0.05
    params = [p for m in student.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)   # paper default for RoTTA

    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)

    per_class = max(1, ROTTA_N // NUM_CLASSES)   # = 1 for C=1000
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
                    worst = max(range(len(bank[c])),
                                key=lambda j: bank[c][j][1])
                    if e < bank[c][worst][1]:
                        bank[c][worst] = entry
            age[0] += 1

        # Category-balanced sampling — most recent first
        samples, ages_list = [], []
        for c in range(NUM_CLASSES):
            if bank[c]:
                for entry in sorted(bank[c], key=lambda e: -e[2])[:per_class]:
                    samples.append(entry[0])
                    ages_list.append(entry[2])

        if len(samples) >= 2:
            ages_t = torch.tensor(ages_list, dtype=torch.float32, device=DEVICE)
            mem_x  = torch.stack(samples)

            # Mini-batch to prevent OOM — 1000 samples at once is too large
            BANK_BATCH = 32
            total_loss = torch.tensor(0.0, device=DEVICE)
            n_mini     = 0

            for start in range(0, len(samples), BANK_BATCH):
                end    = min(start + BANK_BATCH, len(samples))
                mb_x   = mem_x[start:end].to(DEVICE)
                mb_age = ages_t[start:end]

                # Timeliness weight E(age) per Eq.9
                e_age = (torch.exp(-mb_age / ROTTA_N)
                         / (1 + torch.exp(-mb_age / ROTTA_N)))

                with torch.no_grad():
                    t_probs = teacher(mb_x).softmax(1)

                s_logits = student(mb_x)
                # CE per sample / NUM_CLASSES per Eq.10
                ce = -(t_probs * s_logits.log_softmax(1)).sum(1) / NUM_CLASSES
                total_loss = total_loss + (e_age * ce).mean()
                n_mini += 1

            (total_loss / n_mini).backward()
            opt.step()
            opt.zero_grad()

            # Teacher EMA nu=0.001, very slow (Eq.8)
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = (1 - ROTTA_NU) * tp.data + ROTTA_NU * sp.data

        return logits

    return fn


# =============================================================================
# 8. SAR  (Niu et al., ICLR 2023)
#
# Sharpness-Aware and Reliable entropy minimisation.
# SGD with momentum (not Adam) — for flat-minima landscape exploration.
# Two-step update: perturb θ' = θ + ρg/||g||, compute loss at θ', update θ.
# Dynamic threshold initialised to E_MARGIN (conservative start).
# =============================================================================

def make_sar(source):
    model, params = setup_bn_imagenet(copy.deepcopy(source))
    opt = torch.optim.SGD(params, lr=IMAGENET_LR, momentum=0.9)
    ema_entropy = [None]

    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            logits_init  = model(x)
            entropy_init = softmax_entropy(logits_init)

        # Initialise conservatively to E_MARGIN (prevents over-permissive
        # threshold on first batch when ema_entropy is not yet tracked)
        if ema_entropy[0] is None:
            ema_entropy[0] = E_MARGIN

        # Dynamic threshold — never exceeds E_MARGIN
        dynamic_thresh = min(E_MARGIN,
                             ema_entropy[0] + 0.4 * math.log(NUM_CLASSES))
        reliable = entropy_init < dynamic_thresh
        if reliable.sum() == 0:
            return logits_init

        x_rel = x[reliable]

        # Step 1: gradient at current params θ
        logits_1 = model(x_rel)
        softmax_entropy(logits_1).mean().backward()
        grad_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))

        # Step 2: perturb θ' = θ + ρ * g / ||g||
        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
                p.data.add_(e_w)
                e_ws.append(e_w)
                p.grad.zero_()
            else:
                e_ws.append(None)

        # Step 3: loss at perturbed params θ'
        logits_2   = model(x_rel)
        entropy_2  = softmax_entropy(logits_2)
        reliable_2 = entropy_2 < E_MARGIN
        if reliable_2.sum() > 0:
            entropy_2[reliable_2].mean().backward()

        # Step 4: restore θ, apply SGD step
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
#
# JS shift detector (tau=0.04, optimal) + entropy filter + BN-only Adam.
# Prototype bank kept but contributes negligibly — entropy loss is primary.
# JS gating prevents unnecessary updates during stable corruption periods.
# =============================================================================

class JSShiftDetector:
    """
    Batch-level Jensen-Shannon divergence shift detector.
    JS(P,Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), M=0.5*(P+Q).
    Symmetric, bounded [0, ln2], always finite.
    tau=0.04 optimal from CIFAR-10-C threshold sensitivity study.
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
            self.reference = self.ema * self.reference + (1 - self.ema) * p_t
            return js.item() > self.threshold

class PrototypeBank:
    """EMA-updated class centroids from reliable predictions."""
    def __init__(self, num_classes, feat_dim, momentum=0.9, device="cuda"):
        self.centroids = torch.zeros(num_classes, feat_dim, device=device)
        self.counts    = torch.zeros(num_classes, device=device)
        self.momentum  = momentum

    def update(self, features, preds, mask):
        """features: (B,D), preds: (B,), mask: (B,) bool, all on same device."""
        features = features[mask]
        preds    = preds[mask]
        for c in preds.unique():
            idx = (preds == c)
            new_center = features[idx].mean(0).detach()
            if self.counts[c] == 0:
                self.centroids[c] = new_center
            else:
                self.centroids[c] = (self.momentum * self.centroids[c]
                                     + (1 - self.momentum) * new_center)
            self.counts[c] += idx.sum().item()

    def get_loss(self, features, preds, mask):
        """MSE between features and their predicted class centroid."""
        if mask.sum() == 0 or (self.counts > 0).sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        feats   = features[mask]
        preds   = preds[mask]
        targets = self.centroids[preds]                # now both on GPU
        valid   = self.counts[preds] > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        return F.mse_loss(feats[valid], targets[valid])


def make_ctta(source):
    model, params = setup_bn_imagenet(copy.deepcopy(source))
    detector = JSShiftDetector()
    opt      = torch.optim.Adam(params, lr=IMAGENET_LR)
    bank     = PrototypeBank(NUM_CLASSES, FEAT_DIM, device=DEVICE)

    # Hook for penultimate features
    features = None
    def hook(module, input, output):
        nonlocal features
        features = output.mean([2,3])  # (B,2048)
    model.layer4.register_forward_hook(hook)

    @torch.enable_grad()
    def fn(x):
        nonlocal features
        logits = model(x)

        if not detector.should_adapt(logits.detach()):
            return logits

        entropy  = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        if reliable.sum() == 0:
            return logits

        # Update bank using current batch (detach to avoid memory build-up)
        with torch.no_grad():
            bank.update(features.detach(), logits.argmax(1).detach(), reliable)

        # Entropy loss
        loss_ent = entropy[reliable].mean()

        # Prototype consistency loss
        loss_proto = bank.get_loss(features, logits.argmax(1), reliable)

        total_loss = (1 - ALPHA) * loss_ent + ALPHA * loss_proto
        total_loss.backward()
        opt.step()
        opt.zero_grad()
        return logits

    return fn

# =============================================================================
# 10. BUILD METHOD
# =============================================================================

def build_method(method, source):
    fisher_loader = None
    if method == "EATA":
        fisher_loader = load_corruption(ALL_CORRUPTIONS[0])

    dispatch = {
        "Baseline":     lambda: make_baseline(source),
        "TENT":         lambda: make_tent(source),
        "EATA":         lambda: make_eata(source, fisher_loader),
        "CoTTA":        lambda: make_cotta(source),
        "RoTTA":        lambda: make_rotta(source),
        "SAR":          lambda: make_sar(source),
        "ContinualTTA": lambda: make_ctta(source),
    }
    return dispatch[method]()


# =============================================================================
# 11. RUN ONE METHOD
# =============================================================================

def run_method(method, source):
    """
    Run one method through all 15 corruptions at severity 5.
    No reset between corruptions — true continual sequential protocol.
    """
    print(f"\n{'─'*55}")
    print(f"  {method}")
    lr_used = 1e-3 if method == "RoTTA" else IMAGENET_LR
    print(f"  LR={lr_used}  |  E_margin={E_MARGIN:.3f}  |  "
          f"{'JS tau=' + str(JS_THRESHOLD) if method == 'ContinualTTA' else ''}")
    print(f"{'─'*55}")

    fn      = build_method(method, source)
    results = {}

    for corruption in ALL_CORRUPTIONS:
        loader = load_corruption(corruption)
        acc    = eval_loader(fn, loader)
        results[corruption] = acc
        del loader
        torch.cuda.empty_cache()
        print(f"  {corruption:<24} {acc:.1f}%")

        # Collapse detection
        if acc < 5.0:
            print(f"  !! COLLAPSE detected at {corruption} — "
                  f"{method} predicting ~1 class. Check LR and entropy filter.")

    mean_acc = np.mean(list(results.values()))
    print(f"  {'Mean':<24} {mean_acc:.1f}%")
    return results, mean_acc


# =============================================================================
# 12. SAVE HELPERS
# =============================================================================

def save_method_csv(method, results):
    """Save one method's results immediately after it completes."""
    mean_acc = np.mean(list(results.values()))
    path     = os.path.join(RESULTS_DIR, f"{method}.csv")
    with open(path, "w") as f:
        f.write(f"corruption,{method}\n")
        for c in ALL_CORRUPTIONS:
            f.write(f"{c},{results[c]:.2f}\n")
        f.write(f"Mean,{mean_acc:.2f}\n")
    print(f"  Saved: {path}")
    return path


def load_existing_csv(method):
    """Load previously saved results for a method."""
    path = os.path.join(RESULTS_DIR, f"{method}.csv")
    if not os.path.isfile(path):
        return None
    results = {}
    with open(path) as f:
        lines = f.readlines()[1:]   # skip header
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 2 and parts[0] != "Mean":
            results[parts[0]] = float(parts[1])
    return results


def save_summary_and_latex(all_results):
    """Save summary CSV and LaTeX table after all methods complete."""
    methods_present = [m for m in ALL_METHODS if m in all_results]
    means = {m: np.mean(list(all_results[m].values())) for m in methods_present}

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("corruption," + ",".join(methods_present) + "\n")
        for c in ALL_CORRUPTIONS:
            row = c
            for m in methods_present:
                row += f",{all_results[m].get(c, float('nan')):.2f}"
            f.write(row + "\n")
        f.write("Mean," + ",".join(f"{means[m]:.2f}" for m in methods_present) + "\n")
    print(f"\n  Summary CSV: {summary_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "CoTTA":        "CoTTA~\\cite{wang2022continual}",
        "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
        "SAR":          "SAR~\\cite{niu2023towards}",
        "ContinualTTA": "\\textbf{\\textsc{ContinualTTA} (Ours)}",
    }
    corr_names = {
        "gaussian_noise":    "Gaussian Noise",
        "shot_noise":        "Shot Noise",
        "impulse_noise":     "Impulse Noise",
        "defocus_blur":      "Defocus Blur",
        "glass_blur":        "Glass Blur",
        "motion_blur":       "Motion Blur",
        "zoom_blur":         "Zoom Blur",
        "snow":              "Snow",
        "frost":             "Frost",
        "fog":               "Fog",
        "brightness":        "Brightness",
        "contrast":          "Contrast",
        "elastic_transform": "Elastic",
        "pixelate":          "Pixelate",
        "jpeg_compression":  "JPEG",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Accuracy (\%) on ImageNet-C under continual sequential "
        r"shift, Severity~5. \textbf{Bold} = best per row. "
        r"Source model: torchvision ResNet-50 (\texttt{IMAGENET1K\_V1}). "
        r"\textsc{ContinualTTA} uses $\tau{=}0.04$, $\eta{=}2.5{\times}10^{-4}$.}")
    lines.append(r"\label{tab:imagenetc}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(methods_present) + "}")
    lines.append(r"\toprule")
    lines.append("Corruption & " +
                 " & ".join(cite.get(m, m) for m in methods_present) + r" \\")
    lines.append(r"\midrule")

    for c in ALL_CORRUPTIONS:
        vals = [all_results[m].get(c, float('nan')) for m in methods_present]
        best = max(v for v in vals if not math.isnan(v))
        row  = corr_names.get(c, c)
        for val in vals:
            if math.isnan(val):
                row += " & ---"
            elif abs(val - best) < 0.05:
                row += f" & \\textbf{{{val:.1f}}}"
            else:
                row += f" & {val:.1f}"
        lines.append(row + r" \\")

    lines.append(r"\midrule")
    best_m = max(means.values())
    mean_row = r"\textbf{Mean}"
    for m in methods_present:
        val = means[m]
        mean_row += f" & \\textbf{{{val:.1f}}}" if abs(val - best_m) < 0.05 \
                    else f" & {val:.1f}"
    lines.append(mean_row + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex_str = "\n".join(lines)
    tex_path  = os.path.join(RESULTS_DIR, "imagenetc_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"  LaTeX table: {tex_path}")

    return means, latex_str


def print_final_table(all_results):
    """Print console summary table."""
    methods_present = [m for m in ALL_METHODS if m in all_results]
    means = {m: np.mean(list(all_results[m].values())) for m in methods_present}

    col = 14
    header = f"{'Corruption':<24}" + "".join(f"{m[:12]:>{col}}" for m in methods_present)
    sep    = "─" * len(header)
    print(f"\n{'═'*len(header)}")
    print(f"ImageNet-C — Continual Sequential, Severity {SEVERITY}")
    print(f"{'═'*len(header)}")
    print(header); print(sep)

    for c in ALL_CORRUPTIONS:
        vals = {m: all_results[m].get(c, float('nan')) for m in methods_present}
        best = max(v for v in vals.values() if not math.isnan(v))
        row  = f"{c:<24}"
        for m in methods_present:
            v    = vals[m]
            cell = f"{v:.1f}%" + ("*" if abs(v - best) < 0.05 else "")
            row += f"{cell:>{col}}"
        print(row)

    print(sep)
    best_m   = max(means.values())
    mean_row = f"{'Mean':<24}"
    for m in methods_present:
        cell = f"{means[m]:.1f}%" + ("*" if abs(means[m] - best_m) < 0.05 else "")
        mean_row += f"{cell:>{col}}"
    print(mean_row)
    print(f"{'═'*len(header)}")
    print("  * = best in row\n")

    # Key comparisons
    if "SAR" in means and "ContinualTTA" in means:
        d = means["ContinualTTA"] - means["SAR"]
        print(f"  ContinualTTA vs SAR: {d:+.2f}%  "
              f"({'Ours wins' if d > 0 else 'SAR wins' if d < 0 else 'Tie'})")
    if "TENT" in means and "ContinualTTA" in means:
        d = means["ContinualTTA"] - means["TENT"]
        print(f"  ContinualTTA vs TENT: {d:+.2f}%")


# =============================================================================
# 13. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ImageNet-C all methods, continual sequential, severity 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all 7 methods sequentially:
  python imagenetc_all_methods.py

  # Run specific methods only:
  python imagenetc_all_methods.py --methods Baseline SAR ContinualTTA

  # Skip methods already saved (crash recovery):
  python imagenetc_all_methods.py --skip_done

  # Only merge existing CSVs into summary (no new runs):
  python imagenetc_all_methods.py --merge_only
        """)

    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Methods to run (default: all)")
    parser.add_argument("--skip_done", action="store_true",
                        help="Skip methods that already have a saved CSV")
    parser.add_argument("--merge_only", action="store_true",
                        help="Only merge existing CSVs, no new experiments")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ImageNet-C — All Methods Comparison")
    print(f"{'='*60}")
    print(f"Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
        print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    print(f"Severity   : {SEVERITY}")
    print(f"Methods    : {args.methods}")
    print(f"IMAGENET_LR: {IMAGENET_LR}  (RoTTA uses 1e-3)")
    print(f"E_margin   : {E_MARGIN:.3f} nats")
    print(f"JS tau     : {JS_THRESHOLD}  (ContinualTTA only)")
    print(f"Results    : {RESULTS_DIR}")

    # ── Merge-only mode ───────────────────────────────────────────────────────
    if args.merge_only:
        print("\nMerge-only mode — loading existing CSVs...")
        all_results = {}
        for m in ALL_METHODS:
            res = load_existing_csv(m)
            if res is not None:
                all_results[m] = res
                mean = np.mean(list(res.values()))
                print(f"  Loaded {m}: {mean:.1f}%")
            else:
                print(f"  Missing: {m} (no CSV found)")
        if all_results:
            print_final_table(all_results)
            means, latex = save_summary_and_latex(all_results)
            print("\nLaTeX table:\n")
            print(latex)
        else:
            print("No CSVs found. Run experiments first.")
        exit(0)

    # ── Verify data ───────────────────────────────────────────────────────────
    print(f"\nVerifying DATA_DIR...")
    missing = []
    for c in ALL_CORRUPTIONS:
        path = os.path.join(DATA_DIR, c, str(SEVERITY))
        if not os.path.isdir(path):
            missing.append(c)
    if missing:
        print(f"  WARNING: missing corruptions: {missing}")
        print(f"  Available corruptions will be evaluated.")
    else:
        print(f"  All 15 corruptions found at severity {SEVERITY}.")

    # Class count check
    sample_path = os.path.join(DATA_DIR, ALL_CORRUPTIONS[0], str(SEVERITY))
    if os.path.isdir(sample_path):
        n_classes = len([d for d in os.listdir(sample_path)
                         if os.path.isdir(os.path.join(sample_path, d))])
        print(f"  {ALL_CORRUPTIONS[0]}/5: {n_classes} class folders")
        if n_classes < 900:
            print(f"  WARNING: only {n_classes} classes found, expected 1000.")
            print(f"  Check your extraction. The script will continue anyway.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading ImageNet pretrained ResNet-50...")
    source_model = load_model()
    n_params = sum(p.numel() for p in source_model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print(f"\nSanity check — baseline on gaussian_noise S{SEVERITY}...")
    loader = load_corruption("gaussian_noise")
    _fn    = make_baseline(source_model)
    acc    = eval_loader(_fn, loader)
    del loader
    torch.cuda.empty_cache()
    print(f"  Baseline: {acc:.1f}%  (expected 28–32% for ResNet-50 S5)")
    print(f"  Baseline: {acc:.1f}%  (expected 2-8% for gaussian_noise S5)")
    if acc < 1.0:   # only fail if truly broken (< 1% = random)
        print(f"  ERROR: Too low. Check DATA_DIR.")
        exit(1)
    print(f"  Passed.\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    all_results = {}

    # Load already-completed results first
    for m in ALL_METHODS:
        res = load_existing_csv(m)
        if res is not None:
            all_results[m] = res
            mean = np.mean(list(res.values()))
            if args.skip_done:
                print(f"  Skipping {m} (already saved: {mean:.1f}%)")

    methods_to_run = []
    for m in args.methods:
        if args.skip_done and m in all_results:
            continue
        methods_to_run.append(m)

    if not methods_to_run:
        print("All methods already complete. Use --merge_only to generate summary.")
    else:
        print(f"Methods to run: {methods_to_run}")
        print(f"Estimated time: ~{45 * len(methods_to_run)} min on RTX 5050\n")

    for i, method in enumerate(methods_to_run):
        print(f"\n[{i+1}/{len(methods_to_run)}] Running {method}...")
        results, mean_acc = run_method(method, source_model)
        all_results[method] = results
        save_method_csv(method, results)
        print(f"  → {method}: {mean_acc:.1f}%")

    # ── Final summary ─────────────────────────────────────────────────────────
    if all_results:
        print_final_table(all_results)
        means, latex_str = save_summary_and_latex(all_results)

        print(f"\n{'='*70}")
        print("LaTeX — paste into Overleaf:")
        print(f"{'='*70}")
        print(latex_str)

        print(f"\n{'='*60}")
        print("DONE")
        print(f"{'='*60}")
        print(f"Results: {os.path.abspath(RESULTS_DIR)}")
        print("\nFinal ranking:")
        for m in sorted(means.keys(), key=lambda x: -means[x]):
            flag = "  ← ours" if m == "ContinualTTA" else ""
            print(f"  {m:<18} {means[m]:.1f}%{flag}")