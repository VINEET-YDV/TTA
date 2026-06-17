# # =============================================================================
# # ImageNet-C GN — Truly Continual Sequential (NO reset between corruptions)
# # Methods: Baseline | TENT | EATA | CoTTA | RoTTA | SAR | ContinualTTA
# #
# # PROTOCOL: One model per method runs through all 15 corruptions sequentially.
# # No reset of any kind. This is the same protocol as CIFAR-10-C continual.
# # =============================================================================

# import os, copy, math, time
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import timm
# from timm.data import resolve_data_config
# from timm.data.transforms_factory import create_transform
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder

# # ------------------------- CONFIG -------------------------
# DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\ImageNet-C"
# RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\imagenetc_gn_continual"
# os.makedirs(RESULTS_DIR, exist_ok=True)

# DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# BATCH_SIZE  = 64
# NUM_CLASSES = 1000
# SEVERITY    = 5
# NUM_WORKERS = 0

# # Hyperparameters
# IMAGENET_LR  = 2.5e-4
# E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 2.763 nats
# JS_THRESHOLD = 0.1                           # use 0.1 for 1000-class stability
# ROTTA_NU     = 0.001
# ROTTA_N      = 64
# SAR_RHO      = 0.05

# # Model name
# GN_MODEL_NAME = "resnet50_gn"

# # All corruptions in fixed order
# ALL_CORRUPTIONS = [
#     "gaussian_noise", "shot_noise",    "impulse_noise",
#     "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
#     "snow",           "frost",         "fog",           "brightness",
#     "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
# ]


# CORRUPTIONS_LIST = ALL_CORRUPTIONS   # <-- change to ALL_CORRUPTIONS for full run

# # Methods to run
# METHODS = ["Baseline", "TENT", "EATA", "SAR", "ContinualTTA"]

# print(f"Device     : {DEVICE}")
# print(f"Protocol   : Truly continual — NO reset between corruptions")
# print(f"Severity   : {SEVERITY}")
# print(f"Corruptions: {len(CORRUPTIONS_LIST)}")
# print(f"Methods    : {METHODS}")

# # ------------------------- 1. MODEL & DATASET -------------------------
# def load_model():
#     print(f"  Loading {GN_MODEL_NAME} from timm...")
#     model = timm.create_model(GN_MODEL_NAME, pretrained=True)
#     return model.to(DEVICE).eval()

# def get_transform(model):
#     config    = resolve_data_config({}, model=model)
#     transform = create_transform(**config)
#     return transform

# def load_corruption(corruption, transform):
#     path = os.path.join(DATA_DIR, corruption, str(SEVERITY))
#     dataset = ImageFolder(path, transform=transform)
#     return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
#                       num_workers=NUM_WORKERS, pin_memory=True)

# # ------------------------- 2. HELPERS -------------------------
# def softmax_entropy(logits):
#     p = logits.softmax(1)
#     return -(p * p.log()).sum(1)

# def eval_loader(model_fn, loader):
#     correct, total = 0, 0
#     for x, y in loader:
#         x, y = x.to(DEVICE), y.to(DEVICE)
#         logits = model_fn(x)
#         correct += (logits.argmax(1) == y).sum().item()
#         total   += y.size(0)
#     return 100.0 * correct / total

# def setup_gn(model):
#     model.train()
#     model.requires_grad_(False)
#     for m in model.modules():
#         if isinstance(m, nn.GroupNorm):
#             m.requires_grad_(True)
#     params = [p for m in model.modules()
#               if isinstance(m, nn.GroupNorm)
#               for p in m.parameters() if p.requires_grad]
#     return model, params

# # ------------------------- 3. METHODS (GN versions) -------------------------
# # Baseline
# def make_baseline(source):
#     model = copy.deepcopy(source).eval()
#     def fn(x):
#         with torch.no_grad():
#             return model(x)
#     return fn

# # TENT
# def make_tent(source):
#     model, params = setup_gn(copy.deepcopy(source))
#     opt = torch.optim.Adam(params, lr=IMAGENET_LR)
#     @torch.enable_grad()
#     def fn(x):
#         logits = model(x)
#         loss = softmax_entropy(logits).mean()
#         loss.backward()
#         opt.step()
#         opt.zero_grad()
#         return logits
#     return fn

# # EATA
# def make_eata(source, fisher_loader=None):
#     model, params = setup_gn(copy.deepcopy(source))
#     opt = torch.optim.Adam(params, lr=IMAGENET_LR)
#     fisher = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
#     if fisher_loader is not None:
#         model.train()
#         for i,(x,_) in enumerate(fisher_loader):
#             if i >= 10: break
#             x = x.to(DEVICE)
#             softmax_entropy(model(x)).mean().backward()
#             for n,p in model.named_parameters():
#                 if p.requires_grad and p.grad is not None:
#                     fisher[n] += p.grad.pow(2).clone()
#             model.zero_grad()
#         for n in fisher: fisher[n] /= 10
#     ref_probs = [None]
#     d_margin = 0.05
#     @torch.enable_grad()
#     def fn(x):
#         logits = model(x)
#         entropy = softmax_entropy(logits)
#         probs = logits.softmax(1)
#         mask_e = entropy < E_MARGIN
#         if ref_probs[0] is not None:
#             cos_sim = F.cosine_similarity(ref_probs[0].unsqueeze(0).expand(probs.size(0),-1), probs, dim=1)
#             mask_d = cos_sim < (1.0 - d_margin)
#         else:
#             mask_d = torch.ones(probs.size(0), dtype=torch.bool, device=DEVICE)
#         mask = mask_e & mask_d
#         if mask.sum() == 0: return logits
#         with torch.no_grad():
#             ref_probs[0] = probs[mask].mean(0).detach() if ref_probs[0] is None else 0.9*ref_probs[0] + 0.1*probs[mask].mean(0).detach()
#         fisher_reg = sum((fisher[n]*p.pow(2)).sum() for n,p in model.named_parameters() if p.requires_grad and n in fisher)
#         loss = entropy[mask].mean() + 1e-3*fisher_reg
#         loss.backward()
#         opt.step(); opt.zero_grad()
#         return logits
#     return fn

# # CoTTA
# def make_cotta(source):
#     src = copy.deepcopy(source).eval(); src.requires_grad_(False)
#     adapted, params = setup_gn(copy.deepcopy(source))
#     opt = torch.optim.Adam(params, lr=IMAGENET_LR)
#     teacher = copy.deepcopy(source).eval(); teacher.requires_grad_(False)
#     aug = transforms.Compose([transforms.RandomHorizontalFlip(),
#                               transforms.RandomResizedCrop(224, scale=(0.8,1.0))])
#     @torch.enable_grad()
#     def fn(x):
#         with torch.no_grad():
#             pseudo = torch.stack([teacher(aug(x)).softmax(1) for _ in range(4)]).mean(0)
#         logits = adapted(x)
#         loss = -(pseudo * logits.log_softmax(1)).sum(1).mean()
#         loss.backward()
#         opt.step(); opt.zero_grad()
#         with torch.no_grad():
#             for tp,ap in zip(teacher.parameters(), adapted.parameters()):
#                 tp.data = 0.999*tp.data + 0.001*ap.data
#             for (_,pa),(_,ps) in zip(adapted.named_parameters(), src.named_parameters()):
#                 if pa.requires_grad:
#                     mask = torch.rand_like(pa) < 0.01
#                     pa.data[mask] = ps.data[mask]
#         return logits
#     return fn

# # RoTTA
# def make_rotta(source):
#     student = copy.deepcopy(source)
#     student.train(); student.requires_grad_(False)
#     for m in student.modules():
#         if isinstance(m, nn.GroupNorm):
#             m.requires_grad_(True)
#     params = [p for m in student.modules() if isinstance(m, nn.GroupNorm) for p in m.parameters() if p.requires_grad]
#     opt = torch.optim.Adam(params, lr=IMAGENET_LR)
#     teacher = copy.deepcopy(source).eval(); teacher.requires_grad_(False)
#     per_class = max(1, ROTTA_N // NUM_CLASSES)
#     bank = {c:[] for c in range(NUM_CLASSES)}
#     age = [0]
#     @torch.enable_grad()
#     def fn(x):
#         logits = student(x)
#         plabels = logits.argmax(1).detach()
#         ents = softmax_entropy(logits).detach()
#         with torch.no_grad():
#             for i,(c,e) in enumerate(zip(plabels.tolist(), ents.tolist())):
#                 entry = (x[i].detach().cpu(), e, age[0])
#                 if len(bank[c]) < per_class: bank[c].append(entry)
#                 else:
#                     worst = max(range(len(bank[c])), key=lambda j: bank[c][j][1])
#                     if e < bank[c][worst][1]: bank[c][worst] = entry
#             age[0] += 1
#         samples, ages_list = [], []
#         for c in range(NUM_CLASSES):
#             if bank[c]:
#                 for entry in sorted(bank[c], key=lambda e:-e[2])[:per_class]:
#                     samples.append(entry[0]); ages_list.append(entry[2])
#         if len(samples) >= 2:
#             mem_x = torch.stack(samples).to(DEVICE)
#             ages_t = torch.tensor(ages_list, dtype=torch.float32, device=DEVICE)
#             e_age = torch.exp(-ages_t/ROTTA_N) / (1+torch.exp(-ages_t/ROTTA_N))
#             BANK_BATCH = 32
#             total_loss = torch.tensor(0.0, device=DEVICE)
#             n_mini = 0
#             for start in range(0, len(samples), BANK_BATCH):
#                 end = min(start+BANK_BATCH, len(samples))
#                 mb_x = mem_x[start:end].to(DEVICE)
#                 mb_age = e_age[start:end]
#                 with torch.no_grad(): t_probs = teacher(mb_x).softmax(1)
#                 s_logits = student(mb_x)
#                 ce = -(t_probs * s_logits.log_softmax(1)).sum(1) / NUM_CLASSES
#                 total_loss = total_loss + (mb_age * ce).mean()
#                 n_mini += 1
#             (total_loss / n_mini).backward()
#             opt.step(); opt.zero_grad()
#             with torch.no_grad():
#                 for tp,sp in zip(teacher.parameters(), student.parameters()):
#                     tp.data = (1-ROTTA_NU)*tp.data + ROTTA_NU*sp.data
#         return logits
#     return fn

# # SAR
# def make_sar(source):
#     model, params = setup_gn(copy.deepcopy(source))
#     # Freeze layer4 per SAR paper
#     for name, module in model.named_modules():
#         if name.startswith("layer4"):
#             for pname, p in module.named_parameters(recurse=False):
#                 p.requires_grad_(False)
#     params = [p for n,p in model.named_parameters() if p.requires_grad and isinstance(model.get_submodule(".".join(n.split(".")[:-1])), nn.GroupNorm)]
#     opt = torch.optim.SGD(params, lr=IMAGENET_LR, momentum=0.9)
#     init_params = {n: p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
#     ema_entropy = [None]
#     e0 = 0.2
#     @torch.enable_grad()
#     def fn(x):
#         with torch.no_grad():
#             logits_init = model(x)
#             entropy_init = softmax_entropy(logits_init)
#         if ema_entropy[0] is None: ema_entropy[0] = E_MARGIN
#         dynamic_thresh = min(E_MARGIN, ema_entropy[0] + 0.4*math.log(NUM_CLASSES))
#         reliable = entropy_init < dynamic_thresh
#         if reliable.sum() == 0: return logits_init
#         x_rel = x[reliable]
#         logits_1 = model(x_rel)
#         softmax_entropy(logits_1).mean().backward()
#         grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
#         e_ws = []
#         for p in params:
#             if p.grad is not None:
#                 e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
#                 p.data.add_(e_w); e_ws.append(e_w); p.grad.zero_()
#             else: e_ws.append(None)
#         logits_2 = model(x_rel)
#         entropy_2 = softmax_entropy(logits_2)
#         reliable_2 = entropy_2 < E_MARGIN
#         if reliable_2.sum() > 0: entropy_2[reliable_2].mean().backward()
#         for p,e_w in zip(params, e_ws):
#             if e_w is not None: p.data.sub_(e_w)
#         opt.step(); opt.zero_grad()
#         with torch.no_grad():
#             logits_out = model(x)
#             entropy_out = softmax_entropy(logits_out)
#             cur_ent = entropy_out.mean().item()
#             ema_entropy[0] = 0.9*ema_entropy[0] + 0.1*cur_ent
#             if ema_entropy[0] < e0:
#                 for n,p in model.named_parameters():
#                     if p.requires_grad and n in init_params: p.data.copy_(init_params[n])
#                 ema_entropy[0] = None
#         return logits_out
#     return fn

# # ContinualTTA (simplified: JS gate + entropy filter)
# def make_ctta(source):
#     model, params = setup_gn(copy.deepcopy(source))
#     opt = torch.optim.Adam(params, lr=IMAGENET_LR)
#     reference = [None]
#     @torch.enable_grad()
#     def fn(x):
#         logits = model(x)
#         with torch.no_grad():
#             p_t = logits.softmax(1).mean(0)
#             if reference[0] is None:
#                 reference[0] = p_t.clone()
#                 return logits
#             m = 0.5*(reference[0] + p_t)
#             kl_1 = F.kl_div(m.log().unsqueeze(0), reference[0].unsqueeze(0), reduction="batchmean")
#             kl_2 = F.kl_div(m.log().unsqueeze(0), p_t.unsqueeze(0), reduction="batchmean")
#             js = 0.5*(kl_1 + kl_2)
#             reference[0] = 0.9*reference[0] + 0.1*p_t
#             if js.item() <= JS_THRESHOLD:
#                 return logits
#         entropy = softmax_entropy(logits)
#         reliable = entropy < E_MARGIN
#         if reliable.sum() == 0: return logits
#         logits_rel = model(x[reliable])
#         entropy_rel = softmax_entropy(logits_rel)
#         entropy_rel.mean().backward()
#         opt.step(); opt.zero_grad()
#         return logits
#     return fn

# # ------------------------- 4. TRULY CONTINUAL LOOP -------------------------
# def run_continual(method, source, transform):
#     print(f"\n{'='*60}")
#     print(f"Method: {method}  —  Truly Continual (no reset)")
#     print(f"{'='*60}")

#     # Build method once — never rebuilt
#     fn = None
#     if method == "Baseline":
#         fn = make_baseline(source)
#     elif method == "TENT":
#         fn = make_tent(source)
#     elif method == "EATA":
#         # Fisher on first corruption only
#         fisher_ldr = load_corruption(CORRUPTIONS_LIST[0], transform)
#         fn = make_eata(source, fisher_ldr)
#     elif method == "CoTTA":
#         fn = make_cotta(source)
#     elif method == "RoTTA":
#         fn = make_rotta(source)
#     elif method == "SAR":
#         fn = make_sar(source)
#     elif method == "ContinualTTA":
#         fn = make_ctta(source)
#     else:
#         raise ValueError(method)

#     results = {}
#     for corruption in CORRUPTIONS_LIST:
#         loader = load_corruption(corruption, transform)
#         acc = eval_loader(fn, loader)
#         results[corruption] = acc
#         del loader
#         torch.cuda.empty_cache()
#         print(f"  {corruption:<24} {acc:.1f}%")

#     mean_acc = np.mean(list(results.values()))
#     print(f"  Mean (over {len(CORRUPTIONS_LIST)} corruptions): {mean_acc:.1f}%")

#     # Save CSV
#     path = os.path.join(RESULTS_DIR, f"{method}_continual.csv")
#     with open(path, "w") as f:
#         f.write(f"corruption,{method}\n")
#         for c in CORRUPTIONS_LIST:
#             f.write(f"{c},{results[c]:.2f}\n")
#         f.write(f"Mean,{mean_acc:.2f}\n")
#     print(f"  Saved: {path}")

#     return results, mean_acc


# # ------------------------- 5. MAIN -------------------------
# if __name__ == "__main__":
#     print("Loading GN model and transform...")
#     source_model = load_model()
#     transform = get_transform(source_model)

#     # Sanity baseline (fresh)
#     print("\nSanity check — Baseline on gaussian_noise S5 (fresh model)...")
#     loader = load_corruption("gaussian_noise", transform)
#     correct = total = 0
#     with torch.no_grad():
#         for x,y in loader:
#             x,y = x.to(DEVICE), y.to(DEVICE)
#             correct += (source_model(x).argmax(1)==y).sum().item()
#             total += y.size(0)
#     print(f"  Baseline: {100*correct/total:.1f}%")
#     del loader; torch.cuda.empty_cache()

#     all_results = {}
#     all_means = {}

#     for method in METHODS:
#         res, mean = run_continual(method, source_model, transform)
#         all_results[method] = res
#         all_means[method] = mean
#         torch.cuda.empty_cache()

#     # Print final table
#     print("\n" + "="*80)
#     print("Truly Continual Sequential — ImageNet-C (S5) — ResNet-50-GN")
#     print("="*80)
#     print(f"{'Method':<20} {'Mean Acc':>10}")
#     for method in METHODS:
#         print(f"{method:<20} {all_means[method]:>10.1f}%")







# =============================================================================
# ContinualTTA — ImageNet-C GN Truly Continual
# ALL Methods: Baseline | TENT | EATA | CoTTA | RoTTA | SAR | ContinualTTA
#
# Protocol: TRULY CONTINUAL — identical to CIFAR-10-C experiments.
# One model per method runs through ALL 15 corruptions at severity 5
# in fixed order with NO reset between corruptions.
# Total: 15 × 50,000 = 750,000 images per method.
#
# Model: ResNet-50-GN (timm: resnet50_gn)
# Why GN not BN: BN is unstable at ImageNet-C S5 (SAR paper finding).
# GN computes stats per-sample — no running stats to corrupt.
# Only GN gamma/beta adapt — much safer under accumulation.
#
# CRASH SAFETY:
#   Saves after every corruption. Resume with --skip_done.
#
# Output:
#   results/imagenetc_gn_continual/{Method}.csv
#   results/imagenetc_gn_continual/summary.csv
#   results/imagenetc_gn_continual/table_continual.tex
#
# Runtime on RTX A4000:
#   Baseline:     ~60 min  (no backward)
#   TENT:         ~90 min
#   EATA:         ~95 min
#   CoTTA:        ~240 min (4 augmented forwards)
#   RoTTA:        ~120 min
#   SAR:          ~180 min (2 forwards + 2 backwards)
#   ContinualTTA: ~80 min  (JS gated, fewer backwards)
#   Total:        ~14 hours — run overnight, use --skip_done to resume
#
# Run:
#   python imagenetc_gn_continual.py
#   python imagenetc_gn_continual.py --methods Baseline ContinualTTA
#   python imagenetc_gn_continual.py --skip_done
#   python imagenetc_gn_continual.py --table_only
# =============================================================================

import os
import copy
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\ImageNet-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\imagenetc_gn_continual"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
NUM_CLASSES = 1000
SEVERITY    = 5
NUM_WORKERS = 0

# Hyperparameters — paper faithful
IMAGENET_LR  = 2.5e-4
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 2.763 nats
MIN_CONF     = 0.5              # confidence gate for ContinualTTA
ADAPT_BUDGET = 300              # max backward passes per corruption
                                # higher than fresh-per-corruption (100)
                                # because model carries useful state
JS_THRESHOLD = 0.10             # optimal for ImageNet-C GN (1000-class)
ROTTA_NU     = 0.001
ROTTA_N      = 64
SAR_RHO      = 0.05
SAR_E0       = 0.2              # model recovery threshold (paper default)
GN_MODEL     = "resnet50_gn"

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "SAR"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. MODEL & DATASET
# =============================================================================

def load_model():
    print(f"  Loading {GN_MODEL} from timm...")
    model = timm.create_model(GN_MODEL, pretrained=True)
    model = model.to(DEVICE).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_transform(model):
    """Use timm's own transform — critical for correct baseline."""
    config    = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform


def load_corruption(corruption, transform):
    path    = os.path.join(DATA_DIR, corruption, str(SEVERITY))
    dataset = ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


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


def setup_gn(model):
    """
    GN setup: only GroupNorm gamma/beta trainable.
    GN computes stats per-sample — no running stats to corrupt.
    This is why GN is stable under truly continual evaluation.
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            m.requires_grad_(True)
    params = [p for m in model.modules()
              if isinstance(m, nn.GroupNorm)
              for p in m.parameters() if p.requires_grad]
    return model, params


# =============================================================================
# 3. BASELINE
# =============================================================================

def make_baseline(source):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad():
            return model(x)
    return fn


# =============================================================================
# 4. TENT
# =============================================================================

def make_tent(source):
    model, params = setup_gn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=IMAGENET_LR)
    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        loss = softmax_entropy(logits).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        return logits
    return fn

# =============================================================================
# 5. EATA
# =============================================================================

def make_eata(source, fisher_loader=None):
    model, params = setup_gn(copy.deepcopy(source))
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
        for n in fisher: fisher[n] /= 10

    ref_probs = [None]

    @torch.enable_grad()
    def fn(x):
        logits  = model(x)
        entropy = softmax_entropy(logits)
        probs   = logits.softmax(1)
        mask_e  = entropy < E_MARGIN
        if ref_probs[0] is not None:
            cos_sim = F.cosine_similarity(
                ref_probs[0].unsqueeze(0).expand(probs.size(0), -1),
                probs, dim=1)
            mask_d = cos_sim < 0.95
        else:
            mask_d = torch.ones(probs.size(0), dtype=torch.bool, device=DEVICE)
        mask = mask_e & mask_d
        if mask.sum() == 0:
            return logits
        with torch.no_grad():
            ref_probs[0] = probs[mask].mean(0).detach() if ref_probs[0] is None \
                      else 0.9 * ref_probs[0] + 0.1 * probs[mask].mean(0).detach()
        fisher_reg = sum((fisher[n] * p.pow(2)).sum()
                         for n, p in model.named_parameters()
                         if p.requires_grad and n in fisher)
        (entropy[mask].mean() + 1e-3 * fisher_reg).backward()
        opt.step()
        opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 6. CoTTA
# =============================================================================

def make_cotta(source):
    src = copy.deepcopy(source).eval()
    src.requires_grad_(False)
    adapted, params = setup_gn(copy.deepcopy(source))
    opt     = torch.optim.Adam(params, lr=IMAGENET_LR)
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
        (-(pseudo * logits.log_softmax(1)).sum(1).mean()).backward()
        opt.step()
        opt.zero_grad()
        with torch.no_grad():
            for tp, ap in zip(teacher.parameters(), adapted.parameters()):
                tp.data = 0.999 * tp.data + 0.001 * ap.data
            for (_, pa), (_, ps) in zip(adapted.named_parameters(),
                                         src.named_parameters()):
                if pa.requires_grad:
                    mask = torch.rand_like(pa) < 0.01  # single mask — bug fix
                    pa.data[mask] = ps.data[mask]
        return logits

    return fn


# =============================================================================
# 7. RoTTA
# =============================================================================

def make_rotta(source):
    student = copy.deepcopy(source)
    student.train()
    student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, nn.GroupNorm):
            m.requires_grad_(True)
    params = [p for m in student.modules()
              if isinstance(m, nn.GroupNorm)
              for p in m.parameters() if p.requires_grad]
    opt     = torch.optim.Adam(params, lr=IMAGENET_LR)
    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)

    per_class = max(1, ROTTA_N // NUM_CLASSES)
    bank      = {c: [] for c in range(NUM_CLASSES)}
    age       = [0]

    @torch.enable_grad()
    def fn(x):
        logits  = student(x)
        plabels = logits.argmax(1).detach()
        ents    = softmax_entropy(logits).detach()
        with torch.no_grad():
            for i, (c, e) in enumerate(zip(plabels.tolist(), ents.tolist())):
                entry = (x[i].detach().cpu(), e, age[0])
                if len(bank[c]) < per_class: bank[c].append(entry)
                else:
                    worst = max(range(len(bank[c])), key=lambda j: bank[c][j][1])
                    if e < bank[c][worst][1]: bank[c][worst] = entry
            age[0] += 1

        samples, ages_list = [], []
        for c in range(NUM_CLASSES):
            if bank[c]:
                for entry in sorted(bank[c], key=lambda e: -e[2])[:per_class]:
                    samples.append(entry[0]); ages_list.append(entry[2])

        if len(samples) >= 2:
            mem_x  = torch.stack(samples).to(DEVICE)
            ages_t = torch.tensor(ages_list, dtype=torch.float32, device=DEVICE)
            e_age  = torch.exp(-ages_t/ROTTA_N)/(1+torch.exp(-ages_t/ROTTA_N))
            BANK_BATCH = 32
            total_loss = torch.tensor(0.0, device=DEVICE)
            n_mini = 0
            for start in range(0, len(samples), BANK_BATCH):
                end    = min(start+BANK_BATCH, len(samples))
                mb_x   = mem_x[start:end].to(DEVICE)
                mb_age = e_age[start:end]
                with torch.no_grad(): t_probs = teacher(mb_x).softmax(1)
                s_logits = student(mb_x)
                ce = -(t_probs * s_logits.log_softmax(1)).sum(1) / NUM_CLASSES
                total_loss = total_loss + (mb_age * ce).mean()
                n_mini += 1
            (total_loss / n_mini).backward()
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = (1-ROTTA_NU)*tp.data + ROTTA_NU*sp.data

        return logits

    return fn


# =============================================================================
# 8. SAR (paper faithful, GN version)
# =============================================================================

def make_sar(source):
    model, params = setup_gn(copy.deepcopy(source))

    # Freeze layer4 following SAR paper (Appendix C.2)
    for name, module in model.named_modules():
        if name.startswith("layer4"):
            for p in module.parameters():
                p.requires_grad_(False)
    params = [p for m in model.modules()
              if isinstance(m, nn.GroupNorm)
              for p in m.parameters() if p.requires_grad]

    opt = torch.optim.SGD(params, lr=IMAGENET_LR, momentum=0.9)
    init_params = {n: p.data.clone()
                   for n, p in model.named_parameters() if p.requires_grad}
    ema_entropy = [None]

    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            logits_init  = model(x)
            entropy_init = softmax_entropy(logits_init)

        if ema_entropy[0] is None:
            ema_entropy[0] = E_MARGIN

        dynamic_thresh = min(E_MARGIN,
                             ema_entropy[0] + 0.4*math.log(NUM_CLASSES))
        reliable = entropy_init < dynamic_thresh
        if reliable.sum() == 0:
            return logits_init

        x_rel = x[reliable]
        logits_1 = model(x_rel)
        softmax_entropy(logits_1).mean().backward()
        grad_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))

        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
                p.data.add_(e_w); e_ws.append(e_w); p.grad.zero_()
            else: e_ws.append(None)

        logits_2   = model(x_rel)
        entropy_2  = softmax_entropy(logits_2)
        if (entropy_2 < E_MARGIN).sum() > 0:
            entropy_2[entropy_2 < E_MARGIN].mean().backward()

        for p, e_w in zip(params, e_ws):
            if e_w is not None: p.data.sub_(e_w)
        opt.step(); opt.zero_grad()

        with torch.no_grad():
            logits_out  = model(x)
            entropy_out = softmax_entropy(logits_out)
            ema_entropy[0] = 0.9*ema_entropy[0] + 0.1*entropy_out.mean().item()
            if ema_entropy[0] < SAR_E0:
                for n, p in model.named_parameters():
                    if p.requires_grad and n in init_params:
                        p.data.copy_(init_params[n])
                ema_entropy[0] = None

        return logits_out

    return fn


# =============================================================================
# 9. ContinualTTA — GN version
#
# Key difference from BN version:
#   setup_gn() instead of setup_bn_imagenet()
#   JS_THRESHOLD=0.10 (optimal for 1000-class distributions)
#   MIN_CONF=0.5 confidence gate
#   ADAPT_BUDGET=300 (higher than fresh-per-corruption)
#   NO reset between corruptions — truly continual
# =============================================================================

def make_ctta(source):
    model, params = setup_gn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=IMAGENET_LR)

    reference  = [None]
    n_backward = [0]

    @torch.enable_grad()
    def fn(x):
        logits = model(x)

        # JS shift detector
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                return logits   # first batch: init only

            m    = 0.5 * (reference[0] + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5 * (kl_1 + kl_2)
            reference[0] = 0.9 * reference[0] + 0.1 * p_t
            adapt_js = js.item() > JS_THRESHOLD

        if not adapt_js:
            return logits

        if n_backward[0] >= ADAPT_BUDGET:
            return logits

        # Combined filter: entropy AND confidence
        with torch.no_grad():
            entropy = softmax_entropy(logits)
            conf    = logits.softmax(1).max(1).values
        reliable = (entropy < E_MARGIN) & (conf > MIN_CONF)
        if reliable.sum() == 0:
            return logits

        # Entropy minimisation on reliable samples only
        logits_rel  = model(x[reliable])
        softmax_entropy(logits_rel).mean().backward()
        opt.step()
        opt.zero_grad()
        n_backward[0] += 1
        return logits

    return fn


# =============================================================================
# 10. BUILD METHOD
# =============================================================================

def build_method(method, source, transform):
    """
    TRULY CONTINUAL: built ONCE per method, never reset.
    Same instance runs all 15 corruptions sequentially.
    """
    fisher_loader = None
    if method == "EATA":
        print("  Computing Fisher from gaussian_noise S5...")
        fisher_loader = load_corruption(ALL_CORRUPTIONS[0], transform)

    dispatch = {
        "Baseline":     lambda: make_baseline(source),
        "TENT":         lambda: make_tent(source),
        "EATA":         lambda: make_eata(source, fisher_loader),
        "CoTTA":        lambda: make_cotta(source),
        "RoTTA":        lambda: make_rotta(source),
        "SAR":          lambda: make_sar(source),
        "ContinualTTA": lambda: make_ctta(source),
    }
    fn = dispatch[method]()

    if fisher_loader is not None:
        del fisher_loader
        torch.cuda.empty_cache()

    return fn


# =============================================================================
# 11. TRULY CONTINUAL EVALUATION
# =============================================================================

def run_truly_continual(method, source, transform):
    """
    Run one method through all 15 corruptions at S5.
    NO reset between corruptions — same model instance throughout.
    """
    print(f"\n{'='*60}")
    print(f"Method: {method}  —  Truly Continual (no reset)")
    print(f"15 corruptions × 50,000 images, severity {SEVERITY}")
    print(f"{'='*60}")

    fn      = build_method(method, source, transform)
    results = {}

    for corruption in ALL_CORRUPTIONS:
        loader = load_corruption(corruption, transform)
        acc    = eval_loader(fn, loader)
        results[corruption] = acc
        del loader
        torch.cuda.empty_cache()

        flag = "  ✓" if acc > 30 else ("  !!" if acc < 5 else "")
        print(f"  {corruption:<24} {acc:.1f}%{flag}")

        # Save after every corruption — crash safety
        save_partial(method, results)

    mean_acc = np.mean(list(results.values()))
    print(f"  {'Mean':<24} {mean_acc:.1f}%")
    return results, mean_acc


# =============================================================================
# 12. SAVE HELPERS
# =============================================================================

def save_partial(method, results):
    """Save current results (called after every corruption)."""
    path = os.path.join(RESULTS_DIR, f"{method}_partial.csv")
    mean = np.mean(list(results.values())) if results else 0
    with open(path, "w") as f:
        f.write(f"corruption,{method}\n")
        for c in ALL_CORRUPTIONS:
            if c in results:
                f.write(f"{c},{results[c]:.4f}\n")
        if results:
            f.write(f"Mean,{mean:.4f}\n")


def save_final(method, results):
    mean = np.mean(list(results.values()))
    path = os.path.join(RESULTS_DIR, f"{method}.csv")
    with open(path, "w") as f:
        f.write(f"corruption,{method}\n")
        for c in ALL_CORRUPTIONS:
            f.write(f"{c},{results[c]:.4f}\n")
        f.write(f"Mean,{mean:.4f}\n")
    # Remove partial file
    partial = os.path.join(RESULTS_DIR, f"{method}_partial.csv")
    if os.path.isfile(partial):
        os.remove(partial)
    print(f"  Saved: {path}")
    return mean


def load_csv(method):
    path = os.path.join(RESULTS_DIR, f"{method}.csv")
    if not os.path.isfile(path):
        return None, None
    results = {}
    with open(path) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] != "Mean":
                results[parts[0]] = float(parts[1])
    return results, np.mean(list(results.values()))


# =============================================================================
# 13. TABLE GENERATOR
# =============================================================================

def generate_table(all_results, all_means):
    """Generate LaTeX table comparing all methods."""
    present = [m for m in METHODS if m in all_means]

    # Also load fresh-per-corruption results for comparison if available
    fresh_dir = os.path.join(
        os.path.dirname(RESULTS_DIR), "imagenetc_gn")
    fresh_means = {}
    for m in present:
        p = os.path.join(fresh_dir, f"{m}.csv")
        if os.path.isfile(p):
            with open(p) as f:
                for line in f.readlines()[1:]:
                    parts = line.strip().split(",")
                    if parts[0] == "Mean":
                        fresh_means[m] = float(parts[1])

    # Console print
    col = 14
    hdr = f"{'Corruption':<24}" + \
          "".join(f"{m[:11]:>{col}}" for m in present)
    print(f"\n{'='*len(hdr)}")
    print("ImageNet-C GN — Truly Continual, Severity 5")
    print(f"{'='*len(hdr)}")
    print(hdr); print("─"*len(hdr))

    for c in ALL_CORRUPTIONS:
        vals = [all_results[m].get(c, float('nan')) for m in present]
        finite = [v for v in vals if not math.isnan(v)]
        best   = max(finite) if finite else float('nan')
        row    = f"{c:<24}"
        for v in vals:
            cell = (f"{v:.1f}%" + ("*" if abs(v-best)<0.05 else "")) \
                   if not math.isnan(v) else "---"
            row += f"{cell:>{col}}"
        print(row)

    print("─"*len(hdr))
    best_m = max(all_means.values())
    mrow   = f"{'Mean':<24}"
    for m in present:
        v = all_means[m]
        cell = f"{v:.2f}%" + ("*" if abs(v-best_m)<0.05 else "")
        mrow += f"{cell:>{col}}"
    print(mrow)
    print(f"{'='*len(hdr)}\n  * = best\n")

    # Fresh comparison
    if fresh_means:
        print("Protocol comparison (Fresh vs Truly Continual):")
        print(f"  {'Method':<18} {'Fresh':>8} {'Continual':>11} {'Drop':>8}")
        print("  " + "─"*48)
        for m in present:
            fresh = fresh_means.get(m, float('nan'))
            cont  = all_means.get(m, float('nan'))
            if not math.isnan(fresh) and not math.isnan(cont):
                drop = cont - fresh
                flag = "  ←" if m == "ContinualTTA" else ""
                print(f"  {m:<18} {fresh:>7.1f}% {cont:>10.1f}% "
                      f"{drop:>+7.1f}%{flag}")

    # LaTeX
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
        "gaussian_noise":"Gaussian","shot_noise":"Shot","impulse_noise":"Impulse",
        "defocus_blur":"Defocus","glass_blur":"Glass","motion_blur":"Motion",
        "zoom_blur":"Zoom","snow":"Snow","frost":"Frost","fog":"Fog",
        "brightness":"Brightness","contrast":"Contrast",
        "elastic_transform":"Elastic","pixelate":"Pixelate",
        "jpeg_compression":"JPEG",
    }

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Accuracy (\%) on ImageNet-C (severity 5) under the "
        r"truly continual protocol (no model reset between corruption types). "
        r"ResNet-50-GN model following SAR~\cite{niu2023towards}. "
        r"\textbf{Bold} = best per row. "
        r"$\dagger$ = below no-adaptation baseline.}")
    lines.append(r"\label{tab:imagenetc_continual}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c"*len(present) + "}")
    lines.append(r"\toprule")
    lines.append("Corruption & " +
                 " & ".join(cite[m] for m in present) + r" \\")
    lines.append(r"\midrule")

    for c in ALL_CORRUPTIONS:
        vals   = [all_results[m].get(c, float('nan')) for m in present]
        finite = [v for v in vals if not math.isnan(v)]
        best   = max(finite) if finite else float('nan')
        row    = corr_names.get(c, c)
        for val in vals:
            if math.isnan(val): row += " & ---"
            elif abs(val-best)<0.05: row += f" & \\textbf{{{val:.1f}}}"
            else: row += f" & {val:.1f}"
        lines.append(row + r" \\")

    lines.append(r"\midrule")
    mean_vals = [all_means.get(m, float('nan')) for m in present]
    best_m    = max(v for v in mean_vals if not math.isnan(v))
    mrow_tex  = r"\textbf{Mean}"
    for val in mean_vals:
        if math.isnan(val): mrow_tex += " & ---"
        elif abs(val-best_m)<0.05: mrow_tex += f" & \\textbf{{{val:.1f}}}"
        else: mrow_tex += f" & {val:.1f}"
    lines.append(mrow_tex + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    tex_path = os.path.join(RESULTS_DIR, "table_continual.tex")
    with open(tex_path, "w") as f: f.write(latex)

    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("corruption," + ",".join(present) + "\n")
        for c in ALL_CORRUPTIONS:
            row = c + "," + ",".join(
                f"{all_results[m].get(c,float('nan')):.2f}"
                if not math.isnan(all_results[m].get(c,float('nan'))) else "---"
                for m in present)
            f.write(row + "\n")
        f.write("Mean," + ",".join(f"{all_means[m]:.2f}" for m in present) + "\n")

    print(f"\n  LaTeX: {tex_path}")
    print(f"  CSV:   {csv_path}")
    print(f"\n{'='*60}\nLaTeX:\n{'='*60}")
    print(latex)
    return latex


# =============================================================================
# 14. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ImageNet-C GN Truly Continual — All Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Protocol: TRULY CONTINUAL — no reset between corruptions.
Identical to CIFAR-10-C protocol but at ImageNet scale with GN model.

Examples:
  python imagenetc_gn_continual.py
  python imagenetc_gn_continual.py --methods Baseline ContinualTTA
  python imagenetc_gn_continual.py --skip_done
  python imagenetc_gn_continual.py --table_only

Run order recommendation (slowest first for parallel scheduling):
  1. CoTTA, SAR, RoTTA  (slow — run first, overnight)
  2. TENT, EATA, ContinualTTA, Baseline  (fast — run next day)
        """)

    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--table_only", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ImageNet-C GN — Truly Continual Protocol")
    print(f"{'='*60}")
    print(f"Device      : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Model       : {GN_MODEL}")
    print(f"Protocol    : Truly continual — NO reset between corruptions")
    print(f"Severity    : {SEVERITY}")
    print(f"JS tau      : {JS_THRESHOLD}  (ContinualTTA)")
    print(f"E_margin    : {E_MARGIN:.3f} nats")
    print(f"MIN_CONF    : {MIN_CONF}  (ContinualTTA confidence gate)")
    print(f"Budget      : {ADAPT_BUDGET} backward passes  (ContinualTTA)")
    print(f"Results     : {RESULTS_DIR}\n")

    # Table-only mode
    if args.table_only:
        print("Table-only mode...")
        all_results, all_means = {}, {}
        for m in METHODS:
            r, mean = load_csv(m)
            if r: all_results[m]=r; all_means[m]=mean; print(f"  Loaded {m}: {mean:.1f}%")
            else: print(f"  Missing: {m}")
        if all_results: generate_table(all_results, all_means)
        exit(0)

    # Verify data
    print("Verifying DATA_DIR...")
    for c in ALL_CORRUPTIONS[:3]:
        path = os.path.join(DATA_DIR, c, str(SEVERITY))
        if not os.path.isdir(path): print(f"  WARNING: missing {c}")
    print("  Done.\n")

    # Load model and transform
    print("Loading GN model...")
    source    = load_model()
    transform = get_transform(source)
    print(f"  Transform: {transform}\n")

    # Sanity check
    print("Sanity check — Baseline on gaussian_noise S5...")
    loader  = load_corruption("gaussian_noise", transform)
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (source(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    sanity = 100.0 * correct / total
    del loader; torch.cuda.empty_cache()
    print(f"  Baseline gaussian_noise: {sanity:.1f}%  (expected ~18-22%)")
    if sanity < 10.0:
        print("  ERROR: Too low — check model and transform"); exit(1)
    print("  Passed.\n")

    # Load existing results
    all_results, all_means = {}, {}
    for m in METHODS:
        r, mean = load_csv(m)
        if r and args.skip_done:
            all_results[m]=r; all_means[m]=mean
            print(f"  Skipping {m} (saved: {mean:.1f}%)")

    # Run methods
    for method in args.methods:
        if args.skip_done and method in all_results:
            continue
        results, mean = run_truly_continual(method, source, transform)
        all_results[method] = results
        all_means[method]   = save_final(method, results)
        print(f"\n  → {method}: {mean:.1f}%")
        torch.cuda.empty_cache()

    # Final table
    if all_results:
        generate_table(all_results, all_means)

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print("Final ranking:")
    for m in sorted(all_means, key=lambda x: -all_means[x]):
        flag = "  ← ours" if m == "ContinualTTA" else ""
        print(f"  {m:<18} {all_means[m]:.1f}%{flag}")