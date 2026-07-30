# =============================================================================
# ContinualTTA — ImageNet-C GN Multi-Seed Truly Continual Evaluation (FIXED)
# Methods: TENT | EATA | SAR | ContinualTTA
# Seeds:   42, 123, 456
# =============================================================================

import os
import copy
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from multiprocessing import freeze_support

try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
except ImportError:
    print("ERROR: timm not installed. Run: pip install timm")
    exit(1)

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\ImageNet-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\imagenetc_multiseed"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
NUM_CLASSES = 1000
NUM_WORKERS = 0        # Windows-safe
SEVERITY    = 5
SEEDS       = [42, 123, 456]

# Hyperparameters
LR           = 2.5e-4
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 2.763 nats
MIN_CONF     = 0.5     # confidence gate for ContinualTTA
JS_THRESHOLD = 0.10    # ImageNet JS shift threshold
ADAPT_BUDGET = 300     # Max backward passes per corruption shift

SAR_RHO      = 0.05
SAR_E0       = 0.2     # Restored paper default reset threshold

METHODS = ["ContinualTTA"]

# EATA-specific
EATA_D_MARGIN = 0.05
EATA_FISHER_N = 10
EATA_FISHER_W = 1e-3

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

BLOCK_INDEX = {c: i for i, c in enumerate(ALL_CORRUPTIONS)}

FIXED_RESULTS = {
    "TENT":         0.41,
    "EATA":         0.95,
    "SAR":          29.0,
    "ContinualTTA": 34.3,
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. SEEDING & DATA PIPELINE
# =============================================================================

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model():
    print("  Loading resnet50_gn from timm...")
    model = timm.create_model("resnet50_gn", pretrained=True)
    model = model.to(DEVICE).eval()
    return model


def get_imagenet_transform(model):
    """Exact timm transform for resnet50_gn."""
    config = resolve_data_config({}, model=model)
    return create_transform(**config)


def get_loader(corruption, seed, transform):
    path = os.path.join(DATA_DIR, corruption, str(SEVERITY))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Missing path: {path}")

    block_idx = BLOCK_INDEX[corruption]
    g = torch.Generator()
    g.manual_seed(seed * 10000 + block_idx)

    dataset = ImageFolder(path, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                      generator=g, num_workers=NUM_WORKERS,
                      pin_memory=True)


# =============================================================================
# 2. HELPERS
# =============================================================================

def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)


def setup_gn(model):
    """Adapt only GroupNorm affine parameters."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            m.requires_grad_(True)
    params = [p for m in model.modules()
              if isinstance(m, nn.GroupNorm)
              for p in m.parameters() if p.requires_grad]
    return model, params


def eval_loader(model_fn, loader):
    correct, total = 0, 0
    for x, y in loader:
        x, y    = x.to(DEVICE), y.to(DEVICE)
        logits  = model_fn(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


# =============================================================================
# 3. METHOD FACTORIES
# =============================================================================

def make_tent(source):
    model, params = setup_gn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        loss   = softmax_entropy(logits).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        return logits
    return fn


def make_eata(source, fisher_loader=None):
    model, params = setup_gn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    fisher = {n: torch.zeros_like(p)
              for n, p in model.named_parameters() if p.requires_grad}
    if fisher_loader is not None:
        model.train()
        for i, (x, _) in enumerate(fisher_loader):
            if i >= EATA_FISHER_N: break
            x = x.to(DEVICE)
            softmax_entropy(model(x)).mean().backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
            model.zero_grad()
        for n in fisher:
            fisher[n] /= EATA_FISHER_N

    ref_probs = [None]

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
            mask_d = cos_sim < (1.0 - EATA_D_MARGIN)
        else:
            mask_d = torch.ones(probs.size(0), dtype=torch.bool, device=DEVICE)

        mask = mask_e & mask_d
        if mask.sum() == 0:
            return logits

        with torch.no_grad():
            ref_probs[0] = (probs[mask].mean(0).detach()
                            if ref_probs[0] is None
                            else 0.9 * ref_probs[0] + 0.1 * probs[mask].mean(0).detach())

        fisher_reg = sum((fisher[n] * p.pow(2)).sum()
                         for n, p in model.named_parameters()
                         if p.requires_grad and n in fisher)

        (entropy[mask].mean() + EATA_FISHER_W * fisher_reg).backward()
        opt.step()
        opt.zero_grad()
        return logits
    return fn


def make_sar(source):
    """
    FIXED SAR implementation:
    1. Freezes layer4 parameters to protect top-level representations.
    2. Restores SAR_E0 = 0.2 threshold to trigger resets on collapse.
    """
    model = copy.deepcopy(source)
    model.train()
    model.requires_grad_(False)

    # Freeze layer4 following SAR paper (Appendix C.2)
    for name, module in model.named_modules():
        if name.startswith("layer4"):
            for p in module.parameters():
                p.requires_grad_(False)

    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm) and not name.startswith("layer4"):
            module.requires_grad_(True)

    params = [p for name, m in model.named_modules()
              if isinstance(m, nn.GroupNorm) and not name.startswith("layer4")
              for p in m.parameters() if p.requires_grad]

    opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
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
                             ema_entropy[0] + 0.4 * math.log(NUM_CLASSES))
        reliable = entropy_init < dynamic_thresh
        if reliable.sum() == 0:
            return logits_init

        x_rel    = x[reliable]
        logits_1 = model(x_rel)
        softmax_entropy(logits_1).mean().backward()
        grad_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))

        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
                p.data.add_(e_w)
                e_ws.append(e_w)
                p.grad.zero_()
            else:
                e_ws.append(None)

        logits_2  = model(x_rel)
        entropy_2 = softmax_entropy(logits_2)
        if (entropy_2 < E_MARGIN).sum() > 0:
            entropy_2[entropy_2 < E_MARGIN].mean().backward()

        for p, e_w in zip(params, e_ws):
            if e_w is not None:
                p.data.sub_(e_w)
        opt.step()
        opt.zero_grad()

        with torch.no_grad():
            logits_out  = model(x)
            entropy_out = softmax_entropy(logits_out)
            ema_entropy[0] = (0.9 * ema_entropy[0] +
                              0.1 * entropy_out.mean().item())
            if ema_entropy[0] < SAR_E0:
                for n, p in model.named_parameters():
                    if p.requires_grad and n in init_params:
                        p.data.copy_(init_params[n])
                ema_entropy[0] = None

        return logits_out
    return fn


def make_ctta(source):
    model, params = setup_gn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    reference  = [None]
    n_backward = [0]

    @torch.enable_grad()
    def fn(x):
        logits = model(x)

        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                return logits

            m    = 0.5 * (reference[0] + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5 * (kl_1 + kl_2)
            reference[0] = 0.9 * reference[0] + 0.1 * p_t
            adapt_js = js.item() > JS_THRESHOLD

        # 1. THE MISSING GATE: Stop adapting if distribution is stable
        if not adapt_js:
            return logits

        # 2. STRICT BUDGET: Stop adapting if we've exhausted our global budget
        if n_backward[0] >= ADAPT_BUDGET:
            return logits

        with torch.no_grad():
            entropy = softmax_entropy(logits)
            conf    = logits.softmax(1).max(1).values
            
        reliable = (entropy < E_MARGIN) & (conf > MIN_CONF)
        if reliable.sum() == 0:
            return logits

        logits_rel = model(x[reliable])
        softmax_entropy(logits_rel).mean().backward()
        opt.step()
        opt.zero_grad()
        n_backward[0] += 1

        return logits
    return fn


def build_method(method, source, seed, transform):
    fisher_loader = None
    if method == "EATA" and seed is not None:
        fisher_loader = get_loader(ALL_CORRUPTIONS[0], seed, transform)

    dispatch = {
        "TENT":         lambda: make_tent(source),
        "EATA":         lambda: make_eata(source, fisher_loader),
        "SAR":          lambda: make_sar(source),
        "ContinualTTA": lambda: make_ctta(source),
    }
    fn = dispatch[method]()

    if fisher_loader is not None:
        del fisher_loader
        torch.cuda.empty_cache()

    return fn


# =============================================================================
# 4. RUN EXPERIMENTS
# =============================================================================

def run_one_seed(method, seed, source, transform):
    print(f"\n  {'─'*50}")
    print(f"  Method: {method}  |  Seed: {seed}  |  S{SEVERITY} Truly Continual")
    print(f"  {'─'*50}")

    set_global_seed(seed)
    fn = build_method(method, source, seed, transform)

    results = {}
    for corruption in ALL_CORRUPTIONS:
        loader = get_loader(corruption, seed, transform)
        acc    = eval_loader(fn, loader)
        results[corruption] = acc
        del loader
        torch.cuda.empty_cache()
        flag = "  !! collapse" if acc < 5.0 else ""
        print(f"    {corruption:<24} {acc:.1f}%{flag}")

    mean_acc = np.mean(list(results.values()))
    ref = FIXED_RESULTS.get(method, float('nan'))
    diff = mean_acc - ref if not math.isnan(ref) else float('nan')
    print(f"    {'Mean':<24} {mean_acc:.2f}%"
          + (f"  (single-run ref: {ref}%, diff: {diff:+.2f}%)"
             if not math.isnan(ref) else ""))
    return results, mean_acc


def save_csv(method, seed, results, mean_acc):
    path = os.path.join(RESULTS_DIR, f"{method}_seed{seed}.csv")
    with open(path, "w") as f:
        f.write(f"corruption,{method}_seed{seed}\n")
        for c in ALL_CORRUPTIONS:
            f.write(f"{c},{results[c]:.4f}\n")
        f.write(f"Mean,{mean_acc:.4f}\n")
    print(f"    Saved: {path}")


def load_mean(method, seed):
    path = os.path.join(RESULTS_DIR, f"{method}_seed{seed}.csv")
    if not os.path.isfile(path): return None
    with open(path) as f:
        for line in f:
            if line.startswith("Mean,"):
                return float(line.strip().split(",")[1])
    return None


def generate_summary(all_seed_means):
    print(f"\n{'='*60}")
    print("IMAGENET-C MULTI-SEED SUMMARY (Truly Continual, S5)")
    print(f"{'='*60}")

    summary = {}
    for method in METHODS:
        if method not in all_seed_means: continue
        vals = np.array(all_seed_means[method])
        mean = vals.mean()
        std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        summary[method] = (mean, std, vals.tolist())
        ref  = FIXED_RESULTS.get(method, float('nan'))
        flag = "  ← ours" if method == "ContinualTTA" else ""
        print(f"  {method:<18} {mean:.2f}% ± {std:.2f}%"
              + (f"  (single-run: {ref}%)" if not math.isnan(ref) else "")
              + flag)


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fixed ImageNet-C GN Multi-Seed Evaluation")
    parser.add_argument("--methods",   nargs="+", default=METHODS)
    parser.add_argument("--seeds",     nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    source    = load_model()
    transform = get_imagenet_transform(source)

    all_seed_means = {}

    for method in args.methods:
        print(f"\n{'='*60}\nMethod: {method}\n{'='*60}")
        method_means = []

        for seed in args.seeds:
            existing = load_mean(method, seed)
            if args.skip_done and existing is not None:
                print(f"  Skipping seed={seed} (saved: {existing:.2f}%)")
                method_means.append(existing)
                continue

            results, mean_acc = run_one_seed(method, seed, source, transform)
            save_csv(method, seed, results, mean_acc)
            method_means.append(mean_acc)
            torch.cuda.empty_cache()

        all_seed_means[method] = method_means

    generate_summary(all_seed_means)


if __name__ == "__main__":
    freeze_support()
    main()