# =============================================================================
# ContinualTTA — ViT-Small CIFAR-10-C Truly Continual (LayerNorm)
#
# PURPOSE: Validate that JS-gated selective adaptation transfers from
# BN/GN architectures (ResNet-50) to transformer architectures (ViT-S).
# This is the "architectural generalization" experiment for AAAI.
#
# WHAT IS DIFFERENT FROM THE BN EXPERIMENTS:
#   - Model: ViT-Small/16 finetuned on CIFAR-10 (instead of ResNet-50)
#   - Norm layer: nn.LayerNorm (instead of nn.BatchNorm2d)
#   - LN properties: computed per-token, per-sample — NO running stats
#     to corrupt, unlike BN. This means LN is inherently more stable
#     under distribution shift (similar reasoning to why GN is stable
#     on ImageNet-C). The risk of collapse is LOWER than BN.
#   - Params adapted: only LN weight and bias (gamma/beta equivalent)
#     These are the affine parameters of each LayerNorm layer.
#
# METHODS: Baseline | TENT | ContinualTTA (three is enough for AAAI)
#   Three methods shows: (1) baseline performance, (2) naive entropy
#   minimization on LN (TENT), (3) JS-gated selective adaptation.
#   This is sufficient to make the architectural generalization claim.
#   Running all 7 methods would take too long and add marginal value.
#
# PROTOCOL: Truly continual (S1->S5, no reset) — identical to
#   CIFAR-10-C BN experiments for direct comparison.
#
# EXPECTED BEHAVIOR:
#   LN is more stable than BN under distribution shift because it
#   computes statistics per-sample, not per-batch. Expect:
#   - Less collapse than BN under truly continual protocol
#   - TENT may still degrade (unconditional entropy min on LN)
#   - ContinualTTA JS gating should still help at corruption boundaries
#
# MODEL SETUP:
#   timm ViT-Small/16 (vit_small_patch16_224) finetuned on CIFAR-10.
#   Input size: 224x224 (same as BN experiments — consistent pipeline).
#   The head is replaced with a 10-class linear layer during finetuning.
#   Load the finetuned checkpoint via MODEL_PATH.
#
# Run:
#   python cifar10c_vit_continual.py
#   python cifar10c_vit_continual.py --methods Baseline ContinualTTA
#   python cifar10c_vit_continual.py --skip_done
#   python cifar10c_vit_continual.py --table_only
#
# Output:
#   results/vit_continual/{Method}_s{1-5}.csv
#   results/vit_continual/{Method}_averaged.csv
#   results/vit_continual/summary.csv
#   results/vit_continual/table_vit.tex
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
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# =============================================================================
# CONFIG
# =============================================================================

# UPDATE THIS PATH to wherever you saved your finetuned ViT-S checkpoint
MODEL_PATH  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\vit_small_cifar10.pth"
DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\vit_continual"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
NUM_CLASSES = 10
NUM_WORKERS = 0      # Windows-safe
SEVERITIES  = [1, 2, 3, 4, 5]

# Hyperparameters
LR           = 1e-4    # lower than BN experiments — LN is more sensitive
                        # to large updates; 1e-4 is conservative and stable
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 0.921 nats — same as BN
JS_THRESHOLD = 0.04    # same as CIFAR-10-C BN — JS is scale-invariant

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

# Three methods sufficient for architectural generalization claim
METHODS = ["Baseline", "TENT", "ContinualTTA"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. MODEL
# =============================================================================

def load_model():
    """
    Load ViT-Small/16 finetuned on CIFAR-10.
    Architecture uses timm's vit_small_patch16_224.
    The classification head (model.head) has been replaced with
    nn.Linear(384, 10) during finetuning.
    """
    print(f"  Loading ViT-Small/16 from {MODEL_PATH}...")
    model = timm.create_model("vit_small_patch16_224", pretrained=False,
                               num_classes=NUM_CLASSES)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_ln     = sum(1 for m in model.modules() if isinstance(m, nn.LayerNorm))
    n_ln_params = sum(p.numel() for m in model.modules()
                      if isinstance(m, nn.LayerNorm)
                      for p in m.parameters())
    pct = 100.0 * n_ln_params / n_params

    print(f"  Total parameters: {n_params:,}")
    print(f"  LayerNorm layers: {n_ln}")
    print(f"  LayerNorm params: {n_ln_params:,} ({pct:.2f}% of total)")
    return model


# =============================================================================
# 2. DATASET
# =============================================================================

class CIFAR10C_Dataset(Dataset):
    def __init__(self, corruption, severity):
        data        = np.load(f"{DATA_DIR}/{corruption}.npy", mmap_mode='r')
        labels      = np.load(f"{DATA_DIR}/labels.npy",      mmap_mode='r')
        start       = (severity - 1) * 10000
        self.images = data[start:start + 10000]
        self.labels = labels[start:start + 10000]
        # Standard ImageNet normalization — same as used during ViT finetuning
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.transform(Image.fromarray(self.images[idx])),
                int(self.labels[idx]))


def get_loader(corruption, severity):
    return DataLoader(
        CIFAR10C_Dataset(corruption, severity),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)


# =============================================================================
# 3. HELPERS
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


def setup_ln(model):
    """
    LayerNorm adaptation setup.

    ViT uses LayerNorm extensively:
      - After each attention block (norm1)
      - After each MLP block (norm2)
      - Final norm before classification head (norm)

    We adapt ONLY the affine parameters (weight and bias) of each
    LayerNorm layer — the gamma/beta equivalent. This is analogous
    to BN affine adaptation and modifies ~0.04% of total parameters.

    LN key property: statistics computed per-token per-sample, NOT
    per-batch. This means:
      - No running_mean/running_var to corrupt (unlike BN)
      - No need to set track_running_stats=False
      - Naturally more stable under distribution shift
      - No special momentum adjustments needed
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            # LN has no running stats — nothing else to configure
    params = [p for m in model.modules()
              if isinstance(m, nn.LayerNorm)
              for p in m.parameters() if p.requires_grad]
    return model, params


# =============================================================================
# 4. BASELINE
# =============================================================================

def make_baseline(source):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad():
            return model(x)
    return fn


# =============================================================================
# 5. TENT (unconditional entropy minimisation on LN)
# =============================================================================

def make_tent(source):
    model, params = setup_ln(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        loss = softmax_entropy(logits).mean()   # all samples, no filter
        loss.backward()
        opt.step()
        opt.zero_grad()
        return logits

    return fn

# =============================================================================
# 6. ContinualTTA on ViT (JS gate + entropy filter on LayerNorm)
#
# The JS gate and entropy filter are ARCHITECTURE-AGNOSTIC:
#   - JS gate operates on softmax outputs, not on any layer internals
#   - Entropy filter also operates on softmax outputs
#   - Only the adaptation target changes: LN weight/bias instead of BN
#
# This is the key architectural generalization claim for AAAI:
# "The JS-gated selective adaptation principle transfers to transformer
#  architectures by replacing BN affine parameter adaptation with LN
#  affine parameter adaptation, with no other changes to the method."
# =============================================================================

def make_ctta(source):
    """
    ContinualTTA on ViT-Small: JS gate + entropy filter + LN-only Adam.
    The gate and filter are identical to the BN version.
    Only setup_ln() replaces setup_bn().
    """
    model, params = setup_ln(copy.deepcopy(source))
    opt       = torch.optim.Adam(params, lr=LR)
    reference = [None]

    @torch.enable_grad()
    def fn(x):
        logits = model(x)

        # JS shift detector — identical to BN version
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                return logits   # first batch: init reference only
            m    = 0.5 * (reference[0] + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5 * (kl_1 + kl_2)
            reference[0] = 0.9 * reference[0] + 0.1 * p_t
            adapt = js.item() > JS_THRESHOLD

        if not adapt:
            return logits

        # Entropy filter — identical to BN version
        entropy  = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        if reliable.sum() == 0:
            return logits

        # LN affine parameter update — only change from BN version
        entropy[reliable].mean().backward()
        opt.step(); opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 7. BUILD METHOD
# =============================================================================

def build_method(method, source):
    dispatch = {
        "Baseline":     lambda: make_baseline(source),
        "TENT":         lambda: make_tent(source),
        "ContinualTTA": lambda: make_ctta(source),
    }
    if method not in dispatch:
        raise ValueError(f"Unknown method: {method}. Choose from {list(dispatch)}")
    return dispatch[method]()


# =============================================================================
# 8. TRULY CONTINUAL EVALUATION
# =============================================================================

def run_truly_continual(method, source):
    """
    Run one method through all 75 blocks with no reset.
    S1×15 → S2×15 → S3×15 → S4×15 → S5×15.
    Identical protocol to CIFAR-10-C BN experiments.
    """
    print(f"\n{'='*60}")
    print(f"Method: {method}  |  ViT-Small  |  Truly Continual")
    print(f"{'='*60}")

    fn          = build_method(method, source)
    all_results = {s: {} for s in SEVERITIES}

    for severity in SEVERITIES:
        print(f"\n  Severity {severity}")
        for corruption in ALL_CORRUPTIONS:
            loader = get_loader(corruption, severity)
            acc    = eval_loader(fn, loader)
            all_results[severity][corruption] = acc
            del loader
            torch.cuda.empty_cache()
            flag = "  !!" if acc < 30.0 and method != "Baseline" else ""
            print(f"    {corruption:<24} {acc:.1f}%{flag}")

        sev_mean = np.mean(list(all_results[severity].values()))
        print(f"    S{severity} mean: {sev_mean:.2f}%")
        save_severity(method, severity, all_results[severity])

    return all_results


# =============================================================================
# 9. SAVE HELPERS
# =============================================================================

def save_severity(method, severity, results):
    path = os.path.join(RESULTS_DIR, f"{method}_s{severity}.csv")
    mean = np.mean(list(results.values()))
    with open(path, "w") as f:
        f.write(f"corruption,{method}_s{severity}\n")
        for c in ALL_CORRUPTIONS:
            f.write(f"{c},{results[c]:.4f}\n")
        f.write(f"Mean,{mean:.4f}\n")
    print(f"    Saved: {path}")


def compute_averaged(method, all_results):
    averaged = {c: np.mean([all_results[s][c] for s in SEVERITIES])
                for c in ALL_CORRUPTIONS}
    mean_overall = np.mean(list(averaged.values()))
    path = os.path.join(RESULTS_DIR, f"{method}_averaged.csv")
    with open(path, "w") as f:
        f.write(f"corruption,{method}\n")
        for c in ALL_CORRUPTIONS:
            f.write(f"{c},{averaged[c]:.4f}\n")
        f.write(f"Mean,{mean_overall:.4f}\n")
    print(f"  Averaged: {method} = {mean_overall:.2f}%  -> {path}")
    return mean_overall


def load_averaged(method):
    path = os.path.join(RESULTS_DIR, f"{method}_averaged.csv")
    if not os.path.isfile(path): return None, None
    results = {}
    with open(path) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] != "Mean":
                results[parts[0]] = float(parts[1])
    return results, np.mean(list(results.values()))


# =============================================================================
# 10. TABLE GENERATION
# =============================================================================

def generate_table(all_averaged, all_means):
    present = [m for m in METHODS if m in all_means]

    # BN reference for comparison (from main experiments)
    bn_ref = {"Baseline": 77.5, "TENT": 82.07, "ContinualTTA": 86.22}

    print(f"\n{'='*65}")
    print("ViT-Small CIFAR-10-C — Truly Continual (LayerNorm)")
    print(f"{'='*65}")
    print(f"\n  {'Method':<18} {'ViT-S (LN)':>12} {'ResNet-50 (BN)':>16}")
    print("  " + "─"*48)
    for m in present:
        vit = all_means[m]
        bn  = bn_ref.get(m, float('nan'))
        flag = "  ← ours" if m == "ContinualTTA" else ""
        bn_str = f"{bn:.2f}%" if not math.isnan(bn) else "---"
        print(f"  {m:<18} {vit:>11.2f}%  {bn_str:>15}{flag}")

    # LaTeX table
    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "ContinualTTA": "\\textbf{\\textsc{ContinualTTA} (Ours)}",
    }
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Accuracy (\%) on CIFAR-10-C under the truly "
        r"continual protocol with ViT-Small/16 (LayerNorm). "
        r"Only LN affine parameters ($\gamma$, $\beta$) are adapted. "
        r"ResNet-50 (BN) results from Table~\ref{tab:main_cifar} shown "
        r"for reference. \textbf{Bold} = best.}")
    lines.append(r"\label{tab:vit}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & ViT-S/16 (LN) & ResNet-50 (BN) \\")
    lines.append(r"\midrule")
    best_vit = max(all_means[m] for m in present)
    for m in present:
        vit    = all_means[m]
        bn     = bn_ref.get(m, float('nan'))
        vit_s  = f"\\textbf{{{vit:.2f}}}" if abs(vit-best_vit)<0.01 else f"{vit:.2f}"
        bn_s   = f"{bn:.2f}" if not math.isnan(bn) else "---"
        lines.append(f"{cite[m]} & {vit_s} & {bn_s} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines)

    tex_path = os.path.join(RESULTS_DIR, "table_vit.tex")
    with open(tex_path, "w") as f: f.write(latex)

    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("method," + ",".join(ALL_CORRUPTIONS) + ",Mean\n")
        for m in present:
            vals = [f"{all_averaged[m].get(c, float('nan')):.2f}"
                    for c in ALL_CORRUPTIONS]
            f.write(f"{m}," + ",".join(vals) + f",{all_means[m]:.2f}\n")

    print(f"\n  LaTeX: {tex_path}")
    print(f"  CSV:   {csv_path}")
    print(f"\n{'='*60}\nLaTeX:\n{'='*60}")
    print(latex)
    return latex


# =============================================================================
# 11. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ContinualTTA ViT-Small CIFAR-10-C Truly Continual",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architectural generalization experiment for AAAI.
Three methods (Baseline, TENT, ContinualTTA) on ViT-Small/16.
Truly continual protocol identical to main CIFAR-10-C BN experiments.

Examples:
  python cifar10c_vit_continual.py
  python cifar10c_vit_continual.py --methods Baseline ContinualTTA
  python cifar10c_vit_continual.py --skip_done
  python cifar10c_vit_continual.py --table_only
        """)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--table_only", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ContinualTTA — ViT-Small CIFAR-10-C Truly Continual")
    print(f"{'='*60}")
    print(f"Device      : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Model       : ViT-Small/16 (LayerNorm)")
    print(f"Protocol    : Truly continual, S1->S5, no reset")
    print(f"Methods     : {args.methods}")
    print(f"LR          : {LR}  (lower than BN — LN more sensitive)")
    print(f"E_margin    : {E_MARGIN:.3f} nats")
    print(f"JS tau      : {JS_THRESHOLD}")
    print(f"Results     : {RESULTS_DIR}\n")

    if args.table_only:
        all_averaged, all_means = {}, {}
        for m in args.methods:
            avg, mean = load_averaged(m)
            if avg: all_averaged[m]=avg; all_means[m]=mean
            print(f"  {'Loaded' if avg else 'Missing'}: {m}"
                  + (f" ({mean:.2f}%)" if avg else ""))
        if all_averaged: generate_table(all_averaged, all_means)
        exit(0)

    # Verify data
    for c in ALL_CORRUPTIONS[:3]:
        assert os.path.isfile(f"{DATA_DIR}/{c}.npy"), \
            f"Missing: {DATA_DIR}/{c}.npy"
    print("Data check : passed\n")

    print("Loading ViT-Small source model...")
    source = load_model()
    print()

    # Sanity check — baseline on gaussian_noise S1
    print("Sanity check (Baseline, gaussian_noise S1)...")
    _fn = make_baseline(source)
    _loader = get_loader("gaussian_noise", 1)
    acc = eval_loader(_fn, _loader)
    del _loader; torch.cuda.empty_cache()
    print(f"  Baseline gaussian_noise S1: {acc:.1f}%")
    print(f"  Expected: >60%  (ViT-S finetuned on CIFAR-10 should be strong at S1)")
    if acc < 40.0:
        print("  WARNING: lower than expected — check checkpoint path and model arch")
    print()

    # Main loop
    all_averaged, all_means = {}, {}
    for m in args.methods:
        avg, mean = load_averaged(m)
        if avg and args.skip_done:
            all_averaged[m]=avg; all_means[m]=mean
            print(f"Skipping {m} (saved: {mean:.2f}%)")
            continue
        results  = run_truly_continual(m, source)
        mean     = compute_averaged(m, results)
        all_averaged[m] = {c: np.mean([results[s][c] for s in SEVERITIES])
                           for c in ALL_CORRUPTIONS}
        all_means[m]    = mean
        torch.cuda.empty_cache()

    if all_averaged:
        generate_table(all_averaged, all_means)

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print("Summary:")
    for m in sorted(all_means, key=lambda x: -all_means[x]):
        flag = "  ← ours" if m == "ContinualTTA" else ""
        print(f"  {m:<18} {all_means[m]:.2f}%{flag}")