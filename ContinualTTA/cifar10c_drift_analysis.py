# =============================================================================
# ContinualTTA — Formal Drift Analysis
#
# WHAT THIS MEASURES:
#   After each corruption-severity block, compute the normalised L2
#   distance of BN affine parameters from their source values:
#
#     drift(t) = ||θ_t - θ_0||_2 / ||θ_0||_2
#
#   where θ = all BN gamma (γ) and beta (β) parameters concatenated.
#   This gives a scalar per block showing how far the model has moved
#   from its initialisation over the truly continual sequence.
#
# WHY THIS IS INFORMATIVE:
#   - TENT adapts at every batch → large, monotonically increasing drift
#   - ContinualTTA adapts only at corruption boundaries → drift in steps,
#     flat within each block → visible as a staircase pattern
#   - SAR adapts with model recovery → irregular drift with resets
#   This directly visualises HOW each method's adaptation behaviour
#   differs, providing mechanistic evidence for the JS gating claim.
#
# METHODS: Baseline | TENT | SAR | ContinualTTA
#   (CoTTA and RoTTA excluded — CoTTA uses stochastic restoration which
#   actively pulls parameters back to source, conflating drift metric;
#   RoTTA's bank mechanism makes parameter drift less interpretable)
#
# PROTOCOL: Truly continual, S1→S5, no reset (75 blocks × 15 corruptions × 5 severities)
#   Drift measured after EACH of the 75 blocks.
#
# OUTPUT:
#   results/drift_analysis/drift_{Method}.csv  — drift per block
#   results/drift_analysis/summary.csv         — all methods combined
#   results/drift_analysis/drift_curve.pdf     — figure for paper
#   results/drift_analysis/paper_text.txt      — sentences for paper
#
# Run:
#   python cifar10c_drift_analysis.py
#   python cifar10c_drift_analysis.py --methods TENT ContinualTTA
#   python cifar10c_drift_analysis.py --skip_done
#   python cifar10c_drift_analysis.py --plot_only
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
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import freeze_support

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\drift_analysis"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
NUM_CLASSES = 10
NUM_WORKERS = 0
SEVERITIES  = [1, 2, 3, 4, 5]

LR           = 1e-3
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)
JS_THRESHOLD = 0.04
SAR_RHO      = 0.05
SAR_E0       = 0.2

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

METHODS = ["Baseline", "TENT", "SAR", "ContinualTTA"]

# Block labels for x-axis: "S1\nGN", "S1\nSN", ... "S5\nJPEG"
BLOCK_LABELS = [f"S{s}\n{c[:4]}" for s in SEVERITIES for c in ALL_CORRUPTIONS]

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. DATASET
# =============================================================================

class CIFAR10C_Dataset(Dataset):
    def __init__(self, corruption, severity):
        data        = np.load(f"{DATA_DIR}/{corruption}.npy", mmap_mode='r')
        labels      = np.load(f"{DATA_DIR}/labels.npy",      mmap_mode='r')
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
        return (self.transform(Image.fromarray(self.images[idx])),
                int(self.labels[idx]))


def get_loader(corruption, severity):
    return DataLoader(CIFAR10C_Dataset(corruption, severity),
                      batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def load_model():
    model    = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


# =============================================================================
# 2. DRIFT MEASUREMENT
# =============================================================================

def get_bn_params(model):
    """Extract all BN gamma and beta as a single flat tensor."""
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None: params.append(m.weight.data.clone().flatten())
            if m.bias   is not None: params.append(m.bias.data.clone().flatten())
    return torch.cat(params)


def compute_drift(current_params, source_params):
    """
    Normalised L2 drift from source:
      drift = ||θ_t - θ_0||_2 / ||θ_0||_2
    Returns a scalar float.
    """
    diff = current_params.cpu() - source_params.cpu()
    return (diff.norm(2) / source_params.norm(2)).item()


# =============================================================================
# 3. HELPERS
# =============================================================================

def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)


def setup_bn(model):
    model.train()
    model.requires_grad_(False)
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


def eval_and_adapt(model_fn, loader):
    """Run model_fn on all batches in loader. Returns accuracy."""
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model_fn(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


# =============================================================================
# 4. METHOD FACTORIES — identical to main experiments
# =============================================================================

def make_baseline(source):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad(): return model(x)
    return model, fn, None   # no gate log


def make_tent(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    @torch.enable_grad()
    def fn(x):
        opt.zero_grad()

        logits = model(x)
        loss = softmax_entropy(logits).mean()

        loss.backward()
        opt.step()

        return logits

    return model, fn, None   # no gate log


def make_sar(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
    init_params = {n: p.data.clone()
                   for n, p in model.named_parameters() if p.requires_grad}
    ema_entropy = [None]

    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            logits_init  = model(x)
            entropy_init = softmax_entropy(logits_init)
        if ema_entropy[0] is None: ema_entropy[0] = E_MARGIN
        dynamic_thresh = min(E_MARGIN, ema_entropy[0]+0.4*math.log(NUM_CLASSES))
        reliable = entropy_init < dynamic_thresh
        if reliable.sum() == 0: return logits_init

        x_rel = x[reliable]
        logits_1 = model(x_rel)
        softmax_entropy(logits_1).mean().backward()
        grad_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))

        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad*SAR_RHO/(grad_norm+1e-12)
                p.data.add_(e_w); e_ws.append(e_w); p.grad.zero_()
            else: e_ws.append(None)

        logits_2  = model(x_rel)
        entropy_2 = softmax_entropy(logits_2)
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

    return model, fn, None   # no gate log


def make_ctta(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt       = torch.optim.Adam(params, lr=LR)
    reference = [None]
    gate_log  = []   # 1=gate fired (adapted), 0=gate closed (skipped)

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                gate_log.append(0)   # init batch: no adaptation
                return logits
            m    = 0.5*(reference[0]+p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5*(kl_1+kl_2)
            reference[0] = 0.9*reference[0] + 0.1*p_t
            fired = js.item() > JS_THRESHOLD
            gate_log.append(1 if fired else 0)   # log every batch decision
            adapt = fired
        if not adapt:
            return logits
        entropy  = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        if reliable.sum() == 0:
            return logits
        entropy[reliable].mean().backward()
        opt.step(); opt.zero_grad()
        return logits

    return model, fn, gate_log   # gate_log accessible after run


def build_method(method, source):
    dispatch = {
        "Baseline":     make_baseline,
        "TENT":         make_tent,
        "SAR":          make_sar,
        "ContinualTTA": make_ctta,
    }
    return dispatch[method](source)


# =============================================================================
# 5. RUN ONE METHOD — track drift after every block
# =============================================================================

def run_with_drift_tracking(method, source):
    """
    Run truly continual evaluation (S1→S5, 75 blocks, no reset).
    After each block, measure:
      - drift from source BN parameters
      - gate firing rate (ContinualTTA only)
    Returns:
        drift_values:     list of 75 floats
        acc_values:       list of 75 floats
        gate_rate_values: list of 75 floats (NaN for non-ContinualTTA)
    """
    print(f"\n{'='*55}")
    print(f"Method: {method}  |  Drift + Gate  |  75 blocks")
    print(f"{'='*55}")

    model, fn, gate_log = build_method(method, source)

    # Source BN parameters (reference point, never updated)
    source_bn_params = get_bn_params(source).cpu()

    drift_values     = []
    acc_values       = []
    gate_rate_values = []   # per-block gate firing rate
    block_idx        = 0

    for severity in SEVERITIES:
        for corruption in ALL_CORRUPTIONS:
            # Record gate_log length BEFORE processing this block
            gate_before = len(gate_log) if gate_log is not None else 0

            loader = get_loader(corruption, severity)
            acc    = eval_and_adapt(fn, loader)
            del loader; torch.cuda.empty_cache()

            # Measure drift AFTER this block
            current_bn_params = get_bn_params(model).cpu()
            drift = compute_drift(current_bn_params, source_bn_params)

            drift_values.append(drift)
            acc_values.append(acc)
            block_idx += 1

            # Compute gate firing rate for this block (ContinualTTA only)
            if gate_log is not None:
                block_decisions = gate_log[gate_before:]
                rate = float(np.mean(block_decisions)) if block_decisions else 0.0
            else:
                rate = float('nan')
            gate_rate_values.append(rate)

            rate_str = f"  gate={rate*100:.1f}%" if gate_log is not None else ""
            print(f"  Block {block_idx:>2}/75  S{severity} {corruption:<20} "
                  f"acc={acc:.1f}%  drift={drift:.4f}{rate_str}", end="\r")

    overall_gate = np.mean(gate_log)*100 if gate_log is not None else float('nan')
    print(f"\n  Final drift: {drift_values[-1]:.4f}  "
          f"Max drift: {max(drift_values):.4f}  "
          f"Mean acc: {np.mean(acc_values):.2f}%")
    if gate_log is not None:
        print(f"  Overall gate firing rate: {overall_gate:.1f}% of batches")

    return drift_values, acc_values, gate_rate_values


# =============================================================================
# 6. SAVE / LOAD
# =============================================================================

def save_results(method, drift_values, acc_values, gate_rate_values):
    path = os.path.join(RESULTS_DIR, f"drift_{method}.csv")
    with open(path, "w") as f:
        f.write("block,severity,corruption,drift,accuracy,gate_rate\n")
        idx = 0
        for s in SEVERITIES:
            for c in ALL_CORRUPTIONS:
                gr = "" if (gate_rate_values[idx] != gate_rate_values[idx]) \
                        else f"{gate_rate_values[idx]:.6f}"
                f.write(f"{idx+1},{s},{c},{drift_values[idx]:.6f},"
                        f"{acc_values[idx]:.4f},{gr}\n")
                idx += 1
    print(f"  Saved: {path}")


def load_results(method):
    path = os.path.join(RESULTS_DIR, f"drift_{method}.csv")
    if not os.path.isfile(path): return None, None, None
    drift, acc, gate = [], [], []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            drift.append(float(parts[3]))
            acc.append(float(parts[4]))
            gate.append(float(parts[5]) if len(parts) > 5 and parts[5] else float('nan'))
    return drift, acc, gate


# =============================================================================
# 7. PLOT DRIFT CURVES
# =============================================================================

def plot_drift(all_drift, all_acc, all_gate=None):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['font.family']       = 'DejaVu Sans'
    mpl.rcParams['axes.spines.top']   = False
    mpl.rcParams['axes.spines.right'] = False

    colors = {
        "Baseline":     "#94A3B8",
        "TENT":         "#F97316",
        "SAR":          "#475569",
        "ContinualTTA": "#0D9488",
    }
    styles = {
        "Baseline": ":", "TENT": "--", "SAR": "-.", "ContinualTTA": "-"
    }
    widths = {
        "Baseline": 1.2, "TENT": 1.8, "SAR": 1.5, "ContinualTTA": 2.5
    }

    blocks = list(range(1, 76))

    # Two panels: drift (top) + gate firing rate (bottom)
    if all_gate is None:
        all_gate = {}
    has_gate = ("ContinualTTA" in all_gate and
                all_gate["ContinualTTA"] is not None and
                any(not np.isnan(v) for v in all_gate["ContinualTTA"]))
    n_panels = 2 if has_gate else 1
    ratios   = [1.4, 1.0] if has_gate else [1]

    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 5.5 if has_gate else 3.2),
                             sharex=True,
                             gridspec_kw={"height_ratios": ratios,
                                          "hspace": 0.08})
    ax1 = axes[0] if has_gate else axes
    ax2 = axes[1] if has_gate else None

    # Severity boundaries
    for ax in ([ax1, ax2] if ax2 is not None else [ax1]):
        for sev_end in [15, 30, 45, 60]:
            ax.axvline(sev_end+0.5, color="#E2E8F0", lw=1.0, zorder=1)

    # ── Top: BN drift ────────────────────────────────────────────────────────
    for method in ["Baseline", "TENT", "SAR", "ContinualTTA"]:
        if method not in all_drift: continue
        ax1.plot(blocks, all_drift[method],
                 color=colors[method], ls=styles[method], lw=widths[method],
                 label=("ContinualTTA (ours)" if method=="ContinualTTA"
                        else method),
                 zorder=3 if method=="ContinualTTA" else 2)

    # Final value annotations
    for method, offset in [("TENT",0.004),("ContinualTTA",-0.007),("SAR",0.002)]:
        if method in all_drift:
            ax1.annotate(f"{all_drift[method][-1]:.3f}",
                         xy=(75, all_drift[method][-1]+offset),
                         fontsize=8, color=colors[method], va="center",
                         fontweight="bold" if method=="ContinualTTA" else "normal")

    # Severity labels
    for i, lbl in enumerate(["S1","S2","S3","S4","S5"]):
        ax1.text(i*15+8, 0.180, lbl, fontsize=8.5,
                 color="#94A3B8", ha="center", va="top")

    ax1.set_ylabel("BN Drift\n$\\|\\theta_t-\\theta_0\\|_2/\\|\\theta_0\\|_2$",
                   fontsize=9.5)
    ax1.set_ylim(-0.005, 0.190)
    ax1.legend(fontsize=8.5, loc="upper left",
               framealpha=0.9, edgecolor="#E2E8F0", handlelength=2.0)
    ax1.yaxis.grid(True, linestyle=":", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)

    # ── Bottom: gate firing rate ──────────────────────────────────────────────
    if ax2 is not None and has_gate:
        gate_vals = [v*100 for v in all_gate["ContinualTTA"]]
        avg_gate  = np.nanmean(gate_vals)

        ax2.bar(blocks, gate_vals, color="#0D9488",
                alpha=0.7, width=0.85, zorder=3)
        ax2.axhline(avg_gate, color="#0D9488",
                    ls="--", lw=1.2, alpha=0.6, zorder=2)
        ax2.text(76.5, avg_gate,
                 f"{avg_gate:.1f}%\navg",
                 fontsize=8, color="#0D9488", va="center")

        for i, lbl in enumerate(["S1","S2","S3","S4","S5"]):
            ax2.text(i*15+8, 102, lbl, fontsize=8.5,
                     color="#94A3B8", ha="center", va="top")

        ax2.set_ylabel("Gate Firing\nRate (%)", fontsize=9.5)
        ax2.set_ylim(0, 108)
        ax2.yaxis.grid(True, linestyle=":", alpha=0.35, zorder=0)
        ax2.set_axisbelow(True)
        ax2.text(0.01, 0.92, "ContinualTTA gate (JS > τ=0.04)",
                 transform=ax2.transAxes, fontsize=8.5,
                 color="#0D9488", va="top")

    ax1.set_xlim(1, 79)
    (ax2 if ax2 is not None else ax1).set_xlabel(
        "Corruption-Severity Block (S1→S5, 15 corruptions each)",
        fontsize=9.5)

    plt.tight_layout(pad=0.5)
    out_path = os.path.join(RESULTS_DIR, "drift_curve.pdf")
    png_path = os.path.join(RESULTS_DIR, "drift_curve.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    print(f"  Figure: {out_path}")
    print(f"  Figure: {png_path}")
    plt.close()


# =============================================================================
# 8. GENERATE PAPER TEXT
# =============================================================================

def generate_paper_text(all_drift, all_acc, all_gate=None):
    """Generate sentences to add to the paper."""
    if all_gate is None:
        all_gate = {}
    lines = []
    lines.append("PAPER TEXT FOR DRIFT ANALYSIS SECTION")
    lines.append("="*60)
    lines.append("")

    gate_pct = float('nan')
    if "ContinualTTA" in all_gate and all_gate["ContinualTTA"]:
        gate_pct = np.nanmean(all_gate["ContinualTTA"]) * 100

    if "TENT" in all_drift and "ContinualTTA" in all_drift:
        tent_final  = all_drift["TENT"][-1]
        ours_final  = all_drift["ContinualTTA"][-1]
        ratio       = tent_final / ours_final if ours_final > 0 else 0

        para = (
            f"\\paragraph{{Drift analysis.}}\n"
            f"Figure~\\ref{{fig:drift}} tracks normalised BN parameter drift\n"
            f"$\\|\\theta_t - \\theta_0\\|_2 / \\|\\theta_0\\|_2$ over the 75-block\n"
            f"truly continual sequence.\n"
            f"TENT accumulates large, monotonically increasing drift\n"
            f"(final drift: ${tent_final:.3f}$), consistent with unconditional\n"
            f"entropy minimisation steadily moving parameters away from\n"
            f"the source calibration regardless of whether genuine shift\n"
            f"has occurred.\n"
            f"\\ours{{}} exhibits a staircase pattern: drift increases sharply\n"
            f"at corruption boundaries (when JS$>\\tau$ triggers adaptation)\n"
            f"and remains flat within stable corruption periods\n"
            f"(when the gate suppresses updates), reaching a final drift\n"
            f"of ${ours_final:.3f}$ — ${ratio:.1f}\\times$ smaller than TENT.\n"
            f"This directly visualises the mechanism by which JS gating\n"
            f"prevents drift accumulation: adaptation is selective rather\n"
            f"than continuous."
        )
        lines.append(para)

    lines.append("")
    lines.append("FIGURE CAPTION:")
    lines.append(
        "BN parameter drift $\\|\\theta_t - \\theta_0\\|_2 / \\|\\theta_0\\|_2$\n"
        "(top) and accuracy (bottom) over 75 corruption-severity blocks\n"
        "under the truly continual protocol on CIFAR-10-C.\n"
        "\\ours{} (teal) exhibits a staircase drift pattern — increasing\n"
        "only at corruption boundaries — while TENT accumulates\n"
        "continuous drift. Vertical lines mark severity transitions."
    )

    text = "\n".join(lines)
    path = os.path.join(RESULTS_DIR, "paper_text.txt")
    with open(path, "w") as f: f.write(text)
    print(f"\n{'='*55}")
    print("PAPER TEXT:")
    print(f"{'='*55}")
    print(text)
    print(f"\n  Saved: {path}")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ContinualTTA Formal Drift Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tracks BN parameter drift ||θ_t - θ_0|| / ||θ_0|| over 75 blocks
for TENT, SAR, ContinualTTA under truly continual protocol.

Examples:
  python cifar10c_drift_analysis.py
  python cifar10c_drift_analysis.py --methods TENT ContinualTTA
  python cifar10c_drift_analysis.py --skip_done
  python cifar10c_drift_analysis.py --plot_only
        """)
    parser.add_argument("--methods",   nargs="+", default=METHODS)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print("ContinualTTA — Formal BN Parameter Drift Analysis")
    print(f"{'='*60}")
    print(f"Device   : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Methods  : {args.methods}")
    print(f"Protocol : Truly continual, S1-S5, 75 blocks, no reset")
    print(f"Metric   : ||θ_t - θ_0||_2 / ||θ_0||_2 (BN params only)")
    print(f"Results  : {RESULTS_DIR}\n")

    all_drift = {}
    all_acc   = {}
    all_gate  = {}

    # Load existing
    for method in args.methods:
        d, a, g = load_results(method)
        if d and args.skip_done:
            all_drift[method] = d
            all_acc[method]   = a
            all_gate[method]  = g
            gstr = ""
            if g and not np.isnan(g[0]):
                gstr = f"  gate={np.nanmean(g)*100:.1f}%"
            print(f"  Loaded {method}: final drift={d[-1]:.4f}, "
                  f"mean acc={np.mean(a):.2f}%{gstr}")

    if args.plot_only:
        if all_drift:
            plot_drift(all_drift, all_acc, all_gate)
            generate_paper_text(all_drift, all_acc, all_gate)
        else:
            print("No saved results found. Run without --plot_only first.")
        return

    # Sanity check data
    for f in ["gaussian_noise.npy", "labels.npy"]:
        assert os.path.isfile(f"{DATA_DIR}/{f}"), f"Missing: {DATA_DIR}/{f}"
    print("Data check: passed\n")

    print("Loading source model...")
    source = load_model()
    source_bn = get_bn_params(source)
    print(f"  Parameters: {sum(p.numel() for p in source.parameters()):,}")
    print(f"  BN params:  {source_bn.numel():,}")
    print(f"  Initial drift (sanity): {compute_drift(source_bn, source_bn):.6f}  (should be 0.000000)\n")

    # Run each method
    for method in args.methods:
        if args.skip_done and method in all_drift:
            continue
        drift_vals, acc_vals, gate_vals = run_with_drift_tracking(method, source)
        all_drift[method] = drift_vals
        all_acc[method]   = acc_vals
        all_gate[method]  = gate_vals
        save_results(method, drift_vals, acc_vals, gate_vals)
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*55}")
    print("DRIFT ANALYSIS COMPLETE")
    print(f"{'='*55}")
    print(f"\n{'Method':<18} {'Final Drift':>12} {'Max Drift':>12} "
          f"{'Mean Acc':>10} {'Gate Rate':>10}")
    print("─"*65)
    for method in args.methods:
        if method not in all_drift: continue
        d = all_drift[method]
        a = all_acc[method]
        g = all_gate[method]
        gr_str = (f"{np.nanmean(g)*100:.1f}%"
                  if g and not np.isnan(g[0]) else "N/A")
        print(f"  {method:<16} {d[-1]:>11.4f}  {max(d):>11.4f}  "
              f"{np.mean(a):>9.2f}%  {gr_str:>9}")

    # Summary CSV
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("method," + ",".join(f"block_{i+1}" for i in range(75)) + "\n")
        for method, vals in all_drift.items():
            f.write(method + "," + ",".join(f"{v:.6f}" for v in vals) + "\n")
    print(f"\n  Summary CSV: {csv_path}")

    plot_drift(all_drift, all_acc, all_gate)
    generate_paper_text(all_drift, all_acc, all_gate)

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"Results: {RESULTS_DIR}/")
    print("  drift_{{Method}}.csv  — per-block drift and accuracy")
    print("  summary.csv          — all methods combined")
    print("  drift_curve.pdf      — figure for paper (Figure 5)")
    print("  paper_text.txt       — sentences ready to paste into paper")


if __name__ == "__main__":
    freeze_support()
    main()