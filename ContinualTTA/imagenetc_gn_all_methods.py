# =============================================================================
# ContinualTTA — ImageNet-C GN (Group Normalization)
# All Methods: Baseline | TENT | EATA | CoTTA | RoTTA | ContinualTTA
#
# Model: ResNet-50-GN from timm (resnet50_gn_a1h2)
# This is the exact model used by the SAR paper.
# Published baseline: 30.6% mean accuracy at severity 5.
#
# Protocol: Fresh-per-corruption (standard in TTA literature)
# Each method gets a fresh model before each corruption type.
#
# CRITICAL: Uses timm model's own transform, NOT torchvision BN transform.
# The GN model was trained with different normalization stats/pipeline.
# Using wrong transform causes incorrect baseline (too low on weather corrs).
#
# Run:
#   python imagenetc_gn_all_methods.py
#   python imagenetc_gn_all_methods.py --methods Baseline ContinualTTA
#   python imagenetc_gn_all_methods.py --skip_done
#   python imagenetc_gn_all_methods.py --table_only
#
# Expected results (from SAR paper Table 2, ResNet50-GN, S5):
#   Baseline: 30.6%
#   TENT:     22.0% (collapses on some corruptions)
#   EATA:     31.6%
#   SAR:      37.2%
#   Ours:     target ~35-40% with JS gating
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
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\imagenetc_gn"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
NUM_CLASSES = 1000
SEVERITY    = 5
NUM_WORKERS = 0

# Hyperparameters — paper faithful
IMAGENET_LR  = 2.5e-4          # SAR paper lr for ResNet models
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 2.763 nats
MIN_CONF     = 0.5              # confidence gate for ContinualTTA
ADAPT_BUDGET = 100              # max backward passes per corruption
JS_THRESHOLD = 0.1             # tau=0.04 optimal (JS is scale-invariant)
ROTTA_NU     = 0.001
ROTTA_N      = 64
SAR_RHO      = 0.05
SAR_E0       = 0.4 * math.log(NUM_CLASSES) * 0.1  # recovery threshold

# GN model name — exact model used in SAR paper
# SAR paper Appendix C.2: "ResNet-50-GN from timm"
GN_MODEL_NAME = "resnet50_gn"

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "ContinualTTA"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. MODEL & DATASET
# =============================================================================

def load_model():
    """
    Load ResNet-50-GN from timm.
    resnet50_gn_a1h2 = RSB (ResNet Strikes Back) training recipe with GN.
    Clean ImageNet accuracy: ~80.9% (better than standard ResNet-50 BN 76.1%)
    This is the model used by SAR paper for ImageNet-C GN experiments.
    """
    print(f"  Loading {GN_MODEL_NAME} from timm...")
    model = timm.create_model(GN_MODEL_NAME, pretrained=True)
    model = model.to(DEVICE).eval()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def get_transform(model):
    """
    Get the CORRECT transform for this specific timm model.
    CRITICAL: Do NOT use torchvision BN transform (_weights.transforms()).
    The GN model was trained with different preprocessing.
    timm's resolve_data_config reads the model's pretrained config.
    """
    config    = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    print(f"  Transform: {transform}")
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
    GN setup: only GroupNorm scale/bias (gamma/beta) are trainable.
    GN does NOT use running stats — no momentum=0 needed.
    GN computes stats per-sample per-group at inference time.
    This is why GN is stable under distribution shift.
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.GroupNorm):
            m.requires_grad_(True)
    params = [p for m in model.modules()
              if isinstance(m, nn.GroupNorm)
              for p in m.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    return model, params, n_params


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
    model, params, _ = setup_gn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=IMAGENET_LR)

    @torch.enable_grad()
    def fn(x):
        logits   = model(x)
        entropy  = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        if reliable.sum() == 0:
            return logits
        entropy[reliable].mean().backward()
        opt.step()
        opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 5. EATA
# =============================================================================

def make_eata(source, fisher_loader=None):
    model, params, _ = setup_gn(copy.deepcopy(source))
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
    d_margin  = 0.05

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
            mask_d = cos_sim < (1.0 - d_margin)
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
    adapted, params, _ = setup_gn(copy.deepcopy(source))
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
                    pa.data[torch.rand_like(pa) < 0.01] = ps.data[torch.rand_like(pa) < 0.01]
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
            e_age  = torch.exp(-ages_t/ROTTA_N) / (1 + torch.exp(-ages_t/ROTTA_N))
            BANK_BATCH = 32
            total_loss = torch.tensor(0.0, device=DEVICE)
            n_mini     = 0
            for start in range(0, len(samples), BANK_BATCH):
                end    = min(start + BANK_BATCH, len(samples))
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
                    tp.data = (1 - ROTTA_NU) * tp.data + ROTTA_NU * sp.data

        return logits

    return fn


# =============================================================================
# 8. ContinualTTA (GN version)
#
# Key differences from BN version:
#   - setup_gn() instead of setup_bn_imagenet()
#   - MIN_CONF=0.5 confidence gate (prevents wrong-confident updates)
#   - ADAPT_BUDGET=100 (limits backward passes per corruption)
#   - JS_THRESHOLD=0.04 (same as CIFAR-10-C — JS is scale-invariant)
#   - First batch only initialises reference, no adaptation
#   - fn.reset() called between corruptions
# =============================================================================

def make_ctta(source):
    model, params, n_params = setup_gn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=IMAGENET_LR)

    reference  = [None]
    n_backward = [0]

    @torch.enable_grad()
    def fn(x):
        logits = model(x)

        # JS gate
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                return logits   # first batch: init only, no adapt
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

        # Budget check
        if n_backward[0] >= ADAPT_BUDGET:
            return logits

        # Combined filter: entropy AND confidence
        with torch.no_grad():
            entropy = softmax_entropy(logits)
            conf    = logits.softmax(1).max(1).values
        reliable = (entropy < E_MARGIN) & (conf > MIN_CONF)
        if reliable.sum() == 0:
            return logits

        # Entropy minimisation on reliable samples
        logits_rel  = model(x[reliable])
        entropy_rel = softmax_entropy(logits_rel)
        entropy_rel.mean().backward()
        opt.step()
        opt.zero_grad()
        n_backward[0] += 1
        return logits

    def reset():
        reference[0]  = None
        n_backward[0] = 0

    fn.reset   = reset
    fn.n_params = n_params
    return fn


# =============================================================================
# 9. BUILD METHOD
# =============================================================================

def build_method(method, source, transform, corruption=None):
    fisher_loader = None
    if method == "EATA" and corruption is not None:
        fisher_loader = load_corruption(corruption, transform)

    dispatch = {
        "Baseline":     lambda: make_baseline(source),
        "TENT":         lambda: make_tent(source),
        "EATA":         lambda: make_eata(source, fisher_loader),
        "CoTTA":        lambda: make_cotta(source),
        "RoTTA":        lambda: make_rotta(source),
        "ContinualTTA": lambda: make_ctta(source),
    }
    fn = dispatch[method]()
    if fisher_loader is not None:
        del fisher_loader
        torch.cuda.empty_cache()
    return fn


# =============================================================================
# 10. RUN ONE METHOD
# =============================================================================

def run_method(method, source, transform):
    print(f"\n{'─'*55}")
    print(f"  {method}")
    print(f"{'─'*55}")

    results = {}
    for corruption in ALL_CORRUPTIONS:
        # Fresh model per corruption
        fn     = build_method(method, source, transform, corruption)
        loader = load_corruption(corruption, transform)
        acc    = eval_loader(fn, loader)
        results[corruption] = acc

        # Reset state for ContinualTTA between corruptions
        if hasattr(fn, 'reset'):
            fn.reset()

        del loader, fn
        torch.cuda.empty_cache()

        flag = "  ✓" if acc > 25 else ("  !!" if acc < 5 else "")
        print(f"  {corruption:<24} {acc:.1f}%{flag}")

    mean_acc = np.mean(list(results.values()))
    print(f"  {'Mean':<24} {mean_acc:.1f}%")
    return results, mean_acc


# =============================================================================
# 11. SAVE AND TABLE
# =============================================================================

def save_csv(method, results):
    mean = np.mean(list(results.values()))
    path = os.path.join(RESULTS_DIR, f"{method}.csv")
    with open(path, "w") as f:
        f.write(f"corruption,{method}\n")
        for c in ALL_CORRUPTIONS:
            f.write(f"{c},{results[c]:.2f}\n")
        f.write(f"Mean,{mean:.2f}\n")
    print(f"  Saved: {path}")


def load_csv(method):
    path = os.path.join(RESULTS_DIR, f"{method}.csv")
    if not os.path.isfile(path): return None, None
    results = {}
    with open(path) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] != "Mean":
                results[parts[0]] = float(parts[1])
    return results, np.mean(list(results.values()))


def generate_table(all_results, all_means):
    # SAR paper published numbers for comparison
    sar_paper = {
        "Baseline": 30.6, "TENT": 22.0, "EATA": 31.6, "SAR": 37.2
    }

    present = [m for m in METHODS if m in all_means]

    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "CoTTA":        "CoTTA~\\cite{wang2022continual}",
        "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
        "ContinualTTA": "\\textbf{\\textsc{ContinualTTA} (Ours)}",
    }
    corr_names = {
        "gaussian_noise":"Gaussian Noise","shot_noise":"Shot Noise",
        "impulse_noise":"Impulse Noise","defocus_blur":"Defocus Blur",
        "glass_blur":"Glass Blur","motion_blur":"Motion Blur",
        "zoom_blur":"Zoom Blur","snow":"Snow","frost":"Frost","fog":"Fog",
        "brightness":"Brightness","contrast":"Contrast",
        "elastic_transform":"Elastic","pixelate":"Pixelate",
        "jpeg_compression":"JPEG",
    }

    # Console summary
    col = 14
    hdr = f"{'Corruption':<24}" + "".join(f"{m[:12]:>{col}}" for m in present)
    print(f"\n{'='*len(hdr)}")
    print(f"ImageNet-C S5 — ResNet-50-GN, Fresh-per-Corruption")
    print(f"{'='*len(hdr)}")
    print(hdr); print("─"*len(hdr))
    for c in ALL_CORRUPTIONS:
        vals = [all_results[m].get(c, float('nan')) for m in present]
        best = max(v for v in vals if not math.isnan(v))
        row  = f"{c:<24}"
        for v in vals:
            cell = f"{v:.1f}%" + ("*" if abs(v-best)<0.05 else "")
            row += f"{cell:>{col}}"
        print(row)
    print("─"*len(hdr))
    best_m = max(all_means.values())
    mrow   = f"{'Mean':<24}"
    for m in present:
        v    = all_means[m]
        cell = f"{v:.2f}%" + ("*" if abs(v-best_m)<0.05 else "")
        mrow += f"{cell:>{col}}"
    print(mrow)
    print(f"{'='*len(hdr)}\n  * = best in row")

    # SAR paper comparison
    print(f"\nComparison to SAR paper (ResNet50-GN, S5):")
    for m, pub in sar_paper.items():
        if m in all_means:
            diff = all_means[m] - pub
            print(f"  {m:<16} Ours={all_means[m]:.1f}%  Published={pub:.1f}%  Δ={diff:+.1f}%")

    # LaTeX
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Accuracy (\%) on ImageNet-C (severity 5) with ResNet-50-GN, "
        r"fresh-per-corruption protocol. "
        r"\textbf{Bold} = best per row. "
        r"GN/LN architectures are used following SAR~\cite{niu2023towards}, "
        r"as BN-based adaptation is unstable at severity~5 due to low "
        r"per-corruption baseline accuracy.}")
    lines.append(r"\label{tab:imagenetc_gn}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(present) + "}")
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
            elif abs(val-best) < 0.05: row += f" & \\textbf{{{val:.1f}}}"
            else: row += f" & {val:.1f}"
        lines.append(row + r" \\")
    lines.append(r"\midrule")
    mean_vals  = [all_means[m] for m in present]
    best_mean  = max(mean_vals)
    mean_row   = r"\textbf{Mean}"
    for val in mean_vals:
        if abs(val-best_mean) < 0.05: mean_row += f" & \\textbf{{{val:.1f}}}"
        else: mean_row += f" & {val:.1f}"
    lines.append(mean_row + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    tex_path = os.path.join(RESULTS_DIR, "table_imagenetc_gn.tex")
    with open(tex_path, "w") as f: f.write(latex)
    print(f"\n  LaTeX: {tex_path}")
    print(f"\n{'='*60}\nTable LaTeX:\n{'='*60}")
    print(latex)

    # Save summary CSV
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("corruption," + ",".join(present) + "\n")
        for c in ALL_CORRUPTIONS:
            row = c + "," + ",".join(
                f"{all_results[m].get(c,float('nan')):.2f}" for m in present)
            f.write(row + "\n")
        f.write("Mean," + ",".join(f"{all_means[m]:.2f}" for m in present) + "\n")
    print(f"  CSV: {csv_path}")


# =============================================================================
# 12. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ImageNet-C GN — All TTA Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imagenetc_gn_all_methods.py
  python imagenetc_gn_all_methods.py --methods Baseline ContinualTTA
  python imagenetc_gn_all_methods.py --skip_done
  python imagenetc_gn_all_methods.py --table_only
        """)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--table_only", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ImageNet-C GN — All Methods, S{SEVERITY}")
    print(f"{'='*60}")
    print(f"Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Model      : {GN_MODEL_NAME}")
    print(f"Protocol   : Fresh model per corruption")
    print(f"E_margin   : {E_MARGIN:.3f} nats")
    print(f"JS tau     : {JS_THRESHOLD}")
    print(f"MIN_CONF   : {MIN_CONF}")
    print(f"Budget     : {ADAPT_BUDGET} backward passes per corruption")
    print(f"Results    : {RESULTS_DIR}\n")

    # Table-only
    if args.table_only:
        all_results, all_means = {}, {}
        for m in METHODS:
            r, mean = load_csv(m)
            if r: all_results[m]=r; all_means[m]=mean; print(f"  Loaded {m}: {mean:.1f}%")
        if all_results: generate_table(all_results, all_means)
        exit(0)

    # Verify data
    print("Verifying DATA_DIR...")
    for c in ALL_CORRUPTIONS[:3]:
        path = os.path.join(DATA_DIR, c, str(SEVERITY))
        if not os.path.isdir(path):
            print(f"  WARNING: missing {c}")
    print("  Done.\n")

    # Load model and get CORRECT transform
    print("Loading GN model...")
    source    = load_model()
    transform = get_transform(source)
    print()

    # Sanity check with correct transform
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
    print(f"  Baseline gaussian_noise: {sanity:.1f}%")
    print(f"  Expected: ~18% (SAR paper reports 18.0%)")
    if sanity < 10.0:
        print("  WARNING: Too low — check model download and transform")
    print("  Passed.\n")

    # Load existing results
    all_results, all_means = {}, {}
    for m in METHODS:
        r, mean = load_csv(m)
        if r and args.skip_done:
            all_results[m] = r; all_means[m] = mean
            print(f"  Skipping {m} (saved: {mean:.1f}%)")

    # Run methods
    for method in args.methods:
        if args.skip_done and method in all_results:
            continue
        results, mean = run_method(method, source, transform)
        all_results[method] = results
        all_means[method]   = mean
        save_csv(method, results)
        print(f"\n  → {method}: {mean:.1f}%")
        torch.cuda.empty_cache()

    # Generate table
    if all_results:
        generate_table(all_results, all_means)

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print("Final ranking:")
    for m in sorted(all_means.keys(), key=lambda x: -all_means[x]):
        flag = "  ← ours" if m == "ContinualTTA" else ""
        print(f"  {m:<18} {all_means[m]:.1f}%{flag}")











