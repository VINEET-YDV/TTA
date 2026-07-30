# =============================================================================
# ContinualTTA — Multi-Seed ALL-SEVERITY (S1-S5) Truly Continual Evaluation
# Methods: TENT | EATA | SAR | ContinualTTA
# Seeds:   42, 123, 456
#
# Protocol: Truly continual — no reset between any corruption or severity.
#   One model per (method, seed) runs through S1×15 → S2×15 → … → S5×15.
#
# SEEDING: Within-block shuffle via seeded generator per (severity, corruption).
#   Keeps the block SEQUENCE fixed; only within-block batch order varies.
#
# Output:
#   results/multiseed_allsev/{Method}_seed{S}.csv
#   results/multiseed_allsev/summary.csv
#   results/multiseed_allsev/significance.txt
#   results/multiseed_allsev/table_multiseed.tex
#
# Run:
#   python cifar10c_multiseed_allsev.py --verify
#   python cifar10c_multiseed_allsev.py
#   python cifar10c_multiseed_allsev.py --skip_done
#   python cifar10c_multiseed_allsev.py --table_only
# =============================================================================

import os, copy, math, random, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import freeze_support

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\multiseed_allsev"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
NUM_CLASSES = 10
NUM_WORKERS = 0
SEVERITIES  = [1, 2, 3, 4, 5]      # all five severities
SEEDS       = [42, 123, 456]

# Hyperparameters — identical to main experiments
LR           = 1e-3
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 0.921 nats
JS_THRESHOLD = 0.04
SAR_RHO      = 0.05
SAR_E0       = 0.2

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

# Block index for per-block seeded shuffle — now (severity, corruption)
BLOCK_INDEX = {
    (sev, corr): i * len(ALL_CORRUPTIONS) + j
    for i, sev in enumerate(SEVERITIES)
    for j, corr in enumerate(ALL_CORRUPTIONS)
}

METHODS = ["TENT", "EATA", "SAR", "ContinualTTA"]

# =============================================================================
# 1. SEEDING
# =============================================================================
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def get_loader(corruption, severity, seed):
    block_idx = BLOCK_INDEX[(severity, corruption)]
    g = torch.Generator()
    g.manual_seed(seed * 100000 + block_idx)   # ensures unique shuffle per block
    return DataLoader(
        CIFAR10C_Dataset(corruption, severity),
        batch_size=BATCH_SIZE, shuffle=True, generator=g,
        num_workers=NUM_WORKERS, pin_memory=True)

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

# =============================================================================
# 3. VERIFY SEEDS DIFFER
# =============================================================================
def verify_seeds():
    print(f"{'='*55}")
    print("VERIFY: seeds produce different batch compositions")
    print(f"{'='*55}")
    first_batches = {}
    for seed in SEEDS:
        loader = get_loader("gaussian_noise", 1, seed)
        _, y   = next(iter(loader))
        first_batches[seed] = y[:8].tolist()
        print(f"  Seed {seed:>3}: {first_batches[seed]}")
        del loader
    all_same = all(v == first_batches[SEEDS[0]] for v in first_batches.values())
    print()
    if all_same:
        print("  FAIL — seeds produce identical batches. Do not proceed.")
        return False
    print("  PASS — seeds produce genuinely different batch compositions.")
    print("  Safe to run the full evaluation.\n")
    return True

# =============================================================================
# 4. HELPERS
# =============================================================================
def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)

def eval_loader(model_fn, loader):
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model_fn(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total

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

# =============================================================================
# 5. METHODS (paper‑faithful)
# =============================================================================
def make_tent(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        loss = softmax_entropy(logits).mean()
        loss.backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn

def make_eata(source, fisher_loader=None):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    fisher = {n: torch.zeros_like(p) for n,p in model.named_parameters() if p.requires_grad}
    if fisher_loader is not None:
        model.train()
        for i,(x,_) in enumerate(fisher_loader):
            if i >= 10: break
            x = x.to(DEVICE)
            softmax_entropy(model(x)).mean().backward()
            for n,p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()
            model.zero_grad()
        for n in fisher: fisher[n] /= 10
    ref_probs = [None]
    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        entropy = softmax_entropy(logits)
        probs = logits.softmax(1)
        mask_e = entropy < E_MARGIN
        if ref_probs[0] is not None:
            cos_sim = F.cosine_similarity(ref_probs[0].unsqueeze(0).expand(probs.size(0),-1), probs, dim=1)
            mask_d = cos_sim < 0.95
        else:
            mask_d = torch.ones(probs.size(0), dtype=torch.bool, device=DEVICE)
        mask = mask_e & mask_d
        if mask.sum() == 0: return logits
        with torch.no_grad():
            ref_probs[0] = probs[mask].mean(0).detach() if ref_probs[0] is None else 0.9*ref_probs[0] + 0.1*probs[mask].mean(0).detach()
        fisher_reg = sum((fisher[n]*p.pow(2)).sum() for n,p in model.named_parameters() if p.requires_grad and n in fisher)
        loss = entropy[mask].mean() + 1e-3*fisher_reg
        loss.backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn

def make_sar(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
    init_params = {n: p.data.clone() for n,p in model.named_parameters() if p.requires_grad}
    ema_entropy = [None]
    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            logits_init = model(x)
            entropy_init = softmax_entropy(logits_init)
        if ema_entropy[0] is None: ema_entropy[0] = E_MARGIN
        dynamic_thresh = min(E_MARGIN, ema_entropy[0] + 0.4*math.log(NUM_CLASSES))
        reliable = entropy_init < dynamic_thresh
        if reliable.sum() == 0: return logits_init
        x_rel = x[reliable]
        logits_1 = model(x_rel)
        softmax_entropy(logits_1).mean().backward()
        grad_norm = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
                p.data.add_(e_w); e_ws.append(e_w); p.grad.zero_()
            else: e_ws.append(None)
        logits_2 = model(x_rel)
        entropy_2 = softmax_entropy(logits_2)
        if (entropy_2 < E_MARGIN).sum() > 0: entropy_2[entropy_2 < E_MARGIN].mean().backward()
        for p,e_w in zip(params, e_ws):
            if e_w is not None: p.data.sub_(e_w)
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            logits_out = model(x)
            entropy_out = softmax_entropy(logits_out)
            ema_entropy[0] = 0.9*ema_entropy[0] + 0.1*entropy_out.mean().item()
            if ema_entropy[0] < SAR_E0:
                for n,p in model.named_parameters():
                    if p.requires_grad and n in init_params: p.data.copy_(init_params[n])
                ema_entropy[0] = None
        return logits_out
    return fn

def make_ctta(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    reference = [None]
    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                return logits
            m = 0.5 * (reference[0] + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0), reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0), p_t.unsqueeze(0), reduction="batchmean")
            js = 0.5 * (kl_1 + kl_2)
            reference[0] = 0.9 * reference[0] + 0.1 * p_t
            if js.item() <= JS_THRESHOLD: return logits
        entropy = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        if reliable.sum() == 0: return logits
        entropy[reliable].mean().backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn

def build_method(method, source, seed):
    fisher_loader = None
    if method == "EATA":
        # Fisher from first 10 batches of (S1, gaussian_noise) — same seed for reproducibility
        fisher_loader = get_loader(ALL_CORRUPTIONS[0], 1, seed)
    dispatch = {
        "TENT":         lambda: make_tent(source),
        "EATA":         lambda: make_eata(source, fisher_loader),
        "SAR":          lambda: make_sar(source),
        "ContinualTTA": lambda: make_ctta(source),
    }
    fn = dispatch[method]()
    if fisher_loader is not None:
        del fisher_loader; torch.cuda.empty_cache()
    return fn

# =============================================================================
# 6. RUN ONE (METHOD, SEED) — all severities
# =============================================================================
def run_one_seed(method, seed, source):
    print(f"\n{'='*60}")
    print(f"Method: {method}  |  Seed: {seed}  |  Truly Continual S1-S5")
    print(f"{'='*60}")

    set_global_seed(seed)
    fn = build_method(method, source, seed)

    all_results = {sev: {} for sev in SEVERITIES}

    for severity in SEVERITIES:
        print(f"\n  Severity {severity}")
        for corruption in ALL_CORRUPTIONS:
            loader = get_loader(corruption, severity, seed)
            acc    = eval_loader(fn, loader)
            all_results[severity][corruption] = acc
            del loader; torch.cuda.empty_cache()
            print(f"    {corruption:<24} {acc:.1f}%")
        sev_mean = np.mean(list(all_results[severity].values()))
        print(f"    S{severity} mean: {sev_mean:.2f}%")

    # Compute S1-S5 mean per corruption, then overall mean
    averaged = {}
    for corruption in ALL_CORRUPTIONS:
        averaged[corruption] = np.mean([all_results[sev][corruption] for sev in SEVERITIES])
    mean_overall = np.mean(list(averaged.values()))
    return averaged, mean_overall

# =============================================================================
# 7. SAVE / LOAD
# =============================================================================
def save_csv(method, seed, results, mean_acc):
    path = os.path.join(RESULTS_DIR, f"{method}_seed{seed}.csv")
    with open(path, "w") as f:
        f.write(f"corruption,{method}_seed{seed}\n")
        for c in results:
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

# =============================================================================
# 8. SUMMARY + SIGNIFICANCE + LATEX
# =============================================================================
def generate_summary(all_seed_means):
    print(f"\n{'='*60}")
    print("MULTI-SEED SUMMARY (S1-S5 truly continual, 3 seeds)")
    print(f"{'='*60}")

    summary = {}
    for method in METHODS:
        if method not in all_seed_means: continue
        vals = np.array(all_seed_means[method])
        mean = vals.mean()
        std  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        summary[method] = (mean, std, vals.tolist())
        flag = "  ← ours" if method == "ContinualTTA" else ""
        warn = "  ⚠ high variance" if std > 3.0 else ""
        print(f"  {method:<18} {mean:.2f}% ± {std:.2f}%  seeds={vals.tolist()}{flag}{warn}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("method,seed42,seed123,seed456,mean,std\n")
        for method, (mean, std, vals) in summary.items():
            seeds_str = ",".join(f"{v:.4f}" for v in vals)
            f.write(f"{method},{seeds_str},{mean:.4f},{std:.4f}\n")
    print(f"\n  CSV: {csv_path}")

    # Significance
    sig_lines = run_significance_test(summary)
    sig_path  = os.path.join(RESULTS_DIR, "significance.txt")
    with open(sig_path, "w") as f: f.write(sig_lines)
    print(f"  Significance: {sig_path}")

    # LaTeX
    latex = build_latex(summary)
    tex_path = os.path.join(RESULTS_DIR, "table_multiseed.tex")
    with open(tex_path, "w") as f: f.write(latex)
    print(f"  LaTeX: {tex_path}")
    print(f"\n{'='*60}\nLaTeX table:\n{'='*60}")
    print(latex)

    return summary

def run_significance_test(summary):
    lines = []
    lines.append("SIGNIFICANCE TEST — ContinualTTA vs each baseline")
    lines.append("Paired t-test, n=3 seeds, S1-S5 truly continual\n")

    if "ContinualTTA" not in summary:
        lines.append("ContinualTTA not available.")
        return "\n".join(lines)

    ours = np.array(summary["ContinualTTA"][2])
    lines.append(f"ContinualTTA: {ours.tolist()} (mean={ours.mean():.2f}, std={ours.std(ddof=1):.2f})\n")

    try:
        from scipy import stats
        have_scipy = True
    except ImportError:
        have_scipy = False
        lines.append("[scipy not available — t-stat only, install with: pip install scipy]\n")

    for method, (mean, std, vals) in summary.items():
        if method == "ContinualTTA": continue
        other = np.array(vals)
        diff  = ours - other
        n     = len(diff)
        if have_scipy:
            t_stat, p_val = stats.ttest_rel(ours, other)
            sig = "p < 0.05 ✓ SIGNIFICANT" if p_val < 0.05 else "p >= 0.05"
            line = (f"  ContinualTTA ({ours.mean():.2f}%) vs "
                    f"{method} ({other.mean():.2f}% ± {std:.2f}%): "
                    f"diff={diff.mean():+.2f}%, t={t_stat:.2f}, {sig}")
        else:
            mean_diff = diff.mean()
            se = diff.std(ddof=1)/math.sqrt(n) if n > 1 else 1e-9
            t_stat = mean_diff/se if se > 0 else float('inf')
            line = (f"  ContinualTTA ({ours.mean():.2f}%) vs "
                    f"{method} ({other.mean():.2f}% ± {std:.2f}%): "
                    f"diff={diff.mean():+.2f}%, t={t_stat:.2f} (install scipy for p-value)")
        lines.append(line)
        print(line)

    return "\n".join(lines)

def build_latex(summary):
    cite = {
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "SAR":          "SAR~\\cite{niu2023towards}",
        "ContinualTTA": "\\textbf{\\textsc{ContinualTTA} (Ours)}",
    }
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Multi-seed evaluation on CIFAR-10-C truly continual "
        r"(all severities S1--S5, 75 blocks, no reset). "
        r"Mean$\,\pm\,$std over 3 seeds (42, 123, 456) with independently "
        r"shuffled within-block batch composition.}")
    lines.append(r"\label{tab:multiseed}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Mean (\%) & Std (\%) \\")
    lines.append(r"\midrule")

    present = [m for m in METHODS if m in summary]
    best_mean = max(summary[m][0] for m in present)

    for method in present:
        mean, std, _ = summary[method]
        name = cite.get(method, method)
        mean_s = f"\\textbf{{{mean:.2f}}}" if abs(mean-best_mean)<0.01 else f"{mean:.2f}"
        std_s  = f"\\textbf{{{std:.2f}}}"  if abs(mean-best_mean)<0.01 else f"{std:.2f}"
        lines.append(f"{name} & {mean_s} & {std_s} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

# =============================================================================
# 9. MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Seed ALL-SEVERITY CIFAR-10-C Truly Continual",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Always run --verify first to confirm seeds genuinely differ.

Examples:
  python cifar10c_multiseed_allsev.py --verify
  python cifar10c_multiseed_allsev.py
  python cifar10c_multiseed_allsev.py --methods ContinualTTA --seeds 42
  python cifar10c_multiseed_allsev.py --skip_done
  python cifar10c_multiseed_allsev.py --table_only
        """)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--table_only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print("CIFAR-10-C Multi-Seed ALL-SEVERITY Truly Continual")
    print(f"{'='*60}")
    print(f"Device   : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"Severities : {SEVERITIES} (75 blocks total)")
    print(f"Methods  : {args.methods}")
    print(f"Seeds    : {args.seeds}")
    print(f"Results  : {RESULTS_DIR}\n")

    if args.verify:
        ok = verify_seeds()
        exit(0 if ok else 1)

    if args.table_only:
        all_seed_means = {}
        for method in args.methods:
            vals = [load_mean(method, s) for s in args.seeds]
            vals = [v for v in vals if v is not None]
            if vals:
                all_seed_means[method] = vals
                print(f"  Loaded {method}: {len(vals)}/{len(args.seeds)} seeds")
        if all_seed_means:
            generate_summary(all_seed_means)
        exit(0)

    if not args.skip_done:
        print("Pre-flight verification...")
        if not verify_seeds():
            print("ABORT — fix seed variation first.")
            exit(1)

    print("Loading source model...")
    source = load_model()
    print(f"  Parameters: {sum(p.numel() for p in source.parameters()):,}\n")

    # Sanity check (baseline on S1 gaussian_noise)
    print("Sanity check — Baseline on gaussian_noise S1...")
    _m = copy.deepcopy(source).eval()
    _l = get_loader("gaussian_noise", 1, SEEDS[0])
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in _l:
            x = x.to(DEVICE)
            correct += (_m(x).argmax(1) == y.to(DEVICE)).sum().item()
            total   += y.size(0)
    acc = 100.0 * correct / total
    del _m, _l; torch.cuda.empty_cache()
    print(f"  Baseline gaussian_noise S1: {acc:.1f}%  (expected ~70%+)")
    if acc < 50.0:
        print("  ERROR: too low — check MODEL_PATH and DATA_DIR"); exit(1)
    print("  Passed.\n")

    all_seed_means = {}

    for method in args.methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        method_means = []

        for seed in args.seeds:
            existing = load_mean(method, seed)
            if args.skip_done and existing is not None:
                print(f"  Skipping seed={seed} (saved: {existing:.2f}%)")
                method_means.append(existing)
                continue

            averaged, mean_overall = run_one_seed(method, seed, source)
            save_csv(method, seed, averaged, mean_overall)
            method_means.append(mean_overall)
            torch.cuda.empty_cache()

        all_seed_means[method] = method_means
        vals = np.array(method_means)
        print(f"\n  {method} summary: "
              f"{vals.mean():.2f}% ± {vals.std(ddof=1):.2f}%  seeds={method_means}")

    generate_summary(all_seed_means)

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"Results: {RESULTS_DIR}/")

if __name__ == "__main__":
    freeze_support()
    main()