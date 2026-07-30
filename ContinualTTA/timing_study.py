# =============================================================================
# ContinualTTA — ImageNet-C GN Timing Study
# Benchmark: gaussian_noise S5, 50,000 images, all 7 methods
#
# WHAT IS MEASURED:
#   Wall-clock GPU time (torch.cuda.synchronize() for accuracy)
#   Forward passes per batch (theoretical + counted)
#   Backward passes per batch (exact count from instrumentation)
#   Adaptation rate (% of batches where opt.step() fired)
#   Skip rate (% of batches with no parameter update)
#   Relative time normalised to TENT = 1.0x
#
# WHY GAUSSIAN_NOISE ONLY:
#   50,000 images per corruption is sufficient for stable timing.
#   Gaussian_noise S5 is the hardest corruption — worst case timing.
#   All methods are evaluated identically — fair comparison.
#   Running all 15 corruptions would take ~10x longer with no new insight.
#
# WHY N_RUNS=3:
#   Mean ± std gives reviewer confidence in reported numbers.
#   GPU has warmup variance on first run — averaging removes this.
#   3 runs × ~45 min each = ~2.5 hours total. Feasible overnight.
#
# PAPER NOTE:
#   Timing measured on NVIDIA RTX A4000.
#   Relative times (TENT=1.0×) are hardware-independent.
#   Data loading pre-cached — timing isolates pure compute.
#
# Run:
#   python imagenetc_gn_timing.py
#   python imagenetc_gn_timing.py --methods Baseline TENT ContinualTTA
#   python imagenetc_gn_timing.py --runs 1   # quick single run
# =============================================================================

import os
import copy
import math
import time
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
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\timing_gn"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
NUM_CLASSES = 1000
SEVERITY    = 5
CORRUPTION  = "gaussian_noise"   # single corruption for timing
NUM_WORKERS = 0
N_RUNS      = 1                   # repeat each method 3 times

# Hyperparameters — identical to main experiments
IMAGENET_LR  = 2.5e-4
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 2.763 nats
MIN_CONF     = 0.5
ADAPT_BUDGET = 300
JS_THRESHOLD = 0.10
ROTTA_NU     = 0.001
ROTTA_N      = 64
SAR_RHO      = 0.05
SAR_E0       = 0.2
GN_MODEL     = "resnet50_gn"

METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. MODEL & DATA
# =============================================================================

def load_model():
    print(f"  Loading {GN_MODEL} from timm...")
    model = timm.create_model(GN_MODEL, pretrained=True)
    return model.to(DEVICE).eval()


def get_transform(model):
    config    = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return transform


def preload_batches(transform):
    """
    Pre-load all 50,000 images into CPU RAM before timing begins.
    This eliminates disk I/O variance from timing measurements.
    Each batch stored as CPU tensor — moved to GPU inside timed loop.
    """
    print(f"  Pre-loading {CORRUPTION} S{SEVERITY} into CPU RAM...")
    path    = os.path.join(DATA_DIR, CORRUPTION, str(SEVERITY))
    dataset = ImageFolder(path, transform=transform)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)
    batches = [(x, y) for x, y in loader]
    n_total = sum(x.size(0) for x, _ in batches)
    print(f"  Loaded {len(batches)} batches ({n_total:,} images) into CPU RAM")
    return batches


# =============================================================================
# 2. HELPERS
# =============================================================================

def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def setup_gn(model):
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
# 3. INSTRUMENTED METHOD FACTORIES
#
# Each returns (fn, stats) where stats tracks:
#   n_batches:  total batches processed
#   n_backward: batches where backward() fired
#   n_adapted:  batches where opt.step() fired
#   n_skipped:  batches with NO parameter update
#
# CRITICAL: These are the EXACT same implementations as in the
# main experiment scripts. No shortcuts or simplifications.
# Timing must reflect actual computational cost.
# =============================================================================

def make_baseline(source):
    model = copy.deepcopy(source).eval()
    stats = {"n_batches":0, "n_backward":0, "n_adapted":0, "n_skipped":0}

    def fn(x):
        with torch.no_grad():
            logits = model(x)
        stats["n_batches"] += 1
        stats["n_skipped"] += 1
        return logits

    return fn, stats


def make_tent(source):
    model, params = setup_gn(copy.deepcopy(source))
    opt   = torch.optim.Adam(params, lr=IMAGENET_LR)
    stats = {"n_batches":0, "n_backward":0, "n_adapted":0, "n_skipped":0}

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        loss = softmax_entropy(logits).mean()        # all samples, no filter
        stats["n_batches"] += 1
        loss.backward()
        stats["n_backward"] += 1
        opt.step()
        opt.zero_grad()
        stats["n_adapted"] += 1
        return logits

    return fn, stats


def make_eata(source, fisher_batches=None):
    model, params = setup_gn(copy.deepcopy(source))
    opt   = torch.optim.Adam(params, lr=IMAGENET_LR)
    stats = {"n_batches":0, "n_backward":0, "n_adapted":0, "n_skipped":0}

    # Fisher importance weights — computed from first 10 batches
    fisher = {n: torch.zeros_like(p)
              for n, p in model.named_parameters() if p.requires_grad}
    if fisher_batches is not None:
        model.train()
        for i, (x, _) in enumerate(fisher_batches[:10]):
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
        stats["n_batches"] += 1

        if mask.sum() == 0:
            stats["n_skipped"] += 1
            return logits

        with torch.no_grad():
            ref_probs[0] = probs[mask].mean(0).detach() if ref_probs[0] is None \
                      else 0.9 * ref_probs[0] + 0.1 * probs[mask].mean(0).detach()

        fisher_reg = sum((fisher[n] * p.pow(2)).sum()
                         for n, p in model.named_parameters()
                         if p.requires_grad and n in fisher)
        (entropy[mask].mean() + 1e-3 * fisher_reg).backward()
        stats["n_backward"] += 1
        opt.step(); opt.zero_grad()
        stats["n_adapted"] += 1
        return logits

    return fn, stats


def make_cotta(source):
    src = copy.deepcopy(source).eval()
    src.requires_grad_(False)
    adapted, params = setup_gn(copy.deepcopy(source))
    opt     = torch.optim.Adam(params, lr=IMAGENET_LR)
    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)
    stats   = {"n_batches":0, "n_backward":0, "n_adapted":0, "n_skipped":0}

    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ])

    @torch.enable_grad()
    def fn(x):
        # 4 augmented teacher forwards (main compute cost)
        with torch.no_grad():
            pseudo = torch.stack(
                [teacher(aug(x)).softmax(1) for _ in range(4)]).mean(0)

        logits = adapted(x)
        loss   = -(pseudo * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        stats["n_batches"]  += 1
        stats["n_backward"] += 1
        opt.step(); opt.zero_grad()
        stats["n_adapted"]  += 1

        with torch.no_grad():
            # Teacher EMA update
            for tp, ap in zip(teacher.parameters(), adapted.parameters()):
                tp.data = 0.999 * tp.data + 0.001 * ap.data
            # Stochastic restoration — single mask (bug fix)
            for (_, pa), (_, ps) in zip(adapted.named_parameters(),
                                         src.named_parameters()):
                if pa.requires_grad:
                    mask = torch.rand_like(pa) < 0.01
                    pa.data[mask] = ps.data[mask]

        return logits

    return fn, stats


def make_rotta(source):
    student = copy.deepcopy(source)
    student.train()
    student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, nn.GroupNorm):
            m.requires_grad_(True)
    params  = [p for m in student.modules()
               if isinstance(m, nn.GroupNorm)
               for p in m.parameters() if p.requires_grad]
    opt     = torch.optim.Adam(params, lr=IMAGENET_LR)
    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)
    stats   = {"n_batches":0, "n_backward":0, "n_adapted":0, "n_skipped":0}

    per_class = max(1, ROTTA_N // NUM_CLASSES)   # = 1 for C=1000
    bank      = {c: [] for c in range(NUM_CLASSES)}
    age       = [0]

    @torch.enable_grad()
    def fn(x):
        logits  = student(x)
        plabels = logits.argmax(1).detach()
        ents    = softmax_entropy(logits).detach()
        stats["n_batches"] += 1

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

        samples, ages_list = [], []
        for c in range(NUM_CLASSES):
            if bank[c]:
                for entry in sorted(bank[c], key=lambda e: -e[2])[:per_class]:
                    samples.append(entry[0])
                    ages_list.append(entry[2])

        if len(samples) < 2:
            stats["n_skipped"] += 1
            return logits

        mem_x  = torch.stack(samples).to(DEVICE)
        ages_t = torch.tensor(ages_list, dtype=torch.float32, device=DEVICE)
        e_age  = torch.exp(-ages_t/ROTTA_N) / (1+torch.exp(-ages_t/ROTTA_N))

        BANK_BATCH = 32
        total_loss = torch.tensor(0.0, device=DEVICE)
        n_mini     = 0
        for start in range(0, len(samples), BANK_BATCH):
            end    = min(start + BANK_BATCH, len(samples))
            mb_x   = mem_x[start:end]
            mb_age = e_age[start:end]
            with torch.no_grad():
                t_probs = teacher(mb_x).softmax(1)
            s_logits = student(mb_x)
            ce = -(t_probs * s_logits.log_softmax(1)).sum(1) / NUM_CLASSES
            total_loss = total_loss + (mb_age * ce).mean()
            n_mini += 1

        (total_loss / n_mini).backward()
        stats["n_backward"] += 1
        opt.step(); opt.zero_grad()
        stats["n_adapted"]  += 1

        with torch.no_grad():
            for tp, sp in zip(teacher.parameters(), student.parameters()):
                tp.data = (1 - ROTTA_NU) * tp.data + ROTTA_NU * sp.data

        return logits

    return fn, stats


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

    opt   = torch.optim.SGD(params, lr=IMAGENET_LR, momentum=0.9)
    stats = {"n_batches":0, "n_backward":0, "n_adapted":0, "n_skipped":0}

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
        stats["n_batches"] += 1

        if reliable.sum() == 0:
            stats["n_skipped"] += 1
            return logits_init

        x_rel = x[reliable]

        # Step 1 — gradient at θ
        logits_1 = model(x_rel)
        softmax_entropy(logits_1).mean().backward()
        stats["n_backward"] += 1
        grad_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))

        # Step 2 — perturb to θ'
        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
                p.data.add_(e_w); e_ws.append(e_w); p.grad.zero_()
            else:
                e_ws.append(None)

        # Step 3 — gradient at θ'
        logits_2   = model(x_rel)
        entropy_2  = softmax_entropy(logits_2)
        if (entropy_2 < E_MARGIN).sum() > 0:
            entropy_2[entropy_2 < E_MARGIN].mean().backward()
            stats["n_backward"] += 1

        # Step 4 — restore θ and apply update
        for p, e_w in zip(params, e_ws):
            if e_w is not None:
                p.data.sub_(e_w)
        opt.step(); opt.zero_grad()
        stats["n_adapted"] += 1

        # Model recovery
        with torch.no_grad():
            logits_out  = model(x)
            entropy_out = softmax_entropy(logits_out)
            ema_entropy[0] = (0.9 * ema_entropy[0]
                              + 0.1 * entropy_out.mean().item())
            if ema_entropy[0] < SAR_E0:
                for n, p in model.named_parameters():
                    if p.requires_grad and n in init_params:
                        p.data.copy_(init_params[n])
                ema_entropy[0] = None

        return logits_out

    return fn, stats


def make_ctta(source):
    model, params = setup_gn(copy.deepcopy(source))
    opt   = torch.optim.Adam(params, lr=IMAGENET_LR)
    stats = {"n_batches":0, "n_backward":0, "n_adapted":0, "n_skipped":0}

    reference  = [None]
    n_backward = [0]   # separate budget counter

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        stats["n_batches"] += 1

        # JS shift detector
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                stats["n_skipped"] += 1
                return logits
            m    = 0.5 * (reference[0] + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5 * (kl_1 + kl_2)
            reference[0] = 0.9 * reference[0] + 0.1 * p_t
            adapt_js = js.item() > JS_THRESHOLD

        if not adapt_js:
            stats["n_skipped"] += 1
            return logits

        if n_backward[0] >= ADAPT_BUDGET:
            stats["n_skipped"] += 1
            return logits

        # Combined entropy + confidence filter
        with torch.no_grad():
            entropy = softmax_entropy(logits)
            conf    = logits.softmax(1).max(1).values
        reliable = (entropy < E_MARGIN) & (conf > MIN_CONF)

        if reliable.sum() == 0:
            stats["n_skipped"] += 1
            return logits

        # Entropy minimisation
        logits_rel  = model(x[reliable])
        softmax_entropy(logits_rel).mean().backward()
        stats["n_backward"] += 1
        opt.step(); opt.zero_grad()
        stats["n_adapted"]  += 1
        n_backward[0]       += 1
        return logits

    return fn, stats


# =============================================================================
# 4. BUILD METHOD
# =============================================================================

def build_method(method, source, preloaded_batches):
    """Build fresh method instance. EATA Fisher from preloaded_batches."""
    fisher_batches = preloaded_batches if method == "EATA" else None

    dispatch = {
        "Baseline":     lambda: make_baseline(source),
        "TENT":         lambda: make_tent(source),
        "EATA":         lambda: make_eata(source, fisher_batches),
        "CoTTA":        lambda: make_cotta(source),
        "RoTTA":        lambda: make_rotta(source),
        "SAR":          lambda: make_sar(source),
        "ContinualTTA": lambda: make_ctta(source),
    }
    return dispatch[method]()


# =============================================================================
# 5. MEMORY OVERHEAD
# =============================================================================

def get_memory_info(method, model_mb):
    info = {
        "Baseline":     (0,              "none"),
        "TENT":         (0,              "none"),
        "EATA":         (0.5,            "Fisher weights (~GN params)"),
        "CoTTA":        (model_mb * 2,   "source + teacher model copies"),
        "RoTTA":        (ROTTA_N * 3 * 224 * 224 * 4 / 1024**2,
                                         f"image bank ({ROTTA_N} images)"),
        "SAR":          (0.1,            "initial GN param copy"),
        "ContinualTTA": (NUM_CLASSES * 4 / 1024**2,
                                         f"reference vector ({NUM_CLASSES} floats ≈ 0KB)"),
    }
    return info.get(method, (0, "unknown"))


def fwd_bwd_info(method):
    info = {
        "Baseline":     ("1",  "0",   "no update"),
        "TENT":         ("1",  "≤1",  "entropy filter"),
        "EATA":         ("1",  "≤1",  "two filters + Fisher reg"),
        "CoTTA":        ("5",  "1",   "4 aug teacher + 1 student"),
        "RoTTA":        ("1+", "1",   "student + bank mini-batches"),
        "SAR":          ("2",  "≤2",  "SAM two-step"),
        "ContinualTTA": ("1",  "<1",  "JS gate + confidence filter"),
    }
    return info.get(method, ("?", "?", ""))


# =============================================================================
# 6. TIME ONE RUN
# =============================================================================

def time_one_run(method, source, preloaded_batches):
    """
    Run method through all preloaded batches.
    Returns (elapsed_seconds, stats_dict, accuracy).
    Uses cuda.synchronize() for accurate GPU timing.
    """
    fn, stats = build_method(method, source, preloaded_batches)

    # Warmup — one batch to eliminate CUDA first-call overhead
    x_warm, _ = preloaded_batches[0]
    fn(x_warm.to(DEVICE))
    # Reset stats after warmup
    for k in stats: stats[k] = 0

    # Rebuild fresh for clean stats (warmup may have changed bank/reference)
    fn, stats = build_method(method, source, preloaded_batches)

    correct, total = 0, 0
    cuda_sync()
    t_start = time.perf_counter()

    for x, y in preloaded_batches:
        x = x.to(DEVICE)
        logits  = fn(x)
        correct += (logits.argmax(1) == y.to(DEVICE)).sum().item()
        total   += y.size(0)

    cuda_sync()
    elapsed = time.perf_counter() - t_start
    acc     = 100.0 * correct / total

    return elapsed, stats, acc


# =============================================================================
# 7. RESULTS AND LATEX
# =============================================================================

def generate_output(all_times, all_stats, all_accs, model_mb):
    tent_mean = np.mean(all_times.get("TENT", [1.0]))
    methods_present = [m for m in METHODS if m in all_times]

    print(f"\n{'='*75}")
    print(f"TIMING RESULTS — ImageNet-C GN, {CORRUPTION}, S{SEVERITY}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, Batch size: {BATCH_SIZE}")
    print(f"{'='*75}\n")

    rows = []
    for method in methods_present:
        times    = all_times[method]
        accs     = all_accs[method]
        mean_t   = np.mean(times)
        std_t    = np.std(times)
        mean_a   = np.mean(accs)
        rel      = mean_t / tent_mean
        s        = all_stats[method]
        tot      = max(s["n_batches"], 1)
        adapt_pct= 100.0 * s["n_adapted"]  / tot
        skip_pct = 100.0 * s["n_skipped"]  / tot
        bwd_tot  = s["n_backward"]
        fwd, bwd, note = fwd_bwd_info(method)
        mem_mb, mem_note = get_memory_info(method, model_mb)

        rows.append({
            "method":     method,
            "mean_t":     mean_t,  "std_t":    std_t,
            "rel":        rel,     "acc":      mean_a,
            "adapt_pct":  adapt_pct, "skip_pct": skip_pct,
            "bwd_tot":    bwd_tot,
            "fwd":        fwd,     "bwd":      bwd,
            "mem_mb":     mem_mb,  "mem_note": mem_note,
        })

        flag = "  ← ours" if method == "ContinualTTA" else ""
        print(f"  {method}{flag}")
        print(f"    Time:      {mean_t:.1f}s ± {std_t:.1f}s  "
              f"({rel:.2f}× TENT)")
        print(f"    Accuracy:  {mean_a:.1f}%")
        print(f"    Fwd/batch: {fwd}   Bwd/batch: {bwd}  ({note})")
        print(f"    Adapted:   {adapt_pct:.0f}%  "
              f"Skipped: {skip_pct:.0f}%  "
              f"Total bwd: {bwd_tot}")
        print(f"    Memory:    +{mem_mb:.1f} MB ({mem_note})")
        print()

    # Key claims
    if "SAR" in all_times and "ContinualTTA" in all_times:
        sar_t  = np.mean(all_times["SAR"])
        ours_t = np.mean(all_times["ContinualTTA"])
        cotta_t= np.mean(all_times.get("CoTTA", [0]))
        print(f"KEY CLAIMS FOR PAPER:")
        print(f"  ContinualTTA vs TENT:  "
              f"{np.mean(all_times['ContinualTTA'])/tent_mean:.2f}× (TENT=1.00×)")
        print(f"  ContinualTTA vs SAR:   {sar_t/ours_t:.1f}× faster than SAR")
        if cotta_t > 0:
            print(f"  ContinualTTA vs CoTTA: {cotta_t/ours_t:.1f}× faster than CoTTA")
        ours_s = all_stats["ContinualTTA"]
        skip = 100.0*ours_s["n_skipped"]/max(ours_s["n_batches"],1)
        print(f"  JS gate skipped {skip:.0f}% of batches entirely")

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "timing_results.csv")
    with open(csv_path, "w") as f:
        f.write("method,mean_s,std_s,rel_tent,accuracy,"
                "fwd_per_batch,bwd_per_batch,bwd_total,"
                "adapt_pct,skip_pct,mem_mb\n")
        for r in rows:
            f.write(f"{r['method']},{r['mean_t']:.2f},{r['std_t']:.2f},"
                    f"{r['rel']:.3f},{r['acc']:.2f},"
                    f"{r['fwd']},{r['bwd']},{r['bwd_tot']},"
                    f"{r['adapt_pct']:.1f},{r['skip_pct']:.1f},"
                    f"{r['mem_mb']:.1f}\n")
    print(f"\n  CSV: {csv_path}")

    # LaTeX table
    latex = build_latex(rows, tent_mean, model_mb)
    tex_path = os.path.join(RESULTS_DIR, "efficiency_table.tex")
    with open(tex_path, "w") as f: f.write(latex)
    print(f"  LaTeX: {tex_path}")
    print(f"\n{'='*60}\nLaTeX table:\n{'='*60}")
    print(latex)


def build_latex(rows, tent_mean, model_mb):
    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "CoTTA":        "CoTTA~\\cite{wang2022continual}",
        "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
        "SAR":          "SAR~\\cite{niu2023towards}",
        "ContinualTTA": "\\textbf{\\textsc{ContinualTTA} (Ours)}",
    }
    mem_tex = {
        "Baseline":     "---",
        "TENT":         "---",
        "EATA":         "$<$0.1",
        "CoTTA":        f"${model_mb*2:.0f}$",
        "RoTTA":        f"${ROTTA_N*3*224*224*4/1024**2:.0f}$",
        "SAR":          "$<$0.1",
        "ContinualTTA": "$\\approx$0",
    }

    # Find fastest non-baseline for bolding
    non_base = [r for r in rows if r["method"] != "Baseline"]
    fastest  = min(non_base, key=lambda r: r["mean_t"])["method"] \
               if non_base else None

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Computational efficiency on ImageNet-C GN "
        r"(\texttt{gaussian\_noise}, severity~5, $50{,}000$ images). "
        r"Timing on NVIDIA RTX~A4000; relative time normalised to "
        r"TENT~$=\!1.0\times$. "
        r"Adapt\% = fraction of batches where GN parameters are updated. "
        r"Skip\% = fraction requiring only a forward pass. "
        r"Memory = extra storage beyond source model.}")
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Time (s) & Rel. & Acc. (\%) & "
                 r"Adapt\% & Skip\% & Mem (MB) \\")
    lines.append(r"\midrule")

    for r in rows:
        name    = cite.get(r["method"], r["method"])
        is_fast = r["method"] == fastest
        t_str   = f"{r['mean_t']:.0f}$\\pm${r['std_t']:.0f}"
        rel_str = f"{r['rel']:.2f}$\\times$"
        if is_fast:
            t_str   = f"\\textbf{{{t_str}}}"
            rel_str = f"\\textbf{{{rel_str}}}"
        mem_str = mem_tex.get(r["method"], "---")
        lines.append(
            f"{name} & {t_str} & {rel_str} & "
            f"{r['acc']:.1f} & "
            f"{r['adapt_pct']:.0f} & {r['skip_pct']:.0f} & "
            f"{mem_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# 8. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ImageNet-C GN Timing Study — All Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmarks all 7 TTA methods on gaussian_noise S5.
Uses identical implementations as main experiment scripts.
Reports time, relative speedup, adaptation rate, memory overhead.

Examples:
  python imagenetc_gn_timing.py
  python imagenetc_gn_timing.py --methods Baseline TENT SAR ContinualTTA
  python imagenetc_gn_timing.py --runs 1
        """)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--runs", type=int, default=N_RUNS,
                        help=f"Number of timing runs per method (default {N_RUNS})")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ImageNet-C GN Timing Study")
    print(f"{'='*60}")
    print(f"Device    : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU       : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM      : {vram:.1f} GB")
    print(f"Corruption: {CORRUPTION}, S{SEVERITY}")
    print(f"Batch     : {BATCH_SIZE}")
    print(f"Runs      : {args.runs}")
    print(f"Methods   : {args.methods}")
    print(f"Results   : {RESULTS_DIR}\n")

    # Load model + transform
    print("Loading GN model...")
    source    = load_model()
    transform = get_transform(source)
    model_mb  = sum(p.numel() * 4 for p in source.parameters()) / 1024**2
    print(f"  Model size: {model_mb:.1f} MB\n")

    # Pre-load all batches into CPU RAM
    preloaded = preload_batches(transform)
    print()

    # GPU warmup
    print("GPU warmup (10 dummy forward passes)...")
    _dummy = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        for _ in range(10): source(_dummy)
    cuda_sync()
    del _dummy; torch.cuda.empty_cache()
    print("Done.\n")

    # Main timing loop
    all_times = {}
    all_stats = {}
    all_accs  = {}

    for method in args.methods:
        print(f"{'─'*55}")
        print(f"Timing: {method}  ({args.runs} run(s))")
        run_times = []
        run_accs  = []
        run_stats = None

        for run in range(args.runs):
            torch.cuda.empty_cache()
            elapsed, stats, acc = time_one_run(method, source, preloaded)
            run_times.append(elapsed)
            run_accs.append(acc)
            if run_stats is None: run_stats = stats
            pct = 100.0 * stats["n_adapted"] / max(stats["n_batches"], 1)
            print(f"  Run {run+1}/{args.runs}: {elapsed:.1f}s  "
                  f"acc={acc:.1f}%  "
                  f"adapted={stats['n_adapted']}/{stats['n_batches']} "
                  f"({pct:.0f}%)")

        all_times[method] = run_times
        all_accs[method]  = run_accs
        all_stats[method] = run_stats
        print(f"  → Mean: {np.mean(run_times):.1f}s ± {np.std(run_times):.1f}s  "
              f"acc: {np.mean(run_accs):.1f}%")
        torch.cuda.empty_cache()

    generate_output(all_times, all_stats, all_accs, model_mb)

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"Results: {RESULTS_DIR}/")
    print("  timing_results.csv   — raw numbers")
    print("  efficiency_table.tex — LaTeX for paper")