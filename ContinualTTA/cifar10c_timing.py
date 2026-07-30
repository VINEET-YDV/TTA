# =============================================================================
# ContinualTTA — Efficiency Measurement Script
#
# Measures three efficiency metrics for each TTA method:
#   1. Runtime (ms per batch) — forward + adaptation overhead
#   2. Extra GPU memory (MB above baseline forward pass)
#   3. Percentage of batches adapted (from gate analysis)
#
# EXPECTED RESULTS:
#   Baseline:     fastest (no backward pass)
#   TENT:         moderate (backward every batch)
#   SAR:          slowest (2 forward passes per batch)
#   ContinualTTA: near-baseline (backward only 16.1% of batches)
#
# Run:
#   python cifar10c_timing.py
#   python cifar10c_timing.py --n_batches 200
#   python cifar10c_timing.py --methods Baseline TENT ContinualTTA
#   python cifar10c_timing.py --table_only
# =============================================================================

import os
import time
import copy
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from multiprocessing import freeze_support
import torchvision.transforms as transforms

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\efficiency"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
N_WARMUP    = 10
N_BATCHES   = 100
NUM_CLASSES = 10

LR           = 1e-3
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)
JS_THRESHOLD = 0.04
SAR_RHO      = 0.05
SAR_E0       = 0.2

# Gate firing rate from drift analysis experiment
METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]

GATE_RATES = {
    "Baseline":     0.0,
    "TENT":         100.0,
    "EATA":         100.0,
    "CoTTA":        100.0,   # adapts every batch via teacher
    "RoTTA":        100.0,   # adapts every batch via bank
    "SAR":          100.0,
    "ContinualTTA": 16.1,
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. MODEL
# =============================================================================

def load_model():
    model    = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


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


def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)


# =============================================================================
# 2. METHOD FACTORIES
# =============================================================================

def make_baseline(source):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad(): return model(x)
    return fn


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
    return fn


def make_eata(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    ref = [None]
    @torch.enable_grad()
    def fn(x):
        logits  = model(x)
        entropy = softmax_entropy(logits)
        probs   = logits.softmax(1)
        mask_e  = entropy < E_MARGIN
        if ref[0] is not None:
            cos    = F.cosine_similarity(
                ref[0].unsqueeze(0).expand(probs.size(0),-1), probs, dim=1)
            mask_d = cos < 0.95
        else:
            mask_d = torch.ones(probs.size(0), dtype=torch.bool, device=DEVICE)
        mask = mask_e & mask_d
        if mask.sum() == 0: return logits
        ref[0] = (probs[mask].mean(0).detach() if ref[0] is None
                  else 0.9*ref[0] + 0.1*probs[mask].mean(0).detach())
        entropy[mask].mean().backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn

def make_cotta(source):
    src = copy.deepcopy(source).eval()
    src.requires_grad_(False)
    adapted, params = setup_bn(copy.deepcopy(source))
    opt     = torch.optim.Adam(params, lr=LR)
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
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            for tp, ap in zip(teacher.parameters(), adapted.parameters()):
                tp.data = 0.999*tp.data + 0.001*ap.data
        return logits
    return fn


def make_rotta(source):
    student = copy.deepcopy(source)
    student.train()
    student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True
            m.momentum = 0.05
    params  = [p for m in student.modules()
               if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
               for p in m.parameters() if p.requires_grad]
    opt     = torch.optim.Adam(params, lr=LR)
    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)
    # Memory bank: 64 slots across 10 classes = 6 per class
    bank    = {c: [] for c in range(NUM_CLASSES)}

    def fn(x):
        with torch.no_grad():
            logits  = student(x)
            plabels = logits.argmax(1)
            ents    = softmax_entropy(logits)
            for i, (c, e) in enumerate(
                    zip(plabels.tolist(), ents.tolist())):
                entry = (x[i].detach().cpu(), e)
                if len(bank[c]) < 6:
                    bank[c].append(entry)
                else:
                    worst = max(range(len(bank[c])),
                                key=lambda j: bank[c][j][1])
                    if e < bank[c][worst][1]:
                        bank[c][worst] = entry
        samples = []
        for c in range(NUM_CLASSES):
            samples.extend(e[0] for e in bank[c])
        if len(samples) >= 2:
            mem_x    = torch.stack(samples).to(DEVICE)
            with torch.no_grad():
                t_probs  = teacher(mem_x).softmax(1)
            s_logits = student(mem_x)
            loss = -(t_probs*s_logits.log_softmax(1)).sum(1).mean()
            loss.backward()
            opt.step(); opt.zero_grad()
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(),
                                  student.parameters()):
                    tp.data = 0.999*tp.data + 0.001*sp.data
        return logits
    return fn

def make_sar(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
    ip  = {n: p.data.clone() for n,p in model.named_parameters()
           if p.requires_grad}
    ema = [None]
    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            li = model(x)
            ei = softmax_entropy(li)
        if ema[0] is None: ema[0] = E_MARGIN
        thr = min(E_MARGIN, ema[0]+0.4*math.log(NUM_CLASSES))
        rel = ei < thr
        if rel.sum() == 0: return li
        l1 = model(x[rel])
        softmax_entropy(l1).mean().backward()
        gn = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))
        ews = []
        for p in params:
            if p.grad is not None:
                ew = p.grad*SAR_RHO/(gn+1e-12)
                p.data.add_(ew); ews.append(ew); p.grad.zero_()
            else: ews.append(None)
        l2 = model(x[rel])
        e2 = softmax_entropy(l2)
        if (e2<E_MARGIN).sum()>0: e2[e2<E_MARGIN].mean().backward()
        for p,ew in zip(params,ews):
            if ew is not None: p.data.sub_(ew)
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            lo = model(x)
            eo = softmax_entropy(lo)
            ema[0] = 0.9*ema[0]+0.1*eo.mean().item()
            if ema[0]<SAR_E0:
                for n,p in model.named_parameters():
                    if p.requires_grad and n in ip: p.data.copy_(ip[n])
                ema[0] = None
        return lo
    return fn


def make_ctta(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    ref = [None]
    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if ref[0] is None:
                ref[0] = p_t.clone(); return logits
            m   = 0.5*(ref[0]+p_t)
            k1  = F.kl_div(m.log().unsqueeze(0),
                            ref[0].unsqueeze(0), reduction="batchmean")
            k2  = F.kl_div(m.log().unsqueeze(0),
                            p_t.unsqueeze(0),   reduction="batchmean")
            js  = 0.5*(k1+k2)
            ref[0] = 0.9*ref[0]+0.1*p_t
            adapt = js.item() > JS_THRESHOLD
        if not adapt: return logits
        ent = softmax_entropy(logits)
        rel = ent < E_MARGIN
        if rel.sum()==0: return logits
        ent[rel].mean().backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn

def build_method(method, source):
    return {
        "Baseline":     make_baseline,
        "TENT":         make_tent,
        "EATA":         make_eata,
        "CoTTA":        make_cotta,
        "RoTTA":        make_rotta,
        "SAR":          make_sar,
        "ContinualTTA": make_ctta,
    }[method](source)


# =============================================================================
# 3. TIMING — CUDA events for accurate GPU measurement
# =============================================================================

def measure_runtime(fn, n_warmup=N_WARMUP, n_batches=N_BATCHES):
    x = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)
    for _ in range(n_warmup): fn(x)
    if DEVICE == "cuda": torch.cuda.synchronize()

    times = []
    for _ in range(n_batches):
        if DEVICE == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(x); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter(); fn(x)
            times.append((time.perf_counter()-t0)*1000)
    return np.mean(times), np.std(times)


# =============================================================================
# 4. MEMORY
# =============================================================================

def measure_memory(fn, base_mb=0.0):
    if DEVICE != "cuda": return 0.0
    x = torch.randn(BATCH_SIZE, 3, 224, 224, device=DEVICE)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    fn(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    return max(0.0, peak - base_mb)


# =============================================================================
# 5. RUN + SAVE + REPORT
# =============================================================================

def run_all(source, methods, n_batches):
    total  = sum(p.numel() for p in source.parameters())
    bn_aff = sum(p.numel() for m in source.modules()
                 if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d))
                 for p in m.parameters())
    print(f"  Total params: {total:,}  |  BN affine: {bn_aff:,} "
          f"({100*bn_aff/total:.2f}%)\n")

    # Get baseline memory as reference
    bfn      = make_baseline(source)
    base_mb  = measure_memory(bfn)
    del bfn; torch.cuda.empty_cache()

    results = {}
    for method in methods:
        print(f"  [{method}]")
        fn = build_method(method, source)

        ms_mean, ms_std = measure_runtime(fn, n_batches=n_batches)
        extra_mb        = measure_memory(fn, base_mb)
        gate            = GATE_RATES.get(method, 100.0)

        print(f"    Runtime : {ms_mean:.1f} ± {ms_std:.1f} ms/batch")
        print(f"    Memory  : +{extra_mb:.1f} MB above baseline")
        print(f"    Adapted : {gate:.1f}% of batches")

        results[method] = {"ms_mean":ms_mean, "ms_std":ms_std,
                           "extra_mb":extra_mb, "gate_pct":gate}
        del fn; torch.cuda.empty_cache()

    return results


def save_and_report(results, methods):
    # CSV
    csv = os.path.join(RESULTS_DIR, "efficiency_results.csv")
    with open(csv, "w") as f:
        f.write("method,ms_mean,ms_std,extra_mb,gate_pct\n")
        for m in methods:
            if m not in results: continue
            r = results[m]
            f.write(f"{m},{r['ms_mean']:.2f},{r['ms_std']:.2f},"
                    f"{r['extra_mb']:.1f},{r['gate_pct']:.1f}\n")
    print(f"\n  CSV: {csv}")

    # Console table
    print(f"\n{'='*60}")
    print(f"{'Method':<18} {'ms/batch':>10} {'Extra MB':>10} {'Adapted':>10}")
    print("─"*52)
    base_ms = results.get("Baseline",{}).get("ms_mean",1.0)
    for m in methods:
        if m not in results: continue
        r = results[m]
        oh = f"  (+{r['ms_mean']-base_ms:.1f}ms)" if m!="Baseline" else ""
        print(f"  {m:<16} {r['ms_mean']:>7.1f} ms  "
              f"{r['extra_mb']:>7.1f} MB  {r['gate_pct']:>7.1f}%{oh}")

    # LaTeX
    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "SAR":          "SAR~\\cite{niu2023towards}",
        "ContinualTTA": "\\textbf{\\ours{} (ours)}",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Efficiency on CIFAR-10-C (ResNet-50 BN, "
        r"batch size~32, RTX~A4000). Extra memory above baseline "
        r"forward pass. Batches adapted from gate analysis.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & ms/batch & Extra Mem & Adapted \\",
        r"\midrule",
    ]
    best_ms = min(results[m]["ms_mean"] for m in methods if m in results)
    for m in methods:
        if m not in results: continue
        r    = results[m]
        ms_s = (f"\\textbf{{{r['ms_mean']:.1f}}}"
                if abs(r['ms_mean']-best_ms)<0.1 else f"{r['ms_mean']:.1f}")
        g_s  = (f"\\textbf{{{r['gate_pct']:.1f}\\%}}"
                if m=="ContinualTTA" else f"{r['gate_pct']:.1f}\\%")
        lines.append(f"{cite[m]} & {ms_s} & {r['extra_mb']:.0f}~MB & {g_s} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(lines)

    tex = os.path.join(RESULTS_DIR, "table_efficiency.tex")
    with open(tex, "w") as f: f.write(latex)
    print(f"\n  LaTeX: {tex}")
    print(f"\n{'='*55}\nLaTeX table:\n{'='*55}")
    print(latex)

    # Inline sentence
    if all(m in results for m in ["ContinualTTA","TENT","SAR"]):
        c = results["ContinualTTA"]
        t = results["TENT"]
        s = results["SAR"]
        text = (
            f"\\ours{{}} runs at ${c['ms_mean']:.1f}$~ms/batch "
            f"(RTX~A4000, batch~32) vs ${t['ms_mean']:.1f}$~ms for TENT "
            f"and ${s['ms_mean']:.1f}$~ms for SAR (which requires two "
            f"forward passes per batch). By adapting only "
            f"${GATE_RATES['ContinualTTA']}\\%$ of batches, \\ours{{}} "
            f"performs ${100/GATE_RATES['ContinualTTA']:.0f}\\times$ fewer "
            f"optimizer steps than TENT while requiring no memory bank, "
            f"teacher model, or data augmentation."
        )
        txt = os.path.join(RESULTS_DIR, "inline_text.txt")
        with open(txt,"w") as f: f.write(text)
        print(f"\nInline sentence:\n{text}\n  Saved: {txt}")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ContinualTTA Efficiency Measurement")
    parser.add_argument("--n_batches",  type=int,  default=N_BATCHES)
    parser.add_argument("--methods",    nargs="+", default=METHODS)
    parser.add_argument("--table_only", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print("ContinualTTA — Efficiency Measurement")
    print(f"{'='*60}")
    print(f"Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Timing     : {args.n_batches} batches (+ {N_WARMUP} warmup)")
    print(f"Methods    : {args.methods}\n")

    if args.table_only:
        csv = os.path.join(RESULTS_DIR, "efficiency_results.csv")
        if not os.path.isfile(csv):
            print("No results found — run without --table_only first.")
            return
        results = {}
        with open(csv) as f:
            next(f)
            for line in f:
                p = line.strip().split(",")
                results[p[0]] = {"ms_mean":float(p[1]),"ms_std":float(p[2]),
                                 "extra_mb":float(p[3]),"gate_pct":float(p[4])}
        save_and_report(results, args.methods)
        return

    print("Loading source model...")
    source = load_model()
    x = torch.randn(BATCH_SIZE,3,224,224,device=DEVICE)
    out = make_baseline(source)(x)
    print(f"  Sanity: output {out.shape}  ✓\n")
    del x; torch.cuda.empty_cache()

    results = run_all(source, args.methods, args.n_batches)
    save_and_report(results, args.methods)

    print(f"\n{'='*60}\nDONE — Results: {RESULTS_DIR}/\n{'='*60}")


if __name__ == "__main__":
    freeze_support()
    main()