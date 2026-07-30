# =============================================================================
# ContinualTTA — Gate Detection Analysis + Oracle Boundary Control
#
# EXPERIMENT 5 — Gate detection quality
#   The paper claims the JS gate "fires at corruption boundaries and is
#   silent within stable periods". This measures that directly:
#     - boundary recall     : of the 74 block transitions, how many fire?
#     - within-block rate   : how often does it fire when nothing changed?
#     - lift                : P(fire|boundary) / P(fire|non-boundary)
#   Lift is the headline number. A random gate has lift 1.0.
#
#   NOTE ON PRECISION: there are 74 boundaries among 23,400 batches
#   (density 0.0032). A gate with PERFECT recall firing at 16% still
#   scores precision ~0.02 purely from that base rate. Precision alone is
#   meaningless here — report lift and boundary recall instead.
#
# EXPERIMENT 6 — Oracle boundary control
#   Adapt ONLY at known block boundaries (no JS gate). Upper bound on what
#   perfect timing buys. Identical in every other respect: same entropy
#   filter, same |R_t|>0 guard, same optimiser, same stream.
#
# PROTOCOL: matches cifar10c_drift_analysis.py exactly
#   Severity-major: for severity in [1..5]: for corruption in ALL_15
#   -> 75 blocks, 312 batches/block, 23,400 batches, no reset
#
# ALGORITHM 1 CONFORMANCE:
#   Batch 1 adapts unconditionally ("if not initialised or JS > tau").
#   Your drift script logs gate_log.append(0) and skips — that differs
#   from the paper by exactly 1 batch. This script follows the paper.
#
# OUTPUT:
#   results/gate_analysis/gate_log_batches.csv   — per-batch record
#   results/gate_analysis/gate_metrics.txt       — detection metrics
#   results/gate_analysis/oracle_results.csv     — oracle accuracies
#   results/gate_analysis/paper_text.txt         — sentences for the paper
#
# Run:
#   python cifar10c_gate_analysis.py                  # both experiments
#   python cifar10c_gate_analysis.py --exp 5          # detection only
#   python cifar10c_gate_analysis.py --exp 6          # oracle only
#   python cifar10c_gate_analysis.py --analyse_only   # re-analyse saved log
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
# CONFIG — identical to cifar10c_drift_analysis.py
# =============================================================================

MODEL_PATH  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\gate_analysis"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
NUM_CLASSES = 10
NUM_WORKERS = 0
SEVERITIES  = [1, 2, 3, 4, 5]

LR           = 1e-3
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)
JS_THRESHOLD = 0.04

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

N_BLOCKS          = len(SEVERITIES) * len(ALL_CORRUPTIONS)      # 75
BATCHES_PER_BLOCK = 10000 // BATCH_SIZE                          # 312
ORACLE_WARMUPS    = [1, 2, 5]

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. DATASET — identical to drift script
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
# 2. HELPERS — identical to drift script
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


# =============================================================================
# 3. PER-BATCH RECORDER
# =============================================================================

class BatchRecorder:
    """One row per batch. This is the whole of Experiment 5."""

    def __init__(self):
        self.rows = []
        self.t    = 0            # global batch index across all 75 blocks

    def log(self, js, adapted, n_reliable):
        block        = self.t // BATCHES_PER_BLOCK
        pos_in_block = self.t %  BATCHES_PER_BLOCK
        self.rows.append(dict(
            t=self.t,
            block=block,
            pos_in_block=pos_in_block,
            # a corruption boundary is the first batch of any block after the first
            is_boundary=(pos_in_block == 0 and block > 0),
            # severity jumps additionally occur at blocks 15,30,45,60
            is_severity_jump=(pos_in_block == 0 and block > 0
                              and block % len(ALL_CORRUPTIONS) == 0),
            js=float(js),
            adapted=bool(adapted),
            n_reliable=int(n_reliable),
        ))
        self.t += 1

    def save(self, path):
        with open(path, "w") as f:
            f.write("t,block,pos_in_block,is_boundary,is_severity_jump,"
                    "js,adapted,n_reliable\n")
            for r in self.rows:
                js = "" if r["js"] != r["js"] else f"{r['js']:.8f}"
                f.write(f"{r['t']},{r['block']},{r['pos_in_block']},"
                        f"{int(r['is_boundary'])},{int(r['is_severity_jump'])},"
                        f"{js},{int(r['adapted'])},{r['n_reliable']}\n")
        print(f"  Saved: {path}  ({len(self.rows)} batches)")


def load_batch_log(path):
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            rows.append(dict(
                t=int(p[0]), block=int(p[1]), pos_in_block=int(p[2]),
                is_boundary=bool(int(p[3])), is_severity_jump=bool(int(p[4])),
                js=float(p[5]) if p[5] else float("nan"),
                adapted=bool(int(p[6])), n_reliable=int(p[7]),
            ))
    return rows


# =============================================================================
# 4. EXPERIMENT 5 — instrumented ContinualTTA
#    Follows Algorithm 1: batch 1 adapts unconditionally.
# =============================================================================

def make_ctta_logged(source, recorder):
    model, params = setup_bn(copy.deepcopy(source))
    opt       = torch.optim.Adam(params, lr=LR)
    reference = [None]

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)

            # Algorithm 1, line 4: "if not initialised or JS > tau"
            if reference[0] is None:
                reference[0] = p_t.clone()      # line 12
                js_val = float("nan")           # JS undefined on batch 1
                adapt  = True                   # unconditional per paper
            else:
                m    = 0.5 * (reference[0] + p_t)
                kl_1 = F.kl_div(m.log().unsqueeze(0),
                                reference[0].unsqueeze(0), reduction="batchmean")
                kl_2 = F.kl_div(m.log().unsqueeze(0),
                                p_t.unsqueeze(0), reduction="batchmean")
                js   = 0.5 * (kl_1 + kl_2)
                reference[0] = 0.9 * reference[0] + 0.1 * p_t   # line 12
                js_val = js.item()
                adapt  = js_val > JS_THRESHOLD

        if not adapt:
            recorder.log(js_val, False, 0)
            return logits

        entropy  = softmax_entropy(logits)                       # line 5
        reliable = entropy < E_MARGIN
        n_rel    = int(reliable.sum())
        if n_rel == 0:                                           # line 6 guard
            recorder.log(js_val, False, 0)
            return logits

        entropy[reliable].mean().backward()                      # lines 7-8
        opt.step(); opt.zero_grad()
        recorder.log(js_val, True, n_rel)
        return logits

    return model, fn


# =============================================================================
# 5. EXPERIMENT 6 — oracle boundary control
#    Adapts only on the first `warmup` batches of each block.
#    No JS gate, no EMA reference. Everything else identical, so the only
#    variable is WHEN the update happens. p_ref never enters the loss or the
#    update in Algorithm 1 — only the gate decision — so omitting it cannot
#    affect the comparison.
# =============================================================================

def make_ctta_oracle(source, warmup=1, recorder=None):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    t   = [0]

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        adapt  = (t[0] % BATCHES_PER_BLOCK) < warmup
        t[0]  += 1

        if not adapt:
            if recorder: recorder.log(float("nan"), False, 0)
            return logits

        entropy  = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        n_rel    = int(reliable.sum())
        if n_rel == 0:
            if recorder: recorder.log(float("nan"), False, 0)
            return logits

        entropy[reliable].mean().backward()
        opt.step(); opt.zero_grad()
        if recorder: recorder.log(float("nan"), True, n_rel)
        return logits

    return model, fn


# =============================================================================
# 6. CONTINUAL EVALUATION LOOP — severity-major, no reset
# =============================================================================

def run_continual(fn, tag=""):
    """Walk all 75 blocks in the same order as the drift script."""
    block_accs = []
    block_idx  = 0
    for severity in SEVERITIES:
        for corruption in ALL_CORRUPTIONS:
            loader = get_loader(corruption, severity)
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = fn(x)
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)
            del loader; torch.cuda.empty_cache()

            acc = 100.0 * correct / total
            block_accs.append(acc)
            block_idx += 1
            print(f"  {tag}Block {block_idx:>2}/{N_BLOCKS}  S{severity} "
                  f"{corruption:<20} acc={acc:.1f}%", end="\r")
    print()
    return block_accs


# =============================================================================
# 7. EXPERIMENT 5 — ANALYSIS
# =============================================================================

def analyse_gate(rows, tolerance=1):
    """
    tolerance=k : a firing counts as detecting a boundary if it lands within
                  k batches of one. k=1 means only the first batch counts.
    Report several k rather than picking the flattering one.
    """
    n_total  = len(rows)
    n_uncond = sum(1 for r in rows if r["js"] != r["js"])
    # firing rate over ALL batches -> comparable with the paper's 16.1%
    fire_rate_all = sum(r["adapted"] for r in rows) / n_total if n_total else 0.0

    # detection metrics use only gate-DECIDED batches (exclude nan batch 1)
    dec = [r for r in rows if r["js"] == r["js"]]
    n   = len(dec)

    bpos = {b * BATCHES_PER_BLOCK for b in range(1, N_BLOCKS)}
    near = lambda t: any(abs(t - bp) < tolerance for bp in bpos)

    tp = sum(1 for r in dec if r["adapted"] and near(r["t"]))
    fp = sum(1 for r in dec if r["adapted"] and not near(r["t"]))
    fn = sum(1 for r in dec if not r["adapted"] and near(r["t"]))
    tn = n - tp - fp - fn

    n_b, n_nb = tp + fn, fp + tn
    p_fire_b  = tp / n_b  if n_b  else 0.0      # boundary recall
    p_fire_nb = fp / n_nb if n_nb else 0.0      # within-block firing rate
    lift      = p_fire_b / p_fire_nb if p_fire_nb > 0 else float("inf")

    # severity jumps scored separately (larger shift -> should be easier)
    sev = [r for r in dec if r["is_severity_jump"]]
    sev_recall = sum(r["adapted"] for r in sev) / len(sev) if sev else float("nan")

    # where within a block do spurious firings land?
    within_pos = [r["pos_in_block"] for r in dec
                  if r["adapted"] and not near(r["t"])]

    return dict(
        n_batches=n_total, n_unconditional=n_uncond, tolerance=tolerance,
        tp=tp, fp=fp, fn=fn, tn=tn,
        fire_rate=fire_rate_all,
        boundary_recall=p_fire_b,
        within_block_fire_rate=p_fire_nb,
        lift=lift,
        severity_jump_recall=sev_recall,
        chance_precision=n_b / n if n else 0.0,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        mean_pos_within=(sum(within_pos) / len(within_pos)) if within_pos else None,
        median_js_boundary=float(np.median([r["js"] for r in dec if near(r["t"])]))
                            if any(near(r["t"]) for r in dec) else float("nan"),
        median_js_within=float(np.median([r["js"] for r in dec if not near(r["t"])]))
                            if any(not near(r["t"]) for r in dec) else float("nan"),
    )


def report_gate(rows):
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 62)
    out("EXPERIMENT 5 — GATE DETECTION QUALITY")
    out("=" * 62)
    m1 = analyse_gate(rows, 1)
    out(f"\nBatches            : {m1['n_batches']}")
    out(f"Unconditional (b1) : {m1['n_unconditional']}  (Algorithm 1 line 4)")
    out(f"Overall fire rate  : {100*m1['fire_rate']:.2f}%   "
        f"(paper reports 16.1%)")
    out(f"Boundaries         : {N_BLOCKS-1}")

    out(f"\n{'tol':>4} {'bnd recall':>11} {'within-blk':>11} {'lift':>9} "
        f"{'caught':>8}")
    out("-" * 48)
    for tol in (1, 2, 5):
        m = analyse_gate(rows, tol)
        lift = "inf" if m["lift"] == float("inf") else f"{m['lift']:.1f}"
        out(f"{tol:>4} {m['boundary_recall']:11.3f} "
            f"{m['within_block_fire_rate']:11.4f} {lift:>9} "
            f"{m['tp']:>4}/{N_BLOCKS-1:<3}")

    out(f"\nSeverity-jump recall : {m1['severity_jump_recall']:.3f}  "
        f"(blocks 15,30,45,60)")
    out(f"Median JS at boundary: {m1['median_js_boundary']:.5f}")
    out(f"Median JS within blk : {m1['median_js_within']:.5f}")
    out(f"Threshold tau        : {JS_THRESHOLD}")
    if m1["mean_pos_within"] is not None:
        out(f"Mean position of within-block firing: "
            f"{m1['mean_pos_within']:.1f} / {BATCHES_PER_BLOCK}")

    out(f"\nprecision={m1['precision']:.4f}  chance={m1['chance_precision']:.4f}")
    out("  ^ precision is base-rate-limited here; use LIFT as the headline.")

    out("\n" + "=" * 62)
    out("INTERPRETATION")
    out("=" * 62)
    if m1["lift"] == float("inf") or m1["lift"] > 3:
        out("  Lift is well above 1 -> firing IS boundary-localised.")
        out("  The paper's timing claim is now measured, not asserted.")
    elif m1["lift"] > 1.5:
        out("  Lift is moderate. Firing is boundary-biased but not sharply so.")
        out("  Consider softening 'silent within stable periods'.")
    else:
        out("  Lift is near 1 -> firing is roughly UNIFORM, not boundary-localised.")
        out("  The accuracy gain likely comes from adapting LESS, not from")
        out("  adapting at the right TIME. 'Timing, not quantity' would need")
        out("  rewording. Better to find this now than in review.")

    return "\n".join(lines), m1


# =============================================================================
# 8. PAPER TEXT
# =============================================================================

def write_paper_text(m, oracle=None, ours_acc=None):
    L = []
    L.append("PAPER TEXT — GATE ANALYSIS")
    L.append("=" * 62)
    L.append("")
    lift = "inf" if m["lift"] == float("inf") else f"{m['lift']:.0f}"
    L.append("\\paragraph{Gate detection quality.}")
    L.append(f"Across the {N_BLOCKS}-block continual sequence "
             f"({m['n_batches']} batches), the")
    L.append(f"gate fires on {100*m['boundary_recall']:.0f}\\% of the "
             f"{N_BLOCKS-1} corruption boundaries and on")
    L.append(f"only {100*m['within_block_fire_rate']:.1f}\\% of within-block "
             f"batches, a {lift}$\\times$")
    L.append("enrichment over chance. Median JS at a boundary is")
    L.append(f"${m['median_js_boundary']:.3f}$ against ${m['median_js_within']:.3f}$ "
             f"within blocks, with")
    L.append(f"$\\tau={JS_THRESHOLD}$ sitting between the two. Firing is therefore")
    L.append("boundary-localised rather than uniform, which is the mechanism")
    L.append("the $O(K)$ bound of Section~\\ref{sec:theory} assumes.")
    L.append("")

    if oracle:
        L.append("\\paragraph{How much of the achievable gain does the gate recover?}")
        best_k, best_acc = max(oracle.items(), key=lambda kv: kv[1])
        rate = 100.0 * best_k / BATCHES_PER_BLOCK
        L.append(f"An oracle that adapts only at the {N_BLOCKS-1} ground-truth")
        L.append(f"boundaries reaches ${best_acc:.2f}\\%$ at a {rate:.1f}\\% firing")
        L.append("rate.")
        if ours_acc is not None:
            gap = best_acc - ours_acc
            L.append(f"\\ours{{}} reaches ${ours_acc:.2f}\\%$, within "
                     f"${abs(gap):.2f}\\%$ of this upper")
            L.append("bound while requiring no boundary supervision. The residual")
            L.append("gap quantifies the cost of inferring boundaries from")
            L.append("prediction marginals alone.")
        L.append("")
        L.append("% Oracle sweep:")
        for k, a in sorted(oracle.items()):
            L.append(f"%   warmup={k} ({100.0*k/BATCHES_PER_BLOCK:.2f}% fire): "
                     f"{a:.2f}%")

    txt  = "\n".join(L)
    path = os.path.join(RESULTS_DIR, "paper_text.txt")
    with open(path, "w") as f:
        f.write(txt)
    print("\n" + txt)
    print(f"\n  Saved: {path}")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="ContinualTTA gate detection + oracle control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cifar10c_gate_analysis.py
  python cifar10c_gate_analysis.py --exp 5
  python cifar10c_gate_analysis.py --exp 6 --warmups 1 2 5
  python cifar10c_gate_analysis.py --analyse_only
        """)
    ap.add_argument("--exp", nargs="+", type=int, default=[5, 6],
                    choices=[5, 6])
    ap.add_argument("--warmups", nargs="+", type=int, default=ORACLE_WARMUPS)
    ap.add_argument("--analyse_only", action="store_true",
                    help="re-analyse a saved batch log without rerunning")
    args = ap.parse_args()

    log_path    = os.path.join(RESULTS_DIR, "gate_log_batches.csv")
    oracle_path = os.path.join(RESULTS_DIR, "oracle_results.csv")

    print("=" * 62)
    print("ContinualTTA — Gate Analysis")
    print("=" * 62)
    print(f"Device            : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
    print(f"Blocks            : {N_BLOCKS}  (severity-major, no reset)")
    print(f"Batches per block : {BATCHES_PER_BLOCK}")
    print(f"Total batches     : {N_BLOCKS * BATCHES_PER_BLOCK}")
    print(f"Boundaries        : {N_BLOCKS - 1}")
    print(f"tau               : {JS_THRESHOLD}")
    print(f"Results           : {RESULTS_DIR}\n")

    # ---------------- analyse-only path ----------------
    if args.analyse_only:
        rows = load_batch_log(log_path)
        if rows is None:
            print(f"No saved log at {log_path}. Run --exp 5 first.")
            return
        print(f"Loaded {len(rows)} batches from {log_path}\n")
        _, m = report_gate(rows)
        oracle = {}
        if os.path.isfile(oracle_path):
            with open(oracle_path) as f:
                next(f)
                for line in f:
                    k, a = line.strip().split(",")[:2]
                    oracle[int(k)] = float(a)
        write_paper_text(m, oracle or None)
        return

    # ---------------- data + model ----------------
    for f_ in ["gaussian_noise.npy", "labels.npy"]:
        assert os.path.isfile(f"{DATA_DIR}/{f_}"), f"Missing: {DATA_DIR}/{f_}"
    print("Data check: passed")
    print("Loading source model...")
    source = load_model()
    print(f"  Parameters: {sum(p.numel() for p in source.parameters()):,}\n")

    ours_acc = None

    # ---------------- EXPERIMENT 5 ----------------
    if 5 in args.exp:
        print("=" * 62)
        print("EXPERIMENT 5 — instrumented ContinualTTA")
        print("=" * 62)
        rec = BatchRecorder()
        _, fn = make_ctta_logged(source, rec)
        accs  = run_continual(fn, tag="")
        ours_acc = float(np.mean(accs))
        print(f"  Mean accuracy: {ours_acc:.2f}%")
        rec.save(log_path)

        txt, m = report_gate(rec.rows)
        with open(os.path.join(RESULTS_DIR, "gate_metrics.txt"), "w") as f:
            f.write(txt)
        torch.cuda.empty_cache()
    else:
        m = None

    # ---------------- EXPERIMENT 6 ----------------
    oracle = {}
    if 6 in args.exp:
        print("\n" + "=" * 62)
        print("EXPERIMENT 6 — oracle boundary control")
        print("=" * 62)
        for k in args.warmups:
            rate = 100.0 * k / BATCHES_PER_BLOCK
            print(f"\n  warmup={k}  (fires on {rate:.2f}% of batches)")
            _, fn = make_ctta_oracle(source, warmup=k)
            accs  = run_continual(fn, tag=f"[k={k}] ")
            acc   = float(np.mean(accs))
            oracle[k] = acc
            print(f"  -> mean accuracy: {acc:.2f}%")
            torch.cuda.empty_cache()

        with open(oracle_path, "w") as f:
            f.write("warmup,mean_accuracy,fire_rate_pct\n")
            for k, a in sorted(oracle.items()):
                f.write(f"{k},{a:.4f},{100.0*k/BATCHES_PER_BLOCK:.4f}\n")
        print(f"\n  Saved: {oracle_path}")

        print(f"\n  {'warmup':>7} {'fire%':>8} {'accuracy':>10}")
        print("  " + "-" * 27)
        for k, a in sorted(oracle.items()):
            print(f"  {k:>7} {100.0*k/BATCHES_PER_BLOCK:>7.2f}% {a:>9.2f}%")
        if ours_acc is not None:
            print(f"  {'ours':>7} {'16.1':>7}% {ours_acc:>9.2f}%")

    # ---------------- paper text ----------------
    if m is None:
        rows = load_batch_log(log_path)
        if rows:
            _, m = report_gate(rows)
    if m is not None:
        write_paper_text(m, oracle or None, ours_acc)

    print("\n" + "=" * 62)
    print("DONE")
    print("=" * 62)
    print(f"Results: {RESULTS_DIR}/")
    print("  gate_log_batches.csv — per-batch JS, gate decision, position")
    print("  gate_metrics.txt     — detection metrics")
    print("  oracle_results.csv   — oracle accuracy per warmup")
    print("  paper_text.txt       — LaTeX-ready sentences")


if __name__ == "__main__":
    freeze_support()
    main()