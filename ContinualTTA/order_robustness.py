# =============================================================================
# ContinualTTA — Block-Order Robustness (CIFAR-10-C, truly continual)
#
# WHY THIS EXPERIMENT
#   The no-reset protocol is the paper's primary contribution, but the
#   manuscript evaluates exactly ONE block order (severity-major, fixed
#   corruption sequence). A reviewer can ask whether the reported numbers
#   are properties of the methods or artifacts of that ordering.
#   This measures across several orders and reports mean +/- std per method.
#
# WHAT IT VARIES
#   ONLY the order in which the 75 corruption-severity blocks are visited.
#   Everything else is held fixed: source model, seed, batch composition,
#   hyperparameters, optimiser. So any spread is attributable to order.
#
# ORDERS TESTED
#   severity_major  the paper's order (S1 all 15, S2 all 15, ...)   [reference]
#   corruption_major  all 5 severities of each corruption in turn
#   random_1/2/3    uniformly random permutations of the 75 blocks
#   easy_to_hard    ascending by no-adapt baseline accuracy
#   hard_to_easy    descending (adversarial: worst case first)
#
#   The last two are the informative extremes. If a method is order-robust
#   it should survive hard_to_easy; TENT/EATA likely will not, which
#   STRENGTHENS the paper's argument rather than weakening it.
#
# CONFIG: identical to cifar10c_drift_analysis.py. Nothing to adapt.
#
# Run:
#   python cifar10c_order_robustness.py
#   python cifar10c_order_robustness.py --methods ContinualTTA TENT
#   python cifar10c_order_robustness.py --orders severity_major hard_to_easy
#   python cifar10c_order_robustness.py --skip_done
#   python cifar10c_order_robustness.py --report_only
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
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import freeze_support

# =============================================================================
# CONFIG — matches cifar10c_drift_analysis.py exactly
# =============================================================================

MODEL_PATH  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\order_robustness"

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

SEED = 42          # held FIXED: we vary order, not batch composition

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

METHODS = ["Baseline", "TENT", "SAR", "ContinualTTA"]
ORDERS  = ["severity_major", "corruption_major",
           "random_1", "random_2", "random_3",
           "easy_to_hard", "hard_to_easy"]

# Per-corruption no-adapt baseline accuracy (S1-S5 mean), from the paper's
# supplementary Table S6(a). Used only to construct the difficulty orders.
BASELINE_ACC = {
    "gaussian_noise": 72.1, "shot_noise": 74.4, "impulse_noise": 71.7,
    "defocus_blur": 82.3,   "glass_blur": 65.7, "motion_blur": 80.6,
    "zoom_blur": 84.4,      "snow": 80.2,       "frost": 78.3,
    "fog": 76.2,            "brightness": 91.1, "contrast": 74.8,
    "elastic_transform": 80.2, "pixelate": 81.3, "jpeg_compression": 76.4,
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================================
# 1. BLOCK ORDERS
# =============================================================================

def build_order(name):
    """Return a list of 75 (corruption, severity) pairs."""
    if name == "severity_major":
        # the paper's order
        return [(c, s) for s in SEVERITIES for c in ALL_CORRUPTIONS]

    if name == "corruption_major":
        return [(c, s) for c in ALL_CORRUPTIONS for s in SEVERITIES]

    if name.startswith("random_"):
        k = int(name.split("_")[1])
        blocks = [(c, s) for s in SEVERITIES for c in ALL_CORRUPTIONS]
        rng = random.Random(1000 + k)      # fixed per order, reproducible
        rng.shuffle(blocks)
        return blocks

    if name in ("easy_to_hard", "hard_to_easy"):
        # sort corruptions by no-adapt difficulty, keep severity ascending
        # within a corruption so severity is not confounded with difficulty
        corr = sorted(ALL_CORRUPTIONS, key=lambda c: BASELINE_ACC[c],
                      reverse=(name == "easy_to_hard"))
        return [(c, s) for c in corr for s in SEVERITIES]

    raise ValueError(f"unknown order: {name}")


def describe_order(blocks):
    first = ", ".join(f"{c[:8]}/S{s}" for c, s in blocks[:3])
    last  = ", ".join(f"{c[:8]}/S{s}" for c, s in blocks[-2:])
    return f"{first} ... {last}"


# =============================================================================
# 2. DATA / MODEL — identical to the drift script
# =============================================================================

class CIFAR10C_Dataset(Dataset):
    def __init__(self, corruption, severity):
        data   = np.load(f"{DATA_DIR}/{corruption}.npy", mmap_mode='r')
        labels = np.load(f"{DATA_DIR}/labels.npy",      mmap_mode='r')
        start  = (severity - 1) * 10000
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


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# =============================================================================
# 3. METHODS — identical to the drift script
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


def make_baseline(source):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad(): return model(x)
    return fn, None


def make_tent(source):
    """Paper-faithful TENT: entropy minimisation over the FULL batch.
    No reliable-sample filter; that is EATA's contribution, not TENT's."""
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        softmax_entropy(logits).mean().backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn, None


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
        dyn = min(E_MARGIN, ema_entropy[0] + 0.4 * math.log(NUM_CLASSES))
        reliable = entropy_init < dyn
        if reliable.sum() == 0: return logits_init

        x_rel = x[reliable]
        softmax_entropy(model(x_rel)).mean().backward()
        gnorm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))

        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (gnorm + 1e-12)
                p.data.add_(e_w); e_ws.append(e_w); p.grad.zero_()
            else:
                e_ws.append(None)

        ent2 = softmax_entropy(model(x_rel))
        if (ent2 < E_MARGIN).sum() > 0:
            ent2[ent2 < E_MARGIN].mean().backward()

        for p, e_w in zip(params, e_ws):
            if e_w is not None: p.data.sub_(e_w)
        opt.step(); opt.zero_grad()

        with torch.no_grad():
            logits_out = model(x)
            ema_entropy[0] = (0.9 * ema_entropy[0]
                              + 0.1 * softmax_entropy(logits_out).mean().item())
            if ema_entropy[0] < SAR_E0:
                for n, p in model.named_parameters():
                    if p.requires_grad and n in init_params:
                        p.data.copy_(init_params[n])
                ema_entropy[0] = None
        return logits_out
    return fn, None


def make_ctta(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    reference = [None]
    gate_log  = []

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                gate_log.append(0)
                return logits
            m    = 0.5 * (reference[0] + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                            reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                            p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5 * (kl_1 + kl_2)
            reference[0] = 0.9 * reference[0] + 0.1 * p_t
            fired = js.item() > JS_THRESHOLD
            gate_log.append(1 if fired else 0)
        if not fired: return logits

        entropy  = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        if reliable.sum() == 0: return logits
        entropy[reliable].mean().backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn, gate_log


def build(method, source):
    return {"Baseline": make_baseline, "TENT": make_tent,
            "SAR": make_sar, "ContinualTTA": make_ctta}[method](source)


# =============================================================================
# 4. RUN ONE (method, order)
# =============================================================================

def run_one(method, order_name, source):
    blocks = build_order(order_name)
    assert len(blocks) == 75, f"expected 75 blocks, got {len(blocks)}"

    set_seed(SEED)
    fn, gate_log = build(method, source)

    per_block = []
    for i, (corruption, severity) in enumerate(blocks, 1):
        loader = get_loader(corruption, severity)
        correct = total = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = fn(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
        del loader; torch.cuda.empty_cache()
        acc = 100.0 * correct / total
        per_block.append(dict(block=i, corruption=corruption,
                              severity=severity, acc=acc))
        print(f"    [{method}/{order_name}] block {i:>2}/75  "
              f"S{severity} {corruption:<20} {acc:.1f}%", end="\r")

    mean_acc = float(np.mean([b["acc"] for b in per_block]))
    gate_rate = (100.0 * float(np.mean(gate_log))) if gate_log else float("nan")
    print(f"    [{method}/{order_name}] mean {mean_acc:.2f}%"
          + (f"  gate {gate_rate:.1f}%" if gate_log else "") + " " * 20)
    return per_block, mean_acc, gate_rate


def save_run(method, order_name, per_block, mean_acc, gate_rate):
    path = os.path.join(RESULTS_DIR, f"{method}_{order_name}.csv")
    with open(path, "w") as f:
        f.write("block,corruption,severity,accuracy\n")
        for b in per_block:
            f.write(f"{b['block']},{b['corruption']},{b['severity']},"
                    f"{b['acc']:.4f}\n")
        f.write(f"Mean,,,{mean_acc:.4f}\n")
        if gate_rate == gate_rate:
            f.write(f"GateRate,,,{gate_rate:.4f}\n")


def load_run(method, order_name):
    path = os.path.join(RESULTS_DIR, f"{method}_{order_name}.csv")
    if not os.path.isfile(path): return None, None
    mean_acc = gate = None
    with open(path) as f:
        for line in f:
            if line.startswith("Mean,"):     mean_acc = float(line.split(",")[3])
            if line.startswith("GateRate,"): gate     = float(line.split(",")[3])
    return mean_acc, gate


# =============================================================================
# 5. REPORT
# =============================================================================

def report(results, orders, methods):
    """results[(method, order)] = (mean_acc, gate_rate)"""
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 74)
    out("BLOCK-ORDER ROBUSTNESS — CIFAR-10-C continual, seed fixed at 42")
    out("=" * 74)
    out("Only the block visitation order varies. Source model, seed, batch")
    out("composition, and all hyperparameters are held fixed.")
    out("")

    hdr = f"{'Method':<16}" + "".join(f"{o[:11]:>12}" for o in orders) \
          + f"{'mean':>9}{'std':>8}{'range':>9}"
    out(hdr); out("-" * len(hdr))

    summary = {}
    for m in methods:
        vals = [results.get((m, o), (None, None))[0] for o in orders]
        got  = [v for v in vals if v is not None]
        row = f"{m:<16}" + "".join(
            f"{v:>12.2f}" if v is not None else f"{'--':>12}" for v in vals)
        if got:
            mu, sd = float(np.mean(got)), float(np.std(got, ddof=1)) if len(got)>1 else 0.0
            rng = max(got) - min(got)
            row += f"{mu:>9.2f}{sd:>8.2f}{rng:>9.2f}"
            summary[m] = (mu, sd, rng, got)
        out(row)

    out("")
    out("=" * 74)
    out("INTERPRETATION")
    out("=" * 74)
    if summary:
        ranked = sorted(summary.items(), key=lambda kv: kv[1][1])
        out("Order sensitivity, lowest std first:")
        for m, (mu, sd, rng, got) in ranked:
            out(f"  {m:<16} {mu:6.2f} +/- {sd:5.2f}   range {rng:5.2f}")
        best = ranked[0][0]
        out("")
        out(f"Most order-robust: {best}")
        if "ContinualTTA" in summary:
            mu, sd, rng, _ = summary["ContinualTTA"]
            out(f"ContinualTTA std {sd:.2f} across orders.")
            if sd < 1.0:
                out("  -> below 1 point: the reported result is not an artifact")
                out("     of the chosen block order.")
            else:
                out("  -> above 1 point: report this as an order-sensitivity")
                out("     limitation rather than claiming order invariance.")
        if "hard_to_easy" in orders and "severity_major" in orders:
            out("")
            out("Adversarial order (hard_to_easy) vs the paper's order:")
            for m in methods:
                a = results.get((m, "severity_major"), (None,))[0]
                bq = results.get((m, "hard_to_easy"), (None,))[0]
                if a is not None and bq is not None:
                    out(f"  {m:<16} {a:6.2f} -> {bq:6.2f}   ({bq-a:+.2f})")

    out("")
    out("=" * 74)
    out("PAPER TEXT (fill from the table above)")
    out("=" * 74)
    if "ContinualTTA" in summary:
        mu, sd, rng, got = summary["ContinualTTA"]
        out("\\paragraph{Block-order robustness.}")
        out(f"The continual protocol fixes one block order, so we test whether")
        out(f"our conclusions depend on it. Holding the source model, seed, and")
        out(f"batch composition fixed and varying only the visitation order over")
        out(f"{len(got)} orderings (including corruption-major, random")
        out(f"permutations, and difficulty-sorted extremes), \\ours{{}} reaches")
        out(f"${mu:.2f}\\pm{sd:.2f}\\%$ with a range of ${rng:.2f}$ points.")
        out("% add the comparable spread for baselines from the table")

    txt = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "order_report.txt"), "w") as f:
        f.write(txt)

    with open(os.path.join(RESULTS_DIR, "summary.csv"), "w") as f:
        f.write("method," + ",".join(orders) + ",mean,std,range\n")
        for m in methods:
            vals = [results.get((m, o), (None, None))[0] for o in orders]
            got = [v for v in vals if v is not None]
            cells = ",".join(f"{v:.4f}" if v is not None else "" for v in vals)
            if got:
                mu = np.mean(got); sd = np.std(got, ddof=1) if len(got)>1 else 0.0
                f.write(f"{m},{cells},{mu:.4f},{sd:.4f},{max(got)-min(got):.4f}\n")
            else:
                f.write(f"{m},{cells},,,\n")
    print(f"\n  Saved: {RESULTS_DIR}/order_report.txt")
    print(f"  Saved: {RESULTS_DIR}/summary.csv")


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Block-order robustness on CIFAR-10-C",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--orders",  nargs="+", default=ORDERS,  choices=ORDERS)
    ap.add_argument("--skip_done",   action="store_true")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print("Block-Order Robustness — CIFAR-10-C truly continual")
    print("=" * 74)
    print(f"Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Methods : {args.methods}")
    print(f"Orders  : {args.orders}")
    print(f"Seed    : {SEED} (FIXED — only order varies)")
    print(f"Runs    : {len(args.methods) * len(args.orders)}")
    print(f"Results : {RESULTS_DIR}\n")

    print("Block orders:")
    for o in args.orders:
        print(f"  {o:<18} {describe_order(build_order(o))}")
    print()

    results = {}
    for m in args.methods:
        for o in args.orders:
            mean_acc, gate = load_run(m, o)
            if mean_acc is not None:
                results[(m, o)] = (mean_acc, gate)

    if args.report_only:
        if results: report(results, args.orders, args.methods)
        else: print("No saved runs. Run without --report_only first.")
        return

    for f_ in ["gaussian_noise.npy", "labels.npy"]:
        assert os.path.isfile(f"{DATA_DIR}/{f_}"), f"Missing: {DATA_DIR}/{f_}"
    print("Data check: passed")
    print("Loading source model...")
    source = load_model()
    print()

    total = len(args.methods) * len(args.orders)
    done = 0
    for m in args.methods:
        print(f"\n{'='*74}\nMethod: {m}\n{'='*74}")
        for o in args.orders:
            done += 1
            if args.skip_done and (m, o) in results:
                print(f"  [{done}/{total}] skip {m}/{o} "
                      f"(saved: {results[(m,o)][0]:.2f}%)")
                continue
            print(f"  [{done}/{total}] {m} / {o}")
            per_block, mean_acc, gate = run_one(m, o, source)
            save_run(m, o, per_block, mean_acc, gate)
            results[(m, o)] = (mean_acc, gate)
            torch.cuda.empty_cache()

    print()
    report(results, args.orders, args.methods)


if __name__ == "__main__":
    freeze_support()
    main()