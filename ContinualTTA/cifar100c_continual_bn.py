# =============================================================================
# CIFAR-100-C Truly Continual — BN (with adaptation budget for ContinualTTA)
# All Methods: Baseline | TENT | EATA | CoTTA | RoTTA | SAR | ContinualTTA
#
# PROTOCOL: One model per method runs through all 15 corruption types
#           at severity 5 in fixed order, without any reset.
#           Total: 15 × 10,000 = 150,000 images per method.
#
# CONTINUALTTA: JS gate + entropy filter + confidence gate + adaptation budget.
#               Budget resets per corruption (fresh start within stream).
#
# Run:
#   python cifar100c_continual_budget.py --methods ContinualTTA   # pilot
#   python cifar100c_continual_budget.py                          # full run
#   python cifar100c_continual_budget.py --skip_done
#   python cifar100c_continual_budget.py --table_only
# =============================================================================

import os, copy, math, argparse, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ------------------------- CONFIG -------------------------
MODEL_PATH  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar100_source.pth"
DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\CIFAR-100-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\cifar100c_continual_budget"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
NUM_CLASSES = 100
SEVERITY    = 5
NUM_WORKERS = 0

# Hyperparameters — start with these; tune if needed
LR           = 1e-3              # reduced LR to slow down adaptation
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # ≈1.842 nats
MIN_CONF     = 0.5              # stricter confidence gate
JS_THRESHOLD = 0.10            # higher threshold to be more conservative
ADAPT_BUDGET = 100              # max backward passes per corruption
ROTTA_NU     = 0.001
ROTTA_N      = 64
SAR_RHO      = 0.05
SAR_E0       = 0.3

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]

print(f"Device     : {DEVICE}")
if torch.cuda.is_available(): print(f"GPU        : {torch.cuda.get_device_name(0)}")
print(f"Classes    : {NUM_CLASSES}")
print(f"LR         : {LR}")
print(f"E_margin   : {E_MARGIN:.3f} nats")
print(f"JS tau     : {JS_THRESHOLD}")
print(f"MIN_CONF   : {MIN_CONF}")
print(f"Budget     : {ADAPT_BUDGET} backward passes per corruption")
print(f"Methods    : {METHODS}")

# ------------------------- DATASET -------------------------
class CIFAR100C_Dataset(Dataset):
    def __init__(self, corruption, severity):
        data   = np.load(f"{DATA_DIR}/{corruption}.npy", mmap_mode='r')
        labels = np.load(f"{DATA_DIR}/labels.npy",       mmap_mode='r')
        start  = (severity - 1) * 10000
        self.images = data[start:start+10000]
        self.labels = labels[start:start+10000]
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.transform(Image.fromarray(self.images[idx])), int(self.labels[idx])

def get_loader(corruption):
    return DataLoader(CIFAR100C_Dataset(corruption, SEVERITY),
                      batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

# ------------------------- HELPERS -------------------------
def softmax_entropy(logits):
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)

def eval_loader(model_fn, loader):
    correct = total = 0
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
              if isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    return model, params

# ------------------------- METHODS (all paper‑faithful) -------------------------
# Baseline
def make_baseline(source):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad():
            return model(x)
    return fn

# TENT (unfiltered, original)
def make_tent(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        loss = softmax_entropy(logits).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        return logits
    return fn

# EATA (Fisher on first corruption)
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
    d_margin = 0.05
    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        entropy = softmax_entropy(logits)
        probs = logits.softmax(1)
        mask_e = entropy < E_MARGIN
        if ref_probs[0] is not None:
            cos_sim = F.cosine_similarity(ref_probs[0].unsqueeze(0).expand(probs.size(0),-1), probs, dim=1)
            mask_d = cos_sim < (1.0 - d_margin)
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

# CoTTA
def make_cotta(source):
    src = copy.deepcopy(source).eval(); src.requires_grad_(False)
    adapted, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    teacher = copy.deepcopy(source).eval(); teacher.requires_grad_(False)
    aug = transforms.Compose([transforms.RandomHorizontalFlip(),
                              transforms.RandomResizedCrop(224, scale=(0.8,1.0))])
    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            pseudo = torch.stack([teacher(aug(x)).softmax(1) for _ in range(4)]).mean(0)
        logits = adapted(x)
        loss = -(pseudo * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            for tp,ap in zip(teacher.parameters(), adapted.parameters()):
                tp.data = 0.999*tp.data + 0.001*ap.data
            for (_,pa),(_,ps) in zip(adapted.named_parameters(), src.named_parameters()):
                if pa.requires_grad:
                    mask = torch.rand_like(pa) < 0.01
                    pa.data[mask] = ps.data[mask]
        return logits
    return fn

# RoTTA
def make_rotta(source):
    student = copy.deepcopy(source)
    student.train(); student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True
            m.momentum = 0.05
    params = [p for m in student.modules() if isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d)) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    teacher = copy.deepcopy(source).eval(); teacher.requires_grad_(False)
    per_class = max(1, ROTTA_N // NUM_CLASSES)   # 1 for C=100
    bank = {c:[] for c in range(NUM_CLASSES)}
    age = [0]
    @torch.enable_grad()
    def fn(x):
        logits = student(x)
        plabels = logits.argmax(1).detach()
        ents = softmax_entropy(logits).detach()
        with torch.no_grad():
            for i,(c,e) in enumerate(zip(plabels.tolist(), ents.tolist())):
                entry = (x[i].detach().cpu(), e, age[0])
                if len(bank[c]) < per_class: bank[c].append(entry)
                else:
                    worst = max(range(len(bank[c])), key=lambda j: bank[c][j][1])
                    if e < bank[c][worst][1]: bank[c][worst] = entry
            age[0] += 1
        samples, ages_list = [], []
        for c in range(NUM_CLASSES):
            if bank[c]:
                for entry in sorted(bank[c], key=lambda e:-e[2])[:per_class]:
                    samples.append(entry[0]); ages_list.append(entry[2])
        if len(samples) >= 2:
            mem_x = torch.stack(samples).to(DEVICE)
            ages_t = torch.tensor(ages_list, dtype=torch.float32, device=DEVICE)
            e_age = torch.exp(-ages_t/ROTTA_N) / (1+torch.exp(-ages_t/ROTTA_N))
            with torch.no_grad(): t_probs = teacher(mem_x).softmax(1)
            s_logits = student(mem_x)
            ce = -(t_probs * s_logits.log_softmax(1)).sum(1) / NUM_CLASSES
            loss = (e_age * ce).mean()
            loss.backward()
            opt.step(); opt.zero_grad()
            with torch.no_grad():
                for tp,sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = (1-ROTTA_NU)*tp.data + ROTTA_NU*sp.data
        return logits
    return fn

# SAR
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

# ContinualTTA (JS gate + entropy filter + confidence gate + adaptation budget)
def make_ctta(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    reference = [None]
    n_backward = [0]

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        # JS gate
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                return logits
            m    = 0.5*(reference[0] + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0), reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0), p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5*(kl_1 + kl_2)
            reference[0] = 0.9*reference[0] + 0.1*p_t
            if js.item() <= JS_THRESHOLD:
                return logits

        # Budget check
        if n_backward[0] >= ADAPT_BUDGET:
            return logits

        # Combined filter
        with torch.no_grad():
            entropy = softmax_entropy(logits)
            conf    = logits.softmax(1).max(1).values
        reliable = (entropy < E_MARGIN) & (conf > MIN_CONF)
        if reliable.sum() == 0:
            return logits

        logits_rel = model(x[reliable])
        softmax_entropy(logits_rel).mean().backward()
        opt.step(); opt.zero_grad()
        n_backward[0] += 1
        return logits

    def reset_budget():
        n_backward[0] = 0
        reference[0] = None   # fresh reference for new corruption

    fn.reset_budget = reset_budget
    return fn

# ------------------------- BUILD METHOD -------------------------
def build_method(method, source, fisher_loader=None):
    if method == "Baseline":     return make_baseline(source)
    if method == "TENT":         return make_tent(source)
    if method == "EATA":         return make_eata(source, fisher_loader)
    if method == "CoTTA":        return make_cotta(source)
    if method == "RoTTA":        return make_rotta(source)
    if method == "SAR":          return make_sar(source)
    if method == "ContinualTTA": return make_ctta(source)
    raise ValueError(f"Unknown method: {method}")

# ------------------------- TRULY CONTINUAL LOOP -------------------------
def run_truly_continual(method, source):
    print(f"\n{'='*60}")
    print(f"Method: {method}  —  Truly Continual (no reset)")
    print(f"{'='*60}")

    # Build ONCE, never reset
    fisher_loader = get_loader(ALL_CORRUPTIONS[0]) if method == "EATA" else None
    fn = build_method(method, source, fisher_loader)
    if fisher_loader is not None:
        del fisher_loader; torch.cuda.empty_cache()

    results = {}
    for corruption in ALL_CORRUPTIONS:
        # Reset budget/reference before each corruption (only for ContinualTTA)
        if hasattr(fn, 'reset_budget'):
            fn.reset_budget()
        loader = get_loader(corruption)
        acc = eval_loader(fn, loader)
        results[corruption] = acc
        del loader; torch.cuda.empty_cache()
        print(f"  {corruption:<24} {acc:.1f}%")
        if acc < 5.0 and method != "Baseline":
            print("  !! WARNING: possible collapse")

    mean_acc = np.mean(list(results.values()))
    print(f"  Mean: {mean_acc:.1f}%")
    return results, mean_acc

# ------------------------- SAVE & TABLE -------------------------
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
            if len(parts)==2 and parts[0]!="Mean":
                results[parts[0]] = float(parts[1])
    return results, np.mean(list(results.values()))

def generate_table(all_results, all_means):
    present = [m for m in METHODS if m in all_means]
    print(f"\n{'='*60}")
    print("TRULY CONTINUAL CIFAR-100-C (BN, S5) — Mean Accuracy")
    print(f"{'='*60}")
    for m in present:
        print(f"{m:<15} {all_means[m]:.1f}%")

    cite = {
        "Baseline":"Baseline","TENT":"TENT","EATA":"EATA","CoTTA":"CoTTA",
        "RoTTA":"RoTTA","SAR":"SAR","ContinualTTA":"\\textbf{ContinualTTA (Ours)}"
    }
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{CIFAR-100-C (severity 5) truly continual mean accuracy (\%), ResNet-50 BN, with adaptation budget.}")
    latex.append(r"\label{tab:cifar100_continual_budget}")
    latex.append(r"\begin{tabular}{lc}")
    latex.append(r"\toprule")
    latex.append(r"Method & Mean (\%) \\")
    latex.append(r"\midrule")
    best = max(all_means.values())
    for m in present:
        cell = f"{all_means[m]:.1f}"
        if abs(all_means[m]-best)<0.01: cell = f"\\textbf{{{cell}}}"
        latex.append(f"{cite.get(m,m)} & {cell} \\\\")
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    tex_path = os.path.join(RESULTS_DIR, "table_continual_budget.tex")
    with open(tex_path, "w") as f: f.write("\n".join(latex))
    print(f"  LaTeX: {tex_path}")

# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--table_only", action="store_true")
    args = parser.parse_args()

    if args.table_only:
        all_results, all_means = {}, {}
        for m in METHODS:
            r, mean = load_csv(m)
            if r: all_results[m]=r; all_means[m]=mean; print(f"  Loaded {m}: {mean:.1f}%")
        if all_results: generate_table(all_results, all_means)
        exit(0)

    # Data check
    for c in ALL_CORRUPTIONS[:2]:
        assert os.path.isfile(f"{DATA_DIR}/{c}.npy"), f"Missing {c}.npy"
    print("Data check passed.\n")

    source = load_model()
    print(f"Source model loaded: {sum(p.numel() for p in source.parameters()):,} params\n")

    # Sanity baseline
    print("Sanity check — Baseline on gaussian_noise S5...")
    _ldr = get_loader("gaussian_noise")
    _m = copy.deepcopy(source).eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in _ldr:
            x,y = x.to(DEVICE), y.to(DEVICE)
            correct += (_m(x).argmax(1)==y).sum().item()
            total += y.size(0)
    print(f"Baseline: {100*correct/total:.1f}%\n")
    del _ldr, _m; torch.cuda.empty_cache()

    all_results, all_means = {}, {}
    for method in args.methods:
        if args.skip_done and load_csv(method)[0] is not None:
            print(f"Skipping {method} (already done)")
            continue
        res, mean = run_truly_continual(method, source)
        all_results[method] = res
        all_means[method] = mean
        save_csv(method, res)
        print(f"→ {method} mean: {mean:.1f}%\n")
        torch.cuda.empty_cache()

    if all_results:
        generate_table(all_results, all_means)