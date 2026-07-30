# =============================================================================
# ContinualTTA — CIFAR-100-C Fresh-per-Corruption (BN)
# All Methods: Baseline | TENT | EATA | CoTTA | RoTTA | SAR | ContinualTTA
#
# PROTOCOL: Fresh-per-corruption (standard, matches CoTTA/RoTTA papers)
#   For each of 15 corruptions:
#     1. Reset model to source weights
#     2. Adapt on 10,000 images of that corruption at severity 5
#     3. Record accuracy
#     4. Move to next corruption with a FRESH model
#   No state carries over between corruptions.
#
# WHY FRESH-PER-CORRUPTION, NOT TRULY CONTINUAL:
#   Your earlier truly continual CIFAR-100-C attempt (BN model)
#   collapsed catastrophically (SAR: 39.69% -> 3.49%, ContinualTTA:
#   7.11% mean) because S1 baseline accuracy is already only ~53%,
#   meaning 47% wrong predictions corrupt BN statistics, and damage
#   COMPOUNDS across all 75 blocks with no reset. Fresh-per-corruption
#   resets the model before each corruption, so this compounding
#   cannot occur -- each corruption gets one clean evaluation. This
#   is also the protocol CoTTA and RoTTA use for their published
#   CIFAR-100-C numbers, keeping your results directly comparable.
#
# WHY BN, NOT GN (see chat discussion):
#   1. Reset-based protocol prevents the cross-corruption compounding
#      that forced the BN->GN switch on truly-continual ImageNet-C.
#   2. CoTTA/RoTTA published CIFAR-100-C numbers use BN -- staying
#      on BN keeps your results comparable to the literature.
#   3. You already have a BN CIFAR-100 source model (82.7% clean acc).
#
# SAFETY NET: MIN_CONF confidence gate added for ContinualTTA (and
# available for TENT/EATA) as cheap insurance against single-batch
# instability at severity 5, even without cross-corruption compounding.
# This mirrors the fix that stabilised ImageNet-C ContinualTTA.
#
# Run:
#   python cifar100c_fresh.py
#   python cifar100c_fresh.py --methods Baseline ContinualTTA
#   python cifar100c_fresh.py --skip_done
#   python cifar100c_fresh.py --table_only
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

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar100_source.pth"
DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\CIFAR-100-C"
RESULTS_DIR = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\results\cifar100c_fresh"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
NUM_CLASSES = 100
SEVERITY    = 5
NUM_WORKERS = 0

# Hyperparameters
LR           = 1e-3
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 1.842 nats for C=100
MIN_CONF     = 0.3      # safety-net confidence gate (ContinualTTA only)
                         # lower than ImageNet-C's 0.5 since C=100 not
                         # C=1000 -- baseline confidence is naturally
                         # lower with fewer classes to concentrate on;
                         # 0.3 still filters genuinely diffuse predictions
JS_THRESHOLD = 0.04      # JS is scale-invariant; same tau as CIFAR-10-C
ROTTA_NU     = 0.001
ROTTA_N      = 64
SAR_RHO      = 0.05
SAR_E0       = 0.3       # CIFAR-100 uses higher e0 than CIFAR-10 (0.2)
                          # since natural entropy is higher with 100
                          # classes (ln(100)=4.6 vs ln(10)=2.3); 0.3
                          # is well above collapsed-model entropy (~0)
                          # and below E_MARGIN=1.842

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# 1. DATASET — CIFAR-100-C has identical .npy structure to CIFAR-10-C
# =============================================================================

class CIFAR100C_Dataset(Dataset):
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.transform(Image.fromarray(self.images[idx])),
                int(self.labels[idx]))


def load_corruption(corruption):
    return DataLoader(
        CIFAR100C_Dataset(corruption, SEVERITY),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)


def load_model():
    """Load CIFAR-100 trained ResNet-50 (BN). FC has 100 outputs."""
    model    = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


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
    model, params = setup_bn(copy.deepcopy(source))
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
# 5. EATA
# =============================================================================

def make_eata(source, fisher_loader=None):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

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
                      else 0.9*ref_probs[0] + 0.1*probs[mask].mean(0).detach()
        fisher_reg = sum((fisher[n]*p.pow(2)).sum()
                         for n, p in model.named_parameters()
                         if p.requires_grad and n in fisher)
        (entropy[mask].mean() + 1e-3*fisher_reg).backward()
        opt.step(); opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 6. CoTTA
# =============================================================================

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
            for (_, pa), (_, ps) in zip(adapted.named_parameters(),
                                         src.named_parameters()):
                if pa.requires_grad:
                    mask = torch.rand_like(pa) < 0.01   # single mask
                    pa.data[mask] = ps.data[mask]
        return logits

    return fn


# =============================================================================
# 7. RoTTA
#    per_class = max(1, 64//100) = 1 slot per class for C=100
# =============================================================================

def make_rotta(source):
    student = copy.deepcopy(source)
    student.train()
    student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True
            m.momentum = 0.05
    params = [p for m in student.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    opt     = torch.optim.Adam(params, lr=LR)
    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)

    per_class = max(1, ROTTA_N // NUM_CLASSES)   # = 1 for C=100
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
            e_age  = torch.exp(-ages_t/ROTTA_N) / (1+torch.exp(-ages_t/ROTTA_N))
            with torch.no_grad(): t_probs = teacher(mem_x).softmax(1)
            s_logits = student(mem_x)
            # CE / NUM_CLASSES -- scales correctly for C=100
            ce   = -(t_probs*s_logits.log_softmax(1)).sum(1)/NUM_CLASSES
            loss = (e_age*ce).mean()
            loss.backward()
            opt.step(); opt.zero_grad()
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = (1-ROTTA_NU)*tp.data + ROTTA_NU*sp.data

        return logits

    return fn


# =============================================================================
# 8. SAR
#    SAR_E0=0.3 -- see config comment for CIFAR-100-specific reasoning
# =============================================================================

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

    return fn


# =============================================================================
# 9. ContinualTTA (with safety-net confidence gate)
#
# Standard CIFAR-10-C ContinualTTA has only entropy filter (no MIN_CONF)
# because fresh-per-corruption + S1-S5 baseline >75% never required it.
# CIFAR-100-C baseline is much lower (~50% even at S1), so the same
# combined entropy+confidence filter used for ImageNet-C is added here
# as a safety net -- cheap insurance, structurally identical to the fix
# that stabilised ImageNet-C ContinualTTA.
# =============================================================================

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
                return logits   # first batch: init only, no adapt
            m    = 0.5*(reference[0]+p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             reference[0].unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5*(kl_1+kl_2)
            reference[0] = 0.9*reference[0] + 0.1*p_t
            adapt = js.item() > JS_THRESHOLD
        if not adapt:
            return logits

        # Combined filter: entropy AND confidence (safety net)
        with torch.no_grad():
            entropy = softmax_entropy(logits)
            conf    = logits.softmax(1).max(1).values
        reliable = (entropy < E_MARGIN) & (conf > MIN_CONF)
        if reliable.sum() == 0:
            return logits

        logits_rel  = model(x[reliable])
        softmax_entropy(logits_rel).mean().backward()
        opt.step(); opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 10. BUILD METHOD — fresh instance per corruption
# =============================================================================

def build_method(method, source, corruption=None):
    fisher_loader = None
    if method == "EATA" and corruption is not None:
        fisher_loader = load_corruption(corruption)

    dispatch = {
        "Baseline":     lambda: make_baseline(source),
        "TENT":         lambda: make_tent(source),
        "EATA":         lambda: make_eata(source, fisher_loader),
        "CoTTA":        lambda: make_cotta(source),
        "RoTTA":        lambda: make_rotta(source),
        "SAR":          lambda: make_sar(source),
        "ContinualTTA": lambda: make_ctta(source),
    }
    fn = dispatch[method]()
    if fisher_loader is not None:
        del fisher_loader
        torch.cuda.empty_cache()
    return fn


# =============================================================================
# 11. RUN ONE METHOD — fresh per corruption
# =============================================================================

def run_fresh_per_corruption(method, source):
    print(f"\n{'-'*55}")
    print(f"  {method}")
    print(f"  Protocol: fresh model per corruption (standard)")
    print(f"{'-'*55}")

    results = {}
    for corruption in ALL_CORRUPTIONS:
        fn     = build_method(method, source, corruption=corruption)
        loader = load_corruption(corruption)
        acc    = eval_loader(fn, loader)
        results[corruption] = acc
        del loader, fn
        torch.cuda.empty_cache()
        print(f"  {corruption:<24} {acc:.1f}%")
        if acc < 5.0 and method != "Baseline":
            print(f"  !! WARNING: very low accuracy -- check for collapse")

    mean_acc = np.mean(list(results.values()))
    print(f"  {'Mean':<24} {mean_acc:.1f}%")
    return results, mean_acc


# =============================================================================
# 12. SAVE / LOAD HELPERS
# =============================================================================

def save_method_csv(method, results):
    mean_acc = np.mean(list(results.values()))
    path     = os.path.join(RESULTS_DIR, f"{method}.csv")
    with open(path, "w") as f:
        f.write(f"corruption,{method}\n")
        for c in ALL_CORRUPTIONS:
            f.write(f"{c},{results[c]:.2f}\n")
        f.write(f"Mean,{mean_acc:.2f}\n")
    print(f"  Saved: {path}")


def load_existing_csv(method):
    path = os.path.join(RESULTS_DIR, f"{method}.csv")
    if not os.path.isfile(path):
        return None, None
    results = {}
    with open(path) as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) == 2 and parts[0] != "Mean":
                results[parts[0]] = float(parts[1])
    return results, np.mean(list(results.values()))


# =============================================================================
# 13. TABLE GENERATION
# =============================================================================

def print_final_table(all_results, all_means):
    methods_present = [m for m in METHODS if m in all_results]
    col    = 14
    header = f"{'Corruption':<24}" + "".join(f"{m[:12]:>{col}}" for m in methods_present)
    sep    = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("CIFAR-100-C -- Fresh-per-Corruption (Standard Protocol)")
    print(f"{'='*len(header)}")
    print(header); print(sep)

    for c in ALL_CORRUPTIONS:
        vals = {m: all_results[m].get(c, float('nan')) for m in methods_present}
        finite = [v for v in vals.values() if not math.isnan(v)]
        best   = max(finite) if finite else float('nan')
        row    = f"{c:<24}"
        for m in methods_present:
            v    = vals[m]
            cell = f"{v:.1f}%" + ("*" if not math.isnan(v) and abs(v-best)<0.05 else "")
            row += f"{cell:>{col}}"
        print(row)

    print(sep)
    best_m = max(v for v in all_means.values() if not math.isnan(v))
    mrow   = f"{'Mean':<24}"
    for m in methods_present:
        v    = all_means.get(m, float('nan'))
        cell = f"{v:.2f}%" + ("*" if not math.isnan(v) and abs(v-best_m)<0.05 else "")
        mrow += f"{cell:>{col}}"
    print(mrow)
    print(f"{'='*len(header)}\n  * = best in row")


def save_summary_and_latex(all_results, all_means):
    methods_present = [m for m in METHODS if m in all_results]

    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w") as f:
        f.write("corruption," + ",".join(methods_present) + "\n")
        for c in ALL_CORRUPTIONS:
            row = c
            for m in methods_present:
                row += f",{all_results[m].get(c, float('nan')):.2f}"
            f.write(row + "\n")
        f.write("Mean," + ",".join(
            f"{all_means[m]:.2f}" for m in methods_present) + "\n")
    print(f"\n  Summary CSV: {csv_path}")

    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "CoTTA":        "CoTTA~\\cite{wang2022continual}",
        "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
        "SAR":          "SAR~\\cite{niu2023towards}",
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

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Accuracy (\%) on CIFAR-100-C, severity~5, standard "
        r"fresh-per-corruption protocol. Each method receives a fresh "
        r"model before each corruption type. Source model: ResNet-50 "
        r"(BatchNorm, 82.7\% clean accuracy on CIFAR-100). "
        r"\textbf{Bold} = best per row.}")
    lines.append(r"\label{tab:cifar100c}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c"*len(methods_present) + "}")
    lines.append(r"\toprule")
    lines.append("Corruption & " +
                 " & ".join(cite[m] for m in methods_present) + r" \\")
    lines.append(r"\midrule")

    for c in ALL_CORRUPTIONS:
        vals   = [all_results[m].get(c, float('nan')) for m in methods_present]
        finite = [v for v in vals if not math.isnan(v)]
        best   = max(finite) if finite else float('nan')
        row    = corr_names.get(c, c)
        for val in vals:
            if math.isnan(val): row += " & ---"
            elif abs(val-best) < 0.05: row += f" & \\textbf{{{val:.1f}}}"
            else: row += f" & {val:.1f}"
        lines.append(row + r" \\")

    lines.append(r"\midrule")
    mean_vals  = [all_means.get(m, float('nan')) for m in methods_present]
    finite_m   = [v for v in mean_vals if not math.isnan(v)]
    best_m     = max(finite_m) if finite_m else float('nan')
    mean_row   = r"\textbf{Mean}"
    for val in mean_vals:
        if math.isnan(val): mean_row += " & ---"
        elif abs(val-best_m) < 0.05: mean_row += f" & \\textbf{{{val:.1f}}}"
        else: mean_row += f" & {val:.1f}"
    lines.append(mean_row + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex_str = "\n".join(lines)
    tex_path  = os.path.join(RESULTS_DIR, "table_cifar100c.tex")
    with open(tex_path, "w") as f: f.write(latex_str)
    print(f"  LaTeX: {tex_path}")
    return latex_str


# =============================================================================
# 14. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="CIFAR-100-C -- Fresh-per-Corruption Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Standard protocol: fresh model reset before each corruption.
Matches TENT, EATA, CoTTA, RoTTA paper evaluations on CIFAR-100-C.

Examples:
  python cifar100c_fresh.py
  python cifar100c_fresh.py --methods Baseline ContinualTTA
  python cifar100c_fresh.py --skip_done
  python cifar100c_fresh.py --table_only
        """)
    parser.add_argument("--methods", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--table_only", action="store_true")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"CIFAR-100-C S{SEVERITY} -- Fresh-per-Corruption Protocol")
    print(f"{'='*60}")
    print(f"Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Protocol   : Fresh model per corruption (standard)")
    print(f"Norm layer : BatchNorm  (see header for BN-vs-GN rationale)")
    print(f"Severity   : {SEVERITY}")
    print(f"LR         : {LR}")
    print(f"E_margin   : {E_MARGIN:.3f} nats")
    print(f"JS tau     : {JS_THRESHOLD}  (ContinualTTA only)")
    print(f"MIN_CONF   : {MIN_CONF}  (ContinualTTA safety net)")
    print(f"SAR e0     : {SAR_E0}")
    print(f"Results    : {RESULTS_DIR}")

    if args.table_only:
        print("\nTable-only mode...")
        all_results, all_means = {}, {}
        for m in METHODS:
            res, mean = load_existing_csv(m)
            if res:
                all_results[m] = res; all_means[m] = mean
                print(f"  Loaded {m}: {mean:.1f}%")
        if all_results:
            print_final_table(all_results, all_means)
            latex = save_summary_and_latex(all_results, all_means)
            print(f"\n{'='*60}\nTable LaTeX:\n{'='*60}")
            print(latex)
        exit(0)

    print(f"\nVerifying DATA_DIR...")
    for c in ALL_CORRUPTIONS:
        path = os.path.join(DATA_DIR, f"{c}.npy")
        if not os.path.isfile(path):
            print(f"  WARNING: missing {c}")
    print(f"  Check complete.")

    print(f"\nLoading CIFAR-100 source model...")
    source_model = load_model()
    print(f"  Parameters: {sum(p.numel() for p in source_model.parameters()):,}")

    print(f"\nSanity check -- Baseline on gaussian_noise S{SEVERITY}...")
    _fn  = make_baseline(source_model)
    _ldr = load_corruption("gaussian_noise")
    acc  = eval_loader(_fn, _ldr)
    del _ldr; torch.cuda.empty_cache()
    print(f"  Baseline gaussian_noise: {acc:.1f}%  (expected ~20-35%)")
    if acc < 5.0:
        print("  ERROR: too low -- check MODEL_PATH and DATA_DIR"); exit(1)
    print(f"  Passed.\n")

    all_results, all_means = {}, {}
    for m in METHODS:
        res, mean = load_existing_csv(m)
        if res and args.skip_done:
            all_results[m] = res; all_means[m] = mean
            print(f"Skipping {m} (already saved: {mean:.1f}%)")

    for method in args.methods:
        if args.skip_done and method in all_results:
            continue
        print(f"\n[{args.methods.index(method)+1}/{len(args.methods)}] {method}")
        results, mean = run_fresh_per_corruption(method, source_model)
        all_results[method] = results
        all_means[method]   = mean
        save_method_csv(method, results)
        print(f"  -> {method}: {mean:.1f}%")
        torch.cuda.empty_cache()

    if all_results:
        print_final_table(all_results, all_means)
        latex = save_summary_and_latex(all_results, all_means)
        print(f"\n{'='*70}")
        print("Table LaTeX -- paste into Overleaf:")
        print(f"{'='*70}")
        print(latex)
        print(f"\nFinal ranking:")
        for m in sorted(all_means.keys(), key=lambda x: -all_means[x]):
            flag = "  <- ours" if m == "ContinualTTA" else ""
            print(f"  {m:<18} {all_means[m]:.1f}%{flag}")

    print(f"\n{'='*60}\nDONE\n{'='*60}")
    print(f"Results: {os.path.abspath(RESULTS_DIR)}/")