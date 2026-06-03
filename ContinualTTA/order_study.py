# =============================================================================
# ContinualTTA — Corruption Order Study (WACV 2027)
#
# Tests whether ContinualTTA is robust to the ORDER of corruptions,
# not just the sequential protocol itself.
#
# Three orderings:
#   forward  — gaussian_noise → ... → jpeg_compression (standard)
#   reverse  — jpeg_compression → ... → gaussian_noise
#   random   — seeded random permutation (3 seeds for variance estimate)
#
# Run from terminal:
#   python order_study.py --method ContinualTTA --dataset cifar10c
#   python order_study.py --method RoTTA       --dataset cifar10c
#   python order_study.py --method TENT        --dataset cifar10c
#   python order_study.py --method SAR         --dataset cifar10c
#   python order_study.py --method Baseline    --dataset cifar10c
#   python order_study.py --method ContinualTTA --dataset imagenetc
#
#   After all methods done:
#   python order_study.py --merge
#
# Output per run:
#   results/order_study/{method}_{order}_{seed}.csv
#
# Output after merge:
#   results/order_study/order_table.tex
#   results/order_study/order_summary.csv
# =============================================================================

import os
import copy
import math
import argparse
import platform
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from PIL import Image

# =============================================================================
# CONFIG
# =============================================================================

CIFAR_MODEL_PATH  = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
CIFAR_DATA_DIR    = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
IMAGENET_DATA_DIR = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\ImageNet-C"

RESULTS_DIR  = os.path.join("results", "order_study")
NUM_WORKERS  = 0 if platform.system() == "Windows" else 2
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR_BATCH_SIZE    = 32
CIFAR_CLASSES       = 10
CIFAR_SEVERITIES    = [1, 2, 3, 4, 5]

IMAGENET_BATCH_SIZE = 64
IMAGENET_CLASSES    = 1000
IMAGENET_SEVERITY   = 5

LR           = 1e-3
EMA_DECAY    = 0.9
E_MARGIN_FACTOR = 0.4
ALPHA        = 0.5
JS_THRESHOLD = 0.04
ROTTA_NU     = 0.001
ROTTA_N      = 64
SAR_RHO      = 0.05

# Standard fixed corruption order (Setting B)
FORWARD_ORDER = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

REVERSE_ORDER = list(reversed(FORWARD_ORDER))

# Three random seeds for variance estimation
RANDOM_SEEDS = [0, 1, 2]

METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]


def get_random_order(seed):
    rng = random.Random(seed)
    order = FORWARD_ORDER.copy()
    rng.shuffle(order)
    return order


# =============================================================================
# 1. DATASETS
# =============================================================================

class CIFAR10C_Dataset(Dataset):
    def __init__(self, corruption, severity):
        data        = np.load(os.path.join(CIFAR_DATA_DIR, f"{corruption}.npy"), mmap_mode='r')
        labels      = np.load(os.path.join(CIFAR_DATA_DIR, "labels.npy"),        mmap_mode='r')
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
        return self.transform(Image.fromarray(self.images[idx])), int(self.labels[idx])


def cifar_loader(corruption, severity):
    return DataLoader(CIFAR10C_Dataset(corruption, severity),
                      batch_size=CIFAR_BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def imagenet_loader(corruption):
    path = os.path.join(IMAGENET_DATA_DIR, corruption, str(IMAGENET_SEVERITY))
    dataset = ImageFolder(path,
        transform=models.ResNet50_Weights.IMAGENET1K_V1.transforms())
    return DataLoader(dataset, batch_size=IMAGENET_BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def load_cifar_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, CIFAR_CLASSES)
    model.load_state_dict(torch.load(CIFAR_MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def load_imagenet_model():
    return models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1).to(DEVICE).eval()


# =============================================================================
# 2. HELPERS
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


def setup_bn_cifar(model):
    model.train(); model.requires_grad_(False)
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


def setup_bn_imagenet(model):
    model.train(); model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True
            m.momentum = 0
    params = [p for m in model.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    return model, params


# =============================================================================
# 3. METHOD IMPLEMENTATIONS
# =============================================================================

def make_baseline(source, setup_bn_fn):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad(): return model(x)
    return fn


def make_tent(source, setup_bn_fn):
    model, params = setup_bn_fn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        softmax_entropy(logits).mean().backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn


def make_eata(source, setup_bn_fn, num_classes, fisher_loader=None):
    e_margin = E_MARGIN_FACTOR * math.log(num_classes)
    model, params = setup_bn_fn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    fisher = {n: torch.zeros_like(p)
              for n, p in model.named_parameters() if p.requires_grad}
    if fisher_loader is not None:
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
    @torch.enable_grad()
    def fn(x):
        logits  = model(x)
        entropy = softmax_entropy(logits)
        probs   = logits.softmax(1)
        mask_e  = entropy < e_margin
        if ref_probs[0] is not None:
            cos_sim = F.cosine_similarity(
                ref_probs[0].unsqueeze(0).expand(probs.size(0), -1), probs, dim=1)
            mask_d = cos_sim < 0.95
        else:
            mask_d = torch.ones(probs.size(0), dtype=torch.bool, device=DEVICE)
        mask = mask_e & mask_d
        if mask.sum() == 0: return logits
        with torch.no_grad():
            if ref_probs[0] is None: ref_probs[0] = probs[mask].mean(0).detach()
            else: ref_probs[0] = 0.9 * ref_probs[0] + 0.1 * probs[mask].mean(0).detach()
        fisher_reg = sum((fisher[n] * p.pow(2)).sum()
                         for n, p in model.named_parameters()
                         if p.requires_grad and n in fisher)
        (entropy[mask].mean() + 1e-3 * fisher_reg).backward()
        opt.step(); opt.zero_grad()
        return logits
    return fn


def make_cotta(source, setup_bn_fn):
    src = copy.deepcopy(source).eval(); src.requires_grad_(False)
    adapted, params = setup_bn_fn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)
    teacher = copy.deepcopy(source).eval(); teacher.requires_grad_(False)
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ])
    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            pseudo = torch.stack([teacher(aug(x)).softmax(1) for _ in range(4)]).mean(0)
        logits = adapted(x)
        (-(pseudo * logits.log_softmax(1)).sum(1).mean()).backward()
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            for tp, ap in zip(teacher.parameters(), adapted.parameters()):
                tp.data = 0.999 * tp.data + 0.001 * ap.data
            for (_, pa), (_, ps) in zip(adapted.named_parameters(), src.named_parameters()):
                if pa.requires_grad:
                    pa.data[torch.rand_like(pa) < 0.01] = ps.data[torch.rand_like(pa) < 0.01]
        return logits
    return fn


def make_rotta(source, setup_bn_fn, num_classes):
    student = copy.deepcopy(source)
    student.train(); student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True); m.track_running_stats = True; m.momentum = 0.05
    params = [p for m in student.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    teacher = copy.deepcopy(source).eval(); teacher.requires_grad_(False)
    per_class = max(1, ROTTA_N // num_classes)
    bank = {c: [] for c in range(num_classes)}
    age  = [0]
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
        for c in range(num_classes):
            if bank[c]:
                for entry in sorted(bank[c], key=lambda e: -e[2])[:per_class]:
                    samples.append(entry[0]); ages_list.append(entry[2])
        if len(samples) >= 2:
            mem_x  = torch.stack(samples)
            ages_t = torch.tensor(ages_list, dtype=torch.float32, device=DEVICE)
            BANK_BATCH = 32
            total_loss = torch.tensor(0.0, device=DEVICE); n_mini = 0
            for start in range(0, len(samples), BANK_BATCH):
                end    = min(start + BANK_BATCH, len(samples))
                mb_x   = mem_x[start:end].to(DEVICE)
                mb_age = ages_t[start:end]
                e_age  = torch.exp(-mb_age / ROTTA_N) / (1 + torch.exp(-mb_age / ROTTA_N))
                with torch.no_grad(): t_probs = teacher(mb_x).softmax(1)
                s_logits = student(mb_x)
                ce = -(t_probs * s_logits.log_softmax(1)).sum(1) / num_classes
                total_loss = total_loss + (e_age * ce).mean(); n_mini += 1
            (total_loss / n_mini).backward()
            opt.step(); opt.zero_grad()
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = (1 - ROTTA_NU) * tp.data + ROTTA_NU * sp.data
        return logits
    return fn


def make_sar(source, setup_bn_fn, num_classes):
    e_margin = E_MARGIN_FACTOR * math.log(num_classes)
    model, params = setup_bn_fn(copy.deepcopy(source))
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
    ema_entropy = [None]
    @torch.enable_grad()
    def fn(x):
        with torch.no_grad():
            logits_init  = model(x)
            entropy_init = softmax_entropy(logits_init)
        if ema_entropy[0] is None: ema_entropy[0] = entropy_init.mean().item()
        thresh   = min(e_margin, ema_entropy[0] + 0.4 * math.log(num_classes))
        reliable = entropy_init < thresh
        if reliable.sum() == 0: return logits_init
        x_rel = x[reliable]
        softmax_entropy(model(x_rel)).mean().backward()
        grad_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))
        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
                p.data.add_(e_w); e_ws.append(e_w); p.grad.zero_()
            else: e_ws.append(None)
        logits_2  = model(x_rel)
        entropy_2 = softmax_entropy(logits_2)
        if (entropy_2 < e_margin).sum() > 0:
            entropy_2[entropy_2 < e_margin].mean().backward()
        for p, e_w in zip(params, e_ws):
            if e_w is not None: p.data.sub_(e_w)
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            logits_out  = model(x)
            ema_entropy[0] = 0.9 * ema_entropy[0] + 0.1 * softmax_entropy(logits_out).mean().item()
        return logits_out
    return fn


class PrototypeBankModule(nn.Module):
    def __init__(self, num_classes, feat_dim=2048, decay=EMA_DECAY):
        super().__init__()
        self.decay = decay
        self.register_buffer("prototypes",  torch.zeros(num_classes, feat_dim))
        self.register_buffer("initialised", torch.zeros(num_classes).bool())
    @torch.no_grad()
    def update(self, features, pseudo_labels):
        for c in pseudo_labels.unique():
            mask = (pseudo_labels == c); mf = features[mask].mean(0)
            if self.initialised[c]:
                self.prototypes[c] = self.decay * self.prototypes[c] + (1-self.decay) * mf
            else:
                self.prototypes[c] = mf; self.initialised[c] = True
    def consistency_loss(self, features, pseudo_labels):
        loss, count = torch.tensor(0.0, device=features.device), 0
        for c in pseudo_labels.unique():
            if not self.initialised[c]: continue
            mask  = (pseudo_labels == c)
            loss += F.mse_loss(features[mask],
                               self.prototypes[c].unsqueeze(0).expand(mask.sum(), -1))
            count += 1
        return loss / max(count, 1)


def make_ctta(source, setup_bn_fn, num_classes):
    e_margin = E_MARGIN_FACTOR * math.log(num_classes)
    model, params = setup_bn_fn(copy.deepcopy(source))
    bank     = PrototypeBankModule(num_classes).to(DEVICE)
    opt      = torch.optim.Adam(params, lr=LR)
    captured = {}
    reference  = [None]
    model.avgpool.register_forward_hook(
        lambda m, i, o: captured.update({"feat": o.flatten(1)}))

    @torch.enable_grad()
    def fn(x):
        logits        = model(x)
        features      = captured["feat"]
        pseudo_labels = logits.argmax(1).detach()
        # JS detector
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if reference[0] is None:
                reference[0] = p_t.clone()
                adapt = True
            else:
                m    = 0.5 * (reference[0] + p_t)
                kl_1 = F.kl_div(m.log().unsqueeze(0), reference[0].unsqueeze(0), reduction="batchmean")
                kl_2 = F.kl_div(m.log().unsqueeze(0), p_t.unsqueeze(0),          reduction="batchmean")
                adapt = (0.5 * (kl_1 + kl_2)).item() > JS_THRESHOLD
                reference[0] = 0.9 * reference[0] + 0.1 * p_t
        if not adapt: return logits
        entropy  = softmax_entropy(logits)
        reliable = entropy < e_margin
        if reliable.sum() == 0: return logits
        loss = (entropy[reliable].mean()
                + ALPHA * bank.consistency_loss(
                    features[reliable].detach(), pseudo_labels[reliable]))
        loss.backward(); opt.step(); opt.zero_grad()
        bank.update(features[reliable].detach(), pseudo_labels[reliable])
        return logits
    return fn


def build_method(method, source, setup_bn_fn, num_classes, fisher_loader=None):
    if method == "Baseline":     return make_baseline(source, setup_bn_fn)
    if method == "TENT":         return make_tent(source, setup_bn_fn)
    if method == "EATA":         return make_eata(source, setup_bn_fn, num_classes, fisher_loader)
    if method == "CoTTA":        return make_cotta(source, setup_bn_fn)
    if method == "RoTTA":        return make_rotta(source, setup_bn_fn, num_classes)
    if method == "SAR":          return make_sar(source, setup_bn_fn, num_classes)
    if method == "ContinualTTA": return make_ctta(source, setup_bn_fn, num_classes)
    raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 4. ORDER STUDY RUNNER
# =============================================================================

def run_order(method, source, setup_bn_fn, num_classes,
              corruption_order, dataset, order_name, seed=None):
    """
    Run one method through one corruption ordering.
    Returns: {corruption: acc, ..., 'Mean': mean_acc}
    """
    label = f"{order_name}" + (f"_seed{seed}" if seed is not None else "")
    print(f"\n  Order: {label}")
    print(f"  Sequence: {' → '.join(c[:6] for c in corruption_order)}")

    # Build fisher loader for EATA from first corruption in this order
    fisher_loader = None
    if method == "EATA":
        if dataset == "cifar10c":
            fisher_loader = cifar_loader(corruption_order[0], 1)
        else:
            fisher_loader = imagenet_loader(corruption_order[0])

    fn      = build_method(method, source, setup_bn_fn, num_classes, fisher_loader)
    results = {}

    for corruption in corruption_order:
        if dataset == "cifar10c":
            # Average over all severities for fair comparison
            accs = []
            for severity in CIFAR_SEVERITIES:
                loader = cifar_loader(corruption, severity)
                accs.append(eval_loader(fn, loader))
                del loader; torch.cuda.empty_cache()
            acc = np.mean(accs)
        else:
            loader = imagenet_loader(corruption)
            acc    = eval_loader(fn, loader)
            del loader; torch.cuda.empty_cache()

        results[corruption] = acc
        print(f"    {corruption:<24} {acc:.1f}%")

    mean_acc = np.mean(list(results.values()))
    results["Mean"] = mean_acc
    print(f"    {'Mean':<24} {mean_acc:.1f}%")
    return results


def save_order_csv(method, order_name, seed, results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"seed{seed}" if seed is not None else "fixed"
    path   = os.path.join(RESULTS_DIR, f"{method}_{order_name}_{suffix}.csv")
    with open(path, "w") as f:
        f.write(f"corruption,acc\n")
        for c, v in results.items():
            f.write(f"{c},{v:.2f}\n")
    print(f"  Saved: {path}")
    return path


# =============================================================================
# 5. MERGE AND LATEX
# =============================================================================

def merge_and_latex(dataset="cifar10c"):
    """
    Merge all saved CSVs into summary table and LaTeX.
    Computes mean ± std for random orderings.
    """
    print("\nMerging order study results...")

    # Collect all results
    # Structure: method -> order -> list of mean_accs
    summary = {}    # method -> {forward, reverse, random_mean, random_std}

    for method in METHODS:
        summary[method] = {}

        # Forward
        path = os.path.join(RESULTS_DIR, f"{method}_forward_fixed.csv")
        if os.path.isfile(path):
            with open(path) as f: lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if parts[0] == "Mean":
                    summary[method]["forward"] = float(parts[1])

        # Reverse
        path = os.path.join(RESULTS_DIR, f"{method}_reverse_fixed.csv")
        if os.path.isfile(path):
            with open(path) as f: lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(",")
                if parts[0] == "Mean":
                    summary[method]["reverse"] = float(parts[1])

        # Random — collect all seeds
        random_accs = []
        for seed in RANDOM_SEEDS:
            path = os.path.join(RESULTS_DIR, f"{method}_random_seed{seed}.csv")
            if os.path.isfile(path):
                with open(path) as f: lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if parts[0] == "Mean":
                        random_accs.append(float(parts[1]))
        if random_accs:
            summary[method]["random_mean"] = np.mean(random_accs)
            summary[method]["random_std"]  = np.std(random_accs)

    # Print console summary
    print(f"\n{'Method':<20} {'Forward':>10} {'Reverse':>10} "
          f"{'Random (mean±std)':>20} {'Range':>10}")
    print("─" * 75)
    for method in METHODS:
        s = summary[method]
        fwd = f"{s['forward']:.1f}%" if "forward" in s else "---"
        rev = f"{s['reverse']:.1f}%" if "reverse" in s else "---"
        if "random_mean" in s:
            rnd = f"{s['random_mean']:.1f}±{s['random_std']:.1f}%"
            rng = f"{s['random_mean']-s['random_std']:.1f}–{s['random_mean']+s['random_std']:.1f}%"
        else:
            rnd = "---"; rng = "---"
        print(f"{method:<20} {fwd:>10} {rev:>10} {rnd:>20} {rng:>10}")

    # Save summary CSV
    csv_path = os.path.join(RESULTS_DIR, "order_summary.csv")
    with open(csv_path, "w") as f:
        f.write("method,forward,reverse,random_mean,random_std\n")
        for method in METHODS:
            s = summary[method]
            f.write(f"{method},"
                    f"{s.get('forward', ''):.2f},"
                    f"{s.get('reverse', ''):.2f},"
                    f"{s.get('random_mean', ''):.2f},"
                    f"{s.get('random_std', ''):.2f}\n")
    print(f"\nSummary CSV saved: {csv_path}")

    # Generate LaTeX
    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "CoTTA":        "CoTTA~\\cite{wang2022continual}",
        "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
        "SAR":          "SAR~\\cite{niu2023towards}",
        "ContinualTTA": "\\textbf{\\textsc{ContinualTTA} (Ours)}",
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Corruption order sensitivity study on CIFAR-10-C "
                 r"(S1--S5 averaged). "
                 r"\emph{Forward}: standard sequential order "
                 r"(gaussian noise $\to$ jpeg). "
                 r"\emph{Reverse}: reversed order. "
                 r"\emph{Random}: mean $\pm$ std over 3 random permutations. "
                 r"A small range across orderings indicates order robustness. "
                 r"\textbf{Bold} = best per column.}")
    lines.append(r"\label{tab:order_study}")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Forward & Reverse & Random (mean$\pm$std) & Range \\")
    lines.append(r"\midrule")

    for method in METHODS:
        s    = summary[method]
        name = cite[method]

        fwd  = s.get("forward", None)
        rev  = s.get("reverse", None)
        rmn  = s.get("random_mean", None)
        rstd = s.get("random_std",  None)

        fwd_str = f"{fwd:.1f}" if fwd is not None else "---"
        rev_str = f"{rev:.1f}" if rev is not None else "---"
        rng_str = (f"{rmn-rstd:.1f}--{rmn+rstd:.1f}"
                   if rmn is not None else "---")
        rnd_str = (f"${rmn:.1f}\\pm{rstd:.1f}$"
                   if rmn is not None else "---")

        # Bold best in each column
        all_fwd  = [summary[m].get("forward", None) for m in METHODS]
        all_rev  = [summary[m].get("reverse", None) for m in METHODS]
        all_rmn  = [summary[m].get("random_mean", None) for m in METHODS]

        best_fwd = max((v for v in all_fwd if v is not None), default=None)
        best_rev = max((v for v in all_rev if v is not None), default=None)
        best_rmn = max((v for v in all_rmn if v is not None), default=None)

        if fwd is not None and best_fwd is not None and abs(fwd - best_fwd) < 0.05:
            fwd_str = f"\\textbf{{{fwd_str}}}"
        if rev is not None and best_rev is not None and abs(rev - best_rev) < 0.05:
            rev_str = f"\\textbf{{{rev_str}}}"
        if rmn is not None and best_rmn is not None and abs(rmn - best_rmn) < 0.05:
            rnd_str = f"$\\mathbf{{{rmn:.1f}\\pm{rstd:.1f}}}$"

        lines.append(f"{name} & {fwd_str} & {rev_str} & {rnd_str} & {rng_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)
    tex_path  = os.path.join(RESULTS_DIR, "order_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)
    print(f"LaTeX saved: {tex_path}")
    print("\n" + "="*60)
    print("Paste into Overleaf:")
    print("="*60)
    print(latex_str)


# =============================================================================
# 6. MAIN
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ContinualTTA Corruption Order Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run ContinualTTA on all three orderings (CIFAR-10-C):
  python order_study.py --method ContinualTTA --dataset cifar10c

  # Run only forward ordering:
  python order_study.py --method RoTTA --dataset cifar10c --orders forward

  # Run on ImageNet-C:
  python order_study.py --method ContinualTTA --dataset imagenetc

  # Merge results and generate LaTeX after all methods run:
  python order_study.py --merge
        """)

    parser.add_argument("--method",  type=str, default=None,
                        choices=METHODS,
                        help="TTA method to evaluate")
    parser.add_argument("--dataset", type=str, default="cifar10c",
                        choices=["cifar10c", "imagenetc"])
    parser.add_argument("--orders",  type=str, nargs="+",
                        default=["forward", "reverse", "random"],
                        choices=["forward", "reverse", "random"],
                        help="Which orderings to run")
    parser.add_argument("--merge",   action="store_true",
                        help="Merge all CSVs and generate LaTeX table")
    args = parser.parse_args()

    if args.merge:
        merge_and_latex(args.dataset)
        exit(0)

    if args.method is None:
        parser.error("--method is required unless using --merge")

    # Setup
    print(f"\n{'='*60}")
    print(f"Corruption Order Study")
    print(f"{'='*60}")
    print(f"Device  : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"Method  : {args.method}")
    print(f"Dataset : {args.dataset}")
    print(f"Orders  : {args.orders}")

    if args.dataset == "cifar10c":
        source       = load_cifar_model()
        setup_bn_fn  = setup_bn_cifar
        num_classes  = CIFAR_CLASSES
    else:
        source       = load_imagenet_model()
        setup_bn_fn  = setup_bn_imagenet
        num_classes  = IMAGENET_CLASSES

    print(f"Params  : {sum(p.numel() for p in source.parameters()):,}\n")

    # Run each requested ordering
    if "forward" in args.orders:
        print(f"\n{'─'*50}")
        print(f"FORWARD ORDER")
        print(f"{'─'*50}")
        results = run_order(args.method, source, setup_bn_fn, num_classes,
                            FORWARD_ORDER, args.dataset, "forward")
        save_order_csv(args.method, "forward", None, results)

    if "reverse" in args.orders:
        print(f"\n{'─'*50}")
        print(f"REVERSE ORDER")
        print(f"{'─'*50}")
        results = run_order(args.method, source, setup_bn_fn, num_classes,
                            REVERSE_ORDER, args.dataset, "reverse")
        save_order_csv(args.method, "reverse", None, results)

    if "random" in args.orders:
        for seed in RANDOM_SEEDS:
            print(f"\n{'─'*50}")
            print(f"RANDOM ORDER — seed {seed}")
            print(f"{'─'*50}")
            random_order = get_random_order(seed)
            results = run_order(args.method, source, setup_bn_fn, num_classes,
                                random_order, args.dataset, "random", seed=seed)
            save_order_csv(args.method, "random", seed, results)

    print(f"\n{'='*60}")
    print(f"DONE — {args.method} order study complete")
    print(f"Results: {os.path.abspath(RESULTS_DIR)}/")
    print(f"Run with --merge after all methods complete to generate LaTeX.")
    print(f"{'='*60}")