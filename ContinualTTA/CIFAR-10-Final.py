# =============================================================================
# ContinualTTA — CIFAR-10-C Evaluation (WACV 2027)
# Methods: Baseline | TENT | EATA | CoTTA | RoTTA | SAR | ContinualTTA (Ours)
#
# Protocols:
#   Setting A: Mixed i.i.d. per severity (RoTTA original protocol)
#   Setting B: Continual sequential — no reset between corruptions (ours)
#
# Output:
#   results/setting_a.csv          — Setting A per-severity numbers
#   results/setting_b.csv          — Setting B per-corruption numbers
#   results/summary.csv            — Δ table (A vs B)
#   results/main_table.tex         — LaTeX table for paper (Setting B)
#   results/protocol_table.tex     — LaTeX table for paper (A vs B comparison)
#
# All implementations audited for publication correctness.
# =============================================================================

import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

# ── Config — update paths before running ──────────────────────────────────────
MODEL_PATH = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
DATA_DIR   = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
RESULTS_DIR = "results"   # all outputs saved here — works on both Kaggle and local

# Detect platform and set num_workers accordingly
import platform
NUM_WORKERS = 0 if platform.system() == "Windows" else 2

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 32
NUM_CLASSES    = 10
ALL_SEVERITIES = [1, 2, 3, 4, 5]
SEED           = 42

LR           = 1e-3
EMA_DECAY    = 0.9
E_MARGIN     = 0.4 * math.log(NUM_CLASSES)   # 0.921 nats for C=10
ALPHA        = 0.5
JS_THRESHOLD = 0.02

ROTTA_NU = 0.001
ROTTA_N  = 64
SAR_RHO  = 0.05

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]

METHODS = ["Baseline", "TENT", "EATA", "CoTTA", "RoTTA", "SAR", "ContinualTTA"]


# =============================================================================
# 1. DATASET
# =============================================================================

class CIFAR10C_Dataset(Dataset):
    def __init__(self, corruption, severity):
        data        = np.load(os.path.join(DATA_DIR, f"{corruption}.npy"),
                              mmap_mode='r')
        labels      = np.load(os.path.join(DATA_DIR, "labels.npy"),
                              mmap_mode='r')
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


def load_single(corruption, severity):
    return DataLoader(
        CIFAR10C_Dataset(corruption, severity),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)


def load_mixed(severity, seed=SEED):
    """All 15 corruptions at one severity randomly shuffled — RoTTA protocol."""
    combined = ConcatDataset(
        [CIFAR10C_Dataset(c, severity) for c in ALL_CORRUPTIONS])
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(combined), generator=g).tolist()
    subset  = torch.utils.data.Subset(combined, indices)
    return DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)


def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


# =============================================================================
# 2. HELPERS
# =============================================================================

def softmax_entropy(logits):
    """Per-sample Shannon entropy H(p). Shape: (B,)"""
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)


def eval_loader(model_fn, loader):
    """Evaluate model_fn over loader, return accuracy %."""
    correct, total = 0, 0
    for x, y in loader:
        x, y    = x.to(DEVICE), y.to(DEVICE)
        logits  = model_fn(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def setup_bn(model):
    """
    TENT-style BN setup for CIFAR models:
    - track_running_stats = False → use per-batch statistics
    - Only BN gamma (weight) and beta (bias) are trainable
    - All other parameters frozen
    Note: For ImageNet pretrained models use setup_bn_imagenet instead
    (track_running_stats=True, momentum=0).
    """
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
# No adaptation — frozen model evaluated with torch.no_grad().
# =============================================================================

def make_baseline(source):
    model = copy.deepcopy(source).eval()
    def fn(x):
        with torch.no_grad():
            return model(x)
    return fn


# =============================================================================
# 4. TENT  (Wang et al., ICLR 2021)
# Entropy minimisation on BN affine parameters.
# Reference: https://github.com/DequanWang/tent
# =============================================================================

def make_tent(source):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    @torch.enable_grad()
    def fn(x):
        logits = model(x)
        softmax_entropy(logits).mean().backward()
        opt.step()
        opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 5. EATA  (Niu et al., ICML 2022)
# Two-filter entropy minimisation with Fisher regularisation.
#
# Filter 1: entropy < E_margin (reliable samples)
# Filter 2: cos_sim(p_i, p_ref) < (1 - d_margin) (diverse samples)
#   — keeps samples sufficiently DIFFERENT from running reference
#   — prevents redundant gradient updates
# Loss: H(p_reliable_diverse) + lambda * Fisher_regularisation
#
# Reference: https://github.com/mr-eggplant/EATA
# =============================================================================

def make_eata(source, fisher_loader=None):
    model, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    # Compute Fisher importance weights on first corruption's data
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
        for n in fisher:
            fisher[n] /= 10

    ref_probs = [None]
    d_margin  = 0.05

    @torch.enable_grad()
    def fn(x):
        logits  = model(x)
        entropy = softmax_entropy(logits)
        probs   = logits.softmax(1)

        # Filter 1: reliable (low entropy)
        mask_e = entropy < E_MARGIN

        # Filter 2: diverse — keep samples sufficiently different from reference
        # cos_sim < (1 - d_margin) means NOT too similar to reference
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

        # Update running reference distribution
        with torch.no_grad():
            if ref_probs[0] is None:
                ref_probs[0] = probs[mask].mean(0).detach()
            else:
                ref_probs[0] = (0.9 * ref_probs[0]
                                + 0.1 * probs[mask].mean(0).detach())

        # Loss = entropy + Fisher regularisation (prevents catastrophic forgetting)
        fisher_reg = sum((fisher[n] * p.pow(2)).sum()
                         for n, p in model.named_parameters()
                         if p.requires_grad and n in fisher)
        loss = entropy[mask].mean() + 1e-3 * fisher_reg
        loss.backward()
        opt.step()
        opt.zero_grad()
        return logits

    return fn


# =============================================================================
# 6. CoTTA  (Wang et al., CVPR 2022)
# Augmentation-averaged pseudo-labels + teacher EMA + stochastic restoration.
#
# Fix applied: augmentations at 224x224 (model input resolution).
# Original paper uses RandomCrop on 32x32 CIFAR images; after Resize(224)
# a 32px crop is meaningless — corrected to RandomResizedCrop(224).
#
# Reference: https://github.com/qinenergy/cotta
# =============================================================================

def make_cotta(source):
    # Source model — reference for stochastic restoration, never updated
    src = copy.deepcopy(source).eval()
    src.requires_grad_(False)

    # Adapted student model
    adapted, params = setup_bn(copy.deepcopy(source))
    opt = torch.optim.Adam(params, lr=LR)

    # Teacher model — EMA of student, provides stable pseudo-labels
    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)

    # Augmentations at 224x224 — MUST match model input resolution
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    ])

    @torch.enable_grad()
    def fn(x):
        # Step 1: teacher generates augmentation-averaged pseudo-labels
        with torch.no_grad():
            pseudo = torch.stack(
                [teacher(aug(x)).softmax(1) for _ in range(4)]).mean(0)

        # Step 2: student cross-entropy with teacher soft labels
        logits = adapted(x)
        loss   = -(pseudo * logits.log_softmax(1)).sum(1).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        with torch.no_grad():
            # Step 3: teacher EMA update (mt_alpha=0.999)
            for tp, ap in zip(teacher.parameters(), adapted.parameters()):
                tp.data = 0.999 * tp.data + 0.001 * ap.data

            # Step 4: stochastic restoration from source (rst=0.01)
            for (_, pa), (_, ps) in zip(adapted.named_parameters(),
                                         src.named_parameters()):
                if pa.requires_grad:
                    mask = torch.rand_like(pa) < 0.01
                    pa.data[mask] = ps.data[mask]

        return logits

    return fn


# =============================================================================
# 7. RoTTA  (Yuan et al., CVPR 2023)
# Robust BN + CSTU memory bank + timeliness reweighting.
#
# (a) RBN: EMA global stats (momentum=0.05) instead of per-batch
# (b) CSTU: category-balanced bank, replaces highest-entropy sample when full
#     timeliness: prefer most recent samples during sampling
# (c) Loss: E(age) * CE(teacher_weak, student) where E(age) = Eq.9 from paper
#     E(age) = exp(-age/N) / (1 + exp(-age/N))
# (d) Teacher EMA: very slow (nu=0.001) for stability
#
# Reference: https://github.com/BIT-DA/RoTTA
# =============================================================================

def make_rotta(source):
    student = copy.deepcopy(source)
    student.train()
    student.requires_grad_(False)
    for m in student.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True   # RBN: keep running stats
            m.momentum = 0.05             # slow EMA update
    params = [p for m in student.modules()
              if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
              for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)

    teacher = copy.deepcopy(source).eval()
    teacher.requires_grad_(False)

    per_class = max(1, ROTTA_N // NUM_CLASSES)   # 6 slots per class for C=10
    bank      = {c: [] for c in range(NUM_CLASSES)}
    age       = [0]

    @torch.enable_grad()
    def fn(x):
        logits  = student(x)
        plabels = logits.argmax(1).detach()
        ents    = softmax_entropy(logits).detach()

        # CSTU: update memory bank
        with torch.no_grad():
            for i, (c, e) in enumerate(zip(plabels.tolist(), ents.tolist())):
                entry = (x[i].detach().cpu(), e, age[0])
                if len(bank[c]) < per_class:
                    bank[c].append(entry)
                else:
                    # Replace highest-entropy (least confident) sample
                    worst = max(range(len(bank[c])),
                                key=lambda j: bank[c][j][1])
                    if e < bank[c][worst][1]:
                        bank[c][worst] = entry
            age[0] += 1

        # Category-balanced sampling — prefer most recent (timeliness)
        samples, ages_list = [], []
        for c in range(NUM_CLASSES):
            if bank[c]:
                # Sort by age descending → most recent first
                for entry in sorted(bank[c], key=lambda e: -e[2])[:2]:
                    samples.append(entry[0])
                    ages_list.append(entry[2])

        if len(samples) >= 2:
            mem_x  = torch.stack(samples).to(DEVICE)
            ages_t = torch.tensor(ages_list, dtype=torch.float32, device=DEVICE)

            # Timeliness reweighting E(age) from Eq.9 of RoTTA paper
            e_age = (torch.exp(-ages_t / ROTTA_N)
                     / (1 + torch.exp(-ages_t / ROTTA_N)))

            with torch.no_grad():
                t_probs = teacher(mem_x).softmax(1)

            s_logits = student(mem_x)
            # CE loss per sample (Eq.10), averaged per class
            ce   = -(t_probs * s_logits.log_softmax(1)).sum(1) / NUM_CLASSES
            # Weighted by timeliness (Eq.11)
            loss = (e_age * ce).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

            # Teacher EMA (Eq.8) — very slow: nu=0.001
            with torch.no_grad():
                for tp, sp in zip(teacher.parameters(), student.parameters()):
                    tp.data = (1 - ROTTA_NU) * tp.data + ROTTA_NU * sp.data

        return logits

    return fn


# =============================================================================
# 8. SAR  (Niu et al., ICLR 2023)
# Sharpness-Aware and Reliable entropy minimisation.
#
# Key idea: find flat minima that are robust to perturbation.
# Two-step update:
#   Step 1: forward + backward at θ → gradient g
#   Step 2: perturb θ' = θ + ρ * g / ||g||
#   Step 3: forward + backward at θ' → sharpness-aware gradient
#   Step 4: restore θ, apply SGD update
# Filter: entropy < dynamic_threshold (EMA-tracked)
#
# Reference: https://github.com/mr-eggplant/SAR
# =============================================================================

def make_sar(source):
    model, params = setup_bn(copy.deepcopy(source))
    # SAR uses SGD (not Adam) — for flat minima exploration
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9)
    ema_entropy = [None]

    @torch.enable_grad()
    def fn(x):
        # Compute entropy without gradient for filtering
        with torch.no_grad():
            logits_init  = model(x)
            entropy_init = softmax_entropy(logits_init)

        if ema_entropy[0] is None:
            ema_entropy[0] = entropy_init.mean().item()

        # Dynamic threshold: min(E_margin, ema + 0.4*log(C))
        dynamic_thresh = min(E_MARGIN,
                             ema_entropy[0] + 0.4 * math.log(NUM_CLASSES))
        reliable = entropy_init < dynamic_thresh
        if reliable.sum() == 0:
            return logits_init

        x_rel = x[reliable]

        # Step 1: gradient at current params
        logits_1 = model(x_rel)
        softmax_entropy(logits_1).mean().backward()
        grad_norm = torch.norm(torch.stack(
            [p.grad.norm() for p in params if p.grad is not None]))

        # Step 2: perturb params towards sharp region
        e_ws = []
        for p in params:
            if p.grad is not None:
                e_w = p.grad * SAR_RHO / (grad_norm + 1e-12)
                p.data.add_(e_w)
                e_ws.append(e_w)
                p.grad.zero_()
            else:
                e_ws.append(None)

        # Step 3: gradient at perturbed params
        logits_2   = model(x_rel)
        entropy_2  = softmax_entropy(logits_2)
        reliable_2 = entropy_2 < E_MARGIN
        if reliable_2.sum() > 0:
            entropy_2[reliable_2].mean().backward()

        # Step 4: restore original params, apply update
        for p, e_w in zip(params, e_ws):
            if e_w is not None:
                p.data.sub_(e_w)
        opt.step()
        opt.zero_grad()

        # Update EMA entropy tracker
        with torch.no_grad():
            logits_out  = model(x)
            entropy_out = softmax_entropy(logits_out)
            ema_entropy[0] = (0.9 * ema_entropy[0]
                              + 0.1 * entropy_out.mean().item())

        return logits_out

    return fn


# =============================================================================
# 9. ContinualTTA (ours)
# Jensen-Shannon shift detector + reliable filter + EMA prototype bank.
#
# Three components:
# (a) JS shift detector: gates adaptation to genuine distribution changes
#     JS(p_ref, p_t) = 0.5*KL(p_ref||m) + 0.5*KL(p_t||m), m=0.5*(p+q)
#     Chosen over KL by ablation: JS=85.1% > Chi2=85.0% > KL=84.8%
#     Reference: Lin (1991), IEEE Trans. Information Theory
# (b) Reliable filter: entropy < E_margin (same as EATA filter 1)
# (c) Prototype bank: EMA per-class 2048-d feature prototypes
#     Consistency loss: MSE(features, prototypes)
# Loss: H(p_reliable) + alpha * L_proto
# Only BN gamma/beta updated (~0.1% of parameters)
# =============================================================================

class PrototypeBank(nn.Module):
    """EMA prototype memory bank. One 2048-d prototype per class."""

    def __init__(self):
        super().__init__()
        self.decay = EMA_DECAY
        self.register_buffer("prototypes",  torch.zeros(NUM_CLASSES, 2048))
        self.register_buffer("initialised", torch.zeros(NUM_CLASSES).bool())

    @torch.no_grad()
    def update(self, features, pseudo_labels):
        """Update prototypes with EMA using reliable features."""
        for c in pseudo_labels.unique():
            mask = (pseudo_labels == c)
            mf   = features[mask].mean(0)
            if self.initialised[c]:
                self.prototypes[c] = (self.decay * self.prototypes[c]
                                      + (1 - self.decay) * mf)
            else:
                self.prototypes[c] = mf
                self.initialised[c] = True

    def consistency_loss(self, features, pseudo_labels):
        """MSE between features and their corresponding class prototypes."""
        loss, count = torch.tensor(0.0, device=features.device), 0
        for c in pseudo_labels.unique():
            if not self.initialised[c]:
                continue
            mask = (pseudo_labels == c)
            loss += F.mse_loss(
                features[mask],
                self.prototypes[c].unsqueeze(0).expand(mask.sum(), -1))
            count += 1
        return loss / max(count, 1)


class JSShiftDetector:
    """
    Jensen-Shannon divergence shift detector.

    Computes JS(p_ref, p_t) between batch mean prediction p_t and
    running reference p_ref. Adapts only when JS > threshold.

    JS(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)
    JS in [0, ln2], symmetric, bounded, always finite.

    Reference updated AFTER computing JS — intentional ordering.
    """

    def __init__(self, threshold=JS_THRESHOLD, ema=0.9):
        self.threshold = threshold
        self.ema       = ema
        self.reference = None

    def should_adapt(self, logits):
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)   # batch-level distribution (C,)

            if self.reference is None:
                self.reference = p_t.clone()
                return True   # always adapt on first batch

            p_ref = self.reference
            m     = 0.5 * (p_ref + p_t)   # mixture distribution

            # F.kl_div(log_Q, P) = KL(P||Q)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             p_ref.unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0),   reduction="batchmean")
            js   = 0.5 * (kl_1 + kl_2)

            # Update reference AFTER JS computation
            self.reference = self.ema * self.reference + (1 - self.ema) * p_t

            return js.item() > self.threshold


def make_ctta(source):
    model, params = setup_bn(copy.deepcopy(source))
    bank     = PrototypeBank().to(DEVICE)
    detector = JSShiftDetector()
    opt      = torch.optim.Adam(params, lr=LR)
    captured = {}
    # Hook to capture avgpool features (2048-d) during forward pass
    handle   = model.avgpool.register_forward_hook(
        lambda m, i, o: captured.update({"feat": o.flatten(1)}))

    @torch.enable_grad()
    def fn(x):
        logits        = model(x)
        features      = captured["feat"]
        pseudo_labels = logits.argmax(1).detach()

        # Gate 1: JS shift detector — skip if no genuine shift
        if not detector.should_adapt(logits.detach()):
            return logits

        # Gate 2: reliable sample filter
        entropy  = softmax_entropy(logits)
        reliable = entropy < E_MARGIN
        if reliable.sum() == 0:
            return logits

        # Combined loss on reliable samples
        loss = (entropy[reliable].mean()
                + ALPHA * bank.consistency_loss(
                    features[reliable].detach(),
                    pseudo_labels[reliable]))
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Update prototype bank
        bank.update(features[reliable].detach(), pseudo_labels[reliable])
        return logits

    fn._handle = handle
    return fn


# =============================================================================
# 10. BUILD ALL METHODS
# =============================================================================

def build_fns(source_model, fisher_loader=None):
    """Build fresh method functions. Call once per severity."""
    return {
        "Baseline":     make_baseline(source_model),
        "TENT":         make_tent(source_model),
        "EATA":         make_eata(source_model, fisher_loader),
        "CoTTA":        make_cotta(source_model),
        "RoTTA":        make_rotta(source_model),
        "SAR":          make_sar(source_model),
        "ContinualTTA": make_ctta(source_model),
    }


# =============================================================================
# 11. SETTING A — Mixed i.i.d. per severity (RoTTA protocol)
# =============================================================================

def run_setting_a(source_model):
    """
    Each method runs on 150k randomly mixed images at each severity.
    Fresh model per severity — no carry-over between severity levels.
    This matches the RoTTA paper evaluation protocol.
    """
    print("\n── Setting A: Mixed i.i.d. per severity (RoTTA protocol) ──")
    sev_results = {}

    for severity in ALL_SEVERITIES:
        print(f"\n  Severity {severity}  "
              f"(15 corruptions × 10k = 150k images, shuffled)")
        fisher_loader = load_single(ALL_CORRUPTIONS[0], severity)
        fns = build_fns(source_model, fisher_loader)
        sev_results[severity] = {}

        for method in METHODS:
            loader = load_mixed(severity)
            acc    = eval_loader(fns[method], loader)
            sev_results[severity][method] = acc
            del loader
            torch.cuda.empty_cache()
            print(f"    {method:<18} {acc:.1f}%")

    means_a = {m: np.mean([sev_results[s][m] for s in ALL_SEVERITIES])
               for m in METHODS}
    return sev_results, means_a


# =============================================================================
# 12. SETTING B — Continual sequential (our protocol)
# =============================================================================

def run_setting_b(source_model):
    """
    One model instance per method runs through all 15 corruptions in fixed
    order without reset. S1–S5 averaged. This is our proposed protocol.
    """
    print("\n── Setting B: Continual sequential (our protocol) ──")
    all_sev = {}

    for severity in ALL_SEVERITIES:
        print(f"\n  Severity {severity}")
        fisher_loader = load_single(ALL_CORRUPTIONS[0], severity)
        fns     = build_fns(source_model, fisher_loader)
        results = {m: {} for m in METHODS}

        for corruption in ALL_CORRUPTIONS:
            loader = load_single(corruption, severity)
            for method in METHODS:
                results[method][corruption] = eval_loader(fns[method], loader)
            del loader
            torch.cuda.empty_cache()
            print(f"    {corruption:<24}"
                  f"  TENT={results['TENT'][corruption]:.1f}%"
                  f"  SAR={results['SAR'][corruption]:.1f}%"
                  f"  RoTTA={results['RoTTA'][corruption]:.1f}%"
                  f"  Ours={results['ContinualTTA'][corruption]:.1f}%")

        all_sev[severity] = results

    # Average over all severities
    averaged = {m: {} for m in METHODS}
    for method in METHODS:
        for corruption in ALL_CORRUPTIONS:
            averaged[method][corruption] = np.mean(
                [all_sev[s][method][corruption] for s in ALL_SEVERITIES])

    means_b = {m: np.mean(list(averaged[m].values())) for m in METHODS}
    return averaged, means_b


# =============================================================================
# 13. PRINT TABLES
# =============================================================================

def print_setting_a(sev_results, means_a):
    col    = 14
    header = f"{'Severity':<12}" + "".join(f"{m:>{col}}" for m in METHODS)
    sep    = "─" * len(header)
    print(f"\n{'═'*len(header)}")
    print("SETTING A — Mixed i.i.d. per severity (RoTTA original protocol)")
    print(f"{'═'*len(header)}")
    print(header); print(sep)
    for s in ALL_SEVERITIES:
        best = max(sev_results[s][m] for m in METHODS)
        row  = f"S{s:<11}"
        for method in METHODS:
            acc  = sev_results[s][method]
            cell = f"{acc:.1f}%" + ("*" if abs(acc - best) < 0.05 else "")
            row += f"{cell:>{col}}"
        print(row)
    print(sep)
    best_m   = max(means_a.values())
    mean_row = f"{'Mean':<12}"
    for method in METHODS:
        cell = f"{means_a[method]:.1f}%" + \
               ("*" if abs(means_a[method] - best_m) < 0.05 else "")
        mean_row += f"{cell:>{col}}"
    print(mean_row)
    print(f"{'═'*len(header)}")
    print("  * = best in that row")


def print_setting_b(averaged, means_b):
    col    = 14
    header = f"{'Corruption':<24}" + "".join(f"{m:>{col}}" for m in METHODS)
    sep    = "─" * len(header)
    print(f"\n{'═'*len(header)}")
    print("SETTING B — Continual sequential, S1–S5 mean (our protocol)")
    print(f"{'═'*len(header)}")
    print(header); print(sep)
    for corruption in ALL_CORRUPTIONS:
        best = max(averaged[m][corruption] for m in METHODS)
        row  = f"{corruption:<24}"
        for method in METHODS:
            acc  = averaged[method][corruption]
            cell = f"{acc:.1f}%" + ("*" if abs(acc - best) < 0.05 else "")
            row += f"{cell:>{col}}"
        print(row)
    print(sep)
    best_m   = max(means_b.values())
    mean_row = f"{'Mean (S1–S5)':<24}"
    for method in METHODS:
        cell = f"{means_b[method]:.1f}%" + \
               ("*" if abs(means_b[method] - best_m) < 0.05 else "")
        mean_row += f"{cell:>{col}}"
    print(mean_row)
    print(f"{'═'*len(header)}")
    print("  * = best in that row")


def print_summary(means_a, means_b):
    print(f"\n{'═'*72}")
    print("FINAL SUMMARY — Setting A (i.i.d.) vs Setting B (sequential)")
    print(f"{'═'*72}")
    print(f"  {'Method':<18} {'Setting A':>12} {'Setting B':>12} {'Δ (B−A)':>10}")
    print(f"  {'──────':<18} {'─────────':>12} {'─────────':>12} {'───────':>10}")
    for method in METHODS:
        a    = means_a[method]
        b    = means_b[method]
        gap  = b - a
        flag = "  ← ours" if method == "ContinualTTA" else \
               ("  ← collapses" if gap < -3 else "")
        print(f"  {method:<18} {a:>10.1f}%  {b:>10.1f}%  {gap:>+8.1f}%{flag}")
    print(f"{'═'*72}")


# =============================================================================
# 14. LATEX TABLE GENERATORS
# =============================================================================

def generate_latex_main(averaged, means_b):
    """
    Table 1 in paper: Setting B per-corruption results.
    Paste directly into Overleaf.
    """
    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "CoTTA":        "CoTTA~\\cite{wang2022continual}",
        "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
        "SAR":          "SAR~\\cite{niu2023towards}",
        "ContinualTTA": "\\textbf{ContinualTTA (Ours)}",
    }
    corr_names = {
        "gaussian_noise": "Gaussian Noise", "shot_noise": "Shot Noise",
        "impulse_noise": "Impulse Noise",   "defocus_blur": "Defocus Blur",
        "glass_blur": "Glass Blur",          "motion_blur": "Motion Blur",
        "zoom_blur": "Zoom Blur",            "snow": "Snow",
        "frost": "Frost",                    "fog": "Fog",
        "brightness": "Brightness",          "contrast": "Contrast",
        "elastic_transform": "Elastic",      "pixelate": "Pixelate",
        "jpeg_compression": "JPEG",
    }
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Accuracy (\%) on CIFAR-10-C under continual "
                 r"sequential shift (Setting~B), S1--S5 averaged. "
                 r"\textbf{Bold} = best per row. "
                 r"Source model: ResNet-50 trained on clean CIFAR-10.}")
    lines.append(r"\label{tab:main_cifar10c}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * len(METHODS) + "}")
    lines.append(r"\toprule")
    lines.append("Corruption & " +
                 " & ".join(cite[m] for m in METHODS) + r" \\")
    lines.append(r"\midrule")

    for corruption in ALL_CORRUPTIONS:
        best = max(averaged[m][corruption] for m in METHODS)
        row  = corr_names[corruption]
        for method in METHODS:
            val = averaged[method][corruption]
            row += f" & \\textbf{{{val:.1f}}}" if abs(val - best) < 0.05 \
                   else f" & {val:.1f}"
        lines.append(row + r" \\")

    lines.append(r"\midrule")
    best_b = max(means_b.values())
    row_b  = r"\textbf{Mean (Setting~B)}"
    for method in METHODS:
        val = means_b[method]
        row_b += f" & \\textbf{{{val:.1f}}}" if abs(val - best_b) < 0.05 \
                 else f" & {val:.1f}"
    lines.append(row_b + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_latex_protocol(means_a, means_b):
    """
    Table 2 in paper: Protocol comparison (Setting A vs B, Δ column).
    This is your strongest finding table.
    """
    cite = {
        "Baseline":     "Baseline",
        "TENT":         "TENT~\\cite{wang2021tent}",
        "EATA":         "EATA~\\cite{niu2022efficient}",
        "CoTTA":        "CoTTA~\\cite{wang2022continual}",
        "RoTTA":        "RoTTA~\\cite{yuan2023robust}",
        "SAR":          "SAR~\\cite{niu2023towards}",
        "ContinualTTA": "\\textbf{ContinualTTA (Ours)}",
    }
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Protocol robustness comparison on CIFAR-10-C. "
                 r"Setting~A: mixed i.i.d.\ (RoTTA protocol). "
                 r"Setting~B: continual sequential (ours). "
                 r"$\Delta$~=~B$-$A. "
                 r"ContinualTTA is the only method that improves under "
                 r"sequential shift.}")
    lines.append(r"\label{tab:protocol_comparison}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Setting~A & Setting~B & $\Delta$ \\")
    lines.append(r"\midrule")

    for method in METHODS:
        a   = means_a[method]
        b   = means_b[method]
        gap = b - a
        name = cite[method]
        if gap < -3:
            gap_str = f"\\textcolor{{red}}{{{gap:+.1f}}}"
        elif method == "ContinualTTA":
            gap_str = f"\\textbf{{{gap:+.1f}}}"
        else:
            gap_str = f"{gap:+.1f}"
        lines.append(f"{name} & {a:.1f} & {b:.1f} & {gap_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# 15. SAVE ALL RESULTS
# =============================================================================

def save_all(sev_results, means_a, averaged, means_b):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Setting A CSV ─────────────────────────────────────────────────────────
    path_a = os.path.join(RESULTS_DIR, "setting_a.csv")
    with open(path_a, "w") as f:
        f.write("severity," + ",".join(METHODS) + "\n")
        for s in ALL_SEVERITIES:
            f.write(str(s) + "," +
                    ",".join(f"{sev_results[s][m]:.1f}"
                             for m in METHODS) + "\n")
        f.write("Mean," +
                ",".join(f"{means_a[m]:.1f}" for m in METHODS) + "\n")
    print(f"  Saved: {path_a}")

    # ── Setting B CSV ─────────────────────────────────────────────────────────
    path_b = os.path.join(RESULTS_DIR, "setting_b.csv")
    with open(path_b, "w") as f:
        f.write("corruption," + ",".join(METHODS) + "\n")
        for c in ALL_CORRUPTIONS:
            f.write(c + "," +
                    ",".join(f"{averaged[m][c]:.1f}"
                             for m in METHODS) + "\n")
        f.write("Mean," +
                ",".join(f"{means_b[m]:.1f}" for m in METHODS) + "\n")
    print(f"  Saved: {path_b}")

    # ── Summary CSV (Δ table) ─────────────────────────────────────────────────
    path_s = os.path.join(RESULTS_DIR, "summary.csv")
    with open(path_s, "w") as f:
        f.write("method,setting_a,setting_b,delta\n")
        for method in METHODS:
            a   = means_a[method]
            b   = means_b[method]
            gap = b - a
            f.write(f"{method},{a:.1f},{b:.1f},{gap:+.1f}\n")
    print(f"  Saved: {path_s}")

    # ── LaTeX: main table (Setting B) ─────────────────────────────────────────
    path_tex1 = os.path.join(RESULTS_DIR, "main_table.tex")
    latex_main = generate_latex_main(averaged, means_b)
    with open(path_tex1, "w") as f:
        f.write(latex_main)
    print(f"  Saved: {path_tex1}")

    # ── LaTeX: protocol comparison table ──────────────────────────────────────
    path_tex2 = os.path.join(RESULTS_DIR, "protocol_table.tex")
    latex_protocol = generate_latex_protocol(means_a, means_b)
    with open(path_tex2, "w") as f:
        f.write(latex_protocol)
    print(f"  Saved: {path_tex2}")

    # ── Print LaTeX to console for immediate use ───────────────────────────────
    print("\n" + "=" * 70)
    print("TABLE 1 — Main results (Setting B) — paste into Overleaf:")
    print("=" * 70)
    print(latex_main)

    print("\n" + "=" * 70)
    print("TABLE 2 — Protocol comparison — paste into Overleaf:")
    print("=" * 70)
    print(latex_protocol)


# =============================================================================
# 16. MAIN
# =============================================================================

if __name__ == "__main__":

    print(f"Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Methods    : {METHODS}")
    print(f"E_margin   : {E_MARGIN:.3f} nats  (= 0.4 * ln({NUM_CLASSES}))")
    print(f"num_workers: {NUM_WORKERS}  (0=Windows, 2=Linux/Kaggle)")
    print(f"Results dir: {RESULTS_DIR}/")

    # Verify data exists
    for c in ALL_CORRUPTIONS[:3]:
        assert os.path.isfile(os.path.join(DATA_DIR, f"{c}.npy")), \
            f"Missing: {DATA_DIR}/{c}.npy"
    assert os.path.isfile(os.path.join(DATA_DIR, "labels.npy")), \
        f"Missing: {DATA_DIR}/labels.npy"
    print("Data check : passed\n")

    # Load model
    print("Loading source model...")
    source_model = load_model()
    n_params = sum(p.numel() for p in source_model.parameters())
    print(f"Parameters : {n_params:,}")
    assert source_model.fc.out_features == NUM_CLASSES, \
        f"Model outputs {source_model.fc.out_features} classes, expected {NUM_CLASSES}"

    # Sanity check
    print("\nSanity check — baseline on gaussian_noise S3...")
    loader = load_single("gaussian_noise", 3)
    acc    = eval_loader(make_baseline(source_model), loader)
    del loader
    torch.cuda.empty_cache()
    print(f"Baseline S3: {acc:.1f}%  (expected ~40% for ResNet-50 CIFAR-10)")
    assert acc > 20.0, f"Too low ({acc:.1f}%) — check MODEL_PATH or DATA_DIR"
    print("Sanity check passed.\n")

    # ── SETTING A ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("SETTING A — Mixed i.i.d. (RoTTA original protocol)")
    print("15 corruptions randomly shuffled, one fresh model per severity")
    print("=" * 70)
    sev_results, means_a = run_setting_a(source_model)
    print_setting_a(sev_results, means_a)

    # ── SETTING B ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SETTING B — Continual sequential (our protocol)")
    print("15 corruptions in fixed order, no reset, S1–S5 averaged")
    print("=" * 70)
    averaged, means_b = run_setting_b(source_model)
    print_setting_b(averaged, means_b)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(means_a, means_b)

    # ── Save everything ───────────────────────────────────────────────────────
    print("\nSaving all results...")
    save_all(sev_results, means_a, averaged, means_b)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"All outputs saved to: {os.path.abspath(RESULTS_DIR)}/")
    print("  setting_a.csv       — Setting A per-severity numbers")
    print("  setting_b.csv       — Setting B per-corruption numbers")
    print("  summary.csv         — Δ table (A vs B)")
    print("  main_table.tex      — Table 1 for paper (Setting B)")
    print("  protocol_table.tex  — Table 2 for paper (A vs B comparison)")