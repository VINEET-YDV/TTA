# =============================================================================
# ContinualTTA — Performance Improvement Study
#
# Tests all modifications that could improve over the baseline ContinualTTA.
# Run one experiment at a time to preserve Kaggle quota.
#
# EXPERIMENTS (in recommended order):
#
#   python improve.py --exp proto_fix       # Fix prototype gradient flow
#   python improve.py --exp conf_filter     # Add confidence filter to prototypes
#   python improve.py --exp both_fixes      # Both proto fixes together (best guess)
#   python improve.py --exp threshold_sweep # Fine-grained tau sweep
#   python improve.py --exp lr_sweep        # Learning rate sweep
#   python improve.py --exp margin_sweep    # Entropy margin factor sweep
#   python improve.py --exp ema_sweep       # JS reference EMA sweep
#   python improve.py --exp full_sweep      # All hyperparams grid (slow)
#
# Each experiment runs on Setting B S1-S5 and saves CSV + prints results.
# Compare mean accuracy against baseline ContinualTTA = 84.33% (tau=0.02)
# or 86.1% (tau=0.05 from ablation).
#
# Output: results/improve/{exp_name}.csv
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

MODEL_PATH  = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
DATA_DIR    = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
RESULTS_DIR = r"/results/improve"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
NUM_CLASSES = 10
NUM_WORKERS = 2
SEVERITIES  = [1, 2, 3, 4, 5]

# Baselines for comparison
BASELINE_TAU002 = 84.33   # main experiment result, tau=0.02
BASELINE_TAU005 = 86.10   # ablation result, tau=0.05
SAR_BASELINE    = 84.75   # SAR from main experiment

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",    "impulse_noise",
    "defocus_blur",   "glass_blur",    "motion_blur",   "zoom_blur",
    "snow",           "frost",         "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",  "jpeg_compression",
]


# =============================================================================
# 1. DATASET
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


def loader(corruption, severity):
    return DataLoader(
        CIFAR10C_Dataset(corruption, severity),
        batch_size=BATCH_SIZE, shuffle=False,
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
    p = logits.softmax(1)
    return -(p * p.log()).sum(1)


def eval_loader(model_fn, dataloader):
    correct, total = 0, 0
    for x, y in dataloader:
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
# 3. JS SHIFT DETECTOR (shared, parametric)
# =============================================================================

class JSShiftDetector:
    def __init__(self, threshold=0.05, ema=0.9):
        self.threshold = threshold
        self.ema       = ema
        self.reference = None

    def should_adapt(self, logits):
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)
            if self.reference is None:
                self.reference = p_t.clone()
                return True
            m    = 0.5 * (self.reference + p_t)
            kl_1 = F.kl_div(m.log().unsqueeze(0),
                             self.reference.unsqueeze(0), reduction="batchmean")
            kl_2 = F.kl_div(m.log().unsqueeze(0),
                             p_t.unsqueeze(0), reduction="batchmean")
            js   = 0.5 * (kl_1 + kl_2)
            self.reference = self.ema * self.reference + (1 - self.ema) * p_t
            return js.item() > self.threshold


# =============================================================================
# 4. VARIANT FACTORY
#
# All variants controlled by keyword arguments.
# This keeps all logic in one place and makes comparison clean.
#
# Parameters:
#   tau           (float) : JS threshold                    [default 0.05]
#   js_ema        (float) : JS reference EMA decay          [default 0.9]
#   lr            (float) : Adam learning rate              [default 1e-3]
#   e_margin_factor(float): entropy margin = factor*ln(C)   [default 0.4]
#   use_proto     (bool)  : enable prototype consistency     [default True]
#   proto_alpha   (float) : prototype loss weight           [default 0.5]
#   proto_decay   (float) : prototype EMA decay             [default 0.9]
#   fix_detach    (bool)  : swap detach to prototypes       [default False]
#                           False = MSE(feat.detach(), proto)  [original — zero grad]
#                           True  = MSE(feat, proto.detach())  [fixed — grad flows]
#   conf_filter   (float) : min max-prob for proto update   [default 0.0 = off]
#                           0.8 means only update when max(softmax) > 0.8
#   conf_weight   (bool)  : weight proto update by confidence[default False]
# =============================================================================

class PrototypeBank(nn.Module):
    def __init__(self, num_classes, feat_dim, decay):
        super().__init__()
        self.decay = decay
        self.register_buffer("prototypes",  torch.zeros(num_classes, feat_dim))
        self.register_buffer("initialised", torch.zeros(num_classes).bool())

    @torch.no_grad()
    def update(self, features, pseudo_labels, conf_filter=0.0,
               conf_weight=False, probs=None):
        """
        Update prototypes with optional confidence filtering.

        conf_filter: minimum max probability to include a sample.
                     0.0 = include all (original behaviour).
                     0.8 = only high-confidence samples.

        conf_weight: weight contribution by confidence (soft version).
                     Requires probs to be passed.
        """
        for c in pseudo_labels.unique():
            mask = (pseudo_labels == c)
            feats = features[mask]

            if conf_filter > 0.0 and probs is not None:
                # Only keep high-confidence samples for this class
                max_probs = probs[mask].max(1).values
                conf_mask = max_probs > conf_filter
                if conf_mask.sum() == 0:
                    continue
                feats = feats[conf_mask]
                if conf_weight and probs is not None:
                    # Weighted mean by confidence
                    weights = max_probs[conf_mask]
                    weights = weights / weights.sum()
                    mf = (feats * weights.unsqueeze(1)).sum(0)
                else:
                    mf = feats.mean(0)
            elif conf_weight and probs is not None:
                max_probs = probs[mask].max(1).values
                weights   = max_probs / max_probs.sum()
                mf        = (feats * weights.unsqueeze(1)).sum(0)
            else:
                mf = feats.mean(0)

            if self.initialised[c]:
                self.prototypes[c] = (self.decay * self.prototypes[c]
                                      + (1 - self.decay) * mf)
            else:
                self.prototypes[c] = mf
                self.initialised[c] = True

    def consistency_loss(self, features, pseudo_labels, fix_detach=False):
        """
        Prototype consistency loss.

        fix_detach=False (original):
            MSE(features.detach(), prototypes)
            → gradient w.r.t. BN params = 0 (prototypes have no grad path)
            → CONTRIBUTES NOTHING to BN adaptation (this is the bug)

        fix_detach=True (corrected):
            MSE(features, prototypes.detach())
            → gradient flows through features to BN gamma/beta
            → BN params learn to produce features close to stored prototypes
            → This is the correct formulation
        """
        loss, count = torch.tensor(0.0, device=features.device), 0
        for c in pseudo_labels.unique():
            if not self.initialised[c]: continue
            mask  = (pseudo_labels == c)
            proto = self.prototypes[c].unsqueeze(0).expand(mask.sum(), -1)

            if fix_detach:
                # FIXED: gradient flows through features (not prototype)
                # prototype is the fixed target, features adapt toward it
                loss += F.mse_loss(features[mask], proto.detach())
            else:
                # ORIGINAL: zero gradient to BN — prototype loss is inert
                loss += F.mse_loss(features[mask].detach(), proto)

            count += 1
        return loss / max(count, 1)


def make_variant(source, **kwargs):
    """
    Build one ContinualTTA variant with specified hyperparameters.
    All kwargs have defaults matching the original published method.
    """
    # Parse all hyperparameters
    tau             = kwargs.get("tau",            0.05)
    js_ema          = kwargs.get("js_ema",         0.9)
    lr              = kwargs.get("lr",             1e-3)
    e_margin_factor = kwargs.get("e_margin_factor",0.4)
    use_proto       = kwargs.get("use_proto",      True)
    proto_alpha     = kwargs.get("proto_alpha",    0.5)
    proto_decay     = kwargs.get("proto_decay",    0.9)
    fix_detach      = kwargs.get("fix_detach",     False)
    conf_filter     = kwargs.get("conf_filter",    0.0)
    conf_weight     = kwargs.get("conf_weight",    False)

    e_margin = e_margin_factor * math.log(NUM_CLASSES)

    model, params = setup_bn(copy.deepcopy(source))
    detector      = JSShiftDetector(threshold=tau, ema=js_ema)
    opt           = torch.optim.Adam(params, lr=lr)

    bank     = PrototypeBank(NUM_CLASSES, 2048, proto_decay).to(DEVICE)
    captured = {}
    model.avgpool.register_forward_hook(
        lambda m, i, o: captured.update({"feat": o.flatten(1)}))

    @torch.enable_grad()
    def fn(x):
        logits        = model(x)
        features      = captured["feat"]
        probs         = logits.softmax(1)
        pseudo_labels = probs.argmax(1).detach()

        # Gate: JS shift detector
        if not detector.should_adapt(logits.detach()):
            return logits

        # Filter: low-entropy reliable samples
        entropy  = softmax_entropy(logits)
        reliable = entropy < e_margin
        if reliable.sum() == 0:
            return logits

        # Entropy loss on reliable samples
        loss = entropy[reliable].mean()

        # Optional: prototype consistency loss
        if use_proto:
            proto_loss = bank.consistency_loss(
                features[reliable],
                pseudo_labels[reliable],
                fix_detach=fix_detach)
            loss = loss + proto_alpha * proto_loss

        loss.backward()
        opt.step()
        opt.zero_grad()

        # Update prototype bank
        if use_proto:
            bank.update(
                features[reliable].detach(),
                pseudo_labels[reliable],
                conf_filter=conf_filter,
                conf_weight=conf_weight,
                probs=probs[reliable].detach() if (conf_filter > 0 or conf_weight) else None)

        return logits

    return fn


# =============================================================================
# 5. EVALUATION
# =============================================================================

def run_setting_b(source, variants_dict):
    """
    Run all variants in variants_dict through Setting B (S1-S5).
    Returns: {variant_name: mean_acc}
    """
    all_sev = {name: {} for name in variants_dict}

    for severity in SEVERITIES:
        print(f"\n  Severity {severity}")
        # Build fresh instances per severity
        fns = {name: make_variant(source, **kwargs)
               for name, kwargs in variants_dict.items()}

        for corruption in ALL_CORRUPTIONS:
            dl = loader(corruption, severity)
            for name in variants_dict:
                acc = eval_loader(fns[name], dl)
                all_sev[name][f"S{severity}_{corruption}"] = acc
            del dl
            torch.cuda.empty_cache()

            # Print progress
            line = f"    {corruption:<24}"
            for name in variants_dict:
                line += f"  {name[:8]}={all_sev[name][f'S{severity}_{corruption}']:.1f}%"
            print(line)

        # Severity summary
        for name in variants_dict:
            sev_mean = np.mean([all_sev[name][f"S{severity}_{c}"]
                                for c in ALL_CORRUPTIONS])
            print(f"    {'  '+name+' S'+str(severity):<28} mean={sev_mean:.2f}%")

    # Compute final S1-S5 mean per corruption and overall
    results = {}
    for name in variants_dict:
        corr_means = {}
        for c in ALL_CORRUPTIONS:
            corr_means[c] = np.mean([all_sev[name][f"S{s}_{c}"]
                                     for s in SEVERITIES])
        results[name] = {
            "per_corruption": corr_means,
            "mean": np.mean(list(corr_means.values()))
        }

    return results


# =============================================================================
# 6. SAVE AND PRINT
# =============================================================================

def print_results(results, exp_name):
    print(f"\n{'═'*65}")
    print(f"RESULTS — {exp_name}")
    print(f"{'═'*65}")
    print(f"  {'Variant':<30} {'Mean':>8}  {'vs Base(0.05)':>14}  {'vs SAR':>10}")
    print(f"  {'───────':<30} {'────':>8}  {'─────────────':>14}  {'──────':>10}")

    for name, res in sorted(results.items(), key=lambda x: -x[1]["mean"]):
        mean  = res["mean"]
        d005  = mean - BASELINE_TAU005
        dsar  = mean - SAR_BASELINE
        flag  = "  ← BEATS SAR!" if dsar > 0 else ""
        print(f"  {name:<30} {mean:>7.2f}%  {d005:>+13.2f}%  {dsar:>+9.2f}%{flag}")

    print(f"\n  Reference: ContinualTTA (tau=0.05) = {BASELINE_TAU005}%")
    print(f"  Reference: SAR                     = {SAR_BASELINE}%")
    print(f"{'═'*65}")


def save_results(results, exp_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{exp_name}.csv")
    with open(path, "w") as f:
        f.write("variant,mean," +
                ",".join(ALL_CORRUPTIONS) + "\n")
        for name, res in results.items():
            row = f"{name},{res['mean']:.4f}"
            for c in ALL_CORRUPTIONS:
                row += f",{res['per_corruption'][c]:.4f}"
            f.write(row + "\n")
    print(f"\n  Saved: {path}")
    return path


# =============================================================================
# 7. EXPERIMENTS
# =============================================================================

def exp_proto_fix(source):
    """
    EXP 1: Fix prototype gradient flow.

    The original implementation detaches features before the prototype loss,
    making it contribute zero gradient to BN params.
    The fix: detach the prototype (target) instead of the features (prediction).
    This allows gradient to flow through features → BN gamma/beta.
    """
    print("\nEXP: proto_fix — swap detach sides in prototype loss")
    print("  Original: MSE(features.detach(), prototypes)  → zero grad to BN")
    print("  Fixed:    MSE(features, prototypes.detach())  → grad flows to BN")

    variants = {
        "original_no_proto":     {"tau": 0.05, "use_proto": False},
        "original_with_proto":   {"tau": 0.05, "use_proto": True,  "fix_detach": False},
        "fixed_detach":          {"tau": 0.05, "use_proto": True,  "fix_detach": True},
    }
    return run_setting_b(source, variants)


def exp_conf_filter(source):
    """
    EXP 2: Confidence-filtered prototype update.

    Current: updates prototypes for all entropy-filtered samples.
    Fixed: only update when max(softmax) > threshold (high-confidence).
    This reduces pseudo-label noise corrupting the prototype centroids.
    """
    print("\nEXP: conf_filter — confidence threshold for prototype updates")
    print("  Tests: conf_filter = 0.0 (off), 0.6, 0.7, 0.8, 0.9")

    variants = {
        "no_proto":          {"tau": 0.05, "use_proto": False},
        "conf_0.0(original)":{"tau": 0.05, "use_proto": True, "fix_detach": True, "conf_filter": 0.0},
        "conf_0.6":          {"tau": 0.05, "use_proto": True, "fix_detach": True, "conf_filter": 0.6},
        "conf_0.7":          {"tau": 0.05, "use_proto": True, "fix_detach": True, "conf_filter": 0.7},
        "conf_0.8":          {"tau": 0.05, "use_proto": True, "fix_detach": True, "conf_filter": 0.8},
        "conf_0.9":          {"tau": 0.05, "use_proto": True, "fix_detach": True, "conf_filter": 0.9},
    }
    return run_setting_b(source, variants)


def exp_both_fixes(source):
    """
    EXP 3: Both prototype fixes together.

    Best guess at the optimal combined configuration.
    Compares against: no proto, original proto, each fix alone, both fixes.
    This is the most informative experiment — run this first if time is limited.
    """
    print("\nEXP: both_fixes — fix detach + confidence filter together")

    variants = {
        "no_proto":           {"tau": 0.05, "use_proto": False},
        "original_proto":     {"tau": 0.05, "use_proto": True, "fix_detach": False, "conf_filter": 0.0},
        "fix_detach_only":    {"tau": 0.05, "use_proto": True, "fix_detach": True,  "conf_filter": 0.0},
        "conf_only":          {"tau": 0.05, "use_proto": True, "fix_detach": False, "conf_filter": 0.8},
        "both_fixed_cf07":    {"tau": 0.05, "use_proto": True, "fix_detach": True,  "conf_filter": 0.7},
        "both_fixed_cf08":    {"tau": 0.05, "use_proto": True, "fix_detach": True,  "conf_filter": 0.8},
        "conf_weight":        {"tau": 0.05, "use_proto": True, "fix_detach": True,  "conf_filter": 0.0, "conf_weight": True},
    }
    return run_setting_b(source, variants)


def exp_threshold_sweep(source):
    """
    EXP 4: Fine-grained JS threshold sweep.

    Your ablation used coarser steps (0.01, 0.05, 0.1).
    This sweep fills in more values around the optimal region.
    Uses the best prototype configuration found in exp_both_fixes.
    """
    print("\nEXP: threshold_sweep — fine-grained tau values")
    print("  Sweeps tau from 0.01 to 0.10 in steps of 0.01")

    # Use best proto config — update these after running exp_both_fixes
    best_proto_kwargs = {"use_proto": True, "fix_detach": True, "conf_filter": 0.8}

    variants = {}
    for tau in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]:
        variants[f"tau_{tau:.2f}"] = {"tau": tau, **best_proto_kwargs}

    return run_setting_b(source, variants)


def exp_lr_sweep(source):
    """
    EXP 5: Learning rate sweep.

    Current lr=1e-3 was set for CIFAR-10.
    A lower lr might reduce over-adaptation within each corruption period.
    """
    print("\nEXP: lr_sweep — Adam learning rate")
    print("  Tests: 5e-4, 8e-4, 1e-3, 2e-3, 5e-3")

    variants = {
        "lr_5e-4":   {"tau": 0.05, "lr": 5e-4},
        "lr_8e-4":   {"tau": 0.05, "lr": 8e-4},
        "lr_1e-3":   {"tau": 0.05, "lr": 1e-3},   # current default
        "lr_2e-3":   {"tau": 0.05, "lr": 2e-3},
        "lr_5e-3":   {"tau": 0.05, "lr": 5e-3},
    }
    return run_setting_b(source, variants)


def exp_margin_sweep(source):
    """
    EXP 6: Entropy margin factor sweep.

    E_margin = factor * ln(C). Higher factor = more samples included.
    Current factor=0.4 was inherited from EATA paper.
    """
    print("\nEXP: margin_sweep — entropy margin factor")
    print("  Tests: 0.3, 0.35, 0.4, 0.45, 0.5, 0.6")

    variants = {}
    for factor in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        variants[f"factor_{factor}"] = {
            "tau": 0.05, "e_margin_factor": factor}

    return run_setting_b(source, variants)


def exp_ema_sweep(source):
    """
    EXP 7: JS reference EMA decay sweep.

    Current ema=0.9. Higher = reference changes slowly (more stable).
    Lower = reference updates faster (more sensitive to change).
    """
    print("\nEXP: ema_sweep — JS reference EMA decay")
    print("  Tests: 0.7, 0.8, 0.85, 0.9, 0.95, 0.99")

    variants = {}
    for ema in [0.70, 0.80, 0.85, 0.90, 0.95, 0.99]:
        variants[f"ema_{ema}"] = {"tau": 0.05, "js_ema": ema}

    return run_setting_b(source, variants)


def exp_proto_alpha_sweep(source):
    """
    EXP 8: Prototype loss weight alpha sweep.

    Current alpha=0.5. Only meaningful if fix_detach=True.
    """
    print("\nEXP: proto_alpha_sweep — prototype loss weight")
    print("  Tests: 0.1, 0.2, 0.5, 1.0, 2.0 (with fix_detach=True)")

    variants = {}
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0]:
        variants[f"alpha_{alpha}"] = {
            "tau": 0.05,
            "use_proto": True,
            "fix_detach": True,
            "conf_filter": 0.8,
            "proto_alpha": alpha}

    return run_setting_b(source, variants)


def exp_full_sweep(source):
    """
    EXP 9: Best configuration from all experiments combined.
    Run AFTER individual experiments to confirm optimal combination.
    """
    print("\nEXP: full_sweep — best configuration grid")
    print("  Update best_tau and best_cf after running earlier experiments")

    # UPDATE THESE after running exp_threshold_sweep and exp_conf_filter
    best_tau = 0.05   # update with best tau from threshold sweep
    best_cf  = 0.8    # update with best conf_filter from conf_filter sweep
    best_lr  = 1e-3   # update with best lr from lr sweep
    best_ema = 0.9    # update with best ema from ema sweep
    best_factor = 0.4 # update with best factor from margin sweep

    variants = {
        "baseline_no_proto":     {"tau": 0.05, "use_proto": False},
        "original_method":       {"tau": 0.02, "use_proto": True, "fix_detach": False},
        "ablation_method":       {"tau": 0.05, "use_proto": True, "fix_detach": False},
        "best_all":              {
            "tau":             best_tau,
            "js_ema":          best_ema,
            "lr":              best_lr,
            "e_margin_factor": best_factor,
            "use_proto":       True,
            "fix_detach":      True,
            "conf_filter":     best_cf,
            "conf_weight":     False,
        },
    }
    return run_setting_b(source, variants)


# =============================================================================
# 8. QUICK 1-SEVERITY VERSION
# Same experiments but only S3 (moderate severity) for fast iteration.
# Use this to screen candidates before running full S1-S5.
# =============================================================================

def run_s3_only(source, variants_dict):
    """Run only S3 for quick screening. ~1/5 the time of full run."""
    print("  [QUICK MODE: S3 only for screening]")
    results_s3 = {name: {} for name in variants_dict}
    fns = {name: make_variant(source, **kwargs)
           for name, kwargs in variants_dict.items()}

    for corruption in ALL_CORRUPTIONS:
        dl = loader(corruption, 3)
        for name in variants_dict:
            results_s3[name][corruption] = eval_loader(fns[name], dl)
        del dl
        torch.cuda.empty_cache()

    results = {}
    for name in variants_dict:
        results[name] = {
            "per_corruption": results_s3[name],
            "mean": np.mean(list(results_s3[name].values()))
        }
    return results


# =============================================================================
# 9. MAIN
# =============================================================================

EXPERIMENTS = {
    "proto_fix":        exp_proto_fix,
    "conf_filter":      exp_conf_filter,
    "both_fixes":       exp_both_fixes,      # ← START HERE
    "threshold_sweep":  exp_threshold_sweep,
    "lr_sweep":         exp_lr_sweep,
    "margin_sweep":     exp_margin_sweep,
    "ema_sweep":        exp_ema_sweep,
    "proto_alpha":      exp_proto_alpha_sweep,
    "full_sweep":       exp_full_sweep,
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ContinualTTA improvement experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended order:
  1. python improve.py --exp both_fixes           # ~6h on T4 — most informative
  2. python improve.py --exp threshold_sweep       # ~10h on T4
  3. python improve.py --exp lr_sweep              # ~5h on T4
  4. python improve.py --exp full_sweep            # ~4h — best combination

Quick screen (S3 only, ~1/5 time):
  python improve.py --exp both_fixes --quick
  python improve.py --exp threshold_sweep --quick
        """)

    parser.add_argument("--exp",   type=str, required=True,
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--quick", action="store_true",
                        help="Run S3 only for fast screening (~1/5 time)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"ContinualTTA Improvement Study")
    print(f"{'='*60}")
    print(f"Device     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Experiment : {args.exp}")
    print(f"Mode       : {'QUICK (S3 only)' if args.quick else 'FULL (S1-S5)'}")
    print(f"Baseline   : ContinualTTA(tau=0.05) = {BASELINE_TAU005}%")
    print(f"Target     : SAR = {SAR_BASELINE}%")

    print("\nLoading source model...")
    source = load_model()
    print(f"Parameters : {sum(p.numel() for p in source.parameters()):,}")

    # Get the experiment's variant dictionary
    exp_fn = EXPERIMENTS[args.exp]

    # Run the experiment
    # We need to get the variants dict from the function
    # Instead of calling run_setting_b inside exp_fn,
    # call exp_fn which internally calls run_setting_b or run_s3_only

    if args.quick:
        # Override run_setting_b with run_s3_only for quick mode
        import builtins
        _orig = globals()["run_setting_b"]
        globals()["run_setting_b"] = run_s3_only

    results = exp_fn(source)

    if args.quick:
        globals()["run_setting_b"] = _orig

    print_results(results, args.exp)
    path = save_results(results, args.exp)

    print(f"\n{'='*60}")
    print(f"DONE — {args.exp}")
    print(f"{'='*60}")
    print(f"Results: {os.path.abspath(path)}")

    # Final recommendation
    best_name = max(results.keys(), key=lambda k: results[k]["mean"])
    best_mean = results[best_name]["mean"]
    delta_sar = best_mean - SAR_BASELINE

    print(f"\nBest variant: {best_name}")
    print(f"Best mean:    {best_mean:.2f}%")
    print(f"vs SAR:       {delta_sar:+.2f}%")

    if delta_sar > 0:
        print(f"\n✓ BEATS SAR by {delta_sar:.2f}%")
        print(f"  → Use this as your primary method in Table 1")
        print(f"  → Update SETTING_B_MEANS in Setting A script")
    elif delta_sar > -0.5:
        print(f"\n≈ Within 0.5% of SAR — essentially tied")
        print(f"  → Emphasise order robustness and interpretability")
    else:
        print(f"\n→ Still behind SAR. Try combining best configs in full_sweep")