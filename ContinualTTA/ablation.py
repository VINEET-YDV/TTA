# =============================================================================
# ContinualTTA — Component Ablation Study (WACV 2027)
#
# Run from terminal:
#   python ablation.py --ablation A0 --dataset cifar10c --setting b
#   python ablation.py --ablation A0 --dataset cifar10c --setting a
#   python ablation.py --ablation A0 --dataset imagenetc
#
# Ablations (each removes exactly one component from full ContinualTTA):
#   A0  Full ContinualTTA          — all components (reference)
#   A1  No JS Detector             — always adapt (remove gating)
#   A2  No Reliable Filter         — use all samples regardless of entropy
#   A3  No Prototype Bank          — entropy loss only, no consistency term
#   A4  No Prototype + No Filter   — plain entropy minimisation (≈ TENT)
#   A5  KL Detector                — replace JS with KL divergence
#   A6  Entropy Detector           — replace JS with entropy threshold
#   A7  EMA Decay β=0.5            — weaker prototype memory
#   A8  EMA Decay β=0.99           — stronger prototype memory
#   A9  JS Threshold τ=0.01        — more aggressive adaptation
#   A10 JS Threshold τ=0.05        — more conservative adaptation
#
# All ablations:
#   - Use identical BN-only adaptation (setup_bn_cifar / setup_bn_imagenet)
#   - Use identical optimiser (Adam, lr=1e-3)
#   - Differ in EXACTLY ONE component from A0
#   - Save individual CSVs — merge later with --merge flag
#
# Output:
#   results/ablations/A0_Full.csv
#   results/ablations/A1_NoDetector.csv
#   ...
#   results/ablations/ablation_table.tex  (generated after all runs via --merge)
#
# Improvements in PrototypeBankModule (vs original):
#   1. Vectorized scatter_add_ update — no Python loop over classes
#   2. L2-normalized features on unit sphere — scale-invariant
#   3. Confidence-weighted EMA update — uncertain samples contribute less
#   4. Cosine consistency loss — scale-invariant, better than MSE
#   5. Cold-start vs warm-update explicitly separated
#   6. Per-class step counter for future bias-correction use
#   7. Prototype re-normalized to unit sphere after every EMA step
# =============================================================================

from __future__ import annotations

import copy
import math
import os
import argparse
import platform
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder


# =============================================================================
# CONFIG — update paths before running
# =============================================================================

CIFAR_MODEL_PATH  = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\resnet50_cifar10_source.pth"
CIFAR_DATA_DIR    = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\CIFAR-10-C\CIFAR-10-C"
IMAGENET_DATA_DIR = r"C:\Users\vinee\OneDrive\Desktop\ContinualTTA\ImageNet-C"

RESULTS_DIR = os.path.join("results", "ablations")
NUM_WORKERS = 0 if platform.system() == "Windows" else 2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# CIFAR-10-C settings
CIFAR_BATCH_SIZE = 32
CIFAR_CLASSES    = 10
CIFAR_SEVERITIES = [1, 2, 3, 4, 5]
CIFAR_SEED       = 42

# ImageNet-C settings
IMAGENET_BATCH_SIZE = 64
IMAGENET_CLASSES    = 1000
IMAGENET_SEVERITY   = 5

# Shared hyperparameters — identical across ALL ablations
LR              = 1e-3
ALPHA           = 0.5     # weight of prototype consistency loss
JS_THRESHOLD    = 0.04    # optimal JS threshold from sensitivity study
EMA_DECAY       = 0.9     # default prototype EMA decay β
E_MARGIN_FACTOR = 0.4     # entropy margin = factor * log(C)
SAR_RHO         = 0.05    # unused currently, kept for compatibility

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise",      "impulse_noise",
    "defocus_blur",   "glass_blur",      "motion_blur",   "zoom_blur",
    "snow",           "frost",           "fog",           "brightness",
    "contrast",       "elastic_transform", "pixelate",    "jpeg_compression",
]


# =============================================================================
# ABLATION REGISTRY
# Each entry: (id, name, description, kwargs_override)
# kwargs_override modifies EXACTLY ONE component from A0.
# =============================================================================

ABLATIONS = [
    ("A0",  "Full",               "All components (reference)",                   {}),
    ("A1",  "NoDetector",         "Always adapt — no JS gating",                  {"use_detector": False}),
    ("A2",  "NoFilter",           "No entropy filter — all samples used",          {"use_filter": False}),
    ("A3",  "NoPrototype",        "No prototype bank — entropy loss only",         {"use_prototype": False}),
    ("A4",  "NoProtoNoFilter",    "No prototype + no filter (≈ TENT)",             {"use_prototype": False,
                                                                                    "use_filter":    False}),
    ("A5",  "KLDetector",         "KL divergence detector instead of JS",          {"detector_type": "kl"}),
    ("A6",  "EntropyDetector",    "Entropy threshold detector",                    {"detector_type": "entropy"}),
    ("A7",  "WeakMemory",         "Weaker EMA decay β=0.5",                        {"ema_decay": 0.5}),
    ("A8",  "StrongMemory",       "Stronger EMA decay β=0.99",                     {"ema_decay": 0.99}),
    ("A9",  "AggressiveGating",   "More aggressive JS threshold τ=0.01",           {"js_threshold": 0.01}),
    ("A10", "ConservativeGating", "More conservative JS threshold τ=0.05",         {"js_threshold": 0.05}),
]


# =============================================================================
# 1. DATASETS
# =============================================================================

class CIFAR10C_Dataset(Dataset):
    """Single corruption × severity slice from CIFAR-10-C."""

    def __init__(self, corruption: str, severity: int, data_dir: str) -> None:
        data         = np.load(os.path.join(data_dir, f"{corruption}.npy"), mmap_mode="r")
        labels       = np.load(os.path.join(data_dir, "labels.npy"),        mmap_mode="r")
        start        = (severity - 1) * 10_000
        self.images  = data[start : start + 10_000]
        self.labels  = labels[start : start + 10_000]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.transform(Image.fromarray(self.images[idx])), int(self.labels[idx])


def cifar_loader_single(corruption: str, severity: int) -> DataLoader:
    return DataLoader(
        CIFAR10C_Dataset(corruption, severity, CIFAR_DATA_DIR),
        batch_size=CIFAR_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )


def cifar_loader_mixed(severity: int, seed: int = CIFAR_SEED) -> DataLoader:
    """All 15 corruptions concatenated and randomly permuted (i.i.d. Setting A)."""
    combined = ConcatDataset(
        [CIFAR10C_Dataset(c, severity, CIFAR_DATA_DIR) for c in ALL_CORRUPTIONS]
    )
    g = torch.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(len(combined), generator=g).tolist()
    subset  = torch.utils.data.Subset(combined, indices)
    return DataLoader(
        subset, batch_size=CIFAR_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )


def imagenet_loader(corruption: str) -> DataLoader:
    path    = os.path.join(IMAGENET_DATA_DIR, corruption, str(IMAGENET_SEVERITY))
    dataset = ImageFolder(path, transform=models.ResNet50_Weights.IMAGENET1K_V1.transforms())
    return DataLoader(
        dataset, batch_size=IMAGENET_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )


def load_cifar_model() -> nn.Module:
    model    = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, CIFAR_CLASSES)
    model.load_state_dict(torch.load(CIFAR_MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


def load_imagenet_model() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return model.to(DEVICE).eval()


# =============================================================================
# 2. HELPERS
# =============================================================================

def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample Shannon entropy of the softmax distribution. Shape: (N,)"""
    p = logits.softmax(dim=1)
    return -(p * p.log()).sum(dim=1)


def eval_loader(model_fn, loader: DataLoader) -> float:
    """Run model_fn over loader and return top-1 accuracy (%)."""
    correct = total = 0
    for x, y in loader:
        x, y     = x.to(DEVICE), y.to(DEVICE)
        logits   = model_fn(x)
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total


def setup_bn_cifar(model: nn.Module):
    """
    CIFAR setup: compute BN stats from each batch (no running stats).
    Only BN gamma/beta are trainable.
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean        = None
            m.running_var         = None
    params = [
        p for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
        for p in m.parameters() if p.requires_grad
    ]
    return model, params


def setup_bn_imagenet(model: nn.Module):
    """
    ImageNet setup: keep pretrained BN running stats frozen (momentum=0),
    only BN gamma/beta are trainable.
    """
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.requires_grad_(True)
            m.track_running_stats = True
            m.momentum            = 0
    params = [
        p for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
        for p in m.parameters() if p.requires_grad
    ]
    return model, params


# =============================================================================
# 3. SHIFT DETECTOR
# =============================================================================

class ShiftDetector:
    """
    Detects distribution shift between incoming batch and a reference distribution.

    Supported detector types:
        'js'      — Jensen-Shannon divergence (symmetric, bounded in [0, ln2])
        'kl'      — KL divergence (asymmetric, unbounded)
        'entropy' — batch-level entropy threshold
        'none'    — always adapt (no gating)

    The reference distribution is maintained as an EMA of past batch marginals,
    which allows the detector to track slow drift while staying sensitive to
    abrupt shifts.

    Note on threshold choice:
        JS is bounded in [0, ln2 ≈ 0.693]. A threshold of 0.04 corresponds to
        roughly 6% of the maximum possible JS divergence — conservative enough
        to avoid spurious adaptation on clean data.
    """

    def __init__(
        self,
        detector_type: str  = "js",
        threshold:     float = JS_THRESHOLD,
        ema:           float = 0.9,
    ) -> None:
        self.detector_type = detector_type
        self.threshold     = threshold
        self.ema           = ema
        self.reference: Optional[torch.Tensor] = None
        self.ema_entropy:  Optional[float]     = None

    def should_adapt(self, logits: torch.Tensor) -> bool:
        """
        Decide whether the current batch warrants adaptation.

        Args:
            logits: (N, C) — raw model outputs (detached, no grad needed)

        Returns:
            True  → adapt this batch
            False → skip adaptation, return cached logits
        """
        with torch.no_grad():
            p_t = logits.softmax(1).mean(0)   # batch marginal, shape (C,)

            # ── No detector — always adapt ────────────────────────────────────
            if self.detector_type == "none":
                return True

            # ── Entropy detector ──────────────────────────────────────────────
            if self.detector_type == "entropy":
                h = -(p_t * p_t.log().clamp(min=-1e9)).sum().item()
                if self.ema_entropy is None:
                    self.ema_entropy = h
                adapt = h > self.threshold * math.log(logits.size(1))
                self.ema_entropy = 0.9 * self.ema_entropy + 0.1 * h
                return adapt

            # ── JS / KL detector — need a reference distribution ──────────────
            if self.reference is None:
                # First batch: initialise reference, always adapt
                self.reference = p_t.clone()
                return True

            p_ref = self.reference

            if self.detector_type == "js":
                # Jensen-Shannon divergence — symmetric, bounded in [0, ln2]
                m     = 0.5 * (p_ref + p_t)
                kl_1  = F.kl_div(m.log().unsqueeze(0),
                                  p_ref.unsqueeze(0), reduction="batchmean")
                kl_2  = F.kl_div(m.log().unsqueeze(0),
                                  p_t.unsqueeze(0),   reduction="batchmean")
                divergence = 0.5 * (kl_1 + kl_2)

            elif self.detector_type == "kl":
                # KL(p_ref || p_t) — asymmetric, unbounded
                divergence = F.kl_div(
                    p_t.log().unsqueeze(0),
                    p_ref.unsqueeze(0),
                    reduction="batchmean",
                )

            else:
                raise ValueError(f"Unknown detector_type: {self.detector_type!r}")

            # Update reference AFTER divergence computation
            # (ensures comparison is always against the pre-batch reference)
            self.reference = self.ema * self.reference + (1.0 - self.ema) * p_t

            return divergence.item() > self.threshold


# =============================================================================
# 4. PROTOTYPE BANK MODULE (improved)
# =============================================================================

class PrototypeBankModule(nn.Module):
    """
    EMA prototype memory bank — vectorized, confidence-weighted, cosine-loss.

    Design decisions vs original implementation:
    ──────────────────────────────────────────────────────────────────────────
    1. Vectorized update (scatter_add_)
       Original used a Python for-loop over unique classes, which is O(C)
       Python iterations per batch. This becomes expensive for ImageNet (C=1000).
       We use scatter_add_ to accumulate weighted features in one CUDA kernel.

    2. L2-normalized features
       Raw avgpool features (2048-d) have varying L2 norms across batches.
       EMA prototypes built from unnormalized features drift in magnitude,
       making MSE loss scale-sensitive. We project everything to the unit
       sphere before update and loss computation.

    3. Confidence-weighted update
       Each sample's contribution is weighted by its max-softmax confidence.
       High-entropy (uncertain) pseudo-labels contaminate the prototype less.
       If logits are not provided, weights default to uniform.

    4. Cosine consistency loss
       MSE conflates direction and magnitude. On a unit sphere, cosine
       distance (1 - cosine_similarity) isolates directional alignment and
       is invariant to feature scale — better suited after L2 normalization.

    5. Explicit cold-start vs warm-update separation
       On the first batch for a class, the prototype is hard-initialized
       (no decay). Subsequent batches use EMA. This is the same semantics
       as the original but made explicit and vectorized.

    6. Per-class step counter (ema_steps)
       Tracks how many updates each class prototype has received.
       Available for future bias-correction (like Adam's 1/(1-β^t) term)
       without requiring any changes to the update interface.

    7. Prototype re-normalization after EMA
       EMA of unit vectors is NOT generally a unit vector. We re-normalize
       after each update so prototypes stay on the unit sphere and cosine
       distances remain geometrically meaningful.

    Gradient path (important for BN adaptation):
       consistency_loss receives live features (not detached) so gradients
       flow: cosine_loss → features → BN gamma/beta.
       Prototype targets ARE detached — they are fixed reference points.
       Bank update uses features.detach() to avoid contaminating the graph.

    Usage in fn():
       proto_loss = bank.consistency_loss(features[reliable], pseudo_labels[reliable])
       loss = entropy_loss + ALPHA * proto_loss
       loss.backward()          # gradients flow to BN params via features
       opt.step()
       bank.update(features[reliable].detach(), pseudo_labels[reliable], logits[reliable])
    """

    def __init__(
        self,
        num_classes: int,
        feat_dim:    int,
        ema_decay:   float = EMA_DECAY,
    ) -> None:
        super().__init__()
        self.C     = num_classes
        self.D     = feat_dim
        self.decay = ema_decay

        # Persistent buffers — survive deepcopy and device moves
        self.register_buffer("prototypes",  torch.zeros(num_classes, feat_dim))
        self.register_buffer("ema_steps",   torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("initialised", torch.zeros(num_classes, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Internal utility
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Row-wise L2 normalization. Safe for zero-vectors via eps."""
        return F.normalize(x, p=2, dim=-1, eps=eps)

    # ------------------------------------------------------------------
    # Vectorized, confidence-weighted EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(
        self,
        features:      torch.Tensor,
        pseudo_labels: torch.Tensor,
        logits:        Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update prototype bank for all classes present in this batch.

        Args:
            features      : (N, D) raw avgpool features — pass detached
            pseudo_labels : (N,)   argmax pseudo-labels, dtype long
            logits        : (N, C) optional — used to compute per-sample
                            confidence weights. If None, uniform weights.

        Implementation note — scatter_add_:
            weighted_feats[pseudo_labels[i]] += feats[i] * weights[i]
        This replaces the O(C) Python loop with a single fused operation.
        """
        feats = self._l2_norm(features.float())   # (N, D), unit sphere

        # Per-sample confidence weights ────────────────────────────────
        if logits is not None:
            # max softmax probability for each sample's predicted class
            conf = logits.detach().float().softmax(dim=1)              # (N, C)
            conf = conf.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1)  # (N,)
        else:
            conf = torch.ones(feats.size(0), device=feats.device)

        # Scatter: accumulate weighted features per class ──────────────
        weight_sum     = torch.zeros(self.C,       device=feats.device)   # (C,)
        weighted_feats = torch.zeros(self.C, self.D, device=feats.device) # (C, D)

        weight_sum.scatter_add_(
            0,
            pseudo_labels,
            conf,
        )
        weighted_feats.scatter_add_(
            0,
            pseudo_labels.unsqueeze(1).expand(-1, self.D),  # (N, D) index
            feats * conf.unsqueeze(1),                       # (N, D) value
        )

        # Active classes: appeared in this batch with nonzero weight
        active = weight_sum > 0   # (C,) bool

        if not active.any():
            return

        # Weighted mean for active classes, re-normalized
        batch_mean = torch.zeros_like(self.prototypes)         # (C, D)
        batch_mean[active] = weighted_feats[active] / weight_sum[active].unsqueeze(1)
        batch_mean[active] = self._l2_norm(batch_mean[active]) # back to unit sphere

        # Separate cold-start (first time) from warm update (EMA)
        cold = active & ~self.initialised    # (C,) bool
        warm = active &  self.initialised    # (C,) bool

        # Hard-initialize on first batch for each class
        if cold.any():
            self.prototypes[cold] = batch_mean[cold]

        # EMA update for classes seen before
        if warm.any():
            self.prototypes[warm] = (
                self.decay       * self.prototypes[warm]
                + (1.0 - self.decay) * batch_mean[warm]
            )

        # Re-normalize after EMA: EMA of unit vectors ≠ unit vector
        self.prototypes[active] = self._l2_norm(self.prototypes[active])

        # Bookkeeping
        self.ema_steps[active]   += 1
        self.initialised[active]  = True

    # ------------------------------------------------------------------
    # Cosine consistency loss
    # ------------------------------------------------------------------

    def consistency_loss(
        self,
        features:      torch.Tensor,
        pseudo_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine distance between live features and frozen prototype targets.

        Why cosine instead of MSE:
            Prototypes live on the unit sphere (L2-normalized in update()).
            MSE conflates directional error with magnitude error; cosine
            isolates angular alignment, which is what we care about.
            loss = 1 - cosine_similarity ∈ [0, 2]; perfect alignment → 0.

        Gradient path:
            features (NOT detached) → BN gamma/beta     [requires_grad=True]
            prototypes.detach()     → fixed target       [no gradient]

        Only samples whose class prototype is initialised contribute to the
        loss — avoids pulling features toward the zero prototype.

        Args:
            features      : (N, D) live avgpool features, NOT detached
            pseudo_labels : (N,)   argmax pseudo-labels, dtype long

        Returns:
            Scalar cosine consistency loss.
        """
        # Normalize live features — gradient still flows through F.normalize
        feats_norm = self._l2_norm(features.float())   # (N, D)

        # Gather frozen prototype targets for each sample
        # Prototypes are already L2-normalized from update()
        targets = self.prototypes[pseudo_labels].detach()  # (N, D), no grad

        # Mask: only classes with an initialised prototype
        valid = self.initialised[pseudo_labels]   # (N,) bool
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=False)

        cos_sim = F.cosine_similarity(
            feats_norm[valid],
            targets[valid],
            dim=-1,
        )   # (M,) ∈ [-1, 1]

        return (1.0 - cos_sim).mean()   # ∈ [0, 2], lower is better


# =============================================================================
# 5. ABLATION VARIANT FACTORY
# =============================================================================

def make_ablation_variant(
    source:      nn.Module,
    setup_bn_fn,
    num_classes: int,
    feat_dim:    int,
    **kwargs,
):
    """
    Build a ContinualTTA forward function with specific components
    enabled or disabled according to kwargs.

    This is the single point of truth for all ablation variants —
    each ablation changes exactly one kwarg from A0 defaults.

    kwargs (all optional, fall back to module-level defaults):
        use_detector   (bool)  : gate adaptation with shift detector  [default: True]
        use_filter     (bool)  : apply entropy-based reliable filter   [default: True]
        use_prototype  (bool)  : use prototype consistency loss        [default: True]
        detector_type  (str)   : 'js' | 'kl' | 'entropy' | 'none'    [default: 'js']
        js_threshold   (float) : JS / KL detection threshold          [default: JS_THRESHOLD]
        ema_decay      (float) : prototype EMA decay β                [default: EMA_DECAY]

    Returns:
        fn — callable: (x: Tensor) → logits: Tensor
             Performs one step of test-time adaptation and returns logits.
        fn._handle — forward hook handle; call fn._handle.remove() to clean up.

    Design note — update after backward:
        bank.update() is called AFTER loss.backward() and opt.step().
        This means the prototype used in consistency_loss() is always
        one batch stale — intentional. The prototype is a fixed reference
        target; updating it mid-forward would create a moving-target problem.
        The detach() in update() ensures no graph contamination.
    """
    use_detector  = kwargs.get("use_detector",  True)
    use_filter    = kwargs.get("use_filter",    True)
    use_prototype = kwargs.get("use_prototype", True)
    detector_type = kwargs.get("detector_type", "js") if use_detector else "none"
    js_threshold  = kwargs.get("js_threshold",  JS_THRESHOLD)
    ema_decay     = kwargs.get("ema_decay",      EMA_DECAY)

    e_margin = E_MARGIN_FACTOR * math.log(num_classes)

    # Build all components on a fresh copy of the source model
    model, params = setup_bn_fn(copy.deepcopy(source))
    opt           = torch.optim.Adam(params, lr=LR)
    detector      = ShiftDetector(
        detector_type=detector_type,
        threshold=js_threshold,
    )
    bank = PrototypeBankModule(
        num_classes=num_classes,
        feat_dim=feat_dim,
        ema_decay=ema_decay,
    ).to(DEVICE)

    # Hook to capture avgpool output (2048-d features before FC)
    captured: dict[str, torch.Tensor] = {}
    handle = model.avgpool.register_forward_hook(
        lambda m, i, o: captured.update({"feat": o.flatten(1)})
    )

    @torch.enable_grad()
    def fn(x: torch.Tensor) -> torch.Tensor:
        logits        = model(x)                        # forward pass
        features      = captured["feat"]               # (N, D), has grad
        pseudo_labels = logits.argmax(1).detach()      # (N,), no grad

        # ── Gate 1: shift detector ────────────────────────────────────
        # If JS divergence from reference is below threshold, the current
        # batch likely comes from the same distribution — skip adaptation.
        if not detector.should_adapt(logits.detach()):
            return logits

        # ── Gate 2: reliable sample filter ───────────────────────────
        # Keep only samples with entropy below e_margin.
        # High-entropy samples have diffuse predictions — poor pseudo-labels
        # that can destabilize BN parameters.
        if use_filter:
            entropy  = softmax_entropy(logits)
            reliable = entropy < e_margin              # (N,) bool mask
            if reliable.sum() == 0:
                return logits                          # nothing reliable — skip
        else:
            entropy  = softmax_entropy(logits)
            reliable = torch.ones(x.size(0), dtype=torch.bool, device=DEVICE)

        # ── Loss 1: entropy minimisation ──────────────────────────────
        loss = entropy[reliable].mean()

        # ── Loss 2: prototype consistency ─────────────────────────────
        # features[reliable] retains gradient → flows to BN gamma/beta
        if use_prototype:
            proto_loss = bank.consistency_loss(
                features[reliable],
                pseudo_labels[reliable],
            )
            loss = loss + ALPHA * proto_loss

        # ── BN parameter update ───────────────────────────────────────
        loss.backward()
        opt.step()
        opt.zero_grad()

        # ── Prototype bank update (post-backward) ─────────────────────
        # Detach features so the bank holds no reference to the compute graph.
        # logits are passed for confidence weighting.
        if use_prototype:
            bank.update(
                features[reliable].detach(),
                pseudo_labels[reliable],
                logits[reliable].detach(),
            )

        return logits

    fn._handle = handle   # expose handle so caller can remove the hook
    return fn


# =============================================================================
# 6. EVALUATION LOOPS
# =============================================================================

def eval_cifar_setting_b(
    source_model: nn.Module,
    fn,
) -> dict[str, float]:
    """
    Setting B: continual sequential corruption stream.
    Evaluates 15 corruptions × 5 severities; returns per-corruption mean over
    severities. The same fn (and its adaptation state) is used throughout —
    simulating a continually changing environment.
    """
    all_sev: dict[int, dict[str, float]] = {}
    for severity in CIFAR_SEVERITIES:
        results: dict[str, float] = {}
        for corruption in ALL_CORRUPTIONS:
            loader = cifar_loader_single(corruption, severity)
            results[corruption] = eval_loader(fn, loader)
            del loader
            torch.cuda.empty_cache()
        all_sev[severity] = results

    # Average accuracy over severities for each corruption
    averaged: dict[str, float] = {}
    for corruption in ALL_CORRUPTIONS:
        averaged[corruption] = float(np.mean(
            [all_sev[s][corruption] for s in CIFAR_SEVERITIES]
        ))
    return averaged


def eval_cifar_setting_a(
    source_model: nn.Module,
    setup_bn_fn,
    num_classes:  int,
    feat_dim:     int,
    **kwargs,
) -> tuple[float, dict[int, float]]:
    """
    Setting A: mixed i.i.d. stream per severity.
    A fresh model+bank is created for each severity level so there is no
    cross-severity state leak.

    Returns:
        mean_acc   — mean accuracy averaged over 5 severity levels
        sev_results — per-severity accuracy dict {severity: acc}
    """
    sev_results: dict[int, float] = {}
    for severity in CIFAR_SEVERITIES:
        fn_s   = make_ablation_variant(source_model, setup_bn_fn,
                                       num_classes, feat_dim, **kwargs)
        loader = cifar_loader_mixed(severity)
        sev_results[severity] = eval_loader(fn_s, loader)
        del loader
        torch.cuda.empty_cache()
    mean_acc = float(np.mean(list(sev_results.values())))
    return mean_acc, sev_results


def eval_imagenet_sequential(
    source_model: nn.Module,
    fn,
) -> dict[str, float]:
    """
    ImageNet-C: continual sequential stream at severity 5.
    One corruption at a time, 15 corruptions total.
    """
    results: dict[str, float] = {}
    for corruption in ALL_CORRUPTIONS:
        loader = imagenet_loader(corruption)
        results[corruption] = eval_loader(fn, loader)
        del loader
        torch.cuda.empty_cache()
    return results


# =============================================================================
# 7. MAIN RUNNER
# =============================================================================

def run_ablation(
    ablation_id:  str,
    dataset:      str,
    setting:      str,
    source_model: nn.Module,
    setup_bn_fn,
    num_classes:  int,
    feat_dim:     int,
) -> tuple[dict, float]:
    """
    Run one ablation variant end-to-end and save results to a CSV file.

    Args:
        ablation_id  : one of 'A0'..'A10'
        dataset      : 'cifar10c' or 'imagenetc'
        setting      : 'a' or 'b' (cifar10c only)
        source_model : pretrained source model (not modified)
        setup_bn_fn  : setup_bn_cifar or setup_bn_imagenet
        num_classes  : 10 (CIFAR) or 1000 (ImageNet)
        feat_dim     : 2048 for ResNet-50

    Returns:
        results  : {corruption: accuracy} dict
        mean_acc : mean accuracy over all corruptions
    """
    config = next((a for a in ABLATIONS if a[0] == ablation_id), None)
    if config is None:
        raise ValueError(
            f"Unknown ablation ID: {ablation_id!r}. "
            f"Available: {[a[0] for a in ABLATIONS]}"
        )

    abl_id, abl_name, abl_desc, abl_kwargs = config

    print(f"\n{'='*60}")
    print(f"Ablation   : {abl_id} — {abl_name}")
    print(f"Description: {abl_desc}")
    print(f"Kwargs     : {abl_kwargs if abl_kwargs else 'none (full method)'}")
    print(f"Dataset    : {dataset.upper()}  |  Setting: {setting.upper()}")
    print(f"{'='*60}\n")

    # Build the ablation forward function (fresh state)
    fn = make_ablation_variant(
        source_model, setup_bn_fn, num_classes, feat_dim, **abl_kwargs
    )

    # Run the appropriate evaluation loop
    if dataset == "cifar10c":
        if setting == "b":
            results = eval_cifar_setting_b(source_model, fn)
        else:
            mean_a, sev_results = eval_cifar_setting_a(
                source_model, setup_bn_fn, num_classes, feat_dim, **abl_kwargs
            )
            results = {f"S{s}": sev_results[s] for s in CIFAR_SEVERITIES}
            results["Mean"] = mean_a
    elif dataset == "imagenetc":
        results = eval_imagenet_sequential(source_model, fn)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    # Clean up the forward hook
    fn._handle.remove()

    # Compute mean (excluding the pre-computed 'Mean' key if present)
    mean_acc = float(np.mean(
        [v for k, v in results.items() if k != "Mean"]
    ))

    # Print per-corruption results
    print(f"\nResults — {abl_id} ({abl_name}):")
    for k, v in results.items():
        if k != "Mean":
            print(f"  {k:<26}  {v:.2f}%")
    print(f"  {'Mean':<26}  {mean_acc:.2f}%")

    # Save to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{abl_id}_{abl_name}.csv")
    with open(csv_path, "w") as f:
        f.write(f"corruption,{abl_id}_{abl_name}\n")
        for k, v in results.items():
            if k != "Mean":
                f.write(f"{k},{v:.2f}\n")
        f.write(f"Mean,{mean_acc:.2f}\n")
    print(f"\nSaved: {csv_path}")

    return results, mean_acc


# =============================================================================
# 8. MERGE AND LATEX TABLE
# =============================================================================

def merge_ablations_and_latex(dataset: str = "cifar10c", setting: str = "b") -> None:
    """
    Merge all individual ablation CSVs into one summary table and generate
    a LaTeX table suitable for direct inclusion in an Overleaf project.

    Run after all ablations have completed:
        python ablation.py --merge --dataset cifar10c --setting b
    """
    print("\nMerging ablation results...")

    all_results: dict[str, dict[str, float]] = {}
    all_means:   dict[str, float]            = {}

    for abl_id, abl_name, _, _ in ABLATIONS:
        csv_path = os.path.join(RESULTS_DIR, f"{abl_id}_{abl_name}.csv")
        if not os.path.isfile(csv_path):
            print(f"  MISSING: {csv_path} — skipping")
            continue

        with open(csv_path) as f:
            lines = f.readlines()

        results: dict[str, float] = {}
        for line in lines[1:]:   # skip header
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            if parts[0] == "Mean":
                all_means[f"{abl_id}_{abl_name}"] = float(parts[1])
            else:
                results[parts[0]] = float(parts[1])
        all_results[f"{abl_id}_{abl_name}"] = results

    if not all_results:
        print("No results found. Run ablations first.")
        return

    # ── Console table ─────────────────────────────────────────────────────────
    col   = 12
    names = list(all_results.keys())
    header_line = f"{'Corruption':<24}" + "".join(f"{n[:10]:>{col}}" for n in names)
    sep         = "═" * len(header_line)
    print(f"\n{sep}")
    print("ABLATION STUDY — merged results")
    print(sep)
    print(header_line)
    print("─" * len(header_line))
    for corruption in ALL_CORRUPTIONS:
        row = f"{corruption:<24}"
        for name in names:
            val  = all_results[name].get(corruption, float("nan"))
            row += f"{val:.1f}%".rjust(col)
        print(row)
    print("─" * len(header_line))
    mean_row = f"{'Mean':<24}"
    for name in names:
        mean_row += f"{all_means.get(name, float('nan')):.1f}%".rjust(col)
    print(mean_row)
    print(sep)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    corr_display = {
        "gaussian_noise":    "Gauss. Noise",
        "shot_noise":        "Shot Noise",
        "impulse_noise":     "Impulse",
        "defocus_blur":      "Defocus",
        "glass_blur":        "Glass",
        "motion_blur":       "Motion",
        "zoom_blur":         "Zoom",
        "snow":              "Snow",
        "frost":             "Frost",
        "fog":               "Fog",
        "brightness":        "Brightness",
        "contrast":          "Contrast",
        "elastic_transform": "Elastic",
        "pixelate":          "Pixelate",
        "jpeg_compression":  "JPEG",
    }

    col_headers = {
        "A0_Full":               r"\textbf{Full}",
        "A1_NoDetector":         r"w/o Detector",
        "A2_NoFilter":           r"w/o Filter",
        "A3_NoPrototype":        r"w/o Proto.",
        "A4_NoProtoNoFilter":    r"w/o Proto.+Filt.",
        "A5_KLDetector":         r"KL Det.",
        "A6_EntropyDetector":    r"Entr. Det.",
        "A7_WeakMemory":         r"$\beta{=}0.5$",
        "A8_StrongMemory":       r"$\beta{=}0.99$",
        "A9_AggressiveGating":   r"$\tau{=}0.01$",
        "A10_ConservativeGating": r"$\tau{=}0.05$",
    }

    tex_lines = []
    tex_lines.append(r"\begin{table*}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(
        r"\caption{Component ablation of \textsc{ContinualTTA} on CIFAR-10-C "
        r"continual sequential shift (Setting~B), S1--S5 averaged. "
        r"Each column removes or modifies exactly one component from the "
        r"full method (leftmost column). \textbf{Bold} = best per row.}"
    )
    tex_lines.append(r"\label{tab:ablation}")
    tex_lines.append(r"\resizebox{\textwidth}{!}{%")
    tex_lines.append(r"\begin{tabular}{l" + "c" * len(names) + "}")
    tex_lines.append(r"\toprule")

    # Header row
    hdr = "Corruption"
    for name in names:
        hdr += " & " + col_headers.get(name, name)
    tex_lines.append(hdr + r" \\")
    tex_lines.append(r"\midrule")

    # Per-corruption rows
    for corruption in ALL_CORRUPTIONS:
        vals = [all_results[name].get(corruption, float("nan")) for name in names]
        finite_vals = [v for v in vals if not math.isnan(v)]
        best = max(finite_vals) if finite_vals else float("nan")
        row  = corr_display.get(corruption, corruption)
        for val in vals:
            if math.isnan(val):
                row += " & ---"
            elif abs(val - best) < 0.05:
                row += rf" & \textbf{{{val:.1f}}}"
            else:
                row += f" & {val:.1f}"
        tex_lines.append(row + r" \\")

    tex_lines.append(r"\midrule")

    # Mean row
    mean_vals   = [all_means.get(name, float("nan")) for name in names]
    finite_means = [v for v in mean_vals if not math.isnan(v)]
    best_mean   = max(finite_means) if finite_means else float("nan")
    mean_tex    = r"\textbf{Mean}"
    for val in mean_vals:
        if math.isnan(val):
            mean_tex += " & ---"
        elif abs(val - best_mean) < 0.05:
            mean_tex += rf" & \textbf{{{val:.1f}}}"
        else:
            mean_tex += f" & {val:.1f}"
    tex_lines.append(mean_tex + r" \\")
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}}")
    tex_lines.append(r"\end{table*}")

    latex_str = "\n".join(tex_lines)

    tex_path = os.path.join(RESULTS_DIR, "ablation_table.tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)

    print(f"\nLaTeX saved: {tex_path}")
    print("\n" + "=" * 60)
    print("Paste into Overleaf:")
    print("=" * 60)
    print(latex_str)


# =============================================================================
# 9. ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ContinualTTA Component Ablation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full method on CIFAR-10-C Setting B:
  python ablation.py --ablation A0 --dataset cifar10c --setting b

  # Run no-detector ablation on Setting A:
  python ablation.py --ablation A1 --dataset cifar10c --setting a

  # Run on ImageNet-C (severity 5):
  python ablation.py --ablation A0 --dataset imagenetc

  # After all ablations complete, merge and generate LaTeX:
  python ablation.py --merge --dataset cifar10c --setting b

  # List all available ablation variants:
  python ablation.py --list
        """,
    )

    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=[a[0] for a in ABLATIONS],
        help="Which ablation to run (e.g. A0, A1, ..., A10)",
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10c",
        choices=["cifar10c", "imagenetc"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--setting", type=str, default="b",
        choices=["a", "b"],
        help="Evaluation protocol (cifar10c only): a=i.i.d. mixed, b=sequential",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all completed CSVs and generate LaTeX table",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available ablation variants and exit",
    )

    args = parser.parse_args()

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        print(f"\n{'ID':<6} {'Name':<22} Description")
        print("─" * 72)
        for abl_id, name, desc, kw in ABLATIONS:
            kw_str = str(kw) if kw else "(full method — no overrides)"
            print(f"{abl_id:<6} {name:<22} {desc}")
            print(f"{'':6} {'':22} kwargs: {kw_str}")
        print()
        raise SystemExit(0)

    # ── Merge mode ────────────────────────────────────────────────────────────
    if args.merge:
        merge_ablations_and_latex(args.dataset, args.setting)
        raise SystemExit(0)

    # ── Run mode — requires --ablation ────────────────────────────────────────
    if args.ablation is None:
        parser.error("--ablation is required unless using --merge or --list")

    # Environment info
    print(f"\nDevice     : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"Ablation   : {args.ablation}")
    print(f"Dataset    : {args.dataset}")
    if args.dataset == "cifar10c":
        print(f"Setting    : {args.setting.upper()}")

    # Load model
    if args.dataset == "cifar10c":
        source_model = load_cifar_model()
        setup_bn_fn  = setup_bn_cifar
        num_classes  = CIFAR_CLASSES
        feat_dim     = 2048
    else:
        source_model = load_imagenet_model()
        setup_bn_fn  = setup_bn_imagenet
        num_classes  = IMAGENET_CLASSES
        feat_dim     = 2048

    print(f"Parameters : {sum(p.numel() for p in source_model.parameters()):,}")

    # Run
    results, mean_acc = run_ablation(
        ablation_id  = args.ablation,
        dataset      = args.dataset,
        setting      = args.setting,
        source_model = source_model,
        setup_bn_fn  = setup_bn_fn,
        num_classes  = num_classes,
        feat_dim     = feat_dim,
    )

    print(f"\n{'='*60}")
    print(f"DONE — {args.ablation}: {mean_acc:.2f}% mean accuracy")
    print(f"Results dir: {os.path.abspath(RESULTS_DIR)}")
    print(f"{'='*60}")