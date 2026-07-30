# =============================================================================
# CIFAR-100 Source Model Training — Windows-Compatible Version
# Architecture : ResNet-50 with BatchNorm, 100-class head
# Target       : ~80-83% val accuracy (matches your earlier 82.7% run)
#
# WINDOWS FIX APPLIED:
#   All execution code is wrapped in `if __name__ == "__main__":` and
#   `freeze_support()` is called first. This is REQUIRED on Windows
#   when using DataLoader(num_workers > 0), because Windows uses
#   'spawn' (not 'fork') to create worker processes, which re-imports
#   the whole script in each worker -- without the __main__ guard,
#   each worker re-triggers the entire training loop, causing the
#   "An attempt has been made to start a new process before the
#   current process has finished its bootstrapping phase" error.
#
# WHY BN, NOT GN (see discussion):
#   CIFAR-100-C will be evaluated under FRESH-PER-CORRUPTION protocol
#   (model resets before each corruption), not truly continual. Under
#   reset-based evaluation, BN's batch-statistics-corruption problem
#   does not compound across corruptions -- each corruption gets one
#   clean evaluation. BN also matches CoTTA/RoTTA's published
#   CIFAR-100-C numbers, keeping your results directly comparable.
#
# IF YOU ALREADY HAVE A CHECKPOINT from an earlier run (82.7% val acc),
# you do NOT need to re-run this -- check first:
#   C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar100_source.pth
# =============================================================================

import os
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import freeze_support

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\cifar-100-python"
CHECKPOINT  = r"C:\Users\Vineet9.Yadav\Desktop\ContinualTTA\resnet50_cifar100_source.pth"

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES  = 100
BATCH_SIZE   = 128
NUM_EPOCHS   = 30
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
SEED         = 42

# WINDOWS FIX: num_workers=0 avoids spawning subprocess workers entirely,
# sidestepping the Windows 'spawn' multiprocessing issue with zero
# restructuring needed. Data loading runs in the main process -- slightly
# slower than num_workers=4, but the cost is small relative to GPU
# compute time for a 50,000-image dataset, and this is the most robust
# fix for an interactive/script-based Windows workflow (no edge cases
# with how the script is invoked).
NUM_WORKERS  = 0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =============================================================================
# 1. DATASET
# =============================================================================

def load_cifar100_binary(path):
    """Load official CIFAR-100 binary file (pickled dict format)."""
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    images = data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = data[b'fine_labels']   # 100 fine-grained class indices
    return images, labels


class CIFAR100Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=28),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# =============================================================================
# 2. TRAIN / VALIDATE FUNCTIONS
# =============================================================================

def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct      += (out.argmax(1) == y).sum().item()
        total        += x.size(0)

        if (i + 1) % 50 == 0:
            print(f"  [{epoch+1}/{NUM_EPOCHS}] "
                  f"step {i+1}/{len(train_loader)}  "
                  f"loss={running_loss/total:.4f}  "
                  f"acc={100*correct/total:.1f}%", end="\r")

    return running_loss / total, 100.0 * correct / total


def validate(model, val_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y     = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total


# =============================================================================
# 3. MAIN — everything that creates DataLoaders/processes must be here
# =============================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device     : {DEVICE}")
    print(f"Classes    : {NUM_CLASSES}")
    print(f"Epochs     : {NUM_EPOCHS}")
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"Seed       : {SEED}")
    print(f"NumWorkers : {NUM_WORKERS}  (0 = Windows-safe, no spawn issue)")

    print(f"\nLoading CIFAR-100 from {DATA_DIR}...")
    print(f"Files: {os.listdir(DATA_DIR)}")

    train_images, train_labels = load_cifar100_binary(f"{DATA_DIR}/train")
    val_images,   val_labels   = load_cifar100_binary(f"{DATA_DIR}/test")

    print(f"Train : {len(train_images):,} images")
    print(f"Val   : {len(val_images):,} images")

    train_dataset = CIFAR100Dataset(train_images, train_labels, train_transform)
    val_dataset   = CIFAR100Dataset(val_images,   val_labels,   val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    print("\nBuilding ResNet-50 (BatchNorm)...")
    model    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model    = model.to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Normalization layers: "
          f"{sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))} "
          f"BatchNorm2d")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                 momentum=MOMENTUM,
                                 weight_decay=WEIGHT_DECAY,
                                 nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=NUM_EPOCHS)

    print(f"\nTraining for {NUM_EPOCHS} epochs...\n")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<12} "
          f"{'Val Acc':<10} {'LR':<12}")
    print("-" * 58)

    best_acc, best_state = 0.0, None

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch)
        val_acc = validate(model, val_loader)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        marker     = "  <- best" if val_acc > best_acc else ""

        print(f"{epoch+1:<8} {train_loss:<14.4f} {train_acc:<12.1f} "
              f"{val_acc:<10.1f} {current_lr:<12.6f}{marker}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), CHECKPOINT)

    print(f"\n{'='*58}")
    print(f"Training complete.")
    print(f"Best val accuracy : {best_acc:.1f}%")
    print(f"Checkpoint saved  : {CHECKPOINT}")
    print(f"{'='*58}")

    if best_acc >= 80.0:
        print("\nModel is well-trained -- ready for CIFAR-100-C experiments.")
    elif best_acc >= 65.0:
        print("\nAcceptable -- TTA will work but consider more epochs.")
    else:
        print("\nToo low -- try NUM_EPOCHS=50 or reduce LR to 0.05.")

    print("\nNext step: use this checkpoint path in cifar100c_fresh.py:")
    print(f"  MODEL_PATH = r'{CHECKPOINT}'")


# =============================================================================
# ENTRY POINT — REQUIRED guard for Windows multiprocessing
# =============================================================================

if __name__ == "__main__":
    freeze_support()   # harmless no-op on Linux/Mac, required on Windows
    main()