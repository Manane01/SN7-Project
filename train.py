import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
import numpy as np
from pathlib import Path
import random

# Imports de tes modules
from dataset import SpaceNet7PatchDataset, get_train_transform, get_test_transform
from model import get_model

# Paramètres
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

root_dir = "data/train"
model_save_path = "best_model_sn7.pth"

# 1. Split des sites
all_sites = [d.name for d in Path(root_dir).iterdir() if d.is_dir()]
random.seed(42)
random.shuffle(all_sites)
split_idx = int(0.8 * len(all_sites))
train_sites = all_sites[:split_idx]
test_sites = all_sites[split_idx:]

print(f"Train sites: {train_sites}")
print(f"Val sites: {test_sites}")

# 2. Datasets
train_dataset = SpaceNet7PatchDataset(
    root_dir=root_dir,
    patch_size=256,
    stride=128,
    transform=get_train_transform(),
    sites=train_sites,
    min_building_ratio=0.05
)

test_dataset = SpaceNet7PatchDataset(
    root_dir=root_dir,
    patch_size=256,
    stride=256,
    transform=get_test_transform(),
    sites=test_sites,
    min_building_ratio=0.0
)

# 3. DataLoaders
batch_size = 8
num_workers = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 4. Modèle
model = get_model().to(device)
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# 5. Fonctions de perte et métrique
bce_loss = nn.BCELoss()
dice_loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)

def compute_loss(pred, target):
    # pred et target ont tous deux la forme (batch, H, W)
    return bce_loss(pred, target) + dice_loss(pred, target)

def iou_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + 1e-7) / (union + 1e-7)

# 6. Boucle d'entraînement
num_epochs = 10
best_iou = 0.0

for epoch in range(num_epochs):
    # Entraînement
    model.train()
    train_loss = 0.0
    num_batches_train = len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Entraînement en cours...")
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.float().to(device)

        optimizer.zero_grad()
        pred = model(images)          # forme: (batch, 1, H, W)
        pred = pred.squeeze(1)        # supprime la dimension canal -> (batch, H, W)
        loss = compute_loss(pred, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        # Affichage optionnel tous les 100 batchs
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{num_batches_train} - Loss batch: {loss.item():.2f}")
    train_loss /= len(train_dataset)

    # Validation
    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    num_batches_val = len(val_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Validation en cours...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks = masks.float().to(device)
            pred = model(images)
            pred = pred.squeeze(1)       # (batch, H, W)
            loss = compute_loss(pred, masks)
            test_loss += loss.item() * images.size(0)

            batch_iou = iou_score(pred, masks).item()
            test_iou += batch_iou * images.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{num_batches_val} - IoU batch: {batch_iou:.2f}")
    test_loss /= len(test_dataset)
    test_iou /= len(test_dataset)

    print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.2f} | Test Loss: {test_loss:.2f} | Test IoU: {test_iou:.2f}")

    scheduler.step(test_loss)

    if test_iou > best_iou:
        best_iou = test_iou
        torch.save(model.state_dict(), model_save_path)
        print(f"  -> Meilleur modèle sauvegardé (IoU = {best_iou:.2f})")

print("Entraînement terminé.")