import torch
from torch.utils.data import Dataset
import rasterio
from pathlib import Path
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SpaceNet7PatchDataset(Dataset):
    """
    Dataset pour SpaceNet 7 avec sélection des sites.
    """
    def __init__(self,
                 root_dir: str,
                 patch_size: int = 256,
                 stride: int = 256,
                 transform: A.Compose = None,
                 min_building_ratio: float = 0.0,
                 sites: list = None):
        """
        Args:
            root_dir (str): Dossier racine contenant les sous-dossiers des zones.
            patch_size (int): Taille du patch carré.
            stride (int): Pas de déplacement entre patches.
            transform (A.Compose): Transformations (albumentations).
            min_building_ratio (float): Ratio minimum de pixels bâtiment pour conserver un patch.
            sites (list): Liste des noms de sites à inclure. Si None, inclut tous.
        """
        self.root_dir = Path(root_dir)
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.min_building_ratio = min_building_ratio
        self.sites = sites

        # Collecter les dossiers des sites
        all_site_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        if self.sites is not None:
            site_dirs = [d for d in all_site_dirs if d.name in self.sites]
        else:
            site_dirs = all_site_dirs

        # Étape 1 : collecter toutes les paires (image, masque)
        self.samples = []  # liste de tuples (img_path, mask_path)
        for site_dir in site_dirs:
            img_dir = site_dir / "images_masked"
            mask_dir = site_dir / "masks"
            if not img_dir.exists() or not mask_dir.exists():
                print(f"Attention: zone {site_dir.name} manque un dossier images_masked/masks")
                continue
            for img_file in img_dir.glob("*.tif"):
                mask_file = mask_dir / (img_file.stem + "_mask.tif")
                if mask_file.exists():
                    self.samples.append((img_file, mask_file))
                else:
                    print(f"Masque manquant pour {img_file}")

        # Étape 2 : générer les indices de patches (sample_idx, x, y)
        self.patch_indices = []
        for sample_idx, (img_path, mask_path) in enumerate(self.samples):
            with rasterio.open(img_path) as src:
                h, w = src.height, src.width

            # Coordonnées x
            xs = list(range(0, w - self.patch_size + 1, self.stride))
            if (w - self.patch_size) % self.stride != 0:
                xs.append(w - self.patch_size)

            # Coordonnées y
            ys = list(range(0, h - self.patch_size + 1, self.stride))
            if (h - self.patch_size) % self.stride != 0:
                ys.append(h - self.patch_size)

            for x in xs:
                for y in ys:
                    # Filtrer sur le ratio de bâtiments si demandé
                    if self.min_building_ratio > 0:
                        with rasterio.open(mask_path) as src_mask:
                            window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)
                            mask_patch = src_mask.read(1, window=window)
                        ratio = np.sum(mask_patch == 1) / (self.patch_size * self.patch_size)
                        if ratio < self.min_building_ratio:
                            continue
                    self.patch_indices.append((sample_idx, x, y))

        print(f"Dataset initialisé : {len(self.samples)} paires image/masque, "
              f"{len(self.patch_indices)} patches générés.")

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        sample_idx, x, y = self.patch_indices[idx]
        img_path, mask_path = self.samples[sample_idx]

        # Charger le patch d'image
        with rasterio.open(img_path) as src:
            window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)
            img = src.read(window=window)                # shape (bands, H, W)
            if img.shape[0] >= 3:
                img = img[:3]                           # garde les 3 premières bandes (RGB)
            img = np.transpose(img, (1, 2, 0))          # (H, W, C)
            img = img.astype(np.float32) / 255.0        # normalisation [0,1]

        # Charger le patch de masque
        with rasterio.open(mask_path) as src:
            window = rasterio.windows.Window(x, y, self.patch_size, self.patch_size)
            mask = src.read(1, window=window)           # (H, W)

        # Appliquer les transformations
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        else:
            # conversion par défaut
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        return img, mask


def get_train_transform():
    """
    Transformations pour l'entraînement.
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_test_transform():
    """
    Transformations pour le test (seulement normalisation).
    """
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# -----------------------------------------------------------------------------
# Test du dataset avec split des sites
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    from torch.utils.data import DataLoader

    root_dir = "data/train"   

    # Lister tous les sites
    all_sites = [d.name for d in Path(root_dir).iterdir() if d.is_dir()]
    print(f"Sites trouvés : {all_sites}")

    # Split aléatoire (80% de sites pour l,entraînement et 20% pour le test)
    random.seed(42)
    random.shuffle(all_sites)
    split_idx = int(0.8 * len(all_sites))
    train_sites = all_sites[:split_idx]
    val_sites = all_sites[split_idx:]

    print(f"Train sites: {train_sites}")
    print(f"Val sites: {val_sites}")

    # Créer datasets train et test
    train_dataset = SpaceNet7PatchDataset(
        root_dir=root_dir,
        patch_size=256,
        stride=256,
        transform=get_train_transform(),
        sites=train_sites,
        min_building_ratio=0.0
    )

    test_dataset = SpaceNet7PatchDataset(
        root_dir=root_dir,
        patch_size=256,
        stride=256,
        transform=get_test_transform(),
        sites=val_sites,
        min_building_ratio=0.0
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Test sur l'entraînement
    print("Test du DataLoader d'entraînement :")
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images {images.shape}, masks {masks.shape}")
        if batch_idx == 5:
            break

    # Test sur le test
    print("Test du DataLoader de validation :")
    for batch_idx, (images, masks) in enumerate(val_loader):
        print(f"Batch {batch_idx}: images {images.shape}, masks {masks.shape}")
        if batch_idx == 5:
            break
    print("Dataset et DataLoader fonctionnent correctement.")