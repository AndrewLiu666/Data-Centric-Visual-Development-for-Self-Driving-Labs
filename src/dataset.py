from pathlib import Path
import csv
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class LetterboxRect:
    def __init__(self, size_hw=(600, 1500)):
        if isinstance(size_hw, int):
            self.target_h, self.target_w = size_hw, size_hw
        else:
            self.target_h, self.target_w = int(size_hw[0]), int(size_hw[1])

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        H, W = self.target_h, self.target_w
        scale = min(H / h, W / w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BILINEAR)
        pad_w, pad_h = W - nw, H - nh
        pad = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img = ImageOps.expand(img, border=pad, fill=0)
        return img

def build_transforms(split: str, img_size_hw=(600, 1500)):
    base = [LetterboxRect(img_size_hw)]
    if split == "train":
        aug = [
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.02)], p=0.8),
            T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.1),
        ]
    else:
        aug = []
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(base + aug + norm)

class ROIDataset(Dataset):
    def __init__(self, csv_file: Path, split: str, img_size_hw=(600, 1500)):
        self.items = []
        with open(csv_file, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                self.items.append((row["path"], int(row["label"])))
        self.tx = build_transforms(split, img_size_hw)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        pth, lbl = self.items[idx]
        img = Image.open(pth).convert("RGB")
        img = self.tx(img)
        target = torch.tensor([float(lbl)], dtype=torch.float32)
        return img, target

def build_loader(csv_path, split, batch_size=4, img_size_hw=(600, 1500), num_workers=4, shuffle=None):
    if shuffle is None:
        shuffle = (split == "train")
    ds = ROIDataset(Path(csv_path), split, img_size_hw)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
