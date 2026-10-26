"""Track B building blocks: model factory, dataset, sampler weights, losses."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from glass_crack_models.data.rasterize import read_yolo, semantic_mask

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(arch: str, encoder: str, encoder_weights: str | None, n_classes: int = 1):
    import segmentation_models_pytorch as smp
    if arch.lower() != "unet":
        raise ValueError(f"unknown arch {arch}")
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes)


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def build_augment(level: str, imgsz: int):
    """Flips, rot90, and mild colour. Roboflow already applied blur and noise to the pair copies."""
    import albumentations as A
    if level != "base":
        raise ValueError(level)
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.3), A.Resize(imgsz, imgsz),
    ])


class SegDataset(torch.utils.data.Dataset):
    """Images + YOLO polygons -> (image tensor, float mask (1, H, W)). Classes in `classes` become foreground."""

    def __init__(self, images: list[Path], classes: set[int], imgsz: int = 640, augment=None):
        self.images, self.classes, self.imgsz, self.augment = images, classes, imgsz, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        p = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        mask = semantic_mask(read_yolo(p.parent.parent / "labels" / (p.stem + ".txt")), h, w, self.classes).astype(np.uint8)
        if self.augment is not None:
            out = self.augment(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        elif (h, w) != (self.imgsz, self.imgsz):
            img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        return to_tensor(img), torch.from_numpy(mask[None].astype(np.float32))


def train_images_and_weights(data_dir: Path, weighting: str) -> tuple[list[Path], list[float]]:
    """Per-file sampling weights: `balanced` gives each Roboflow pair copy half weight,
    `no_augpairs` drops the pair copies, `all` is uniform."""
    with (data_dir / "manifest.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    rows.sort(key=lambda r: r["image"])
    images, weights = [], []
    for r in rows:
        pair = r["copy_kind"] == "original_or_augmented"
        if weighting == "no_augpairs" and pair:
            continue
        images.append(data_dir / "train" / "images" / r["image"])
        weights.append(0.5 if (weighting == "balanced" and pair) else 1.0)
    if weighting not in ("all", "balanced", "no_augpairs"):
        raise ValueError(weighting)
    return images, weights


# ---- losses -----------------------------------------------------------------------------------

def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def make_loss(name: str, pos_weight: float | None = None):
    pw = torch.tensor([pos_weight]) if pos_weight else None

    def bce(logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw.to(logits.device) if pw is not None else None)

    if name == "bce_dice":
        return lambda l, t: bce(l, t) + dice_loss(l, t)
    raise ValueError(name)
