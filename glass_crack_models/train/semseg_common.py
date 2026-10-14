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
    arch = arch.lower()
    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes)
    if arch == "segformer":
        return smp.Segformer(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes)
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights=encoder_weights, classes=n_classes)
    raise ValueError(f"unknown arch {arch}")


def to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    x = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def build_augment(level: str, imgsz: int):
    """`base`: flips/rot90 + mild colour (Roboflow already applied blur/noise to the pair copies).
    `robust` (Phase 4): adds what Roboflow did not apply: downscale jitter, JPEG, motion blur, stronger colour."""
    import albumentations as A
    t = [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
         A.RandomBrightnessContrast(0.1, 0.1, p=0.3)]
    if level == "robust":
        t += [A.Affine(scale=(0.35, 1.0), p=0.5),
              A.ImageCompression(quality_range=(40, 95), p=0.4),
              A.MotionBlur(blur_limit=(3, 7), p=0.2),
              A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5)]
    elif level != "base":
        raise ValueError(level)
    t.append(A.Resize(imgsz, imgsz))
    return A.Compose(t)


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


def tversky_loss(logits, target, alpha=0.3, beta=0.7, eps=1.0):
    """beta > alpha penalizes false negatives more (thin cracks)."""
    p = torch.sigmoid(logits)
    tp = (p * target).sum((1, 2, 3))
    fp = (p * (1 - target)).sum((1, 2, 3))
    fn = ((1 - p) * target).sum((1, 2, 3))
    return (1 - (tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def _soft_erode(x):
    return -F.max_pool2d(-x, 3, 1, 1)


def _soft_dilate(x):
    return F.max_pool2d(x, 3, 1, 1)


def soft_skeleton(x, iters: int = 10):
    skel = F.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(iters):
        x = _soft_erode(x)
        delta = F.relu(x - _soft_dilate(_soft_erode(x)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(logits, target, iters: int = 10, eps=1.0):
    """Soft clDice (Shit et al., CVPR 2021)."""
    p = torch.sigmoid(logits)
    sp, st = soft_skeleton(p, iters), soft_skeleton(target, iters)
    tprec = ((sp * target).sum((1, 2, 3)) + eps) / (sp.sum((1, 2, 3)) + eps)
    tsens = ((st * p).sum((1, 2, 3)) + eps) / (st.sum((1, 2, 3)) + eps)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


def make_loss(name: str, pos_weight: float | None = None):
    pw = torch.tensor([pos_weight]) if pos_weight else None

    def bce(logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw.to(logits.device) if pw is not None else None)

    if name == "bce_dice":
        return lambda l, t: bce(l, t) + dice_loss(l, t)
    if name == "bce_tversky":
        return lambda l, t: bce(l, t) + tversky_loss(l, t)
    if name == "bce_dice_cldice":
        return lambda l, t: bce(l, t) + dice_loss(l, t) + 0.5 * cldice_loss(l, t)
    raise ValueError(name)
