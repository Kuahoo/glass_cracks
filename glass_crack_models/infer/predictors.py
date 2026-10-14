"""Model wrappers that turn RGB tiles into per-class score maps in [0, 1].

Track A (YOLO-seg): score = max over detections of (confidence x mask), per evaluated class.
Track B (smp): score = sigmoid probability.
Evaluated classes: `linear_defect` always (for 4-class models: crack OR scratch), plus `crack` when the
model has a separate crack class.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def eval_class_sources(class_names: list[str]) -> dict[str, list[int]]:
    idx = {n: i for i, n in enumerate(class_names)}
    if "linear_defect" in idx:
        return {"linear_defect": [idx["linear_defect"]]}
    if "crack" in idx:
        out = {"linear_defect": [idx[n] for n in ("crack", "scratch") if n in idx], "crack": [idx["crack"]]}
        return out
    raise ValueError(f"model classes {class_names} have neither linear_defect nor crack")


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def torch_device(device: str | None) -> str:
    """Accept Ultralytics-style ids ('0', '0,1') as well as torch strings."""
    if not device:
        return default_device()
    return f"cuda:{device.split(',')[0]}" if device[0].isdigit() else device


class YoloPredictor:
    track = "A"

    def __init__(self, checkpoint: str | Path, device: str | None = None, conf: float = 0.01, max_det: int = 300):
        from ultralytics import YOLO
        self.checkpoint = str(checkpoint)
        self.model = YOLO(self.checkpoint)
        names = self.model.names
        self.class_names = [names[i] for i in sorted(names)]
        self.sources = eval_class_sources(self.class_names)
        self.eval_classes = list(self.sources)
        self.device = device or default_device()
        self.conf, self.max_det = conf, max_det

    def predict_batch(self, tiles: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        bgr = [cv2.cvtColor(t, cv2.COLOR_RGB2BGR) for t in tiles]
        imgsz = max(max(t.shape[:2]) for t in tiles)
        imgsz = int(np.ceil(imgsz / 32) * 32)
        results = self.model.predict(bgr, imgsz=imgsz, conf=self.conf, max_det=self.max_det, retina_masks=True,
                                     device=self.device, verbose=False)
        out = []
        for t, r in zip(tiles, results):
            h, w = t.shape[:2]
            maps = {c: np.zeros((h, w), np.float32) for c in self.eval_classes}
            if r.masks is not None and len(r.boxes):
                masks = r.masks.data.float().cpu().numpy()
                if masks.shape[1:] != (h, w):
                    masks = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) for m in masks])
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                for name, ids in self.sources.items():
                    sel = np.isin(cls, ids)
                    if sel.any():
                        maps[name] = np.max(masks[sel] * conf[sel, None, None], axis=0).astype(np.float32)
            out.append(maps)
        return out


class SemsegPredictor:
    track = "B"

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        from glass_crack_models.train.semseg_common import build_model
        self.checkpoint = str(checkpoint)
        ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.class_names = list(ck["classes"])
        self.model = build_model(cfg["arch"], cfg["encoder"], None, len(self.class_names))
        self.model.load_state_dict(ck["model"])
        self.device = torch_device(device)
        self.model.to(self.device).eval()
        self.sources = eval_class_sources(self.class_names)
        self.eval_classes = list(self.sources)

    @torch.no_grad()
    def predict_batch(self, tiles: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        from glass_crack_models.train.semseg_common import to_tensor
        h, w = tiles[0].shape[:2]
        ph, pw = int(np.ceil(h / 32) * 32), int(np.ceil(w / 32) * 32)
        x = torch.stack([to_tensor(t) for t in tiles]).to(self.device)
        if (ph, pw) != (h, w):
            x = torch.nn.functional.pad(x, (0, pw - w, 0, ph - h), mode="reflect")
        with torch.autocast(device_type="cuda", enabled=self.device.startswith("cuda")):
            prob = torch.sigmoid(self.model(x).float())[:, :, :h, :w].cpu().numpy()
        out = []
        for p in prob:
            out.append({name: np.max(p[ids], axis=0) for name, ids in self.sources.items()})
        return out


def load_predictor(checkpoint: str | Path, device: str | None = None):
    p = Path(checkpoint)
    if p.suffix == ".pth":
        return SemsegPredictor(p, device)
    return YoloPredictor(p, device)
