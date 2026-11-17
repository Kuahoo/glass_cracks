"""Model wrappers that turn RGB tiles into per-class score maps in [0, 1].

Track A (YOLO-seg): score = max over detections of (confidence x mask), per evaluated class.
Track B (smp): score = sigmoid probability.
Evaluated classes: `linear_defect` always (for 4-class models: crack OR scratch), plus `crack` when the
model has a separate crack class.
"""
from __future__ import annotations

import json
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

    def predict_batch(self, tiles: list[np.ndarray], imgsz: int | None = None) -> list[dict[str, np.ndarray]]:
        bgr = [cv2.cvtColor(t, cv2.COLOR_RGB2BGR) for t in tiles]
        if imgsz is None:
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


def score_maps_from_instances(
    height: int,
    width: int,
    masks: np.ndarray,
    confs: np.ndarray,
    class_ids: np.ndarray,
    class_names: list[str],
) -> dict[str, np.ndarray]:
    """Per evaluated class: max over instances of (confidence x mask)."""
    sources = eval_class_sources(class_names)
    maps = {c: np.zeros((height, width), np.float32) for c in sources}
    if masks.size == 0:
        return maps
    if masks.shape[-2:] != (height, width):
        masks = np.stack([cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR) > 0.5
                          for m in masks])
    for name, ids in sources.items():
        sel = np.isin(class_ids, ids)
        if sel.any():
            maps[name] = np.max(masks[sel].astype(np.float32) * confs[sel, None, None], axis=0).astype(np.float32)
    return maps


def prefer_rfdetr_checkpoint(run_dir: Path) -> Path:
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        p = Path(run_dir) / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no RF-DETR best checkpoint under {run_dir}")


def _experiment_track(checkpoint: Path) -> str | None:
    side = checkpoint.parent / "experiment.json"
    if side.exists():
        try:
            return json.loads(side.read_text()).get("track")
        except Exception:
            return None
    return None


class RfDetrPredictor:
    track = "C"

    def __init__(self, checkpoint: str | Path, device: str | None = None, conf: float = 0.01):
        from rfdetr import RFDETR
        self.checkpoint = str(checkpoint)
        self.device = torch_device(device)
        self.conf = conf
        # trust_checkpoint: local training artifacts we produced
        self.model = RFDETR.from_checkpoint(self.checkpoint, trust_checkpoint=True)
        names = getattr(self.model, "class_names", None) or getattr(self.model, "names", None)
        if names is None:
            raise RuntimeError(f"RF-DETR checkpoint missing class names: {self.checkpoint}")
        if isinstance(names, dict):
            self.class_names = [names[i] for i in sorted(names)]
        else:
            self.class_names = list(names)
        self.sources = eval_class_sources(self.class_names)
        self.eval_classes = list(self.sources)

    def predict_batch(self, tiles: list[np.ndarray]) -> list[dict[str, np.ndarray]]:
        out = []
        for tile in tiles:
            h, w = tile.shape[:2]
            det = self.model.predict(tile, threshold=self.conf)
            masks = getattr(det, "mask", None)
            if masks is None or len(masks) == 0:
                out.append({c: np.zeros((h, w), np.float32) for c in self.eval_classes})
                continue
            confs = np.asarray(det.confidence, dtype=np.float32)
            class_ids = np.asarray(det.class_id, dtype=np.int64)
            out.append(score_maps_from_instances(h, w, np.asarray(masks), confs, class_ids, self.class_names))
        return out


def load_predictor(checkpoint: str | Path, device: str | None = None):
    p = Path(checkpoint)
    # Track C first: RF-DETR checkpoints are also .pth, so check before the generic Track B route.
    if _experiment_track(p) == "C" or p.name.lower().startswith("checkpoint_best_") or "rfdetr" in p.name.lower():
        return RfDetrPredictor(p, device)
    if p.suffix == ".pth":
        return SemsegPredictor(p, device)
    return YoloPredictor(p, device)
