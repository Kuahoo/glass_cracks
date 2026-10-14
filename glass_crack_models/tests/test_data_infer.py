import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from glass_crack_models.data.rasterize import mask_to_polygons, polygon_to_mask, read_yolo
from glass_crack_models.data.relabel_classes import relabel_lines
from glass_crack_models.infer.tiling import auto_scales, predict_multiscale, predict_tiled, tile_starts
from glass_crack_models.train import semseg_common as S


class FakePredictor:
    """Score = red channel / 255, so tiled output must reproduce the input exactly."""
    eval_classes = ["linear_defect"]

    def __init__(self):
        self.calls = 0

    def predict_batch(self, tiles):
        self.calls += len(tiles)
        return [{"linear_defect": t[..., 0].astype(np.float32) / 255.0} for t in tiles]


class RasterizeTests(unittest.TestCase):
    def test_read_and_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("1 0.1 0.1 0.6 0.1 0.6 0.3 0.1 0.3\n0 0.5 0.5\n")  # second line too short: ignored
            labels = read_yolo(p)
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0][0], 1)
            m = polygon_to_mask(labels[0][1], 200, 300)
            polys = mask_to_polygons(m)
            self.assertEqual(len(polys), 1)
            back = polygon_to_mask(polys[0] / [300, 200], 200, 300)
            iou = np.logical_and(m, back).sum() / np.logical_or(m, back).sum()
            self.assertGreater(iou, 0.95)
        self.assertEqual(read_yolo("/nonexistent.txt"), [])

    def test_relabel(self):
        lines = ["0 .1 .1 .2 .2 .3 .1", "1 .1 .1 .2 .2 .3 .1", "3 .5 .5 .6 .6 .7 .5", "2 .1 .1 .2 .2 .3 .1"]
        self.assertEqual(relabel_lines(lines, {1, 3}), ["0 .1 .1 .2 .2 .3 .1", "0 .5 .5 .6 .6 .7 .5"])


class TilingTests(unittest.TestCase):
    def test_tile_starts_cover(self):
        for length in (640, 641, 1000, 4500):
            s = tile_starts(length, 640, 0.25)
            self.assertEqual(s[0], 0)
            self.assertEqual(s[-1] + 640, max(length, 640) if length > 640 else 640)
            self.assertTrue(all(b - a <= 480 for a, b in zip(s, s[1:])))

    def test_tiled_reconstructs(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (1100, 1500, 3), dtype=np.uint8)
        fp = FakePredictor()
        out = predict_tiled(fp, img, 640, 0.25, batch=3)["linear_defect"]
        np.testing.assert_allclose(out, img[..., 0] / 255.0, atol=1e-5)
        self.assertGreater(fp.calls, 1)

    def test_small_image_single_pass(self):
        img = np.full((480, 640, 3), 128, np.uint8)
        fp = FakePredictor()
        out = predict_tiled(fp, img)["linear_defect"]
        self.assertEqual(out.shape, (480, 640))
        self.assertEqual(fp.calls, 1)

    def test_multiscale_shape_and_auto(self):
        self.assertEqual(auto_scales(3000, 4500), [1.0, 0.5, 0.25])
        self.assertEqual(auto_scales(640, 640), [1.0])
        img = np.zeros((900, 1300, 3), np.uint8)
        img[400:405, 100:1200, 0] = 255
        out = predict_multiscale(FakePredictor(), img, [1.0, 0.5], thr=0.5)["linear_defect"]
        self.assertEqual(out.shape, (900, 1300))
        self.assertGreater(out[402, 600], 0.9)


class SemsegTests(unittest.TestCase):
    def test_losses_prefer_correct(self):
        t = torch.zeros(1, 1, 32, 32)
        t[..., 14:17, 4:28] = 1
        good, bad = (t * 20 - 10), torch.zeros_like(t) - 10
        for name in ("bce_dice", "bce_tversky", "bce_dice_cldice"):
            fn = S.make_loss(name)
            self.assertLess(fn(good, t).item(), fn(bad, t).item(), name)

    def test_weights(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with (d / "manifest.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["split", "image", "copy_kind"])
                w.writeheader()
                w.writerows([{"split": "train", "image": "a.jpg", "copy_kind": "original"},
                             {"split": "train", "image": "b.jpg", "copy_kind": "original_or_augmented"},
                             {"split": "train", "image": "c.jpg", "copy_kind": "original_or_augmented"},
                             {"split": "valid", "image": "v.jpg", "copy_kind": "original"}])
            _, w_bal = S.train_images_and_weights(d, "balanced")
            self.assertEqual(w_bal, [1.0, 0.5, 0.5])
            imgs, _ = S.train_images_and_weights(d, "no_augpairs")
            self.assertEqual([p.name for p in imgs], ["a.jpg"])
            _, w_all = S.train_images_and_weights(d, "all")
            self.assertEqual(w_all, [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
