import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from glass_crack_models.infer.predictors import (
    RfDetrPredictor,
    load_predictor,
    prefer_rfdetr_checkpoint,
    score_maps_from_instances,
)


class ScoreMapTests(unittest.TestCase):
    def test_max_conf_times_mask_for_linear_defect(self):
        h, w = 8, 8
        m0 = np.zeros((h, w), bool); m0[2:5, 2:5] = True
        m1 = np.zeros((h, w), bool); m1[4:7, 4:7] = True
        maps = score_maps_from_instances(
            height=h, width=w,
            masks=np.stack([m0, m1]),
            confs=np.array([0.9, 0.4], np.float32),
            class_ids=np.array([1, 3], np.int64),  # crack, scratch
            class_names=["bubble", "crack", "inclusion", "scratch"],
        )
        self.assertIn("linear_defect", maps)
        self.assertIn("crack", maps)
        self.assertAlmostEqual(float(maps["linear_defect"][3, 3]), 0.9, places=5)
        self.assertAlmostEqual(float(maps["linear_defect"][5, 5]), 0.4, places=5)
        self.assertAlmostEqual(float(maps["crack"][3, 3]), 0.9, places=5)
        self.assertEqual(float(maps["crack"][5, 5]), 0.0)


class RfDetrClassIdMappingTests(unittest.TestCase):
    def _make(self, names):
        import sys
        import types
        model = MagicMock()
        model.class_names = names
        fake = types.ModuleType("rfdetr")
        fake.RFDETR = MagicMock()
        fake.RFDETR.from_checkpoint.return_value = model
        with patch.dict(sys.modules, {"rfdetr": fake}):
            return RfDetrPredictor("checkpoint_best_total.pth", device="cpu"), model

    def test_sparse_coco_ids_remapped_to_contiguous(self):
        pred, model = self._make({1: "bubble", 2: "crack", 3: "inclusion", 4: "scratch"})
        self.assertEqual(pred.class_names, ["bubble", "crack", "inclusion", "scratch"])
        h = w = 8
        mask = np.zeros((1, h, w), bool); mask[0, 2:5, 2:5] = True
        det = MagicMock()
        det.mask = mask
        det.confidence = np.array([0.8], np.float32)
        det.class_id = np.array([2])  # COCO-style id for "crack"
        model.predict.return_value = det
        out = pred.predict_batch([np.zeros((h, w, 3), np.uint8)])[0]
        self.assertAlmostEqual(float(out["crack"][3, 3]), 0.8, places=5)
        self.assertAlmostEqual(float(out["linear_defect"][3, 3]), 0.8, places=5)

    def test_sparse_ids_inclusion_does_not_leak_into_linear_defect(self):
        pred, model = self._make({1: "bubble", 2: "crack", 3: "inclusion", 4: "scratch"})
        h = w = 8
        mask = np.ones((1, h, w), bool)
        det = MagicMock()
        det.mask = mask
        det.confidence = np.array([0.9], np.float32)
        det.class_id = np.array([3])  # inclusion
        model.predict.return_value = det
        out = pred.predict_batch([np.zeros((h, w, 3), np.uint8)])[0]
        self.assertEqual(float(out["linear_defect"].max()), 0.0)
        self.assertEqual(float(out["crack"].max()), 0.0)

    def test_list_names_identity_mapping(self):
        pred, model = self._make(["bubble", "crack", "inclusion", "scratch"])
        h = w = 8
        det = MagicMock()
        det.mask = np.ones((1, h, w), bool)
        det.confidence = np.array([0.7], np.float32)
        det.class_id = np.array([1])
        model.predict.return_value = det
        out = pred.predict_batch([np.zeros((h, w, 3), np.uint8)])[0]
        self.assertAlmostEqual(float(out["crack"][0, 0]), 0.7, places=5)


class RoutingTests(unittest.TestCase):
    def test_prefer_checkpoint_order(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "checkpoint_best_regular.pth").write_bytes(b"a")
            (d / "checkpoint_best_ema.pth").write_bytes(b"b")
            (d / "checkpoint_best_total.pth").write_bytes(b"c")
            self.assertEqual(prefer_rfdetr_checkpoint(d).name, "checkpoint_best_total.pth")

    def test_load_predictor_uses_track_c_experiment_json(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            ck = d / "checkpoint_best_total.pth"
            ck.write_bytes(b"x")
            (d / "experiment.json").write_text(json.dumps({"track": "C", "variant": "large"}))
            with patch("glass_crack_models.infer.predictors.RfDetrPredictor") as cls:
                cls.return_value = MagicMock(track="C")
                pred = load_predictor(ck, device="cpu")
                cls.assert_called_once()
                self.assertEqual(pred.track, "C")

    def test_explicit_non_c_track_not_overridden_by_filename(self):
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "checkpoint_best_rfdetr.pth"
            ck.write_bytes(b"x")
            (Path(td) / "experiment.json").write_text(json.dumps({"track": "B"}))
            with patch("glass_crack_models.infer.predictors.SemsegPredictor") as sem, \
                 patch("glass_crack_models.infer.predictors.RfDetrPredictor") as rf:
                load_predictor(ck, device="cpu")
                sem.assert_called_once()
                rf.assert_not_called()

    def test_filename_heuristic_when_track_none(self):
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "checkpoint_best_total.pth"
            ck.write_bytes(b"x")
            with patch("glass_crack_models.infer.predictors.RfDetrPredictor") as rf:
                load_predictor(ck, device="cpu")
                rf.assert_called_once()

    def test_track_b_pth_still_routes_to_semseg(self):
        with tempfile.TemporaryDirectory() as td:
            ck = Path(td) / "best.pth"
            ck.write_bytes(b"x")
            with patch("glass_crack_models.infer.predictors.SemsegPredictor") as sem, \
                 patch("glass_crack_models.infer.predictors.RfDetrPredictor") as rf:
                load_predictor(ck, device="cpu")
                sem.assert_called_once()
                rf.assert_not_called()


if __name__ == "__main__":
    unittest.main()
