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
