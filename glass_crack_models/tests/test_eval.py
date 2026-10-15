"""Toy-case tests for the evaluator. Run: python -m unittest discover -s glass_crack_models/tests -t ."""
import unittest

import cv2
import numpy as np

from glass_crack_models.eval import agdd as A
from glass_crack_models.eval import metrics as M
from glass_crack_models.eval.bootstrap import cluster_bootstrap
from glass_crack_models.eval.match import match, pair_stats
from glass_crack_models.eval.regions import extract_regions, region_from_mask, suppress_small


def line_mask(h, w, p0, p1, thickness=3):
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, p0, p1, 1, thickness)
    return m.astype(bool)


def box_mask(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


class RegionTests(unittest.TestCase):
    def test_min_area_filter(self):
        s = np.zeros((64, 64), np.float32)
        s[2:6, 2:8] = 0.9    # 24 px: kept
        s[20:23, 20:27] = 0.9  # 21 px: dropped
        regs = extract_regions(s, 0.5)
        self.assertEqual([r.area for r in regs], [24])
        sup = suppress_small(s, 0.5)
        self.assertEqual(int((sup >= 0.5).sum()), 24)

    def test_eight_connectivity_and_score(self):
        s = np.zeros((32, 32), np.float32)
        for i in range(30):
            s[i, i] = 0.6
        s[10, 10] = 0.95
        regs = extract_regions(s, 0.5, min_area=1)
        self.assertEqual(len(regs), 1)
        self.assertAlmostEqual(regs[0].score, 0.95, places=5)


class MatchTests(unittest.TestCase):
    def test_identical(self):
        g = region_from_mask(line_mask(100, 100, (10, 10), (90, 80)))
        m = match([g], [g])
        self.assertEqual((m.tp, m.fp, m.fn), (1, 0, 0))

    def test_thin_offset_matches_leniently_not_strictly(self):
        gt = region_from_mask(line_mask(100, 100, (10, 50), (90, 50), 3))
        pr = region_from_mask(line_mask(100, 100, (10, 56), (90, 56), 3))  # disjoint, gap < 2 x 3 px dilation
        iou, near = pair_stats(pr, gt)
        self.assertEqual(iou, 0.0)
        self.assertGreater(near, 0)
        self.assertEqual(match([pr], [gt]).tp, 1)
        self.assertEqual(match([pr], [gt], strict=True).tp, 0)

    def test_far_apart(self):
        gt = region_from_mask(box_mask(100, 100, 0, 0, 10, 10))
        pr = region_from_mask(box_mask(100, 100, 50, 50, 60, 60))
        m = match([pr], [gt])
        self.assertEqual((m.tp, m.fp, m.fn), (0, 1, 1))

    def test_one_to_one_greedy_by_score(self):
        gt = region_from_mask(box_mask(100, 100, 10, 10, 30, 30))
        hi = region_from_mask(box_mask(100, 100, 12, 12, 30, 30), score=0.9)
        lo = region_from_mask(box_mask(100, 100, 10, 10, 28, 28), score=0.4)
        m = match([lo, hi], [gt])
        self.assertEqual((m.tp, m.fp, m.fn), (1, 1, 0))
        self.assertEqual(m.pairs[0][0], 1)  # the high-score prediction took the GT

    def test_merged_prediction_credits_touching_gts(self):
        a, b = line_mask(100, 100, (10, 50), (50, 50)), line_mask(100, 100, (50, 50), (90, 20))
        gts = [region_from_mask(a), region_from_mask(b)]
        pr = region_from_mask(a | b)
        for strict in (False, True):
            m = match([pr], gts, strict=strict)
            self.assertEqual((m.tp, m.tp_pred, m.fp, m.fn), (2, 1, 0, 0))

    def test_nested_gt_is_credited(self):
        outer = region_from_mask(box_mask(100, 100, 10, 10, 60, 20))
        inner = region_from_mask(box_mask(100, 100, 20, 12, 30, 18))
        m = match([outer], [outer, inner])
        self.assertEqual((m.tp, m.tp_pred, m.fp, m.fn), (2, 1, 0, 0))

    def test_neighbour_gt_not_stolen(self):
        a = region_from_mask(box_mask(100, 100, 10, 10, 40, 14))
        b = region_from_mask(box_mask(100, 100, 10, 16, 40, 20))  # 2 px gap: touches after dilation
        m = match([a, b], [a, b])
        self.assertEqual((m.tp, m.tp_pred, m.fp, m.fn), (2, 2, 0, 0))

    def test_blob_touching_two_gts_credits_one(self):
        gts = [region_from_mask(box_mask(100, 100, 10, 10, 14, 14)), region_from_mask(box_mask(100, 100, 80, 80, 84, 84))]
        blob = region_from_mask(box_mask(100, 100, 12, 12, 82, 82))  # IoU with the GT union is tiny
        m = match([blob], gts)
        self.assertEqual((m.tp, m.tp_pred, m.fp, m.fn), (1, 1, 0, 1))

    def test_strict_iou_half(self):
        gt = region_from_mask(box_mask(100, 100, 0, 0, 20, 20))
        pr = region_from_mask(box_mask(100, 100, 0, 0, 20, 12))  # IoU 0.6
        self.assertEqual(match([pr], [gt], strict=True).tp, 1)
        pr2 = region_from_mask(box_mask(100, 100, 0, 0, 20, 8))  # IoU 0.4: lenient only
        self.assertEqual(match([pr2], [gt], strict=True).tp, 0)
        self.assertEqual(match([pr2], [gt]).tp, 1)


class MetricTests(unittest.TestCase):
    def test_record_and_slices(self):
        h = w = 64
        gt = region_from_mask(box_mask(h, w, 5, 5, 25, 10))
        s = np.zeros((h, w), np.float32)
        s[5:10, 5:25] = 0.9
        s[40:50, 40:50] = 0.8  # FP
        rec = M.image_record(s, [gt], 0.5, "linear")
        self.assertEqual((rec["tp"], rec["fp"], rec["fn"]), (1, 1, 0))
        neg = M.image_record(s, [], 0.5, "unannotated")
        summ = M.summarize([rec, neg])
        self.assertAlmostEqual(summ["recall"], 1.0)
        self.assertAlmostEqual(summ["precision"], 1 / 4)  # 1 TP; 1 FP + 2 FP on the negative image
        self.assertAlmostEqual(summ["fp100"], 150.0)
        self.assertAlmostEqual(summ["fp100_unannotated"], 200.0)
        self.assertTrue(np.isnan(summ["fp100_negdefect"]))
        self.assertGreater(summ["cldice"], 0.0)

    def test_crack_counts(self):
        h = w = 64
        crack = region_from_mask(box_mask(h, w, 5, 5, 25, 10), label=1)
        scratch = region_from_mask(box_mask(h, w, 40, 40, 60, 45), label=3)
        s = np.zeros((h, w), np.float32)
        s[5:10, 5:25] = 0.9
        rec = M.image_record(s, [crack, scratch], 0.5, "linear", crack_label=1)
        self.assertEqual((rec["gt_crack"], rec["tp_crack"], rec["fn"]), (1, 1, 1))

    def test_merged_prediction_precision_counts_predictions(self):
        h = w = 100
        a, b = line_mask(h, w, (10, 50), (50, 50)), line_mask(h, w, (50, 50), (90, 20))
        s = np.zeros((h, w), np.float32)
        s[a | b] = 0.9
        s[5:10, 80:95] = 0.8  # FP
        summ = M.summarize([M.image_record(s, [region_from_mask(a), region_from_mask(b)], 0.5, "linear")])
        self.assertAlmostEqual(summ["recall"], 1.0)
        self.assertAlmostEqual(summ["precision"], 0.5)  # 1 matched prediction, 1 FP


class BootstrapTests(unittest.TestCase):
    def test_constant_groups_give_degenerate_ci(self):
        recs = [{"tp": 1, "fn": 1} for _ in range(10)]
        pt, lo, hi = cluster_bootstrap(recs, list(range(10)), M.recall, n=200)
        self.assertEqual((pt, lo, hi), (0.5, 0.5, 0.5))

    def test_clusters_are_resampled_whole(self):
        # Two groups: 9 near-duplicates all hit, 1 image missed. Cluster resampling sees 2 units,
        # so the CI is much wider than if the 10 images were treated as independent.
        recs = [{"tp": 1, "fn": 0}] * 9 + [{"tp": 0, "fn": 1}]
        _, lo_c, hi_c = cluster_bootstrap(recs, [0] * 9 + [1], M.recall, n=2000)
        _, lo_i, hi_i = cluster_bootstrap(recs, list(range(10)), M.recall, n=2000)
        self.assertGreater(hi_c - lo_c, hi_i - lo_i)
        self.assertLessEqual(lo_c, 0.9)
        self.assertGreaterEqual(hi_c, 0.9)

    def test_seeded(self):
        rng = np.random.default_rng(1)
        recs = [{"tp": int(rng.integers(0, 3)), "fn": int(rng.integers(0, 3))} for _ in range(40)]
        g = [i // 3 for i in range(40)]
        self.assertEqual(cluster_bootstrap(recs, g, M.recall, n=300), cluster_bootstrap(recs, g, M.recall, n=300))


class AgddTests(unittest.TestCase):
    def _img(self, classes, boxes, obbs):
        return A.AgddImage("t/x", {}, classes, np.array(boxes, float).reshape(-1, 4),
                           [np.array(o, float) for o in obbs], 640, 640)

    def test_box_iou(self):
        self.assertAlmostEqual(A.box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)

    def test_thin_diagonal_hits_via_oriented_box(self):
        # GT: a long diagonal crack; its axis-aligned box is huge, the thin prediction's box IoU is low
        # only if it covers part; the oriented box rule still matches it.
        obb = [[95, 105], [105, 95], [505, 495], [495, 505]]
        img = self._img([2], [[95, 95, 505, 505]], [obb])
        pred = region_from_mask(line_mask(640, 640, (100, 100), (250, 250), 3), score=0.9)
        self.assertLess(A.box_iou((pred.x0, pred.y0, pred.x1, pred.y1), img.boxes[0]), 0.3)
        rec, matched = A.eval_view([pred], img)
        self.assertEqual((rec["tp"], rec["fp"], rec["fn"], rec["tp_crack"]), (1, 0, 0, 1))
        self.assertEqual(matched, {0})

    def test_nonlinear_confusion(self):
        img = self._img([0, 1], [[10, 10, 60, 60], [300, 300, 400, 310]],
                        [[[10, 10], [60, 10], [60, 60], [10, 60]], [[300, 300], [400, 300], [400, 310], [300, 310]]])
        pred = region_from_mask(box_mask(640, 640, 12, 12, 58, 58), score=0.8)
        rec, _ = A.eval_view([pred], img)
        self.assertEqual((rec["tp"], rec["fp"], rec["fn"], rec["confusions"], rec["gt"]), (0, 1, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
