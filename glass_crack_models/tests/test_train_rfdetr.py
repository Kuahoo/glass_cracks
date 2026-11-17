import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from glass_crack_models.train import train_rfdetr as T


def fake_merged(root: Path, n_train: int = 20, n_valid: int = 10) -> Path:
    """Minimal glass_merged_seg: image/label stubs, data_all.yaml and a balanced list repeating the first image."""
    for split, n in (("train", n_train), ("valid", n_valid)):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for i in range(n):
            (root / split / "images" / f"{split}{i:02d}.jpg").write_bytes(b"img")
            (root / split / "labels" / f"{split}{i:02d}.txt").write_text("1 0.1 0.1 0.9 0.1 0.9 0.9\n")
    (root / "data_all.yaml").write_text("names: [bubble, crack, inclusion, scratch]\n")
    lines = ["./train/images/train00.jpg"] + [f"./train/images/train{i:02d}.jpg" for i in range(n_train)]
    (root / "train_balanced.txt").write_text("\n".join(lines) + "\n")
    return root


class BuildDatasetTests(unittest.TestCase):
    def build(self, td, **kw):
        src = fake_merged(Path(td) / "src")
        with patch.dict(T.DATASETS, {"merged": lambda: src}):
            return T.build_dataset({"dataset": "merged", "weighting": "balanced"}, Path(td) / "out", **kw)

    def test_balanced_repeats_become_separate_files_and_test_is_left_out(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.build(td)
            imgs = sorted(p.name for p in (out / "train" / "images").iterdir())
            self.assertEqual(len(imgs), 21)
            self.assertIn("train00_w1.jpg", imgs)
            self.assertEqual((out / "train" / "labels" / "train00_w1.txt").read_text(),
                             (out / "train" / "labels" / "train00.txt").read_text())
            self.assertEqual(len(list((out / "valid" / "images").iterdir())), 10)
            self.assertFalse((out / "test").exists())
            self.assertIn("scratch", (out / "data.yaml").read_text())

    def test_smoke_is_a_small_subset_without_repeats(self):
        with tempfile.TemporaryDirectory() as td:
            out = self.build(td, smoke=True)
            imgs = [p.name for p in (out / "train" / "images").iterdir()]
            self.assertEqual(len(imgs), T.SMOKE_TRAIN)
            self.assertFalse(any("_w" in n for n in imgs))
            self.assertEqual(len(list((out / "valid" / "images").iterdir())), T.SMOKE_VALID)

    def test_missing_weighting_list_names_the_fix(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(FileNotFoundError, "relabel_classes"):
                T.train_images(Path(td), "balanced")


class VariantTests(unittest.TestCase):
    def test_unknown_variant(self):
        try:
            import rfdetr  # noqa: F401
        except ImportError:
            self.skipTest("rfdetr not installed")
        self.assertEqual(T.variant_class("large").__name__, "RFDETRSegLarge")
        with self.assertRaises(KeyError):
            T.variant_class("huge")


if __name__ == "__main__":
    unittest.main()
