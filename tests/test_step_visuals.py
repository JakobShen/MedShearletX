import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from medshearletx.step_visuals import StepVisualizer
from medshearletx.transforms import IdentityTransform


class StepVisualTests(unittest.TestCase):
    @patch("medshearletx.scoring.Scorer.evaluate", side_effect=AssertionError("unexpected model scoring"))
    def test_all_steps_use_fixed_paths_and_preserve_original_pixels(self, scorer_call):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "runs" / "run_dog"
            pixels = np.random.default_rng(8).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            image = Image.fromarray(pixels)
            transform = IdentityTransform()
            coefficients = transform.encode(pixels.astype(float) / 255)
            visualizer = StepVisualizer(root, image, transform, coefficients, model="Gemini 3.5 Flash-Lite",
                                        target_label="Afghan hound", normalize_final=False, grid_size=2)
            history = [{"step": 0, "loss": 1.2, "mask_energy": 1}]
            paths = visualizer(np.ones((1, 2, 2)), history)
            self.assertEqual(paths["figure"], str(root / "figures/f_n/step000.png"))
            self.assertEqual(paths["kept"], str(root / "images/f_n/step000.png"))
            with Image.open(root / "input.png") as original:
                np.testing.assert_array_equal(np.asarray(original), pixels)
            with Image.open(paths["kept"]) as kept:
                np.testing.assert_array_equal(np.asarray(kept), pixels)
            with Image.open(paths["removed"]) as removed:
                self.assertEqual(np.asarray(removed).max(), 0)
            history.append({"step": 1, "loss": 0.9, "best_loss": 0.8, "requests": 12})
            mask = np.full((1, 2, 2), 0.5)
            visualizer(mask, history)
            np.testing.assert_array_equal(np.load(root / "masks/step001.npy"), mask)
            with Image.open(root / "figures/f_n/step001.png") as figure:
                self.assertEqual(figure.size, (1800, 750))
            metrics = json.loads((root / "metrics.json").read_text())
            self.assertEqual([row["step"] for row in metrics["steps"]], [0, 1])
            self.assertEqual(metrics["steps"][1]["preview_mask_mean"], 0.5)
            page = (root / "index.html").read_text()
            self.assertIn('type="range"', page)
            self.assertIn("figures/f_n/step001.png", page)
            self.assertIn("no per-step measured class probability", page)
            self.assertNotIn("Retained Prob", page)
            self.assertNotIn("fetch(", page)
            self.assertEqual(image.tobytes(), pixels.tobytes())
            scorer_call.assert_not_called()

    def test_duplicate_step_atomically_replaces_artifacts_and_normalizes_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.full((16, 16, 3), 100, dtype=np.uint8)
            pixels[0, 0] = 200
            image = Image.fromarray(pixels)
            transform = IdentityTransform()
            visualizer = StepVisualizer(root, image, transform, transform.encode(pixels / 255),
                                        model="test", target_label="dog", normalize_final=True, grid_size=2)
            visualizer(np.ones((1, 2, 2)), {"step": 1, "loss": 1})
            final_mask = np.array([[[0.5, 0.25], [0.25, 0.25]]])
            visualizer(final_mask, {"step": 1, "loss": 0.4})
            metrics = json.loads((root / "metrics.json").read_text())
            self.assertEqual(len(metrics["steps"]), 1)
            self.assertEqual(metrics["steps"][0]["loss"], 0.4)
            np.testing.assert_array_equal(np.load(root / "masks/step001.npy"), final_mask)
            with Image.open(root / "images/f_n/step001.png") as kept:
                self.assertEqual(np.asarray(kept).max(), 255)
                self.assertEqual(kept.getpixel((10, 10)), (64, 64, 64))
            with Image.open(root / "input.png") as original:
                np.testing.assert_array_equal(np.asarray(original), pixels)
            self.assertFalse(list(root.rglob(".step-*")))

    def test_final_index_records_independent_samples_and_selected_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = Image.new("RGB", (16, 16), "white")
            transform = IdentityTransform()
            visualizer = StepVisualizer(root, image, transform, transform.encode(np.ones((16, 16, 3))),
                                        model="test", target_label="dog", normalize_final=False, grid_size=2)
            visualizer(np.ones((1, 2, 2)), {"step": 3, "loss": 0.4})
            result = {"optimization_diagnostics": {"selected_step": 1}, "retained_frequency_ratio": 0.8}
            for name, count in (("reference", 10), ("retained", 8), ("removed", 2)):
                result[name] = {"sample_counts": {"dog": count}, "requests": 16,
                                "target_score": count / 16,
                                "diagnostics": {"sample_wilson_95": {"dog": [0.1, 0.9]}}}
            (root / "optimization.png").write_bytes(b"fixture")
            visualizer.finalize(result)
            data = json.loads((root / "metrics.json").read_text())
            self.assertEqual(data["final"]["selected_step"], 1)
            self.assertEqual([row["count"] for row in data["final"]["samples"]], [10, 8, 2])
            self.assertEqual(data["final"]["links"]["optimization"], "optimization.png")
            page = (root / "index.html").read_text()
            self.assertIn("Final selected-mask evaluation", page)
            self.assertIn("may differ from the last iteration preview", page)
            self.assertIn("comparison.png", page)

    @patch("medshearletx.scoring.Scorer.evaluate", side_effect=AssertionError("unexpected model scoring"))
    def test_full_mask_keeps_independent_pixel_values_without_block_expansion(self, scorer_call):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pixels = np.full((16, 16, 3), 200, dtype=np.uint8)
            image = Image.fromarray(pixels)
            transform = IdentityTransform()
            visualizer = StepVisualizer(root, image, transform, transform.encode(pixels / 255),
                                        model="test", target_label="dog", normalize_final=False,
                                        grid_size=2, mask_resolution="full")
            mask = np.ones((1, 16, 16))
            mask[0, 0, 0], mask[0, 0, 1], mask[0, 1, 0] = 0.25, 0.5, 0.75
            visualizer(mask, {"step": 1, "loss": 0.4})
            self.assertEqual(np.load(root / "masks/step001.npy").shape, (1, 16, 16))
            with Image.open(root / "images/f_n/step001.png") as kept:
                self.assertEqual(kept.getpixel((0, 0)), (50, 50, 50))
                self.assertEqual(kept.getpixel((1, 0)), (100, 100, 100))
                self.assertEqual(kept.getpixel((0, 1)), (150, 150, 150))
                self.assertEqual(kept.getpixel((1, 1)), (200, 200, 200))
            with Image.open(root / "input.png") as original:
                np.testing.assert_array_equal(np.asarray(original), pixels)
            self.assertEqual(json.loads((root / "metrics.json").read_text())["mask_resolution"], "full")
            scorer_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
