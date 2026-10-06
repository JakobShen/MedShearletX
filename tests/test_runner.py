import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from medshearletx.cli import main
from medshearletx.data import ImageDataset
from medshearletx.explainer import to_image
from medshearletx.runner import plan_run, preprocess, preprocess_pixels, run


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        Image.new("RGB", (48, 32), (230, 230, 230)).save(self.images / "sample.png")
        self.dataset = ImageDataset.from_folder(self.images)
        self.config = {
            "model": {"backend": "mock", "seed": 7},
            "task": {"labels": ["bright", "dark"], "question": "Is it bright?"},
            "sampling": {"repeats": 3, "temperature": 1.0},
            "transform": {"name": "identity"}, "image_size": 32,
            "explainer": {"steps": 1, "max_requests": 100}, "max_total_requests": 100,
        }

    def test_comparison_saves_artifacts_and_reuses_native_reference(self):
        output = self.root / "results"
        rows = run(self.config, self.dataset, output)
        self.assertEqual([row["status"] for row in rows], ["ok"] * 3)
        self.assertTrue(rows[1]["reference_reused"])
        self.assertEqual(set(rows[0]["reference"]["log_probabilities"]), {"bright", "dark"})
        self.assertIsNone(rows[2]["reference"]["log_probabilities"])
        self.assertEqual(len({row["target"] for row in rows}), 1)
        for mode in ("probability", "log_margin", "agreement"):
            for name in ("retained.png", "removed.png", "mask.npy", "history.json"):
                self.assertTrue((output / "0000" / mode / name).exists())
            self.assertTrue((output / "0000" / mode / "figures/f_n/step001.png").is_file())
            self.assertTrue((output / "0000" / mode / "index.html").is_file())
        summary = json.loads((output / "summary.json").read_text())
        self.assertLessEqual(summary["prediction_attempts"], summary["request_bound"])
        self.assertEqual(sum(row["requests"] for row in rows), summary["prediction_attempts"])
        self.assertEqual(rows[0]["original_size"], [48, 32])

    def test_unavailable_logprobs_are_explicit_and_sampling_still_runs(self):
        self.config["model"]["supports_logprobs"] = False
        rows = run(self.config, self.dataset, self.root / "partial", probe=True)
        self.assertEqual([row["status"] for row in rows], ["unavailable", "unavailable", "ok"])
        self.assertGreater(rows[0]["prediction_attempts"], 0)

    def test_dry_run_never_sends_model_request_or_writes_results(self):
        cfg = self.root / "config.json"
        cfg.write_text(json.dumps(self.config))
        with patch("medshearletx.backends.MockBackend.predict", side_effect=AssertionError("network attempted")):
            self.assertEqual(main(["run", "--config", str(cfg), "--images", str(self.images),
                                   "--output", str(self.root / "dry"), "--dry-run"]), 0)
        self.assertFalse((self.root / "dry").exists())

    def test_budget_check_runs_before_calls_and_existing_outputs_are_protected(self):
        self.config["max_total_requests"] = 1
        with self.assertRaisesRegex(ValueError, "max_total_requests"):
            plan_run(self.config, self.dataset)
        self.config["max_total_requests"] = 100
        output = self.root / "existing"
        output.mkdir()
        (output / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            run(self.config, self.dataset, output)
        self.assertEqual((output / "keep.txt").read_text(), "keep")

    def test_bad_dimensions_and_label_type_fail_before_predictions(self):
        self.config["explainer"]["grid_size"] = 64
        with self.assertRaisesRegex(ValueError, "image_size"):
            plan_run(self.config, self.dataset)
        self.config["explainer"]["grid_size"] = 4
        self.config["task"]["labels"] = "AB"
        with self.assertRaisesRegex(ValueError, "labels"):
            plan_run(self.config, self.dataset)

    def test_bad_transform_preflight_never_calls_model(self):
        with patch("medshearletx.transforms.IdentityTransform.encode", side_effect=ValueError("bad local transform")):
            with patch("medshearletx.backends.MockBackend.predict", side_effect=AssertionError("model called")):
                with self.assertRaisesRegex(ValueError, "bad local transform"):
                    run(self.config, self.dataset, self.root / "bad_transform")

    def test_letterbox_preserves_geometry_and_stretch_is_explicit(self):
        source = Image.new("RGB", (48, 24), "white")
        processed, metadata = preprocess(source, self.config)
        self.assertEqual(metadata["fitted_size"], [32, 16])
        self.assertEqual(metadata["offset"], [0, 8])
        self.assertEqual(processed.getpixel((16, 0)), (0, 0, 0))
        self.assertEqual(processed.getpixel((16, 16)), (255, 255, 255))
        stretch, metadata = preprocess(source, {**self.config, "resize_mode": "stretch"})
        self.assertEqual(metadata["mode"], "stretch")
        self.assertEqual(stretch.getpixel((16, 0)), (255, 255, 255))

    def test_missing_optional_plotting_dependency_preserves_core_run(self):
        self.config["scores"] = ["probability"]
        output = self.root / "without-figures"
        with patch("medshearletx.runner.find_spec", return_value=None):
            rows = run(self.config, self.dataset, output)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertIn("figures", rows[0]["iteration_visuals_unavailable"])
        self.assertNotIn("iteration_index", rows[0])
        self.assertTrue((output / "0000/probability/retained.png").is_file())

    def test_tensor_resize_uses_half_pixel_coordinates_and_preserves_fractions(self):
        source = np.repeat(np.array([[0, 100], [200, 255]], dtype=np.uint8)[..., None], 3, axis=2)
        image = Image.fromarray(source)
        config = {"image_size": 4, "resize_mode": "stretch", "resize_filter": "tensor_bilinear"}
        expected = np.array([[0, 25, 75, 100],
                             [50, 72.1875, 116.5625, 138.75],
                             [150, 166.5625, 199.6875, 216.25],
                             [200, 213.75, 241.25, 255]])
        pixels = preprocess_pixels(image, config)
        self.assertEqual(pixels.dtype, np.float32)
        np.testing.assert_allclose(pixels[..., 0] * 255, expected, atol=3e-5, rtol=0)
        processed, metadata = preprocess(image, config)
        self.assertEqual(processed.tobytes(), to_image(pixels.astype(np.float64)).tobytes())
        self.assertFalse(metadata["align_corners"])
        self.assertFalse(metadata["antialias"])
        self.assertEqual(metadata["tensor_dtype"], "float32")
        self.assertGreater(abs(pixels[1, 1, 0] * 255 - processed.getpixel((1, 1))[0]), 0.1)

    def test_tensor_downsample_has_no_antialias_prefilter(self):
        source = np.repeat(np.arange(16, dtype=np.uint8).reshape(4, 4, 1), 3, axis=2)
        pixels = preprocess_pixels(Image.fromarray(source), {
            "image_size": 2, "resize_mode": "stretch", "resize_filter": "tensor_bilinear"})
        np.testing.assert_allclose(pixels[..., 0] * 255, [[2.5, 4.5], [10.5, 12.5]], atol=2e-6, rtol=0)

    def test_full_mask_run_preserves_float_input_and_last_step_query_bound(self):
        self.config.update(scores=["agreement"], resize_mode="stretch", resize_filter="tensor_bilinear",
                           sampling={"repeats": 2}, max_total_requests=14)
        self.config["explainer"].update(mask_resolution="full", mask_selection="last", steps=1,
                                        noise_samples=1, resample_noise=True, max_requests=14)
        source = np.repeat(np.array([[0, 100], [200, 255]], dtype=np.uint8)[..., None], 3, axis=2)
        Image.fromarray(source).save(self.images / "sample.png")
        output = self.root / "full-mask-run"
        self.assertEqual(plan_run(self.config, self.dataset)["total_requests_bound"], 14)
        rows = run(self.config, self.dataset, output)
        row = rows[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["requests"], 14)
        self.assertEqual(row["diagnostics"]["selected_step"], 1)
        self.assertEqual(row["diagnostics"]["mask_resolution"], "full")
        mask = np.load(output / "0000/agreement/mask.npy")
        self.assertEqual(mask.shape, (1, 32, 32))
        tensor = np.load(output / "0000/input_tensor.npy")
        self.assertEqual(tensor.dtype, np.float32)
        with Image.open(output / "0000/input.png") as image:
            self.assertEqual(image.tobytes(), to_image(tensor.astype(np.float64)).tobytes())
        with Image.open(output / "0000/agreement/images/f_n/step001.png") as image:
            self.assertEqual(image.tobytes(), to_image(tensor * mask[0, ..., None]).tobytes())
        self.assertEqual(json.loads((output / "summary.json").read_text())["prediction_attempts"], 14)


if __name__ == "__main__":
    unittest.main()
