import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from medshearletx.cli import main
from medshearletx.data import ImageDataset
from medshearletx.runner import plan_run, preprocess, run


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
        self.assertEqual(len({row["target"] for row in rows}), 1)
        for mode in ("probability", "log_margin", "agreement"):
            for name in ("retained.png", "removed.png", "mask.npy", "history.json"):
                self.assertTrue((output / "0000" / mode / name).exists())
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


if __name__ == "__main__":
    unittest.main()
