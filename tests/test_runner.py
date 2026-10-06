from contextlib import redirect_stdout
import io
import json
import math
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from medshearletx.cli import main
from medshearletx.data import ImageDataset
from medshearletx.explainer import to_image
from medshearletx.runner import plan_run, prepare, preprocess, preprocess_pixels, run
from medshearletx.types import Prediction


class NativeEvidenceBackend:
    """Offline reported evidence with an actual sampled class."""

    model = "offline-native-evidence"

    def __init__(self, evidence, observed):
        self.evidence = evidence
        self.observed = observed
        self.calls = []

    def predict(self, image, task, *, require_logprobs, temperature=1.0):
        self.calls.append((task.labels, require_logprobs, temperature))
        return Prediction(label_logprobs=self.evidence if require_logprobs else None,
                          sampled_label=self.observed, metadata={"logprob_scope": "reported"})


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

    def test_fixed_target_planning_requires_context_before_output_or_backend(self):
        self.config["scores"] = ["target_probability"]
        output = self.root / "missing-target"
        with patch("medshearletx.runner.create_backend", side_effect=AssertionError("backend constructed")):
            for action in (
                lambda: prepare(self.config),
                lambda: plan_run(self.config, self.dataset),
                lambda: run(self.config, self.dataset, output),
                lambda: run(self.config, self.dataset, output, target="unknown"),
            ):
                with self.subTest(action=action), self.assertRaises(ValueError):
                    action()
        self.assertFalse(output.exists())

    def test_fixed_target_planning_accepts_config_and_argument_with_native_budget(self):
        self.config.update(scores=["target_probability"], target="dark")
        task, scorers, _ = prepare(self.config, target="bright")
        self.assertEqual(task.labels, ("bright", "dark"))
        self.assertEqual(scorers["target_probability"].target, "bright")
        plan = plan_run(self.config, self.dataset)
        self.assertEqual(plan["target"], "dark")
        self.assertEqual(plan["total_requests_bound"], 7)
        self.assertEqual(plan["requests_per_image_bound"], {"target_probability": 7})
        probe = plan_run(self.config, self.dataset, probe=True, target="bright")
        self.assertEqual(probe["target"], "bright")
        self.assertEqual(probe["total_requests_bound"], 1)
        self.config["max_total_requests"] = 6
        with self.assertRaisesRegex(ValueError, "max_total_requests"):
            plan_run(self.config, self.dataset)

    def test_partial_1000_class_target_run_preserves_raw_score_and_observed_label(self):
        labels = [f"class_{index}" for index in range(1000)]
        target, observed = labels[:2]
        self.config.update(task={"labels": labels, "question": "Which class?"},
                           scores=["target_probability"], target=target, step_visuals=False)
        backend = NativeEvidenceBackend({target: math.log(0.9), observed: math.log(0.05)}, observed)
        output = self.root / "partial-target-run"
        with patch("medshearletx.runner.create_backend", return_value=backend):
            rows = run(self.config, self.dataset, output)
        row = rows[0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["target"], target)
        self.assertFalse(row["reference_reused"])
        for name in ("reference", "retained", "removed"):
            result = row[name]
            self.assertAlmostEqual(result["target_score"], 0.9)
            self.assertEqual(set(result["probabilities"]), {target})
            self.assertAlmostEqual(result["log_probabilities"][target], math.log(0.9))
            self.assertEqual(result["observed_label"], observed)
            self.assertEqual(result["predicted_label"], observed)
            self.assertEqual(result["diagnostics"]["normalization"], "raw_unnormalized")
        self.assertTrue(all(call[0] == tuple(labels) and call[1] for call in backend.calls))
        self.assertEqual(row["prediction_attempts"], len(backend.calls))
        summary = json.loads((output / "summary.json").read_text())
        self.assertLessEqual(len(backend.calls), summary["request_bound"])
        self.assertTrue((output / "0000/target_probability/mask.npy").is_file())
        self.assertEqual(json.loads((output / "plan.json").read_text())["target"], target)

    def test_only_complete_native_modes_share_reference(self):
        for modes in (
            ["probability", "target_probability", "log_margin"],
            ["target_probability", "probability", "log_margin"],
        ):
            with self.subTest(modes=modes):
                backend = NativeEvidenceBackend({"bright": math.log(0.3), "dark": math.log(0.1)}, "dark")
                output = self.root / modes[0]
                with patch("medshearletx.runner.create_backend", return_value=backend):
                    rows = run(self.config, self.dataset, output, scores=modes, target="bright", probe=True)
                self.assertEqual([row["status"] for row in rows], ["ok"] * 3)
                by_mode = {row["mode"]: row for row in rows}
                self.assertFalse(by_mode["target_probability"]["reference_reused"])
                self.assertFalse(by_mode["probability"]["reference_reused"])
                self.assertTrue(by_mode["log_margin"]["reference_reused"])
                self.assertAlmostEqual(by_mode["probability"]["reference"]["target_score"], 0.75)
                self.assertAlmostEqual(by_mode["target_probability"]["reference"]["target_score"], 0.3)
                self.assertAlmostEqual(by_mode["log_margin"]["reference"]["target_score"], math.log(3))
                self.assertEqual(by_mode["target_probability"]["reference"]["predicted_label"], "dark")
                self.assertEqual(by_mode["probability"]["reference"]["predicted_label"], "bright")
                self.assertEqual(len(backend.calls), 2)
                self.assertEqual(sum(row["requests"] for row in rows), 2)

    def test_partial_target_evidence_is_unavailable_for_complete_modes(self):
        labels = [f"class_{index}" for index in range(1000)]
        target, observed = labels[:2]
        self.config["task"] = {"labels": labels, "question": "Which class?"}
        backend = NativeEvidenceBackend({target: math.log(0.9), observed: math.log(0.05)}, observed)
        with patch("medshearletx.runner.create_backend", return_value=backend):
            rows = run(self.config, self.dataset, self.root / "scope-check", target=target, probe=True,
                       scores=["target_probability", "probability", "log_margin"])
        self.assertEqual([row["status"] for row in rows], ["ok", "unavailable", "unavailable"])
        self.assertEqual([row["prediction_attempts"] for row in rows], [1, 1, 1])
        self.assertTrue(all("every candidate label" in row["error"] for row in rows[1:]))

    def test_unreported_fixed_target_fails_explicitly_at_probe(self):
        backend = NativeEvidenceBackend({"dark": math.log(0.9)}, "dark")
        with patch("medshearletx.runner.create_backend", return_value=backend):
            rows = run(self.config, self.dataset, self.root / "unreported-target", probe=True,
                       scores=["target_probability"], target="bright")
        self.assertEqual(rows[0]["status"], "unavailable")
        self.assertEqual(rows[0]["target"], "bright")
        self.assertIn("does not report the fixed target", rows[0]["error"])
        self.assertNotIn("reference", rows[0])
        self.assertEqual(len(backend.calls), 1)

    def test_cli_fixed_target_dry_run_passes_explicit_override(self):
        self.config["target"] = "dark"
        cfg = self.root / "target-config.json"
        cfg.write_text(json.dumps(self.config))
        for command in ("run", "probe"):
            output = self.root / f"target-{command}-dry"
            printed = io.StringIO()
            with redirect_stdout(printed), patch("medshearletx.backends.MockBackend.predict",
                                                  side_effect=AssertionError("model called")):
                code = main([command, "--config", str(cfg), "--images", str(self.images),
                             "--output", str(output), "--scores", "target_probability",
                             "--target", "bright", "--dry-run"])
            self.assertEqual(code, 0)
            plan = json.loads(printed.getvalue())
            self.assertEqual(plan["target"], "bright")
            self.assertEqual(plan["total_requests_bound"], 1 if command == "probe" else 7)
            self.assertFalse(output.exists())
        del self.config["target"]
        cfg.write_text(json.dumps(self.config))
        output = self.root / "cli-missing-target"
        with redirect_stdout(io.StringIO()), patch("medshearletx.runner.create_backend",
                                                   side_effect=AssertionError("backend constructed")):
            code = main(["run", "--config", str(cfg), "--images", str(self.images),
                         "--output", str(output), "--scores", "target_probability", "--dry-run"])
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())

    def test_cli_target_probe_records_effective_context_and_real_prediction(self):
        self.config["target"] = "dark"
        cfg = self.root / "target-probe-config.json"
        cfg.write_text(json.dumps(self.config))
        output = self.root / "cli-target-probe"
        backend = NativeEvidenceBackend({"bright": math.log(0.9), "dark": math.log(0.05)}, "dark")
        with redirect_stdout(io.StringIO()), patch("medshearletx.runner.create_backend", return_value=backend):
            code = main(["probe", "--config", str(cfg), "--images", str(self.images),
                         "--output", str(output), "--scores", "target_probability", "--target", "bright"])
        self.assertEqual(code, 0)
        saved_config = json.loads((output / "config.json").read_text())
        self.assertEqual(saved_config["target"], "bright")
        self.assertEqual(saved_config["scores"], ["target_probability"])
        result = json.loads((output / "results.json").read_text())[0]
        self.assertEqual(result["target"], "bright")
        self.assertAlmostEqual(result["reference"]["target_score"], 0.9)
        self.assertEqual(result["reference"]["predicted_label"], "dark")
        self.assertEqual(len(backend.calls), 1)

    def test_final_unknown_target_scores_are_serialized_without_zero_fill(self):
        self.config.update(scores=["target_probability"], target="bright", step_visuals=False)
        backend = NativeEvidenceBackend({"bright": math.log(0.9)}, "bright")
        explanation = SimpleNamespace(
            image=Image.new("RGB", (32, 32)), removed_image=Image.new("RGB", (32, 32)),
            mask=np.ones((4, 4)), history=[], retained=None, removed=None,
            diagnostics={"requests": 0, "final_score_status": "target_not_reported"},
        )
        output = self.root / "unknown-final-scores"
        with patch("medshearletx.runner.create_backend", return_value=backend), \
                patch("medshearletx.runner.BlackBoxShearletX.explain", return_value=explanation):
            rows = run(self.config, self.dataset, output)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertIsNone(rows[0]["retained"])
        self.assertIsNone(rows[0]["removed"])
        self.assertEqual(rows[0]["diagnostics"]["final_score_status"], "target_not_reported")
        saved = json.loads((output / "results.json").read_text())[0]
        self.assertIsNone(saved["retained"])
        self.assertIsNone(saved["removed"])
        self.assertTrue((output / "0000/target_probability/retained.png").is_file())

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
