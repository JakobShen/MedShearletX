"""Offline 1000-class experiment, independent samples, and durable artifacts."""

from contextlib import redirect_stdout
from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from medshearletx.backends import MockBackend
from medshearletx.cli import main
from medshearletx.explainer import to_image
from medshearletx.imagenet_experiment import prepare_experiment, run_experiment
from medshearletx.tasks import load_imagenet_task


REPOSITORY = Path(__file__).resolve().parents[1]


class UsageMockBackend(MockBackend):
    """Keep the real seeded mock classifier; add offline token-usage evidence."""

    def __init__(self):
        super().__init__(seed=7)
        self.prompts = []
        self.calls = []

    def predict(self, image, task, *, require_logprobs, temperature=1.0):
        self.prompts.append(task.prompt)
        self.calls.append((len(task.labels), require_logprobs, temperature))
        prediction = super().predict(image, task, require_logprobs=require_logprobs, temperature=temperature)
        return replace(prediction, metadata={**prediction.metadata,
                                            "usage": {"promptTokenCount": 3, "candidatesTokenCount": 1}})


class ImageNetExperimentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        shutil.copyfile(REPOSITORY / "code" / "imagenet_utils" / "imagenet_labels.py",
                        self.root / "labels.py")
        pixels = np.full((16, 16, 3), 32, dtype=np.uint8)
        pixels[4:12, 4:12] = 220
        Image.fromarray(pixels).save(self.root / "image.png")
        self.config = {
            "model": {"backend": "mock", "seed": 7, "model": "mock-spatial"},
            "labels_path": "labels.py", "image_path": "image.png",
            "image_size": 16, "resize_mode": "stretch", "transform": {"name": "identity"},
            "selection_samples": 2, "evaluation_samples": 4, "evaluation_workers": 1,
            "optimization_sampling": {"repeats": 2, "temperature": 1.0, "workers": 1},
            "explainer": {"steps": 0, "grid_size": 2, "noise": "zeros", "noise_samples": 1,
                          "mask_init": 0.75, "unbiased_sampling_distortion": True,
                          "max_requests": 8, "seed": 42},
            "max_total_requests": 22,
        }

    def test_complete_offline_run_preserves_all_classes_and_independent_evidence(self):
        backend = UsageMockBackend()
        output = self.root / "run"
        progress = []
        with patch("medshearletx.imagenet_experiment.create_backend", return_value=backend):
            result = run_experiment(self.config, self.root, output, progress=progress.append)
        task = load_imagenet_task(self.root / "labels.py")
        prompt = task.prompt
        records = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["classes"], 1000)
        self.assertEqual(result["request_bound"], 22)
        # Reference selection is reused during optimization, saving two calls.
        self.assertEqual(result["requests"], 20)
        self.assertEqual(len(records), result["requests"])
        self.assertLessEqual(result["requests"], self.config["max_total_requests"])
        self.assertEqual({record["sequence_id"] for record in records}, set(range(1, 21)))
        self.assertTrue(all(record["status"] == "success" for record in records))
        self.assertTrue(all(record["prompt_sha256"] == sha256(prompt.encode("utf-8")).hexdigest()
                            for record in records))
        self.assertEqual(backend.prompts, [prompt] * result["requests"])
        self.assertEqual(backend.calls, [(1000, False, 1.0)] * result["requests"])
        self.assertEqual((output / "classification_prompt.txt").read_text(), prompt)
        self.assertIn("000: tench, Tinca tinca", prompt)
        self.assertIn("167: English foxhound", prompt)
        self.assertIn("999: toilet tissue, toilet paper, bathroom tissue", prompt)

        selection_counts = self._counts(records[:2], task)
        reference_counts = self._counts(records[2:6], task)
        self.assertEqual(result["selection"]["sample_counts"], selection_counts)
        self.assertEqual(result["reference"]["sample_counts"], reference_counts)
        self.assertNotEqual(selection_counts, reference_counts)
        self.assertEqual(result["selection"]["requests"], 2)
        self.assertEqual(result["reference"]["requests"], 4)
        target = max(selection_counts, key=selection_counts.get)
        self.assertEqual(result["target"], target)
        self.assertEqual(result["target_index"], task.labels.index(target))
        self.assertEqual(result["selection"]["target_score"], selection_counts[target] / 2)
        self.assertEqual(result["reference"]["target_score"], reference_counts[target] / 4)
        self.assertEqual(self._read(output / "selection.json"), result["selection"])
        self.assertEqual(self._read(output / "reference.json"), result["reference"])
        self.assertEqual(result["retained"]["sample_counts"], self._counts(records[-8:-4], task))
        self.assertEqual(result["removed"]["sample_counts"], self._counts(records[-4:], task))
        before = result["reference"]["target_score"]
        expected_ratio = result["retained"]["target_score"] / before if before else None
        self.assertEqual(result["retained_frequency_ratio"], expected_ratio)
        self.assertEqual(result["usage"], {"promptTokenCount": 60, "candidatesTokenCount": 20})
        self.assertEqual(self._read(output / "config.json"), self.config)
        self.assertEqual(self._read(output / "result.json"), result)
        self.assertFalse(result["calibrated_correctness"])
        self.assertTrue(progress)

        artifacts = {
            "input.png", "retained.png", "removed.png", "mask.npy", "history.json",
            "optimization_diagnostics.json", "plan.json", "selection.json", "reference.json",
            "retained.json", "removed.json", "config.json", "result.json", "requests.jsonl",
            "classification_prompt.txt", "explanation.png", "comparison.png",
            "explanation.pdf", "comparison.pdf",
        }
        self.assertTrue(artifacts.issubset({path.name for path in output.iterdir()}))
        for name in ("input.png", "retained.png", "removed.png"):
            with Image.open(output / name) as image:
                self.assertEqual(image.size, (16, 16))
        mask = np.load(output / "mask.npy", allow_pickle=False)
        self.assertEqual(mask.shape, (1, 2, 2))
        self.assertTrue(np.allclose(mask, 0.75))
        self.assertEqual(len(self._read(output / "history.json")), 1)
        self.assertEqual(result["optimization_diagnostics"]["requests"], 6)
        self.assertEqual(set(result["figures"]), {"explanation_png", "comparison_png",
                                                 "explanation_pdf", "comparison_pdf"})
        for name, path in result["figures"].items():
            path = Path(path)
            self.assertEqual(path.parent, output)
            self.assertGreater(path.stat().st_size, 1000)
            if name.endswith("png"):
                with Image.open(path) as figure:
                    self.assertGreater(min(figure.size), 500)

    def test_planning_rejects_global_and_optimizer_caps_before_creating_backend(self):
        for change in ("global", "optimizer"):
            config = json.loads(json.dumps(self.config))
            if change == "global":
                config["max_total_requests"] = 21
            else:
                config["explainer"]["max_requests"] = 7
            with self.subTest(change=change):
                with patch("medshearletx.imagenet_experiment.create_backend") as factory:
                    with self.assertRaisesRegex(ValueError, "requests|exceeds"):
                        prepare_experiment(config, self.root)
                    factory.assert_not_called()

    def test_existing_run_directory_is_preserved_before_any_prediction(self):
        backend = UsageMockBackend()
        output = self.root / "existing"
        output.mkdir()
        evidence = output / "earlier.json"
        evidence.write_text("preserve me")
        with patch("medshearletx.imagenet_experiment.create_backend", return_value=backend):
            with self.assertRaisesRegex(ValueError, "empty"):
                run_experiment(self.config, self.root, output)
        self.assertEqual(evidence.read_text(), "preserve me")
        self.assertEqual(backend.calls, [])

    def test_cli_dry_run_needs_no_vertex_key_and_never_contacts_provider(self):
        config = json.loads(json.dumps(self.config))
        config["model"] = {"backend": "vertex", "model": "gemini-3.5-flash-lite",
                           "api_key_env": "ABSENT_IMAGENET_TEST_KEY"}
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config))
        output = self.root / "unused-output"
        stdout = io.StringIO()
        with patch.dict(os.environ, {"ABSENT_IMAGENET_TEST_KEY": ""}):
            with patch("medshearletx.backends.vertex.http_json",
                       side_effect=AssertionError("dry run contacted provider")) as transport:
                with redirect_stdout(stdout):
                    status = main(["imagenet", "--config", str(config_path), "--root", str(self.root),
                                   "--output", str(output), "--dry-run"])
                transport.assert_not_called()
        self.assertEqual(status, 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["classes"], 1000)
        self.assertEqual(plan["total_requests_bound"], 22)
        self.assertEqual(plan["model"]["model"], "gemini-3.5-flash-lite")
        self.assertFalse(output.exists())

    def test_generic_five_and_complete_imagenet_task_planning(self):
        for definition, classes in (({"labels": ["Afghan hound", "beagle", "poodle", "cat", "bird"]}, 5),
                                    ({"imagenet_labels_path": "labels.py"}, 1000)):
            config = json.loads(json.dumps(self.config))
            del config["labels_path"]
            config["task"] = definition
            config["image_metadata"] = {"source": "offline fixture", "reference_label": "Afghan hound"}
            task, dataset, _, _, _, _, plan = prepare_experiment(config, self.root)
            self.assertEqual(len(task.labels), classes)
            self.assertEqual(plan["classes"], classes)
            self.assertEqual(dataset[0].sample_id, "image")
            self.assertEqual(dataset[0].metadata, config["image_metadata"])
            self.assertNotIn("paper_predicted_class", dataset[0].metadata)
            self.assertEqual(config["task"], definition)

    def test_generic_run_automatically_saves_current_iteration_visuals(self):
        config = json.loads(json.dumps(self.config))
        config["task"] = {"labels": ["Afghan hound", "beagle", "poodle", "cat", "bird"]}
        del config["labels_path"]
        config["sample_id"] = "afghan_fixture"
        config["image_metadata"] = {"source": "offline fixture"}
        config["explainer"].update(steps=1, mask_init=1, max_requests=14)
        config["max_total_requests"] = 28
        backend = UsageMockBackend()
        output = self.root / "generic-run"
        progress = []
        with patch("medshearletx.imagenet_experiment.create_backend", return_value=backend):
            result = run_experiment(config, self.root, output, progress=progress.append)
        self.assertEqual(result["classes"], 5)
        self.assertEqual(result["sample_id"], "afghan_fixture")
        self.assertEqual(result["image_metadata"], config["image_metadata"])
        self.assertIsNone(result["paper_predicted_class"])
        self.assertEqual(result["requests"], 26)  # Visual export adds no predictions.
        self.assertTrue(all(call[0] == 5 for call in backend.calls))
        self.assertIn("full 5-class task", progress[0])
        for step in (0, 1):
            for folder, suffix in (("images/f_n", "png"), ("images/removed", "png"),
                                   ("figures/f_n", "png"), ("masks", "npy")):
                self.assertTrue((output / folder / f"step{step:03d}.{suffix}").is_file())
        with Image.open(output / "input.png") as image:
            original = np.asarray(image)
        with Image.open(output / "images/f_n/step000.png") as kept:
            np.testing.assert_array_equal(np.asarray(kept), original)
        mask = np.load(output / "masks/step001.npy")
        indices = np.arange(16) * 2 // 16
        dense = mask[0, indices[:, None], indices[None, :]][:, :, None]
        expected = np.rint(original * dense).astype(np.uint8)
        with Image.open(output / "images/f_n/step001.png") as kept:
            np.testing.assert_array_equal(np.asarray(kept), expected)
        metrics = self._read(output / "metrics.json")
        history = self._read(output / "history.json")
        self.assertEqual([row["loss"] for row in metrics["steps"]], [row["loss"] for row in history])
        self.assertEqual(metrics["final"]["selected_step"], result["optimization_diagnostics"]["selected_step"])
        self.assertEqual(metrics["final"]["samples"][1]["count"], result["retained"]["sample_counts"][result["target"]])
        self.assertTrue(Path(result["step_visuals"]["index"]).is_file())
        self.assertTrue((output / "optimization.png").is_file())
        self.assertEqual(set(result["optimization_figures"]), {"optimization_png", "optimization_pdf"})

    def test_experiment_cli_alias_uses_default_run_folder_and_explicit_output(self):
        config = json.loads(json.dumps(self.config))
        config["run_name"] = "afghan-five"
        config["task"] = {"labels": ["a", "b", "c", "d", "e"]}
        del config["labels_path"]
        config_path = self.root / "fiveway.json"
        config_path.write_text(json.dumps(config))
        for explicit in (None, self.root / "chosen-output"):
            arguments = ["experiment", "--config", str(config_path), "--root", str(self.root), "--dry-run"]
            if explicit:
                arguments.extend(("--output", str(explicit)))
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(arguments), 0)
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["classes"], 5)
            destination = Path(plan["output"])
            if explicit:
                self.assertEqual(destination, explicit)
            else:
                self.assertEqual(destination.parent, self.root / "runs")
                self.assertRegex(destination.name, r"^afghan-five-\d{8}-\d{6}$")
            self.assertFalse(destination.exists())

    def test_full_mask_experiment_retains_float_resize_and_last_step_without_extra_calls(self):
        config = json.loads(json.dumps(self.config))
        config["task"] = {"labels": ["bright", "dark"]}
        del config["labels_path"]
        config["resize_filter"] = "tensor_bilinear"
        config["explainer"].update(steps=1, mask_init=1, mask_resolution="full",
                                    mask_selection="last", resample_noise=True, max_requests=14)
        config["max_total_requests"] = 28
        source = np.repeat(np.array([[0, 100], [200, 255]], dtype=np.uint8)[..., None], 3, axis=2)
        Image.fromarray(source).save(self.root / "image.png")
        backend = UsageMockBackend()
        output = self.root / "full-mask-experiment"
        with patch("medshearletx.imagenet_experiment.create_backend", return_value=backend):
            result = run_experiment(config, self.root, output, progress=lambda message: None)
        self.assertEqual(result["request_bound"], 28)
        self.assertEqual(result["requests"], 26)
        self.assertEqual(len(backend.calls), 26)
        self.assertEqual(result["optimization_diagnostics"]["selected_step"], 1)
        self.assertEqual(result["optimization_diagnostics"]["mask_resolution"], "full")
        tensor = np.load(output / "input_tensor.npy")
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(np.any(abs(tensor * 255 - np.rint(tensor * 255)) > 0.1))
        with Image.open(output / "input.png") as image:
            self.assertEqual(image.tobytes(), to_image(tensor.astype(np.float64)).tobytes())
        mask = np.load(output / "mask.npy")
        self.assertEqual(mask.shape, (1, 16, 16))
        expected = to_image(tensor * mask[0, ..., None])
        for path in ("retained.png", "images/f_n/step001.png"):
            with Image.open(output / path) as image:
                self.assertEqual(image.tobytes(), expected.tobytes())

    def test_keyboard_interrupt_preserves_checkpoint_and_records_interrupted_status(self):
        output = self.root / "interrupted"

        def interrupted_run(config, root, directory, progress):
            directory.mkdir()
            (directory / "plan.json").write_text("{}")
            np.save(directory / "checkpoint-mask.npy", np.array([[[0.25]]]))
            raise KeyboardInterrupt()

        with patch("medshearletx.imagenet_experiment._run_experiment", side_effect=interrupted_run):
            with self.assertRaises(KeyboardInterrupt):
                run_experiment(self.config, self.root, output)
        failure = self._read(output / "failure.json")
        self.assertEqual(failure["status"], "interrupted")
        self.assertEqual(failure["error_type"], "KeyboardInterrupt")
        self.assertTrue(failure["checkpoint_available"])
        np.testing.assert_array_equal(np.load(output / "checkpoint-mask.npy"), [[[0.25]]])

    @staticmethod
    def _counts(records, task):
        counts = dict.fromkeys(task.labels, 0)
        for record in records:
            counts[record["prediction"]["sampled_label"]] += 1
        return counts

    @staticmethod
    def _read(path):
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
