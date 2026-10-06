"""Task configs keep complete or explicitly restricted candidate sets."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from medshearletx.backends.base import parse_label
from medshearletx.tasks import load_imagenet_task, load_task


REPOSITORY = Path(__file__).resolve().parents[1]
AFGHAN_FIVE_CLASSES = ("Afghan hound", "beagle", "golden retriever", "English foxhound", "bulldog")


class TaskFactoryTests(unittest.TestCase):
    def test_imagenet_relative_path_keeps_all_original_classes_and_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.py"
            shutil.copyfile(REPOSITORY / "code" / "imagenet_utils" / "imagenet_labels.py", path)
            task = load_task({"imagenet_labels_path": "labels.py"}, root=directory)
        original = load_imagenet_task(REPOSITORY / "code" / "imagenet_utils" / "imagenet_labels.py")
        self.assertEqual(task, original)
        self.assertEqual(len(task.labels), 1000)
        self.assertEqual(task.codes[160], "160")
        self.assertEqual(task.labels[160], "Afghan hound, Afghan")
        self.assertIn("ALL 1000", task.prompt)
        self.assertIn("160: Afghan hound, Afghan", task.prompt)
        self.assertEqual(parse_label("160", task), "Afghan hound, Afghan")

    def test_five_way_task_keeps_order_spelling_and_original_letter_codes(self):
        config = {"labels": list(AFGHAN_FIVE_CLASSES), "question": "What breed is the dog in this image?"}
        before = json.loads(json.dumps(config))
        task = load_task(config)
        self.assertEqual(task.labels, AFGHAN_FIVE_CLASSES)
        self.assertEqual(task.codes, ("A", "B", "C", "D", "E"))
        self.assertEqual(task.question, config["question"])
        self.assertEqual(parse_label("A", task), "Afghan hound")
        self.assertEqual(parse_label("D", task), "English foxhound")
        self.assertEqual(config, before)
        self.assertIn("A: Afghan hound", task.prompt)
        self.assertIn("E: bulldog", task.prompt)

    def test_task_sources_are_mutually_exclusive_and_required(self):
        for config in ({}, {"question": "Classify."},
                       {"labels": ["first", "second"], "imagenet_labels_path": "missing.py"}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "exactly one"):
                load_task(config)

    def test_question_override_preserves_complete_imagenet_classes(self):
        source = REPOSITORY / "code" / "imagenet_utils" / "imagenet_labels.txt"
        original = load_imagenet_task(source)
        task = load_task({"imagenet_labels_path": source, "question": "Classify the image."},
                         root="/unused-root")
        self.assertEqual(task.labels, original.labels)
        self.assertEqual(task.display_labels, original.display_labels)
        self.assertEqual(task.codes, original.codes)
        self.assertEqual(task.question, "Classify the image.")

    def test_invalid_generic_labels_and_imagenet_paths_fail_explicitly(self):
        for config in ({"labels": "first, second"}, {"labels": {"first": 1, "second": 2}},
                       {"labels": None}, {"labels": ["same", "same"]},
                       {"labels": ["one"]}, {"labels": ["first", "second"], "question": " "},
                       {"imagenet_labels_path": " "}, {"imagenet_labels_path": None}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                load_task(config)
        with self.assertRaisesRegex(ValueError, "mapping"):
            load_task(None)

    def test_afghan_templates_use_the_same_image_and_model_with_distinct_candidate_sets(self):
        full = self._config("vertex-afghan-imagenet.json")
        five = self._config("vertex-afghan-fiveway.json")
        for config in (full, five):
            self.assertEqual(config["image_path"], "code/imgs/ILSVRC2012_val_00017625.JPEG")
            self.assertEqual(config["image_size"], 256)
            self.assertEqual(config["resize_mode"], "stretch")
            self.assertEqual(config["transform"], {"name": "shearlet", "scales": 4})
            self.assertEqual(config["model"]["backend"], "vertex")
            self.assertEqual(config["model"]["model"], "gemini-3.5-flash-lite")
            self.assertEqual(config["model"]["max_output_tokens"], 128)
            self.assertEqual(config["model"]["generation_options"]["thinkingConfig"]["thinkingLevel"], "MINIMAL")
        self.assertEqual(full["model"], five["model"])
        full_task = load_task(full["task"], root=REPOSITORY)
        five_task = load_task(five["task"], root=REPOSITORY)
        self.assertEqual(len(full_task.labels), 1000)
        self.assertEqual(five_task.labels, AFGHAN_FIVE_CLASSES)
        self.assertEqual(full_task.codes[160], "160")
        self.assertEqual(five_task.codes[0], "A")

    @staticmethod
    def _config(name):
        return json.loads((REPOSITORY / "configs" / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
