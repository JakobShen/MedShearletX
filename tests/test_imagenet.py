"""The VLM uses the same complete ImageNet label set as the original task."""

import math
from pathlib import Path
import tempfile
import unittest

from medshearletx.backends.base import aggregate_code_logprobs, parse_label
from medshearletx.tasks import load_imagenet_task
from medshearletx.types import ClassificationTask, InvalidPredictionError


LABELS_DIR = Path(__file__).resolve().parents[1] / "code" / "imagenet_utils"


class ImageNetTaskTests(unittest.TestCase):
    def setUp(self):
        self.task = load_imagenet_task(LABELS_DIR / "imagenet_labels.py")

    def test_original_py_and_txt_have_the_same_complete_ordered_labels(self):
        text_task = load_imagenet_task(LABELS_DIR / "imagenet_labels.txt")
        self.assertEqual(self.task.labels, text_task.labels)
        self.assertEqual(self.task.display_labels, text_task.display_labels)
        self.assertEqual(len(self.task.labels), 1000)
        self.assertEqual(self.task.labels[0], "tench, Tinca tinca")
        self.assertEqual(self.task.labels[167], "English foxhound")
        self.assertEqual(self.task.labels[999], "toilet tissue, toilet paper, bathroom tissue")

    def test_shared_crane_name_keeps_two_distinct_class_keys(self):
        self.assertEqual(self.task.display_labels[134], "crane")
        self.assertEqual(self.task.display_labels[517], "crane")
        self.assertEqual(self.task.labels[134], "crane [ImageNet 134]")
        self.assertEqual(self.task.labels[517], "crane [ImageNet 517]")
        self.assertEqual(len(set(self.task.labels)), 1000)
        self.assertIn("134: crane\n", self.task.prompt)
        self.assertIn("517: crane\n", self.task.prompt)

    def test_codes_preserve_class_indices_and_prompt_includes_every_class(self):
        self.assertEqual(self.task.codes, tuple(f"{index:03d}" for index in range(1000)))
        self.assertIn("dominant object", self.task.prompt)
        self.assertIn("ALL 1000", self.task.prompt)
        self.assertIn("000: tench, Tinca tinca", self.task.prompt)
        self.assertIn("167: English foxhound", self.task.prompt)
        self.assertIn("999: toilet tissue, toilet paper, bathroom tissue", self.task.prompt)
        self.assertIn("zero-padded numeric option code", self.task.prompt)
        self.assertIn("no explanation", self.task.prompt)
        self.assertIn("or confidence value", self.task.prompt)
        option_lines = [line for line in self.task.prompt.splitlines()
                        if len(line) >= 5 and line[:3].isdigit() and line[3:5] == ": "]
        self.assertEqual(len(option_lines), 1000)

    def test_small_tasks_keep_original_letter_codes(self):
        task = ClassificationTask(tuple(str(index) for index in range(20)), "Classify.")
        self.assertEqual(task.codes[0], "A")
        self.assertEqual(task.codes[-1], "T")
        self.assertIn("uppercase option code", task.prompt)

    def test_numeric_codes_parse_to_actual_labels_without_tokenizer_assumptions(self):
        for index in (0, 20, 167, 999):
            self.assertEqual(parse_label(f" {index:03d}\n", self.task), self.task.labels[index])
        for invalid in ("A", "16", "167 English foxhound", "0167", "1000", "167\n168"):
            with self.subTest(invalid=invalid), self.assertRaises(InvalidPredictionError):
                parse_label(invalid, self.task)

    def test_native_evidence_cannot_invent_the_other_999_class_probabilities(self):
        with self.assertRaisesRegex(InvalidPredictionError, "omit one or more class codes"):
            aggregate_code_logprobs(
                [{"token": "167", "logprob": math.log(0.9)}], self.task,
                logprob_key="logprob",
            )

    def test_loader_orders_by_index_even_if_dictionary_order_is_reversed(self):
        labels = {index: self.task.display_labels[index] for index in reversed(range(1000))}
        self.assertEqual(self._load(repr(labels)).labels, self.task.labels)

    def test_loader_rejects_missing_extra_or_noninteger_indices(self):
        labels = dict(enumerate(self.task.display_labels))
        cases = [
            {key: value for key, value in labels.items() if key != 999},
            {**labels, 1000: "extra class"},
            {str(key): value for key, value in labels.items()},
            {True if key == 1 else key: value for key, value in labels.items()},
        ]
        for malformed in cases:
            with self.subTest(keys=list(malformed)[:3]), self.assertRaises(ValueError):
                self._load(repr(malformed))
        # AST parsing also catches duplicate literal keys that dict() would hide.
        with self.assertRaisesRegex(ValueError, "without duplicates"):
            self._load(repr(labels)[:-1] + ", 999: 'overridden'}")

    def test_loader_rejects_empty_or_nonstring_label_values(self):
        for value in ("", None):
            labels = dict(enumerate(self.task.display_labels))
            labels[999] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self._load(repr(labels))

    def test_generic_tasks_still_reject_duplicate_class_identifiers(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            ClassificationTask(("crane", "crane"), "Classify.")

    def test_display_names_require_one_nonempty_name_per_identifier(self):
        for names in (("one",), ("one", ""), "names"):
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "display_labels"):
                ClassificationTask(("first", "second"), "Classify.", display_labels=names)

    def test_loader_never_executes_source_or_dictionary_values(self):
        for source in (
            "import os\nimagenet_labels_dict = {}",
            "imagenet_labels_dict = dict(enumerate([]))",
            "other_name = {}",
            "{0: __import__('os').getcwd()}",
            "{**{0: 'class'}}",
            "imagenet_labels_dict = {",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                self._load(source)

    def test_question_can_be_set_without_modifying_labels(self):
        question = "Classify the object using all of the labels."
        task = load_imagenet_task(LABELS_DIR / "imagenet_labels.py", question=question)
        self.assertEqual(task.question, question)
        self.assertEqual(task.labels, self.task.labels)

    def _load(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.txt"
            path.write_text(source, encoding="utf-8")
            return load_imagenet_task(path)


if __name__ == "__main__":
    unittest.main()
