import math
import unittest

from PIL import Image

from medshearletx.scoring import Scorer
from medshearletx.types import CapabilityError, ClassificationTask, InvalidPredictionError, Prediction


class StubBackend:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.calls = []

    def predict(self, image, task, *, require_logprobs, temperature=1.0):
        self.calls.append((require_logprobs, temperature))
        return next(self.predictions)


class ScoringTests(unittest.TestCase):
    def test_float32_probability_roundoff_is_accepted(self):
        from unittest.mock import Mock
        backend = Mock(spec=["predict"])
        task = ClassificationTask(("first", "second"), "Which class?")
        backend.predict.return_value = Prediction(label_logprobs={
            "first": -0.31708189845085144, "second": -1.302950382232666})
        result = Scorer(backend, task).evaluate(Image.new("RGB", (2, 2)))
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)

    def test_agreement_rejects_fixed_provider_seed_including_zero(self):
        from medshearletx.backends import create_backend
        task = ClassificationTask(("first", "second"), "Which class?")
        for provider in ("vllm", "gemini"):
            backend = create_backend({"backend": provider, "model": "test-model",
                                      "generation_options": {"seed": 0}})
            with self.assertRaisesRegex(ValueError, "independent draws"):
                Scorer(backend, task, mode="agreement")
            Scorer(backend, task, mode="probability")

    def setUp(self):
        self.image = Image.new("RGB", (4, 4))
        self.task = ClassificationTask(("normal", "abnormal"), "What does this image show?")

    def test_native_scores_condition_on_candidates_and_reuse_evidence(self):
        backend = StubBackend([Prediction(
            label_logprobs={"normal": math.log(0.3), "abnormal": math.log(0.1)}
        )])
        result = Scorer(backend, self.task).evaluate(self.image)
        self.assertAlmostEqual(result.score("normal"), 0.75)
        self.assertAlmostEqual(result.score("abnormal"), 0.25)
        self.assertAlmostEqual(result.diagnostics["class_mass"], 0.4)
        self.assertAlmostEqual(result.diagnostics["entropy"], -0.75 * math.log(0.75) - 0.25 * math.log(0.25))
        margins = result.with_mode("log_margin")
        self.assertAlmostEqual(margins.score("normal"), math.log(3))
        self.assertAlmostEqual(margins.score("abnormal"), -math.log(3))
        self.assertEqual(margins.predicted_label, "normal")
        self.assertEqual(set(margins.scores), set(self.task.labels))
        self.assertEqual(backend.calls, [(True, 1.0)])
        self.assertFalse(result.diagnostics["calibrated_correctness"])

    def test_tiny_native_probabilities_remain_numerically_stable(self):
        backend = StubBackend([Prediction(label_logprobs={"normal": -10000, "abnormal": -10001})])
        result = Scorer(backend, self.task, mode="log_margin").evaluate(self.image)
        self.assertAlmostEqual(result.score("normal"), 1)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1)
        self.assertTrue(math.isfinite(result.diagnostics["log_class_mass"]))

    def test_incomplete_or_invalid_native_evidence_fails_explicitly(self):
        cases = [
            {"normal": -0.1},
            {"normal": -0.1, "abnormal": -2, "unknown": -3},
            {"normal": -0.1, "abnormal": float("nan")},
            {"normal": -0.1, "abnormal": float("-inf")},
            {"normal": 0.1, "abnormal": -2},
            {"normal": "-0.1", "abnormal": -2},
            {"normal": 0, "abnormal": 0},
            {"normal": math.log(0.6), "abnormal": math.log(0.6)},
        ]
        for evidence in cases:
            with self.subTest(evidence=evidence):
                backend = StubBackend([Prediction(label_logprobs=evidence)])
                with self.assertRaises(InvalidPredictionError):
                    Scorer(backend, self.task).evaluate(self.image)

    def test_no_native_evidence_does_not_fall_back_to_self_reported_label(self):
        backend = StubBackend([Prediction(sampled_label="normal", metadata={"confidence": 0.99})])
        with self.assertRaises(CapabilityError):
            Scorer(backend, self.task).evaluate(self.image)

    def test_agreement_measures_samples_and_reports_sampling_error(self):
        backend = StubBackend([Prediction(sampled_label=label) for label in ["normal", "normal", "abnormal", "normal"]])
        result = Scorer(backend, self.task, mode="agreement", repeats=4, temperature=0.7).evaluate(self.image)
        self.assertEqual(result.sample_counts, {"normal": 3, "abnormal": 1})
        self.assertEqual(result.score("normal"), 0.75)
        self.assertEqual(result.requests, 4)
        self.assertEqual(backend.calls, [(False, 0.7)] * 4)
        self.assertAlmostEqual(result.diagnostics["sample_standard_error"]["normal"], math.sqrt(0.75 * 0.25 / 4))
        low, high = result.diagnostics["sample_wilson_95"]["normal"]
        self.assertLess(low, 0.75)
        self.assertGreater(high, 0.75)
        with self.assertRaises(CapabilityError):
            result.with_mode("probability")
        with self.assertRaises(CapabilityError):
            result.with_mode("log_margin")

    def test_invalid_agreement_sample_cannot_be_silently_excluded(self):
        for invalid_label in ("A", None, "probably normal", []):
            with self.subTest(invalid_label=invalid_label):
                backend = StubBackend([Prediction(sampled_label="normal"), Prediction(sampled_label=invalid_label)])
                with self.assertRaisesRegex(InvalidPredictionError, "sample 2/3"):
                    Scorer(backend, self.task, mode="agreement", repeats=3).evaluate(self.image)

    def test_agreement_requires_stochastic_sampling(self):
        with self.assertRaisesRegex(ValueError, "temperature > 0"):
            Scorer(StubBackend([]), self.task, mode="agreement", temperature=0)

    def test_unanimous_small_sample_does_not_imply_certainty(self):
        backend = StubBackend([Prediction(sampled_label="normal") for _ in range(4)])
        result = Scorer(backend, self.task, mode="agreement", repeats=4).evaluate(self.image)
        self.assertEqual(result.diagnostics["sample_standard_error"]["normal"], 0)
        normal_low, normal_high = result.diagnostics["sample_wilson_95"]["normal"]
        abnormal_low, abnormal_high = result.diagnostics["sample_wilson_95"]["abnormal"]
        self.assertLess(normal_low, normal_high)
        self.assertGreater(abnormal_high, abnormal_low)
        self.assertAlmostEqual(normal_high, 1)
        self.assertAlmostEqual(abnormal_low, 0)

    def test_attempt_counter_accumulates_successful_evaluations(self):
        backend = StubBackend([Prediction(label_logprobs={"normal": -1, "abnormal": -2})] * 2)
        scorer = Scorer(backend, self.task)
        self.assertEqual(scorer.requests, 0)
        self.assertEqual(scorer.evaluate(self.image).requests, 1)
        self.assertEqual(scorer.evaluate(self.image).requests, 1)
        self.assertEqual(scorer.requests, 2)

    def test_attempt_counter_includes_failed_samples_and_preflight(self):
        backend = StubBackend([Prediction(sampled_label="normal"), Prediction(sampled_label="invalid")])
        scorer = Scorer(backend, self.task, mode="agreement", repeats=4)
        with self.assertRaises(InvalidPredictionError):
            scorer.evaluate(self.image)
        self.assertEqual(scorer.requests, 2)

        class UnsupportedBackend:
            def predict(self, image, task, *, require_logprobs, temperature=1.0):
                raise CapabilityError("provider cannot return log probabilities")

        scorer = Scorer(UnsupportedBackend(), self.task)
        with self.assertRaises(CapabilityError):
            scorer.evaluate(self.image)
        self.assertEqual(scorer.requests, 1)

    def test_unknown_target_rejected(self):
        backend = StubBackend([Prediction(label_logprobs={"normal": -1, "abnormal": -2})])
        result = Scorer(backend, self.task).evaluate(self.image)
        with self.assertRaises(ValueError):
            result.score("A")


class ClassificationTaskTests(unittest.TestCase):
    def test_codes_and_prompt_keep_fixed_label_order(self):
        task = ClassificationTask(("healthy", "disease present", "indeterminate"), "Classify the image.")
        self.assertEqual(task.codes, ("A", "B", "C"))
        self.assertIn("B: disease present", task.prompt)
        self.assertIn("no explanation", task.prompt)

    def test_invalid_tasks_rejected(self):
        for labels, question in [(("one",), "question"), (("same", "same"), "question"), (("a", " "), "question"), (("a", "b"), " "), (tuple(map(str, range(21))), "question")]:
            with self.subTest(labels=labels, question=question):
                with self.assertRaises(ValueError):
                    ClassificationTask(labels, question)


if __name__ == "__main__":
    unittest.main()
