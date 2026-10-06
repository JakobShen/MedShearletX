import math
from threading import Lock
import time
import unittest

from PIL import Image

from medshearletx.scoring import Scorer
from medshearletx.types import (
    CapabilityError, ClassificationTask, InvalidPredictionError, MissingTargetScoreError, Prediction,
)


class StubBackend:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.calls = []

    def predict(self, image, task, *, require_logprobs, temperature=1.0):
        self.calls.append((require_logprobs, temperature))
        return next(self.predictions)


class ConcurrentBackend:
    """Atomic offline fake with varying delays and visible in-flight calls."""

    def __init__(self, *, invalid_at=None, fail_at=None, thread_safe=True):
        self.thread_safe = thread_safe
        self.invalid_at = invalid_at
        self.fail_at = fail_at
        self.lock = Lock()
        self.calls = []
        self.active = 0
        self.maximum_active = 0

    def predict(self, image, task, *, require_logprobs, temperature=1.0):
        with self.lock:
            index = len(self.calls)
            self.calls.append((require_logprobs, temperature))
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            time.sleep(0.005 * (4 - index % 4))
            if index == self.fail_at:
                raise CapabilityError("offline provider failure")
            return Prediction(
                sampled_label="invalid" if index == self.invalid_at else task.labels[index % 2],
                metadata={"sample_index": index, "effective_temperature": 1.0},
            )
        finally:
            with self.lock:
                self.active -= 1


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

    def test_fixed_target_scores_keep_partial_1000_class_native_probability(self):
        task = ClassificationTask(tuple(f"class_{index}" for index in range(1000)), "Which class?")
        target, observed = task.labels[:2]
        backend = StubBackend([
            Prediction(label_logprobs={target: math.log(probability), observed: math.log(0.05)},
                       sampled_label=observed, metadata={"logprob_scope": "reported"})
            for probability in (0.9, 0.8)
        ])
        scorer = Scorer(backend, task, mode="target_probability", target=target)
        reference = scorer.evaluate(self.image)
        candidate = scorer.evaluate(self.image)
        self.assertAlmostEqual(reference.score(target), 0.9)
        self.assertAlmostEqual(candidate.score(target), 0.8)
        self.assertAlmostEqual((reference.score(target) - candidate.score(target)) ** 2, 0.01)
        self.assertEqual(reference.observed_label, observed)
        self.assertEqual(reference.predicted_label, observed)
        self.assertEqual(candidate.predicted_label, observed)
        self.assertEqual(set(reference.probabilities), {target})
        self.assertEqual(set(reference.log_probabilities), {target})
        self.assertAlmostEqual(reference.log_probabilities[target], math.log(0.9))
        self.assertEqual(reference.scores, reference.probabilities)
        self.assertNotAlmostEqual(sum(reference.probabilities.values()), 1.0)
        self.assertAlmostEqual(reference.diagnostics["reported_class_mass"], 0.95)
        self.assertEqual(reference.diagnostics["scope"], "fixed_target")
        self.assertEqual(reference.diagnostics["normalization"], "raw_unnormalized")
        self.assertFalse(reference.diagnostics["normalized"])
        self.assertEqual(reference.diagnostics["probability_event"], "output_token_event")
        self.assertEqual(reference.diagnostics["target"], target)
        self.assertAlmostEqual(reference.diagnostics["raw_logprob"], math.log(0.9))
        self.assertFalse(reference.diagnostics["calibrated_correctness"])
        with self.assertRaises(ValueError):
            reference.score(observed)
        self.assertEqual(backend.calls, [(True, 1.0)] * 2)
        self.assertEqual(scorer.requests, 2)

    def test_fixed_target_requires_an_explicit_known_target(self):
        for target in (None, "unknown", 1, True, []):
            with self.subTest(target=target):
                backend = StubBackend([])
                with self.assertRaisesRegex(ValueError, "fixed target from task.labels"):
                    Scorer(backend, self.task, mode="target_probability", target=target)
                self.assertEqual(backend.calls, [])
        for mode in ("probability", "log_margin", "agreement"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "target is only supported"):
                Scorer(StubBackend([]), self.task, mode=mode, target="normal")

    def test_fixed_target_preserves_provider_output_event_semantics(self):
        backend = StubBackend([Prediction(
            label_logprobs={"normal": math.log(0.8)}, sampled_label="normal",
            metadata={"probability_event": "output_digit_sequence_event"},
        )])
        result = Scorer(backend, self.task, mode="target_probability", target="normal").evaluate(self.image)
        self.assertEqual(result.diagnostics["probability_event"], "output_digit_sequence_event")
        self.assertAlmostEqual(result.score("normal"), 0.8)

    def test_fixed_target_absent_from_reported_evidence_fails_without_zero_fill(self):
        for evidence in ({}, {"abnormal": math.log(0.9)}):
            with self.subTest(evidence=evidence):
                backend = StubBackend([Prediction(label_logprobs=evidence, sampled_label="abnormal")])
                scorer = Scorer(backend, self.task, mode="target_probability", target="normal")
                with self.assertRaisesRegex(MissingTargetScoreError, "does not report the fixed target.*normal"):
                    scorer.evaluate(self.image)
                self.assertEqual(scorer.requests, 1)
        backend = StubBackend([Prediction(sampled_label="normal", metadata={"confidence": 0.99})])
        with self.assertRaisesRegex(CapabilityError, "native label log probabilities"):
            Scorer(backend, self.task, mode="target_probability", target="normal").evaluate(self.image)

    def test_fixed_target_validates_all_reported_native_evidence(self):
        cases = [
            {"normal": math.log(0.8), "unknown": math.log(0.1)},
            {"normal": math.log(0.8), "abnormal": float("nan")},
            {"normal": math.log(0.8), "abnormal": float("-inf")},
            {"normal": math.log(0.8), "abnormal": 0.1},
            {"normal": math.log(0.8), "abnormal": "-2"},
            {"normal": math.log(0.8), "abnormal": True},
            {"normal": math.log(0.8), "abnormal": math.log(0.3)},
            {"normal": math.log(0.6), "abnormal": math.log(0.6)},
        ]
        for evidence in cases:
            with self.subTest(evidence=evidence), self.assertRaises(InvalidPredictionError):
                backend = StubBackend([Prediction(label_logprobs=evidence, sampled_label="normal")])
                Scorer(backend, self.task, mode="target_probability", target="normal").evaluate(self.image)

    def test_fixed_target_does_not_invent_an_observed_label(self):
        backend = StubBackend([Prediction(label_logprobs={"normal": math.log(0.9)})])
        result = Scorer(backend, self.task, mode="target_probability", target="normal").evaluate(self.image)
        self.assertIsNone(result.observed_label)
        self.assertIsNone(result.predicted_label)
        for invalid in ("unknown", "A", []):
            for evidence in ({"normal": math.log(0.9)}, {"abnormal": math.log(0.9)}):
                with self.subTest(invalid=invalid, evidence=evidence), self.assertRaisesRegex(
                    InvalidPredictionError, "sampled label"
                ):
                    backend = StubBackend([Prediction(label_logprobs=evidence, sampled_label=invalid)])
                    Scorer(backend, self.task, mode="target_probability", target="normal").evaluate(self.image)

    def test_fixed_target_evidence_cannot_change_to_other_score_scopes(self):
        backend = StubBackend([Prediction(label_logprobs={"normal": math.log(0.9)},
                                          sampled_label="normal")])
        result = Scorer(backend, self.task, mode="target_probability", target="normal").evaluate(self.image)
        self.assertEqual(result.with_mode("target_probability"), result)
        for mode in ("probability", "log_margin", "agreement"):
            with self.subTest(mode=mode), self.assertRaises(CapabilityError):
                result.with_mode(mode)
        backend = StubBackend([Prediction(label_logprobs={"normal": math.log(0.3), "abnormal": math.log(0.1)})])
        complete = Scorer(backend, self.task).evaluate(self.image)
        with self.assertRaises(CapabilityError):
            complete.with_mode("target_probability")
        backend = StubBackend([Prediction(sampled_label="normal")])
        agreement = Scorer(backend, self.task, mode="agreement", repeats=1).evaluate(self.image)
        with self.assertRaises(CapabilityError):
            agreement.with_mode("target_probability")

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

    def test_parallel_agreement_counts_samples_and_provider_temperature(self):
        backend = ConcurrentBackend()
        scorer = Scorer(backend, self.task, mode="agreement", repeats=20, temperature=0.7, workers=4)
        result = scorer.evaluate(self.image)
        self.assertEqual(result.sample_counts, {"normal": 10, "abnormal": 10})
        self.assertEqual(result.requests, 20)
        self.assertEqual(scorer.requests, 20)
        self.assertGreater(backend.maximum_active, 1)
        self.assertLessEqual(backend.maximum_active, 4)
        self.assertEqual(backend.active, 0)
        self.assertEqual(backend.calls, [(False, 0.7)] * 20)
        self.assertEqual(len(result.diagnostics["backend"]), 20)
        self.assertEqual({item["sample_index"] for item in result.diagnostics["backend"]}, set(range(20)))
        self.assertEqual(result.diagnostics["requested_temperature"], 0.7)
        self.assertEqual(result.diagnostics["effective_temperature"], 1.0)
        self.assertEqual(result.diagnostics["requested_workers"], 4)
        self.assertEqual(result.diagnostics["workers"], 4)

    def test_parallel_failure_counts_all_completed_attempts(self):
        for failure in ("invalid_at", "fail_at"):
            with self.subTest(failure=failure):
                backend = ConcurrentBackend(**{failure: 1})
                scorer = Scorer(backend, self.task, mode="agreement", repeats=12, workers=4)
                with self.assertRaises((InvalidPredictionError, CapabilityError)):
                    scorer.evaluate(self.image)
                self.assertEqual(scorer.requests, len(backend.calls))
                self.assertGreaterEqual(scorer.requests, 2)
                self.assertLessEqual(scorer.requests, 12)
                if failure == "invalid_at":
                    self.assertEqual(scorer.requests, 12)
                self.assertEqual(backend.active, 0)

    def test_workers_are_bounded_and_unsafe_backends_require_sequential_sampling(self):
        for workers in (0, -1, 33, 1.5, True, None):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "workers"):
                Scorer(StubBackend([]), self.task, workers=workers)
        backend = ConcurrentBackend(thread_safe=False)
        with self.assertRaisesRegex(ValueError, "workers=1"):
            Scorer(backend, self.task, mode="agreement", workers=2)
        self.assertEqual(backend.calls, [])
        Scorer(backend, self.task, mode="agreement", workers=1)

    def test_effective_workers_do_not_exceed_sample_count(self):
        result = Scorer(ConcurrentBackend(), self.task, mode="agreement", repeats=2, workers=8).evaluate(self.image)
        self.assertEqual(result.diagnostics["requested_workers"], 8)
        self.assertEqual(result.diagnostics["workers"], 2)

    def test_effective_temperature_uses_metadata_and_does_not_guess(self):
        for metadata, expected in (
            ([{"temperature": 0.6}] * 2, 0.6),
            ([{}, {}], None),
            ([{"effective_temperature": 1.0}, {"effective_temperature": 0.5}], None),
        ):
            with self.subTest(metadata=metadata):
                backend = StubBackend([Prediction(sampled_label="normal", metadata=item) for item in metadata])
                result = Scorer(backend, self.task, mode="agreement", repeats=2).evaluate(self.image)
                self.assertEqual(result.diagnostics["effective_temperature"], expected)

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
        for labels, question in [(("one",), "question"), (("same", "same"), "question"), (("a", " "), "question"), (("a", "b"), " "), (tuple(map(str, range(1001))), "question")]:
            with self.subTest(labels=labels, question=question):
                with self.assertRaises(ValueError):
                    ClassificationTask(labels, question)


if __name__ == "__main__":
    unittest.main()
