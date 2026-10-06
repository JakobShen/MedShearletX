"""Offline evidence durability, concurrency, quota, and credential isolation."""

import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from threading import Lock
import time
import unittest

from PIL import Image

from medshearletx.audit import RecordedBackend, RequestBudgetExceeded
from medshearletx.backends.gemini import GeminiBackend
from medshearletx.backends.base import png_base64
from medshearletx.types import ClassificationTask, Prediction


class AuditStubBackend:
    model = "offline-vision"
    fixed_sampling_seed = None
    thread_safe = True

    def __init__(self, prediction=None, error=None):
        self.prediction = prediction or Prediction(sampled_label="normal")
        self.error = error
        self.calls = 0
        self.lock = Lock()

    def predict(self, image, task, *, require_logprobs, temperature=1.0):
        with self.lock:
            self.calls += 1
        time.sleep(0.002)
        if self.error is not None:
            raise self.error
        return self.prediction


class RecordedBackendTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "predictions.jsonl"
        self.image = Image.new("RGB", (4, 4), "gray")
        self.task = ClassificationTask(("normal", "abnormal"), "Classify this image.")

    def test_success_records_exact_image_prompt_and_prediction_before_return(self):
        prediction = Prediction(
            sampled_label="normal", label_logprobs={"normal": -0.2, "abnormal": -2.0},
            metadata={"backend": "offline", "model": "offline-vision", "temperature": 1.0,
                      "usage": {"promptTokenCount": 31}},
        )
        stub = AuditStubBackend(prediction=prediction)
        backend = RecordedBackend(stub, self.path, max_requests=2)
        result = backend.predict(self.image, self.task, require_logprobs=True, temperature=0.7)
        self.assertIs(result, prediction)
        self.assertEqual(backend.requests, 1)
        entry = self._entries()[0]
        self.assertEqual(entry["status"], "success")
        self.assertEqual(entry["sequence_id"], 1)
        self.assertEqual(entry["run_id"], backend.run_id)
        self.assertEqual(entry["image_sha256"], sha256(base64.b64decode(png_base64(self.image))).hexdigest())
        self.assertEqual(entry["prompt_sha256"], sha256(self.task.prompt.encode("utf-8")).hexdigest())
        self.assertEqual(entry["prediction"]["sampled_label"], "normal")
        self.assertEqual(entry["prediction"]["label_logprobs"], prediction.label_logprobs)
        self.assertEqual(entry["prediction"]["metadata"], prediction.metadata)
        self.assertEqual(entry["temperature"], 0.7)
        self.assertTrue(entry["require_logprobs"])
        self.assertGreaterEqual(entry["elapsed_seconds"], 0)
        self.assertEqual(backend.model, stub.model)
        self.assertIsNone(backend.fixed_sampling_seed)
        self.assertTrue(backend.thread_safe)

    def test_error_is_recorded_once_without_secret_message_or_transport_data(self):
        stub = AuditStubBackend(error=RuntimeError("SECRET_API_KEY at https://secret-provider.test?key=SECRET"))
        backend = RecordedBackend(stub, self.path, max_requests=2)
        with self.assertRaises(RuntimeError):
            backend.predict(self.image, self.task, require_logprobs=False)
        entries = self._entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "error")
        self.assertEqual(entries[0]["error_type"], "RuntimeError")
        self.assertEqual(backend.requests, 1)
        text = self.path.read_text()
        self.assertNotIn("SECRET", text)
        self.assertNotIn("secret-provider", text)
        self.assertNotIn("error_message", entries[0])

    def test_metadata_allowlist_excludes_credentials_urls_and_headers(self):
        metadata = {
            "backend": "offline", "model": "offline-vision", "api_key": "SECRET_API_KEY",
            "base_url": "https://secret-provider.test", "headers": {"Authorization": "SECRET"},
            "config": {"api_key": "SECRET"}, "response_id": "https://secret-provider.test/response",
            "usage": {"promptTokenCount": 22, "url": "https://secret-provider.test", "key": "SECRET"},
            "generation_option_names": ["thinkingConfig", "https://secret-provider.test", {"key": "SECRET"}],
            "numeric_padding_normalized": True,
        }
        backend = RecordedBackend(AuditStubBackend(prediction=Prediction(sampled_label="normal", metadata=metadata)),
                                  self.path, max_requests=1)
        backend.predict(self.image, self.task, require_logprobs=False)
        saved = self._entries()[0]["prediction"]["metadata"]
        self.assertEqual(saved, {"backend": "offline", "model": "offline-vision",
                                 "usage": {"promptTokenCount": 22},
                                 "generation_option_names": ["thinkingConfig"],
                                 "numeric_padding_normalized": True})
        self.assertNotIn("SECRET", self.path.read_text())
        self.assertNotIn("https://", self.path.read_text())

    def test_reported_native_evidence_retains_partial_coverage_without_filling_labels(self):
        task = ClassificationTask(tuple(f"label-{index}" for index in range(1000)), "Classify image.")
        metadata = {"logprob_scope": "reported", "returned_class_count": 1,
                    "class_coverage_complete": False, "headers": {"Authorization": "SECRET"},
                    "unapproved_coverage_detail": "SECRET"}
        prediction = Prediction(sampled_label="label-160", label_logprobs={"label-160": -0.2},
                                metadata=metadata)
        backend = RecordedBackend(AuditStubBackend(prediction=prediction), self.path, max_requests=1)
        backend.predict(self.image, task, require_logprobs=True)
        saved = self._entries()[0]["prediction"]
        self.assertEqual(saved["label_logprobs"], {"label-160": -0.2})
        self.assertNotIn("label-161", saved["label_logprobs"])
        self.assertEqual(saved["metadata"], {"logprob_scope": "reported", "returned_class_count": 1,
                                             "class_coverage_complete": False})
        self.assertNotIn("SECRET", self.path.read_text())

    def test_partial_native_metadata_fields_are_type_checked_before_recording(self):
        invalid = [
            {"logprob_scope": "SECRET", "returned_class_count": True, "class_coverage_complete": "false"},
            {"logprob_scope": {"api_key": "SECRET"}, "returned_class_count": -1, "class_coverage_complete": None},
            {"logprob_scope": ["reported"], "returned_class_count": 1001, "class_coverage_complete": 0},
            {"logprob_scope": None, "returned_class_count": 1.5, "class_coverage_complete": []},
        ]
        backend = RecordedBackend(AuditStubBackend(), self.path, max_requests=len(invalid))
        for metadata in invalid:
            backend.backend.prediction = Prediction(sampled_label="normal", metadata=metadata)
            backend.predict(self.image, self.task, require_logprobs=False)
        self.assertTrue(all(not row["prediction"]["metadata"] for row in self._entries()))
        self.assertNotIn("SECRET", self.path.read_text())

    def test_gemini_digit_event_scope_and_actual_prefix_are_preserved_in_request_log(self):
        task = ClassificationTask(tuple(f"label-{index}" for index in range(1000)), "Classify image.")
        chosen = [{"token": digit, "tokenId": 10 + int(digit), "logProbability": -0.1}
                  for digit in "160"]
        top = [{"candidates": [entry]} for entry in chosen]
        top[-1]["candidates"] = [chosen[-1], {"token": "1", "tokenId": 11, "logProbability": -3.0}]
        response = {"candidates": [{"content": {"parts": [{"text": "160"}]}, "finishReason": "STOP",
                                     "logprobsResult": {"chosenCandidates": chosen, "topCandidates": top}}]}
        gemini = GeminiBackend(model="offline", api_key_env=None, supports_logprobs=True,
                               logprob_scope="reported", transport=lambda *args, **kwargs: response)
        backend = RecordedBackend(gemini, self.path, max_requests=1)
        backend.predict(self.image, task, require_logprobs=True)
        saved = self._entries()[0]["prediction"]
        metadata = saved["metadata"]
        self.assertEqual(metadata["probability_event"], "output_digit_sequence_event")
        self.assertEqual(metadata["verbalizer_policy"], "reported_digit_sequence_codes")
        self.assertEqual(metadata["evaluated_prefix"], "16")
        self.assertEqual(metadata["output_token_count"], 3)
        self.assertEqual(metadata["returned_class_count"], 2)
        self.assertFalse(metadata["class_coverage_complete"])
        self.assertAlmostEqual(saved["label_logprobs"]["label-160"], -0.3)
        self.assertAlmostEqual(saved["label_logprobs"]["label-161"], -3.2)

    def test_native_event_metadata_filters_invalid_event_count_and_prefix(self):
        invalid = [
            {"probability_event": "SECRET", "output_token_count": True, "evaluated_prefix": "secret"},
            {"probability_event": [], "output_token_count": 0, "evaluated_prefix": "１６"},
            {"probability_event": None, "output_token_count": 1001, "evaluated_prefix": "0" * 1000},
            {"probability_event": "sequence", "output_token_count": 1.5, "evaluated_prefix": ["16"]},
        ]
        backend = RecordedBackend(AuditStubBackend(), self.path, max_requests=len(invalid) + 1)
        for metadata in invalid:
            backend.backend.prediction = Prediction(sampled_label="normal", metadata=metadata)
            backend.predict(self.image, self.task, require_logprobs=False)
        backend.backend.prediction = Prediction(sampled_label="normal", metadata={
            "probability_event": "output_token_event", "output_token_count": 1, "evaluated_prefix": ""})
        backend.predict(self.image, self.task, require_logprobs=True)
        rows = self._entries()
        self.assertTrue(all(not row["prediction"]["metadata"] for row in rows[:-1]))
        self.assertEqual(rows[-1]["prediction"]["metadata"], {
            "probability_event": "output_token_event", "output_token_count": 1, "evaluated_prefix": ""})
        self.assertNotIn("SECRET", self.path.read_text())

    def test_concurrent_quota_reservations_never_exceed_hard_limit(self):
        stub = AuditStubBackend()
        backend = RecordedBackend(stub, self.path, max_requests=7)

        def attempt(_):
            try:
                backend.predict(self.image, self.task, require_logprobs=False)
                return True
            except RequestBudgetExceeded:
                return False

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(attempt, range(30)))
        self.assertEqual(sum(results), 7)
        self.assertEqual(stub.calls, 7)
        self.assertEqual(backend.requests, 7)
        entries = self._entries()
        self.assertEqual(len(entries), 7)
        self.assertEqual({entry["sequence_id"] for entry in entries}, set(range(1, 8)))
        self.assertTrue(all(entry["status"] == "success" for entry in entries))

    def test_failed_attempts_consume_quota_and_following_denials_do_not_append(self):
        stub = AuditStubBackend(error=ValueError("offline failure"))
        backend = RecordedBackend(stub, self.path, max_requests=1)
        with self.assertRaises(ValueError):
            backend.predict(self.image, self.task, require_logprobs=False)
        before = self.path.read_bytes()
        with self.assertRaises(RequestBudgetExceeded):
            backend.predict(self.image, self.task, require_logprobs=False)
        self.assertEqual(stub.calls, 1)
        self.assertEqual(backend.requests, 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_existing_evidence_is_preserved_and_runs_have_distinct_ids(self):
        first = RecordedBackend(AuditStubBackend(), self.path, max_requests=1)
        first.predict(self.image, self.task, require_logprobs=False)
        before = self.path.read_bytes()
        second = RecordedBackend(AuditStubBackend(), self.path, max_requests=1)
        second.predict(self.image, self.task, require_logprobs=False)
        self.assertTrue(self.path.read_bytes().startswith(before))
        entries = self._entries()
        self.assertEqual(len(entries), 2)
        self.assertNotEqual(entries[0]["run_id"], entries[1]["run_id"])

    def test_nonpositive_or_noninteger_quota_is_rejected_without_calls(self):
        stub = AuditStubBackend()
        for value in (0, -1, True, 1.2, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                RecordedBackend(stub, self.path, max_requests=value)
        self.assertEqual(stub.calls, 0)
        self.assertFalse(self.path.exists())

    def _entries(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
