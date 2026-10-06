import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
import unittest
from unittest.mock import patch
import urllib.error

from PIL import Image

from medshearletx.audit import RecordedBackend, RequestBudgetExceeded
from medshearletx.backends.base import BackendRequestError, http_json
from medshearletx.backends.retry import RetryingBackend
from medshearletx.types import ClassificationTask, InvalidPredictionError, Prediction


class Stub:
    thread_safe = True

    def __init__(self, failures=1, error=None):
        self.failures = failures
        self.error = error or BackendRequestError("timeout", retryable=True)
        self.calls = 0
        self.lock = Lock()

    def predict(self, image, task, **kwargs):
        with self.lock:
            self.calls += 1
            failed = self.calls <= self.failures
        if failed:
            raise self.error
        return Prediction(sampled_label=task.labels[0])


class RetryTests(unittest.TestCase):
    image = Image.new("RGB", (16, 16))
    task = ClassificationTask(("dog", "cat"), "What animal?")

    def call(self, backend):
        return backend.predict(self.image, self.task, require_logprobs=False)

    def test_recovery_audits_failed_and_successful_physical_attempts(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "requests.jsonl"
            recorded = RecordedBackend(Stub(), path, max_requests=4)
            waits = []
            backend = RetryingBackend(recorded, retry_budget=2, sleep=waits.append)
            self.assertEqual(self.call(backend).sampled_label, "dog")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["error", "success"])
            self.assertEqual(recorded.requests, 2)
            self.assertEqual(backend.retry_count, 1)
            self.assertEqual(waits, [0.5])

    def test_invalid_labels_and_http_400_are_not_retried(self):
        for error in (InvalidPredictionError("invalid label"), BackendRequestError("HTTP400", status_code=400)):
            stub = Stub(failures=100, error=error)
            with self.assertRaises(type(error)):
                self.call(RetryingBackend(stub, retry_budget=32, sleep=lambda _: None))
            self.assertEqual(stub.calls, 1)

    def test_concurrent_retry_allowance_and_attempt_limit(self):
        stub = Stub(failures=100)
        backend = RetryingBackend(stub, retry_budget=3, sleep=lambda _: None)
        def attempt(_):
            with self.assertRaises(BackendRequestError):
                self.call(backend)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(attempt, range(8)))
        self.assertEqual(backend.retry_count, 3)
        self.assertEqual(stub.calls, 8 + 3)
        stub = Stub(failures=100)
        with self.assertRaises(BackendRequestError):
            self.call(RetryingBackend(stub, retry_budget=32, max_attempts=3, sleep=lambda _: None))
        self.assertEqual(stub.calls, 3)

    def test_physical_request_cap_still_prevents_retries(self):
        with TemporaryDirectory() as folder:
            stub = Stub(failures=100)
            recorded = RecordedBackend(stub, Path(folder) / "log.jsonl", max_requests=2)
            with self.assertRaises(RequestBudgetExceeded):
                self.call(RetryingBackend(recorded, retry_budget=32, sleep=lambda _: None))
            self.assertEqual(stub.calls, 2)

    def test_transport_errors_have_explicit_retryability(self):
        for code in (400, 401, 403, 429, 500, 502, 503, 504):
            error = urllib.error.HTTPError("https://secret.invalid", code, "private", {}, None)
            with patch("urllib.request.build_opener") as opener:
                opener.return_value.open.side_effect = error
                with self.assertRaises(BackendRequestError) as caught:
                    http_json("https://provider.invalid", headers={}, payload={}, timeout=1)
            self.assertEqual(caught.exception.retryable, code in {429, 500, 502, 503, 504})
            self.assertEqual(caught.exception.status_code, code)
            self.assertNotIn("secret", str(caught.exception))
        with patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError("private URL")
            with self.assertRaises(BackendRequestError) as caught:
                http_json("https://provider.invalid", headers={}, payload={}, timeout=1)
        self.assertTrue(caught.exception.retryable)
