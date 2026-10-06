"""Vertex Express fixtures do not contact a provider or read real credentials."""

import copy
import math
import os
import unittest
from unittest.mock import patch

from PIL import Image

from medshearletx.backends import BackendRequestError, VertexBackend, create_backend
from medshearletx.types import CapabilityError, ClassificationTask, InvalidPredictionError


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, *, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return copy.deepcopy(self.response)


def response(text="167", *, native=False):
    candidate = {
        "content": {"parts": [{"thought": True, "text": "hidden"}, {"text": text}]},
        "finishReason": "STOP",
    }
    if native:
        candidate["logprobsResult"] = {
            "chosenCandidates": [{"token": text, "logProbability": math.log(0.6)}],
            "topCandidates": [{"candidates": [
                {"token": "A", "logProbability": math.log(0.6)},
                {"token": "B", "logProbability": math.log(0.3)},
            ]}],
        }
    return {"candidates": [candidate], "usageMetadata": {"candidatesTokenCount": 3}}


class VertexTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (16, 16), "gray")
        self.task = ClassificationTask(tuple(f"label-{index}" for index in range(1000)), "Classify image.")

    def test_express_path_header_and_full_imagenet_sampling(self):
        transport = FakeTransport(response())
        backend = create_backend({"backend": "vertex", "model": "gemini-3.5-flash-lite"}, transport=transport)
        self.assertIsInstance(backend, VertexBackend)
        self.assertEqual(transport.calls, [])
        with patch.dict(os.environ, {"VERTEX_API_KEY": "offline-secret"}):
            result = backend.predict(self.image, self.task, require_logprobs=False, temperature=0)
        request = transport.calls[0]
        self.assertEqual(
            request["url"],
            "https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-3.5-flash-lite:generateContent",
        )
        self.assertEqual(request["headers"], {"x-goog-api-key": "offline-secret"})
        self.assertNotIn("offline-secret", request["url"])
        config = request["payload"]["generationConfig"]
        self.assertNotIn("temperature", config)
        self.assertNotIn("responseLogprobs", config)
        self.assertEqual(config["maxOutputTokens"], 128)
        self.assertIn("999: label-999", request["payload"]["contents"][0]["parts"][0]["text"])
        self.assertEqual(result.sampled_label, "label-167")
        self.assertIsNone(result.label_logprobs)
        self.assertEqual(result.metadata["backend"], "vertex")
        self.assertEqual(result.metadata["requested_temperature"], 0)
        self.assertEqual(result.metadata["effective_temperature"], 1)
        self.assertEqual(result.metadata["temperature_control"], "provider_default_ignored")
        self.assertTrue(result.metadata["native_logprobs_deprecated"])
        self.assertNotIn("offline-secret", repr(result))

    def test_other_models_preserve_requested_temperature_and_options(self):
        transport = FakeTransport(response())
        options = {"thinkingConfig": {"thinkingLevel": "MINIMAL"}}
        backend = VertexBackend(
            model="models/gemini-2.5-flash", api_key_env=None, max_output_tokens=32,
            generation_options=options, transport=transport,
        )
        options["thinkingConfig"]["thinkingLevel"] = "HIGH"
        result = backend.predict(self.image, self.task, require_logprobs=False, temperature=0.7)
        config = transport.calls[0]["payload"]["generationConfig"]
        self.assertEqual(config["temperature"], 0.7)
        self.assertEqual(config["maxOutputTokens"], 32)
        self.assertEqual(config["thinkingConfig"]["thinkingLevel"], "MINIMAL")
        self.assertEqual(result.metadata["temperature_control"], "requested")
        self.assertEqual(result.metadata["effective_temperature"], 0.7)
        self.assertFalse(result.metadata["native_logprobs_deprecated"])

    def test_native_capability_is_explicit_and_all_class_evidence_stays_required(self):
        transport = FakeTransport(response("A", native=True))
        task = ClassificationTask(("normal", "abnormal"), "Classify image.")
        backend = VertexBackend(model="gemini-3.5-flash-lite", api_key_env=None, transport=transport)
        with self.assertRaises(CapabilityError):
            backend.predict(self.image, task, require_logprobs=True)
        self.assertEqual(transport.calls, [])
        backend = VertexBackend(
            model="gemini-3.5-flash-lite", api_key_env=None, supports_logprobs=True, transport=transport,
        )
        result = backend.predict(self.image, task, require_logprobs=True)
        self.assertAlmostEqual(result.label_logprobs["normal"], math.log(0.6))
        self.assertTrue(transport.calls[0]["payload"]["generationConfig"]["responseLogprobs"])
        self.assertEqual(transport.calls[0]["payload"]["generationConfig"]["logprobs"], 20)
        transport.response = response("167", native=True)
        with self.assertRaises(InvalidPredictionError):
            backend.predict(self.image, self.task, require_logprobs=True)

    def test_provider_http_errors_remain_sanitized(self):
        def failing_transport(*args, **kwargs):
            raise BackendRequestError("Provider returned HTTP 400; model capability unsupported.")

        backend = VertexBackend(model="gemini-3.5-flash-lite", api_key_env=None, transport=failing_transport)
        with self.assertRaisesRegex(BackendRequestError, "HTTP 400"):
            backend.predict(self.image, self.task, require_logprobs=False)

    def test_custom_base_url_is_encoded_and_credentials_are_deferred(self):
        transport = FakeTransport(response())
        backend = VertexBackend(model="custom/model", base_url="https://example.invalid/v1/", transport=transport)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(BackendRequestError):
            backend.predict(self.image, self.task, require_logprobs=False)
        self.assertEqual(transport.calls, [])
        with patch.dict(os.environ, {"VERTEX_API_KEY": "offline-secret"}):
            backend.predict(self.image, self.task, require_logprobs=False)
        self.assertEqual(
            transport.calls[0]["url"],
            "https://example.invalid/v1/publishers/google/models/custom%2Fmodel:generateContent",
        )


if __name__ == "__main__":
    unittest.main()
