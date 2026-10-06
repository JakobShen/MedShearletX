"""Offline provider fixtures exercise the native evidence boundary."""

import base64
import copy
import io
import math
import os
import unittest
from unittest.mock import patch

from PIL import Image

from medshearletx.backends import (
    BackendRequestError, GeminiBackend, MockBackend, OpenAICompatibleBackend,
    create_backend, register_backend,
)
from medshearletx.backends.base import parse_label
from medshearletx.types import Backend, CapabilityError, ClassificationTask, InvalidPredictionError


def openai_response():
    return {
        "id": "offline-completion", "usage": {"prompt_tokens": 22, "completion_tokens": 1},
        "choices": [{"message": {"content": " A", "refusal": None}, "finish_reason": "length",
                     "logprobs": {"content": [{"token": " A", "logprob": math.log(0.3), "bytes": [32, 65],
                                               "top_logprobs": [
                                                   {"token": " A", "logprob": math.log(0.3)},
                                                   {"token": "A", "logprob": math.log(0.2)},
                                                   {"token": "B", "logprob": math.log(0.4)},
                                                   {"token": "other", "logprob": math.log(0.1)},
                                               ]}], "refusal": None}}],
    }


def gemini_response():
    return {
        "responseId": "offline-gemini", "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 1},
        "candidates": [{"content": {"parts": [{"text": "B"}]}, "finishReason": "MAX_TOKENS",
                        "logprobsResult": {
                            "topCandidates": [{"candidates": [
                                {"token": "A", "tokenId": 65, "logProbability": math.log(0.6)},
                                {"token": "B", "tokenId": 66, "logProbability": math.log(0.3)},
                            ]}],
                            "chosenCandidates": [{"token": "B", "tokenId": 66, "logProbability": math.log(0.3)}],
                        }}],
    }


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, *, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return copy.deepcopy(self.response)


class BackendsTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (16, 16), "gray")
        self.task = ClassificationTask(("normal", "abnormal"), "Does this image contain an abnormality?")

    def test_openai_native_score_and_png_request(self):
        transport = FakeTransport(openai_response())
        backend = create_backend({"backend": "openai", "model": "test-vision", "api_key_env": None,
                                  "supports_logprobs": True}, transport=transport)
        self.assertIsInstance(backend, Backend)
        self.assertEqual(transport.calls, [])
        result = backend.predict(self.image, self.task, require_logprobs=True)
        self.assertEqual(result.sampled_label, "normal")
        self.assertAlmostEqual(result.label_logprobs["normal"], math.log(0.5))
        self.assertAlmostEqual(result.label_logprobs["abnormal"], math.log(0.4))
        self.assertTrue(result.metadata["class_mass_is_lower_bound"])
        request = transport.calls[0]
        self.assertEqual(request["url"], "https://api.openai.com/v1/chat/completions")
        self.assertTrue(request["payload"]["logprobs"])
        self.assertEqual(request["payload"]["top_logprobs"], 20)
        self.assertEqual(request["payload"]["max_tokens"], 1)
        self.assertNotIn("logit_bias", request["payload"])
        content = request["payload"]["messages"][0]["content"]
        self.assertEqual(content[0]["text"], self.task.prompt)
        encoded = content[1]["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as decoded:
            self.assertEqual(decoded.format, "PNG")
            self.assertEqual(decoded.size, self.image.size)

    def test_native_requires_explicit_capability_without_request(self):
        transport = FakeTransport(openai_response())
        backend = OpenAICompatibleBackend(model="any", api_key_env=None, transport=transport)
        with self.assertRaises(CapabilityError):
            backend.predict(self.image, self.task, require_logprobs=True)
        self.assertEqual(transport.calls, [])
        result = backend.predict(self.image, self.task, require_logprobs=False, temperature=0.8)
        self.assertIsNone(result.label_logprobs)
        self.assertEqual(result.sampled_label, "normal")
        self.assertNotIn("logprobs", transport.calls[0]["payload"])

    def test_numeric_padding_parser_is_opt_in_and_class_preserving(self):
        task = ClassificationTask(tuple(f"label-{index}" for index in range(1000)), "Classify image.")
        for text, index in (("0160", 160), ("160", 160), ("016", 16), ("16", 16),
                            ("001", 1), ("1", 1), ("000", 0), ("00", 0), (" 0160\n", 160)):
            with self.subTest(text=text):
                self.assertEqual(parse_label(text, task, allow_numeric_padding=True), f"label-{index}")
        for text in ("160", "016", "001", "000"):
            self.assertEqual(parse_label(text, task), f"label-{int(text)}")
        for text in ("0160", "16", "1"):
            with self.subTest(strict_text=text), self.assertRaises(InvalidPredictionError):
                parse_label(text, task)
        for text in ("1000", "01000", "-160", "+160", "160.0", "1e2", "class 160",
                     "160 Afghan", '"160"', "１６０", "١٦٠", "", None, 160):
            with self.subTest(invalid_text=text), self.assertRaises(InvalidPredictionError):
                parse_label(text, task, allow_numeric_padding=True)
        with self.assertRaises(InvalidPredictionError):
            parse_label("0", self.task, allow_numeric_padding=True)
        twenty_one = ClassificationTask(tuple(f"label-{index}" for index in range(21)), "Classify image.")
        self.assertEqual(parse_label("001", twenty_one, allow_numeric_padding=True), "label-1")
        with self.assertRaises(InvalidPredictionError):
            parse_label("21", twenty_one, allow_numeric_padding=True)

    def test_providers_normalize_only_sampled_numeric_codes_and_report_changes(self):
        task = ClassificationTask(tuple(f"label-{index}" for index in range(1000)), "Classify image.")
        for provider, fixture in ((OpenAICompatibleBackend, openai_response), (GeminiBackend, gemini_response)):
            for text, index, normalized in (("0160", 160, True), ("160", 160, False),
                                            ("016", 16, False), ("16", 16, True), ("001", 1, False)):
                with self.subTest(provider=provider.__name__, text=text):
                    response = fixture()
                    if provider is GeminiBackend:
                        response["candidates"][0]["content"]["parts"] = [{"thought": True, "text": "hidden"}, {"text": text}]
                    else:
                        response["choices"][0]["message"]["content"] = text
                    backend = provider(model="offline", api_key_env=None, supports_logprobs=True,
                                       transport=FakeTransport(response))
                    result = backend.predict(self.image, task, require_logprobs=False)
                    self.assertEqual(result.sampled_label, f"label-{index}")
                    self.assertIs(result.metadata["numeric_padding_normalized"], normalized)
                    self.assertIsNone(result.label_logprobs)
                    if normalized:
                        with self.assertRaises(InvalidPredictionError):
                            backend.predict(self.image, task, require_logprobs=True)

    def test_native_chosen_numeric_token_also_keeps_canonical_padding(self):
        task = ClassificationTask(tuple(f"label-{index}" for index in range(1000)), "Classify image.")
        for provider, fixture in ((OpenAICompatibleBackend, openai_response), (GeminiBackend, gemini_response)):
            with self.subTest(provider=provider.__name__):
                response = fixture()
                if provider is GeminiBackend:
                    candidate = response["candidates"][0]
                    candidate["content"]["parts"] = [{"text": "160"}]
                    candidate["logprobsResult"]["chosenCandidates"][0]["token"] = "0160"
                else:
                    choice = response["choices"][0]
                    choice["message"]["content"] = "160"
                    choice["logprobs"]["content"][0]["token"] = "0160"
                backend = provider(model="offline", api_key_env=None, supports_logprobs=True,
                                   transport=FakeTransport(response))
                with self.assertRaisesRegex(InvalidPredictionError, "configured class code"):
                    backend.predict(self.image, task, require_logprobs=True)

    def test_missing_class_and_sentinel_fail_without_floor(self):
        for top_entries in [
            [{"token": " A", "logprob": math.log(0.3)}],
            [{"token": " A", "logprob": math.log(0.3)}, {"token": "B", "logprob": -9999.0}],
            [{"token": " A", "logprob": math.log(0.3)}, {"token": "B", "logprob": float("nan")}],
        ]:
            with self.subTest(top_entries=top_entries):
                response = openai_response()
                response["choices"][0]["logprobs"]["content"][0]["top_logprobs"] = top_entries
                backend = OpenAICompatibleBackend(model="any", api_key_env=None, supports_logprobs=True,
                                                  transport=FakeTransport(response))
                with self.assertRaises(InvalidPredictionError):
                    backend.predict(self.image, self.task, require_logprobs=True)

    def test_refusal_multitoken_missing_logprobs_and_content_mismatch_fail(self):
        responses = []
        refused = openai_response()
        refused["choices"][0]["message"]["refusal"] = "refused"
        responses.append(refused)
        multitoken = openai_response()
        multitoken["choices"][0]["logprobs"]["content"].append({"token": "\n", "logprob": -0.1})
        responses.append(multitoken)
        no_logprobs = openai_response()
        no_logprobs["choices"][0]["logprobs"] = None
        responses.append(no_logprobs)
        mismatch = openai_response()
        mismatch["choices"][0]["message"]["content"] = "B"
        responses.append(mismatch)
        for response in responses:
            with self.subTest(response=response):
                backend = OpenAICompatibleBackend(model="any", api_key_env=None, supports_logprobs=True,
                                                  transport=FakeTransport(response))
                with self.assertRaises(InvalidPredictionError):
                    backend.predict(self.image, self.task, require_logprobs=True)

    def test_custom_budget_and_reasoning_configuration(self):
        transport = FakeTransport(openai_response())
        options = {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
        backend = OpenAICompatibleBackend(model="any", api_key_env=None, max_tokens=8,
                                          token_budget_field="max_completion_tokens", generation_options=options,
                                          transport=transport)
        options["chat_template_kwargs"]["enable_thinking"] = True
        backend.predict(self.image, self.task, require_logprobs=False)
        payload = transport.calls[0]["payload"]
        self.assertEqual(payload["max_completion_tokens"], 8)
        self.assertNotIn("max_tokens", payload)
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        for option in ["messages", "model", "temperature", "logprobs", "logit_bias", "response_format", "top_p",
                       "guided_regex", "bad_words", "structured_outputs", "logits_processors"]:
            with self.subTest(option=option), self.assertRaises(ValueError):
                OpenAICompatibleBackend(model="any", generation_options={option: 1})

    def test_vllm_defaults_and_explicit_class_tokens(self):
        response = openai_response()
        token = response["choices"][0]["logprobs"]["content"][0]
        token.update(token="A", token_id=65, logprob=math.log(0.6))
        token["top_logprobs"] = [
            {"token": "A", "token_id": 65, "logprob": math.log(0.6)},
            {"token": "B", "token_id": 66, "logprob": math.log(0.3)},
        ]
        transport = FakeTransport(response)
        backend = create_backend({"backend": "vllm", "model": "local-model", "supports_logprobs": True,
                                  "class_token_ids": {"normal": 65, "abnormal": 66}}, transport=transport)
        result = backend.predict(self.image, self.task, require_logprobs=True)
        self.assertEqual(transport.calls[0]["headers"], {})
        self.assertEqual(transport.calls[0]["payload"]["logprob_token_ids"], [65, 66])
        self.assertEqual(result.metadata["verbalizer_policy"], "explicit_token_ids")
        self.assertFalse(result.metadata["class_mass_is_lower_bound"])
        self.assertTrue(result.metadata["class_token_ids_verified"])
        self.assertAlmostEqual(result.label_logprobs["normal"], math.log(0.6))
        response["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["token_id"] = 65
        transport.response = response
        with self.assertRaises(InvalidPredictionError):
            backend.predict(self.image, self.task, require_logprobs=True)

    def test_vllm_sampled_whitespace_variant_outside_declared_ids_is_excluded(self):
        response = openai_response()
        token = response["choices"][0]["logprobs"]["content"][0]
        token.update(token=" A", token_id=999, logprob=math.log(0.1))
        token["top_logprobs"] = [
            {"token": "A", "token_id": 65, "logprob": math.log(0.6)},
            {"token": "B", "token_id": 66, "logprob": math.log(0.3)},
        ]
        backend = create_backend({"backend": "vllm", "model": "local-model", "supports_logprobs": True,
                                  "class_token_ids": {"normal": 65, "abnormal": 66}},
                                 transport=FakeTransport(response))
        result = backend.predict(self.image, self.task, require_logprobs=True)
        self.assertEqual(result.sampled_label, "normal")
        self.assertAlmostEqual(result.label_logprobs["normal"], math.log(0.6))

    def test_vllm_without_returned_ids_marks_verbalizers_unverified(self):
        response = openai_response()
        token = response["choices"][0]["logprobs"]["content"][0]
        token.update(token="A", logprob=math.log(0.6))
        token["top_logprobs"] = [
            {"token": "A", "logprob": math.log(0.6)},
            {"token": "B", "logprob": math.log(0.3)},
        ]
        response["choices"][0]["message"]["content"] = "A"
        backend = create_backend({"backend": "vllm", "model": "local-model", "supports_logprobs": True,
                                  "class_token_ids": {"normal": 65, "abnormal": 66}},
                                 transport=FakeTransport(response))
        result = backend.predict(self.image, self.task, require_logprobs=True)
        self.assertFalse(result.metadata["class_token_ids_verified"])
        self.assertTrue(result.metadata["class_mass_is_lower_bound"])

    def test_gemini_native_schema_and_header_auth(self):
        transport = FakeTransport(gemini_response())
        backend = create_backend({"backend": "gemini", "model": "models/vision-test",
                                  "api_key_env": "TEST_GEMINI_KEY", "supports_logprobs": True,
                                  "generation_options": {"thinkingConfig": {"thinkingBudget": 0}}},
                                 transport=transport)
        with patch.dict(os.environ, {"TEST_GEMINI_KEY": "offline-secret"}):
            result = backend.predict(self.image, self.task, require_logprobs=True)
        request = transport.calls[0]
        self.assertNotIn("offline-secret", request["url"])
        self.assertEqual(request["headers"], {"x-goog-api-key": "offline-secret"})
        config = request["payload"]["generationConfig"]
        self.assertTrue(config["responseLogprobs"])
        self.assertEqual(config["logprobs"], 20)
        self.assertEqual(result.sampled_label, "abnormal")
        self.assertAlmostEqual(result.label_logprobs["normal"], math.log(0.6))
        self.assertNotIn("offline-secret", repr(result))

    def test_gemini_chosen_token_outside_top_candidates_is_retained(self):
        response = gemini_response()
        response["candidates"][0]["logprobsResult"]["topCandidates"][0]["candidates"].pop()
        backend = GeminiBackend(model="any", api_key_env=None, supports_logprobs=True,
                                transport=FakeTransport(response))
        result = backend.predict(self.image, self.task, require_logprobs=True)
        self.assertAlmostEqual(result.label_logprobs["abnormal"], math.log(0.3))

    def test_gemini_missing_class_blocked_prompt_and_extra_steps_fail(self):
        missing = gemini_response()
        missing["candidates"][0]["logprobsResult"]["topCandidates"][0]["candidates"].pop(0)
        extra_steps = gemini_response()
        extra_steps["candidates"][0]["logprobsResult"]["chosenCandidates"].append(
            {"token": "\n", "tokenId": 10, "logProbability": -0.1})
        for response in [missing, extra_steps, {"promptFeedback": {"blockReason": "SAFETY"}}]:
            with self.subTest(response=response):
                backend = GeminiBackend(model="any", api_key_env=None, supports_logprobs=True,
                                        transport=FakeTransport(response))
                with self.assertRaises(InvalidPredictionError):
                    backend.predict(self.image, self.task, require_logprobs=True)

    def test_sampling_does_not_accept_model_written_confidence(self):
        response = openai_response()
        response["choices"][0]["message"]["content"] = "A, confidence=0.99"
        backend = OpenAICompatibleBackend(model="any", api_key_env=None, transport=FakeTransport(response))
        with self.assertRaises(InvalidPredictionError):
            backend.predict(self.image, self.task, require_logprobs=False)

    def test_fixed_provider_seed_is_exposed_without_blocking_native_scores(self):
        openai_transport = FakeTransport(openai_response())
        openai = OpenAICompatibleBackend(
            model="any", api_key_env=None, supports_logprobs=True,
            generation_options={"seed": 0}, transport=openai_transport,
        )
        self.assertEqual(openai.fixed_sampling_seed, 0)
        openai.predict(self.image, self.task, require_logprobs=True)
        self.assertEqual(openai_transport.calls[0]["payload"]["seed"], 0)
        gemini_transport = FakeTransport(gemini_response())
        gemini = GeminiBackend(
            model="any", api_key_env=None, generation_options={"seed": 123},
            transport=gemini_transport,
        )
        self.assertEqual(gemini.fixed_sampling_seed, 123)
        gemini.predict(self.image, self.task, require_logprobs=False)
        self.assertEqual(gemini_transport.calls[0]["payload"]["generationConfig"]["seed"], 123)
        self.assertIsNone(OpenAICompatibleBackend(model="any").fixed_sampling_seed)
        self.assertIsNone(GeminiBackend(model="any").fixed_sampling_seed)

    def test_credential_lookup_is_deferred_and_error_sanitized(self):
        transport = FakeTransport(openai_response())
        backend = OpenAICompatibleBackend(model="any", api_key_env="MISSING_OFFLINE_KEY", transport=transport)
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(BackendRequestError):
            backend.predict(self.image, self.task, require_logprobs=False)
        self.assertEqual(transport.calls, [])

        def failing_transport(*args, **kwargs):
            raise RuntimeError("https://provider?key=offline-secret raw-response=patient-data")

        backend = OpenAICompatibleBackend(model="any", api_key_env=None, transport=failing_transport)
        with self.assertRaises(BackendRequestError) as caught:
            backend.predict(self.image, self.task, require_logprobs=False)
        self.assertNotIn("offline-secret", str(caught.exception))
        self.assertNotIn("patient-data", str(caught.exception))

    def test_mock_native_scores_depend_on_spatial_content(self):
        dark = Image.new("L", (32, 32), 0)
        bright = dark.copy()
        bright.paste(255, (12, 12, 20, 20))
        backend = MockBackend(seed=4)
        left = backend.predict(dark, self.task, require_logprobs=True, temperature=0)
        right = backend.predict(bright, self.task, require_logprobs=True, temperature=0)
        self.assertLess(left.label_logprobs["normal"], right.label_logprobs["normal"])
        self.assertAlmostEqual(sum(math.exp(value) for value in right.label_logprobs.values()), 1.0)
        before = backend.predict(bright, self.task, require_logprobs=True, temperature=0.5)
        after = backend.predict(bright, self.task, require_logprobs=True, temperature=2.0)
        self.assertEqual(before.label_logprobs, after.label_logprobs)

    def test_mock_samples_reproducible_stream(self):
        first, second = MockBackend(seed=42), MockBackend(seed=42)
        sequence_a = [first.predict(self.image, self.task, require_logprobs=False).sampled_label for _ in range(30)]
        sequence_b = [second.predict(self.image, self.task, require_logprobs=False).sampled_label for _ in range(30)]
        self.assertEqual(sequence_a, sequence_b)
        self.assertEqual(set(sequence_a), set(self.task.labels))

    def test_config_validation_and_protocol_registration(self):
        for config in [
            {},
            {"backend": "missing"}, {"backend": "openai"},
            {"backend": "openai", "model": "any", "api_key": "offline-secret"},
            {"backend": "gemini", "model": "any", "base_url": "https://server?key=offline-secret"},
            {"backend": "mock", "typo": 1},
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                create_backend(config)
        register_backend("offline_test_protocol", MockBackend)
        self.assertIsInstance(create_backend({"backend": "offline_test_protocol", "seed": 2}), MockBackend)
        with self.assertRaises(ValueError):
            register_backend("mock", MockBackend)


if __name__ == "__main__":
    unittest.main()
