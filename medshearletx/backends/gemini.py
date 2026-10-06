"""Gemini generateContent adapter with optional native token logprobs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from PIL import Image

from ..types import CapabilityError, ClassificationTask, InvalidPredictionError, Prediction
from .base import (
    Transport, aggregate_code_logprobs, api_key, http_json, parse_label, png_base64,
    positive_int, positive_number, request_json, safe_usage, single_item,
    validate_api_key_env, validate_base_url, validate_options, validate_temperature,
)


_PROTECTED = {
    "model", "contents", "systemInstruction", "temperature", "responseLogprobs", "logprobs",
    "candidateCount", "maxOutputTokens", "responseMimeType", "responseSchema", "responseJsonSchema",
    "_responseJsonSchema",
    "responseModalities", "responseFormat", "topP", "topK", "presencePenalty", "frequencyPenalty",
    "generationConfig", "tools", "toolConfig", "api_key", "headers",
}


class GeminiBackend:
    """Keep Gemini's protocol separate while exposing the same classification API.

    Native logprob availability varies by model/deployment. Enable it explicitly
    only after a probe confirms that every class code is returned.
    """

    def __init__(
        self, *, model: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key_env: str | None = "GEMINI_API_KEY", supports_logprobs: bool = False,
        timeout: float = 60.0, max_output_tokens: int = 1,
        generation_options: Mapping[str, Any] | None = None, transport: Transport | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty deployment name.")
        if not isinstance(supports_logprobs, bool):
            raise ValueError("supports_logprobs must be a boolean.")
        self.model = model.removeprefix("models/")
        if not self.model.strip():
            raise ValueError("model must name a Gemini deployment.")
        self.base_url = validate_base_url(base_url)
        self.api_key_env = validate_api_key_env(api_key_env)
        self.supports_logprobs = supports_logprobs
        self.timeout = positive_number(timeout, "timeout")
        self.max_output_tokens = positive_int(max_output_tokens, "max_output_tokens")
        self.generation_options = validate_options(generation_options, _PROTECTED)
        self.transport = transport or http_json

    @property
    def fixed_sampling_seed(self) -> Any:
        """A repeated provider seed can correlate independent sampling requests."""
        return self.generation_options.get("seed")

    def predict(
        self, image: Image.Image, task: ClassificationTask, *,
        require_logprobs: bool, temperature: float = 1.0,
    ) -> Prediction:
        temperature = validate_temperature(temperature)
        if require_logprobs and not self.supports_logprobs:
            raise CapabilityError("This Gemini deployment has not declared native vision logprob support.")
        generation_config = {
            **self.generation_options, "temperature": temperature, "candidateCount": 1,
            "maxOutputTokens": self.max_output_tokens,
        }
        if require_logprobs:
            generation_config.update(responseLogprobs=True, logprobs=20)
        payload = {
            "contents": [{"role": "user", "parts": [
                {"text": task.prompt},
                {"inlineData": {"mimeType": "image/png", "data": png_base64(image)}},
            ]}], "generationConfig": generation_config,
        }
        key = api_key(self.api_key_env)
        response = request_json(
            self.transport, self.base_url + "/models/" + quote(self.model, safe="") + ":generateContent",
            # Header auth keeps the API key out of URLs and URL-bearing errors.
            headers={"x-goog-api-key": key} if key else {}, payload=payload, timeout=self.timeout,
        )
        feedback = response.get("promptFeedback")
        if isinstance(feedback, Mapping) and feedback.get("blockReason"):
            raise InvalidPredictionError("Gemini blocked the classification prompt.")
        candidate = single_item(response.get("candidates"), "Gemini response candidate")
        if candidate.get("finishReason") not in {"STOP", "MAX_TOKENS", None}:
            raise InvalidPredictionError("Gemini did not finish a usable classification.")
        content = candidate.get("content")
        if not isinstance(content, Mapping) or not isinstance(content.get("parts"), list):
            raise InvalidPredictionError("Gemini returned no classification content.")
        parts = content["parts"]
        if any(not isinstance(part, Mapping) for part in parts):
            raise InvalidPredictionError("Gemini returned malformed classification parts.")
        # Hidden reasoning parts are excluded from the visible answer, but the
        # native logprob path still requires exactly one decoding step below.
        visible_parts = [part for part in parts if isinstance(part, Mapping) and not part.get("thought")]
        if not visible_parts or any(not isinstance(part.get("text"), str) for part in visible_parts):
            raise InvalidPredictionError("Gemini returned a non-text classification response.")
        sampled_label = parse_label("".join(part["text"] for part in visible_parts), task)
        class_logprobs = None
        if require_logprobs:
            logprobs = candidate.get("logprobsResult")
            if not isinstance(logprobs, Mapping):
                raise InvalidPredictionError("Gemini native token logprobs are missing.")
            chosen = single_item(logprobs.get("chosenCandidates"), "Gemini chosen output token")
            if parse_label(chosen.get("token"), task) != sampled_label:
                raise InvalidPredictionError("Gemini generated text and native class token disagree.")
            top = single_item(logprobs.get("topCandidates"), "Gemini token distribution")
            top_tokens = top.get("candidates")
            if not isinstance(top_tokens, list):
                raise InvalidPredictionError("Gemini native top token candidates are missing.")
            class_logprobs = aggregate_code_logprobs(
                [*top_tokens, chosen], task, logprob_key="logProbability", token_id_key="tokenId",
            )
        metadata = {
            "backend": "gemini", "model": self.model, "temperature": temperature,
            "native_logprobs": require_logprobs, "token_budget": self.max_output_tokens,
            "generation_option_names": sorted(self.generation_options),
            "finish_reason": candidate.get("finishReason"),
            "usage": safe_usage(response.get("usageMetadata")),
        }
        if require_logprobs:
            metadata.update(
                verbalizer_policy="reported_single_token_codes", class_mass_is_lower_bound=True,
                logprob_scale="deployment_native",
            )
        if isinstance(response.get("responseId"), str):
            metadata["response_id"] = response["responseId"]
        return Prediction(label_logprobs=class_logprobs, sampled_label=sampled_label, metadata=metadata)
