"""Gemini generateContent adapter with optional native token logprobs."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any
from urllib.parse import quote

from PIL import Image

from ..types import CapabilityError, ClassificationTask, InvalidPredictionError, Prediction
from .base import (
    Transport, aggregate_code_logprobs, api_key, http_json, parse_label, png_base64,
    positive_int, positive_number, request_json, safe_usage, single_item,
    validate_api_key_env, validate_base_url, validate_logprob_scope, validate_options, validate_temperature,
)


_PROTECTED = {
    "model", "contents", "systemInstruction", "temperature", "responseLogprobs", "logprobs",
    "candidateCount", "maxOutputTokens", "responseMimeType", "responseSchema", "responseJsonSchema",
    "_responseJsonSchema",
    "responseModalities", "responseFormat", "topP", "topK", "presencePenalty", "frequencyPenalty",
    "generationConfig", "tools", "toolConfig", "api_key", "headers",
}


def _reported_digit_sequence(logprobs, task, visible_text):
    """Measure canonical code paths sharing the actual chosen digit prefix.

    Top-k entries at earlier decoding positions describe different conditional
    distributions. Only the last digit can vary while reusing the measured
    chosen prefix; no probability is inferred for another prefix.
    """
    codes = task.codes
    width = len(codes[0])
    if width < 2 or any(len(code) != width or any(char not in "0123456789" for char in code) for code in codes):
        raise InvalidPredictionError("Gemini multi-token native scores require uniform numeric class codes.")
    chosen = logprobs.get("chosenCandidates")
    top = logprobs.get("topCandidates")
    if (not isinstance(chosen, list) or len(chosen) != width or
            not isinstance(top, list) or len(top) != width):
        raise InvalidPredictionError("Gemini digit-code scores require exactly one native decoding step per digit.")
    code_parts = []
    distributions = []
    token_texts = {}
    for selected, step in zip(chosen, top):
        if not isinstance(selected, Mapping) or selected.get("token") not in tuple("0123456789"):
            raise InvalidPredictionError("Gemini chosen code path must contain only individual ASCII digits.")
        if not isinstance(step, Mapping) or not isinstance(step.get("candidates"), list):
            raise InvalidPredictionError("Gemini native digit top candidates are missing or malformed.")
        measured = {}
        for entry in [*step["candidates"], selected]:
            if not isinstance(entry, Mapping):
                raise InvalidPredictionError("Malformed Gemini native digit token entry.")
            token, token_id, value = entry.get("token"), entry.get("tokenId"), entry.get("logProbability")
            if not isinstance(token, str) or not token or type(token_id) is not int or token_id < 0:
                raise InvalidPredictionError("Malformed Gemini native digit token or token ID.")
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value > 0 or value <= -9999.0):
                raise InvalidPredictionError("A digit-path token has an invalid or unmeasured native logprob.")
            if token_id in token_texts and token_texts[token_id] != token:
                raise InvalidPredictionError("Gemini returned conflicting text for the same native token ID.")
            token_texts[token_id] = token
            if token_id in measured and not math.isclose(measured[token_id]["logProbability"], value, abs_tol=1e-6):
                raise InvalidPredictionError("Gemini returned conflicting logprobs for the same native digit token.")
            measured[token_id] = entry
        if math.fsum(math.exp(entry["logProbability"]) for entry in measured.values()) > 1.0 + 1e-5:
            raise InvalidPredictionError("Gemini native digit token probability mass exceeds one.")
        code_parts.append(selected["token"])
        distributions.append(list(measured.values()))
    code = "".join(code_parts)
    if code not in codes or code != visible_text:
        raise InvalidPredictionError("Gemini visible class code and native digit sequence disagree.")
    prefix = code[:-1]
    prefix_logprob = math.fsum(entry["logProbability"] for entry in chosen[:-1])
    last_entries = [
        {"token": prefix + entry["token"], "tokenId": entry["tokenId"],
         "logProbability": entry["logProbability"]}
        for entry in distributions[-1] if entry["token"] in tuple("0123456789")
    ]
    values = aggregate_code_logprobs(last_entries, task, logprob_key="logProbability", token_id_key="tokenId",
                                    require_all=False)
    joint_logprobs = {label: prefix_logprob + value for label, value in values.items()}
    return joint_logprobs, {"probability_event": "output_digit_sequence_event",
                    "verbalizer_policy": "reported_digit_sequence_codes",
                    "output_token_count": width, "evaluated_prefix": prefix}


class GeminiBackend:
    """Keep Gemini's protocol separate while exposing the same classification API.

    Native logprob availability varies by model/deployment. Enable it explicitly
    only after a probe confirms native vision support. The default complete scope
    requires every class code; reported scope preserves a partial distribution
    for a scorer that explicitly handles missing fixed-target evidence.
    """

    def __init__(
        self, *, model: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        api_key_env: str | None = "GEMINI_API_KEY", supports_logprobs: bool = False,
        logprob_scope: str = "complete",
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
        self.logprob_scope = validate_logprob_scope(logprob_scope)
        self.timeout = positive_number(timeout, "timeout")
        self.max_output_tokens = positive_int(max_output_tokens, "max_output_tokens")
        self.generation_options = validate_options(generation_options, _PROTECTED)
        self.transport = transport or http_json

    @property
    def fixed_sampling_seed(self) -> Any:
        """A repeated provider seed can correlate independent sampling requests."""
        return self.generation_options.get("seed")

    def _request_url(self) -> str:
        return self.base_url + "/models/" + quote(self.model, safe="") + ":generateContent"

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
            self.transport, self._request_url(),
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
        # native scores require a verified one-token or numeric digit-code path.
        visible_parts = [part for part in parts if isinstance(part, Mapping) and not part.get("thought")]
        if not visible_parts or any(not isinstance(part.get("text"), str) for part in visible_parts):
            raise InvalidPredictionError("Gemini returned a non-text classification response.")
        visible_text = "".join(part["text"] for part in visible_parts)
        sampled_label = parse_label(visible_text, task, allow_numeric_padding=not require_logprobs)
        class_logprobs = None
        native_metadata = {}
        if require_logprobs:
            logprobs = candidate.get("logprobsResult")
            if not isinstance(logprobs, Mapping):
                raise InvalidPredictionError("Gemini native token logprobs are missing.")
            chosen_tokens = logprobs.get("chosenCandidates")
            if self.logprob_scope == "reported" and isinstance(chosen_tokens, list) and len(chosen_tokens) > 1:
                class_logprobs, native_metadata = _reported_digit_sequence(logprobs, task, visible_text)
            else:
                chosen = single_item(chosen_tokens, "Gemini chosen output token")
                if parse_label(chosen.get("token"), task) != sampled_label:
                    raise InvalidPredictionError("Gemini generated text and native class token disagree.")
                top = single_item(logprobs.get("topCandidates"), "Gemini token distribution")
                top_tokens = top.get("candidates")
                if not isinstance(top_tokens, list):
                    raise InvalidPredictionError("Gemini native top token candidates are missing.")
                class_logprobs = aggregate_code_logprobs(
                    [*top_tokens, chosen], task, logprob_key="logProbability", token_id_key="tokenId",
                    require_all=self.logprob_scope == "complete",
                )
                native_metadata = {"probability_event": "output_token_event", "output_token_count": 1}
        metadata = {
            "backend": "gemini", "model": self.model, "temperature": temperature,
            "native_logprobs": require_logprobs, "token_budget": self.max_output_tokens,
            "numeric_padding_normalized": visible_text.strip() not in task.codes,
            "generation_option_names": sorted(self.generation_options),
            "finish_reason": candidate.get("finishReason"),
            "usage": safe_usage(response.get("usageMetadata")),
        }
        if require_logprobs:
            metadata.update(
                verbalizer_policy="reported_single_token_codes", class_mass_is_lower_bound=True,
                logprob_scale="deployment_native",
                logprob_scope=self.logprob_scope,
                returned_class_count=len(class_logprobs),
                class_coverage_complete=set(class_logprobs) == set(task.labels),
            )
            metadata.update(native_metadata)
        if isinstance(response.get("responseId"), str):
            metadata["response_id"] = response["responseId"]
        return Prediction(label_logprobs=class_logprobs, sampled_label=sampled_label, metadata=metadata)
