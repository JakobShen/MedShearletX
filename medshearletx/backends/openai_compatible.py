"""OpenAI Chat Completions wire format, also used by vLLM deployments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import Image

from ..types import CapabilityError, ClassificationTask, InvalidPredictionError, Prediction
from .base import (
    Transport, aggregate_code_logprobs, api_key, http_json, parse_label, png_base64,
    positive_int, positive_number, request_json, safe_usage, single_item,
    validate_api_key_env, validate_base_url, validate_options, validate_temperature,
)


_PROTECTED = {
    "model", "messages", "temperature", "logprobs", "top_logprobs", "stream", "n",
    "max_tokens", "max_completion_tokens", "logprob_token_ids", "logit_bias",
    "allowed_token_ids", "response_format", "structured_outputs", "guided_choice",
    "guided_json", "guided_regex", "guided_grammar", "guided_decoding_backend",
    "guided_whitespace_pattern", "bad_words", "logits_processors", "use_beam_search", "watermarking",
    "tools", "tool_choice", "functions", "function_call", "prediction",
    "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty",
    "extra_body", "extra_headers", "api_key", "headers",
}


class OpenAICompatibleBackend:
    """Native one-token class scores without constrained decoding or logit bias.

    ``supports_logprobs`` is an explicit deployment capability declaration;
    merely supporting the Chat API does not imply supporting vision logprobs.
    For recent vLLM, ``class_token_ids`` can request all declared verbalizers
    without depending on top-k coverage. IDs must come from that deployment's
    tokenizer and decode to the corresponding A/B/... code.
    """

    def __init__(
        self, *, model: str, provider: str = "openai", base_url: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY", supports_logprobs: bool = False,
        timeout: float = 60.0, max_tokens: int = 1, token_budget_field: str = "max_tokens",
        generation_options: Mapping[str, Any] | None = None, image_detail: str = "auto",
        class_token_ids: Mapping[str, int] | None = None, transport: Transport | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty deployment name.")
        if provider not in {"openai", "vllm"}:
            raise ValueError("provider must be openai or vllm.")
        if not isinstance(supports_logprobs, bool):
            raise ValueError("supports_logprobs must be a boolean.")
        if token_budget_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("token_budget_field must be max_tokens or max_completion_tokens.")
        if image_detail not in {"auto", "low", "high", "original"}:
            raise ValueError("image_detail must be auto, low, high, or original.")
        if class_token_ids is not None:
            if provider != "vllm" or not isinstance(class_token_ids, Mapping) or not class_token_ids:
                raise ValueError("class_token_ids is supported only for vLLM and must map labels to IDs.")
            if any(not isinstance(label, str) or isinstance(token_id, bool) or
                   not isinstance(token_id, int) or token_id < 0 for label, token_id in class_token_ids.items()):
                raise ValueError("class_token_ids must map labels to non-negative integer token IDs.")
            if len(set(class_token_ids.values())) != len(class_token_ids):
                raise ValueError("Each class must use a distinct token ID.")
        self.model = model
        self.provider = provider
        self.base_url = validate_base_url(base_url or (
            "http://localhost:8000/v1" if provider == "vllm" else "https://api.openai.com/v1"
        ))
        self.api_key_env = validate_api_key_env(api_key_env)
        self.supports_logprobs = supports_logprobs
        self.timeout = positive_number(timeout, "timeout")
        self.max_tokens = positive_int(max_tokens, "max_tokens")
        self.token_budget_field = token_budget_field
        self.image_detail = image_detail
        self.generation_options = validate_options(generation_options, _PROTECTED)
        self.class_token_ids = dict(class_token_ids) if class_token_ids is not None else None
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
            raise CapabilityError("This deployment has not declared native vision logprob support.")
        if self.class_token_ids is not None and set(self.class_token_ids) != set(task.labels):
            raise ValueError("class_token_ids must have exactly the task's actual labels as keys.")
        payload = {
            **self.generation_options, "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": task.prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + png_base64(image), "detail": self.image_detail,
                }},
            ]}],
            "temperature": temperature, self.token_budget_field: self.max_tokens,
        }
        if require_logprobs:
            payload.update(logprobs=True, top_logprobs=20)
            if self.class_token_ids is not None:
                payload["logprob_token_ids"] = [self.class_token_ids[label] for label in task.labels]
                # Recent vLLM gives explicit IDs precedence. Matching the length
                # also supports versions that validate top_logprobs against IDs.
                payload["top_logprobs"] = len(task.labels)
        key = api_key(self.api_key_env)
        headers = {"Authorization": "Bearer " + key} if key else {}
        response = request_json(
            self.transport, self.base_url + "/chat/completions", headers=headers,
            payload=payload, timeout=self.timeout,
        )
        choice = single_item(response.get("choices"), "completion choice")
        message = choice.get("message")
        if not isinstance(message, Mapping) or message.get("refusal") or message.get("tool_calls"):
            raise InvalidPredictionError("The model refused or returned a non-classification response.")
        if choice.get("finish_reason") not in {"stop", "length", None}:
            raise InvalidPredictionError("The provider did not finish a usable classification.")
        sampled_label = parse_label(message.get("content"), task)
        class_logprobs = None
        class_token_ids_verified = False
        if require_logprobs:
            logprobs = choice.get("logprobs")
            if not isinstance(logprobs, Mapping) or logprobs.get("refusal"):
                raise InvalidPredictionError("Native token logprobs are missing or describe a refusal.")
            token = single_item(logprobs.get("content"), "visible output token with native logprobs")
            if parse_label(token.get("token"), task) != sampled_label:
                raise InvalidPredictionError("Generated text and native class token disagree.")
            top_tokens = token.get("top_logprobs")
            if not isinstance(top_tokens, list):
                raise InvalidPredictionError("Native top_logprobs are missing.")
            entries = [*top_tokens, token]
            class_logprobs = aggregate_code_logprobs(
                entries, task, logprob_key="logprob", token_id_key="token_id",
                class_token_ids=self.class_token_ids,
            )
            if self.class_token_ids is not None:
                class_token_ids_verified = all(
                    any(entry.get("token_id") == token_id for entry in entries)
                    for token_id in self.class_token_ids.values()
                )
        metadata = {
            "backend": self.provider, "model": self.model, "temperature": temperature,
            "native_logprobs": require_logprobs, "image_detail": self.image_detail,
            "token_budget_field": self.token_budget_field, "token_budget": self.max_tokens,
            "generation_option_names": sorted(self.generation_options),
            "finish_reason": choice.get("finish_reason"), "usage": safe_usage(response.get("usage")),
        }
        if require_logprobs:
            metadata.update(
                verbalizer_policy="explicit_token_ids" if self.class_token_ids else "reported_single_token_codes",
                class_mass_is_lower_bound=not class_token_ids_verified,
                class_token_ids_verified=class_token_ids_verified,
                logprob_scale="deployment_native",
            )
        if isinstance(response.get("id"), str):
            metadata["response_id"] = response["id"]
        return Prediction(label_logprobs=class_logprobs, sampled_label=sampled_label, metadata=metadata)
