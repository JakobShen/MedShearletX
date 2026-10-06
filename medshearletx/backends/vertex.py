"""Vertex AI Express mode using Gemini's existing multimodal protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from PIL import Image

from ..types import ClassificationTask, Prediction
from .base import Transport, http_json, validate_temperature
from .gemini import GeminiBackend


class VertexBackend(GeminiBackend):
    """Use an Express API key without a project or location in the REST path.

    Gemini 3.5 Flash-Lite ignores custom sampling temperatures. Omit that field
    and record its provider default so sampling provenance remains accurate
    even when a caller requested another temperature.
    """

    def __init__(
        self, *, model: str, base_url: str = "https://aiplatform.googleapis.com/v1",
        api_key_env: str | None = "VERTEX_API_KEY", supports_logprobs: bool = False,
        timeout: float = 60.0, max_output_tokens: int = 128,
        generation_options: Mapping[str, Any] | None = None, transport: Transport | None = None,
    ) -> None:
        self._raw_transport = transport or http_json
        super().__init__(
            model=model, base_url=base_url, api_key_env=api_key_env,
            supports_logprobs=supports_logprobs, timeout=timeout,
            max_output_tokens=max_output_tokens, generation_options=generation_options,
            transport=self._vertex_transport,
        )

    @property
    def _provider_default_temperature(self) -> bool:
        return self.model.startswith("gemini-3.5-flash-lite")

    def _request_url(self) -> str:
        return (
            self.base_url + "/publishers/google/models/" + quote(self.model, safe="")
            + ":generateContent"
        )

    def _vertex_transport(
        self, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float,
    ) -> Mapping[str, Any]:
        if self._provider_default_temperature:
            config = dict(payload["generationConfig"])
            config.pop("temperature", None)
            payload = {**payload, "generationConfig": config}
        return self._raw_transport(url, headers=headers, payload=payload, timeout=timeout)

    def predict(
        self, image: Image.Image, task: ClassificationTask, *,
        require_logprobs: bool, temperature: float = 1.0,
    ) -> Prediction:
        requested = validate_temperature(temperature)
        result = super().predict(image, task, require_logprobs=require_logprobs, temperature=requested)
        effective = 1.0 if self._provider_default_temperature else requested
        metadata = {
            **result.metadata, "backend": "vertex", "temperature": effective,
            "requested_temperature": requested, "effective_temperature": effective,
            "temperature_control": (
                "provider_default_ignored" if self._provider_default_temperature else "requested"
            ),
            "native_logprobs_deprecated": self.model.startswith("gemini-3"),
        }
        return Prediction(
            label_logprobs=result.label_logprobs, sampled_label=result.sampled_label, metadata=metadata,
        )
