"""Append provider evidence before scoring or explanation can reject it."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import io
import json
import math
from numbers import Real
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

from PIL import Image

from .types import Backend, ClassificationTask, InvalidPredictionError, Prediction


class RequestBudgetExceeded(RuntimeError):
    """An audit wrapper has exhausted its hard limit of provider attempts."""


# Provider adapters already sanitize these fields. An explicit allowlist keeps
# new configuration, transport headers, and credential fields out of evidence.
_METADATA_FIELDS = {
    "backend", "model", "seed", "temperature", "requested_temperature",
    "effective_temperature", "temperature_control", "native_logprobs",
    "native_logprobs_deprecated", "token_budget", "token_budget_field",
    "generation_option_names", "finish_reason", "usage", "response_id",
    "image_detail", "verbalizer_policy", "class_mass_is_lower_bound",
    "class_token_ids_verified", "logprob_scale", "numeric_padding_normalized",
    "logprob_scope", "returned_class_count", "class_coverage_complete",
    "probability_event", "output_token_count", "evaluated_prefix",
}


class RecordedBackend:
    """Wrap a backend with an append-only JSONL log and an atomic request cap.

    Sequence IDs describe reservation order, while lines describe completion
    order. They are scoped by ``run_id`` so appending another wrapper's run
    preserves the earlier evidence. ``requests`` counts reserved attempts,
    including failures; requests denied by the cap never call the provider.
    Provider error messages, transport configuration, and credentials are not
    logged. Image hashes cover the same RGB PNG representation adapters send.
    """

    def __init__(
        self, backend: Backend, path: str | Path, *, max_requests: int,
    ) -> None:
        if isinstance(max_requests, bool) or not isinstance(max_requests, int) or max_requests < 1:
            raise ValueError("max_requests must be a positive integer")
        self.backend = backend
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_requests = max_requests
        self.run_id = str(uuid4())
        self._requests = 0
        self._lock = Lock()

    def __getattr__(self, name: str):
        return getattr(self.backend, name)

    @property
    def requests(self) -> int:
        with self._lock:
            return self._requests

    def predict(
        self, image: Image.Image, task: ClassificationTask, *,
        require_logprobs: bool, temperature: float = 1.0,
    ) -> Prediction:
        with self._lock:
            if self._requests >= self.max_requests:
                raise RequestBudgetExceeded("Recorded backend request budget exhausted.")
            self._requests += 1
            sequence_id = self._requests
        start = perf_counter()
        record = {
            "run_id": self.run_id,
            "sequence_id": sequence_id,
            "require_logprobs": require_logprobs if isinstance(require_logprobs, bool) else None,
            "temperature": _finite_number(temperature),
            "image_sha256": None,
            "prompt_sha256": None,
        }
        try:
            image_bytes = io.BytesIO()
            image.convert("RGB").save(image_bytes, format="PNG")
            record["image_sha256"] = sha256(image_bytes.getvalue()).hexdigest()
            record["prompt_sha256"] = sha256(task.prompt.encode("utf-8")).hexdigest()
            prediction = self.backend.predict(
                image, task, require_logprobs=require_logprobs, temperature=temperature,
            )
            if not isinstance(prediction, Prediction):
                raise InvalidPredictionError("Backend must return a Prediction.")
            record.update(status="success", prediction={
                "sampled_label": prediction.sampled_label,
                "label_logprobs": None if prediction.label_logprobs is None else {
                    label: _finite_number(value)
                    for label, value in prediction.label_logprobs.items() if isinstance(label, str)
                },
                "metadata": _safe_metadata(prediction.metadata),
            })
            return prediction
        except BaseException as error:
            record.update(status="error", error_type=type(error).__name__)
            status_code = getattr(error, "status_code", None)
            if type(status_code) is int and 100 <= status_code <= 599:
                record["http_status"] = status_code
            retryable = getattr(error, "retryable", None)
            if type(retryable) is bool:
                record["retryable"] = retryable
            raise
        finally:
            record["elapsed_seconds"] = perf_counter() - start
            line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            with self._lock:
                with self.path.open("a", encoding="utf-8") as log:
                    log.write(line)


def _finite_number(value) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    return int(value) if isinstance(value, int) else float(value)


def _safe_metadata(metadata: Mapping) -> dict:
    result = {}
    for key in _METADATA_FIELDS & metadata.keys():
        value = metadata[key]
        if key == "usage":
            if isinstance(value, Mapping):
                result[key] = {
                    name: number for name, raw in value.items()
                    if isinstance(name, str) and name.isidentifier()
                    and (number := _finite_number(raw)) is not None
                }
        elif key == "logprob_scope":
            if isinstance(value, str) and value in {"complete", "reported"}:
                result[key] = value
        elif key == "returned_class_count":
            if type(value) is int and 0 <= value <= 1000:
                result[key] = value
        elif key == "class_coverage_complete":
            if type(value) is bool:
                result[key] = value
        elif key == "probability_event":
            if isinstance(value, str) and value in {"output_token_event", "output_digit_sequence_event"}:
                result[key] = value
        elif key == "output_token_count":
            if type(value) is int and 1 <= value <= 1000:
                result[key] = value
        elif key == "evaluated_prefix":
            if isinstance(value, str) and len(value) <= 999 and all(char in "0123456789" for char in value):
                result[key] = value
        elif key == "generation_option_names":
            if isinstance(value, (list, tuple)):
                result[key] = [name for name in value if isinstance(name, str) and name.isidentifier()]
        elif value is None or isinstance(value, bool):
            result[key] = value
        elif isinstance(value, str):
            # Models and response IDs are strings, never transport URLs.
            if "://" not in value:
                result[key] = value
        elif (number := _finite_number(value)) is not None:
            result[key] = number
    return result
