"""Provider interface and small, dependency-free HTTP/token helpers."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from PIL import Image

from ..types import Backend, ClassificationTask, InvalidPredictionError


class BackendRequestError(RuntimeError):
    """An HTTP/configuration failure without provider bodies or credentials."""

    def __init__(self, message, *, retryable=False, status_code=None):
        super().__init__(message)
        self.retryable = retryable is True
        self.status_code = status_code if type(status_code) is int else None


Transport = Callable[..., Mapping[str, Any]]


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward an authorization header to a redirected host.
        return None


def http_json(
    url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float,
) -> Mapping[str, Any]:
    """POST JSON once. Error strings intentionally omit URLs and response bodies."""
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.build_opener(_NoRedirects).open(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise BackendRequestError(
            f"Provider returned HTTP {error.code}; check credentials, model capability, and configuration.",
            retryable=error.code in {429, 500, 502, 503, 504}, status_code=error.code,
        ) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise BackendRequestError("Provider connection failed or timed out.", retryable=True) from None
    except (UnicodeError, ValueError):
        raise BackendRequestError("Provider returned invalid JSON.") from None
    if not isinstance(result, Mapping):
        raise BackendRequestError("Provider returned a JSON response with the wrong shape.")
    return result


def request_json(transport: Transport, url: str, **kwargs: Any) -> Mapping[str, Any]:
    try:
        result = transport(url, **kwargs)
    except BackendRequestError:
        raise
    except Exception:
        # SDK/custom transport errors often include the URL, headers, or body.
        raise BackendRequestError("Provider transport failed; inspect its configuration.") from None
    if not isinstance(result, Mapping):
        raise BackendRequestError("Provider transport must return a decoded JSON object.")
    return result


def validate_base_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("base_url must be an HTTP(S) URL.")
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("base_url must be an HTTP(S) URL without credentials, query, or fragment.")
    return value.rstrip("/")


def validate_api_key_env(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or
                              not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value)):
        raise ValueError("api_key_env must be an environment variable name or null.")
    return value


def api_key(env_name: str | None) -> str | None:
    if env_name is None:
        return None
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise BackendRequestError(f"Set the {env_name} environment variable before calling this backend.")
    return key


def png_base64(image: Image.Image) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image.")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def validate_temperature(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("temperature must be a finite non-negative number.")
    return float(value)


def positive_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return float(value)


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def validate_options(options: Mapping[str, Any] | None, protected: set[str]) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping) or not all(isinstance(key, str) for key in options):
        raise ValueError("generation_options must be a JSON object.")
    if set(options) & protected:
        raise ValueError("generation_options cannot override classification or score settings.")
    try:
        # Copy nested options so a caller cannot silently mutate later requests.
        result = json.loads(json.dumps(dict(options), allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("generation_options must contain JSON-compatible values.") from None
    return result


def parse_label(text: Any, task: ClassificationTask) -> str:
    if not isinstance(text, str) or text.strip() not in task.codes:
        raise InvalidPredictionError("Expected exactly one configured class code; got an invalid or refused answer.")
    return task.labels[task.codes.index(text.strip())]


def single_item(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise InvalidPredictionError(f"Expected exactly one {description}.")
    return value[0]


def aggregate_code_logprobs(
    entries: Sequence[Mapping[str, Any]], task: ClassificationTask, *,
    logprob_key: str, token_id_key: str | None = None,
    class_token_ids: Mapping[str, int] | None = None,
) -> dict[str, float]:
    """Sum reported single-token code variants, never invent missing probabilities.

    Top-k APIs expose a subset of vocabulary tokens. Whitespace variants present
    in that subset are summed; unseen variants are not assumed to have zero mass.
    In explicit-ID mode the declared ID per class defines the verbalizer set.
    """
    values: dict[str, dict[Any, float]] = {label: {} for label in task.labels}
    expected = {code: label for code, label in zip(task.codes, task.labels)}
    selected_ids = {token_id: label for label, token_id in (class_token_ids or {}).items()}
    entries = list(entries)
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise InvalidPredictionError("Malformed native token logprob entry.")
    for entry in entries:
        token_id = entry.get(token_id_key) if token_id_key else None
        if token_id is not None and (isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0):
            raise InvalidPredictionError("Malformed native token ID.")
    if class_token_ids is not None and any(entry.get(token_id_key) is not None for entry in entries):
        # The provider can include a sampled whitespace variant outside our
        # declared ID set. It remains a valid classification, but its mass must
        # not enter the score for the declared verbalizers.
        entries = [entry for entry in entries if entry.get(token_id_key) in selected_ids]
        for entry in entries:
            token = entry.get("token")
            if not isinstance(token, str) or expected.get(token.strip()) != selected_ids[entry[token_id_key]]:
                raise InvalidPredictionError("A configured class token ID does not decode to its expected code.")
    for entry in entries:
        token = entry.get("token")
        if not isinstance(token, str) or token.strip() not in expected:
            continue
        label = expected[token.strip()]
        value = entry.get(logprob_key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value > 0 or value <= -9999.0):
            raise InvalidPredictionError("A class token has an invalid or unmeasured native logprob.")
        token_id = entry.get(token_id_key) if token_id_key else None
        identity = ("id", token_id) if token_id is not None else ("text", token)
        if identity in values[label] and not math.isclose(values[label][identity], value, abs_tol=1e-6):
            raise InvalidPredictionError("The provider returned conflicting logprobs for the same token.")
        values[label][identity] = float(value)
    missing = [label for label, variants in values.items() if not variants]
    if missing:
        raise InvalidPredictionError(
            "Native logprobs omit one or more class codes; choose another deployment or use sampling agreement."
        )
    if class_token_ids is not None and any(len(variants) != 1 for variants in values.values()):
        raise InvalidPredictionError("Explicit class_token_ids requires one reported token per class.")
    result = {}
    for label, variants in values.items():
        maximum = max(variants.values())
        result[label] = maximum + math.log(math.fsum(math.exp(value - maximum) for value in variants.values()))
    if math.fsum(math.exp(value) for value in result.values()) > 1.0 + 1e-5:
        raise InvalidPredictionError("The provider returned class probability mass greater than one.")
    return result


def safe_usage(value: Any) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): number for key, number in value.items()
            if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number)}
