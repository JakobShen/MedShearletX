"""Configuration-based backend factory.

New deployments using an existing protocol require only a new configuration.
New protocols implement ``Backend.predict`` and register a builder here (or via
``register_backend``); scoring, loaders, and explainers do not import providers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import Backend, BackendRequestError, Transport
from .gemini import GeminiBackend
from .mock import MockBackend
from .openai_compatible import OpenAICompatibleBackend


BackendFactory = Callable[..., Backend]
_REGISTRY: dict[str, BackendFactory] = {
    "mock": MockBackend,
    "openai": lambda **kwargs: OpenAICompatibleBackend(provider="openai", **kwargs),
    "vllm": lambda **kwargs: OpenAICompatibleBackend(provider="vllm", **kwargs),
    "gemini": GeminiBackend,
}


def register_backend(name: str, factory: BackendFactory) -> None:
    """Register one protocol builder; reject accidental built-in replacement."""
    if not isinstance(name, str) or not name or name != name.lower() or name in _REGISTRY:
        raise ValueError("Use a new, non-empty lowercase backend name.")
    if not callable(factory):
        raise TypeError("factory must be callable.")
    _REGISTRY[name] = factory


def create_backend(config: Mapping[str, Any], *, transport: Transport | None = None) -> Backend:
    """Create a backend without reading keys or making a network request.

    ``config`` contains ``backend`` and constructor options, including a model
    deployment name and, for remote providers, an ``api_key_env`` variable name.
    Passing ``api_key_env: null`` allows an unauthenticated local vLLM server.
    ``transport`` is injectable for offline fixtures/custom networking.
    """
    if not isinstance(config, Mapping):
        raise TypeError("backend config must be a mapping.")
    options = dict(config)
    name = options.pop("backend", None)
    if not isinstance(name, str) or name not in _REGISTRY:
        raise ValueError("Unknown backend; choose a registered protocol name.")
    if any(key in options for key in {"api_key", "token", "password", "transport", "provider"}):
        raise ValueError("Use api_key_env for credentials and the explicit transport argument for transport.")
    if name == "vllm":
        options.setdefault("api_key_env", None)
    if transport is not None:
        if name == "mock":
            raise ValueError("MockBackend does not use an HTTP transport.")
        options["transport"] = transport
    try:
        return _REGISTRY[name](**options)
    except TypeError:
        raise ValueError("Backend config contains unknown options or is missing required options.") from None


__all__ = [
    "Backend", "BackendRequestError", "GeminiBackend", "MockBackend",
    "OpenAICompatibleBackend", "create_backend", "register_backend",
]
