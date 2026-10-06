"""A spatial, deterministic mock for offline pipeline tests; not a medical model."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..types import CapabilityError, ClassificationTask, Prediction
from .base import validate_temperature


class MockBackend:
    thread_safe = False

    def __init__(self, *, seed: int = 0, model: str = "mock-spatial", supports_logprobs: bool = True) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not isinstance(supports_logprobs, bool):
            raise ValueError("supports_logprobs must be a boolean.")
        self.seed = seed
        self.model = model
        self.supports_logprobs = supports_logprobs
        self._rng = np.random.default_rng(seed)

    def predict(
        self, image: Image.Image, task: ClassificationTask, *,
        require_logprobs: bool, temperature: float = 1.0,
    ) -> Prediction:
        temperature = validate_temperature(temperature)
        if require_logprobs and not self.supports_logprobs:
            raise CapabilityError("Mock native logprobs were disabled in this configuration.")
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        height, width = gray.shape
        y, x = np.mgrid[:height, :width]
        x = (x + 0.5) / width
        y = (y + 0.5) / height
        weights = np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.035)
        signal = float(np.sum((gray - 0.5) * weights) / np.sum(weights))
        # Two classes produce opposite evidence; additional classes span it.
        logits = np.linspace(1.0, -1.0, len(task.labels)) * 8.0 * signal
        logprobs = logits - np.logaddexp.reduce(logits)
        if temperature == 0:
            sampled = int(np.argmax(logits))
        else:
            sampling_logits = logits / temperature
            probabilities = np.exp(sampling_logits - np.logaddexp.reduce(sampling_logits))
            sampled = int(self._rng.choice(len(task.labels), p=probabilities))
        return Prediction(
            label_logprobs={label: float(value) for label, value in zip(task.labels, logprobs)} if require_logprobs else None,
            sampled_label=task.labels[sampled],
            metadata={"backend": "mock", "model": self.model, "seed": self.seed,
                      "temperature": temperature, "native_logprobs": require_logprobs,
                      "logprob_scale": "raw_before_sampling_temperature", "class_mass_is_lower_bound": False},
        )
