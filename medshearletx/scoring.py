"""Classification evidence and scores without self-reported confidence."""

from dataclasses import dataclass, field, replace
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from threading import Lock

import numpy as np
from PIL import Image

from .types import Backend, CapabilityError, ClassificationTask, InvalidPredictionError


MODES = ("probability", "log_margin", "agreement")


@dataclass(frozen=True)
class ScoreResult:
    """One reusable evaluation of an image.

    Candidate probabilities and sampling frequency measure model behavior;
    neither is a calibrated probability that a medical label is correct.
    Native evidence can be viewed as either probability or log margin without
    querying the model again.
    """

    mode: str
    probabilities: dict[str, float]
    log_probabilities: dict[str, float] | None = None
    sample_counts: dict[str, int] = field(default_factory=dict)
    requests: int = 1
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown score mode: {self.mode!r}")
        object.__setattr__(self, "probabilities", dict(self.probabilities))
        if self.log_probabilities is not None:
            object.__setattr__(self, "log_probabilities", dict(self.log_probabilities))
        object.__setattr__(self, "sample_counts", dict(self.sample_counts))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def scores(self) -> dict[str, float]:
        """Scores for every candidate, using this result's selected mode."""
        return {label: self.score(label) for label in self.probabilities}

    @property
    def predicted_label(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    def score(self, target: str) -> float:
        if target not in self.probabilities:
            raise ValueError(f"unknown target label: {target!r}")
        if self.mode == "log_margin":
            if self.log_probabilities is None:
                raise CapabilityError("log margin requires native label log probabilities")
            competitor = max(
                value for label, value in self.log_probabilities.items() if label != target
            )
            return self.log_probabilities[target] - competitor
        return self.probabilities[target]

    def with_mode(self, mode: str) -> "ScoreResult":
        """Reuse native evidence across probability and log-margin objectives."""
        if mode not in MODES:
            raise ValueError(f"unknown score mode: {mode!r}")
        if mode == "agreement" and self.mode != "agreement":
            raise CapabilityError("agreement requires repeated sampled predictions")
        if self.mode == "agreement" and mode != "agreement":
            raise CapabilityError("sample frequency cannot substitute for native probabilities")
        return replace(self, mode=mode)


class Scorer:
    """Score an image with native log probabilities or repeated label samples.

    ``requests`` counts backend prediction attempts across evaluations, even
    when a prediction fails. A backend may fail during capability or credential
    preflight, so this is an upper bound on requests actually sent to a server.
    Native temperature zero is allowed; backend metadata records whether a
    provider reports probabilities before or after temperature adjustment.
    A retry wrapper can perform additional HTTP attempts inside ``predict``;
    use its RecordedBackend log and counter for those physical requests.
    ``workers`` bounds concurrent agreement samples. Parallel batches finish
    their running attempts before reporting a failure, so an invalid sample
    can consume the full batch of requests. A backend can declare
    ``thread_safe = False`` to require sequential sampling.
    """

    def __init__(
        self,
        backend: Backend,
        task: ClassificationTask,
        mode: str = "probability",
        repeats: int = 16,
        temperature: float = 1.0,
        workers: int = 1,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown score mode: {mode!r}")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
            raise ValueError("workers must be an integer between 1 and 32")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, Real)
            or not np.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be a finite nonnegative number")
        if mode == "agreement" and temperature <= 0:
            raise ValueError("agreement requires temperature > 0 to measure sampling variability")
        if mode == "agreement" and getattr(backend, "fixed_sampling_seed", None) is not None:
            raise ValueError("agreement requires independent draws; remove the provider's fixed sampling seed")
        if mode == "agreement" and workers > 1 and getattr(backend, "thread_safe", True) is False:
            raise ValueError("this backend requires workers=1 for reproducible, thread-safe sampling")
        self.backend = backend
        self.task = task
        self.mode = mode
        self.repeats = repeats
        self.temperature = float(temperature)
        self.workers = workers
        self.requests = 0
        self._requests_lock = Lock()

    def _predict(self, image: Image.Image, *, require_logprobs: bool):
        with self._requests_lock:
            self.requests += 1
        return self.backend.predict(
            image, self.task, require_logprobs=require_logprobs, temperature=self.temperature
        )

    def evaluate(self, image: Image.Image) -> ScoreResult:
        if self.mode == "agreement":
            return self._sample(image)
        prediction = self._predict(image, require_logprobs=True)
        raw = prediction.label_logprobs
        if raw is None:
            raise CapabilityError("backend did not return native label log probabilities")
        if set(raw) != set(self.task.labels):
            missing = sorted(set(self.task.labels) - set(raw))
            extra = sorted(set(raw) - set(self.task.labels), key=str)
            raise InvalidPredictionError(
                f"native evidence must contain every candidate label exactly; "
                f"missing={missing}, extra={extra}"
            )
        values = []
        for label in self.task.labels:
            value = raw[label]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value > 0
            ):
                raise InvalidPredictionError(
                    f"log probability for {label!r} must be finite and <= 0"
                )
            values.append(float(value))
        values = np.asarray(values, dtype=float)
        maximum = float(values.max())
        log_mass = maximum + float(np.log(np.exp(values - maximum).sum()))
        if log_mass > 1e-5:
            raise InvalidPredictionError(
                "native probabilities for mutually exclusive labels sum to more than 1"
            )
        normalized = values - log_mass
        probabilities = np.exp(normalized)
        return ScoreResult(
            mode=self.mode,
            probabilities=dict(zip(self.task.labels, map(float, probabilities))),
            log_probabilities=dict(zip(self.task.labels, map(float, normalized))),
            requests=1,
            diagnostics={
                "class_mass": float(np.exp(log_mass)),
                "log_class_mass": log_mass,
                "entropy": _entropy(probabilities),
                "temperature": self.temperature,
                "evidence": "native_candidate_logprobs",
                "calibrated_correctness": False,
                "backend": dict(prediction.metadata),
            },
        )

    def _sample(self, image: Image.Image) -> ScoreResult:
        counts = dict.fromkeys(self.task.labels, 0)
        metadata = []
        workers = min(self.workers, self.repeats)
        if workers == 1:
            predictions = (
                self._predict(image, require_logprobs=False) for _ in range(self.repeats)
            )
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # map retains sample order; materializing waits for the batch
                # before validating labels, without silently dropping failures.
                predictions = list(executor.map(
                    lambda _: self._predict(image, require_logprobs=False), range(self.repeats)
                ))
        for index, prediction in enumerate(predictions):
            label = prediction.sampled_label
            if not isinstance(label, str) or label not in counts:
                raise InvalidPredictionError(
                    f"sample {index + 1}/{self.repeats} returned invalid label {label!r}; "
                    "invalid samples cannot be silently dropped or renormalized"
                )
            counts[label] += 1
            metadata.append(dict(prediction.metadata))
        probabilities = {label: count / self.repeats for label, count in counts.items()}
        standard_error = {
            label: float(np.sqrt(value * (1 - value) / self.repeats))
            for label, value in probabilities.items()
        }
        return ScoreResult(
            mode="agreement",
            probabilities=probabilities,
            sample_counts=counts,
            requests=self.repeats,
            diagnostics={
                "entropy": _entropy(np.asarray(list(probabilities.values()))),
                "sample_standard_error": standard_error,
                "sample_wilson_95": {
                    label: _wilson_interval(value, self.repeats)
                    for label, value in probabilities.items()
                },
                "temperature": self.temperature,
                "requested_temperature": self.temperature,
                "effective_temperature": _effective_temperature(metadata),
                "workers": workers,
                "requested_workers": self.workers,
                "repeats": self.repeats,
                "evidence": "sampled_label_frequency",
                "calibrated_correctness": False,
                "backend": metadata,
            },
        )


def _effective_temperature(metadata: list[dict]) -> float | None:
    """Report a provider-confirmed common temperature, including overrides."""
    values = [item.get("effective_temperature", item.get("temperature")) for item in metadata]
    if (not values or any(isinstance(value, bool) or not isinstance(value, Real)
                          or not np.isfinite(value) or value < 0 for value in values)):
        return None
    return float(values[0]) if all(value == values[0] for value in values) else None


def _entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0]
    return -float(np.sum(positive * np.log(positive)))


def _wilson_interval(probability: float, samples: int) -> list[float]:
    """A 95% binomial interval that remains informative for unanimous samples."""
    z_squared = 1.959963984540054 ** 2
    denominator = 1 + z_squared / samples
    center = (probability + z_squared / (2 * samples)) / denominator
    radius = np.sqrt(
        z_squared * (probability * (1 - probability) / samples + z_squared / (4 * samples ** 2))
    ) / denominator
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]
