"""Shared classification contracts, independent of any model provider."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from PIL import Image


class CapabilityError(RuntimeError):
    """A backend cannot supply the evidence requested by a scorer."""


class InvalidPredictionError(ValueError):
    """A model response does not satisfy the classification contract."""


@dataclass(frozen=True)
class ClassificationTask:
    """An image question with a fixed, ordered set of candidate labels."""

    labels: tuple[str, ...]
    question: str

    def __post_init__(self) -> None:
        if isinstance(self.labels, str):
            raise ValueError("labels must be a sequence of distinct label strings")
        labels = tuple(self.labels)
        if not 2 <= len(labels) <= 20:
            raise ValueError("classification requires between 2 and 20 labels")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("labels must be nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be distinct")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a nonempty string")
        object.__setattr__(self, "labels", labels)

    @property
    def codes(self) -> tuple[str, ...]:
        """Short output codes keep label wording out of token scoring."""
        return tuple(chr(ord("A") + index) for index in range(len(self.labels)))

    @property
    def prompt(self) -> str:
        options = "\n".join(
            f"{code}: {label}" for code, label in zip(self.codes, self.labels)
        )
        return (
            f"{self.question.strip()}\n\n"
            f"Choose exactly one of these image classification labels:\n{options}\n\n"
            "Return exactly one uppercase option code, with no explanation, "
            "punctuation, or confidence value."
        )


@dataclass(frozen=True)
class Prediction:
    """Provider evidence keyed by actual labels, rather than output codes.

    ``label_logprobs`` contains natural logs of mutually exclusive candidate
    output probabilities before conditioning on the candidate set. Missing
    evidence must be represented by ``None``, never by invented probabilities.
    """

    label_logprobs: dict[str, float] | None = None
    sampled_label: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label_logprobs is not None:
            object.__setattr__(self, "label_logprobs", dict(self.label_logprobs))
        object.__setattr__(self, "metadata", dict(self.metadata))


@runtime_checkable
class Backend(Protocol):
    """The sole model interface consumed by scoring and explanation code."""

    def predict(
        self,
        image: Image.Image,
        task: ClassificationTask,
        *,
        require_logprobs: bool,
        temperature: float = 1.0,
    ) -> Prediction:
        ...
