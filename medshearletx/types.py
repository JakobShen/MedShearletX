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
    """An image question with fixed, ordered class identifiers and display names.

    ``labels`` are unique keys used for scores. Optional ``display_labels`` keep
    the original names when different classes share the same human label.
    """

    labels: tuple[str, ...]
    question: str
    display_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.labels, str):
            raise ValueError("labels must be a sequence of distinct label strings")
        labels = tuple(self.labels)
        if not 2 <= len(labels) <= 1000:
            raise ValueError("classification requires between 2 and 1000 labels")
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("labels must be nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be distinct")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a nonempty string")
        if self.display_labels is not None:
            if isinstance(self.display_labels, str):
                raise ValueError("display_labels must be a sequence of label strings")
            display_labels = tuple(self.display_labels)
            if (len(display_labels) != len(labels)
                    or any(not isinstance(label, str) or not label.strip() for label in display_labels)):
                raise ValueError("display_labels must contain one nonempty string per class")
            object.__setattr__(self, "display_labels", display_labels)
        object.__setattr__(self, "labels", labels)

    @property
    def codes(self) -> tuple[str, ...]:
        """Stable output codes; tokenizer support must be checked by the backend."""
        if len(self.labels) <= 20:
            return tuple(chr(ord("A") + index) for index in range(len(self.labels)))
        width = len(str(len(self.labels) - 1))
        return tuple(f"{index:0{width}d}" for index in range(len(self.labels)))

    @property
    def prompt(self) -> str:
        options = "\n".join(
            f"{code}: {label}" for code, label in zip(self.codes, self.display_labels or self.labels)
        )
        if len(self.labels) <= 20:
            choice_instruction = "Choose exactly one of these image classification labels:"
            code_description = "uppercase option code"
        else:
            choice_instruction = f"Choose exactly one label from ALL {len(self.labels)} image classification labels below:"
            code_description = "zero-padded numeric option code"
        return (
            f"{self.question.strip()}\n\n"
            f"{choice_instruction}\n{options}\n\n"
            f"Return exactly one {code_description}, with no explanation, "
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
