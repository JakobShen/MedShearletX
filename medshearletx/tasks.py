"""Load classification task definitions without executing label-file code."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from .types import ClassificationTask


IMAGENET_QUESTION = (
    "What is the dominant object visible in this image? "
    "Classify it using the complete ImageNet-1k candidate set."
)

CLASSIFICATION_QUESTION = (
    "What is the dominant object visible in this image? "
    "Classify it using the complete candidate set."
)


def load_task(config: Mapping, *, root: str | Path = ".") -> ClassificationTask:
    """Create one generic or complete ImageNet task from a task config.

    Specify exactly one of ``labels`` (an ordered sequence) or
    ``imagenet_labels_path`` (relative to ``root``, or an absolute path).
    An optional ``question`` changes the question without changing class names
    or output codes. The supplied mapping is never modified.
    """
    if not isinstance(config, Mapping):
        raise ValueError("task config must be a mapping")
    if ("labels" in config) == ("imagenet_labels_path" in config):
        raise ValueError("task requires exactly one of labels or imagenet_labels_path")
    if "imagenet_labels_path" in config:
        path = config["imagenet_labels_path"]
        if not isinstance(path, (str, Path)) or isinstance(path, str) and not path.strip():
            raise ValueError("imagenet_labels_path must be a nonempty path")
        return load_imagenet_task(Path(root) / path, question=config.get("question", IMAGENET_QUESTION))
    labels = config["labels"]
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        raise ValueError("labels must be an ordered sequence of class names")
    return ClassificationTask(tuple(labels), config.get("question", CLASSIFICATION_QUESTION))


def load_imagenet_task(
    labels_path: str | Path, *, question: str = IMAGENET_QUESTION,
) -> ClassificationTask:
    """Load the original, ordered 1,000 ImageNet labels from .py or .txt.

    Supported files contain either a literal dictionary or one assignment of
    that dictionary to ``imagenet_labels_dict``. Only AST literals are read;
    Python imports, calls, and other executable statements are never executed.
    """
    source = Path(labels_path).read_text(encoding="utf-8")
    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("ImageNet labels must be a literal dictionary or imagenet_labels_dict assignment.") from error
    if len(module.body) != 1:
        raise ValueError("ImageNet label files must contain exactly one dictionary or assignment.")
    statement = module.body[0]
    if isinstance(statement, ast.Expr):
        dictionary = statement.value
    elif (isinstance(statement, ast.Assign) and len(statement.targets) == 1
          and isinstance(statement.targets[0], ast.Name)
          and statement.targets[0].id == "imagenet_labels_dict"):
        dictionary = statement.value
    else:
        raise ValueError("Expected a literal dictionary or imagenet_labels_dict assignment.")
    if not isinstance(dictionary, ast.Dict):
        raise ValueError("ImageNet labels must be a literal dictionary.")
    try:
        keys = [ast.literal_eval(key) for key in dictionary.keys]
        labels = ast.literal_eval(dictionary)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError("ImageNet labels must contain only literal keys and labels.") from error
    if (len(keys) != 1000 or any(type(key) is not int for key in keys)
            or set(keys) != set(range(1000))):
        raise ValueError("ImageNet labels must have exactly the integer keys 0 through 999, without duplicates.")
    display_labels = tuple(labels[index] for index in range(1000))
    if any(not isinstance(label, str) or not label.strip() for label in display_labels):
        raise ValueError("ImageNet labels must be nonempty strings.")
    counts = Counter(display_labels)
    # The original list has two different ImageNet classes named 'crane': the
    # bird (134) and the machine (517). Preserve all original display names while
    # keeping provider evidence and score dictionaries keyed by distinct classes.
    identifiers = tuple(
        f"{label} [ImageNet {index:03d}]" if counts[label] > 1 else label
        for index, label in enumerate(display_labels)
    )
    return ClassificationTask(identifiers, question, display_labels=display_labels)
