"""Publication figures drawn from the exact input and explained image pixels."""

from collections.abc import Mapping
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image


def _caption(target_label, score_label, caption, metadata):
    lines = [f"Fixed target: {target_label}."]
    if score_label == "sampling frequency":
        counts = []
        for name in ("reference", "retained"):
            count, samples = metadata.get(f"{name}_count"), metadata.get(f"{name}_samples")
            if count is not None and samples is not None:
                detail = f"{name.capitalize()}: {count}/{samples} target responses"
                interval = metadata.get(f"{name}_interval")
                if interval is not None:
                    detail += f" (95% Wilson CI {100 * interval[0]:.1f}–{100 * interval[1]:.1f}%)"
                counts.append(detail)
        if counts:
            lines.append("; ".join(counts) + ".")
        lines.append("Score: sampled class frequency; confidence values written by the model are unused.")
    else:
        lines.append("Score: measured native class probabilities, conditional on the reported candidate set.")
    lines.append(metadata.get("approximation", "API adaptation: black-box optimization with a grouped shearlet mask."))
    if caption:
        lines.append(caption.strip())
    return " ".join(lines)


def save_explanation_figure(
    input_image: Image.Image,
    explanation_image: Image.Image,
    output_dir: str | Path,
    *,
    model: str,
    target_label: str,
    retained_ratio: float | None,
    score_label: str = "sampling frequency",
    caption: str | None = None,
    metadata: Mapping | None = None,
    save_pdf: bool = True,
) -> dict[str, str]:
    """Save a single explanation and an original/explanation comparison.

    ``retained_ratio`` is the fixed target's explained-image score divided by
    its original-image score; it can exceed one. Pass ``None`` if that ratio
    is undefined. This function only composes already scored image pixels.
    Native probability titles require explicit evidence metadata.
    """
    if not isinstance(input_image, Image.Image) or not isinstance(explanation_image, Image.Image):
        raise TypeError("input_image and explanation_image must be PIL images")
    if score_label not in {"sampling frequency", "probability"}:
        raise ValueError("score_label must be sampling frequency or probability")
    metadata = dict(metadata or {})
    if score_label == "probability" and metadata.get("evidence") != "native_candidate_logprobs":
        raise ValueError("probability figures require measured native_candidate_logprobs evidence")
    if retained_ratio is not None and (
        isinstance(retained_ratio, bool) or not np.isfinite(retained_ratio) or retained_ratio < 0
    ):
        raise ValueError("retained_ratio must be finite and nonnegative, or None")
    if not model.strip() or not target_label.strip():
        raise ValueError("model and target_label must be nonempty")
    try:
        from matplotlib import rc_context
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("Install matplotlib to export explanation figures") from exc

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_title = "Gemini 3.5 Flash-Lite" if model in {"gemini-3.5-flash-lite", "gemini3.5flashlite"} else model
    value = "undefined" if retained_ratio is None else f"{100 * retained_ratio:.2f}%"
    title = f"ShearletX ({model_title})\nRetained {score_label}: {value}"
    footer = _caption(target_label, score_label, caption, metadata)
    paths = {}
    with rc_context({"font.family": "DejaVu Serif", "pdf.fonttype": 42}):
        for comparison in (False, True):
            width = 12 if comparison else 7
            columns = 2 if comparison else 1
            footer_lines = textwrap.wrap(footer, width=145 if comparison else 85)
            footer_height = 0.13 + len(footer_lines) * 0.13
            image_height = width / columns - 0.18
            height = image_height + 1.05 + footer_height
            figure = Figure(figsize=(width, height), dpi=180, facecolor="white")
            FigureCanvasAgg(figure)
            figure.text(0.5, 1 - 0.12 / height, title, ha="center", va="top", fontsize=21 if comparison else 19)
            images = [input_image, explanation_image] if comparison else [explanation_image]
            for index, image in enumerate(images):
                axis = figure.add_axes([
                    (index + 0.015) / columns, footer_height / height,
                    0.97 / columns, image_height / height,
                ])
                axis.imshow(image.convert("RGB"), interpolation="nearest")
                axis.set_axis_off()
                if comparison:
                    axis.set_title("Original image" if index == 0 else "Explanation", fontsize=12, pad=7)
            figure.text(0.025, 0.09 / height, "\n".join(footer_lines), ha="left", va="bottom", fontsize=7.5)
            stem = "comparison" if comparison else "explanation"
            for suffix in ("png", "pdf") if save_pdf else ("png",):
                path = output / f"{stem}.{suffix}"
                figure.savefig(path, dpi=180, facecolor="white", metadata={"Title": title})
                paths[f"{stem}_{suffix}"] = str(path)
            figure.clear()
    return paths


def save_optimization_figure(history, output_dir: str | Path) -> dict[str, str]:
    """Plot sampled training objectives and local penalties, without rescoring."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    figure = Figure(figsize=(11, 4.6), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    steps = [row["step"] for row in history]
    for name, label in (("loss", "Total training loss"), ("distortion", "Sampled fidelity distortion")):
        axes[0].plot(steps, [row[name] for row in history], label=label)
    for name, label in (("mask_energy", "Mean mask value"), ("spatial_energy", "Spatial L1 mean")):
        axes[1].plot(steps, [row[name] for row in history], label=label)
    for axis in axes:
        axis.set_xlabel("Optimization step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=9)
    axes[0].set_title("Noisy training estimates")
    axes[1].set_title("Local sparsity penalties")
    figure.suptitle("ShearletX optimization history")
    figure.text(0.5, 0.025, "Training estimates use resampled noisy images; final clean-image validation is independent.",
                ha="center", fontsize=9)
    figure.subplots_adjust(left=0.065, right=0.98, bottom=0.16, top=0.82, wspace=0.25)
    paths = {}
    for suffix in ("png", "pdf"):
        path = output / f"optimization.{suffix}"
        figure.savefig(path, dpi=150)
        paths[f"optimization_{suffix}"] = str(path)
    figure.clear()
    return paths
