"""Dataset orchestration and inspectable per-image comparison artifacts."""

import json
from importlib.util import find_spec
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .backends import BackendRequestError, create_backend
from .data import ImageDataset
from .explainer import BlackBoxShearletX, ExplainerConfig, to_image
from .scoring import MODES, Scorer
from .tasks import load_task
from .transforms import create_transform
from .types import CapabilityError, InvalidPredictionError


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def describe_result(result, target):
    if result is None:
        return None
    return {"mode": result.mode, "target_score": result.score(target),
            "scores": result.scores, "probabilities": result.probabilities,
            "log_probabilities": result.log_probabilities, "sample_counts": result.sample_counts,
            "observed_label": result.observed_label, "predicted_label": result.predicted_label,
            "diagnostics": result.diagnostics, "requests": result.requests}


def prepare(config, scores=None, target=None):
    task = load_task(config["task"], root=config.get("root", "."))
    modes = scores if scores is not None else config.get("scores", ["probability", "log_margin", "agreement"])
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError("scores must contain distinct probability, log_margin, agreement or target_probability modes")
    target = target if target is not None else config.get("target")
    if target is not None and (not isinstance(target, str) or target not in task.labels):
        raise ValueError("target must be one of the task labels")
    if "target_probability" in modes and target is None:
        raise ValueError("target_probability requires an explicit --target or top-level config target")
    if "api_key" in config.get("model", {}):
        raise ValueError("Use api_key_env, not plaintext api_key in config")
    backend = create_backend(config["model"])
    settings = config.get("sampling", {})
    if "target" in settings:
        raise ValueError("Use top-level config target or --target instead of sampling.target")
    scorers = {
        mode: Scorer(backend, task, mode=mode,
                     target=target if mode == "target_probability" else None, **settings)
        for mode in modes
    }
    optimizer = ExplainerConfig(**config.get("explainer", {}))
    if type(config.get("step_visuals", True)) is not bool:
        raise ValueError("step_visuals must be a boolean")
    size = config.get("image_size", 128)
    if type(size) is not int or size < 16 or (optimizer.mask_resolution == "grid" and optimizer.grid_size > size):
        raise ValueError("image_size must be an integer >=16 and >= grid_size")
    if config.get("resize_mode", "letterbox") not in {"letterbox", "stretch"}:
        raise ValueError("resize_mode must be letterbox or stretch")
    if config.get("resize_filter", "lanczos") not in {"lanczos", "tensor_bilinear"}:
        raise ValueError("resize_filter must be lanczos or tensor_bilinear")
    if type(config.get("pad_value", 0)) is not int or not 0 <= config.get("pad_value", 0) <= 255:
        raise ValueError("pad_value must be an integer in [0,255]")
    return task, scorers, optimizer


def _tensor_bilinear(pixels, width, height):
    """Tensor Resize coordinates: align_corners=False, antialias=False."""
    source_height, source_width = pixels.shape[:2]
    x = np.clip((np.arange(width) + 0.5) * source_width / width - 0.5, 0, source_width - 1)
    y = np.clip((np.arange(height) + 0.5) * source_height / height - 0.5, 0, source_height - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, source_width - 1), np.minimum(y0 + 1, source_height - 1)
    wx = (x - x0).astype(np.float32)[None, :, None]
    wy = (y - y0).astype(np.float32)[:, None, None]
    top = pixels[y0[:, None], x0[None, :]] * (1 - wx) + pixels[y0[:, None], x1[None, :]] * wx
    bottom = pixels[y1[:, None], x0[None, :]] * (1 - wx) + pixels[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bottom * wy


def _preprocessed_pixels(image, config):
    size = config.get("image_size", 128)
    mode = config.get("resize_mode", "letterbox")
    resize_filter = config.get("resize_filter", "lanczos")
    if type(size) is not int or size < 1:
        raise ValueError("image_size must be a positive integer")
    if mode not in {"letterbox", "stretch"}:
        raise ValueError("resize_mode must be letterbox or stretch")
    if resize_filter not in {"lanczos", "tensor_bilinear"}:
        raise ValueError("resize_filter must be lanczos or tensor_bilinear")
    image = image.convert("RGB")
    if resize_filter == "tensor_bilinear":
        pixels = np.asarray(image, dtype=np.float32) / 255
        metadata = {"mode": mode, "size": size, "resampling": "tensor_bilinear",
                    "align_corners": False, "antialias": False, "tensor_dtype": "float32"}
        if mode == "stretch":
            return _tensor_bilinear(pixels, size, size), metadata
        scale = min(size / image.width, size / image.height)
        width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        fitted = _tensor_bilinear(pixels, width, height)
        offset = ((size - width) // 2, (size - height) // 2)
        value = config.get("pad_value", 0)
        square = np.full((size, size, 3), value / 255, dtype=np.float32)
        square[offset[1]:offset[1] + height, offset[0]:offset[0] + width] = fitted
        metadata.update(fitted_size=[width, height], offset=list(offset), pad_value=value)
        return square, metadata
    if mode == "stretch":
        resized = image.resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.float64) / 255, {"mode": mode, "size": size}
    fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    value = config.get("pad_value", 0)
    square = Image.new("RGB", (size, size), (value, value, value))
    square.paste(fitted, offset)
    return np.asarray(square, dtype=np.float64) / 255, {"mode": mode, "size": size, "fitted_size": list(fitted.size),
                                                    "offset": list(offset), "pad_value": value, "resampling": "Lanczos"}


def preprocess_pixels(image, config):
    """Return float RGB pixels; the tensor policy keeps interpolation fractions."""
    return _preprocessed_pixels(image, config)[0]


def preprocess(image, config):
    """Return the exact quantized API/display image and preprocessing metadata."""
    pixels, metadata = _preprocessed_pixels(image, config)
    # Transform preparation promotes to float64. Quantize the same values here;
    # float32 multiplication can otherwise round an exact half pixel differently.
    return to_image(pixels.astype(np.float64)), metadata


def plan_run(config, dataset, limit=1, scores=None, probe=False, target=None):
    _, scorers, optimizer = prepare(config, scores, target)
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    count = min(limit, len(dataset))
    if not count:
        raise ValueError("Dataset is empty")
    bounds = {}
    for mode, scorer in scorers.items():
        repeats = scorer.repeats if mode == "agreement" else 1
        bounds[mode] = repeats if probe else optimizer.request_bound(repeats)
        if not probe and bounds[mode] > optimizer.max_requests:
            raise ValueError(f"{mode} needs up to {bounds[mode]} requests per image; increase max_requests or reduce steps/repeats")
    total = count * sum(bounds.values())
    maximum = config.get("max_total_requests", 500)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_total_requests must be a positive integer")
    if total > maximum:
        raise ValueError(f"Planned bound {total} exceeds max_total_requests={maximum}")
    return {"images": count, "scores": list(scorers), "requests_per_image_bound": bounds,
            "total_requests_bound": total, "max_total_requests": maximum,
            "target": target if target is not None else config.get("target"),
            "transform": config.get("transform", {"name": "shearlet", "scales": 2}),
            "probe": probe}


def run(config, dataset: ImageDataset, output, *, limit=1, scores=None,
        target=None, probe=False, progress=None):
    plan = plan_run(config, dataset, limit, scores, probe, target)
    task, scorers, optimizer = prepare(config, scores, target)
    target = plan["target"]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    # Config contains env variable names only, never resolved credentials.
    effective_config = {**config, "scores": list(scorers)}
    if target is not None:
        effective_config["target"] = target
    write_json(output / "config.json", effective_config)
    write_json(output / "plan.json", plan)
    transform = None if probe else create_transform(plan["transform"])
    results = []
    for index, sample in enumerate(dataset[:plan["images"]]):
        original = dataset.load(sample)
        image, preprocessing = preprocess(original, config)
        pixels = preprocess_pixels(original, config) if config.get("resize_filter") == "tensor_bilinear" else None
        representation = None
        if not probe:
            representation = BlackBoxShearletX(next(iter(scorers.values())), transform, optimizer).prepare_image(image, pixels=pixels)
        folder = output / f"{index:04d}"
        folder.mkdir()
        image.save(folder / "input.png")
        if pixels is not None:
            np.save(folder / "input_tensor.npy", pixels)
        native_reference = None
        fixed_target = target
        for mode, scorer in scorers.items():
            started = time.monotonic()
            requests_before = scorer.requests
            row = {"sample_id": sample.sample_id, "mode": mode, "ground_truth": sample.label,
                   "original_size": list(original.size), "processed_size": list(image.size),
                   "original_mode": original.info["original_mode"],
                   "sample_metadata": sample.metadata, "preprocessing": preprocessing,
                   "folder": f"{index:04d}/{mode}"}
            if progress:
                progress(f"Image {index + 1}/{plan['images']}: {mode}")
            try:
                reused = mode in {"probability", "log_margin"} and native_reference is not None
                reference = native_reference.with_mode(mode) if reused else scorer.evaluate(image)
                reference_requests = 0 if reused else reference.requests
                if mode in {"probability", "log_margin"}:
                    native_reference = reference
                if fixed_target is None:
                    fixed_target = max(reference.probabilities, key=reference.probabilities.get)
                row.update(target=fixed_target, reference=describe_result(reference, fixed_target),
                           reference_reused=reused)
                if probe:
                    row.update(status="ok", requests=reference_requests)
                else:
                    from .step_visuals import StepVisualizer
                    mode_folder = folder / mode
                    mode_folder.mkdir()
                    visualizer = None
                    if config.get("step_visuals", True):
                        if find_spec("matplotlib") is None:
                            row["iteration_visuals_unavailable"] = "Install the figures extra: pip install -e '.[figures]'"
                            if progress:
                                progress(row["iteration_visuals_unavailable"])
                        else:
                            visualizer = StepVisualizer(
                                mode_folder, image, transform, representation[1],
                                model=config["model"].get("model", getattr(scorer.backend, "model", mode)),
                                target_label=fixed_target, normalize_final=optimizer.normalize_final,
                                grid_size=optimizer.grid_size,
                                mask_resolution=optimizer.mask_resolution,
                            )
                    explanation = BlackBoxShearletX(scorer, transform, optimizer).explain(
                        image, fixed_target, reference=reference, representation=representation,
                        on_iteration=visualizer)
                    explanation.image.save(mode_folder / "retained.png")
                    explanation.removed_image.save(mode_folder / "removed.png")
                    np.save(mode_folder / "mask.npy", explanation.mask)
                    write_json(mode_folder / "history.json", explanation.history)
                    row.update(status="ok", retained=describe_result(explanation.retained, fixed_target),
                               removed=describe_result(explanation.removed, fixed_target),
                               diagnostics=explanation.diagnostics,
                               requests=reference_requests + explanation.diagnostics["requests"])
                    if visualizer is not None:
                        row["iteration_index"] = f"{index:04d}/{mode}/index.html"
            except (CapabilityError, InvalidPredictionError, BackendRequestError) as exc:
                # No automatic method switch: comparison records unsupported modes explicitly.
                row.update(status="unavailable", error=str(exc), target=fixed_target)
            row["prediction_attempts"] = scorer.requests - requests_before
            row["elapsed_seconds"] = time.monotonic() - started
            results.append(row)
            write_json(output / "results.json", results)
    write_json(output / "summary.json", {
        "comparisons": len(results), "successful": sum(row["status"] == "ok" for row in results),
        "prediction_attempts": sum(row["prediction_attempts"] for row in results),
        "request_bound": plan["total_requests_bound"],
        "note": "Attempts include failed backend preflight; external HTTP calls can be fewer.",
    })
    return results
