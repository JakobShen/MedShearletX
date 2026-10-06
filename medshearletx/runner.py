"""Dataset orchestration and inspectable per-image comparison artifacts."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .backends import BackendRequestError, create_backend
from .data import ImageDataset
from .explainer import BlackBoxShearletX, ExplainerConfig
from .scoring import Scorer
from .transforms import create_transform
from .types import CapabilityError, ClassificationTask, InvalidPredictionError


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def describe_result(result, target):
    return {"mode": result.mode, "target_score": result.score(target),
            "scores": result.scores, "probabilities": result.probabilities,
            "log_probabilities": result.log_probabilities, "sample_counts": result.sample_counts,
            "diagnostics": result.diagnostics, "requests": result.requests}


def prepare(config, scores=None):
    task = ClassificationTask(config["task"]["labels"], config["task"]["question"])
    modes = scores or config.get("scores", ["probability", "log_margin", "agreement"])
    if not modes or len(set(modes)) != len(modes) or any(m not in {"probability", "log_margin", "agreement"} for m in modes):
        raise ValueError("scores must contain distinct probability, log_margin or agreement modes")
    if "api_key" in config.get("model", {}):
        raise ValueError("Use api_key_env, not plaintext api_key in config")
    backend = create_backend(config["model"])
    settings = config.get("sampling", {})
    scorers = {mode: Scorer(backend, task, mode=mode, **settings) for mode in modes}
    optimizer = ExplainerConfig(**config.get("explainer", {}))
    size = config.get("image_size", 128)
    if type(size) is not int or size < 16 or optimizer.grid_size > size:
        raise ValueError("image_size must be an integer >=16 and >= grid_size")
    if config.get("resize_mode", "letterbox") not in {"letterbox", "stretch"}:
        raise ValueError("resize_mode must be letterbox or stretch")
    if type(config.get("pad_value", 0)) is not int or not 0 <= config.get("pad_value", 0) <= 255:
        raise ValueError("pad_value must be an integer in [0,255]")
    return task, scorers, optimizer


def preprocess(image, config):
    """Aspect-preserving square input by default; record the exact policy."""
    size = config.get("image_size", 128)
    mode = config.get("resize_mode", "letterbox")
    if mode == "stretch":
        return image.resize((size, size), Image.Resampling.LANCZOS), {"mode": mode, "size": size}
    fitted = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    offset = ((size - fitted.width) // 2, (size - fitted.height) // 2)
    value = config.get("pad_value", 0)
    square = Image.new("RGB", (size, size), (value, value, value))
    square.paste(fitted, offset)
    return square, {"mode": mode, "size": size, "fitted_size": list(fitted.size),
                    "offset": list(offset), "pad_value": value, "resampling": "Lanczos"}


def plan_run(config, dataset, limit=1, scores=None, probe=False):
    _, scorers, optimizer = prepare(config, scores)
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
            "transform": config.get("transform", {"name": "shearlet", "scales": 2}),
            "probe": probe}


def run(config, dataset: ImageDataset, output, *, limit=1, scores=None,
        target=None, probe=False, progress=None):
    plan = plan_run(config, dataset, limit, scores, probe)
    task, scorers, optimizer = prepare(config, scores)
    if target is not None and target not in task.labels:
        raise ValueError("target must be one of the task labels")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    # Config contains env variable names only, never resolved credentials.
    write_json(output / "config.json", config)
    write_json(output / "plan.json", plan)
    transform = None if probe else create_transform(plan["transform"])
    results = []
    for index, sample in enumerate(dataset[:plan["images"]]):
        original = dataset.load(sample)
        image, preprocessing = preprocess(original, config)
        representation = None
        if not probe:
            representation = BlackBoxShearletX(next(iter(scorers.values())), transform, optimizer).prepare_image(image)
        folder = output / f"{index:04d}"
        folder.mkdir()
        image.save(folder / "input.png")
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
                reused = mode != "agreement" and native_reference is not None
                reference = native_reference.with_mode(mode) if reused else scorer.evaluate(image)
                reference_requests = 0 if reused else reference.requests
                if mode != "agreement":
                    native_reference = reference
                if fixed_target is None:
                    fixed_target = max(reference.probabilities, key=reference.probabilities.get)
                row.update(target=fixed_target, reference=describe_result(reference, fixed_target),
                           reference_reused=reused)
                if probe:
                    row.update(status="ok", requests=reference_requests)
                else:
                    explanation = BlackBoxShearletX(scorer, transform, optimizer).explain(
                        image, fixed_target, reference=reference, representation=representation)
                    mode_folder = folder / mode
                    mode_folder.mkdir()
                    explanation.image.save(mode_folder / "retained.png")
                    explanation.removed_image.save(mode_folder / "removed.png")
                    np.save(mode_folder / "mask.npy", explanation.mask)
                    write_json(mode_folder / "history.json", explanation.history)
                    row.update(status="ok", retained=describe_result(explanation.retained, fixed_target),
                               removed=describe_result(explanation.removed, fixed_target),
                               diagnostics=explanation.diagnostics,
                               requests=reference_requests + explanation.diagnostics["requests"])
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
