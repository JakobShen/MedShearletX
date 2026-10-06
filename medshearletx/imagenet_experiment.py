"""Single-image classification experiments with independent final evaluation."""

from collections.abc import Mapping
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from .audit import RecordedBackend
from .backends import create_backend
from .backends.retry import RetryingBackend
from .data import ImageDataset, ImageSample
from .explainer import BlackBoxShearletX, ExplainerConfig
from .figures import save_explanation_figure, save_optimization_figure
from .runner import describe_result, preprocess, preprocess_pixels, write_json
from .scoring import Scorer
from .step_visuals import StepVisualizer
from .tasks import load_task
from .transforms import create_transform


def _optimization_scorer(config, backend, task, target=None):
    mode = config.get("optimization_score", "agreement")
    if mode not in {"agreement", "target_probability"}:
        raise ValueError("optimization_score must be agreement or target_probability")
    settings = {"repeats": 2 if mode == "agreement" else 1,
                **config.get("optimization_sampling", {})}
    if mode == "target_probability":
        if settings["repeats"] != 1:
            raise ValueError("target_probability uses one native request; set optimization_sampling.repeats=1")
        # Before target selection this is used for local validation only.
        return Scorer(backend, task, mode=mode, target=target or task.labels[0], **settings)
    return Scorer(backend, task, mode=mode, **settings)


def prepare_experiment(config, root):
    """Resolve dataset/task paths relative to the explicit project root."""
    root = Path(root).resolve()
    task_config = config["task"] if "task" in config else {"imagenet_labels_path": config["labels_path"]}
    task = load_task(task_config, root=root)
    image_path = root / config["image_path"]
    metadata = config.get("image_metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("image_metadata must be a mapping")
    dataset = ImageDataset([ImageSample(
        sample_id=config.get("sample_id", image_path.stem), image_path=image_path,
        metadata=metadata,
    )])
    optimizer = ExplainerConfig(**config.get("explainer", {}))
    mode = config.get("optimization_score", "agreement")
    if mode not in {"agreement", "target_probability"}:
        raise ValueError("optimization_score must be agreement or target_probability")
    sampling = {"repeats": 2 if mode == "agreement" else 1,
                **config.get("optimization_sampling", {})}
    repeats = sampling["repeats"] if mode == "agreement" else 1
    if mode == "target_probability" and optimizer.unbiased_sampling_distortion:
        raise ValueError("target_probability requires unbiased_sampling_distortion=false")
    selection = config.get("selection_samples", 64)
    evaluation = config.get("evaluation_samples", 128)
    maximum = config.get("max_total_requests", 1600)
    for name, value in (("selection_samples", selection), ("evaluation_samples", evaluation),
                        ("max_total_requests", maximum)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    # Includes a conservative redundant optimization reference even though reused.
    retry = {"retry_budget": 0, **config.get("retry", {})}
    # A shared retry allowance adds to the whole run, rather than multiplying
    # the nominal bound by a per-call retry count.
    retry_budget = retry["retry_budget"]
    if type(retry_budget) is not int or retry_budget < 0:
        raise ValueError("retry_budget must be a nonnegative integer")
    bound = selection + 3 * evaluation + optimizer.request_bound(repeats) + retry_budget
    if bound > maximum:
        raise ValueError(f"Planned {bound} requests exceeds max_total_requests={maximum}")
    if optimizer.request_bound(repeats) + retry_budget > optimizer.max_requests:
        raise ValueError("Optimization exceeds explainer.max_requests")
    backend = create_backend(config["model"])
    if mode == "target_probability" and getattr(backend, "supports_logprobs", None) is False:
        raise ValueError("target_probability requires a deployment with declared native logprob support")
    RetryingBackend(backend, **retry)  # Validate the remaining reliability options.
    _optimization_scorer(config, backend, task)
    Scorer(backend, task, mode="agreement", repeats=evaluation,
           workers=config.get("evaluation_workers", 8))
    image, preprocessing = preprocess(dataset.load(dataset[0]), config)
    return task, dataset, optimizer, backend, image, preprocessing, {
        "classes": len(task.labels), "prompt_characters": len(task.prompt),
        "sample_id": dataset[0].sample_id, "image_metadata": dict(metadata),
        "selection_samples": selection, "evaluation_samples_per_image": evaluation,
        "total_requests_bound": bound, "max_total_requests": maximum,
        "retry_budget": retry_budget,
        "optimization_score": mode,
        "optimization": asdict(optimizer), "preprocessing": preprocessing,
        "model": config["model"],
        "native_probability": ("fixed target output-code event during optimization; held-out sampled_label_frequency"
                               if mode == "target_probability" else "unavailable; sampled_label_frequency"),
    }


def _usage(path):
    """Sum provider-reported usage without inventing absent billing information."""
    totals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for key, value in row.get("prediction", {}).get("metadata", {}).get("usage", {}).items():
            if type(value) is int:
                totals[key] = totals.get(key, 0) + value
    return totals


def run_experiment(config, root, output, progress=print):
    """Run the protocol, retaining a safe failure summary and any checkpoints."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    try:
        return _run_experiment(config, root, output, progress)
    except (Exception, KeyboardInterrupt) as error:
        if (output / "plan.json").is_file():
            log = output / "requests.jsonl"
            records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            write_json(output / "failure.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                "error_type": type(error).__name__,
                "recorded_attempts": len(records),
                "failed_attempts": sum(row["status"] == "error" for row in records),
                "checkpoint_available": (output / "checkpoint-mask.npy").is_file(),
                "usage": _usage(log) if log.exists() else {},
            })
        raise


def _run_experiment(config, root, output, progress):
    task, dataset, optimizer, backend, image, preprocessing, plan = prepare_experiment(config, root)
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "plan.json", plan)
    (output / "classification_prompt.txt").write_text(task.prompt, encoding="utf-8")
    image.save(output / "input.png")
    recorded = RecordedBackend(backend, output / "requests.jsonl", max_requests=plan["max_total_requests"])
    reliable = RetryingBackend(recorded, retry_budget=plan["retry_budget"],
                              **{key: value for key, value in config.get("retry", {}).items() if key != "retry_budget"})
    scoring = _optimization_scorer(config, reliable, task)
    explainer = BlackBoxShearletX(scoring, create_transform(config["transform"]), optimizer)
    pixels = preprocess_pixels(dataset.load(dataset[0]), config) if config.get("resize_filter") == "tensor_bilinear" else None
    if pixels is not None:
        np.save(output / "input_tensor.npy", pixels)
    representation = explainer.prepare_image(image, pixels=pixels)  # Local preflight before any billable call.
    started = time.monotonic()
    selector = Scorer(reliable, task, mode="agreement", repeats=plan["selection_samples"],
                      workers=config.get("evaluation_workers", 8))
    progress(f"Selecting the fixed target from the full {len(task.labels)}-class task")
    selection = selector.evaluate(image)
    target = selection.predicted_label
    if plan["optimization_score"] == "target_probability":
        explainer = BlackBoxShearletX(_optimization_scorer(config, reliable, task, target),
                                     explainer.transform, optimizer)
    write_json(output / "selection.json", describe_result(selection, target))
    progress(f"Fixed target: {target}; {selection.sample_counts[target]}/{selection.requests} samples")
    # Hold out the following samples from target selection and optimization.
    evaluator = Scorer(reliable, task, mode="agreement", repeats=plan["evaluation_samples_per_image"],
                       workers=config.get("evaluation_workers", 8))
    reference = evaluator.evaluate(image)
    write_json(output / "reference.json", describe_result(reference, target))

    def on_step(row):
        write_json(output / "latest_step.json", row)
        progress(f"Step {row['step']}/{optimizer.steps}: loss={row['loss']:.4f}, "
                 f"mask={row['mask_energy']:.3f}, requests={recorded.requests}")

    def checkpoint(mask, history):
        np.save(output / "checkpoint-mask.npy", mask)
        write_json(output / "checkpoint-history.json", history)

    visualizer = StepVisualizer(
        output, image, explainer.transform, representation[1],
        model=config["model"].get("model", getattr(backend, "model", config["model"]["backend"])),
        target_label=target, normalize_final=optimizer.normalize_final, grid_size=optimizer.grid_size,
        mask_resolution=optimizer.mask_resolution,
    )
    explanation = explainer.explain(image, target=target,
                                    reference=selection if plan["optimization_score"] == "agreement" else None,
                                    representation=representation, progress=on_step,
                                    checkpoint=checkpoint, on_iteration=visualizer)
    explanation.image.save(output / "retained.png")
    explanation.removed_image.save(output / "removed.png")
    np.save(output / "mask.npy", explanation.mask)
    write_json(output / "history.json", explanation.history)
    optimization_figures = save_optimization_figure(explanation.history, output)
    write_json(output / "optimization_diagnostics.json", explanation.diagnostics)
    optimization_scores = {name: describe_result(score, target) for name, score in (
        ("reference", explanation.reference), ("retained", explanation.retained),
        ("removed", explanation.removed))}
    write_json(output / "optimization_scores.json", optimization_scores)
    progress("Evaluating clean retained and removed images with held-out samples")
    retained = evaluator.evaluate(explanation.image)
    write_json(output / "retained.json", describe_result(retained, target))
    removed = evaluator.evaluate(explanation.removed_image)
    write_json(output / "removed.json", describe_result(removed, target))
    before = reference.probabilities[target]
    ratio = retained.probabilities[target] / before if before else None
    mask_description = "full coefficient mask" if optimizer.mask_resolution == "full" else "grouped mask"
    score_description = ("native fixed-target code score; held-out sampled evaluation"
                         if plan["optimization_score"] == "target_probability" else "finite sampling; no native logprobs")
    metadata = {"evidence": "sampled_label_frequency",
                "approximation": f"API adaptation: {optimizer.optimizer}, {mask_description}, {score_description}."}
    for name, result in (("reference", reference), ("retained", retained)):
        metadata.update({f"{name}_count": result.sample_counts[target],
                         f"{name}_samples": result.requests,
                         f"{name}_interval": result.diagnostics["sample_wilson_95"][target]})
    figures = save_explanation_figure(image, explanation.image, output,
                                     model=config["model"]["model"], target_label=target,
                                     retained_ratio=ratio, metadata=metadata)
    result = {"status": "ok", "model": config["model"], "classes": len(task.labels),
              "target": target, "target_index": task.labels.index(target),
              "sample_id": dataset[0].sample_id, "image_metadata": dataset[0].metadata,
              "paper_predicted_class": dataset[0].metadata.get("paper_predicted_class"),
              "selection": describe_result(selection, target),
              "reference": describe_result(reference, target),
              "retained": describe_result(retained, target), "removed": describe_result(removed, target),
              "retained_frequency_ratio": ratio,
              "removed_target_frequency_drop": before - removed.probabilities[target],
              "optimization_score": plan["optimization_score"], "optimization_scores": optimization_scores,
              "optimization_diagnostics": explanation.diagnostics, "preprocessing": preprocessing,
              "requests": recorded.requests, "request_bound": plan["total_requests_bound"],
              "retry_count": reliable.retry_count,
              "elapsed_seconds": time.monotonic() - started, "usage": _usage(output / "requests.jsonl"),
              "figures": figures, "evaluation": "independent held-out class samples; fixed target",
              "optimization_figures": optimization_figures,
              "step_visuals": {"index": str(output / "index.html"), "metrics": str(output / "metrics.json"),
                               "steps": optimizer.steps + 1},
              "calibrated_correctness": False}
    write_json(output / "result.json", result)
    visualizer.finalize(result)
    return result
