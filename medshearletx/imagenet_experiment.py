"""Reproducible 1000-way ImageNet experiment using independent core components."""

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
from .figures import save_explanation_figure
from .runner import describe_result, preprocess, write_json
from .scoring import Scorer
from .tasks import load_imagenet_task
from .transforms import create_transform


def prepare_experiment(config, root):
    """Resolve dataset/task paths relative to the explicit project root."""
    root = Path(root).resolve()
    task = load_imagenet_task(root / config["labels_path"])
    dataset = ImageDataset([ImageSample(
        sample_id="paper_english_foxhound", image_path=root / config["image_path"],
        metadata={"source": "arXiv:2211.12857v3, page 1, embedded original photo",
                  "paper_predicted_class": "English foxhound", "paper_class_index": 167},
    )])
    optimizer = ExplainerConfig(**config.get("explainer", {}))
    repeats = config.get("optimization_sampling", {}).get("repeats", 2)
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
    RetryingBackend(backend, **retry)  # Validate the remaining reliability options.
    Scorer(backend, task, mode="agreement", **config.get("optimization_sampling", {}))
    Scorer(backend, task, mode="agreement", repeats=evaluation,
           workers=config.get("evaluation_workers", 8))
    image, preprocessing = preprocess(dataset.load(dataset[0]), config)
    return task, dataset, optimizer, backend, image, preprocessing, {
        "classes": len(task.labels), "prompt_characters": len(task.prompt),
        "selection_samples": selection, "evaluation_samples_per_image": evaluation,
        "total_requests_bound": bound, "max_total_requests": maximum,
        "retry_budget": retry_budget,
        "optimization": asdict(optimizer), "preprocessing": preprocessing,
        "model": config["model"], "native_probability": "unavailable; sampled_label_frequency",
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
    except Exception as error:
        if (output / "plan.json").is_file():
            log = output / "requests.jsonl"
            records = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
            write_json(output / "failure.json", {
                "status": "error", "error_type": type(error).__name__,
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
    scoring = Scorer(reliable, task, mode="agreement", **config.get("optimization_sampling", {}))
    explainer = BlackBoxShearletX(scoring, create_transform(config["transform"]), optimizer)
    representation = explainer.prepare_image(image)  # Local preflight before any billable call.
    started = time.monotonic()
    selector = Scorer(reliable, task, mode="agreement", repeats=plan["selection_samples"],
                      workers=config.get("evaluation_workers", 8))
    progress("Selecting the fixed target from the full 1000-class task")
    selection = selector.evaluate(image)
    target = selection.predicted_label
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

    explanation = explainer.explain(image, target=target, reference=selection,
                                    representation=representation, progress=on_step,
                                    checkpoint=checkpoint)
    explanation.image.save(output / "retained.png")
    explanation.removed_image.save(output / "removed.png")
    np.save(output / "mask.npy", explanation.mask)
    write_json(output / "history.json", explanation.history)
    write_json(output / "optimization_diagnostics.json", explanation.diagnostics)
    progress("Evaluating clean retained and removed images with held-out samples")
    retained = evaluator.evaluate(explanation.image)
    write_json(output / "retained.json", describe_result(retained, target))
    removed = evaluator.evaluate(explanation.removed_image)
    write_json(output / "removed.json", describe_result(removed, target))
    before = reference.probabilities[target]
    ratio = retained.probabilities[target] / before if before else None
    metadata = {"evidence": "sampled_label_frequency",
                "approximation": "API adaptation: hybrid Adam/SPSA, grouped shearlet mask, finite sampling; no native logprobs."}
    for name, result in (("reference", reference), ("retained", retained)):
        metadata.update({f"{name}_count": result.sample_counts[target],
                         f"{name}_samples": result.requests,
                         f"{name}_interval": result.diagnostics["sample_wilson_95"][target]})
    figures = save_explanation_figure(image, explanation.image, output,
                                     model=config["model"]["model"], target_label=target,
                                     retained_ratio=ratio, metadata=metadata)
    result = {"status": "ok", "model": config["model"], "classes": len(task.labels),
              "target": target, "target_index": task.labels.index(target),
              "paper_predicted_class": dataset[0].metadata["paper_predicted_class"],
              "selection": describe_result(selection, target),
              "reference": describe_result(reference, target),
              "retained": describe_result(retained, target), "removed": describe_result(removed, target),
              "retained_frequency_ratio": ratio,
              "removed_target_frequency_drop": before - removed.probabilities[target],
              "optimization_diagnostics": explanation.diagnostics, "preprocessing": preprocessing,
              "requests": recorded.requests, "request_bound": plan["total_requests_bound"],
              "retry_count": reliable.retry_count,
              "elapsed_seconds": time.monotonic() - started, "usage": _usage(output / "requests.jsonl"),
              "figures": figures, "evaluation": "independent held-out class samples; fixed target",
              "calibrated_correctness": False}
    write_json(output / "result.json", result)
    return result
