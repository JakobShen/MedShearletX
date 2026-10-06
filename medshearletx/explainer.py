"""Finite-query, grouped-mask adaptation of the original distortion objective."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image

from .scoring import Scorer, ScoreResult
from .transforms import ImageTransform


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


@dataclass(frozen=True)
class ExplainerConfig:
    steps: int = 20
    grid_size: int = 4
    learning_rate: float = 0.1
    perturbation_size: float = 0.1
    mask_init: float = 0.9
    distortion_weight: float = 1.0
    mask_weight: float = 0.05
    spatial_weight: float = 0.05
    noise: str = "uniform"
    noise_samples: int = 1
    max_requests: int = 500
    seed: int = 42

    def __post_init__(self):
        for name in ("steps", "grid_size", "noise_samples", "max_requests"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "steps" else 1):
                raise ValueError(f"{name} must be an integer in its valid range")
        if self.noise not in {"uniform", "gaussian", "zeros"}:
            raise ValueError("noise must be uniform, gaussian or zeros")
        if not np.isfinite(self.mask_init) or not 0 <= self.mask_init <= 1:
            raise ValueError("mask_init must be in [0,1]")
        for name in ("learning_rate", "perturbation_size", "distortion_weight", "mask_weight", "spatial_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.perturbation_size == 0:
            raise ValueError("learning_rate and perturbation_size must be positive")

    def request_bound(self, repeats=1):
        # Reference, initial objective, three probes per step, clean kept/removed images.
        return repeats * (3 + self.noise_samples * (1 + 3 * self.steps))


@dataclass
class Explanation:
    image: Image.Image
    removed_image: Image.Image
    mask: np.ndarray
    target: str
    reference: ScoreResult
    retained: ScoreResult
    removed: ScoreResult
    history: list[dict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


class BlackBoxShearletX:
    """SPSA on a per-band spatial grid, shared across RGB channels.

    The VLM is queried with quantized PNG images. Common coefficient noise is
    fixed for optimization; final diagnostics use clean kept/removed images.
    """

    def __init__(self, scorer: Scorer, transform: ImageTransform, config=None):
        self.scorer = scorer
        self.transform = transform
        self.config = config or ExplainerConfig()

    def prepare_image(self, image):
        """Validate the local transform before making an external prediction."""
        x = np.asarray(image.convert("RGB"), dtype=np.float64) / 255
        coeffs = self.transform.encode(x)
        if coeffs.ndim != 4 or not np.isfinite(coeffs).all():
            raise ValueError("Transform must produce finite C,K,H,W coefficients")
        if self.config.grid_size > min(coeffs.shape[-2:]):
            raise ValueError("grid_size cannot exceed image dimensions")
        round_trip = self.transform.decode(coeffs)
        if not np.allclose(round_trip, x, atol=1e-6, rtol=1e-6):
            raise ValueError("Transform does not reconstruct the input within tolerance")
        return x, coeffs, float(np.max(np.abs(round_trip - x)))

    def explain(self, image: Image.Image, target=None, reference=None, representation=None,
                progress: Callable[[dict], None] | None = None) -> Explanation:
        cfg = self.config
        repeats = self.scorer.repeats if self.scorer.mode == "agreement" else 1
        bound = cfg.request_bound(repeats)
        if bound > cfg.max_requests:
            raise ValueError(f"Run needs at most {bound} requests; max_requests={cfg.max_requests}")
        image = image.convert("RGB")
        x, coeffs, round_trip_error = representation or self.prepare_image(image)
        if not np.array_equal(x, np.asarray(image, dtype=np.float64) / 255):
            raise ValueError("Prepared representation belongs to a different input image")
        _, bands, height, width = coeffs.shape
        requests = 0

        def evaluate(array):
            nonlocal requests
            if requests + repeats > cfg.max_requests:
                raise ValueError("Query budget exhausted")
            result = self.scorer.evaluate(to_image(array))
            requests += result.requests
            return result

        if reference is None:
            reference = evaluate(x)
        else:
            reference = reference.with_mode(self.scorer.mode)
        if target is None:
            target = max(reference.probabilities, key=reference.probabilities.get)
        if target not in reference.probabilities:
            raise ValueError(f"Unknown fixed target class: {target}")
        reference_score = reference.score(target)
        rng = np.random.default_rng(cfg.seed)
        mean = coeffs.mean(axis=(-2, -1), keepdims=True)
        std = coeffs.std(axis=(-2, -1), keepdims=True)
        shape = (cfg.noise_samples, *coeffs.shape)
        if cfg.noise == "uniform":
            noise = rng.uniform(-1, 1, shape) * std + mean
        elif cfg.noise == "gaussian":
            noise = rng.normal(size=shape) * std + mean
        else:
            noise = np.zeros(shape)
        y_index = np.arange(height) * cfg.grid_size // height
        x_index = np.arange(width) * cfg.grid_size // width

        def expand(mask):
            return mask[:, y_index[:, None], x_index[None, :]][None]

        def objective(mask):
            dense = expand(mask)
            clean = self.transform.decode(coeffs * dense)
            distortion = np.mean([
                (evaluate(self.transform.decode(coeffs * dense + (1 - dense) * sample)).score(target)
                 - reference_score) ** 2
                for sample in noise
            ])
            mask_energy = float(np.mean(np.abs(dense)))
            spatial_energy = float(np.mean(np.abs(clean)))
            loss = cfg.distortion_weight * distortion + cfg.mask_weight * mask_energy + cfg.spatial_weight * spatial_energy
            return {"loss": float(loss), "distortion": float(distortion),
                    "mask_energy": mask_energy, "spatial_energy": spatial_energy}

        mask = np.full((bands, cfg.grid_size, cfg.grid_size), cfg.mask_init)
        best_mask = mask.copy()
        best = objective(mask)
        history = [dict(step=0, requests=requests, **best)]
        for step in range(1, cfg.steps + 1):
            direction = rng.choice([-1., 1.], size=mask.shape)
            radius = cfg.perturbation_size / step ** 0.101
            plus = objective(np.clip(mask + radius * direction, 0, 1))
            minus = objective(np.clip(mask - radius * direction, 0, 1))
            gradient = (plus["loss"] - minus["loss"]) / (2 * radius) * direction
            rate = cfg.learning_rate / step ** 0.602
            mask = np.clip(mask - rate * gradient, 0, 1)
            current = objective(mask)
            if current["loss"] < best["loss"]:
                best, best_mask = current, mask.copy()
            record = dict(step=step, requests=requests, best_loss=best["loss"], **current)
            history.append(record)
            if progress is not None:
                progress(record)
        dense = expand(best_mask)
        kept = self.transform.decode(coeffs * dense)
        removed = self.transform.decode(coeffs * (1 - dense))
        retained_result = evaluate(kept)
        removed_result = evaluate(removed)
        before = reference.probabilities[target]
        return Explanation(
            image=to_image(kept), removed_image=to_image(removed), mask=best_mask,
            target=target, reference=reference, retained=retained_result,
            removed=removed_result, history=history,
            diagnostics={"transform": self.transform.name, "optimization": "grouped_spsa",
                         "requests": requests, "request_bound": bound,
                         "mask_parameters": int(best_mask.size), "coefficient_shape": list(coeffs.shape),
                         "mask_energy": best["mask_energy"], "spatial_energy": best["spatial_energy"],
                         "probability_drop_removed": before - removed_result.probabilities[target],
                         "retained_probability_ratio": retained_result.probabilities[target] / before if before > 0 else None,
                         "retained_score_distortion": (retained_result.score(target) - reference_score) ** 2,
                         "evaluation": "clean_kept_and_removed_images; fixed_noise_optimization",
                         "spatial_energy_domain": "unclipped_RGB_reconstruction",
                         "retained_clipped_fraction": float(np.mean((kept < 0) | (kept > 1))),
                         "retained_raw_range": [float(kept.min()), float(kept.max())],
                         "removed_clipped_fraction": float(np.mean((removed < 0) | (removed > 1))),
                         "removed_raw_range": [float(removed.min()), float(removed.max())],
                         "round_trip_max_error": round_trip_error})
