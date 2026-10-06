"""Finite-query shearlet masks with explicit author-code and API choices."""

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Callable

import numpy as np
from PIL import Image

from .scoring import Scorer, ScoreResult
from .transforms import ImageTransform


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def unbiased_squared_error(target_count: int, samples: int, reference_score: float) -> float:
    """Estimate (p - reference_score)^2 from independent Bernoulli responses.

    The pair-count term estimates p^2 without the finite-sampling variance in
    (k/n)^2. Individual estimates can be negative for a reference between 0
    and 1; clipping them would reintroduce bias.
    """
    if type(samples) is not int or samples < 2:
        raise ValueError("unbiased distortion requires at least two independent samples")
    if type(target_count) is not int or not 0 <= target_count <= samples:
        raise ValueError("target_count must be an integer between zero and samples")
    if not np.isfinite(reference_score) or not 0 <= reference_score <= 1:
        raise ValueError("reference_score must be finite and in [0,1]")
    return float(reference_score ** 2 - 2 * reference_score * target_count / samples
                 + target_count * (target_count - 1) / (samples * (samples - 1)))


def clipped_spsa_gradient(plus_value, minus_value, plus_mask, minus_mask):
    """Use each coordinate's actual probe displacement at box boundaries.

    Dividing by a nominal 2*radius halves the expected gradient at an all-one
    mask because projection makes that displacement only radius. Independent
    Rademacher directions cancel cross-coordinate terms for a linear function.
    """
    displacement = plus_mask - minus_mask
    return np.divide(plus_value - minus_value, displacement,
                     out=np.zeros_like(displacement), where=displacement != 0)


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
    noise_workers: int = 1
    max_requests: int = 500
    seed: int = 42
    fidelity_reference: str = "original"
    noise_channels: str = "rgb"
    resample_noise: bool = False
    spatial_domain: str = "rgb_raw"
    normalize_final: bool = False
    unbiased_sampling_distortion: bool = False
    optimizer: str = "spsa"
    directions: int = 1
    mask_resolution: str = "grid"
    obfuscation_coefficient_clip: bool = False
    mask_selection: str = "best"
    fidelity_loss: str = "squared_error"

    def __post_init__(self):
        for name in ("steps", "grid_size", "noise_samples", "max_requests", "directions"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "steps" else 1):
                raise ValueError(f"{name} must be an integer in its valid range")
        if type(self.noise_workers) is not int or not 1 <= self.noise_workers <= 16:
            raise ValueError("noise_workers must be an integer in [1,16]")
        if self.noise not in {"uniform", "gaussian", "zeros"}:
            raise ValueError("noise must be uniform, gaussian or zeros")
        for name, choices in (("fidelity_reference", {"original", "one"}),
                              ("noise_channels", {"rgb", "shared_gray"}),
                              ("spatial_domain", {"rgb_raw", "gray_clipped"}),
                              ("mask_resolution", {"grid", "full"}),
                              ("mask_selection", {"best", "last"}),
                              ("fidelity_loss", {"squared_error", "one_minus_score"}),
                              ("optimizer", {"spsa", "hybrid_adam"})):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        for name in ("resample_noise", "normalize_final", "unbiased_sampling_distortion", "obfuscation_coefficient_clip"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if not np.isfinite(self.mask_init) or not 0 <= self.mask_init <= 1:
            raise ValueError("mask_init must be in [0,1]")
        for name in ("learning_rate", "perturbation_size", "distortion_weight", "mask_weight", "spatial_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.perturbation_size == 0:
            raise ValueError("learning_rate and perturbation_size must be positive")

    def request_bound(self, repeats=1):
        # Best selection reevaluates its candidate on the step's common noise
        # batch. Author-code last selection does not need that extra objective.
        probes = 2 * self.directions + 1 + int(self.resample_noise and self.mask_selection == "best")
        return repeats * (3 + self.noise_samples * (1 + probes * self.steps))


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
    """SPSA on full or grouped shearlet masks, shared across RGB channels.

    The VLM is queried with quantized PNG images. Coefficient noise is common
    within each finite-difference pair; optional paper settings resample that
    batch each step and normalize the final kept image before scoring it.
    """

    def __init__(self, scorer: Scorer, transform: ImageTransform, config=None):
        self.scorer = scorer
        self.transform = transform
        self.config = config or ExplainerConfig()
        if self.config.fidelity_reference == "one" and scorer.mode == "log_margin":
            raise ValueError("fidelity_reference='one' requires probability or agreement")
        if self.config.fidelity_loss == "one_minus_score" and scorer.mode == "log_margin":
            raise ValueError("one_minus_score requires probability or agreement")
        if self.config.unbiased_sampling_distortion and (
            scorer.mode != "agreement" or
            (self.config.fidelity_loss == "squared_error" and scorer.repeats < 2)
        ):
            raise ValueError("unbiased_sampling_distortion requires agreement with repeats >=2")
        if self.config.noise_workers > 1 and getattr(scorer.backend, "thread_safe", True) is False:
            raise ValueError("noise_workers >1 requires a thread-safe backend")
        if self.config.optimizer == "hybrid_adam" and not callable(getattr(transform, "decode_adjoint", None)):
            raise ValueError("hybrid_adam requires transform.decode_adjoint")

    def _expand_mask(self, mask, y_index=None, x_index=None):
        """Add the shared RGB axis; full masks retain every coefficient entry."""
        if self.config.mask_resolution == "full":
            return mask[None]
        return mask[:, y_index[:, None], x_index[None, :]][None]

    def _regularizer_gradient(self, mask, coeffs, y_index=None, x_index=None):
        """Exact full/grid-mask gradient of the local regularization penalties."""
        _, bands, height, width = coeffs.shape
        if self.config.mask_resolution == "full":
            if mask.shape != (bands, height, width):
                raise ValueError("full mask must have shape K,H,W matching the coefficients")
            # Match torch.abs's zero subgradient for the author-code full mask.
            gradient = self.config.mask_weight * np.sign(mask) / (bands * height * width)
        else:
            grid = self.config.grid_size
            y_index = np.arange(height) * grid // height if y_index is None else y_index
            x_index = np.arange(width) * grid // width if x_index is None else x_index
            cells = (y_index[:, None] * grid + x_index[None, :]).ravel()
            areas = np.bincount(cells, minlength=grid * grid).reshape(grid, grid)
            gradient = np.broadcast_to(
                self.config.mask_weight * areas / (bands * height * width), mask.shape,
            ).copy()
        if self.config.spatial_weight == 0:
            return gradient
        dense = self._expand_mask(mask, y_index, x_index)
        clean = self.transform.decode(coeffs * dense)
        if self.config.spatial_domain == "gray_clipped":
            gray = clean.mean(axis=-1)
            upper_active = gray <= 1 if self.config.mask_resolution == "full" else gray < 1
            pixel_gradient = np.broadcast_to(
                ((gray > 0) & upper_active)[:, :, None] / clean.size, clean.shape,
            )
        else:
            pixel_gradient = np.sign(clean) / clean.size
        coefficient_gradient = np.asarray(self.transform.decode_adjoint(pixel_gradient))
        if coefficient_gradient.shape != coeffs.shape or not np.isfinite(coefficient_gradient).all():
            raise ValueError("decode_adjoint must return finite C,K,H,W coefficients")
        dense_gradient = np.sum(coeffs * coefficient_gradient, axis=0)
        if self.config.mask_resolution == "full":
            spatial_gradient = dense_gradient
        else:
            spatial_gradient = np.stack([
                np.bincount(cells, weights=band.ravel(), minlength=grid * grid).reshape(grid, grid)
                for band in dense_gradient
            ])
        return gradient + self.config.spatial_weight * spatial_gradient

    def prepare_image(self, image, *, pixels=None):
        """Keep optional resized float pixels, validating their API image first."""
        image = image.convert("RGB")
        x = np.asarray(image, dtype=np.float64) / 255 if pixels is None else np.asarray(pixels, dtype=np.float64)
        if (x.shape != (image.height, image.width, 3) or not np.isfinite(x).all()
                or np.any((x < 0) | (x > 1))):
            raise ValueError("input pixels must be finite H,W,3 values in [0,1] matching the image dimensions")
        if to_image(x).tobytes() != image.tobytes():
            raise ValueError("Float input pixels belong to a different API input image")
        coeffs = self.transform.encode(x)
        if coeffs.ndim != 4 or not np.isfinite(coeffs).all():
            raise ValueError("Transform must produce finite C,K,H,W coefficients")
        if self.config.mask_resolution == "grid" and self.config.grid_size > min(coeffs.shape[-2:]):
            raise ValueError("grid_size cannot exceed image dimensions")
        round_trip = self.transform.decode(coeffs)
        if not np.allclose(round_trip, x, atol=1e-6, rtol=1e-6):
            raise ValueError("Transform does not reconstruct the input within tolerance")
        return x, coeffs, float(np.max(np.abs(round_trip - x)))

    def explain(self, image: Image.Image, target=None, reference=None, representation=None,
                progress: Callable[[dict], None] | None = None,
                checkpoint: Callable[[np.ndarray, list[dict]], None] | None = None,
                on_iteration: Callable[[np.ndarray, dict], None] | None = None) -> Explanation:
        """Explain an image; checkpoints follow selection, previews use current masks."""
        cfg = self.config
        repeats = self.scorer.repeats if self.scorer.mode == "agreement" else 1
        bound = cfg.request_bound(repeats)
        if bound > cfg.max_requests:
            raise ValueError(f"Run needs at most {bound} requests; max_requests={cfg.max_requests}")
        image = image.convert("RGB")
        x, coeffs, round_trip_error = representation or self.prepare_image(image)
        if (np.shape(x) != (image.height, image.width, 3) or not np.isfinite(x).all()
                or np.any((x < 0) | (x > 1))
                or to_image(x).tobytes() != image.tobytes()):
            raise ValueError("Prepared representation belongs to a different input image")
        _, bands, height, width = coeffs.shape
        requests = 0
        request_lock = Lock()

        def evaluate(array):
            nonlocal requests
            with request_lock:
                if requests + repeats > cfg.max_requests:
                    raise ValueError("Query budget exhausted")
                requests += repeats
            result = self.scorer.evaluate(to_image(array))
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
        fidelity_score = 1.0 if cfg.fidelity_reference == "one" or cfg.fidelity_loss == "one_minus_score" else reference_score
        rng = np.random.default_rng(cfg.seed)
        noise_coeffs = coeffs.mean(axis=0, keepdims=True) if cfg.noise_channels == "shared_gray" else coeffs
        mean = noise_coeffs.mean(axis=(-2, -1), keepdims=True)
        # Upstream torch.std uses the sample standard deviation. Preserve the
        # existing RGB profile's population std for backward compatibility.
        std = noise_coeffs.std(axis=(-2, -1), keepdims=True,
                               ddof=1 if cfg.noise_channels == "shared_gray" else 0)
        shape = (cfg.noise_samples, *noise_coeffs.shape)

        def draw_noise():
            if cfg.noise == "zeros":
                return np.zeros(shape)
            sample = rng.uniform(-1, 1, shape) if cfg.noise == "uniform" else rng.normal(size=shape)
            sample *= std
            sample += mean
            return sample

        noise = draw_noise()
        y_index = np.arange(height) * cfg.grid_size // height
        x_index = np.arange(width) * cfg.grid_size // width

        def expand(mask):
            return self._expand_mask(mask, y_index, x_index)

        def objective(mask):
            dense = expand(mask)
            clean = self.transform.decode(coeffs * dense)
            def sample_distortion(sample):
                mixed_coefficients = coeffs * dense + (1 - dense) * sample
                if cfg.obfuscation_coefficient_clip:
                    mixed_coefficients = np.clip(mixed_coefficients, 0, 1)
                prediction = evaluate(self.transform.decode(mixed_coefficients))
                if cfg.unbiased_sampling_distortion or (cfg.fidelity_loss == "one_minus_score" and self.scorer.mode == "agreement"):
                    count = prediction.sample_counts.get(target)
                    samples = sum(prediction.sample_counts.values())
                    if count is None:
                        samples = repeats
                        count_float = prediction.probabilities[target] * samples
                        count = round(count_float)
                        if not np.isclose(count_float, count, atol=1e-8, rtol=0):
                            raise ValueError("sampling probability does not correspond to an integer count")
                    if cfg.fidelity_loss == "one_minus_score":
                        return 1.0 - count / samples
                    return unbiased_squared_error(count, samples, fidelity_score)
                if cfg.fidelity_loss == "one_minus_score":
                    return 1.0 - prediction.score(target)
                return (prediction.score(target) - fidelity_score) ** 2

            if cfg.noise_workers > 1:
                with ThreadPoolExecutor(max_workers=min(cfg.noise_workers, cfg.noise_samples)) as pool:
                    distortions = list(pool.map(sample_distortion, noise))
            else:
                distortions = [sample_distortion(sample) for sample in noise]
            distortion = np.mean(distortions)
            mask_energy = float(np.mean(np.abs(dense)))
            spatial = np.clip(clean.mean(axis=-1), 0, 1) if cfg.spatial_domain == "gray_clipped" else clean
            spatial_energy = float(np.mean(np.abs(spatial)))
            loss = cfg.distortion_weight * distortion + cfg.mask_weight * mask_energy + cfg.spatial_weight * spatial_energy
            return {"loss": float(loss), "distortion": float(distortion),
                    "mask_energy": mask_energy, "spatial_energy": spatial_energy}

        mask_shape = (bands, height, width) if cfg.mask_resolution == "full" else (bands, cfg.grid_size, cfg.grid_size)
        mask = np.full(mask_shape, cfg.mask_init, dtype=np.float64)
        adam_mean, adam_variance = np.zeros_like(mask), np.zeros_like(mask)
        best_mask = mask.copy()
        best_step = 0
        best = objective(mask)
        history = [dict(step=0, requests=requests, best_step=best_step, **best)]
        if checkpoint is not None:
            checkpoint(best_mask.copy(), list(history))
        if on_iteration is not None:
            on_iteration(mask.copy(), dict(history[-1]))
        for step in range(1, cfg.steps + 1):
            if cfg.resample_noise:
                noise = draw_noise()
            radius = cfg.perturbation_size / step ** 0.101
            gradient = np.zeros_like(mask)
            probe_distortions = []
            for _ in range(cfg.directions):
                direction = rng.choice([-1., 1.], size=mask.shape)
                plus_mask = np.clip(mask + radius * direction, 0, 1)
                minus_mask = np.clip(mask - radius * direction, 0, 1)
                plus = objective(plus_mask)
                minus = objective(minus_mask)
                probe_distortions.append([plus["distortion"], minus["distortion"]])
                if cfg.optimizer == "hybrid_adam":
                    gradient += cfg.distortion_weight * clipped_spsa_gradient(
                        plus["distortion"], minus["distortion"], plus_mask, minus_mask)
                else:
                    gradient += (plus["loss"] - minus["loss"]) / (2 * radius) * direction
            gradient /= cfg.directions
            fidelity_gradient_norm = float(np.linalg.norm(gradient))
            previous_mask = mask.copy()
            regularizer_gradient_norm = None
            if cfg.optimizer == "hybrid_adam":
                regularizer_gradient = self._regularizer_gradient(mask, coeffs, y_index, x_index)
                regularizer_gradient_norm = float(np.linalg.norm(regularizer_gradient))
                gradient += regularizer_gradient
                adam_mean = 0.9 * adam_mean + 0.1 * gradient
                adam_variance = 0.999 * adam_variance + 0.001 * gradient ** 2
                corrected_mean = adam_mean / (1 - 0.9 ** step)
                corrected_variance = adam_variance / (1 - 0.999 ** step)
                mask = np.clip(mask - cfg.learning_rate * corrected_mean / (np.sqrt(corrected_variance) + 1e-8), 0, 1)
            else:
                rate = cfg.learning_rate / step ** 0.602
                mask = np.clip(mask - rate * gradient, 0, 1)
            current = objective(mask)
            if cfg.mask_selection == "last":
                best, best_mask = current, mask.copy()
                best_step = step
            else:
                if cfg.resample_noise:
                    best = objective(best_mask)
                if current["loss"] < best["loss"]:
                    best, best_mask = current, mask.copy()
                    best_step = step
            record = dict(step=step, requests=requests, best_loss=best["loss"],
                          best_step=best_step, probe_distortions=probe_distortions,
                          estimated_gradient_norm=fidelity_gradient_norm,
                          regularizer_gradient_norm=regularizer_gradient_norm,
                          mean_mask_change=float(np.mean(np.abs(mask - previous_mask))), **current)
            history.append(record)
            if checkpoint is not None:
                checkpoint(best_mask.copy(), list(history))
            if on_iteration is not None:
                on_iteration(mask.copy(), dict(record))
            if progress is not None:
                progress(record)
        dense = expand(best_mask)
        kept = self.transform.decode(coeffs * dense)
        removed = self.transform.decode(coeffs * (1 - dense))
        displayed_kept = np.clip(kept, 0, 1)
        normalizer = float(displayed_kept.max())
        if cfg.normalize_final and normalizer > 0:
            displayed_kept = displayed_kept / normalizer
        retained_result = evaluate(displayed_kept)
        removed_result = evaluate(removed)
        before = reference.probabilities[target]
        return Explanation(
            image=to_image(displayed_kept), removed_image=to_image(removed), mask=best_mask,
            target=target, reference=reference, retained=retained_result,
            removed=removed_result, history=history,
            diagnostics={"transform": self.transform.name,
                         "optimization": ("full_" if cfg.mask_resolution == "full" else "grouped_") + cfg.optimizer,
                         "model_gradient": "black_box_spsa_estimate",
                         "model_score_evidence": "sampled_label_frequency" if self.scorer.mode == "agreement" else "native_candidate_logprobs",
                         "finite_difference_denominator": "actual_clipped_probe_displacement" if cfg.optimizer == "hybrid_adam" else "symmetric_nominal_radius",
                         "requests": requests, "request_bound": bound,
                         "mask_parameters": int(best_mask.size), "coefficient_shape": list(coeffs.shape),
                         "mask_resolution": cfg.mask_resolution, "mask_shape": list(best_mask.shape),
                         "mask_selection": cfg.mask_selection,
                         "obfuscation_coefficient_clip": cfg.obfuscation_coefficient_clip,
                         "fidelity_loss": cfg.fidelity_loss,
                         "perturbation_directions": cfg.directions, "selected_step": best_step,
                         "mask_energy": best["mask_energy"], "spatial_energy": best["spatial_energy"],
                         "probability_drop_removed": before - removed_result.probabilities[target],
                         "retained_probability_ratio": retained_result.probabilities[target] / before if before > 0 else None,
                         "retained_score_distortion": (retained_result.score(target) - reference_score) ** 2,
                         "evaluation": "clean_kept_and_removed_images; " + (
                             "resampled_common_noise_optimization" if cfg.resample_noise else "fixed_noise_optimization"),
                         "fidelity_reference": cfg.fidelity_reference,
                         "fidelity_reference_score": fidelity_score,
                         "noise_channels": cfg.noise_channels,
                         "noise_workers": cfg.noise_workers,
                         "spatial_energy_domain": "clipped_grayscale_reconstruction" if cfg.spatial_domain == "gray_clipped" else "unclipped_RGB_reconstruction",
                         "unbiased_sampling_distortion": cfg.unbiased_sampling_distortion,
                         "normalize_final": cfg.normalize_final,
                         "final_normalization_divisor": normalizer if cfg.normalize_final and normalizer > 0 else 1.0,
                         "retained_clipped_fraction": float(np.mean((kept < 0) | (kept > 1))),
                         "retained_raw_range": [float(kept.min()), float(kept.max())],
                         "removed_clipped_fraction": float(np.mean((removed < 0) | (removed > 1))),
                         "removed_raw_range": [float(removed.min()), float(removed.max())],
                         "round_trip_max_error": round_trip_error})
