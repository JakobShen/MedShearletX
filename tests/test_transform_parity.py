"""Independent upstream FFT formulas on the actual 49-band Afghan example.

These references translate code/shearletx.py, not PyShearLab's decomposition or
reconstruction functions. NumPy checks the formula at native precision; SciPy's
dtype-preserving FFTs model the upstream float32 filter/weight casts. No Torch,
model weights, API calls, or new dependencies are required.
"""

from hashlib import sha256
import importlib.util
import math
from pathlib import Path
import platform
import unittest

import numpy as np
from PIL import Image

from medshearletx.explainer import BlackBoxShearletX, ExplainerConfig, to_image
from medshearletx.runner import preprocess_pixels
from medshearletx.scoring import Scorer
from medshearletx.transforms import ShearletTransform
from medshearletx.types import ClassificationTask, Prediction


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "code" / "imgs" / "ILSVRC2012_val_00017625.JPEG"
EXTRAS_AVAILABLE = (importlib.util.find_spec("pyshearlab") is not None
                    and importlib.util.find_spec("scipy") is not None)


def author_analysis(image, real_filters, fft=np.fft):
    """Literal author BH W -> BH WK FFT operations, then return KHW."""
    image = image[None]
    filters = real_filters[None]
    frequency = fft.ifftshift(fft.fft2(fft.fftshift(image)))
    product = np.einsum("bij,bijk->bijk", frequency, filters)
    # Preserve all-axis shifts, including the odd 49-band axis. They cancel
    # around an inverse FFT on the author's explicit spatial axes (-3,2).
    coefficients = fft.ifftshift(fft.ifft2(fft.fftshift(product), axes=(-3, 2)))
    return coefficients.real[0].transpose(2, 0, 1)


def author_synthesis(coefficients, real_filters, dual_weights, fft=np.fft):
    """Literal author BK HW synthesis with dual-frame weights, returning HW."""
    coefficients = coefficients[None]
    frequency = fft.fftshift(fft.fft2(fft.ifftshift(coefficients)))
    image_frequency = np.sum(frequency * real_filters[None].transpose(0, 3, 1, 2), axis=1)
    image = fft.fftshift(fft.ifft2(fft.ifftshift(image_frequency / dual_weights)))
    return image.real[0]


def _rgb_analysis(pixels, filters, fft=np.fft):
    return np.stack([author_analysis(pixels[..., channel], filters, fft) for channel in range(3)])


def _rgb_synthesis(coefficients, filters, weights, fft=np.fft):
    return np.stack([author_synthesis(channel, filters, weights, fft) for channel in coefficients], axis=-1)


def _display(pixels):
    clipped = np.clip(pixels, 0, 1)
    maximum = float(clipped.max())
    return clipped / maximum if maximum > 0 else clipped


def _png_pixels(pixels):
    return np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)


def _difference(actual, reference):
    errors = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {"max_abs_error": float(np.max(np.abs(errors))),
            "root_mean_square_error": float(np.sqrt(np.mean(errors * errors)))}


class _CaptureBackend:
    """Only capture local PNGs; fixed fake probabilities are not model evidence."""

    thread_safe = True

    def __init__(self):
        self.images = []

    def predict(self, image, task, *, require_logprobs, temperature=1.0):
        self.images.append(np.asarray(image).copy())
        return Prediction(label_logprobs={task.labels[0]: math.log(0.75), task.labels[1]: math.log(0.25)})


def build_parity_evidence():
    """Return small diagnostics plus arrays used by independent assertions."""
    import scipy
    from scipy import fft as float32_fft

    with Image.open(IMAGE_PATH) as original:
        pixels32 = preprocess_pixels(original, {
            "image_size": 256, "resize_mode": "stretch", "resize_filter": "tensor_bilinear",
        })
    pixels64 = pixels32.astype(np.float64)  # Production prepare_image promotion.
    transform = ShearletTransform(scales=4)
    native_coefficients = transform.encode(pixels64)
    filters = transform.system["shearlets"]
    real64 = filters.real
    weights64 = transform.system["dualFrameWeights"]
    real32, weights32 = real64.astype(np.float32), weights64.astype(np.float32)
    author64 = _rgb_analysis(pixels64, real64)
    author32 = _rgb_analysis(pixels32, real32, float32_fft)
    # A spatially varying, band-varying mask detects permutations that a round
    # trip with mask=1 could conceal. Same mask is shared across RGB channels.
    mask = np.random.default_rng(4629).random((49, 256, 256), dtype=np.float32)
    native_kept = transform.decode(native_coefficients * mask[None])
    author_kept64 = _rgb_synthesis(author64 * mask[None], real64, weights64)
    author_kept32 = _rgb_synthesis(author32 * mask[None], real32, weights32, float32_fft)
    gray64 = author_analysis(pixels64.mean(axis=-1), real64)
    gray32 = author_analysis(pixels32.mean(axis=-1), real32, float32_fft)
    native_gray = native_coefficients.mean(axis=0)
    native_display, author_display64, author_display32 = map(_display, (native_kept, author_kept64, author_kept32))
    png_native, png_author32 = _png_pixels(native_display), _png_pixels(author_display32)
    png_error = np.abs(png_native.astype(int) - png_author32.astype(int))

    # Exercise the real explainer's pre-inverse clipping and final display path
    # with a local capture backend, independently constructing the expected FFT
    # reconstruction. No prediction is sent to a provider.
    backend = _CaptureBackend()
    config = ExplainerConfig(steps=0, mask_resolution="full", mask_selection="last", mask_init=0.63,
                            noise="uniform", noise_samples=1, noise_channels="shared_gray", seed=913,
                            spatial_domain="gray_clipped", obfuscation_coefficient_clip=True,
                            normalize_final=True, fidelity_reference="one")
    scorer = Scorer(backend, ClassificationTask(("target", "other"), "Local capture only."))
    explainer = BlackBoxShearletX(scorer, transform, config)
    image = to_image(pixels64)
    explanation = explainer.explain(image, "target", representation=(pixels64, native_coefficients, 0.0))
    noise = np.random.default_rng(config.seed).uniform(-1, 1, (49, 256, 256))
    noise = noise * gray64.std(axis=(-2, -1), keepdims=True, ddof=1) + gray64.mean(axis=(-2, -1), keepdims=True)
    mixed = config.mask_init * author64 + (1 - config.mask_init) * noise[None]
    expected_obfuscated = _rgb_synthesis(np.clip(mixed, 0, 1), real64, weights64)
    wrong_clip_position = _rgb_synthesis(mixed, real64, weights64)
    expected_final = _display(_rgb_synthesis(config.mask_init * author64, real64, weights64))
    expected_removed = _rgb_synthesis((1 - config.mask_init) * author64, real64, weights64)
    expected_spatial = np.clip(author_synthesis(config.mask_init * gray64, real64, weights64), 0, 1)
    captured = {
        "obfuscated": backend.images[1], "final": np.asarray(explanation.image),
        "removed": np.asarray(explanation.removed_image),
        "expected_obfuscated": _png_pixels(expected_obfuscated),
        "expected_final": _png_pixels(expected_final), "expected_removed": _png_pixels(expected_removed),
        "wrong_clip_position": _png_pixels(wrong_clip_position),
    }
    comparisons = {
        "analysis_rgb_float64_formula": _difference(native_coefficients, author64),
        "analysis_rgb_author_float32": _difference(native_coefficients, author32),
        "random_full_mask_synthesis_float64_formula": _difference(native_kept, author_kept64),
        "random_full_mask_synthesis_author_float32": _difference(native_kept, author_kept32),
        "gray_analysis_equals_rgb_mean_float64": _difference(native_gray, gray64),
        "gray_analysis_author_float32": _difference(native_gray, gray32),
        # FFT outputs are transposed views. NumPy's float32 reduction order on
        # that strided layout is not Torch/CUDA's reduction implementation;
        # retain it as a numerical diagnostic and isolate accumulation error
        # by changing only layout or accumulator precision.
        "shared_gray_mean_float32_strided_reduction": _difference(native_gray.mean(axis=(-2, -1)),
                                                                   gray32.mean(axis=(-2, -1))),
        "shared_gray_mean_float32_contiguous_reduction": _difference(native_gray.mean(axis=(-2, -1)),
                                                                      np.ascontiguousarray(gray32).mean(axis=(-2, -1))),
        "shared_gray_mean_float32_float64_accumulator": _difference(native_gray.mean(axis=(-2, -1)),
                                                                    gray32.mean(axis=(-2, -1), dtype=np.float64)),
        "shared_gray_sample_std_author_float32": _difference(native_gray.std(axis=(-2, -1), ddof=1),
                                                                  gray32.std(axis=(-2, -1), ddof=1)),
        "final_clip_max_float64_formula": _difference(native_display, author_display64),
        "final_clip_max_author_float32": _difference(native_display, author_display32),
        "core_obfuscation_png": _difference(captured["obfuscated"], captured["expected_obfuscated"]),
        "core_final_clip_max_png": _difference(captured["final"], captured["expected_final"]),
        "core_removed_png": _difference(captured["removed"], captured["expected_removed"]),
        "core_spatial_penalty_mean": _difference(explanation.diagnostics["spatial_energy"], expected_spatial.mean()),
    }
    evidence = {
        "status": "pass",
        "scope": "Actual default 4-scale 256x256 transform and local rendering/obfuscation paths; no optimizer or model-feature claim.",
        "reference": "Independent literal NumPy/SciPy translations of torchsheardec2D/torchshearrec2D in code/shearletx.py.",
        "upstream_source_sha256": sha256((ROOT / "code" / "shearletx.py").read_bytes()).hexdigest(),
        "transform_source_sha256": sha256((ROOT / "medshearletx" / "transforms.py").read_bytes()).hexdigest(),
        "input_sha256": sha256(IMAGE_PATH.read_bytes()).hexdigest(),
        "input": "code/imgs/ILSVRC2012_val_00017625.JPEG", "input_float_dtype": str(pixels32.dtype),
        "preprocessing": "256 stretch tensor bilinear, align_corners=False, antialias=False; float input retained",
        "coefficient_shape": list(native_coefficients.shape), "mask_shape": list(mask.shape),
        "mask_parameters": int(mask.size), "mask_seed": 4629,
        "filter_dtype": str(filters.dtype), "filter_max_abs_imaginary": float(np.max(np.abs(filters.imag))),
        "filter_max_abs_real": float(np.max(np.abs(filters.real))),
        "filter_conjugation_difference": float(np.max(np.abs(filters - np.conj(filters)))),
        "float64_absolute_tolerance": 5e-11, "author_float32_absolute_tolerance": 3e-6,
        "float32_strided_mean_absolute_tolerance": 1e-5,
        "comparisons": comparisons,
        "author_float32_png_max_lsb_difference": int(png_error.max()),
        "author_float32_png_changed_value_fraction": float(np.mean(png_error != 0)),
        "clip_before_inverse_vs_after_inverse_changed_value_fraction": float(np.mean(
            captured["expected_obfuscated"] != captured["wrong_clip_position"])),
        "vlm_calls": 0, "offline_capture_predictions": len(backend.images),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "limits": ["NumPy/SciPy float32 arithmetic and reduction order are not a CUDA FFT bitwise reproduction; the strided mean discrepancy disappears with contiguous layout or float64 accumulation.",
                   "Even 256 spatial dimensions and this effectively real default filter bank are tested; custom complex or odd-size banks are outside scope.",
                   "Transform parity does not establish optimization convergence or explain Gemini's reliance on any image feature."],
    }
    return evidence, captured


@unittest.skipUnless(EXTRAS_AVAILABLE, "optional shearlet/SciPy extras missing")
class AuthorTransformParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence, cls.captured = build_parity_evidence()

    def test_real_default_filters_and_full_unpooled_mask_shape(self):
        self.assertEqual(self.evidence["coefficient_shape"], [3, 49, 256, 256])
        self.assertEqual(self.evidence["mask_shape"], [49, 256, 256])
        self.assertEqual(self.evidence["mask_parameters"], 3211264)
        self.assertEqual(self.evidence["input_float_dtype"], "float32")
        self.assertLess(self.evidence["filter_max_abs_imaginary"], 1e-12)

    def test_float64_fft_formulas_match_elementwise_analysis_and_random_mask_synthesis(self):
        for name in ("analysis_rgb_float64_formula", "random_full_mask_synthesis_float64_formula",
                     "gray_analysis_equals_rgb_mean_float64", "final_clip_max_float64_formula"):
            with self.subTest(stage=name):
                self.assertLess(self.evidence["comparisons"][name]["max_abs_error"], 5e-11)

    def test_upstream_float32_casts_preserve_rgb_gray_noise_statistics_and_final_display(self):
        for name in ("analysis_rgb_author_float32", "random_full_mask_synthesis_author_float32",
                     "gray_analysis_author_float32", "shared_gray_mean_float32_float64_accumulator",
                     "shared_gray_sample_std_author_float32", "final_clip_max_author_float32"):
            with self.subTest(stage=name):
                self.assertLess(self.evidence["comparisons"][name]["max_abs_error"], 3e-6)
        self.assertLessEqual(self.evidence["author_float32_png_max_lsb_difference"], 1)
        self.assertLess(self.evidence["author_float32_png_changed_value_fraction"], 0.001)

    def test_float32_mean_discrepancy_is_reduction_layout_not_transformed_coefficients(self):
        comparisons = self.evidence["comparisons"]
        self.assertLess(comparisons["shared_gray_mean_float32_strided_reduction"]["max_abs_error"], 1e-5)
        for name in ("shared_gray_mean_float32_contiguous_reduction",
                     "shared_gray_mean_float32_float64_accumulator"):
            with self.subTest(reduction=name):
                self.assertLess(comparisons[name]["max_abs_error"], 1e-8)

    def test_real_core_clips_mixed_coefficients_before_inverse_not_only_png(self):
        np.testing.assert_array_equal(self.captured["obfuscated"], self.captured["expected_obfuscated"])
        self.assertGreater(self.evidence["clip_before_inverse_vs_after_inverse_changed_value_fraction"], 0.05)

    def test_real_core_final_clip_max_and_gray_spatial_penalty_match_author_path(self):
        np.testing.assert_array_equal(self.captured["final"], self.captured["expected_final"])
        np.testing.assert_array_equal(self.captured["removed"], self.captured["expected_removed"])
        self.assertLess(self.evidence["comparisons"]["core_spatial_penalty_mean"]["max_abs_error"], 5e-11)
        self.assertEqual(self.evidence["vlm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
