import importlib.util
import math
from itertools import product
from threading import Barrier, Lock
import unittest

import numpy as np
from PIL import Image

from medshearletx.backends import MockBackend
from medshearletx.explainer import BlackBoxShearletX, ExplainerConfig, clipped_spsa_gradient, unbiased_squared_error
from medshearletx.scoring import Scorer
from medshearletx.transforms import IdentityTransform, ShearletTransform
from medshearletx.types import ClassificationTask, Prediction


class ConstantBackend:
    thread_safe = True

    def predict(self, image, task, **kwargs):
        return Prediction(label_logprobs={"bright": math.log(0.8), "dark": math.log(0.2)},
                          sampled_label="bright")


class ExplainerTests(unittest.TestCase):
    def test_checkpoint_preserves_best_mask_before_later_provider_failure(self):
        class FailingBackend(ConstantBackend):
            calls = 0

            def predict(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 6:
                    raise RuntimeError("transient test failure")
                return super().predict(*args, **kwargs)

        scorer = Scorer(FailingBackend(), ClassificationTask(("bright", "dark"), "Is it bright?"))
        saved = []
        explainer = BlackBoxShearletX(scorer, IdentityTransform(), ExplainerConfig(steps=3))
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            explainer.explain(Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)),
                              checkpoint=lambda mask, history: saved.append((mask, history)))
        self.assertEqual([history[-1]["step"] for _, history in saved], [0, 1])
        self.assertTrue(np.isfinite(saved[-1][0]).all())
        self.assertTrue(((saved[-1][0] >= 0) & (saved[-1][0] <= 1)).all())

    def test_probability_maximization_reference_rejects_log_margin_before_calls(self):
        scorer = Scorer(MockBackend(), ClassificationTask(("bright", "dark"), "Is it bright?"),
                        mode="log_margin")
        with self.assertRaisesRegex(ValueError, "requires probability or agreement"):
            BlackBoxShearletX(scorer, IdentityTransform(), ExplainerConfig(fidelity_reference="one"))
        self.assertEqual(scorer.requests, 0)

    def test_projected_probes_preserve_linear_gradient_in_expectation(self):
        coefficients = np.array([0.7, -1.3])
        for mask in (np.array([1., 1.]), np.array([0., 0.]), np.array([1., 0.4])):
            gradients = []
            for signs in product((-1., 1.), repeat=2):
                direction = np.array(signs)
                plus = np.clip(mask + 0.1 * direction, 0, 1)
                minus = np.clip(mask - 0.1 * direction, 0, 1)
                gradients.append(clipped_spsa_gradient(coefficients @ plus,
                                                      coefficients @ minus, plus, minus))
            np.testing.assert_allclose(np.mean(gradients, axis=0), coefficients, atol=1e-12)

    def setUp(self):
        self.image = Image.fromarray(np.full((32, 32, 3), 220, dtype=np.uint8))
        self.scorer = Scorer(MockBackend(seed=8), ClassificationTask(("bright", "dark"), "Is it bright?"))

    def test_budget_refuses_before_model_call(self):
        explainer = BlackBoxShearletX(self.scorer, IdentityTransform(), ExplainerConfig(steps=3, max_requests=4))
        with self.assertRaisesRegex(ValueError, "requests"):
            explainer.explain(self.image)
        self.assertEqual(self.scorer.requests, 0)

    def test_finite_bounded_mask_fixed_target_and_request_accounting(self):
        cfg = ExplainerConfig(steps=2, noise_samples=2, seed=12)
        result = BlackBoxShearletX(self.scorer, IdentityTransform(), cfg).explain(self.image, "dark")
        self.assertEqual(result.target, "dark")
        self.assertEqual(result.diagnostics["requests"], cfg.request_bound())
        self.assertEqual(self.scorer.requests, cfg.request_bound())
        self.assertTrue(np.isfinite(result.mask).all())
        self.assertTrue(np.all((result.mask >= 0) & (result.mask <= 1)))
        self.assertLessEqual(result.history[-1]["best_loss"], result.history[0]["loss"])
        self.assertEqual(result.image.size, self.image.size)
        self.assertEqual(result.diagnostics["transform"], "identity")

    def test_shared_reference_saves_call_and_keeps_score_semantics(self):
        reference = self.scorer.evaluate(self.image)
        margin = Scorer(self.scorer.backend, self.scorer.task, mode="log_margin")
        cfg = ExplainerConfig(steps=0)
        result = BlackBoxShearletX(margin, IdentityTransform(), cfg).explain(self.image, reference=reference)
        self.assertEqual(result.reference.mode, "log_margin")
        self.assertEqual(margin.requests, cfg.request_bound() - 1)

    def test_agreement_budget_includes_every_repeat(self):
        scorer = Scorer(self.scorer.backend, self.scorer.task, mode="agreement", repeats=3)
        cfg = ExplainerConfig(steps=1, seed=12)
        result = BlackBoxShearletX(scorer, IdentityTransform(), cfg).explain(self.image)
        self.assertEqual(result.diagnostics["requests"], cfg.request_bound(3))
        self.assertEqual(scorer.requests, cfg.request_bound(3))

    def test_unbiased_distortion_has_correct_binomial_expectation(self):
        for samples in (2, 3, 16):
            for probability in (0, 0.2, 0.6, 1):
                for reference in (0, 0.4, 1):
                    expectation = sum(
                        math.comb(samples, count) * probability ** count
                        * (1 - probability) ** (samples - count)
                        * unbiased_squared_error(count, samples, reference)
                        for count in range(samples + 1)
                    )
                    self.assertAlmostEqual(expectation, (probability - reference) ** 2, places=12)
        self.assertAlmostEqual(unbiased_squared_error(3, 5, 1), 2 * 1 / (5 * 4))
        with self.assertRaisesRegex(ValueError, "two"):
            unbiased_squared_error(1, 1, 1)

    def test_profile_settings_and_unbiased_mode_fail_before_model_calls(self):
        for options in ({"fidelity_reference": "bad"}, {"noise_channels": "bad"},
                        {"spatial_domain": "bad"}, {"resample_noise": 1},
                        {"normalize_final": 1}, {"unbiased_sampling_distortion": 1},
                        {"noise_workers": 0}, {"noise_workers": 17}, {"noise_workers": True}):
            with self.assertRaises(ValueError):
                ExplainerConfig(**options)
        with self.assertRaisesRegex(ValueError, "agreement"):
            BlackBoxShearletX(self.scorer, IdentityTransform(), ExplainerConfig(unbiased_sampling_distortion=True))
        scorer = Scorer(self.scorer.backend, self.scorer.task, mode="agreement", repeats=1)
        with self.assertRaisesRegex(ValueError, "repeats"):
            BlackBoxShearletX(scorer, IdentityTransform(), ExplainerConfig(unbiased_sampling_distortion=True))
        self.assertEqual(self.scorer.requests, 0)

    def test_one_reference_matches_upstream_maximization(self):
        for setting, expected in (("original", 0), ("one", 0.04)):
            scorer = Scorer(ConstantBackend(), self.scorer.task)
            config = ExplainerConfig(steps=0, fidelity_reference=setting)
            explanation = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(self.image)
            self.assertAlmostEqual(explanation.history[0]["distortion"], expected)

    def test_shared_gray_noise_uses_gray_statistics_and_shared_rgb_samples(self):
        class RecordingTransform(IdentityTransform):
            def __init__(self):
                self.decoded = []

            def decode(self, coefficients):
                self.decoded.append(coefficients.copy())
                return super().decode(coefficients)

        pixels = np.random.default_rng(9).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        transform = RecordingTransform()
        config = ExplainerConfig(steps=0, mask_init=0, noise_channels="shared_gray", seed=5)
        BlackBoxShearletX(self.scorer, transform, config).explain(Image.fromarray(pixels))
        obfuscated = transform.decoded[2]
        for channel in (1, 2):
            np.testing.assert_array_equal(obfuscated[0], obfuscated[channel])
        gray = pixels.mean(axis=-1) / 255
        expected = np.random.default_rng(5).uniform(-1, 1, (1, 1, 1, 32, 32))
        expected = expected * gray.std(ddof=1) + gray.mean()
        np.testing.assert_allclose(obfuscated[0], expected[0, 0], atol=1e-15)

    def test_resampled_profile_budget_and_final_scored_pixels(self):
        class RecordingScorer(Scorer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.images = []

            def evaluate(self, image):
                self.images.append(image.copy())
                return super().evaluate(image)

        scorer = RecordingScorer(ConstantBackend(), self.scorer.task, mode="agreement", repeats=3)
        config = ExplainerConfig(steps=2, noise_samples=2, mask_init=0.5,
                                fidelity_reference="one", noise_channels="shared_gray",
                                resample_noise=True, spatial_domain="gray_clipped",
                                normalize_final=True, unbiased_sampling_distortion=True)
        explanation = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(self.image)
        expected_requests = 3 * (3 + 2 * (1 + 4 * 2))
        self.assertEqual(config.request_bound(3), expected_requests)
        self.assertEqual(explanation.diagnostics["requests"], expected_requests)
        self.assertEqual(scorer.requests, expected_requests)
        self.assertEqual(scorer.images[-2].tobytes(), explanation.image.tobytes())
        self.assertEqual(np.asarray(explanation.image).max(), 255)
        self.assertEqual(explanation.diagnostics["fidelity_reference_score"], 1)
        self.assertEqual(explanation.diagnostics["spatial_energy_domain"], "clipped_grayscale_reconstruction")

    def test_hybrid_exact_regularizer_gradient_matches_finite_difference(self):
        coefficients = np.random.default_rng(10).normal(size=(3, 1, 5, 7))
        mask = np.array([[[0.3, 0.55], [0.65, 0.8]]])
        y_index, x_index = np.arange(5) * 2 // 5, np.arange(7) * 2 // 7
        for domain in ("rgb_raw", "gray_clipped"):
            config = ExplainerConfig(grid_size=2, optimizer="hybrid_adam", mask_weight=0.7,
                                    spatial_weight=1.2, spatial_domain=domain)
            explainer = BlackBoxShearletX(self.scorer, IdentityTransform(), config)
            actual = explainer._regularizer_gradient(mask, coefficients, y_index, x_index)

            def regularizer(candidate):
                dense = candidate[:, y_index[:, None], x_index[None, :]][None]
                clean = explainer.transform.decode(coefficients * dense)
                spatial = np.clip(clean.mean(axis=-1), 0, 1) if domain == "gray_clipped" else clean
                return config.mask_weight * np.mean(np.abs(dense)) + config.spatial_weight * np.mean(np.abs(spatial))

            expected = np.empty_like(mask)
            for index in np.ndindex(mask.shape):
                plus, minus = mask.copy(), mask.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                expected[index] = (regularizer(plus) - regularizer(minus)) / 2e-6
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_hybrid_adam_changes_mask_and_requires_adjoint_before_calls(self):
        scorer = Scorer(ConstantBackend(), self.scorer.task)
        config = ExplainerConfig(steps=2, grid_size=2, mask_init=1, optimizer="hybrid_adam",
                                mask_weight=1, spatial_weight=2, normalize_final=True)
        explanation = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(self.image)
        self.assertLess(explanation.mask.mean(), 0.9)
        self.assertTrue(np.all((explanation.mask >= 0) & (explanation.mask <= 1)))
        self.assertEqual(explanation.diagnostics["optimization"], "grouped_hybrid_adam")
        self.assertEqual(scorer.requests, config.request_bound())
        no_adjoint = IdentityTransform()
        no_adjoint.decode_adjoint = None
        scorer = Scorer(ConstantBackend(), self.scorer.task)
        with self.assertRaisesRegex(ValueError, "decode_adjoint"):
            BlackBoxShearletX(scorer, no_adjoint, config)
        self.assertEqual(scorer.requests, 0)

    def test_parallel_noise_queries_and_thread_safe_backend_guard(self):
        class ParallelBackend(ConstantBackend):
            def __init__(self):
                self.calls = 0
                self.lock = Lock()
                self.barrier = Barrier(4)

            def predict(self, image, task, **kwargs):
                with self.lock:
                    index = self.calls
                    self.calls += 1
                if 1 <= index <= 4:
                    self.barrier.wait(timeout=3)
                return super().predict(image, task, **kwargs)

        backend = ParallelBackend()
        scorer = Scorer(backend, self.scorer.task)
        config = ExplainerConfig(steps=0, noise_samples=4, noise_workers=4)
        result = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(self.image)
        self.assertEqual(backend.calls, config.request_bound())
        self.assertEqual(result.diagnostics["requests"], config.request_bound())
        backend.thread_safe = False
        with self.assertRaisesRegex(ValueError, "thread-safe"):
            BlackBoxShearletX(scorer, IdentityTransform(), config)

    @unittest.skipUnless(importlib.util.find_spec("pyshearlab"), "optional shearlet extra missing")
    def test_real_shearlet_signed_coefficients_and_round_trip(self):
        import pyshearlab.pySLUtilities as utility
        original_filter = utility.dfilters
        pixels = np.random.default_rng(4).random((128, 128, 3))
        transform = ShearletTransform(scales=2)
        coefficients = transform.encode(pixels)
        self.assertEqual(coefficients.shape[:2], (3, 17))
        self.assertTrue(np.any(coefficients < 0))
        np.testing.assert_allclose(transform.decode(coefficients), pixels, atol=1e-10)
        self.assertIs(utility.dfilters, original_filter)
        with self.assertRaisesRegex(ValueError, "square"):
            transform.encode(np.zeros((128, 64, 3)))


if __name__ == "__main__":
    unittest.main()
