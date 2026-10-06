import importlib.util
import math
from itertools import product
from threading import Barrier, Lock
import unittest

import numpy as np
from PIL import Image

from medshearletx.backends import MockBackend
from medshearletx.explainer import BlackBoxShearletX, ExplainerConfig, clipped_spsa_gradient, to_image, unbiased_squared_error
from medshearletx.scoring import Scorer
from medshearletx.transforms import IdentityTransform, ShearletTransform
from medshearletx.types import ClassificationTask, Prediction


class ConstantBackend:
    thread_safe = True

    def predict(self, image, task, **kwargs):
        return Prediction(label_logprobs={"bright": math.log(0.8), "dark": math.log(0.2)},
                          sampled_label="bright")


class ExplainerTests(unittest.TestCase):
    def test_iteration_preview_uses_current_mask_when_checkpoint_keeps_earlier_best(self):
        class CandidateFailure(ConstantBackend):
            calls = 0

            def predict(self, *args, **kwargs):
                self.calls += 1
                probability = 0.1 if self.calls == 5 else 0.8
                return Prediction(label_logprobs={"bright": math.log(probability),
                                                 "dark": math.log(1 - probability)})

        scorer = Scorer(CandidateFailure(), ClassificationTask(("bright", "dark"), "Is it bright?"))
        config = ExplainerConfig(steps=1, grid_size=2, mask_init=1, optimizer="hybrid_adam",
                                mask_weight=1, spatial_weight=0)
        previews, checkpoints = [], []
        result = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(
            Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)),
            on_iteration=lambda mask, row: previews.append((mask, row)),
            checkpoint=lambda mask, history: checkpoints.append(mask),
        )
        self.assertEqual([row["step"] for _, row in previews], [0, 1])
        self.assertAlmostEqual(previews[-1][0].mean(), previews[-1][1]["mask_energy"])
        self.assertLess(previews[-1][0].mean(), checkpoints[-1].mean())
        np.testing.assert_array_equal(result.mask, checkpoints[-1])
        self.assertEqual(result.diagnostics["selected_step"], 0)

    def test_last_selection_checkpoints_current_mask_and_skips_best_reevaluation(self):
        class CandidateFailure(ConstantBackend):
            calls = 0

            def predict(self, *args, **kwargs):
                self.calls += 1
                probability = 0.1 if self.calls == 5 else 0.8
                return Prediction(label_logprobs={"bright": math.log(probability),
                                                 "dark": math.log(1 - probability)})

        results = {}
        for policy in ("best", "last"):
            scorer = Scorer(CandidateFailure(), self.scorer.task)
            config = ExplainerConfig(steps=1, grid_size=2, mask_init=1, optimizer="hybrid_adam",
                                    mask_weight=1, spatial_weight=0, resample_noise=True,
                                    mask_selection=policy)
            previews, checkpoints = [], []
            result = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(
                self.image, on_iteration=lambda mask, row: previews.append(mask),
                checkpoint=lambda mask, history: checkpoints.append(mask))
            self.assertEqual(scorer.requests, config.request_bound())
            np.testing.assert_array_equal(result.mask, checkpoints[-1])
            if policy == "last":
                np.testing.assert_array_equal(result.mask, previews[-1])
                self.assertEqual(result.diagnostics["selected_step"], 1)
                self.assertGreater(result.history[-1]["loss"], result.history[0]["loss"])
            else:
                np.testing.assert_array_equal(result.mask, checkpoints[0])
                self.assertEqual(result.diagnostics["selected_step"], 0)
            results[policy] = result
        self.assertLess(results["last"].mask.mean(), results["best"].mask.mean())
        self.assertEqual(results["best"].diagnostics["requests"] - results["last"].diagnostics["requests"], 1)

    def test_full_mask_keeps_independent_pixel_entries_without_block_expansion(self):
        full = BlackBoxShearletX(self.scorer, IdentityTransform(),
                                ExplainerConfig(mask_resolution="full", grid_size=2))
        coefficients = np.ones((3, 1, 5, 7))
        mask = np.zeros((1, 5, 7))
        mask[0, 2, 3] = 1
        reconstruction = full.transform.decode(coefficients * full._expand_mask(mask))
        self.assertEqual(np.count_nonzero(reconstruction[:, :, 0]), 1)
        np.testing.assert_array_equal(reconstruction[2, 3], np.ones(3))
        grid = BlackBoxShearletX(self.scorer, IdentityTransform(), ExplainerConfig(grid_size=2))
        coarse = np.zeros((1, 2, 2))
        coarse[0, 0, 0] = 1
        indices = np.arange(5) * 2 // 5, np.arange(7) * 2 // 7
        coarse_image = grid.transform.decode(coefficients * grid._expand_mask(coarse, *indices))
        self.assertEqual(np.count_nonzero(coarse_image[:, :, 0]), 12)

    def test_full_geometry_and_last_multiple_direction_budget(self):
        scorer = Scorer(ConstantBackend(), self.scorer.task, mode="agreement", repeats=2)
        config = ExplainerConfig(steps=2, grid_size=999, mask_resolution="full", directions=2,
                                noise_samples=2, resample_noise=True, mask_selection="last",
                                optimizer="hybrid_adam", max_requests=100)
        expected = 2 * (3 + 2 * (1 + 5 * 2))
        self.assertEqual(config.request_bound(2), expected)
        result = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(self.image)
        self.assertEqual(result.mask.shape, (1, 32, 32))
        self.assertEqual(result.diagnostics["mask_parameters"], 1024)
        self.assertEqual(result.diagnostics["mask_shape"], [1, 32, 32])
        self.assertEqual(result.diagnostics["mask_resolution"], "full")
        self.assertEqual(result.diagnostics["optimization"], "full_hybrid_adam")
        self.assertEqual(result.diagnostics["model_gradient"], "black_box_spsa_estimate")
        self.assertEqual(result.diagnostics["model_score_evidence"], "sampled_label_frequency")
        self.assertEqual(result.diagnostics["selected_step"], config.steps)
        self.assertEqual(scorer.requests, expected)

    def test_float_preprocessing_pixels_survive_until_api_quantization(self):
        pixels = np.asarray(self.image, dtype=np.float64) / 255 + 0.001
        explainer = BlackBoxShearletX(self.scorer, IdentityTransform(),
                                     ExplainerConfig(steps=0, mask_resolution="full"))
        representation = explainer.prepare_image(self.image, pixels=pixels)
        np.testing.assert_array_equal(representation[0], pixels)
        self.assertFalse(np.array_equal(representation[0], np.asarray(self.image) / 255))
        self.assertEqual(to_image(pixels).tobytes(), self.image.tobytes())
        result = explainer.explain(self.image, representation=representation)
        self.assertEqual(result.diagnostics["round_trip_max_error"], 0)
        requests_before = self.scorer.requests
        with self.assertRaisesRegex(ValueError, "different API"):
            explainer.prepare_image(self.image, pixels=pixels + 0.01)
        with self.assertRaisesRegex(ValueError, "different input"):
            explainer.explain(self.image, representation=(pixels + 0.01, *representation[1:]))
        for bad in (np.full(pixels.shape, np.nan), pixels[:, :, :1], np.ones_like(pixels) * 2):
            with self.assertRaisesRegex(ValueError, "input pixels"):
                explainer.prepare_image(self.image, pixels=bad)
        self.assertEqual(self.scorer.requests, requests_before)

    def test_obfuscation_coefficients_are_clipped_before_synthesis_only(self):
        class SignedTransform(IdentityTransform):
            def encode(self, image):
                channels = np.asarray(image).transpose(2, 0, 1)
                return np.stack((2 * channels, -channels), axis=1)

            def decode(self, coefficients):
                return coefficients.sum(axis=1).transpose(1, 2, 0)

        class RecordingScorer(Scorer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.images = []

            def evaluate(self, image):
                self.images.append(np.asarray(image).copy())
                return super().evaluate(image)

        image = Image.fromarray(np.tile(np.array([51, 204], dtype=np.uint8)[None, :, None], (4, 1, 3)))
        for clipped in (False, True):
            scorer = RecordingScorer(ConstantBackend(), self.scorer.task)
            config = ExplainerConfig(steps=0, mask_resolution="full", mask_init=1, noise="zeros",
                                    obfuscation_coefficient_clip=clipped)
            result = BlackBoxShearletX(scorer, SignedTransform(), config).explain(image)
            expected = np.tile(np.array([102, 255] if clipped else [51, 204], dtype=np.uint8)[None, :, None], (4, 1, 3))
            np.testing.assert_array_equal(scorer.images[1], expected)
            np.testing.assert_array_equal(np.asarray(result.image), np.asarray(image))
            self.assertIs(result.diagnostics["obfuscation_coefficient_clip"], clipped)

    def test_one_minus_sample_frequency_is_unbiased_even_with_one_sample(self):
        for samples in (1, 2, 4):
            estimates = []
            for count in range(samples + 1):
                class CountBackend:
                    def __init__(self):
                        self.calls = 0

                    def predict(self, *args, **kwargs):
                        label = "bright" if self.calls % samples < count else "dark"
                        self.calls += 1
                        return Prediction(sampled_label=label)

                scorer = Scorer(CountBackend(), self.scorer.task, mode="agreement", repeats=samples)
                config = ExplainerConfig(steps=0, fidelity_loss="one_minus_score",
                                        unbiased_sampling_distortion=True)
                result = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(self.image, target="bright")
                estimates.append(result.history[0]["distortion"])
                self.assertAlmostEqual(estimates[-1], 1 - count / samples)
                self.assertEqual(result.diagnostics["fidelity_reference_score"], 1)
            for probability in (0, 0.2, 0.6, 1):
                expectation = sum(math.comb(samples, k) * probability ** k * (1 - probability) ** (samples-k)
                                  * estimates[k] for k in range(samples + 1))
                self.assertAlmostEqual(expectation, 1 - probability)
        native = Scorer(ConstantBackend(), self.scorer.task)
        result = BlackBoxShearletX(native, IdentityTransform(),
                                  ExplainerConfig(steps=0, fidelity_loss="one_minus_score")).explain(self.image)
        self.assertAlmostEqual(result.history[0]["distortion"], 0.2)

    def test_multiple_directions_budget_counts_each_probe_and_rejects_before_calls(self):
        scorer = Scorer(ConstantBackend(), ClassificationTask(("bright", "dark"), "Is it bright?"),
                        mode="agreement", repeats=4)
        config = ExplainerConfig(steps=2, grid_size=2, directions=2, noise_samples=2,
                                resample_noise=True, optimizer="hybrid_adam", max_requests=200)
        expected = 4 * (3 + 2 * (1 + 6 * 2))
        self.assertEqual(config.request_bound(4), expected)
        result = BlackBoxShearletX(scorer, IdentityTransform(), config).explain(
            Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)))
        self.assertEqual(scorer.requests, expected)
        self.assertEqual(result.diagnostics["perturbation_directions"], 2)
        self.assertEqual(len(result.history[-1]["probe_distortions"]), 2)
        for invalid in (0, True, 1.5):
            with self.assertRaises(ValueError):
                ExplainerConfig(directions=invalid)
        refused = Scorer(ConstantBackend(), scorer.task, mode="agreement", repeats=4)
        with self.assertRaisesRegex(ValueError, "requests"):
            BlackBoxShearletX(refused, IdentityTransform(),
                             ExplainerConfig(steps=2, directions=2, noise_samples=2,
                                             resample_noise=True, max_requests=expected-1)).explain(
                Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)))
        self.assertEqual(refused.requests, 0)

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
                        {"noise_workers": 0}, {"noise_workers": 17}, {"noise_workers": True},
                        {"mask_resolution": "bilinear"}, {"mask_selection": "bad"},
                        {"fidelity_loss": "bad"}, {"obfuscation_coefficient_clip": 1}):
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

    def test_full_regularizer_gradient_matches_pixelwise_finite_difference(self):
        rng = np.random.default_rng(10)
        coefficients = rng.normal(size=(3, 1, 5, 7))
        mask = rng.uniform(0.3, 0.8, (1, 5, 7))
        for domain in ("rgb_raw", "gray_clipped"):
            config = ExplainerConfig(mask_resolution="full", optimizer="hybrid_adam", mask_weight=0.7,
                                    spatial_weight=1.2, spatial_domain=domain)
            explainer = BlackBoxShearletX(self.scorer, IdentityTransform(), config)
            actual = explainer._regularizer_gradient(mask, coefficients)

            def regularizer(candidate):
                clean = explainer.transform.decode(coefficients * candidate[None])
                spatial = np.clip(clean.mean(axis=-1), 0, 1) if domain == "gray_clipped" else clean
                return config.mask_weight * np.mean(np.abs(candidate)) + config.spatial_weight * np.mean(np.abs(spatial))

            expected = np.empty_like(mask)
            for index in np.ndindex(mask.shape):
                plus, minus = mask.copy(), mask.copy()
                plus[index] += 1e-6
                minus[index] -= 1e-6
                expected[index] = (regularizer(plus) - regularizer(minus)) / 2e-6
            np.testing.assert_allclose(actual, expected, atol=1e-9)

    def test_full_regularizer_matches_author_subgradients_at_zero_and_clip_boundary(self):
        mask = np.array([[[0., 0.5, 1., 1., 1.]]])
        coefficients = np.tile(np.array([1., 1., 1., 2., -1.])[None, None, None], (3, 1, 1, 1))
        for mask_weight, spatial_weight, expected in (
                (1, 0, [0., 0.2, 0.2, 0.2, 0.2]),
                (0, 1, [0., 0.2, 0.2, 0., 0.])):
            config = ExplainerConfig(mask_resolution="full", optimizer="hybrid_adam", spatial_domain="gray_clipped",
                                    mask_weight=mask_weight, spatial_weight=spatial_weight)
            gradient = BlackBoxShearletX(self.scorer, IdentityTransform(), config)._regularizer_gradient(mask, coefficients)
            np.testing.assert_allclose(gradient.ravel(), expected, atol=1e-15)

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
        config = ExplainerConfig(mask_resolution="full", optimizer="hybrid_adam", mask_weight=0.7,
                                spatial_weight=1.2, spatial_domain="gray_clipped")
        explainer = BlackBoxShearletX(self.scorer, transform, config)
        rng = np.random.default_rng(27)
        mask = rng.uniform(0.3, 0.7, coefficients.shape[1:])
        direction = rng.normal(size=mask.shape)
        direction /= np.linalg.norm(direction)
        gradient = explainer._regularizer_gradient(mask, coefficients)

        def regularizer(candidate):
            clean = transform.decode(coefficients * candidate[None])
            return 0.7 * np.mean(np.abs(candidate)) + 1.2 * np.mean(np.clip(clean.mean(axis=-1), 0, 1))

        epsilon = 1e-3
        expected = (regularizer(mask + epsilon * direction) - regularizer(mask - epsilon * direction)) / (2 * epsilon)
        self.assertAlmostEqual(float(np.sum(gradient * direction)), expected, places=9)
        self.assertIs(utility.dfilters, original_filter)
        with self.assertRaisesRegex(ValueError, "square"):
            transform.encode(np.zeros((128, 64, 3)))


if __name__ == "__main__":
    unittest.main()
