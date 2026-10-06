import importlib.util
import unittest

import numpy as np
from PIL import Image

from medshearletx.backends import MockBackend
from medshearletx.explainer import BlackBoxShearletX, ExplainerConfig
from medshearletx.scoring import Scorer
from medshearletx.transforms import IdentityTransform, ShearletTransform
from medshearletx.types import ClassificationTask


class ExplainerTests(unittest.TestCase):
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
