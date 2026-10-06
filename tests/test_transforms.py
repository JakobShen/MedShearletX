"""Adjoint identities verify analytic image regularizer gradients."""

import importlib.util
import unittest

import numpy as np

from medshearletx.transforms import IdentityTransform, ShearletTransform


class IdentityAdjointTests(unittest.TestCase):
    def test_synthesis_adjoint_dot_product_and_signed_values(self):
        random = np.random.default_rng(22)
        transform = IdentityTransform()
        coefficients = random.normal(size=(3, 1, 13, 19))
        image_gradient = random.normal(size=(13, 19, 3))
        adjoint = transform.decode_adjoint(image_gradient)
        self.assertEqual(adjoint.shape, coefficients.shape)
        self.assertTrue(np.any(adjoint < 0))
        np.testing.assert_allclose(
            np.vdot(transform.decode(coefficients), image_gradient),
            np.vdot(coefficients, adjoint), rtol=1e-13, atol=1e-12,
        )


@unittest.skipUnless(importlib.util.find_spec("pyshearlab"), "optional shearlet extra missing")
class ShearletAdjointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transform = ShearletTransform(scales=2)
        cls.coefficient_shape = cls.transform.encode(np.zeros((128, 128, 3), dtype=np.float64)).shape

    def test_synthesis_adjoint_on_arbitrary_coefficients(self):
        # A frame round trip alone would not detect confusing analysis with the
        # adjoint of synthesis. These coefficients are outside the image range.
        random = np.random.default_rng(36)
        coefficients = random.normal(size=self.coefficient_shape)
        image_gradient = random.normal(size=(128, 128, 3))
        synthesized = self.transform.decode(coefficients)
        adjoint = self.transform.decode_adjoint(image_gradient)
        self.assertEqual(adjoint.shape, coefficients.shape)
        self.assertTrue(np.any(coefficients < 0))
        self.assertTrue(np.any(adjoint < 0))
        np.testing.assert_allclose(
            np.vdot(synthesized, image_gradient), np.vdot(coefficients, adjoint),
            rtol=1e-11, atol=1e-9,
        )
        # The non-tight redundant frame requires its dual weights; ordinary
        # encode(image_gradient) does not provide the correct synthesis adjoint.
        self.assertGreater(np.linalg.norm(adjoint - self.transform.encode(image_gradient)), 1.0)

    def test_synthesized_energy_gradient_matches_finite_difference(self):
        random = np.random.default_rng(52)
        coefficients = random.normal(size=self.coefficient_shape)
        direction = random.normal(size=self.coefficient_shape)
        direction /= np.linalg.norm(direction)
        reconstructed = self.transform.decode(coefficients)
        derivative = np.vdot(self.transform.decode_adjoint(reconstructed), direction)
        # This energy is quadratic, so a larger central step has no truncation
        # error and avoids cancellation when subtracting large image energies.
        epsilon = 0.1
        plus = self.transform.decode(coefficients + epsilon * direction)
        minus = self.transform.decode(coefficients - epsilon * direction)
        finite_difference = (0.5 * np.vdot(plus, plus) - 0.5 * np.vdot(minus, minus)) / (2 * epsilon)
        np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7, atol=1e-8)

    def test_adjoint_requires_initialized_frame_and_matching_image_shape(self):
        with self.assertRaisesRegex(ValueError, "encode"):
            ShearletTransform(scales=2).decode_adjoint(np.zeros((128, 128, 3)))
        for shape in [(128, 128), (64, 128, 3), (128, 64, 3)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "HWC"):
                self.transform.decode_adjoint(np.zeros(shape))


if __name__ == "__main__":
    unittest.main()
