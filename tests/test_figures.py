import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from medshearletx.figures import save_explanation_figure


class FigureTests(unittest.TestCase):
    def test_actual_raster_and_pdf_exports_preserve_supplied_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            original = Image.new("RGB", (256, 256), (230, 230, 230))
            explanation = Image.new("RGB", (256, 256), (25, 50, 100))
            before = explanation.tobytes()
            files = save_explanation_figure(
                original, explanation, folder, model="gemini-3.5-flash-lite",
                target_label="English foxhound", retained_ratio=0.3729,
                metadata={"reference_count": 16, "reference_samples": 16,
                          "retained_count": 6, "retained_samples": 16,
                          "retained_interval": [0.18, 0.61]},
            )
            self.assertEqual(set(files), {"explanation_png", "explanation_pdf", "comparison_png", "comparison_pdf"})
            self.assertEqual(explanation.tobytes(), before)
            for name, path in files.items():
                self.assertTrue(Path(path).is_absolute())
                self.assertGreater(Path(path).stat().st_size, 1000)
                if name.endswith("pdf"):
                    self.assertTrue(Path(path).read_bytes().startswith(b"%PDF-"))
            with Image.open(files["explanation_png"]) as figure:
                self.assertGreaterEqual(figure.width, 1000)
                self.assertGreater(figure.height, figure.width)
                # A large area contains exactly the supplied explanation color.
                pixels = np.asarray(figure.convert("RGB"))
                self.assertGreater(np.mean(np.all(pixels == (25, 50, 100), axis=-1)), 0.5)
            with Image.open(files["comparison_png"]) as figure:
                self.assertGreater(figure.width, figure.height)

    def test_probability_requires_native_evidence_and_invalid_ratios_fail(self):
        image = Image.new("RGB", (32, 32))
        with tempfile.TemporaryDirectory() as folder:
            common = {"model": "test", "target_label": "dog", "retained_ratio": 1}
            with self.assertRaisesRegex(ValueError, "native_candidate"):
                save_explanation_figure(image, image, folder, score_label="probability", **common)
            for ratio in (-1, float("nan"), float("inf"), True):
                with self.assertRaisesRegex(ValueError, "retained_ratio"):
                    save_explanation_figure(image, image, folder, **{**common, "retained_ratio": ratio})
            files = save_explanation_figure(
                image, image, folder, **{**common, "retained_ratio": None}, save_pdf=False,
            )
            self.assertEqual(set(files), {"explanation_png", "comparison_png"})


if __name__ == "__main__":
    unittest.main()
