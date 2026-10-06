import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from medshearletx.data import ImageDataset, ImageSample


class ImageDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_image(self, relative_path, mode="RGB", size=(4, 2)):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, size).save(path)
        return path

    def write_manifest(self, rows, headers=("image_path", "label", "sample_id", "patient_group")):
        path = self.root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def test_folder_is_recursive_sorted_and_filters_nonimages(self):
        self.write_image("z.JPG")
        self.write_image("nested/a.png")
        self.write_image("a.webp")
        (self.root / "notes.txt").write_text("example", encoding="utf-8")
        dataset = ImageDataset.from_folder(self.root)
        self.assertEqual([sample.sample_id for sample in dataset], ["a.webp", "nested/a.png", "z.JPG"])
        self.assertEqual(len(dataset[:2]), 2)
        self.assertEqual(len(ImageDataset.from_folder(self.root, recursive=False)), 2)
        self.assertIsNone(dataset[0].label)

    def test_manifest_resolves_relative_paths_and_preserves_optional_fields(self):
        self.write_image("nested/a.png", mode="L")
        self.write_image("b.png")
        manifest = self.write_manifest([
            ("nested/a.png", "abnormal", "case-1", "external"),
            ("b.png", "", "", "validation"),
        ])
        dataset = ImageDataset.from_manifest(manifest)
        self.assertEqual(dataset[0].sample_id, "case-1")
        self.assertEqual(dataset[0].image_path, (self.root / "nested/a.png").resolve())
        self.assertEqual(dataset[0].label, "abnormal")
        self.assertEqual(dataset[0].metadata, {"patient_group": "external"})
        self.assertEqual(dataset[1].sample_id, "b.png")
        self.assertIsNone(dataset[1].label)
        image = dataset.load(dataset[0])
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.info["original_mode"], "L")
        self.assertEqual(image.info["original_size"], (4, 2))
        self.assertEqual(image.info["sample_metadata"], {"patient_group": "external"})

    def test_exif_orientation_is_applied_before_rgb_conversion(self):
        path = self.root / "oriented.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (4, 2)).save(path, exif=exif)
        dataset = ImageDataset.from_folder(self.root)
        image = dataset.load(dataset[0])
        self.assertEqual(image.size, (2, 4))
        self.assertEqual(image.info["original_size"], (4, 2))
        self.assertNotIn(274, image.getexif())
        path.unlink()
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))

    def test_duplicate_ids_and_image_paths_fail_early(self):
        first = self.write_image("a.png")
        second = self.write_image("b.png")
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            ImageDataset([ImageSample("same", first), ImageSample("same", second)])
        with self.assertRaisesRegex(ValueError, "duplicate image path"):
            ImageDataset([ImageSample("a", first), ImageSample("b", first)])

    def test_missing_files_and_malformed_manifests_fail_early(self):
        with self.assertRaises(FileNotFoundError):
            ImageDataset.from_manifest(self.write_manifest([("missing.png", "", "", "")]))
        with self.assertRaisesRegex(ValueError, "image_path column"):
            ImageDataset.from_manifest(self.write_manifest([], headers=("path",)))
        with self.assertRaisesRegex(ValueError, "malformed CSV"):
            ImageDataset.from_manifest(self.write_manifest([("a.png",)], headers=("image_path", "label")))

    def test_image_path_only_manifest_is_supported(self):
        self.write_image("a.png")
        dataset = ImageDataset.from_manifest(self.write_manifest([("a.png",)], headers=("image_path",)))
        self.assertEqual(dataset[0].sample_id, "a.png")
        self.assertIsNone(dataset[0].label)

    def test_high_bit_png_and_tiff_require_explicit_windowing(self):
        pixels = np.array([[0, 128, 256, 1024, 4095, 65535]], dtype=np.uint16)
        for extension in (".png", ".tiff"):
            with self.subTest(extension=extension):
                path = self.root / f"high-bit{extension}"
                Image.fromarray(pixels).save(path)
                sample = ImageSample(extension, path)
                dataset = ImageDataset([sample])
                with self.assertRaisesRegex(ValueError, "explicit medical windowing"):
                    dataset.load(sample)

    def test_integer_and_float_tiff_require_explicit_windowing(self):
        for mode in ("I", "F"):
            with self.subTest(mode=mode):
                path = self.root / f"mode-{mode}.tiff"
                Image.new(mode, (4, 4)).save(path)
                sample = ImageSample(mode, path)
                dataset = ImageDataset([sample])
                with self.assertRaisesRegex(ValueError, "explicit medical windowing"):
                    dataset.load(sample)

    def test_multiframe_tiff_requires_explicit_frame_selection(self):
        path = self.root / "multiframe.tiff"
        first = Image.new("RGB", (4, 4), "black")
        second = Image.new("RGB", (4, 4), "white")
        first.save(path, save_all=True, append_images=[second])
        sample = ImageSample("multiple-frames", path)
        dataset = ImageDataset([sample])
        with self.assertRaisesRegex(ValueError, "explicit frame selection"):
            dataset.load(sample)


if __name__ == "__main__":
    unittest.main()
