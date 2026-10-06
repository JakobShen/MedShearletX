"""Local image datasets with no model or tensor dependencies."""

import csv
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})


@dataclass(frozen=True)
class ImageSample:
    sample_id: str
    image_path: Path
    label: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a nonempty string")
        if self.label is not None and (not isinstance(self.label, str) or not self.label.strip()):
            raise ValueError("label must be a nonempty string or None")
        object.__setattr__(self, "image_path", Path(self.image_path).expanduser().resolve())
        object.__setattr__(self, "metadata", dict(self.metadata))


class ImageDataset(Sequence[ImageSample]):
    """An ordered sample manifest; image pixels are loaded only on request."""

    def __init__(self, samples: Sequence[ImageSample]) -> None:
        self._samples = tuple(samples)
        seen_ids = set()
        seen_paths = set()
        for sample in self._samples:
            if sample.sample_id in seen_ids:
                raise ValueError(f"duplicate sample_id: {sample.sample_id!r}")
            if sample.image_path in seen_paths:
                raise ValueError(f"duplicate image path: {sample.image_path}")
            if not sample.image_path.is_file():
                raise FileNotFoundError(f"image file does not exist: {sample.image_path}")
            if sample.image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"unsupported image extension: {sample.image_path}")
            seen_ids.add(sample.sample_id)
            seen_paths.add(sample.image_path)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int | slice) -> ImageSample | tuple[ImageSample, ...]:
        return self._samples[index]

    def __iter__(self) -> Iterator[ImageSample]:
        return iter(self._samples)

    @classmethod
    def from_folder(cls, root: str | Path, recursive: bool = True) -> "ImageDataset":
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"image folder does not exist: {root}")
        paths = root.rglob("*") if recursive else root.iterdir()
        images = sorted(
            (path for path in paths if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
            key=lambda path: path.relative_to(root).as_posix(),
        )
        return cls([
            ImageSample(sample_id=path.relative_to(root).as_posix(), image_path=path)
            for path in images
        ])

    @classmethod
    def from_manifest(cls, path: str | Path) -> "ImageDataset":
        """Read CSV columns image_path, optional label/sample_id, and metadata."""
        path = Path(path).expanduser().resolve()
        samples = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "image_path" not in reader.fieldnames:
                raise ValueError("CSV manifest requires an image_path column")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("CSV manifest contains duplicate column names")
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"malformed CSV manifest row {row_number}")
                image_path = row["image_path"].strip()
                if not image_path:
                    raise ValueError(f"missing image_path in CSV manifest row {row_number}")
                source_path = Path(image_path).expanduser()
                if not source_path.is_absolute():
                    source_path = path.parent / source_path
                samples.append(ImageSample(
                    sample_id=(row.get("sample_id") or "").strip() or image_path,
                    image_path=source_path,
                    label=(row.get("label") or "").strip() or None,
                    metadata={
                        key: value for key, value in row.items()
                        if key not in {"image_path", "label", "sample_id"}
                    },
                ))
        return cls(samples)

    def load(self, sample: ImageSample) -> Image.Image:
        """Return independent RGB pixels with EXIF orientation applied.

        The source mode/size and manifest metadata remain available in the
        returned image's ``info`` dictionary for audit records. High-bit,
        integer, floating-point, and multiframe inputs require explicit
        windowing or frame selection before they enter this RGB workflow.
        """
        with Image.open(sample.image_path) as source:
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError(
                    "Multiframe images require explicit frame selection before loading: "
                    f"{sample.image_path}"
                )
            if source.mode in {"I", "F"} or source.mode.startswith("I;16"):
                raise ValueError(
                    f"Image mode {source.mode!r} requires explicit medical windowing "
                    f"to an 8-bit image before loading: {sample.image_path}"
                )
            original_size, original_mode = source.size, source.mode
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
        image.info.update({
            "original_size": original_size,
            "original_mode": original_mode,
            "source_path": str(sample.image_path),
            "sample_id": sample.sample_id,
            "sample_metadata": dict(sample.metadata),
        })
        return image
