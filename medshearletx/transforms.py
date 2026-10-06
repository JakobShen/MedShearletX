"""Invertible representations; only the shearlet option is ShearletX."""

from threading import RLock
from typing import Protocol

import numpy as np


_SYSTEM_LOCK = RLock()


class _FilterPair(tuple):
    """Preserve upstream filter-wise division for differently sized arrays."""

    def __truediv__(self, scalar):
        return tuple(value / scalar for value in self)


def _build_system(library, height, width, scales):
    # PyShearLab 0.0.1 divides a ragged tuple of filters by a NumPy scalar.
    # Adapt only those two module-local references during system construction;
    # do not alter NumPy or installed source. Restore even on upstream failure.
    import pyshearlab.pyShearLab2D as transform_module
    import pyshearlab.pySLUtilities as utilities_module

    with _SYSTEM_LOCK:
        modules = (transform_module, utilities_module)
        originals = [module.dfilters for module in modules]
        try:
            for module, original in zip(modules, originals):
                module.dfilters = lambda *args, _original=original: _FilterPair(_original(*args))
            return library.SLgetShearletSystem2D(0, height, width, scales)
        except SystemExit as exc:
            raise ValueError(str(exc)) from exc
        finally:
            for module, original in zip(modules, originals):
                module.dfilters = original


class ImageTransform(Protocol):
    """Representations consumed by the explainer.

    Transforms may optionally implement ``decode_adjoint(image_gradient)`` to
    map a real HWC image gradient back to channel/band/height/width coefficients.
    This is the adjoint of synthesis, which need not equal ``encode`` for a
    redundant frame. Gradient-based regularizers can use it when available.
    """

    name: str

    def encode(self, image: np.ndarray) -> np.ndarray:
        """HWC image -> channel, band, height, width coefficients."""

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """Coefficients -> HWC image, without clipping signed coefficients."""


class IdentityTransform:
    """Explicit pixel-space control for plumbing tests; never a shearlet fallback."""

    name = "identity"

    def encode(self, image):
        return np.asarray(image, dtype=np.float64).transpose(2, 0, 1)[:, None].copy()

    def decode(self, coefficients):
        return coefficients[:, 0].transpose(1, 2, 0).copy()

    def decode_adjoint(self, image_gradient):
        """Transpose the pixel synthesis map without clipping signed values."""
        return self.encode(image_gradient)


class ShearletTransform:
    """The same PyShearLab representation family used by the paper."""

    name = "shearlet"

    def __init__(self, scales=2):
        if not isinstance(scales, int) or scales < 1:
            raise ValueError("scales must be a positive integer")
        try:
            import pyshearlab
        except ImportError as exc:
            raise ImportError("Install the shearlet extra: pip install -e '.[shearlet]'") from exc
        self.library = pyshearlab
        self.scales = scales
        self.system = None
        self.shape = None

    def encode(self, image):
        height, width, _ = image.shape
        if height != width:
            raise ValueError("PyShearLab requires a square image; set image_size in the run config")
        if self.shape != (height, width):
            self.system = _build_system(self.library, height, width, self.scales)
            self.shape = (height, width)
        return np.stack([
            self.library.SLsheardec2D(image[:, :, channel], self.system).transpose(2, 0, 1)
            for channel in range(image.shape[2])
        ])

    def decode(self, coefficients):
        if self.system is None:
            raise ValueError("encode an image before decoding coefficients")
        return np.stack([
            self.library.SLshearrec2D(channel.transpose(1, 2, 0), self.system)
            for channel in coefficients
        ], axis=-1)

    def decode_adjoint(self, image_gradient):
        """Apply the real adjoint of synthesis, including dual-frame weights.

        PyShearLab implements each band as the centered inverse FFT of
        ``FFT(image_gradient) * conj(shearlet_band) / dualFrameWeights``.
        Its FFT shifts and normalization match ``SLshearrec2D`` exactly.
        ``encode`` initializes the frame before either synthesis operation.
        """
        if self.system is None:
            raise ValueError("encode an image before computing the decode adjoint")
        image_gradient = np.asarray(image_gradient, dtype=np.float64)
        if image_gradient.ndim != 3 or image_gradient.shape[:2] != self.shape:
            raise ValueError("image gradient must be HWC with the initialized spatial shape")
        return np.stack([
            self.library.SLshearrecadjoint2D(image_gradient[:, :, channel], self.system).transpose(2, 0, 1)
            for channel in range(image_gradient.shape[2])
        ])


def create_transform(config):
    options = dict(config)
    name = options.pop("name", "shearlet")
    if name == "shearlet":
        return ShearletTransform(**options)
    if name == "identity" and not options:
        return IdentityTransform()
    raise ValueError(f"Unknown transform or options: {name}")
