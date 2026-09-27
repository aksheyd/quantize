"""Python bindings for quantize."""

from quantize._native import (
    InvalidBitsError,
    InvalidBlockError,
    InvalidToleranceError,
    LengthMismatchError,
    NotAMatrixError,
    QuantizeError,
    Quantized,
    Scale,
    Scheme,
    ShapeMismatchError,
)

from . import adaptive, asymmetric, learned, symmetric
from .symmetric import quantize, quantize_tensor

__version__ = "0.2.2"

__all__ = [
    "Scale",
    "Scheme",
    "Quantized",
    "QuantizeError",
    "InvalidBitsError",
    "InvalidBlockError",
    "InvalidToleranceError",
    "LengthMismatchError",
    "ShapeMismatchError",
    "NotAMatrixError",
    "quantize",
    "quantize_tensor",
    "symmetric",
    "asymmetric",
    "adaptive",
    "learned",
]
