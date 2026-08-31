"""Python bindings for quantize."""

from quantize._native import (
    InvalidBitsError,
    InvalidBlockError,
    InvalidToleranceError,
    LengthMismatchError,
    QuantizeError,
    Quantized,
    Scale,
    Scheme,
)

from . import adaptive, asymmetric, learned, symmetric
from .symmetric import quantize, quantize_tensor

__version__ = "0.2.1"

__all__ = [
    "Scale",
    "Scheme",
    "Quantized",
    "QuantizeError",
    "InvalidBitsError",
    "InvalidBlockError",
    "InvalidToleranceError",
    "LengthMismatchError",
    "quantize",
    "quantize_tensor",
    "symmetric",
    "asymmetric",
    "adaptive",
    "learned",
]
