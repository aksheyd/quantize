import numpy as np
import pytest

from quantize import LengthMismatchError, QuantizeError, learned, quantize


def test_fit_recovers_known_line():
    codes = [0, 1, 2, 3, 4]
    values = [0.5 * code + 1.0 for code in codes]
    scale, zero_point = learned.fit_scale_and_zero_point(values, codes)
    assert abs(scale - 0.5) < 1e-5
    assert abs(zero_point + 2.0) < 1e-5


def test_refine_keeps_kind_and_identity():
    weights = [0.42, -0.10, 0.70, -0.50]
    quantized = quantize(weights, bits=8, block=4)
    before = quantized.copy()
    result = learned.refine(quantized, weights)
    assert result is quantized
    assert quantized.kind == "symmetric"
    assert quantized.nbytes == before.nbytes
    np.testing.assert_array_equal(quantized.unpacked_codes, before.unpacked_codes)


def test_refine_takes_the_matrix_it_refines():
    weights = np.linspace(-0.5, 0.5, 64, dtype=np.float32).reshape(2, 32)
    quantized = quantize(weights, bits=4, block=32)
    learned.refine(quantized, weights)
    assert quantized.shape == (2, 32)


def test_refine_length_mismatch_including_empty():
    quantized = quantize([0.1] * 4, bits=8, block=4)
    with pytest.raises(LengthMismatchError):
        learned.refine(quantized, [0.1] * 3)
    empty = quantize([], bits=8, block=4)
    with pytest.raises(LengthMismatchError):
        learned.refine(empty, [0.1])
    assert learned.refine(empty, []) is empty


def test_fit_rejects_packed_codes():
    weights = [0.42, -0.10, 0.70, -0.50]
    quantized = quantize(weights, bits=8, block=4)
    with pytest.raises(TypeError, match="unpacked_codes"):
        learned.fit_scale_and_zero_point(weights, quantized.codes)
    scale, zero_point = learned.fit_scale_and_zero_point(
        weights, quantized.unpacked_codes
    )
    assert isinstance(scale, float)
    assert isinstance(zero_point, float)


def test_except_quantize_error_catches_length():
    with pytest.raises(QuantizeError, match="codes must have length 1, got 2") as raised:
        learned.fit_scale_and_zero_point([0.1], [0, 1])
    assert isinstance(raised.value, ValueError)
