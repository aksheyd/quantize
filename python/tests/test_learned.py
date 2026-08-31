import numpy as np
import pytest

from quantize import LengthMismatchError, QuantizeError, learned, quantize


def test_fit_recovers_known_line():
    codes = [0, 1, 2, 3, 4]
    values = [0.5 * code + 1.0 for code in codes]
    scale, zero_point = learned.fit_scale_and_zero_point(values, codes)
    assert abs(scale - 0.5) < 1e-5
    assert abs(zero_point + 2.0) < 1e-5


def test_refine_flips_kind_and_keeps_identity():
    weights = [0.42, -0.10, 0.70, -0.50]
    quantized = quantize(weights, bits=8, block=4)
    assert quantized.kind == "symmetric"
    before = quantized.copy()
    before_nbytes = quantized.nbytes
    result = learned.refine(quantized, weights)
    assert result is quantized
    assert quantized.kind == "asymmetric"
    assert before.kind == "symmetric"
    assert quantized.nbytes > before_nbytes
    np.testing.assert_array_equal(quantized.unpacked_codes, before.unpacked_codes)


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
    with pytest.raises(QuantizeError):
        learned.fit_scale_and_zero_point([0.1], [0, 1])
