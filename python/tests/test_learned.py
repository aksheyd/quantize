import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from quantize import LengthMismatchError, QuantizeError, adaptive, asymmetric, learned, quantize


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


def test_alternate_moves_a_value_to_a_closer_code_in_place():
    # Seven small values and one outlier. Once the fit moves the line,
    # 0.02 sits closer to code 6 than to its code 5.
    values = [0.02, -0.09, 0.10, 0.03, -0.04, 0.13, 0.04, -0.90]
    quantized = asymmetric.quantize(values, bits=4, block=8)
    assert list(quantized.unpacked_codes) == [5, 4, 7, 6, 5, 7, 6, -8]
    assert learned.alternate(quantized, values) is quantized
    assert list(quantized.unpacked_codes) == [6, 4, 7, 6, 5, 7, 6, -8]


def squared_error(quantized, weights):
    return np.sum((quantized.dequantize() - weights) ** 2)


def test_alternate_ends_no_higher_than_refine_and_keeps_the_tensor():
    # Each block is wider than the last, so adaptive packs them at 4 to 7 bits.
    weights = (np.sin(np.arange(256)) * (1 + np.arange(256) // 32)).astype(np.float32)
    weights = weights.reshape(8, 32)
    for original in [
        quantize(weights, bits=4),
        asymmetric.quantize(weights, bits=4),
        adaptive.quantize(weights, block=32, tolerance=0.1),
    ]:
        refined = learned.refine(original.copy(), weights)
        alternated = learned.alternate(original.copy(), weights)
        assert squared_error(alternated, weights) <= squared_error(refined, weights)
        assert alternated.kind == original.kind
        assert alternated.shape == (8, 32)
        assert alternated.nbytes == original.nbytes
    with pytest.raises(LengthMismatchError):
        learned.alternate(quantize(weights), weights[:4])


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


def test_refine_and_alternate_work_while_another_thread_uses_the_tensor():
    weights = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    quantized = quantize(weights, bits=4)
    stop = threading.Event()

    def multiply_until_stopped():
        while not stop.is_set():
            quantized.matmul(weights)
            quantized.dot(weights.ravel())

    with ThreadPoolExecutor() as pool:
        multiplying = pool.submit(multiply_until_stopped)
        try:
            for _ in range(10):
                learned.refine(quantized, weights)
                learned.alternate(quantized, weights)
        finally:
            stop.set()
        multiplying.result()
