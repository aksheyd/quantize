import pickle

import numpy as np
import pytest

from quantize import (
    InvalidBitsError,
    InvalidBlockError,
    InvalidToleranceError,
    LengthMismatchError,
    QuantizeError,
    Scale,
    Scheme,
    adaptive,
    asymmetric,
    quantize,
    quantize_tensor,
)


def test_eight_bit_roundtrip_stays_within_half_step():
    weights = [0.42, -0.10, 0.70, -0.50]
    quantized = quantize(weights, bits=8, block=4)
    back = quantized.dequantize()
    for original, reconstructed in zip(weights, back):
        assert abs(original - reconstructed) < 0.01


def test_four_bit_packed_byte_count():
    weights = [0.1] * 32
    quantized = quantize(weights, bits=4, block=32)
    assert quantized.codes.size == 16


def test_remainder_block_length():
    weights = [i * 0.01 - 0.2 for i in range(40)]
    quantized = quantize(weights, bits=8, block=32)
    assert len(quantized) == 40


def test_dequantize_out_length_error():
    quantized = quantize([0.1] * 8, bits=8, block=8)
    out = np.zeros(3, dtype=np.float32)
    with pytest.raises(LengthMismatchError) as raised:
        quantized.dequantize(out)
    assert raised.value.expected == 8
    assert raised.value.got == 3


def test_dequantize_out_returns_same_object():
    quantized = quantize([0.1] * 8, bits=8, block=8)
    out = np.zeros(8, dtype=np.float32)
    result = quantized.dequantize(out)
    assert result is out


def test_fused_dot_matches_dequant_then_dot():
    weights = [i * 0.01 - 0.3 for i in range(64)]
    quantized = quantize(weights, bits=8, block=32)
    reconstructed = quantized.dequantize()
    naive = float(np.dot(reconstructed, np.array(weights, dtype=np.float32)))
    fused = quantized.dot(weights)
    assert abs(naive - fused) < 1e-4


def test_dot_length_mismatch():
    quantized = quantize([0.1] * 8, bits=8, block=8)
    with pytest.raises(LengthMismatchError):
        quantized.dot([0.1] * 3)


def test_fused_matmul_matches_dequant_then_multiply():
    columns = 32
    weights = [i * 0.01 - 0.3 for i in range(4 * columns)]
    quantized = quantize(weights, bits=8, block=32)
    reconstructed = np.asarray(quantized.dequantize(), dtype=np.float32).reshape(4, columns)
    rhs = np.array([i * 0.02 - 0.1 for i in range(columns)], dtype=np.float32)
    naive = reconstructed @ rhs
    fused = quantized.matmul(rhs)
    np.testing.assert_allclose(naive, fused, atol=1e-4)


def test_matmul_length_mismatch():
    quantized = quantize([0.1] * 64, bits=8, block=32)
    with pytest.raises(LengthMismatchError) as raised:
        quantized.matmul([0.1] * 3, columns=32)
    assert raised.value.expected == 32
    assert raised.value.got == 3


def test_matmul_zero_columns():
    quantized = quantize([0.1] * 8, bits=8, block=8)
    with pytest.raises(InvalidBlockError) as raised:
        quantized.matmul([0.1] * 8, columns=0)
    assert raised.value.block == 0


def test_matmul_batch_is_row_major():
    columns = 32
    rows = 4
    weights = [i * 0.01 - 0.3 for i in range(rows * columns)]
    quantized = quantize(weights, bits=8, block=32)
    reconstructed = np.asarray(quantized.dequantize(), dtype=np.float32).reshape(rows, columns)
    rhs = np.array(
        [
            [i * 0.02 - 0.1 for i in range(columns)],
            [i * 0.01 + 0.05 for i in range(columns)],
        ],
        dtype=np.float32,
    )
    naive = (rhs @ reconstructed.T).ravel()
    fused = quantized.matmul(rhs)
    np.testing.assert_allclose(naive, fused, atol=1e-4)
    assert fused.shape == (2 * rows,)


def test_invalid_bits():
    with pytest.raises(InvalidBitsError) as raised:
        quantize([0.1], bits=1, block=1)
    assert raised.value.bits == 1
    with pytest.raises(QuantizeError):
        quantize([0.1], bits=17, block=1)


def test_invalid_block():
    with pytest.raises(InvalidBlockError) as raised:
        quantize([0.1], bits=8, block=0)
    assert raised.value.block == 0


def test_invalid_tolerance():
    with pytest.raises(InvalidToleranceError):
        adaptive.quantize([0.1], block=1, tolerance=0.0)


def test_scheme_constants_and_eq():
    assert Scheme.Q8_32 == Scheme.symmetric(8, 32)
    assert Scheme.Q4_32 == Scheme.symmetric(4, 32)
    assert Scheme.Q8_32.kind == "symmetric"
    assert repr(Scheme.Q8_32) == "Scheme(kind='symmetric', bits=8, block=32)"
    assert "adaptive" in repr(Scheme.adaptive(block=32, tolerance=0.001))


def test_scheme_factory_does_not_validate():
    scheme = Scheme.symmetric(bits=1)
    assert scheme.bits == 1
    with pytest.raises(InvalidBitsError):
        scheme.quantize([0.1])


def test_ndim_not_one():
    with pytest.raises(TypeError, match="1-D"):
        quantize(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="1-D"):
        quantize(np.array(0.1, dtype=np.float32))


def test_scale_enum_selects_storage():
    weights = [0.42, -0.10, 0.70, -0.50]
    assert quantize(weights, bits=8, block=4).scale == Scale.F32
    assert quantize(weights, bits=8, block=4, scale=Scale.F16).scale == Scale.F16
    assert quantize(weights, bits=8, block=4, scale=Scale.Bf16).scale == Scale.Bf16
    assert Scheme.Q8_32.quantize(weights, scale=Scale.F16).scale == Scale.F16
    assert Scale.F32 != Scale.F16
    assert (
        repr(quantize(weights, bits=8, block=4, scale=Scale.F16))
        == "Quantized(kind='symmetric', bits=8, block=4, len=4, scale=Scale.F16)"
    )


def test_bad_scale():
    with pytest.raises(TypeError):
        quantize([0.1], scale="f32")
    with pytest.raises(TypeError):
        quantize([0.1], scale="float32")


def test_negative_bits_overflow():
    with pytest.raises(OverflowError):
        quantize([0.1], bits=-1)


def test_asymmetric_and_adaptive_paths():
    weights = [0.42, -0.10, 0.70, -0.50]
    asymmetric.quantize(weights, bits=8, block=4)
    mixed = adaptive.quantize(weights, block=2, tolerance=0.001)
    assert mixed.kind == "adaptive"
    assert mixed.block_bits is not None
    assert mixed.bits is None


def test_quantize_tensor_one_scale():
    weights = [0.1, 0.2, 0.3]
    quantized = quantize_tensor(weights, bits=8)
    assert quantized.block == 3
    assert len(quantized.scales) == 1


def test_pickle_roundtrip_including_f16():
    weights = [0.42, -0.10, 0.70, -0.50]
    quantized = quantize(weights, bits=8, block=4, scale=Scale.F16)
    restored = pickle.loads(pickle.dumps(quantized))
    assert restored is not quantized
    assert restored.kind == quantized.kind
    assert restored.scale == Scale.F16
    assert restored.nbytes == quantized.nbytes
    assert list(restored.codes) == list(quantized.codes)
    assert list(restored.unpacked_codes) == list(quantized.unpacked_codes)
    np.testing.assert_array_equal(restored.dequantize(), quantized.dequantize())
    assert pickle.loads(pickle.dumps(Scheme.Q8_32)) == Scheme.symmetric(8, 32)
    assert pickle.loads(pickle.dumps(Scale.F16)) == Scale.F16
