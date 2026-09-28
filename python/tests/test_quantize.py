import array
import io
import pickle
import re
from collections.abc import Hashable

import numpy as np
import pytest

from quantize import (
    InvalidBitsError,
    InvalidBlockError,
    InvalidToleranceError,
    LengthMismatchError,
    NotAMatrixError,
    QuantizeError,
    Quantized,
    Scale,
    ScaleOutOfRangeError,
    Scheme,
    ShapeMismatchError,
    ToleranceTooTightError,
    adaptive,
    asymmetric,
    quantize,
    quantize_tensor,
)


def weight_matrix(rows, columns):
    return np.linspace(-0.6, 0.6, rows * columns, dtype=np.float32).reshape(rows, columns)


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


def test_values_decode_as_the_docs_say():
    weights = weight_matrix(3, 30)
    for quantized in [
        quantize(weights, bits=4, block=32),
        asymmetric.quantize(weights, bits=5, block=16),
        adaptive.quantize(weights, block=32, tolerance=0.001),
    ]:
        block_of_each_value = np.arange(len(quantized)) // quantized.block
        codes = quantized.unpacked_codes
        scales = quantized.scales[block_of_each_value]
        if quantized.kind == "symmetric":
            decoded = codes * scales
        else:
            decoded = (codes - quantized.zero_points[block_of_each_value]) * scales
        np.testing.assert_allclose(decoded, quantized.dequantize().ravel(), rtol=1e-6)
    four_bit = quantize(weights, bits=4)
    low_nibbles = four_bit.codes & 0x0F
    first_codes = np.where(low_nibbles > 7, low_nibbles.astype(np.int32) - 16, low_nibbles)
    np.testing.assert_array_equal(first_codes, four_bit.unpacked_codes[::2])
    assert (four_bit.scales < 0).any()


def test_help_shows_the_default_scale():
    for function in [
        quantize,
        quantize_tensor,
        asymmetric.quantize,
        asymmetric.quantize_tensor,
        adaptive.quantize,
        Scheme.Q8_32.quantize,
    ]:
        assert function.__doc__
        assert "scale='f32'" in function.__text_signature__


def test_remainder_block_length():
    weights = [i * 0.01 - 0.2 for i in range(40)]
    quantized = quantize(weights, bits=8, block=32)
    assert len(quantized) == 40


def test_dequantize_out_length_error():
    quantized = quantize([0.1] * 8, bits=8, block=8)
    out = np.zeros(3, dtype=np.float32)
    with pytest.raises(LengthMismatchError, match="out must have length 8, got 3") as raised:
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
    with pytest.raises(LengthMismatchError, match="values must have length 8, got 3"):
        quantized.dot([0.1] * 3)


def test_matrix_keeps_its_shape():
    weights = weight_matrix(8, 32)
    for quantized in [
        quantize(weights, bits=8, block=32),
        quantize_tensor(weights, bits=8),
        asymmetric.quantize(weights, bits=4, block=16),
        adaptive.quantize(weights, block=32, tolerance=0.001),
        Scheme.Q4_32.quantize(weights),
    ]:
        assert quantized.shape == (8, 32)
        assert len(quantized) == 256
        back = quantized.dequantize()
        assert back.shape == (8, 32)
        np.testing.assert_allclose(back, weights, atol=0.05)
    assert quantize(weights.ravel()).shape == (256,)
    assert quantize([0.1, 0.2]).dequantize().shape == (2,)


def test_quantize_reads_any_real_dtype_and_layout_row_by_row():
    weights = weight_matrix(32, 8)
    expected = quantize(np.ascontiguousarray(weights.T)).dequantize()
    for values in [weights.T, np.asfortranarray(weights.T), weights.T.astype(np.float64)]:
        np.testing.assert_array_equal(quantize(values).dequantize(), expected)
    assert quantize(np.arange(-4, 4).reshape(2, 4)).shape == (2, 4)


class Tensor:
    """Stands in for a PyTorch tensor, which NumPy reads through __array__."""

    def __init__(self, values):
        self.values = values

    def __array__(self, dtype=None, copy=None):
        return self.values


def test_quantize_reads_anything_numpy_asarray_reads():
    weights = weight_matrix(4, 32)
    expected = quantize(weights)
    for values in [
        Tensor(weights),
        weights.tolist(),
        weights.astype(">f4"),
        memoryview(weights),
    ]:
        assert quantize(values) == expected
    row = array.array("f", weights[0])
    assert quantize(row) == quantize(weights[0])
    assert quantize(Tensor(weights)).matmul(Tensor(weights[:2])).shape == (2, 4)


def test_values_that_are_not_real_numbers_are_rejected_with_what_arrived():
    for values, dtype in [(np.array([1 + 2j]), "complex128"), (b"ab", "|S2"), ("0.5", "<U3")]:
        with pytest.raises(TypeError, match=f"real numbers, got .* dtype {re.escape(dtype)}"):
            quantize(values)
    with pytest.raises(ValueError, match=r"Tensor with shape \(2, 2, 2\) and dtype float32"):
        quantize(Tensor(np.zeros((2, 2, 2), np.float32)))


def test_masked_arrays_are_rejected_instead_of_losing_their_mask():
    masked = np.ma.masked_array([0.1, 0.2, 99.0], mask=[False, False, True])
    with pytest.raises(TypeError, match=r"masked array.*values\.filled\(0\)"):
        quantize(masked, bits=8, block=3)
    with pytest.raises(TypeError, match=r"inputs can't be a masked array.*inputs\.filled\(0\)"):
        quantize(weight_matrix(2, 3)).matmul(masked)
    back = quantize(masked.filled(0), bits=8, block=3).dequantize()
    np.testing.assert_allclose(back, [0.1, 0.2, 0.0], atol=1e-3)


def test_quantize_rejects_a_matrix_with_no_columns():
    with pytest.raises(ShapeMismatchError, match="rows of 0 columns") as raised:
        quantize(np.zeros((4, 0), np.float32))
    assert raised.value.columns == 0


def test_dequantize_out_must_have_the_tensor_shape():
    quantized = quantize(weight_matrix(8, 32))
    out = np.empty((8, 32), np.float32)
    assert quantized.dequantize(out) is out
    np.testing.assert_array_equal(out, quantized.dequantize())
    with pytest.raises(ValueError, match=r"shape \(8, 32\), got \(32, 8\)"):
        quantized.dequantize(np.empty((32, 8), np.float32))
    with pytest.raises(LengthMismatchError):
        quantized.dequantize(np.empty((8, 16), np.float32))


def test_fused_matmul_matches_dequant_then_multiply():
    quantized = quantize(weight_matrix(4, 32), bits=8, block=32)
    inputs = np.array([i * 0.02 - 0.1 for i in range(32)], dtype=np.float32)
    naive = quantized.dequantize() @ inputs
    fused = quantized.matmul(inputs=inputs)
    assert fused.shape == (4,)
    np.testing.assert_allclose(naive, fused, atol=1e-4)


def test_matmul_length_mismatch():
    quantized = quantize(weight_matrix(2, 32), bits=8, block=32)
    with pytest.raises(LengthMismatchError, match="each vector in inputs must have length 32") as raised:
        quantized.matmul([0.1] * 3)
    assert raised.value.expected == 32
    assert raised.value.got == 3


def test_matmul_batch_returns_batch_by_rows():
    rows = 4
    quantized = quantize(weight_matrix(rows, 32), bits=8, block=32)
    values = np.array(
        [
            [i * 0.02 - 0.1 for i in range(32)],
            [i * 0.01 + 0.05 for i in range(32)],
        ],
        dtype=np.float32,
    )
    naive = values @ quantized.dequantize().T
    fused = quantized.matmul(values)
    np.testing.assert_allclose(naive, fused, atol=1e-4)
    assert fused.shape == (2, rows)


def test_matmul_rejects_a_flat_tensor():
    quantized = quantize([0.1] * 64, bits=8, block=32)
    with pytest.raises(NotAMatrixError, match="flat vector of 64 values") as raised:
        quantized.matmul([0.1] * 32)
    assert raised.value.len == 64


def test_matmul_rejects_column_vectors_and_transposed_batches():
    quantized = quantize(weight_matrix(4, 32), bits=8, block=32)
    for values in [np.zeros((32, 1), np.float32), np.zeros((32, 8), np.float32)]:
        with pytest.raises(LengthMismatchError) as raised:
            quantized.matmul(values)
        assert raised.value.expected == 32
        assert raised.value.got == values.shape[1]


def test_matmul_reads_a_square_input_as_a_batch():
    columns = 32
    rows = 4
    quantized = quantize(weight_matrix(rows, columns), bits=8, block=32)
    square = np.linspace(-1, 1, columns * columns, dtype=np.float32).reshape(columns, columns)
    fused = quantized.matmul(square)
    assert fused.shape == (columns, rows)
    np.testing.assert_allclose(square @ quantized.dequantize().T, fused, atol=1e-4)


def test_matmul_reads_any_real_dtype_and_layout():
    quantized = quantize(weight_matrix(4, 32), bits=8, block=32)
    batch = np.linspace(-1, 1, 3 * 32).reshape(32, 3).T
    expected = quantized.matmul(np.ascontiguousarray(batch, dtype=np.float32))
    np.testing.assert_array_equal(quantized.matmul(batch), expected)


def test_matmul_rejects_three_dimensional_values():
    quantized = quantize(weight_matrix(2, 32), bits=8, block=32)
    with pytest.raises(ValueError, match="inputs must be a 1-D or 2-D array"):
        quantized.matmul(np.zeros((2, 2, 32), np.float32))


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


def test_adaptive_needs_a_tolerance_by_name():
    with pytest.raises(TypeError, match="tolerance"):
        adaptive.quantize([0.1])
    with pytest.raises(TypeError, match="tolerance"):
        Scheme.adaptive()
    with pytest.raises(TypeError, match="positional"):
        adaptive.quantize([0.1], 32, 0.001)
    with pytest.raises(TypeError, match="positional"):
        Scheme.adaptive(32, 0.001)


def test_a_tolerance_8_bits_cannot_meet_raises_with_one_they_can():
    values = [0.0, 1.0, 0.0, 2.0]
    with pytest.raises(ToleranceTooTightError, match="block 0 .* at least 0.0039") as raised:
        adaptive.quantize(values, block=2, tolerance=0.001)
    assert isinstance(raised.value, QuantizeError)
    assert raised.value.block_index == 0
    assert raised.value.smallest_tolerance == pytest.approx(2 / 255 / 2)
    adaptive.quantize(values, block=2, tolerance=raised.value.smallest_tolerance)


def test_a_zero_point_that_f16_cannot_hold_raises():
    values = [0.02, 0.03, 0.04, 0.05]
    with pytest.raises(ScaleOutOfRangeError, match="use f32 scales") as raised:
        asymmetric.quantize(values, bits=16, block=4, scale=Scale.F16)
    assert isinstance(raised.value, QuantizeError)
    assert raised.value.block_index == 0
    assert raised.value.scale_type == "f16"
    back = asymmetric.quantize(values, bits=16, block=4).dequantize()
    np.testing.assert_allclose(back, values, atol=1e-6)


def test_scheme_constants_and_eq():
    assert Scheme.Q8_32 == Scheme.symmetric(8, 32)
    assert Scheme.Q4_32 == Scheme.symmetric(4, 32)
    assert not isinstance(Scheme.Q8_32, Hashable)
    assert Scheme.Q8_32.kind == "symmetric"
    assert repr(Scheme.Q8_32) == "Scheme(kind='symmetric', bits=8, block=32)"
    assert "adaptive" in repr(Scheme.adaptive(block=32, tolerance=0.001))


def test_scheme_factory_does_not_validate():
    scheme = Scheme.symmetric(bits=1)
    assert scheme.bits == 1
    with pytest.raises(InvalidBitsError):
        scheme.quantize([0.1])


def test_quantize_rejects_other_dimensions():
    with pytest.raises(ValueError, match=r"1-D or 2-D array, got .* shape \(2, 2, 2\)"):
        quantize(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match=r"1-D or 2-D array, got .* shape \(\)"):
        quantize(np.array(0.1, dtype=np.float32))


def test_scale_enum_selects_storage():
    weights = [0.42, -0.10, 0.70, -0.50]
    assert quantize(weights, bits=8, block=4).scale == Scale.F32
    assert quantize(weights, bits=8, block=4, scale=Scale.F16).scale == Scale.F16
    assert quantize(weights, bits=8, block=4, scale=Scale.BF16).scale == Scale.BF16
    assert Scheme.Q8_32.quantize(weights, scale=Scale.F16).scale == Scale.F16
    assert Scale.F32 != Scale.F16
    assert (
        repr(quantize(weights, bits=8, block=4, scale=Scale.F16))
        == "Quantized(kind='symmetric', bits=8, block=4, shape=(4,), scale=Scale.F16)"
    )
    assert repr(Scale.BF16) == "Scale.BF16"
    assert repr(quantize(weights, scale=Scale.BF16)).endswith("scale=Scale.BF16)")


def test_scale_can_be_given_by_name():
    for scale in [Scale.F32, Scale.F16, Scale.BF16]:
        assert quantize([0.1], scale=scale.name).scale == scale
    assert Scale.BF16.name == "bf16"
    with pytest.raises(QuantizeError, match="scale must be a Scale or its name"):
        quantize([0.1], scale="float32")


def test_bad_values_raise_value_errors_that_name_the_argument():
    for bits in [-4, 1, 17]:
        with pytest.raises(InvalidBitsError, match=f"bits must be from 2 to 16, got {bits}") as raised:
            quantize([0.1], bits=bits)
        assert raised.value.bits == bits
        assert isinstance(raised.value, ValueError)
    for block in [-32, 0]:
        with pytest.raises(InvalidBlockError, match=f"block must be at least 1, got {block}") as raised:
            asymmetric.quantize([0.1], block=block)
        assert raised.value.block == block
        assert isinstance(raised.value, ValueError)
    with pytest.raises(InvalidBitsError, match="got -1"):
        Scheme.symmetric(bits=-1)
    with pytest.raises(ValueError, match="tolerance"):
        adaptive.quantize([0.1], tolerance=-1.0)


def test_asymmetric_and_adaptive_paths():
    weights = [0.42, -0.10, 0.70, -0.50]
    asymmetric.quantize(weights, bits=8, block=4)
    mixed = adaptive.quantize(weights, block=2, tolerance=0.01)
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
    assert restored == quantized
    assert restored.kind == quantized.kind
    assert restored.scale == Scale.F16
    assert restored.nbytes == quantized.nbytes
    assert list(restored.codes) == list(quantized.codes)
    assert list(restored.unpacked_codes) == list(quantized.unpacked_codes)
    np.testing.assert_array_equal(restored.dequantize(), quantized.dequantize())
    assert pickle.loads(pickle.dumps(Scheme.Q8_32)) == Scheme.symmetric(8, 32)
    assert pickle.loads(pickle.dumps(Scale.F16)) == Scale.F16


def test_pickle_keeps_the_matrix_shape():
    quantized = quantize(weight_matrix(8, 32), bits=4, block=32)
    restored = pickle.loads(pickle.dumps(quantized))
    assert restored.shape == (8, 32)
    np.testing.assert_array_equal(restored.matmul(np.ones(32)), quantized.matmul(np.ones(32)))


def test_bytes_round_trip_every_kind_and_scale_type_through_numpy():
    weights = weight_matrix(3, 30)
    for scale in [Scale.F32, Scale.F16, Scale.BF16]:
        for quantized in [
            quantize(weights, bits=4, block=32, scale=scale),
            asymmetric.quantize(weights, bits=5, block=16, scale=scale),
            adaptive.quantize(weights.ravel(), block=32, tolerance=0.001, scale=scale),
            quantize([], scale=scale),
        ]:
            data = quantized.to_bytes()
            assert Quantized.from_bytes(data) == quantized
            saved = saved_and_loaded_with_numpy({"layer": np.frombuffer(data, np.uint8)})
            assert Quantized.from_bytes(saved["layer"]) == quantized


def test_pickles_hold_the_bytes_that_from_bytes_loads():
    quantized = quantize(weight_matrix(8, 32), bits=4, scale=Scale.F16)
    rebuild, (data,) = quantized.__reduce__()
    assert rebuild == Quantized.from_bytes
    assert data == quantized.to_bytes()


def test_from_bytes_rejects_bytes_that_do_not_hold_a_tensor():
    for quantized in [
        quantize([0.1] * 64, bits=4, block=32),
        adaptive.quantize([i * 0.01 for i in range(40)], block=32, tolerance=0.001),
    ]:
        data = quantized.to_bytes()
        with pytest.raises(ValueError, match="malformed"):
            Quantized.from_bytes(data[:-1])
        with pytest.raises(ValueError, match="saved with f64 scales, not f32"):
            Quantized.from_bytes(data.replace(b"f32", b"f64", 1))
        with pytest.raises(ValueError, match="QNTZ"):
            Quantized.from_bytes(b"not a tensor")


# quantize([0.42, -0.10, 0.70, -0.50], bits=8, block=4), pickled by quantize-py 0.2.2.
PICKLED_BY_0_2 = (
    b"\x80\x04\x95\xa5\x00\x00\x00\x00\x00\x00\x00\x8c\x08builtins\x94\x8c\x07geta"
    b"ttr\x94\x93\x94\x8c\x08quantize\x94\x8c\tQuantized\x94\x93\x94\x8c\x0c_from_"
    b"pickle\x94\x86\x94R\x94(K\x01\x8c\tsymmetric\x94h\x02\x8c\x08quantize\x94"
    b"\x8c\x05Scale\x94\x93\x94\x8c\x0c_from_pickle\x94\x86\x94R\x94\x8c\x03f32"
    b"\x94\x85\x94R\x94K\x04K\x04C\x04L\xee\x7f\xa5\x94K\x08)C\x04l\x9c\xb4;\x94C"
    b"\x00\x94t\x94\x85\x94R\x94."
)


def test_a_pickle_from_0_2_says_how_to_move_the_tensor_over():
    with pytest.raises(ValueError, match="pickled by quantize-py 0.2") as raised:
        pickle.loads(PICKLED_BY_0_2)
    assert "quantize the original weights again" in str(raised.value)
    assert "Quantized.from_parts" in str(raised.value)
    with pytest.raises(ValueError, match="malformed"):
        Quantized._from_pickle((2, "symmetric"))


def test_quantized_compares_by_value():
    weights = weight_matrix(4, 32)
    quantized = quantize(weights, bits=4)
    assert quantized == quantized.copy()
    assert quantized == quantize(weights, bits=4)
    assert quantized != quantize(weights, bits=8)
    assert quantized != quantize(weights.ravel(), bits=4)
    assert quantized != quantize(weights, bits=4, scale=Scale.F16)
    assert not isinstance(quantized, Hashable)
    with pytest.raises(TypeError, match="unhashable"):
        hash(quantized)


def saved_and_loaded_with_numpy(arrays):
    file = io.BytesIO()
    np.savez(file, **arrays)
    file.seek(0)
    with np.load(file) as loaded:
        return dict(loaded)


def test_from_parts_rebuilds_parts_saved_with_numpy_as_the_readme_says():
    weights = weight_matrix(3, 30)
    for scale in [Scale.F32, Scale.F16, Scale.BF16]:
        for quantized in [
            quantize(weights, bits=4, block=32, scale=scale),
            asymmetric.quantize(weights, bits=5, block=16, scale=scale),
            adaptive.quantize(weights.ravel(), block=32, tolerance=0.001, scale=scale),
            quantize([], scale=scale),
        ]:
            parts = {
                "kind": quantized.kind,
                "shape": quantized.shape,
                "block": quantized.block,
                "codes": quantized.codes,
                "scales": quantized.scales,
                "zero_points": quantized.zero_points,
                "scale": quantized.scale.name,
            }
            if quantized.kind == "adaptive":
                parts["block_bits"] = quantized.block_bits
            else:
                parts["bits"] = quantized.bits
            rebuilt = Quantized.from_parts(**saved_and_loaded_with_numpy(parts))
            assert rebuilt == quantized


def test_from_parts_rejects_parts_that_do_not_fit_together():
    quantized = asymmetric.quantize(weight_matrix(3, 30), bits=4, block=32)
    parts = {
        "kind": "asymmetric",
        "shape": (3, 30),
        "block": 32,
        "codes": quantized.codes,
        "scales": quantized.scales,
        "zero_points": quantized.zero_points,
        "bits": 4,
        "scale": "f32",
    }
    assert Quantized.from_parts(**parts) == quantized
    with pytest.raises(TypeError, match="scale"):
        Quantized.from_parts(**{name: part for name, part in parts.items() if name != "scale"})
    for changed, error, message in [
        ({"bits": 17}, InvalidBitsError, "bits must be from 2 to 16, got 17"),
        ({"block": 0}, InvalidBlockError, "block must be at least 1, got 0"),
        ({"shape": (90, 0)}, ShapeMismatchError, "rows of 0 columns"),
        ({"shape": (4, 30)}, ValueError, "one scale"),
        ({"shape": (3, 3, 10)}, ValueError, "shape must be"),
        ({"codes": quantized.codes[:-1]}, ValueError, "codes must fill"),
        ({"codes": quantized.unpacked_codes}, TypeError, "uint8"),
        ({"zero_points": None}, ValueError, "one zero-point"),
        ({"kind": "symmetric"}, ValueError, "no zero_points"),
        ({"kind": "adaptive"}, ValueError, "'adaptive', with block_bits"),
        ({"block_bits": [4, 4, 4]}, ValueError, "'asymmetric', with bits"),
        ({"kind": "int4"}, ValueError, "kind must be"),
    ]:
        with pytest.raises(error, match=message):
            Quantized.from_parts(**{**parts, **changed})


def test_adaptive_block_widths_take_one_byte_each():
    quantized = adaptive.quantize(weight_matrix(3, 30), block=32, tolerance=0.001)
    assert quantized.block_bits.dtype == np.uint8
    parts = [quantized.codes, quantized.scales, quantized.zero_points, quantized.block_bits]
    assert quantized.nbytes == sum(part.nbytes for part in parts)
    with pytest.raises(InvalidBitsError, match="got 300"):
        Quantized.from_parts(
            kind="adaptive",
            shape=quantized.shape,
            block=quantized.block,
            codes=quantized.codes,
            scales=quantized.scales,
            zero_points=quantized.zero_points,
            block_bits=[300, 8, 8],
            scale="f32",
        )
