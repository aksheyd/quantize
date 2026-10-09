import re

import numpy as np
import pytest

from quantize import QuantizeError, Quantized, Scale, adaptive, asymmetric, quantize


def weights(rows, columns):
    """Weights spread like a layer's, with a block of zeros and outliers."""
    values = 0.02 * np.random.default_rng(0).standard_normal((rows, columns), np.float32)
    values[0, :32] = 0
    values[1, ::37] *= 20
    return values


def bit_patterns(values):
    """Floats as their bits, so that -0 differs from 0."""
    return values.view(np.uint32)


def test_four_and_eight_bit_matrices_round_trip_through_ggml_blocks():
    for bits, ggml_type, block_bytes in [(4, "Q4_0", 18), (8, "Q8_0", 34)]:
        quantized = quantize(weights(4, 96), bits=bits, scale=Scale.F16)
        blocks, to_type = quantized.to_ggml()
        assert to_type == ggml_type
        assert blocks.dtype == np.uint8 and blocks.shape == (4, 3 * block_bytes)
        assert Quantized.from_ggml(blocks, ggml_type) == quantized
        # The block of zeros has codes of zero, and each other block puts its
        # value farthest from zero on the lowest code: -8, or -128, which
        # ggml's own quantizer never writes.
        assert not quantized.unpacked_codes[:32].any()
        assert quantized.unpacked_codes.min() == -(2 ** (bits - 1))


def test_each_block_holds_its_scale_then_its_codes():
    for bits in (4, 8):
        quantized = quantize(weights(2, 64), bits=bits, scale=Scale.F16)
        blocks, _ = quantized.to_ggml()
        blocks = blocks.reshape(4, -1)
        np.testing.assert_array_equal(blocks[:, :2].copy().view(np.float16).ravel(), quantized.scales)
        codes = quantized.unpacked_codes.reshape(4, 32)
        if bits == 8:
            np.testing.assert_array_equal(blocks[:, 2:].view(np.int8), codes)
        else:
            # Byte j holds code j plus 8 in its low 4 bits, and code j + 16
            # plus 8 in its high 4.
            np.testing.assert_array_equal(blocks[:, 2:] & 0x0F, codes[:, :16] + 8)
            np.testing.assert_array_equal(blocks[:, 2:] >> 4, codes[:, 16:] + 8)


def test_blocks_of_any_dimensions_load_as_a_matrix_of_their_rows():
    quantized = quantize(weights(6, 64), bits=8, scale=Scale.F16)
    blocks, ggml_type = quantized.to_ggml()
    assert Quantized.from_ggml(blocks.reshape(2, 3, -1), ggml_type) == quantized
    first_row = Quantized.from_ggml(blocks[0], ggml_type)
    assert first_row.shape == (1, 64)
    np.testing.assert_array_equal(first_row.dequantize()[0], quantized.dequantize()[0])


def test_tensors_that_q4_0_and_q8_0_cannot_hold_are_refused_with_why():
    matrix = weights(2, 64)
    cases = [
        (asymmetric.quantize(matrix, bits=4, scale=Scale.F16), "is asymmetric, but Q4_0 and Q8_0 are symmetric"),
        (adaptive.quantize(matrix, tolerance=0.01, scale=Scale.F16), "is adaptive, but Q4_0 and Q8_0 give every block one width"),
        (quantize(matrix, bits=6, scale=Scale.F16), "has 6-bit codes, but Q4_0 and Q8_0 have 4-bit and 8-bit ones"),
        (quantize(matrix, bits=8, block=64, scale=Scale.F16), "has blocks of 64, but Q4_0 and Q8_0 have blocks of 32"),
        (quantize(matrix, bits=4), "has f32 scales, but Q4_0 and Q8_0 hold f16 ones"),
        (quantize(matrix, bits=8, scale=Scale.BF16), "has bf16 scales, but Q4_0 and Q8_0 hold f16 ones"),
        (quantize(matrix.ravel(), bits=4, scale=Scale.F16), "is a flat vector of 128 values, but Q4_0 and Q8_0 blocks run along a matrix's rows"),
        (quantize(weights(2, 48), bits=8, scale=Scale.F16), "has rows of 48 values, but Q4_0 and Q8_0 split each row into blocks of 32"),
    ]
    for quantized, why in cases:
        message = f"the tensor {why}; safetensors keeps any quantized tensor"
        with pytest.raises(QuantizeError, match=re.escape(message)):
            quantized.to_ggml()


def test_blocks_that_do_not_fit_their_type_are_refused():
    blocks, _ = quantize(weights(2, 64), bits=4, scale=Scale.F16).to_ggml()
    with pytest.raises(QuantizeError, match=re.escape('"q4_0" isn\'t Q4_0 or Q8_0, the ggml types that hold')):
        Quantized.from_ggml(blocks, "q4_0")
    with pytest.raises(QuantizeError, match="rows of 36 bytes, which don't split into Q8_0 blocks of 34 bytes"):
        Quantized.from_ggml(blocks, "Q8_0")
    with pytest.raises(QuantizeError, match="Q4_0 with rows of no values"):
        Quantized.from_ggml(np.zeros((2, 0), np.uint8), "Q4_0")
    for wrong in [blocks.view(np.int8), blocks.tolist(), np.uint8(7)]:
        with pytest.raises(TypeError, match="data must be a uint8 array of ggml blocks"):
            Quantized.from_ggml(wrong, "Q4_0")


def test_gguf_decodes_the_blocks_to_the_same_values_bit_for_bit():
    gguf = pytest.importorskip("gguf")
    for bits in (4, 8):
        quantized = quantize(weights(8, 256), bits=bits, scale=Scale.F16)
        blocks, ggml_type = quantized.to_ggml()
        decoded = gguf.quants.dequantize(blocks, gguf.GGMLQuantizationType[ggml_type])
        np.testing.assert_array_equal(bit_patterns(decoded), bit_patterns(quantized.dequantize()))


def test_blocks_that_gguf_quantizes_load_and_decode_as_gguf_decodes_them():
    gguf = pytest.importorskip("gguf")
    for ggml_type, bits in [("Q4_0", 4), ("Q8_0", 8)]:
        qtype = gguf.GGMLQuantizationType[ggml_type]
        blocks = gguf.quants.quantize(weights(8, 256), qtype)
        quantized = Quantized.from_ggml(blocks, ggml_type)
        assert (quantized.kind, quantized.bits, quantized.block) == ("symmetric", bits, 32)
        assert (quantized.scale, quantized.shape) == (Scale.F16, (8, 256))
        decoded = gguf.quants.dequantize(blocks, qtype)
        np.testing.assert_array_equal(bit_patterns(quantized.dequantize()), bit_patterns(decoded))
        np.testing.assert_array_equal(quantized.to_ggml()[0], blocks)


def test_a_file_that_gguf_writes_reads_back_equal(tmp_path):
    gguf = pytest.importorskip("gguf")
    tensors = {f"{bits}-bit": quantize(weights(4, 64), bits=bits, scale=Scale.F16) for bits in (4, 8)}
    writer = gguf.GGUFWriter(tmp_path / "model.gguf", arch="llama")
    for name, quantized in tensors.items():
        blocks, ggml_type = quantized.to_ggml()
        writer.add_tensor(name, blocks, raw_dtype=gguf.GGMLQuantizationType[ggml_type])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    read_back = gguf.GGUFReader(tmp_path / "model.gguf").tensors
    assert sorted(tensor.name for tensor in read_back) == sorted(tensors)
    for tensor in read_back:
        # GGUFReader lists dimensions innermost first.
        assert list(tensor.shape) == [64, 4]
        assert Quantized.from_ggml(tensor.data, tensor.tensor_type.name) == tensors[tensor.name]
