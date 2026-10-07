//! Reconstruct f32 from packed codes.

use crate::kernels::{
    dequant_asym_into, dequant_i4_blocks, dequant_i8_blocks, dequant_sym_into, dot_asym,
    dot_i4_blocks, dot_i8_blocks, dot_sym,
};
use crate::packed::{Packed, nbytes, read_code};
use crate::params::assert_bits_in_range;
use crate::scale::Scale;
use crate::tensor::Quantized;

pub(crate) fn dequant_sym<S: Scale>(scales: &[S], codes: &Packed, block: usize, out: &mut [f32]) {
    match codes.bits() {
        8 => dequant_i8_blocks(scales, codes.as_bytes(), block, out),
        4 => dequant_i4_blocks(scales, codes.as_bytes(), block, out),
        _ => dequant_sym_into(scales, codes, block, out),
    }
}

pub(crate) fn dequant_asym<S: Scale>(
    scales: &[S],
    zero_points: &[S],
    codes: &Packed,
    block: usize,
    out: &mut [f32],
) {
    dequant_asym_into(scales, zero_points, codes, block, out);
}

/// Decode values `start..start + out.len()` of an adaptive tensor.
/// `first_byte` is where the codes of the block holding `start` begin.
///
/// Like [`dot_adaptive`], it reads each code in place: each block's codes are
/// packed at its own width, starting on the byte after the block before it.
pub(crate) fn dequant_adaptive<S: Scale>(
    quantized: &Quantized<S>,
    start: usize,
    first_byte: usize,
    out: &mut [f32],
) {
    let Quantized::Adaptive {
        scales,
        zero_points,
        codes,
        block_bits,
        block,
        ..
    } = quantized
    else {
        unreachable!("only adaptive tensors pack each block at its own width")
    };
    let end = start + out.len();
    let mut byte_offset = first_byte;
    for block_index in start / block..end.div_ceil(*block) {
        let bit_width = u32::from(block_bits[block_index]);
        assert_bits_in_range(bit_width);
        let block_codes = &codes[byte_offset..];
        let scale = scales[block_index].to_f32();
        let zero_point = zero_points[block_index].to_f32();
        let block_start = block_index * block;
        let indices = block_start.max(start)..(block_start + block).min(end);
        let slots = &mut out[indices.start - start..indices.end - start];
        for (slot, index) in slots.iter_mut().zip(indices) {
            let code = read_code(block_codes, index - block_start, bit_width);
            *slot = (code as f32 - zero_point) * scale;
        }
        // Only the tensor's last block can be short, and no block follows it.
        byte_offset += nbytes(*block, bit_width);
    }
}

/// Like [`dot_asym`], one block at a time: each block's codes are packed at
/// its own width, starting on the byte after the block before it.
pub(crate) fn dot_adaptive<S: Scale>(
    scales: &[S],
    zero_points: &[S],
    bytes: &[u8],
    bits: &[u8],
    block: usize,
    rhs: &[f32],
) -> f32 {
    let mut total = 0.0_f32;
    let mut byte_offset = 0;
    for (block_index, block_rhs) in rhs.chunks(block).enumerate() {
        let bit_width = u32::from(bits[block_index]);
        assert_bits_in_range(bit_width);
        let block_codes = &bytes[byte_offset..];
        let mut code_total = 0.0_f32;
        let mut rhs_total = 0.0_f32;
        for (code_index, &x) in block_rhs.iter().enumerate() {
            code_total += read_code(block_codes, code_index, bit_width) as f32 * x;
            rhs_total += x;
        }
        let scale = scales[block_index].to_f32();
        let zero_point = zero_points[block_index].to_f32();
        total += scale * (code_total - zero_point * rhs_total);
        byte_offset += nbytes(block_rhs.len(), bit_width);
    }
    total
}

pub(crate) fn dot_of<S: Scale>(quantized: &Quantized<S>, rhs: &[f32]) -> f32 {
    match quantized {
        Quantized::Symmetric {
            scales,
            codes,
            block,
            ..
        } if codes.bits() == 8 => dot_i8_blocks(scales, codes.as_bytes(), *block, rhs),
        Quantized::Symmetric {
            scales,
            codes,
            block,
            ..
        } if codes.bits() == 4 => dot_i4_blocks(scales, codes.as_bytes(), *block, rhs),
        Quantized::Symmetric {
            scales,
            codes,
            block,
            ..
        } => dot_sym(scales, codes, *block, rhs),
        Quantized::Asymmetric {
            scales,
            zero_points,
            codes,
            block,
            ..
        } => dot_asym(scales, zero_points, codes, *block, rhs),
        Quantized::Adaptive {
            scales,
            zero_points,
            codes,
            block_bits,
            block,
            ..
        } => dot_adaptive(scales, zero_points, codes, block_bits, *block, rhs),
    }
}

pub(crate) fn unpack_codes<S: Scale>(quantized: &Quantized<S>, out: &mut [i32]) {
    match quantized {
        Quantized::Symmetric { codes, .. } | Quantized::Asymmetric { codes, .. } => {
            codes.unpack_into(out);
        }
        Quantized::Adaptive {
            codes,
            block_bits,
            block,
            len,
            ..
        } => {
            let mut byte_offset = 0;
            let mut value_index = 0;
            for &bit_width in block_bits {
                let count = (*len - value_index).min(*block);
                let byte_count = nbytes(count, bit_width.into());
                Packed::unpack_slice(
                    &codes[byte_offset..byte_offset + byte_count],
                    bit_width.into(),
                    &mut out[value_index..value_index + count],
                    count,
                );
                byte_offset += byte_count;
                value_index += count;
            }
        }
    }
}

pub(crate) fn matmul_into<S: Scale>(
    quantized: &Quantized<S>,
    inputs: &[f32],
    columns: usize,
    out: &mut [f32],
) {
    let rows = quantized.len() / columns;
    if quantized.is_empty() || inputs.is_empty() {
        return;
    }

    // Decode each weight row once, then reuse it for every vector in the batch.
    let mut row_weights = vec![0.0; columns];
    for row in 0..rows {
        decode_row(quantized, row, &mut row_weights);
        for (vector, input) in inputs.chunks_exact(columns).enumerate() {
            out[vector * rows + row] = dot(&row_weights, input);
        }
    }
}

fn packed_rows_ok<S: Scale>(quantized: &Quantized<S>, columns: usize) -> bool {
    let block = quantized.block();
    if block == 0 || !columns.is_multiple_of(block) {
        return false;
    }
    match quantized {
        Quantized::Symmetric { codes, .. } | Quantized::Asymmetric { codes, .. } => {
            (columns * codes.bits() as usize).is_multiple_of(8)
        }
        Quantized::Adaptive { .. } => false,
    }
}

/// Decode row `row` of a matrix with `out.len()` columns.
///
/// When a block sits inside one row and each row starts on a byte, a 4-bit or
/// 8-bit row's codes and scales are slices of the tensor's, and the packed
/// kernels decode them directly. Any other row goes through `decode_values`.
pub(crate) fn decode_row<S: Scale>(quantized: &Quantized<S>, row: usize, out: &mut [f32]) {
    let columns = out.len();
    if !packed_rows_ok(quantized, columns) {
        decode_values(quantized, row, out);
        return;
    }
    let block = quantized.block();
    let scales_per_row = columns / block;
    let row_scales = &quantized.scales()[row * scales_per_row..(row + 1) * scales_per_row];
    match quantized {
        Quantized::Symmetric { codes, .. } if codes.bits() == 8 => {
            let start = row * columns;
            dequant_i8_blocks(
                row_scales,
                &codes.as_bytes()[start..start + columns],
                block,
                out,
            )
        }
        Quantized::Symmetric { codes, .. } if codes.bits() == 4 => {
            let bytes_per_row = columns / 2;
            let start = row * bytes_per_row;
            dequant_i4_blocks(
                row_scales,
                &codes.as_bytes()[start..start + bytes_per_row],
                block,
                out,
            )
        }
        _ => decode_values(quantized, row, out),
    }
}

/// Decode row `row` of a matrix with `out.len()` columns, reading only the
/// blocks it falls in. Unlike the packed kernels, this works for a row that
/// starts anywhere, even partway through a block or a byte.
fn decode_values<S: Scale>(quantized: &Quantized<S>, row: usize, out: &mut [f32]) {
    let block = quantized.block();
    let start = row * out.len();
    match quantized {
        Quantized::Symmetric { codes, .. } | Quantized::Asymmetric { codes, .. } => {
            // A symmetric tensor has no zero-points: it decodes as if each were 0.
            let (scales, zero_points) = (quantized.scales(), quantized.zero_points());
            let end = start + out.len();
            let (first_block, end_block) = (start / block, end.div_ceil(block));
            for (block_index, scale) in (first_block..).zip(&scales[first_block..end_block]) {
                let scale = scale.to_f32();
                let zero_point = zero_points
                    .get(block_index)
                    .map_or(0.0, |zero_point| zero_point.to_f32());
                let block_start = block_index * block;
                let indices = block_start.max(start)..(block_start + block).min(end);
                let slots = &mut out[indices.start - start..indices.end - start];
                for (slot, index) in slots.iter_mut().zip(indices) {
                    *slot = (codes.code(index) as f32 - zero_point) * scale;
                }
            }
        }
        Quantized::Adaptive { row_starts, .. } => {
            dequant_adaptive(quantized, start, row_starts[row], out);
        }
    }
}

/// Multiply two slices element by element and add up the products.
///
/// Float addition is not associative, so with one running total the compiler
/// must add the products in order, one at a time. Sixteen separate totals are
/// independent, so it can add them side by side in SIMD registers.
fn dot(left: &[f32], right: &[f32]) -> f32 {
    const LANES: usize = 16;
    let (left_chunks, left_remainder) = left.as_chunks::<LANES>();
    let (right_chunks, right_remainder) = right.as_chunks::<LANES>();
    let remainder = left_remainder.iter().zip(right_remainder);
    let remainder_total: f32 = remainder.map(|(a, b)| a * b).sum();

    let mut totals = [0.0_f32; LANES];
    for (left_chunk, right_chunk) in left_chunks.iter().zip(right_chunks) {
        for ((total, a), b) in totals.iter_mut().zip(left_chunk).zip(right_chunk) {
            *total += a * b;
        }
    }
    totals.iter().sum::<f32>() + remainder_total
}

#[cfg(test)]
mod tests {
    use crate::{Quantized, adaptive, asymmetric, symmetric};

    #[test]
    fn dot_stays_precise_when_every_product_is_positive() {
        // Summing a non-negative tensor, by dotting it with ones, adds 262,144
        // positive products. One running total over all of them would lose up
        // to 0.1% of the sum.
        let values: Vec<f32> = (0..262_144)
            .map(|i| (i as f32 * 0.37).sin().abs())
            .collect();
        let ones = vec![1.0; values.len()];
        let tensors: [Quantized<f32>; 4] = [
            symmetric::quantize_with(&values, 3, 32).unwrap(),
            symmetric::quantize_with(&values, 5, 32).unwrap(),
            asymmetric::quantize_with(&values, 4, 32).unwrap(),
            adaptive::quantize_with(&values, 32, 0.01).unwrap(),
        ];
        for quantized in tensors {
            let exact: f64 = quantized
                .dequantize()
                .iter()
                .map(|&value| value as f64)
                .sum();
            let got = quantized.dot(&ones).unwrap() as f64;
            let relative_error = ((got - exact) / exact).abs();
            assert!(relative_error < 2e-5, "{relative_error}");
        }
    }

    #[test]
    fn dot_matches_the_decoded_values_for_every_layout() {
        // Blocks of 7 codes at 3, 4, or 5 bits end partway through a byte,
        // 120 values leave a last block of 1, and the adaptive blocks widen
        // from 2 to 7 bits as the values do.
        let values: Vec<f32> = (0..120)
            .map(|i| (i as f32 * 0.37).sin() * i as f32 / 120.0)
            .collect();
        let rhs: Vec<f32> = (0..120).map(|i| (i as f32 * 0.11).cos()).collect();
        let tensors: [Quantized<f32>; 5] = [
            symmetric::quantize_with(&values, 8, 7).unwrap(),
            symmetric::quantize_with(&values, 4, 7).unwrap(),
            symmetric::quantize_with(&values, 5, 7).unwrap(),
            asymmetric::quantize_with(&values, 3, 7).unwrap(),
            adaptive::quantize_with(&values, 7, 0.01).unwrap(),
        ];
        for quantized in tensors {
            let decoded = quantized.dequantize();
            let products = decoded.iter().zip(&rhs);
            let expected: f32 = products.map(|(weight, x)| weight * x).sum();
            let got = quantized.dot(&rhs).unwrap();
            assert!((expected - got).abs() < 1e-5, "{expected} vs {got}");
        }
    }

    /// A 4 × 30 matrix in every layout a row can have. The first two decode
    /// each row with the packed 4-bit and 8-bit kernels. The rest read each
    /// code in place: 12-bit and asymmetric rows start on a block and a byte,
    /// 5-bit rows of 30 values end partway through a byte, and blocks of 8, 7,
    /// and 9 cross from one row into the next. The last has 4-bit and 5-bit
    /// blocks of 8, so its rows start after a mix of widths.
    fn matrices_in_every_layout() -> [Quantized<f32>; 9] {
        let values: Vec<f32> = (0..120).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut matrices = [
            symmetric::quantize_with(&values, 8, 10).unwrap(),
            symmetric::quantize_with(&values, 4, 6).unwrap(),
            symmetric::quantize_with(&values, 12, 15).unwrap(),
            asymmetric::quantize_with(&values, 4, 10).unwrap(),
            symmetric::quantize_with(&values, 5, 10).unwrap(),
            symmetric::quantize_with(&values, 4, 8).unwrap(),
            asymmetric::quantize_with(&values, 3, 7).unwrap(),
            adaptive::quantize_with(&values, 9, 0.01).unwrap(),
            adaptive::quantize_with(&values, 8, 0.05).unwrap(),
        ];
        for matrix in &mut matrices {
            matrix.set_shape(4, 30).unwrap();
        }
        matrices
    }

    #[test]
    fn matmul_matches_dequant_then_multiply_for_every_layout() {
        let inputs: Vec<f32> = (0..90).map(|i| (i as f32) * 0.01 - 0.3).collect();
        for quantized in matrices_in_every_layout() {
            let (rows, columns) = quantized.shape().unwrap();
            let weights = quantized.dequantize();
            let fused = quantized.matmul(&inputs).unwrap();
            for (vector, input) in inputs.chunks_exact(columns).enumerate() {
                for (row, row_weights) in weights.chunks_exact(columns).enumerate() {
                    let naive: f32 = row_weights.iter().zip(input).map(|(a, b)| a * b).sum();
                    let got = fused[vector * rows + row];
                    assert!((naive - got).abs() < 1e-4, "{naive} vs {got}");
                }
            }
        }
    }

    #[test]
    fn dequantize_row_into_matches_dequantize_for_every_layout() {
        for quantized in matrices_in_every_layout() {
            let (_, columns) = quantized.shape().unwrap();
            let every_value = quantized.dequantize();
            let mut row_values = vec![0.0; columns];
            for (row, expected) in every_value.chunks_exact(columns).enumerate() {
                quantized.dequantize_row_into(row, &mut row_values).unwrap();
                assert_eq!(row_values, expected, "row {row}");
            }
        }
    }
}
