//! Reconstruct f32 from packed codes.

use crate::kernels::{
    dequant_asym_into, dequant_i4_blocks, dequant_i8_blocks, dequant_sym_into, dot_asym,
    dot_i4_blocks, dot_i8_blocks, dot_sym,
};
use crate::packed::{Packed, nbytes};
use crate::scale::Scale;
use crate::tensor::Quantized;

pub(crate) fn as_f32<S: Scale>(values: &[S]) -> Vec<f32> {
    values.iter().copied().map(S::to_f32).collect()
}

pub(crate) fn dequant_sym<S: Scale>(scales: &[S], codes: &Packed, block: usize, out: &mut [f32]) {
    let scales = as_f32(scales);
    match codes.bits() {
        8 => dequant_i8_blocks(&scales, codes.as_bytes(), block, out),
        4 => dequant_i4_blocks(&scales, codes.as_bytes(), block, out),
        _ => dequant_sym_into(&scales, codes, block, out),
    }
}

pub(crate) fn dequant_asym<S: Scale>(
    scales: &[S],
    zero_points: &[S],
    codes: &Packed,
    block: usize,
    out: &mut [f32],
) {
    dequant_asym_into(&as_f32(scales), &as_f32(zero_points), codes, block, out);
}

pub(crate) fn dequant_adaptive<S: Scale>(
    scales: &[S],
    zero_points: &[S],
    bytes: &[u8],
    bits: &[u8],
    block: usize,
    len: usize,
    out: &mut [f32],
) {
    let mut byte_offset = 0;
    let mut value_index = 0;
    for (block_index, &bit_width) in bits.iter().enumerate() {
        let count = (len - value_index).min(block);
        let byte_count = nbytes(count, bit_width.into());
        let mut codes = vec![0i32; count];
        Packed::unpack_slice(
            &bytes[byte_offset..byte_offset + byte_count],
            bit_width.into(),
            &mut codes,
            count,
        );
        let scale = scales[block_index].to_f32();
        let zero_point = zero_points[block_index].to_f32();
        for (slot, &code) in out[value_index..value_index + count].iter_mut().zip(&codes) {
            *slot = (code as f32 - zero_point) * scale;
        }
        byte_offset += byte_count;
        value_index += count;
    }
}

pub(crate) fn dot_of<S: Scale>(quantized: &Quantized<S>, rhs: &[f32]) -> f32 {
    match quantized {
        Quantized::Symmetric {
            scales,
            codes,
            block,
            ..
        } if codes.bits() == 8 => dot_i8_blocks(&as_f32(scales), codes.as_bytes(), *block, rhs),
        Quantized::Symmetric {
            scales,
            codes,
            block,
            ..
        } if codes.bits() == 4 => dot_i4_blocks(&as_f32(scales), codes.as_bytes(), *block, rhs),
        Quantized::Symmetric {
            scales,
            codes,
            block,
            ..
        } => dot_sym(&as_f32(scales), codes, *block, rhs),
        Quantized::Asymmetric {
            scales,
            zero_points,
            codes,
            block,
            ..
        } => dot_asym(&as_f32(scales), &as_f32(zero_points), codes, *block, rhs),
        Quantized::Adaptive { block, .. } => {
            // Added up block by block, like the kernels above.
            let weights = quantized.dequantize();
            let mut total = 0.0;
            for (block_weights, block_rhs) in weights.chunks(*block).zip(rhs.chunks(*block)) {
                let products = block_weights.iter().zip(block_rhs);
                total += products.map(|(weight, x)| weight * x).sum::<f32>();
            }
            total
        }
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
    // Rows decode fastest when a block sits inside one row and each row starts
    // on a byte (for 4-bit, an even column count). Otherwise decode the whole
    // matrix once.
    if packed_rows_ok(quantized, columns) {
        let mut row_weights = vec![0.0; columns];
        for row in 0..rows {
            decode_row(quantized, row, &mut row_weights);
            for (vector, input) in inputs.chunks_exact(columns).enumerate() {
                out[vector * rows + row] = dot(&row_weights, input);
            }
        }
        return;
    }

    let weights = quantized.dequantize();
    for (row, row_weights) in weights.chunks_exact(columns).enumerate() {
        for (vector, input) in inputs.chunks_exact(columns).enumerate() {
            out[vector * rows + row] = dot(row_weights, input);
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
/// When a block sits inside one row and each row starts on a byte, the row's
/// codes and scales are slices of the tensor's, and the packed kernels decode
/// them directly. Any other row goes through `decode_values`.
pub(crate) fn decode_row<S: Scale>(quantized: &Quantized<S>, row: usize, out: &mut [f32]) {
    let columns = out.len();
    if !packed_rows_ok(quantized, columns) {
        decode_values(quantized, row * columns, out);
        return;
    }
    let block = quantized.block();
    let scales_per_row = columns / block;
    let row_blocks = row * scales_per_row..(row + 1) * scales_per_row;
    let row_scales = as_f32(&quantized.scales()[row_blocks.clone()]);
    match quantized {
        Quantized::Symmetric { codes, .. } if codes.bits() == 8 => {
            let start = row * columns;
            dequant_i8_blocks(
                &row_scales,
                &codes.as_bytes()[start..start + columns],
                block,
                out,
            )
        }
        Quantized::Symmetric { codes, .. } if codes.bits() == 4 => {
            let bytes_per_row = columns / 2;
            let start = row * bytes_per_row;
            dequant_i4_blocks(
                &row_scales,
                &codes.as_bytes()[start..start + bytes_per_row],
                block,
                out,
            )
        }
        Quantized::Symmetric { codes, .. } => {
            let bytes_per_row = nbytes(columns, codes.bits());
            let start = row * bytes_per_row;
            let packed = Packed::from_raw(
                codes.as_bytes()[start..start + bytes_per_row].to_vec(),
                codes.bits(),
                columns,
            );
            dequant_sym_into(&row_scales, &packed, block, out)
        }
        Quantized::Asymmetric {
            codes, zero_points, ..
        } => {
            let bytes_per_row = nbytes(columns, codes.bits());
            let start = row * bytes_per_row;
            let packed = Packed::from_raw(
                codes.as_bytes()[start..start + bytes_per_row].to_vec(),
                codes.bits(),
                columns,
            );
            let row_zero_points = as_f32(&zero_points[row_blocks]);
            dequant_asym_into(&row_scales, &row_zero_points, &packed, block, out)
        }
        Quantized::Adaptive { .. } => unreachable!("adaptive rows are never packed rows"),
    }
}

/// Decode values `start..start + out.len()`, reading only the blocks they fall
/// in. Unlike the packed kernels, this works for a range that starts anywhere,
/// even partway through a block or a byte.
fn decode_values<S: Scale>(quantized: &Quantized<S>, start: usize, out: &mut [f32]) {
    let block = quantized.block();
    match quantized {
        Quantized::Symmetric { codes, .. } | Quantized::Asymmetric { codes, .. } => {
            let mut range_codes = vec![0; out.len()];
            codes.unpack_range(start, &mut range_codes);
            // A symmetric tensor has no zero-points: it decodes as if each were 0.
            let (scales, zero_points) = (quantized.scales(), quantized.zero_points());
            for (offset, (slot, code)) in out.iter_mut().zip(range_codes).enumerate() {
                let block_index = (start + offset) / block;
                let zero_point = zero_points
                    .get(block_index)
                    .map_or(0.0, |zero_point| zero_point.to_f32());
                *slot = (code as f32 - zero_point) * scales[block_index].to_f32();
            }
        }
        Quantized::Adaptive {
            scales,
            zero_points,
            codes,
            block_bits,
            len,
            ..
        } => {
            // Each block is packed at its own width, so the first block's bytes
            // start after those of every block before it. Only the last block
            // can be short, so those blocks are full, and when `block` is a
            // multiple of 8, `block` codes of `bit_width` bits fill exactly
            // `block / 8 × bit_width` bytes: adding up the widths is enough.
            let first_block = start / block;
            let end_block = (start + out.len()).div_ceil(block);
            let blocks_before = &block_bits[..first_block];
            let byte_offset: usize = if block.is_multiple_of(8) {
                let width_total: usize = blocks_before
                    .iter()
                    .map(|&bit_width| usize::from(bit_width))
                    .sum();
                block / 8 * width_total
            } else {
                blocks_before
                    .iter()
                    .map(|&bit_width| nbytes(block, bit_width.into()))
                    .sum()
            };
            let first_value = first_block * block;
            let mut decoded = vec![0.0; (end_block * block).min(*len) - first_value];
            let blocks = first_block..end_block;
            dequant_adaptive(
                &scales[blocks.clone()],
                &zero_points[blocks.clone()],
                &codes[byte_offset..],
                &block_bits[blocks],
                block,
                decoded.len(),
                &mut decoded,
            );
            let offset = start - first_value;
            out.copy_from_slice(&decoded[offset..offset + out.len()]);
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
    fn matmul_matches_dequant_then_multiply_for_every_scheme() {
        let values: Vec<f32> = (0..80).map(|i| (i as f32) * 0.02 - 0.8).collect();
        let inputs: Vec<f32> = (0..80).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let (rows, columns) = (2, 40);
        let tensors: [Quantized<f32>; 5] = [
            symmetric::quantize_with(&values, 8, 8).unwrap(),
            symmetric::quantize_with(&values, 4, 8).unwrap(),
            symmetric::quantize_with(&values, 5, 8).unwrap(),
            asymmetric::quantize_with(&values, 4, 8).unwrap(),
            adaptive::quantize_with(&values, 8, 0.001).unwrap(),
        ];
        for mut quantized in tensors {
            let weights = quantized.dequantize();
            quantized.set_shape(rows, columns).unwrap();
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
        let values: Vec<f32> = (0..120).map(|i| (i as f32 * 0.37).sin()).collect();
        let (rows, columns) = (4, 30);
        // The first four decode each row with the packed kernels. The rest
        // don't: 5-bit rows of 30 values end partway through a byte, and
        // blocks of 8, 7, and 9 cross from one row into the next. The last
        // has 4-bit and 5-bit blocks of 8, so its rows start after a mix of
        // widths.
        let tensors: [Quantized<f32>; 9] = [
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
        for mut quantized in tensors {
            let every_value = quantized.dequantize();
            quantized.set_shape(rows, columns).unwrap();
            let mut row_values = vec![0.0; columns];
            for (row, expected) in every_value.chunks_exact(columns).enumerate() {
                quantized.dequantize_row_into(row, &mut row_values).unwrap();
                assert_eq!(row_values, expected, "row {row}");
            }
        }
    }
}
