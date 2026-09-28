//! Reconstruct f32 from packed codes.

use crate::kernels::{
    dequant_asym_into, dequant_i4_blocks, dequant_i8_blocks, dequant_sym_into, dot_asym,
    dot_i4_blocks, dot_i8_blocks, dot_sym,
};
use crate::packed::{nbytes, Packed};
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
    bits: &[u32],
    block: usize,
    len: usize,
    out: &mut [f32],
) {
    let mut byte_offset = 0;
    let mut value_index = 0;
    for (block_index, &bit_width) in bits.iter().enumerate() {
        let count = (len - value_index).min(block);
        let byte_count = nbytes(count, bit_width);
        let mut codes = vec![0i32; count];
        Packed::unpack_slice(
            &bytes[byte_offset..byte_offset + byte_count],
            bit_width,
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
            bytes,
            bits,
            block,
            len,
            ..
        } => {
            let mut byte_offset = 0;
            let mut value_index = 0;
            for &bit_width in bits {
                let count = (*len - value_index).min(*block);
                let byte_count = nbytes(count, bit_width);
                Packed::unpack_slice(
                    &bytes[byte_offset..byte_offset + byte_count],
                    bit_width,
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
    // A block must sit inside one row so a row can be decoded on its own.
    // 4-bit rows also have to start on a byte (even column count).
    if packed_rows_ok(quantized, columns) {
        let scales = as_f32(quantized.scales());
        let zero_points = as_f32(quantized.zero_points());
        let mut row_weights = vec![0.0; columns];
        for row in 0..rows {
            decode_row(quantized, &scales, &zero_points, row, &mut row_weights);
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

fn decode_row<S: Scale>(
    quantized: &Quantized<S>,
    scales: &[f32],
    zero_points: &[f32],
    row: usize,
    out: &mut [f32],
) {
    let columns = out.len();
    let block = quantized.block();
    let scales_per_row = columns / block;
    let scale_offset = row * scales_per_row;
    let row_scales = &scales[scale_offset..scale_offset + scales_per_row];
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
        Quantized::Symmetric { codes, .. } => {
            let bytes_per_row = nbytes(columns, codes.bits());
            let start = row * bytes_per_row;
            let packed = Packed::from_raw(
                codes.as_bytes()[start..start + bytes_per_row].to_vec(),
                codes.bits(),
                columns,
            );
            dequant_sym_into(row_scales, &packed, block, out)
        }
        Quantized::Asymmetric { codes, .. } => {
            let bytes_per_row = nbytes(columns, codes.bits());
            let start = row * bytes_per_row;
            let packed = Packed::from_raw(
                codes.as_bytes()[start..start + bytes_per_row].to_vec(),
                codes.bits(),
                columns,
            );
            let row_zero_points = &zero_points[scale_offset..scale_offset + scales_per_row];
            dequant_asym_into(row_scales, row_zero_points, &packed, block, out)
        }
        Quantized::Adaptive { .. } => {
            let weights = quantized.dequantize();
            out.copy_from_slice(&weights[row * columns..(row + 1) * columns]);
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
    use crate::{adaptive, asymmetric, symmetric, Quantized};

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
        for quantized in tensors {
            let weights = quantized.dequantize();
            let matrix = quantized.into_matrix(rows, columns).unwrap();
            let fused = matrix.matmul(&inputs).unwrap();
            for (vector, input) in inputs.chunks_exact(columns).enumerate() {
                for (row, row_weights) in weights.chunks_exact(columns).enumerate() {
                    let naive: f32 = row_weights.iter().zip(input).map(|(a, b)| a * b).sum();
                    let got = fused[vector * rows + row];
                    assert!((naive - got).abs() < 1e-4, "{naive} vs {got}");
                }
            }
        }
    }
}
