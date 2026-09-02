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
        Quantized::Adaptive { .. } => quantized
            .dequantize()
            .iter()
            .zip(rhs)
            .map(|(left, right)| left * right)
            .sum(),
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

pub(crate) fn matmul_of<S: Scale>(
    quantized: &Quantized<S>,
    rhs: &[f32],
    columns: usize,
) -> Vec<f32> {
    let rows = quantized.len() / columns;
    let vectors = rhs.len() / columns;
    let mut out = vec![0.0; vectors * rows];
    if quantized.is_empty() || rhs.is_empty() {
        return out;
    }

    // A block must sit inside one row so each row can reuse the packed fused dots.
    // 4-bit rows also have to start on a byte (even column count).
    if packed_rows_ok(quantized, columns) {
        let scales = as_f32(quantized.scales());
        let zero_points = as_f32(quantized.zero_points());
        for vector in 0..vectors {
            let rhs_vec = &rhs[vector * columns..(vector + 1) * columns];
            for row in 0..rows {
                out[vector * rows + row] =
                    packed_row_dot(quantized, &scales, &zero_points, row, columns, rhs_vec);
            }
        }
        return out;
    }

    let weights = quantized.dequantize();
    for vector in 0..vectors {
        let rhs_vec = &rhs[vector * columns..(vector + 1) * columns];
        for row in 0..rows {
            let left = &weights[row * columns..(row + 1) * columns];
            out[vector * rows + row] = left.iter().zip(rhs_vec).map(|(w, x)| w * x).sum();
        }
    }
    out
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

fn packed_row_dot<S: Scale>(
    quantized: &Quantized<S>,
    scales: &[f32],
    zero_points: &[f32],
    row: usize,
    columns: usize,
    rhs: &[f32],
) -> f32 {
    let block = quantized.block();
    let scales_per_row = columns / block;
    let scale_offset = row * scales_per_row;
    let row_scales = &scales[scale_offset..scale_offset + scales_per_row];
    match quantized {
        Quantized::Symmetric { codes, .. } if codes.bits() == 8 => {
            let start = row * columns;
            dot_i8_blocks(
                row_scales,
                &codes.as_bytes()[start..start + columns],
                block,
                rhs,
            )
        }
        Quantized::Symmetric { codes, .. } if codes.bits() == 4 => {
            let bytes_per_row = columns / 2;
            let start = row * bytes_per_row;
            dot_i4_blocks(
                row_scales,
                &codes.as_bytes()[start..start + bytes_per_row],
                block,
                rhs,
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
            dot_sym(row_scales, &packed, block, rhs)
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
            dot_asym(row_scales, row_zero_points, &packed, block, rhs)
        }
        Quantized::Adaptive { .. } => {
            let weights = quantized.dequantize();
            let left = &weights[row * columns..(row + 1) * columns];
            left.iter().zip(rhs).map(|(w, x)| w * x).sum()
        }
    }
}
