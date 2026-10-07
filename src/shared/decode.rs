//! Reconstruct f32 from packed codes.

use crate::kernels::{
    decode_i4_32, decode_i8_32, dequant_asym_into, dequant_i4_blocks, dequant_i8_blocks,
    dequant_sym_into, dot_asym, dot_i4_blocks, dot_i8_blocks, dot_sym,
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
/// `first_byte` is where the codes of the block holding `start` begin. Returns
/// where the codes of the block holding `start + out.len()` begin, so the
/// values after these can carry on from there.
///
/// Like [`dot_adaptive`], it reads each code in place: each block's codes are
/// packed at its own width, starting on the byte after the block before it.
pub(crate) fn dequant_adaptive<S: Scale>(
    quantized: &Quantized<S>,
    start: usize,
    first_byte: usize,
    out: &mut [f32],
) -> usize {
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
        // Step past only the blocks that end within the range: the values
        // after these start in a block that runs past `end`. Those blocks are
        // all full, since only the tensor's last block can be short.
        if block_start + block <= end {
            byte_offset += nbytes(*block, bit_width);
        }
    }
    byte_offset
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

    // A single vector, as when a language model generates one token, leaves
    // no batch to reuse a decoded row for, and storing each row only to read
    // it straight back costs about as much as decoding it. So a 4-bit or 8-bit
    // matrix whose blocks are whole groups of 32 codes multiplies each group
    // as it decodes it instead.
    if let Quantized::Symmetric {
        scales,
        codes,
        block,
        ..
    } = quantized
        && inputs.len() == columns
        && matches!(codes.bits(), 4 | 8)
        && block.is_multiple_of(32)
        && packed_rows_ok(quantized, columns)
    {
        let bytes_per_row = columns * codes.bits() as usize / 8;
        let scales_per_row = columns / block;
        let mut row_scales = vec![0.0; scales_per_row];
        for (row, slot) in out.iter_mut().enumerate() {
            let row_codes = &codes.as_bytes()[row * bytes_per_row..(row + 1) * bytes_per_row];
            let stored_scales = &scales[row * scales_per_row..(row + 1) * scales_per_row];
            scales_to_f32(stored_scales, &mut row_scales);
            *slot = match codes.bits() {
                4 => dot_i4_row(&row_scales, row_codes, *block, inputs),
                _ => dot_i8_row(&row_scales, row_codes, *block, inputs),
            };
        }
        return;
    }

    // Decode each weight row once, then reuse it for every vector in the batch.
    // Finding an adaptive row on its own means adding up the widths of every
    // block before it, so here each adaptive row carries on from the byte
    // where the row before it stopped.
    let mut row_weights = vec![0.0; columns];
    let mut row_first_byte = 0;
    for row in 0..rows {
        match quantized {
            Quantized::Adaptive { .. } => {
                let start = row * columns;
                row_first_byte =
                    dequant_adaptive(quantized, start, row_first_byte, &mut row_weights);
            }
            _ => decode_row(quantized, row, &mut row_weights),
        }
        for (vector, input) in inputs.chunks_exact(columns).enumerate() {
            out[vector * rows + row] = dot(&row_weights, input);
        }
    }
}

/// Convert `stored` into `out`, four scales at a time.
///
/// On Apple silicon, `half` converts each f16 with an instruction of its own,
/// so a loop that converted one scale per step spent about as long on its
/// steps as on the conversions.
fn scales_to_f32<S: Scale>(stored: &[S], out: &mut [f32]) {
    let (out_quads, out_rest) = out.as_chunks_mut::<4>();
    let (stored_quads, stored_rest) = stored.as_chunks::<4>();
    for (quad, stored) in out_quads.iter_mut().zip(stored_quads) {
        *quad = [
            stored[0].to_f32(),
            stored[1].to_f32(),
            stored[2].to_f32(),
            stored[3].to_f32(),
        ];
    }
    for (scale, stored) in out_rest.iter_mut().zip(stored_rest) {
        *scale = stored.to_f32();
    }
}

/// One row of a 4-bit matrix times `input`, for blocks of whole groups of 32
/// codes, without storing the decoded row: each group is decoded into
/// registers and multiplied right away. Its products go into the same totals
/// as `dot`'s, in the same order, and a row of whole groups of 32 leaves `dot`
/// no remainder, so the result is bit for bit what decoding the row and
/// calling `dot` gives.
///
/// It takes the row's scales as f32 instead of being generic over the scale
/// type. A generic function compiles in the crate that calls it, at that
/// crate's optimization level, so this loop would run unoptimized in a debug
/// build, even one that sets `opt-level = 3` for its dependencies.
fn dot_i4_row(scales: &[f32], bytes: &[u8], block: usize, input: &[f32]) -> f32 {
    let mut totals = [0.0_f32; LANES];
    let blocks = bytes.chunks_exact(block / 2).zip(input.chunks_exact(block));
    for ((block_bytes, block_input), &scale) in blocks.zip(scales) {
        let (groups, _) = block_bytes.as_chunks::<16>();
        let (group_inputs, _) = block_input.as_chunks::<32>();
        for (group, group_input) in groups.iter().zip(group_inputs) {
            let weights = decode_i4_32(group, scale);
            let (weight_chunks, _) = weights.as_chunks::<LANES>();
            let (input_chunks, _) = group_input.as_chunks::<LANES>();
            add_products(&mut totals, weight_chunks, input_chunks);
        }
    }
    totals.iter().sum()
}

/// Like [`dot_i4_row`], for an 8-bit matrix, where each group of 32 codes
/// takes 32 bytes instead of 16.
fn dot_i8_row(scales: &[f32], bytes: &[u8], block: usize, input: &[f32]) -> f32 {
    let mut totals = [0.0_f32; LANES];
    let blocks = bytes.chunks_exact(block).zip(input.chunks_exact(block));
    for ((block_bytes, block_input), &scale) in blocks.zip(scales) {
        let (groups, _) = block_bytes.as_chunks::<32>();
        let (group_inputs, _) = block_input.as_chunks::<32>();
        for (group, group_input) in groups.iter().zip(group_inputs) {
            let weights = decode_i8_32(group, scale);
            let (weight_chunks, _) = weights.as_chunks::<LANES>();
            let (input_chunks, _) = group_input.as_chunks::<LANES>();
            add_products(&mut totals, weight_chunks, input_chunks);
        }
    }
    totals.iter().sum()
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
        decode_values(quantized, row * columns, out);
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
        _ => decode_values(quantized, row * columns, out),
    }
}

/// Decode values `start..start + out.len()`, reading only the blocks they fall
/// in. Unlike the packed kernels, this works for a range that starts anywhere,
/// even partway through a block or a byte.
fn decode_values<S: Scale>(quantized: &Quantized<S>, start: usize, out: &mut [f32]) {
    let block = quantized.block();
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
        Quantized::Adaptive { block_bits, .. } => {
            // Each block is packed at its own width, so the first block's bytes
            // start after those of every block before it. Only the last block
            // can be short, so those blocks are full, and when `block` is a
            // multiple of 8, `block` codes of `bit_width` bits fill exactly
            // `block / 8 × bit_width` bytes: adding up the widths is enough.
            let blocks_before = &block_bits[..start / block];
            let first_byte: usize = if block.is_multiple_of(8) {
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
            dequant_adaptive(quantized, start, first_byte, out);
        }
    }
}

const LANES: usize = 16;

/// Multiply two slices element by element and add up the products.
///
/// Float addition is not associative, so with one running total the compiler
/// must add the products in order, one at a time. Sixteen separate totals are
/// independent, so it can add them side by side in SIMD registers.
fn dot(left: &[f32], right: &[f32]) -> f32 {
    let (left_chunks, left_remainder) = left.as_chunks::<LANES>();
    let (right_chunks, right_remainder) = right.as_chunks::<LANES>();
    let remainder = left_remainder.iter().zip(right_remainder);
    let remainder_total: f32 = remainder.map(|(a, b)| a * b).sum();

    let mut totals = [0.0_f32; LANES];
    add_products(&mut totals, left_chunks, right_chunks);
    totals.iter().sum::<f32>() + remainder_total
}

/// Add each product `left[i][lane] × right[i][lane]` to `totals[lane]`.
#[inline]
fn add_products(totals: &mut [f32; LANES], left: &[[f32; LANES]], right: &[[f32; LANES]]) {
    for (left_chunk, right_chunk) in left.iter().zip(right) {
        for ((total, a), b) in totals.iter_mut().zip(left_chunk).zip(right_chunk) {
            *total += a * b;
        }
    }
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
    fn one_vector_gives_bit_for_bit_what_it_gives_in_a_batch() {
        // 4-bit and 8-bit blocks of 32 and 64 multiply a single vector as
        // they decode it. Every other layout decodes each row first, whatever
        // the batch.
        let values: Vec<f32> = (0..6 * 128).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut wide_rows = [
            symmetric::quantize_with(&values, 4, 32).unwrap(),
            symmetric::quantize_with(&values, 4, 64).unwrap(),
            symmetric::quantize_with(&values, 8, 32).unwrap(),
            symmetric::quantize_with(&values, 8, 64).unwrap(),
        ];
        for matrix in &mut wide_rows {
            matrix.set_shape(6, 128).unwrap();
        }
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        for quantized in wide_rows.into_iter().chain(matrices_in_every_layout()) {
            let (rows, columns) = quantized.shape().unwrap();
            let inputs: Vec<f32> = (0..2 * columns).map(|i| (i as f32 * 0.11).cos()).collect();
            let batch = quantized.matmul(&inputs).unwrap();
            for (vector, input) in inputs.chunks_exact(columns).enumerate() {
                let alone = quantized.matmul(input).unwrap();
                assert_eq!(
                    bits(&alone),
                    bits(&batch[vector * rows..(vector + 1) * rows])
                );
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
