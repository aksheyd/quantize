//! Arbitrary bit-width and asymmetric loops. 4/8-bit symmetric bypasses this.

use crate::error::Result;
use crate::packed::Packed;
use crate::params::{asymmetric_params, largest_code, smallest_code, symmetric_scale};
use crate::scale::{store_scale, store_zero_point, Scale};

use super::i4::pack_sym_i4;
use super::i8::pack_sym_i8;
use super::reduce::{min_max, signed_extreme};

/// Symmetric codes, plus each block's scale as stored in `S`.
///
/// Codes are picked against the stored scale, since that is what decoding
/// multiplies by. f16 and bf16 round a scale when they store it: bf16 keeps 8
/// significant bits, so its scale can be off by 1/256, and at 8 bits a code of
/// -128 picked against the unrounded scale would decode half a tick away.
/// [`store_scale`] rounds away from zero, so the value farthest from zero
/// still has a code.
pub(crate) fn quantize_sym_packed<S: Scale>(
    values: &[f32],
    bits: u32,
    block: usize,
) -> Result<(Vec<S>, Packed)> {
    match bits {
        8 => pack_sym_i8(values, block),
        4 => pack_sym_i4(values, block),
        _ => pack_sym_general(values, bits, block),
    }
}

fn pack_sym_general<S: Scale>(values: &[f32], bits: u32, block: usize) -> Result<(Vec<S>, Packed)> {
    let mut scales = Vec::with_capacity(values.len().div_ceil(block));
    let mut codes = Vec::with_capacity(values.len());
    for (block_index, chunk) in values.chunks(block).enumerate() {
        let scale: S = store_scale(symmetric_scale(signed_extreme(chunk), bits), block_index)?;
        let one_over_scale = 1.0 / scale.to_f32();
        let code_min = smallest_code(bits) as f32;
        let code_max = largest_code(bits) as f32;
        for &value in chunk {
            codes.push((value * one_over_scale).round().clamp(code_min, code_max) as i32);
        }
        scales.push(scale);
    }
    Ok((scales, Packed::from_i32s(&codes, bits)))
}

/// Asymmetric codes for one block, plus its scale and zero-point as stored in
/// `S`. As in [`quantize_sym_packed`], codes are picked against the stored
/// pair. That matters most for the zero-point: for values near 100 it is about
/// -1451, which f16 rounds to a whole number and bf16 to a multiple of 8.
pub(crate) fn quantize_asym_block<S: Scale>(
    block: &[f32],
    block_index: usize,
    bits: u32,
    codes: &mut Vec<i32>,
) -> Result<(S, S)> {
    let (lowest, highest) = min_max(block);
    let (scale, mut zero_point) = asymmetric_params(lowest, highest, bits);
    let scale: S = store_scale(scale, block_index)?;
    if lowest < highest {
        // The stored scale can be a little wider than the range needs, so
        // place the zero-point again from it: `lowest` stays on the smallest
        // code, and `highest` lands on or below the largest.
        zero_point = smallest_code(bits) as f32 - lowest / scale.to_f32();
    }
    let zero_point: S = store_zero_point(zero_point, block_index)?;
    let one_over_scale = 1.0 / scale.to_f32();
    let code_min = smallest_code(bits);
    let code_max = largest_code(bits);
    for &value in block {
        let code = (value * one_over_scale + zero_point.to_f32()).round() as i32;
        codes.push(code.clamp(code_min, code_max));
    }
    Ok((scale, zero_point))
}

pub(crate) fn dequant_sym_into(scales: &[f32], packed: &Packed, block: usize, out: &mut [f32]) {
    let mut codes = vec![0i32; packed.len()];
    packed.unpack_into(&mut codes);
    let blocks = out.chunks_mut(block).zip(codes.chunks(block));
    for (block_index, (block_out, block_codes)) in blocks.enumerate() {
        let scale = scales[block_index];
        for (slot, &code) in block_out.iter_mut().zip(block_codes) {
            *slot = code as f32 * scale;
        }
    }
}

pub(crate) fn dequant_asym_into(
    scales: &[f32],
    zero_points: &[f32],
    packed: &Packed,
    block: usize,
    out: &mut [f32],
) {
    let mut codes = vec![0i32; packed.len()];
    packed.unpack_into(&mut codes);
    let blocks = out.chunks_mut(block).zip(codes.chunks(block));
    for (block_index, (block_out, block_codes)) in blocks.enumerate() {
        let scale = scales[block_index];
        let zero_point = zero_points[block_index];
        for (slot, &code) in block_out.iter_mut().zip(block_codes) {
            *slot = (code as f32 - zero_point) * scale;
        }
    }
}

pub(crate) fn dot_sym(scales: &[f32], packed: &Packed, block: usize, rhs: &[f32]) -> f32 {
    let mut codes = vec![0i32; packed.len()];
    packed.unpack_into(&mut codes);
    let mut total = 0.0_f32;
    for (index, &code) in codes.iter().enumerate() {
        total += code as f32 * scales[index / block] * rhs[index];
    }
    total
}

pub(crate) fn dot_asym(
    scales: &[f32],
    zero_points: &[f32],
    packed: &Packed,
    block: usize,
    rhs: &[f32],
) -> f32 {
    let mut codes = vec![0i32; packed.len()];
    packed.unpack_into(&mut codes);
    let mut total = 0.0_f32;
    for (index, &code) in codes.iter().enumerate() {
        let scale = scales[index / block];
        let zero_point = zero_points[index / block];
        total += (code as f32 - zero_point) * scale * rhs[index];
    }
    total
}
