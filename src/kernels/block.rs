//! Arbitrary bit-width and asymmetric loops. 4/8-bit symmetric bypasses this.

use crate::packed::Packed;
use crate::params::{asymmetric_params, largest_code, smallest_code, symmetric_scale};

use super::i4::pack_sym_i4;
use super::i8::pack_sym_i8;
use super::reduce::{abs_max, min_max};

pub(crate) fn quantize_sym_packed(values: &[f32], bits: u32, block: usize) -> (Vec<f32>, Packed) {
    match bits {
        8 => pack_sym_i8(values, block),
        4 => pack_sym_i4(values, block),
        _ => pack_sym_general(values, bits, block),
    }
}

fn pack_sym_general(values: &[f32], bits: u32, block: usize) -> (Vec<f32>, Packed) {
    let mut scales = Vec::with_capacity(values.len().div_ceil(block));
    let mut codes = Vec::with_capacity(values.len());
    for chunk in values.chunks(block) {
        let scale = symmetric_scale(abs_max(chunk), bits);
        let one_over_scale = 1.0 / scale;
        let code_min = smallest_code(bits) as f32;
        let code_max = largest_code(bits) as f32;
        for &value in chunk {
            codes.push((value * one_over_scale).round().clamp(code_min, code_max) as i32);
        }
        scales.push(scale);
    }
    (scales, Packed::from_i32s(&codes, bits))
}

pub(crate) fn quantize_asym_block(block: &[f32], bits: u32, codes: &mut Vec<i32>) -> (f32, f32) {
    let (lowest, highest) = min_max(block);
    let (scale, zero_point) = asymmetric_params(lowest, highest, bits);
    let one_over_scale = 1.0 / scale;
    let code_min = smallest_code(bits);
    let code_max = largest_code(bits);
    for &value in block {
        let code = (value * one_over_scale + zero_point).round() as i32;
        codes.push(code.clamp(code_min, code_max));
    }
    (scale, zero_point)
}

pub(crate) fn dequant_sym_into(scales: &[f32], packed: &Packed, block: usize, out: &mut [f32]) {
    let mut codes = vec![0i32; packed.len()];
    packed.unpack_into(&mut codes);
    for (index, &code) in codes.iter().enumerate() {
        out[index] = code as f32 * scales[index / block];
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
    for (index, &code) in codes.iter().enumerate() {
        let scale = scales[index / block];
        let zero_point = zero_points[index / block];
        out[index] = (code as f32 - zero_point) * scale;
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
