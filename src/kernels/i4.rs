//! Symmetric 4-bit: two codes per byte, low nibble first.

use crate::error::Result;
use crate::packed::{Packed, nbytes};
use crate::params::symmetric_scale;
use crate::scale::{Scale, store_scale};

use super::reduce::signed_extreme;

pub(crate) fn pack_sym_i4<S: Scale>(values: &[f32], block: usize) -> Result<(Vec<S>, Packed)> {
    let mut scales = Vec::with_capacity(values.len().div_ceil(block));
    let mut bytes = vec![0u8; nbytes(values.len(), 4)];
    let mut i = 0usize;
    for (block_index, chunk) in values.chunks(block).enumerate() {
        let scale: S = store_scale(symmetric_scale(signed_extreme(chunk), 4), block_index)?;
        scales.push(scale);
        quant_chunk(chunk, scale.to_f32(), &mut bytes, &mut i);
    }
    Ok((scales, Packed::from_raw(bytes, 4, values.len())))
}

fn quant_chunk(values: &[f32], scale: f32, bytes: &mut [u8], value_index: &mut usize) {
    let one_over_scale = 1.0 / scale;
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    if value_index.is_multiple_of(2) {
        // SAFETY: 16 floats → 8 packed bytes.
        unsafe {
            while i + 16 <= values.len() {
                quant_16(
                    values.as_ptr().add(i),
                    one_over_scale,
                    bytes.as_mut_ptr().add(*value_index / 2),
                );
                i += 16;
                *value_index += 16;
            }
        }
    }
    while i < values.len() {
        let code = (values[i] * one_over_scale).round().clamp(-8.0, 7.0) as i32;
        let byte = *value_index / 2;
        if value_index.is_multiple_of(2) {
            bytes[byte] = (code as u8) & 0x0F;
        } else {
            bytes[byte] |= ((code as u8) & 0x0F) << 4;
        }
        *value_index += 1;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn quant_16(src: *const f32, inv: f32, dst: *mut u8) {
    unsafe {
        use core::arch::aarch64::*;
        let vinv = vdupq_n_f32(inv);
        let vmin = vdupq_n_f32(-8.0);
        let vmax = vdupq_n_f32(7.0);
        let q = |v| {
            vcvtq_s32_f32(vmaxq_f32(
                vminq_f32(vrndaq_f32(vmulq_f32(v, vinv)), vmax),
                vmin,
            ))
        };
        let p0 = vcombine_s16(
            vmovn_s32(q(vld1q_f32(src))),
            vmovn_s32(q(vld1q_f32(src.add(4)))),
        );
        let p1 = vcombine_s16(
            vmovn_s32(q(vld1q_f32(src.add(8)))),
            vmovn_s32(q(vld1q_f32(src.add(12)))),
        );
        let codes = vcombine_s8(vmovn_s16(p0), vmovn_s16(p1));
        let masked = vandq_u8(vreinterpretq_u8_s8(codes), vdupq_n_u8(0x0F));
        vst1_u8(
            dst,
            vget_low_u8(vorrq_u8(
                vuzp1q_u8(masked, masked),
                vshlq_n_u8(vuzp2q_u8(masked, masked), 4),
            )),
        );
    }
}

pub(crate) fn dequant_i4_blocks<S: Scale>(
    scales: &[S],
    bytes: &[u8],
    block: usize,
    out: &mut [f32],
) {
    assert!(bytes.len() >= nbytes(out.len(), 4));
    let mut i = 0usize;
    for (bi, chunk) in out.chunks_mut(block).enumerate() {
        let s = scales[bi].to_f32();
        let mut j = 0;
        // Value `i` is in byte `i / 2`: its low nibble when `i` is even, its
        // high nibble when `i` is odd. So after a block of odd length, the
        // next block starts on a high nibble.
        if !i.is_multiple_of(2) {
            chunk[0] = high_code(bytes[i / 2]) as f32 * s;
            i += 1;
            j += 1;
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: 32 codes = 16 packed bytes, which the assert above
            // guarantees are inside `bytes`.
            unsafe {
                while j + 32 <= chunk.len() {
                    dequant_32(bytes.as_ptr().add(i / 2), s, chunk.as_mut_ptr().add(j));
                    i += 32;
                    j += 32;
                }
            }
        }
        // Decode both of a byte's codes at once. With no branch on which
        // nibble comes next, the compiler can decode many bytes side by side
        // in SIMD registers.
        let (pairs, _) = chunk[j..].as_chunks_mut::<2>();
        for (pair, &byte) in pairs.iter_mut().zip(&bytes[i / 2..]) {
            *pair = [low_code(byte) as f32 * s, high_code(byte) as f32 * s];
        }
        i += 2 * pairs.len();
        j += 2 * pairs.len();
        // And a block of odd length ends on a low nibble.
        if j < chunk.len() {
            chunk[j] = low_code(bytes[i / 2]) as f32 * s;
            i += 1;
        }
    }
}

/// The code in a byte's low nibble. Shifting it to the top of the byte and
/// back with an arithmetic shift copies its sign bit into the upper four bits.
fn low_code(byte: u8) -> i8 {
    ((byte << 4) as i8) >> 4
}

/// The code in a byte's high nibble, sign-extended the same way.
fn high_code(byte: u8) -> i8 {
    (byte as i8) >> 4
}

/// Decode 16 bytes into their 32 codes times `scale`, as
/// [`dequant_i4_blocks`] would. With a fixed size, the compiler can keep all
/// 32 values in SIMD registers, so a caller can multiply them without storing
/// them first.
#[inline]
pub(crate) fn decode_32(bytes: &[u8; 16], scale: f32) -> [f32; 32] {
    let mut values = [0.0; 32];
    for (pair, &byte) in values.as_chunks_mut::<2>().0.iter_mut().zip(bytes) {
        *pair = [
            low_code(byte) as f32 * scale,
            high_code(byte) as f32 * scale,
        ];
    }
    values
}

// `dequant_i4_blocks` is generic, so it's compiled in each crate that calls
// it. Without `#[inline]`, the code it compiles to there calls these two
// functions in this crate every 32 codes.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn dequant_32(src: *const u8, scale: f32, dst: *mut f32) {
    unsafe {
        use core::arch::aarch64::*;
        let raw = vld1q_u8(src);
        let lo = vshrq_n_s8(
            vshlq_n_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), 4),
            4,
        );
        let hi = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), 4), 4);
        store_i8x16(vzip1q_s8(lo, hi), scale, dst);
        store_i8x16(vzip2q_s8(lo, hi), scale, dst.add(16));
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn store_i8x16(q: core::arch::aarch64::int8x16_t, scale: f32, dst: *mut f32) {
    unsafe {
        use core::arch::aarch64::*;
        let lo = vmovl_s8(vget_low_s8(q));
        let hi = vmovl_s8(vget_high_s8(q));
        let vs = vdupq_n_f32(scale);
        vst1q_f32(
            dst,
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vs),
        );
        vst1q_f32(
            dst.add(4),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vs),
        );
        vst1q_f32(
            dst.add(8),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vs),
        );
        vst1q_f32(
            dst.add(12),
            vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vs),
        );
    }
}

pub(crate) fn dot_i4_blocks<S: Scale>(
    scales: &[S],
    bytes: &[u8],
    block: usize,
    rhs: &[f32],
) -> f32 {
    let mut acc = 0.0_f32;
    let mut value_index = 0;
    for (block_index, chunk) in rhs.chunks(block).enumerate() {
        let mut inner = 0.0_f32;
        for &x in chunk {
            let byte = bytes[value_index / 2];
            let code = if value_index.is_multiple_of(2) {
                low_code(byte)
            } else {
                high_code(byte)
            };
            inner += code as f32 * x;
            value_index += 1;
        }
        acc += scales[block_index].to_f32() * inner;
    }
    acc
}
