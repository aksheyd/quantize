//! Symmetric 4-bit: two codes per byte, low nibble first.

use crate::packed::{nbytes, Packed};
use crate::params::symmetric_scale;

use super::reduce::abs_max;

pub(crate) fn pack_sym_i4(values: &[f32], block: usize) -> (Vec<f32>, Packed) {
    let mut scales = Vec::with_capacity(values.len().div_ceil(block));
    let mut bytes = vec![0u8; nbytes(values.len(), 4)];
    let mut i = 0usize;
    for chunk in values.chunks(block) {
        let scale = symmetric_scale(abs_max(chunk), 4);
        scales.push(scale);
        quant_chunk(chunk, scale, &mut bytes, &mut i);
    }
    (scales, Packed::from_raw(bytes, 4, values.len()))
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

pub(crate) fn dequant_i4_blocks(scales: &[f32], bytes: &[u8], block: usize, out: &mut [f32]) {
    let mut i = 0usize;
    for (bi, chunk) in out.chunks_mut(block).enumerate() {
        let s = scales[bi];
        let mut j = 0;
        #[cfg(target_arch = "aarch64")]
        if i.is_multiple_of(2) {
            // SAFETY: 32 codes = 16 packed bytes.
            unsafe {
                while j + 32 <= chunk.len() {
                    dequant_32(bytes.as_ptr().add(i / 2), s, chunk.as_mut_ptr().add(j));
                    i += 32;
                    j += 32;
                }
            }
        }
        while j < chunk.len() {
            let byte = bytes[i / 2];
            let nib = if i.is_multiple_of(2) {
                byte & 0x0F
            } else {
                byte >> 4
            };
            chunk[j] = ((((nib as i8) << 4) >> 4) as i32 as f32) * s;
            i += 1;
            j += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dequant_32(src: *const u8, scale: f32, dst: *mut f32) {
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

#[cfg(target_arch = "aarch64")]
unsafe fn store_i8x16(q: core::arch::aarch64::int8x16_t, scale: f32, dst: *mut f32) {
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

pub(crate) fn dot_i4_blocks(scales: &[f32], bytes: &[u8], block: usize, rhs: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mut value_index = 0;
    for (block_index, chunk) in rhs.chunks(block).enumerate() {
        let mut inner = 0.0_f32;
        for &x in chunk {
            let byte = bytes[value_index / 2];
            let nibble = if value_index.is_multiple_of(2) {
                byte & 0x0F
            } else {
                byte >> 4
            };
            inner += (((nibble as i8) << 4) >> 4) as f32 * x;
            value_index += 1;
        }
        acc += scales[block_index] * inner;
    }
    acc
}
