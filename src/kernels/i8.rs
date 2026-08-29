//! Symmetric 8-bit: one i8 per value. NEON convert on aarch64.

use crate::packed::Packed;
use crate::params::symmetric_scale;

use super::reduce::abs_max;

pub(crate) fn pack_sym_i8(values: &[f32], block: usize) -> (Vec<f32>, Packed) {
    let mut scales = Vec::with_capacity(values.len().div_ceil(block));
    let mut bytes = vec![0u8; values.len()];
    let mut off = 0;
    for chunk in values.chunks(block) {
        let scale = symmetric_scale(abs_max(chunk), 8);
        scales.push(scale);
        quant_chunk(chunk, scale, &mut bytes[off..off + chunk.len()]);
        off += chunk.len();
    }
    (scales, Packed::from_raw(bytes, 8, values.len()))
}

fn quant_chunk(values: &[f32], scale: f32, out: &mut [u8]) {
    let one_over_scale = 1.0 / scale;
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `i + 16 <= len`.
        unsafe {
            while i + 16 <= values.len() {
                quant_16(
                    values.as_ptr().add(i),
                    one_over_scale,
                    out.as_mut_ptr().add(i),
                );
                i += 16;
            }
        }
    }
    while i < values.len() {
        out[i] = (values[i] * one_over_scale).round().clamp(-128.0, 127.0) as i32 as i8 as u8;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn quant_16(src: *const f32, inv: f32, dst: *mut u8) {
    use core::arch::aarch64::*;
    let vinv = vdupq_n_f32(inv);
    let vmin = vdupq_n_f32(-128.0);
    let vmax = vdupq_n_f32(127.0);
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
    vst1q_s8(dst.cast(), vcombine_s8(vmovn_s16(p0), vmovn_s16(p1)));
}

pub(crate) fn dequant_i8_blocks(scales: &[f32], bytes: &[u8], block: usize, out: &mut [f32]) {
    let mut off = 0;
    for (bi, chunk) in out.chunks_mut(block).enumerate() {
        dequant_chunk(&bytes[off..off + chunk.len()], scales[bi], chunk);
        off += chunk.len();
    }
}

fn dequant_chunk(bytes: &[u8], scale: f32, out: &mut [f32]) {
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `i + 16 <= len`.
        unsafe {
            while i + 16 <= out.len() {
                dequant_16(bytes.as_ptr().add(i), scale, out.as_mut_ptr().add(i));
                i += 16;
            }
        }
    }
    while i < out.len() {
        out[i] = (bytes[i] as i8 as f32) * scale;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dequant_16(src: *const u8, scale: f32, dst: *mut f32) {
    use core::arch::aarch64::*;
    let q = vld1q_s8(src.cast());
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

pub(crate) fn dot_i8_blocks(scales: &[f32], bytes: &[u8], block: usize, rhs: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    let mut off = 0;
    for (bi, chunk) in rhs.chunks(block).enumerate() {
        let mut inner = 0.0_f32;
        for (j, &x) in chunk.iter().enumerate() {
            inner += (bytes[off + j] as i8 as f32) * x;
        }
        acc += scales[bi] * inner;
        off += chunk.len();
    }
    acc
}
