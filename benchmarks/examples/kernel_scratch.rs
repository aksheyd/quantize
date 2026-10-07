#![allow(clippy::all, unused_mut, dead_code)]
//! Scratch: batch-1 matmul kernel variants for 4-bit, block-32, f16 rows.
//! Not committed.

use half::f16;
use quantize::{Quantized, quantize};
use std::hint::black_box;
use std::time::Instant;

const HIDDEN: usize = 576;
const KV: usize = 192;
const INTERMEDIATE: usize = 1536;
const VOCAB: usize = 49152;
const LAYERS: usize = 30;
const BLOCK: usize = 32;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut seed = seed;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.05
        })
        .collect()
}

// ---------- shared helpers ----------

/// The crate's 16-lane dot.
fn dot16(left: &[f32], right: &[f32]) -> f32 {
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

#[inline(always)]
fn low_code(byte: u8) -> i8 {
    ((byte << 4) as i8) >> 4
}

#[inline(always)]
fn high_code(byte: u8) -> i8 {
    (byte as i8) >> 4
}

// ---------- copy of the crate's current 4-bit decode (NEON on aarch64) ----------

fn dequant_i4_blocks_copy(scales: &[f16], bytes: &[u8], block: usize, out: &mut [f32]) {
    assert!(bytes.len() >= (out.len() * 4).div_ceil(8));
    let mut i = 0usize;
    for (bi, chunk) in out.chunks_mut(block).enumerate() {
        let s = scales[bi].to_f32();
        let mut j = 0;
        #[cfg(target_arch = "aarch64")]
        if i.is_multiple_of(2) {
            unsafe {
                while j + 32 <= chunk.len() {
                    neon::dequant_32(bytes.as_ptr().add(i / 2), s, chunk.as_mut_ptr().add(j));
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

// ---------- copy of the crate's current fused dot (scalar, branchy) ----------

fn dot_i4_blocks_copy(scales: &[f16], bytes: &[u8], block: usize, rhs: &[f32]) -> f32 {
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
        acc += scales[block_index].to_f32() * inner;
    }
    acc
}

// ---------- portable branch-free decode (pairs) ----------

fn dequant_i4_pairs(scales: &[f16], bytes: &[u8], block: usize, out: &mut [f32]) {
    // block even: each block starts on a byte.
    for ((block_out, block_bytes), scale) in out
        .chunks_mut(block)
        .zip(bytes.chunks(block / 2))
        .zip(scales)
    {
        let scale = scale.to_f32();
        let (pairs, _) = block_out.as_chunks_mut::<2>();
        for (pair, &byte) in pairs.iter_mut().zip(block_bytes) {
            pair[0] = low_code(byte) as f32 * scale;
            pair[1] = high_code(byte) as f32 * scale;
        }
    }
}

// ---------- fused, bit-identical to decode + dot16 (block % 16 == 0, columns % 16 == 0) ----------

fn dot_i4_bitexact(scales: &[f16], bytes: &[u8], block: usize, x: &[f32]) -> f32 {
    let mut totals = [0.0_f32; 16];
    for ((block_bytes, block_x), scale) in bytes.chunks(block / 2).zip(x.chunks(block)).zip(scales)
    {
        let scale = scale.to_f32();
        let (byte_chunks, _) = block_bytes.as_chunks::<8>();
        let (x_chunks, _) = block_x.as_chunks::<16>();
        for (eight, sixteen) in byte_chunks.iter().zip(x_chunks) {
            for i in 0..8 {
                let low = low_code(eight[i]) as f32 * scale;
                let high = high_code(eight[i]) as f32 * scale;
                totals[2 * i] += low * sixteen[2 * i];
                totals[2 * i + 1] += high * sixteen[2 * i + 1];
            }
        }
    }
    totals.iter().sum::<f32>() + 0.0
}

// Same as bitexact, but decode 16 weights into a small array first, then the dot16 inner loop.
fn dot_i4_bitexact_staged(scales: &[f16], bytes: &[u8], block: usize, x: &[f32]) -> f32 {
    let mut totals = [0.0_f32; 16];
    for ((block_bytes, block_x), scale) in bytes.chunks(block / 2).zip(x.chunks(block)).zip(scales)
    {
        let scale = scale.to_f32();
        let (byte_chunks, _) = block_bytes.as_chunks::<8>();
        let (x_chunks, _) = block_x.as_chunks::<16>();
        for (eight, sixteen) in byte_chunks.iter().zip(x_chunks) {
            let mut weights = [0.0_f32; 16];
            for (pair, &byte) in weights.as_chunks_mut::<2>().0.iter_mut().zip(eight) {
                *pair = [
                    low_code(byte) as f32 * scale,
                    high_code(byte) as f32 * scale,
                ];
            }
            for ((total, weight), input) in totals.iter_mut().zip(weights).zip(sixteen) {
                *total += weight * input;
            }
        }
    }
    totals.iter().sum::<f32>() + 0.0
}

// ---------- fused, factored scale: Σ code·x per block, × scale once ----------

fn dot_i4_factored(scales: &[f16], bytes: &[u8], block: usize, x: &[f32]) -> f32 {
    let mut row_totals = [0.0_f32; 8];
    for ((block_bytes, block_x), scale) in bytes.chunks(block / 2).zip(x.chunks(block)).zip(scales)
    {
        let mut block_totals = [0.0_f32; 8];
        let (byte_chunks, _) = block_bytes.as_chunks::<8>();
        let (x_chunks, _) = block_x.as_chunks::<16>();
        for (eight, sixteen) in byte_chunks.iter().zip(x_chunks) {
            for i in 0..8 {
                let low = low_code(eight[i]) as f32;
                let high = high_code(eight[i]) as f32;
                block_totals[i] += low * sixteen[2 * i] + high * sixteen[2 * i + 1];
            }
        }
        let scale = scale.to_f32();
        for (row_total, block_total) in row_totals.iter_mut().zip(block_totals) {
            *row_total += scale * block_total;
        }
    }
    row_totals.iter().sum()
}

// Factored, but staged: codes into [f32;16] in order, then 16-lane multiply-add per block.
fn dot_i4_factored_staged(scales: &[f16], bytes: &[u8], block: usize, x: &[f32]) -> f32 {
    let mut row_totals = [0.0_f32; 16];
    for ((block_bytes, block_x), scale) in bytes.chunks(block / 2).zip(x.chunks(block)).zip(scales)
    {
        let mut block_totals = [0.0_f32; 16];
        let (byte_chunks, _) = block_bytes.as_chunks::<8>();
        let (x_chunks, _) = block_x.as_chunks::<16>();
        for (eight, sixteen) in byte_chunks.iter().zip(x_chunks) {
            let mut codes = [0.0_f32; 16];
            for (pair, &byte) in codes.as_chunks_mut::<2>().0.iter_mut().zip(eight) {
                *pair = [low_code(byte) as f32, high_code(byte) as f32];
            }
            for ((total, code), input) in block_totals.iter_mut().zip(codes).zip(sixteen) {
                *total += code * input;
            }
        }
        let scale = scale.to_f32();
        for (row_total, block_total) in row_totals.iter_mut().zip(block_totals) {
            *row_total += scale * block_total;
        }
    }
    row_totals.iter().sum()
}

// ---------- NEON fused ----------

#[cfg(target_arch = "aarch64")]
mod neon {
    use core::arch::aarch64::*;

    pub unsafe fn dequant_32(src: *const u8, scale: f32, dst: *mut f32) {
        unsafe {
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

    unsafe fn store_i8x16(q: int8x16_t, scale: f32, dst: *mut f32) {
        unsafe {
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

    /// Σ code·x over 32 codes, as 4 lanes.
    unsafe fn dot_32(src: *const u8, x: *const f32) -> float32x4_t {
        unsafe {
            let raw = vreinterpretq_s8_u8(vld1q_u8(src));
            let lo = vshrq_n_s8(vshlq_n_s8(raw, 4), 4);
            let hi = vshrq_n_s8(raw, 4);
            let a = vzip1q_s8(lo, hi);
            let b = vzip2q_s8(lo, hi);
            let a0 = vmovl_s8(vget_low_s8(a));
            let a1 = vmovl_high_s8(a);
            let b0 = vmovl_s8(vget_low_s8(b));
            let b1 = vmovl_high_s8(b);
            let mut acc0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0))), vld1q_f32(x));
            let mut acc1 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(a0)), vld1q_f32(x.add(4)));
            acc0 = vfmaq_f32(
                acc0,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(a1))),
                vld1q_f32(x.add(8)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vcvtq_f32_s32(vmovl_high_s16(a1)),
                vld1q_f32(x.add(12)),
            );
            acc0 = vfmaq_f32(
                acc0,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(b0))),
                vld1q_f32(x.add(16)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vcvtq_f32_s32(vmovl_high_s16(b0)),
                vld1q_f32(x.add(20)),
            );
            acc0 = vfmaq_f32(
                acc0,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(b1))),
                vld1q_f32(x.add(24)),
            );
            acc1 = vfmaq_f32(
                acc1,
                vcvtq_f32_s32(vmovl_high_s16(b1)),
                vld1q_f32(x.add(28)),
            );
            vaddq_f32(acc0, acc1)
        }
    }

    /// Bit-exact with decode + dot16: 16 lanes as 4 registers, mul then add.
    pub fn dot_row_bitexact(scales: &[half::f16], bytes: &[u8], x: &[f32]) -> f32 {
        assert_eq!(bytes.len() * 2, x.len());
        assert_eq!(scales.len() * 32, x.len());
        unsafe {
            let mut totals = [vdupq_n_f32(0.0); 4];
            for (block_index, scale) in scales.iter().enumerate() {
                let vs = vdupq_n_f32(scale.to_f32());
                let raw = vreinterpretq_s8_u8(vld1q_u8(bytes.as_ptr().add(block_index * 16)));
                let lo = vshrq_n_s8(vshlq_n_s8(raw, 4), 4);
                let hi = vshrq_n_s8(raw, 4);
                for (half, codes) in [vzip1q_s8(lo, hi), vzip2q_s8(lo, hi)]
                    .into_iter()
                    .enumerate()
                {
                    let xs = x.as_ptr().add(block_index * 32 + half * 16);
                    let w0 = vmovl_s8(vget_low_s8(codes));
                    let w1 = vmovl_high_s8(codes);
                    let weights = [
                        vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0))),
                        vcvtq_f32_s32(vmovl_high_s16(w0)),
                        vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1))),
                        vcvtq_f32_s32(vmovl_high_s16(w1)),
                    ];
                    for k in 0..4 {
                        let weight = vmulq_f32(weights[k], vs);
                        totals[k] =
                            vaddq_f32(totals[k], vmulq_f32(weight, vld1q_f32(xs.add(4 * k))));
                    }
                }
            }
            let mut lanes = [0.0f32; 16];
            for k in 0..4 {
                vst1q_f32(lanes.as_mut_ptr().add(4 * k), totals[k]);
            }
            lanes.iter().sum::<f32>() + 0.0
        }
    }

    pub fn dot_row(scales: &[half::f16], bytes: &[u8], x: &[f32]) -> f32 {
        assert_eq!(bytes.len() * 2, x.len());
        assert_eq!(scales.len() * 32, x.len());
        unsafe {
            let mut acc = vdupq_n_f32(0.0);
            for (block_index, scale) in scales.iter().enumerate() {
                let block = dot_32(
                    bytes.as_ptr().add(block_index * 16),
                    x.as_ptr().add(block_index * 32),
                );
                acc = vfmaq_n_f32(acc, block, scale.to_f32());
            }
            vaddvq_f32(acc)
        }
    }
}

// ---------- variants over the whole matrix ----------

type Kernel = fn(&Quantized<f16>, &[f32], &mut [f32]);

fn crate_matmul(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    q.matmul_into(input, out).unwrap();
}

fn rows_of<'a>(q: &'a Quantized<f16>) -> (usize, impl Iterator<Item = (&'a [u8], &'a [f16])>) {
    let (_, columns) = q.shape().unwrap();
    let rows = q
        .codes()
        .chunks_exact(columns / 2)
        .zip(q.scales().chunks_exact(columns / BLOCK));
    (columns, rows)
}

fn decode_dot_hoisted(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (columns, rows) = rows_of(q);
    let mut row_weights = vec![0.0; columns];
    for ((bytes, scales), slot) in rows.zip(out) {
        dequant_i4_blocks_copy(scales, bytes, BLOCK, &mut row_weights);
        *slot = dot16(&row_weights, input);
    }
}

fn portable_decode_dot(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (columns, rows) = rows_of(q);
    let mut row_weights = vec![0.0; columns];
    for ((bytes, scales), slot) in rows.zip(out) {
        dequant_i4_pairs(scales, bytes, BLOCK, &mut row_weights);
        *slot = dot16(&row_weights, input);
    }
}

fn fused_existing_dot(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = dot_i4_blocks_copy(scales, bytes, BLOCK, input);
    }
}

fn fused_bitexact(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = dot_i4_bitexact(scales, bytes, BLOCK, input);
    }
}

fn fused_bitexact_staged(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = dot_i4_bitexact_staged(scales, bytes, BLOCK, input);
    }
}

fn fused_factored(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = dot_i4_factored(scales, bytes, BLOCK, input);
    }
}

fn fused_factored_staged(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = dot_i4_factored_staged(scales, bytes, BLOCK, input);
    }
}

#[cfg(target_arch = "aarch64")]
fn fused_neon(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = neon::dot_row(scales, bytes, input);
    }
}

#[cfg(target_arch = "aarch64")]
fn fused_neon_bitexact(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = neon::dot_row_bitexact(scales, bytes, input);
    }
}

/// Whole-tensor 4-bit decode: crate (NEON on aarch64) vs portable pairs.
fn time_whole_decode() {
    let side = 1024;
    let weights = values(side * side, 7);
    let q = quantize::<f16, 4, 32>(&weights).unwrap();
    let mut out = vec![0.0f32; side * side];
    let mut time = |f: &mut dyn FnMut(&mut [f32])| {
        for _ in 0..20 {
            f(&mut out);
        }
        let mut samples: Vec<f64> = (0..50)
            .map(|_| {
                let start = Instant::now();
                f(&mut out);
                black_box(&out);
                start.elapsed().as_secs_f64()
            })
            .collect();
        samples.sort_by(f64::total_cmp);
        samples[25] * 1e9 / (side * side) as f64
    };
    let crate_ns = time(&mut |out| q.dequantize_into(out).unwrap());
    let reference = q.dequantize();
    let portable_ns = time(&mut |out| dequant_i4_pairs(q.scales(), q.codes(), BLOCK, out));
    dequant_i4_pairs(q.scales(), q.codes(), BLOCK, &mut out);
    let same = out
        .iter()
        .zip(&reference)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "\nwhole 1024x1024 4-bit dequantize_into, ns/value: crate {crate_ns:.3}, portable pairs {portable_ns:.3}, bitexact {same}"
    );
}

fn main() {
    let shapes = [
        (HIDDEN, HIDDEN),
        (KV, HIDDEN),
        (KV, HIDDEN),
        (HIDDEN, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (HIDDEN, INTERMEDIATE),
    ];
    let mut layers: Vec<Quantized<f16>> = Vec::new();
    for layer in 0..LAYERS {
        for (index, &(rows, columns)) in shapes.iter().enumerate() {
            let weights = values(rows * columns, (layer * 7 + index) as u32 + 1);
            let mut q = quantize::<f16, 4, 32>(&weights).unwrap();
            q.set_shape(rows, columns).unwrap();
            layers.push(q);
        }
    }
    let mut lm = quantize::<f16, 4, 32>(&values(VOCAB * HIDDEN, 999)).unwrap();
    lm.set_shape(VOCAB, HIDDEN).unwrap();
    layers.push(lm);

    let input_hidden = values(HIDDEN, 4242);
    let input_intermediate = values(INTERMEDIATE, 4343);
    let input_for = |columns: usize| -> &[f32] {
        if columns == HIDDEN {
            &input_hidden
        } else {
            &input_intermediate
        }
    };
    let mut outs: Vec<Vec<f32>> = layers
        .iter()
        .map(|q| vec![0.0; q.shape().unwrap().0])
        .collect();

    #[allow(unused_mut)]
    let mut variants: Vec<(&str, Kernel)> = vec![
        ("crate matmul (now)", crate_matmul),
        ("decode+dot, hoisted", decode_dot_hoisted),
        ("portable decode+dot", portable_decode_dot),
        ("fused, existing dot", fused_existing_dot),
        ("fused bitexact", fused_bitexact),
        ("fused bitexact staged", fused_bitexact_staged),
        ("fused factored", fused_factored),
        ("fused factored staged", fused_factored_staged),
    ];
    #[cfg(target_arch = "aarch64")]
    variants.push(("fused neon factored", fused_neon));
    #[cfg(target_arch = "aarch64")]
    variants.push(("fused neon bitexact", fused_neon_bitexact));

    // Reference outputs.
    let mut reference: Vec<Vec<f32>> = Vec::new();
    for q in &layers {
        let (rows, columns) = q.shape().unwrap();
        let mut out = vec![0.0; rows];
        q.matmul_into(input_for(columns), &mut out).unwrap();
        reference.push(out);
    }

    let repeats = 25;
    println!("batch-1 linears of one SmolLM-135M token (4-bit, block 32, f16 scales)");
    println!(
        "{:<24}{:>10}{:>10}{:>14}{:>10}",
        "variant", "ms/token", "ns/value", "max rel diff", "bitexact"
    );
    let total_values: usize = layers.iter().map(|q| q.len()).sum();
    for (name, kernel) in &variants {
        // Check.
        let mut max_relative = 0.0f32;
        let mut bitexact = true;
        for ((q, out), expected) in layers.iter().zip(outs.iter_mut()).zip(&reference) {
            let (_, columns) = q.shape().unwrap();
            kernel(q, input_for(columns), out);
            for (got, want) in out.iter().zip(expected) {
                if got.to_bits() != want.to_bits() {
                    bitexact = false;
                }
                let relative = (got - want).abs() / want.abs().max(1e-3);
                max_relative = max_relative.max(relative);
            }
        }
        let mut run = || {
            let start = Instant::now();
            for (q, out) in layers.iter().zip(outs.iter_mut()) {
                let (_, columns) = q.shape().unwrap();
                kernel(q, input_for(columns), out);
                black_box(&out);
            }
            start.elapsed().as_secs_f64()
        };
        for _ in 0..3 {
            run();
        }
        let mut samples: Vec<f64> = (0..repeats).map(|_| run()).collect();
        samples.sort_by(f64::total_cmp);
        let seconds = samples[repeats / 2];
        println!(
            "{name:<24}{:>10.2}{:>10.3}{:>14.2e}{:>10}",
            seconds * 1e3,
            seconds * 1e9 / total_values as f64,
            max_relative,
            bitexact
        );
    }
    time_whole_decode();
}
