#![allow(clippy::all, unused_mut, dead_code)]
//! Scratch: batch-1 matmul kernel variants, with the block size known only at
//! run time, as in the crate. Not committed.

use half::f16;
use quantize::{Quantized, quantize};
use std::hint::black_box;
use std::time::Instant;

const HIDDEN: usize = 576;
const KV: usize = 192;
const INTERMEDIATE: usize = 1536;
const VOCAB: usize = 49152;
const LAYERS: usize = 30;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut seed = seed;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.05
        })
        .collect()
}

/// The crate's 16-lane dot.
#[inline(never)]
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

// ---------- the crate's current decode, verbatim ----------

#[inline(never)]
fn dequant_now(scales: &[f16], bytes: &[u8], block: usize, out: &mut [f32]) {
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

// ---------- the decode as changed in the working tree ----------

#[inline(never)]
fn dequant_new(scales: &[f16], bytes: &[u8], block: usize, out: &mut [f32]) {
    assert!(bytes.len() >= (out.len() * 4).div_ceil(8));
    let mut i = 0usize;
    for (bi, chunk) in out.chunks_mut(block).enumerate() {
        let s = scales[bi].to_f32();
        let mut j = 0;
        if !i.is_multiple_of(2) {
            chunk[0] = high_code(bytes[i / 2]) as f32 * s;
            i += 1;
            j += 1;
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            while j + 32 <= chunk.len() {
                neon::dequant_32(bytes.as_ptr().add(i / 2), s, chunk.as_mut_ptr().add(j));
                i += 32;
                j += 32;
            }
        }
        let (pairs, _) = chunk[j..].as_chunks_mut::<2>();
        for (pair, &byte) in pairs.iter_mut().zip(&bytes[i / 2..]) {
            *pair = [low_code(byte) as f32 * s, high_code(byte) as f32 * s];
        }
        i += 2 * pairs.len();
        j += 2 * pairs.len();
        if j < chunk.len() {
            chunk[j] = low_code(bytes[i / 2]) as f32 * s;
            i += 1;
        }
    }
}

// Portable only, no NEON.
#[inline(never)]
fn dequant_portable(scales: &[f16], bytes: &[u8], block: usize, out: &mut [f32]) {
    assert!(bytes.len() >= (out.len() * 4).div_ceil(8));
    let mut i = 0usize;
    for (bi, chunk) in out.chunks_mut(block).enumerate() {
        let s = scales[bi].to_f32();
        let mut j = 0;
        if !i.is_multiple_of(2) {
            chunk[0] = high_code(bytes[i / 2]) as f32 * s;
            i += 1;
            j += 1;
        }
        let (pairs, _) = chunk[j..].as_chunks_mut::<2>();
        for (pair, &byte) in pairs.iter_mut().zip(&bytes[i / 2..]) {
            *pair = [low_code(byte) as f32 * s, high_code(byte) as f32 * s];
        }
        i += 2 * pairs.len();
        j += 2 * pairs.len();
        if j < chunk.len() {
            chunk[j] = low_code(bytes[i / 2]) as f32 * s;
            i += 1;
        }
    }
}

// ---------- fused dot kernels, factored by block, any even block ----------

/// Σ code·x over pairs, 8 lanes, then the lanes added up.
#[inline(always)]
fn pairs_dot(bytes: &[u8], x: &[f32]) -> f32 {
    let (byte_groups, byte_rest) = bytes.as_chunks::<4>();
    let (x_groups, x_rest) = x.as_chunks::<8>();
    let mut lanes = [0.0_f32; 8];
    for (four, eight) in byte_groups.iter().zip(x_groups) {
        for i in 0..4 {
            lanes[2 * i] += low_code(four[i]) as f32 * eight[2 * i];
            lanes[2 * i + 1] += high_code(four[i]) as f32 * eight[2 * i + 1];
        }
    }
    let mut total: f32 = lanes.iter().sum();
    for (&byte, pair) in byte_rest.iter().zip(x_rest.chunks(2)) {
        total += low_code(byte) as f32 * pair[0];
        if let Some(&high_x) = pair.get(1) {
            total += high_code(byte) as f32 * high_x;
        }
    }
    total
}

/// Whole row, any even block, block sums × scale. NEON 32 at a time on aarch64 if `use_neon`.
#[inline(always)]
fn dot_row_factored<const USE_NEON: bool>(
    scales: &[f16],
    bytes: &[u8],
    block: usize,
    x: &[f32],
) -> f32 {
    let mut total = 0.0_f32;
    for ((block_x, block_bytes), scale) in x
        .chunks(block)
        .zip(bytes.chunks(block.div_ceil(2)))
        .zip(scales)
    {
        let mut inner = 0.0_f32;
        let mut j = 0;
        #[cfg(target_arch = "aarch64")]
        if USE_NEON {
            unsafe {
                while j + 32 <= block_x.len() {
                    inner += core::arch::aarch64::vaddvq_f32(neon::dot_32(
                        block_bytes.as_ptr().add(j / 2),
                        block_x.as_ptr().add(j),
                    ));
                    j += 32;
                }
            }
        }
        inner += pairs_dot(&block_bytes[j / 2..], &block_x[j..]);
        total += scale.to_f32() * inner;
    }
    total
}

/// NEON, llama.cpp style: one vector accumulator for the row, block % 32 == 0.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn dot_row_neon_vector(scales: &[f16], bytes: &[u8], block: usize, x: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        for ((block_x, block_bytes), scale) in x
            .chunks_exact(block)
            .zip(bytes.chunks_exact(block / 2))
            .zip(scales)
        {
            let mut inner = vdupq_n_f32(0.0);
            let mut j = 0;
            while j + 32 <= block_x.len() {
                inner = vaddq_f32(
                    inner,
                    neon::dot_32(block_bytes.as_ptr().add(j / 2), block_x.as_ptr().add(j)),
                );
                j += 32;
            }
            acc = vfmaq_n_f32(acc, inner, scale.to_f32());
        }
        vaddvq_f32(acc)
    }
}

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

    /// Like `dequant_32`, but multiplies the codes by `x` and adds up the products in 4 lanes.
    #[inline(always)]
    pub unsafe fn dot_32(src: *const u8, x: *const f32) -> float32x4_t {
        unsafe {
            let raw = vld1q_u8(src);
            let lo = vshrq_n_s8(
                vshlq_n_s8(vreinterpretq_s8_u8(vandq_u8(raw, vdupq_n_u8(0x0F))), 4),
                4,
            );
            let hi = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), 4), 4);
            let first = dot_i8x16(vzip1q_s8(lo, hi), x);
            let second = dot_i8x16(vzip2q_s8(lo, hi), x.add(16));
            vaddq_f32(first, second)
        }
    }

    /// Like `store_i8x16`, but multiplies by `x` and adds up instead of storing.
    #[inline(always)]
    unsafe fn dot_i8x16(q: int8x16_t, x: *const f32) -> float32x4_t {
        unsafe {
            let lo = vmovl_s8(vget_low_s8(q));
            let hi = vmovl_s8(vget_high_s8(q));
            let a = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(x));
            let b = vmulq_f32(
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                vld1q_f32(x.add(4)),
            );
            let a = vfmaq_f32(
                a,
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                vld1q_f32(x.add(8)),
            );
            let b = vfmaq_f32(
                b,
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                vld1q_f32(x.add(12)),
            );
            vaddq_f32(a, b)
        }
    }
}

// ---------- variants over the whole matrix ----------

type Kernel = fn(&Quantized<f16>, &[f32], &mut [f32]);

fn crate_matmul(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    q.matmul_into(input, out).unwrap();
}

fn rows_of<'a>(
    q: &'a Quantized<f16>,
) -> (usize, usize, impl Iterator<Item = (&'a [u8], &'a [f16])>) {
    let (_, columns) = q.shape().unwrap();
    let block = q.block();
    let rows = q
        .codes()
        .chunks_exact(columns / 2)
        .zip(q.scales().chunks_exact(columns / block));
    (columns, block, rows)
}

/// The crate's `matmul_into` and `decode_row`, per-row dispatch included, with a chosen decode.
fn crate_like(
    q: &Quantized<f16>,
    inputs: &[f32],
    out: &mut [f32],
    decode: fn(&[f16], &[u8], usize, &mut [f32]),
) {
    let (_, columns) = q.shape().unwrap();
    let rows = q.len() / columns;
    let mut row_weights = vec![0.0; columns];
    for row in 0..rows {
        decode_row_like(q, row, &mut row_weights, decode);
        for (vector, input) in inputs.chunks_exact(columns).enumerate() {
            out[vector * rows + row] = dot16(&row_weights, input);
        }
    }
}

fn packed_rows_ok_like(q: &Quantized<f16>, columns: usize) -> bool {
    let block = q.block();
    if block == 0 || !columns.is_multiple_of(block) {
        return false;
    }
    match q {
        Quantized::Symmetric { codes, .. } | Quantized::Asymmetric { codes, .. } => {
            (columns * codes.bits() as usize).is_multiple_of(8)
        }
        Quantized::Adaptive { .. } => false,
    }
}

fn decode_row_like(
    q: &Quantized<f16>,
    row: usize,
    out: &mut [f32],
    decode: fn(&[f16], &[u8], usize, &mut [f32]),
) {
    let columns = out.len();
    if !packed_rows_ok_like(q, columns) {
        unreachable!();
    }
    let block = q.block();
    let scales_per_row = columns / block;
    let row_scales = &q.scales()[row * scales_per_row..(row + 1) * scales_per_row];
    match q {
        Quantized::Symmetric { codes, .. } if codes.bits() == 4 => {
            let bytes_per_row = columns / 2;
            let start = row * bytes_per_row;
            decode(
                row_scales,
                &codes.as_bytes()[start..start + bytes_per_row],
                block,
                out,
            )
        }
        _ => unreachable!(),
    }
}

fn crate_like_now(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    crate_like(q, input, out, dequant_now);
}

fn crate_like_new(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    crate_like(q, input, out, dequant_new);
}

fn decode_now_hoisted(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (columns, block, rows) = rows_of(q);
    let mut row_weights = vec![0.0; columns];
    for ((bytes, scales), slot) in rows.zip(out) {
        dequant_now(scales, bytes, block, &mut row_weights);
        *slot = dot16(&row_weights, input);
    }
}

fn decode_new_hoisted(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (columns, block, rows) = rows_of(q);
    let mut row_weights = vec![0.0; columns];
    for ((bytes, scales), slot) in rows.zip(out) {
        dequant_new(scales, bytes, block, &mut row_weights);
        *slot = dot16(&row_weights, input);
    }
}

fn decode_portable_hoisted(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (columns, block, rows) = rows_of(q);
    let mut row_weights = vec![0.0; columns];
    for ((bytes, scales), slot) in rows.zip(out) {
        dequant_portable(scales, bytes, block, &mut row_weights);
        *slot = dot16(&row_weights, input);
    }
}

fn fused_factored_neon(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, block, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = dot_row_factored::<true>(scales, bytes, block, input);
    }
}

fn fused_factored_portable(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, block, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = dot_row_factored::<false>(scales, bytes, block, input);
    }
}

#[cfg(target_arch = "aarch64")]
fn fused_neon_vector(q: &Quantized<f16>, input: &[f32], out: &mut [f32]) {
    let (_, block, rows) = rows_of(q);
    for ((bytes, scales), slot) in rows.zip(out) {
        *slot = dot_row_neon_vector(scales, bytes, block, input);
    }
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
    let layer_count: usize = std::env::var("LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(LAYERS);
    let vocab: usize = std::env::var("VOCAB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(VOCAB);
    let mut layers: Vec<Quantized<f16>> = Vec::new();
    for layer in 0..layer_count {
        for (index, &(rows, columns)) in shapes.iter().enumerate() {
            let weights = values(rows * columns, (layer * 7 + index) as u32 + 1);
            let mut q = quantize::<f16, 4, 32>(&weights).unwrap();
            q.set_shape(rows, columns).unwrap();
            layers.push(q);
        }
    }
    let mut lm = quantize::<f16, 4, 32>(&values(vocab * HIDDEN, 999)).unwrap();
    lm.set_shape(vocab, HIDDEN).unwrap();
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
        ("crate-like, decode now", crate_like_now),
        ("crate-like, decode new", crate_like_new),
        ("decode now, hoisted", decode_now_hoisted),
        ("decode new, hoisted", decode_new_hoisted),
        ("decode portable, hoist", decode_portable_hoisted),
        ("fused fact. neon+pairs", fused_factored_neon),
        ("fused fact. pairs only", fused_factored_portable),
    ];
    #[cfg(target_arch = "aarch64")]
    variants.push(("fused neon vector acc", fused_neon_vector));

    let mut reference: Vec<Vec<f32>> = Vec::new();
    for q in &layers {
        let (rows, columns) = q.shape().unwrap();
        let mut out = vec![0.0; rows];
        q.matmul_into(input_for(columns), &mut out).unwrap();
        reference.push(out);
    }

    let repeats: usize = std::env::var("REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);
    println!(
        "batch-1 linears of one SmolLM-135M token (4-bit, block 32 known at run time, f16 scales)"
    );
    println!(
        "{:<24}{:>10}{:>10}{:>14}{:>10}",
        "variant", "ms/token", "ns/value", "max rel diff", "bitexact"
    );
    let total_values: usize = layers.iter().map(|q| q.len()).sum();
    for (name, kernel) in &variants {
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

    // Whole-tensor decode.
    let side = 1024;
    let q = quantize::<f16, 4, 32>(&values(side * side, 7)).unwrap();
    let block = q.block();
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
    let now = time(&mut |out| dequant_now(q.scales(), q.codes(), block, out));
    let new = time(&mut |out| dequant_new(q.scales(), q.codes(), block, out));
    let portable = time(&mut |out| dequant_portable(q.scales(), q.codes(), block, out));
    println!(
        "\nwhole 1024x1024 4-bit decode, ns/value: now {now:.3}, new {new:.3}, portable only {portable:.3}"
    );
}
