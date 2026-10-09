//! ggml's `Q4_0` and `Q8_0` blocks, which hold quantize's
//! [`Scheme::Q4_32`] and [`Scheme::Q8_32`] tensors with f16 scales.
//!
//! Both formats split each row of a matrix into blocks of 32 values, give
//! each block one f16 scale, and decode value `i` of a block as
//! `code[i] × scale`. Only where the bytes go differs. quantize keeps every
//! scale in one list and every code in another. ggml writes each block's
//! scale just before its codes:
//!
//! | type | a block's bytes |
//! | --- | --- |
//! | `Q4_0` | 18: the scale, then 16 bytes, where byte `j` holds `code[j] + 8` in its low 4 bits and `code[j + 16] + 8` in its high 4 |
//! | `Q8_0` | 34: the scale, then 32 bytes, where byte `i` holds `code[i]` as a signed byte |
//!
//! A 4-bit code runs from -8 to 7. `Q4_0` stores it plus 8, from 0 to 15,
//! and pairs value `j` with value `j + 16`. quantize stores the code itself,
//! as a signed 4-bit number, and pairs neighbors: codes `2j` and `2j + 1`
//! share byte `j`, low 4 bits first. So a `Q4_0` block's low halves, then
//! its high halves, give its 32 codes in order, which quantize packs its own
//! way.
//!
//! An 8-bit code is one signed byte in both, so `Q8_0`'s codes are
//! quantize's bytes, block by block. quantize puts each block's value
//! farthest from zero on -128, a code that ggml's own quantizer never
//! writes, but that its decoder reads as it reads any other.
//!
//! [`Scheme::Q4_32`]: quantize::Scheme::Q4_32
//! [`Scheme::Q8_32`]: quantize::Scheme::Q8_32

use half::f16;
use quantize::{Packed, Quantized};

/// How many values share each scale, in `Q4_0` and `Q8_0` alike.
pub(super) const BLOCK: usize = 32;
/// The bytes of one `Q4_0` block: the scale, then 32 codes of 4 bits.
pub(super) const Q4_0_BLOCK_BYTES: usize = 2 + BLOCK / 2;
/// The bytes of one `Q8_0` block: the scale, then 32 codes of 8 bits.
pub(super) const Q8_0_BLOCK_BYTES: usize = 2 + BLOCK;

/// The `Q4_0` blocks of `tensor`, a symmetric matrix with 4-bit codes in
/// blocks of 32 that split its rows evenly.
pub(super) fn q4_0_blocks(tensor: &Quantized<f16>) -> Vec<u8> {
    let codes = tensor.unpacked_codes();
    let mut blocks = Vec::with_capacity(tensor.scales().len() * Q4_0_BLOCK_BYTES);
    for (scale, block_codes) in tensor.scales().iter().zip(codes.chunks(BLOCK)) {
        blocks.extend(scale.to_le_bytes());
        let (first_half, second_half) = block_codes.split_at(BLOCK / 2);
        for (low, high) in first_half.iter().zip(second_half) {
            blocks.push((low + 8) as u8 | ((high + 8) as u8) << 4);
        }
    }
    blocks
}

/// The matrix, with rows of `columns` values, that `Q4_0` `blocks` hold.
/// `columns` is a multiple of 32, and `blocks` holds whole blocks.
pub(super) fn from_q4_0(blocks: &[u8], columns: usize) -> Quantized<f16> {
    let mut scales = Vec::new();
    let mut codes = Vec::new();
    let (blocks, _) = blocks.as_chunks::<Q4_0_BLOCK_BYTES>();
    for block in blocks {
        let (scale, pairs) = block.split_at(2);
        scales.push(f16::from_le_bytes([scale[0], scale[1]]));
        codes.extend(pairs.iter().map(|byte| i32::from(byte & 0x0F) - 8));
        codes.extend(pairs.iter().map(|byte| i32::from(byte >> 4) - 8));
    }
    matrix(scales, Packed::from_i32s(&codes, 4), columns)
}

/// The `Q8_0` blocks of `tensor`, a symmetric matrix with 8-bit codes in
/// blocks of 32 that split its rows evenly.
pub(super) fn q8_0_blocks(tensor: &Quantized<f16>) -> Vec<u8> {
    let mut blocks = Vec::with_capacity(tensor.scales().len() * Q8_0_BLOCK_BYTES);
    for (scale, block_codes) in tensor.scales().iter().zip(tensor.codes().chunks(BLOCK)) {
        blocks.extend(scale.to_le_bytes());
        blocks.extend(block_codes);
    }
    blocks
}

/// The matrix, with rows of `columns` values, that `Q8_0` `blocks` hold.
/// `columns` is a multiple of 32, and `blocks` holds whole blocks.
pub(super) fn from_q8_0(blocks: &[u8], columns: usize) -> Quantized<f16> {
    let mut scales = Vec::new();
    let mut codes = Vec::new();
    let (blocks, _) = blocks.as_chunks::<Q8_0_BLOCK_BYTES>();
    for block in blocks {
        let (scale, block_codes) = block.split_at(2);
        scales.push(f16::from_le_bytes([scale[0], scale[1]]));
        codes.extend(block_codes);
    }
    let len = codes.len();
    matrix(scales, Packed::from_raw(codes, 8, len), columns)
}

/// A symmetric matrix in blocks of 32, with rows of `columns` values.
fn matrix(scales: Vec<f16>, codes: Packed, columns: usize) -> Quantized<f16> {
    Quantized::Symmetric {
        scales,
        len: codes.len(),
        codes,
        block: BLOCK,
        columns: Some(columns),
    }
}
