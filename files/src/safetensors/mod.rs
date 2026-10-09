//! [safetensors](https://github.com/huggingface/safetensors), the format that
//! Hugging Face ships models in. A file holds, in order:
//!
//! | bytes | field |
//! | --- | --- |
//! | 8 | `N`, the header's length, as a little-endian `u64` |
//! | `N` | the header: JSON text that starts with `{`, padded with spaces |
//! | the rest | the data: every tensor's bytes, back to back |
//!
//! The header gives each tensor's name, its dtype, its shape, and the bytes
//! of the data that hold it, `begin` up to but not including `end`:
//!
//! ```json
//! {"norm": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
//!  "embed": {"dtype": "BF16", "shape": [3, 4], "data_offsets": [16, 40]},
//!  "__metadata__": {"format": "pt"}}
//! ```
//!
//! A tensor's values fill its bytes row after row, each one little-endian,
//! so the `3 × 4` matrix of 2-byte `BF16` values takes 24. Every byte of the
//! data belongs to exactly one tensor, so nothing can hide between them.
//! `__metadata__` is optional and maps strings to strings: [`read()`] checks
//! it and leaves it out, and [`write()`] doesn't write one.
//!
//! The format has no dtype for quantize's tensors, so [`write()`] saves each
//! one as a one-dimensional `U8` tensor holding the bytes of
//! [`Quantized::to_bytes`](quantize::Quantized::to_bytes). Those start with
//! `QNTZ`, which is how [`read()`] tells them from other bytes. In Python,
//! `safetensors.numpy.load_file` reads one as a uint8 array, and quantize's
//! `Quantized.from_bytes` loads that array.

mod dtypes;
mod read;
mod write;

#[cfg(test)]
mod tests;

pub use read::read;
pub use write::write;

/// How many values a tensor of `shape` holds, or `None` if a `usize` can't
/// count them. A single value's empty shape holds 1.
fn value_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |count, &length| count.checked_mul(length))
}
