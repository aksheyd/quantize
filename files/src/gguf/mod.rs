//! [gguf](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), the
//! format that llama.cpp runs models from. A file holds, in order, with every
//! number little-endian:
//!
//! | bytes | field |
//! | --- | --- |
//! | 4 | `GGUF` |
//! | 4 | the version, 3, as a `u32` |
//! | 8 | how many tensors the file holds, as a `u64` |
//! | 8 | how many metadata entries follow, as a `u64` |
//! | ... | each metadata entry: its key, a string; its value's type, as a `u32`; then the value |
//! | ... | each tensor's info: its name, a string; how many dimensions it has, as a `u32`; each dimension, innermost first, as a `u64`; its ggml type, as a `u32`; and its offset, as a `u64` |
//! | ... | zeros, up to a multiple of the alignment |
//! | the rest | the data: each tensor's bytes, at its offset into the data, then zeros up to a multiple of the alignment |
//!
//! Everything before the data is the header. A string is its length in
//! bytes, as a `u64`, then its UTF-8 bytes. A value is one of the types that
//! [`Value`] lists, and an array is its elements' type, as a `u32`, how many
//! there are, as a `u64`, then each element.
//!
//! A file with one metadata entry, `general.architecture = "llama"`, and a
//! `2 × 3` matrix of `F32` values named `w` lays out like this:
//!
//! ```text
//! byte 0     GGUF, 3, 1 tensor, 1 metadata entry
//! byte 24    20, "general.architecture", 8 for a string, 5, "llama"
//! byte 69    1, "w", 2 dimensions, 3 columns, 2 rows, ggml type 0 for F32, offset 0
//! byte 110   zeros up to byte 128
//! byte 128   the 6 values, row after row, then zeros up to byte 160
//! ```
//!
//! Dimensions run innermost first, the reverse of how numpy lists a shape:
//! the matrix is `[3, 2]`. [`read()`] turns them around, so a
//! [`Tensor::Float`](crate::Tensor::Float)'s shape is outermost first,
//! `[2, 3]`, as in [`safetensors`](crate::safetensors).
//!
//! A tensor's size in bytes isn't stored: it follows from its dimensions and
//! its ggml type. Tensors' data follow one another in the order of their
//! infos, so each tensor starts where the one before it ends, padded to the
//! alignment: 32 bytes, unless the metadata entry `general.alignment`, a
//! `u32` power of two, says otherwise.
//!
//! ggml types `F32` (0), `F16` (1), and `BF16` (30) read as
//! [`Tensor::Float`](crate::Tensor::Float), widened to `f32`, and float
//! tensors write as `F32`. Any other ggml type, and a
//! [`Tensor::Quantized`](crate::Tensor::Quantized), is an error that names
//! the tensor. Tensors are `Tensor<f16>`, since ggml's quantized blocks, like
//! `Q4_0` and `Q8_0`, hold f16 scales.

mod ggml_types;
mod read;
mod reader;
mod value;
mod write;

#[cfg(test)]
mod tests;

use std::collections::BTreeMap;

pub use read::read;
pub use value::Value;
pub use write::write;

use crate::error::{Error, invalid};

/// A file's metadata: each entry's value, by its key, like
/// `general.architecture` or `llama.block_count`.
pub type Metadata = BTreeMap<String, Value>;

const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: usize = 32;

/// The alignment of the data section and of each tensor's data:
/// `general.alignment`, if the metadata has it, or 32 bytes.
fn alignment(metadata: &Metadata) -> Result<usize, Error> {
    match metadata.get("general.alignment") {
        None => Ok(DEFAULT_ALIGNMENT),
        Some(&Value::U32(alignment)) if alignment.is_power_of_two() => Ok(alignment as usize),
        Some(other) => Err(invalid(format!(
            "general.alignment must be a U32 that is a power of two, not {other:?}"
        ))),
    }
}

/// How many values a tensor of `shape` holds, or `None` if a `usize` can't
/// count them. A single value's empty shape holds 1.
fn value_count(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1_usize, |count, &length| count.checked_mul(length))
}
