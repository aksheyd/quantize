//! # quantize-files
//!
//! Read and write the files that models ship in, with
//! [quantize](https://docs.rs/quantize)'s tensors inside, so a model can be
//! loaded, quantized, and saved again.
//!
//! A model file maps each tensor's name to its values. Each format has a
//! module with a `read` function, which gives back a [`BTreeMap`] from names
//! to [`Tensor`]s, and a `write` function, which takes one:
//!
//! - [`safetensors`], Hugging Face's format. `F32`, `F16`, and `BF16` tensors
//!   read as `f32`, floats write as `F32`, and quantized tensors of any
//!   scheme save as the bytes that [`Quantized::to_bytes`] writes.
//! - [`gguf`], llama.cpp's format. `F32`, `F16`, and `BF16` tensors read as
//!   `f32`, and floats write as `F32`. Quantized tensors read and write as
//!   ggml's `Q4_0` and `Q8_0` when they are
//!   [`Scheme::Q4_32`](quantize::Scheme::Q4_32) and
//!   [`Scheme::Q8_32`](quantize::Scheme::Q8_32) matrices with f16 scales,
//!   which hold the same values in other bytes. Its metadata, like the
//!   model's architecture and its tokenizer's vocabulary, reads and writes
//!   alongside the tensors, as a map from keys to [`gguf::Value`]s.
//!
//! Start with [`Tensor`], then read the module of your format: its docs lay
//! out the file byte by byte, and its `read` and `write` follow that layout
//! in order.
//!
//! ```
//! use std::collections::BTreeMap;
//!
//! use quantize::{Scheme, f16};
//! use quantize_files::{Tensor, gguf, safetensors};
//!
//! // A 2 × 32 weight matrix, quantized to 4 bits, and a norm kept as floats.
//! let values: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
//! let mut weight = Scheme::Q4_32.quantize::<f16>(&values)?;
//! weight.set_shape(2, 32)?;
//! let norm = Tensor::Float { shape: vec![32], values: vec![1.0; 32] };
//!
//! let tensors = BTreeMap::from([
//!     ("norm".to_string(), norm),
//!     ("weight".to_string(), Tensor::Quantized(weight)),
//! ]);
//! let path = std::env::temp_dir().join("quantize-files-example.safetensors");
//! safetensors::write(&path, &tensors)?;
//!
//! // The weight was saved with f16 scales, so it loads back with f16 scales.
//! assert_eq!(safetensors::read::<f16>(&path)?, tensors);
//!
//! // gguf holds the weight as ggml's Q4_0 blocks, which decode to the same
//! // values, so it loads back the same too.
//! let path = std::env::temp_dir().join("quantize-files-example.gguf");
//! gguf::write(&path, &gguf::Metadata::new(), &tensors)?;
//! let (_, loaded) = gguf::read(&path)?;
//! assert_eq!(loaded, tensors);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! [`BTreeMap`]: std::collections::BTreeMap

#![warn(missing_docs)]

mod error;
pub mod gguf;
pub mod safetensors;

use quantize::{Quantized, Scale};

pub use error::Error;

/// One tensor in a model file: plain floats, or a tensor that quantize
/// quantized.
///
/// `S` is the scale type of the quantized tensors: `f32`, `f16`, or `bf16`.
/// A safetensors file of floats alone reads with any of them. A gguf file's
/// tensors are `Tensor<f16>`, since ggml's quantized blocks hold f16 scales.
#[derive(Clone, Debug, PartialEq)]
pub enum Tensor<S: Scale> {
    /// Values of any float type, widened to `f32`.
    Float {
        /// The length of each dimension, outermost first, as numpy and
        /// PyTorch list them: `[rows, columns]` for a matrix, `[len]` for a
        /// vector, and `[]` for a single value.
        shape: Vec<usize>,
        /// Every value, row after row, so the last dimension changes fastest.
        values: Vec<f32>,
    },
    /// A tensor of any scheme, with the shape that
    /// [`set_shape`](Quantized::set_shape) recorded, if any.
    Quantized(Quantized<S>),
}
