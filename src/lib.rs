//! # quantize
//!
//! A tiny, readable quantization library — block-wise symmetric or asymmetric,
//! from 2 to 16 bits.
//!
//! ## Example
//!
//! ```
//! use quantize::{f16, quantize};
//!
//! let weights = [0.42_f32, -0.10, 0.70, -0.50];
//!
//! // f16 scales, 8-bit codes, blocks of 32 values
//! let q = quantize::<f16, 8, 32>(&weights).unwrap();
//!
//! let back = q.dequantize(); // [0.421, -0.098, 0.700, -0.498]
//! let dot = q.dot(&weights).unwrap(); // 0.926
//!
//! assert!((back[0] - weights[0]).abs() < 0.01);
//! ```
//!
//! `S` is the scale type: `f32`, [`f16`](struct@f16), or [`bf16`], the last two
//! re-exported from the `half` crate, so import them with
//! `use quantize::{f16, bf16}`. Each block shares one scale, so one `f16` per
//! 32 values adds 16 / 32 = 0.5 bits to each value: 4-bit codes cost 4.5 bits
//! per value.
//!
//! `BITS` and `BLOCK` are const generics, so a width outside 2 to 16 or a
//! block of 0 stops the build instead of returning an error. To choose them at
//! run time, call the scheme's `quantize_with`, like
//! [`symmetric::quantize_with`], which takes them as ordinary arguments.
//!
//! [`quantize`] is symmetric: each block gets one scale. Everything below
//! uses the same [`Quantized`] type:
//!
//! - [`asymmetric::quantize`] adds a zero-point per block, for values that
//!   aren't centered on zero
//! - [`adaptive::quantize`] picks each block's bit width from an error
//!   tolerance in the values' own units, like a tenth of their standard
//!   deviation
//! - [`learned::refine`] refits each block's scale, and its zero-point if it
//!   has one, to lower the mean squared error
//! - [`learned::alternate`] refits too, then rounds each value to the nearest
//!   code on its block's new line, and repeats until no code moves. Both can
//!   raise the worst error past an adaptive tensor's tolerance
//! - [`Scheme`] picks one at run time, like
//!   `Scheme::Q4_32.quantize::<f16>(&weights)`
//!
//! For a weight matrix, [`set_shape`](Quantized::set_shape) records its shape,
//! so [`matmul`](Quantized::matmul) can multiply a batch of inputs by it, like
//! a linear layer, and [`dequantize_row_into`](Quantized::dequantize_row_into)
//! can decode one row, like an embedding lookup.
//! [`to_bytes`](Quantized::to_bytes) saves a tensor, shape included, and
//! [`from_bytes`](Quantized::from_bytes) loads it back.
//!
//! To learn how the library got here, see the
//! [chapters](https://github.com/aksheyd/quantize/tree/main/chapters).
//!
//! ## NaN and infinity
//!
//! Quantization expects finite input. A NaN is skipped when its block
//! measures its range and is stored as code 0, so the other values in that
//! block are unaffected. An infinity is kept, which stretches its block's
//! range to infinity, so every finite value in that block decodes to NaN.

#![warn(missing_docs)]

mod kernels;
mod methods;
mod shared;

pub use methods::{adaptive, asymmetric, learned, symmetric};

#[doc(no_inline)]
pub use half::{bf16, f16};

pub use shared::error::{Error, Result};
pub use shared::packed::Packed;
pub use shared::params;
pub use shared::scale::Scale;
pub use shared::scheme::Scheme;
pub use shared::tensor::Quantized;
pub use symmetric::{quantize, quantize_tensor};

pub(crate) use shared::{decode, error, packed, scale, tensor};
