//! # quantize
//!
//! A tiny, readable quantization library — block-wise symmetric or asymmetric,
//! any bit width.
//!
//! ## Example
//!
//! ```
//! use quantize::quantize;
//!
//! let weights = [0.42_f32, -0.10, 0.70, -0.50];
//!
//! // 8-bit, block-size-32, f32 scales
//! let q = quantize::<f32, 8, 32>(&weights).unwrap();
//! let back = q.dequantize();
//!
//! assert!((back[0] - weights[0]).abs() < 0.01);
//! ```
//!
//! `BITS` and `BLOCK` are const generics, so `quantize::<f32, 4, 32>(...)`,
//! `quantize::<f32, 8, 64>(...)`, etc. all compile to specialized code.
//!
//! See `symmetric`, `asymmetric`, and `adaptive` for the other schemes.
//! To learn how the library got here, please see `chapters/`.
//!
//! ## NaN and infinity
//!
//! Quantization expects finite input. A NaN is skipped when its block
//! measures its range and is stored as code 0, so the other values in that
//! block are unaffected. An infinity is kept, which stretches its block's
//! range to infinity, so every finite value in that block decodes to NaN.

mod kernels;
mod methods;
mod shared;

pub use methods::{adaptive, asymmetric, learned, symmetric};

pub use shared::error::{Error, Result};
pub use shared::packed::Packed;
pub use shared::scale::Scale;
pub use shared::scheme::Scheme;
pub use shared::tensor::Quantized;
pub use shared::{error, packed, params, scale, scheme, tensor};
pub use symmetric::{quantize, quantize_tensor};

pub(crate) use shared::decode;
