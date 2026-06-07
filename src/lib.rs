//! # quantize
//!
//! A tiny, readable quantization library — block-wise symmetric or asymmetric, any bit width.
//!
//! ## Example
//!
//! ```
//! use quantize::{quantize, dequantize};
//!
//! let weights = [0.42_f32, -0.10, 0.70, -0.50];
//!
//! // 8-bit, block-size-32, f32 scales
//! let (scales, codes) = quantize::<f32, 8, 32>(&weights);
//! let back = dequantize::<_, 32>(&scales, &codes);
//!
//! assert!((back[0] - weights[0]).abs() < 0.01);
//! ```
//!
//! `BITS` and `BLOCK` are const generics, so `quantize::<4, 32>(...)`,
//! `quantize::<8, 64>(...)`, etc. all compile to specialized code with
//! zero runtime overhead.
//!
//! See the `symmetric` and `asymmetric` modules for tensor-granularity variants
//! and precision-aware per-block bit widths.
//!
//! To learn how the library got here, please see `chapters/`.

pub mod asymmetric;
pub mod scale;
pub mod symmetric;

pub use asymmetric::{dequantize as dequantize_asym, quantize as quantize_asym};
pub use scale::Scale;
pub use symmetric::{dequantize, quantize};
