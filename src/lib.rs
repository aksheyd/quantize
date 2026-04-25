//! # quantize
//!
//! A tiny, readable quantization library — symmetric, any bit width.
//!
//! ## Example
//!
//! ```
//! use quantize::{choose_scale_bits, dequantize_bits, quantize_bits};
//!
//! let weights = [0.42_f32, -0.10, 0.70, -0.50];
//!
//! // 8-bit symmetric quantization round-trip
//! let scale = choose_scale_bits::<8>(&weights);
//! let q: i32 = quantize_bits::<8>(weights[0], scale); // q in -128..=127
//! let back = dequantize_bits(q, scale);               // back ≈ weights[0]
//!
//! assert!((back - weights[0]).abs() < 0.01);
//! ```
//!
//! Bit width is a const generic, so `quantize_bits::<4>(...)` (4-bit),
//! `quantize_bits::<16>(...)` (16-bit), etc. all compile to specialized
//! code with zero runtime overhead.

// To learn how the library got here, please see `chapters/`.

pub mod bits;

pub use bits::{choose_scale_bits, dequantize_bits, quantize_bits};
