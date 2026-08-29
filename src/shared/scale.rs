//! Scale trait — how per-block scale factors are stored.
//!
//! Implement this for any type that can round-trip through `f32`.
//! The crate ships impls for `f32` (lossless), `f16`, and `bf16`.

use half::{bf16, f16};

/// A type that can serve as a per-block scale (or zero-point) factor.
pub trait Scale: Copy {
    /// Convert from the working-precision `f32` value.
    fn from_f32(v: f32) -> Self;
    /// Convert back to `f32` for arithmetic.
    fn to_f32(self) -> f32;
}

impl Scale for f32 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
}

impl Scale for f16 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
}

impl Scale for bf16 {
    #[inline]
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }
}
