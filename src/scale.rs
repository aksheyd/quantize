//! Scale trait — defines how per-block scale factors are stored.
//!
//! Implement this for any type that can round-trip through f32.
//! The library ships impls for `f32` (lossless) and `f16` (GGML-compatible).

use half::{bf16, f16};

/// A type that can serve as a per-block scale factor.
pub trait Scale: Copy {
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
}

impl Scale for f32 {
    fn from_f32(v: f32) -> Self {
        v
    }
    fn to_f32(self) -> f32 {
        self
    }
}

impl Scale for f16 {
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
}

impl Scale for bf16 {
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }
}
