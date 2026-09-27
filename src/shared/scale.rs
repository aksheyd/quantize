//! Scale trait — how per-block scale factors are stored.
//!
//! Implement this for any type that can round-trip through `f32`.
//! The crate ships impls for `f32` (lossless), `f16`, and `bf16`.

use half::{bf16, f16};

/// A type that can serve as a per-block scale (or zero-point) factor.
pub trait Scale: Copy {
    /// Short name that [`Quantized::to_bytes`](crate::Quantized::to_bytes)
    /// records, so a tensor loads back with the same scale type.
    const NAME: &'static str;
    /// Convert from the working-precision `f32` value.
    fn from_f32(v: f32) -> Self;
    /// Convert back to `f32` for arithmetic.
    fn to_f32(self) -> f32;
    /// Append this value's `size_of::<Self>()` bytes, little-endian.
    fn write_le_bytes(self, out: &mut Vec<u8>);
    /// Read a value back from the bytes that
    /// [`write_le_bytes`](Self::write_le_bytes) wrote.
    fn read_le_bytes(bytes: &[u8]) -> Self;
}

impl Scale for f32 {
    const NAME: &'static str = "f32";
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn read_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl Scale for f16 {
    const NAME: &'static str = "f16";
    #[inline]
    fn from_f32(v: f32) -> Self {
        f16::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn read_le_bytes(bytes: &[u8]) -> Self {
        f16::from_le_bytes([bytes[0], bytes[1]])
    }
}

impl Scale for bf16 {
    const NAME: &'static str = "bf16";
    #[inline]
    fn from_f32(v: f32) -> Self {
        bf16::from_f32(v)
    }
    #[inline]
    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }
    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn read_le_bytes(bytes: &[u8]) -> Self {
        bf16::from_le_bytes([bytes[0], bytes[1]])
    }
}
