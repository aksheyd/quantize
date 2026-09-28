//! Scale trait — how per-block scale factors are stored.
//!
//! Implement this for any type that can round-trip through `f32`.
//! The crate ships impls for `f32` (lossless), `f16`, and `bf16`.

use half::{bf16, f16};

use crate::error::{Error, Result};

/// A type that can serve as a per-block scale (or zero-point) factor.
#[diagnostic::on_unimplemented(
    note = "scales can be `f32`, `quantize::f16`, or `quantize::bf16`; for `f16`, add `use quantize::f16`, since without it `f16` names Rust's unstable primitive"
)]
pub trait Scale: Copy {
    /// Short name that [`Quantized::to_bytes`](crate::Quantized::to_bytes)
    /// records, so a tensor loads back with the same scale type.
    const NAME: &'static str;
    /// Convert from the working-precision `f32` value.
    fn from_f32(v: f32) -> Self;
    /// Convert from `f32`, rounding away from zero instead of to the nearest
    /// value, so the result is never smaller in magnitude than `v`.
    ///
    /// ```
    /// use quantize::{f16, Scale};
    ///
    /// // f16 can't hold 0.1 exactly; its nearest value is just below.
    /// assert!(f16::from_f32(0.1).to_f32() < 0.1);
    /// assert!(f16::from_f32_away_from_zero(0.1).to_f32() > 0.1);
    /// assert!(f16::from_f32_away_from_zero(-0.1).to_f32() < -0.1);
    /// ```
    fn from_f32_away_from_zero(v: f32) -> Self;
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
    fn from_f32_away_from_zero(v: f32) -> Self {
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
    fn from_f32_away_from_zero(v: f32) -> Self {
        let nearest = f16::from_f32(v);
        if nearest.to_f32().abs() < v.abs() {
            // Bit patterns of values with the same sign count up with their
            // magnitude, so the next pattern is the next value from zero.
            f16::from_bits(nearest.to_bits() + 1)
        } else {
            nearest
        }
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
    fn from_f32_away_from_zero(v: f32) -> Self {
        let nearest = bf16::from_f32(v);
        if nearest.to_f32().abs() < v.abs() {
            bf16::from_bits(nearest.to_bits() + 1)
        } else {
            nearest
        }
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

/// Block `block_index`'s scale, as stored in `S`, rounded away from zero.
///
/// Rounded to the nearest value instead, the scale could come out smaller in
/// magnitude than the block needs. Its value farthest from zero would then
/// need a code past the end, and be clamped: at 16 bits, up to 16 ticks off
/// with f16, which keeps 11 significant bits, and 128 with bf16, which keeps 8.
///
/// # Errors
///
/// [`Error::ScaleOutOfRange`] if the scale is too large for `S`, so it rounds
/// to infinity, which would decode every value in the block wrongly.
pub(crate) fn store_scale<S: Scale>(scale: f32, block_index: usize) -> Result<S> {
    let stored = S::from_f32_away_from_zero(scale);
    if scale.is_finite() && stored.to_f32().is_infinite() {
        return Err(Error::ScaleOutOfRange {
            block_index,
            scale_type: S::NAME,
        });
    }
    Ok(stored)
}

/// Block `block_index`'s zero-point, as stored in `S`.
///
/// # Errors
///
/// [`Error::ScaleOutOfRange`] if the zero-point is too large for `S`, so it
/// rounds to infinity. It grows with the bit width: at 16 bits, a block from
/// 0.02 to 0.05 needs a zero-point of about -76,000, past f16's 65504.
pub(crate) fn store_zero_point<S: Scale>(zero_point: f32, block_index: usize) -> Result<S> {
    let stored = S::from_f32(zero_point);
    if zero_point.is_finite() && stored.to_f32().is_infinite() {
        return Err(Error::ScaleOutOfRange {
            block_index,
            scale_type: S::NAME,
        });
    }
    Ok(stored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Scheme;

    #[test]
    fn sixteen_bit_codes_beat_eight_bit_codes_with_every_scale_type() {
        // Each extra bit halves the tick, so 16-bit codes should decode about
        // 256 times closer than 8-bit ones. Two sines give blocks with
        // different extremes, and so scales that f16 and bf16 round both ways.
        let values: Vec<f32> = (0..1024)
            .map(|i| ((i as f32).sin() + 0.3 * (i as f32 * 3.7).sin()) * 0.05)
            .collect();
        // How many times smaller the worst error is with 16-bit codes.
        fn improvement<S: Scale>(scheme: fn(u32) -> Scheme, values: &[f32]) -> f32 {
            let worst_error = |bits| {
                let back = scheme(bits).quantize::<S>(values).unwrap().dequantize();
                let errors = values.iter().zip(back).map(|(a, b)| (a - b).abs());
                errors.fold(0.0, f32::max)
            };
            worst_error(8) / worst_error(16)
        }
        let symmetric = |bits| Scheme::Symmetric { bits, block: 32 };
        let asymmetric = |bits| Scheme::Asymmetric { bits, block: 32 };
        let improvements = [
            improvement::<f32>(symmetric, &values),
            improvement::<f16>(symmetric, &values),
            improvement::<bf16>(symmetric, &values),
            improvement::<f32>(asymmetric, &values),
            improvement::<f16>(asymmetric, &values),
            improvement::<bf16>(asymmetric, &values),
        ];
        assert!(
            improvements.iter().all(|&improvement| improvement > 10.0),
            "{improvements:?}"
        );
    }
}
