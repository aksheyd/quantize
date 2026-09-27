//! Per-block range: min/max (asymmetric), and from it the value farthest from
//! zero (symmetric).
//!
//! NaN is skipped and infinity is kept. The NEON loop uses the "number" forms
//! (`vmaxnmq_f32`, `vminnmq_f32`), which skip NaN the same way `f32::max` and
//! `f32::min` do, so every architecture measures the same range.

/// The value farthest from zero, sign kept: `[0.5, -2.0, 1.0]` gives `-2.0`.
/// A block of only NaN has no extreme and gives 0.
#[inline]
pub(crate) fn signed_extreme(xs: &[f32]) -> f32 {
    let (lowest, highest) = min_max(xs);
    if lowest > highest {
        0.0
    } else if -lowest > highest {
        lowest
    } else {
        highest
    }
}

#[inline]
pub(crate) fn min_max(xs: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    {
        min_max_neon(xs)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &x in xs {
            lo = lo.min(x);
            hi = hi.max(x);
        }
        (lo, hi)
    }
}

#[cfg(target_arch = "aarch64")]
fn min_max_neon(xs: &[f32]) -> (f32, f32) {
    use core::arch::aarch64::*;
    if xs.is_empty() {
        return (f32::INFINITY, f32::NEG_INFINITY);
    }
    let n = xs.len();
    let mut i = 0;
    // SAFETY: loads stay inside `xs`.
    unsafe {
        let mut vlo = vdupq_n_f32(f32::INFINITY);
        let mut vhi = vdupq_n_f32(f32::NEG_INFINITY);
        while i + 8 <= n {
            let p = xs.as_ptr().add(i);
            let a = vld1q_f32(p);
            let b = vld1q_f32(p.add(4));
            vlo = vminnmq_f32(vlo, vminnmq_f32(a, b));
            vhi = vmaxnmq_f32(vhi, vmaxnmq_f32(a, b));
            i += 8;
        }
        let mut lo = vminnmvq_f32(vlo);
        let mut hi = vmaxnmvq_f32(vhi);
        while i < n {
            lo = lo.min(xs[i]);
            hi = hi.max(xs[i]);
            i += 1;
        }
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_extreme_keeps_the_sign() {
        assert_eq!(signed_extreme(&[0.1, -3.5, 2.0, -0.25]), -3.5);
        assert_eq!(signed_extreme(&[0.1, -2.0, 3.5, -0.25]), 3.5);
    }

    #[test]
    fn min_max_matches_iterator() {
        assert_eq!(min_max(&[0.1, -3.5, 2.0]), (-3.5, 2.0));
    }

    // On aarch64 the first 16 values take the NEON loop and the last 4 the scalar tail.
    #[test]
    fn range_skips_nan() {
        let mut values = [0.5_f32; 20];
        values[3] = f32::NAN;
        values[5] = -2.0;
        values[17] = f32::NAN;
        assert_eq!(signed_extreme(&values), -2.0);
        assert_eq!(min_max(&values), (-2.0, 0.5));
        assert_eq!(signed_extreme(&[f32::NAN; 4]), 0.0);
    }

    #[test]
    fn range_keeps_infinity() {
        let mut values = [0.5_f32; 20];
        values[3] = f32::NEG_INFINITY;
        assert_eq!(signed_extreme(&values), f32::NEG_INFINITY);
        assert_eq!(min_max(&values), (f32::NEG_INFINITY, 0.5));
    }
}
