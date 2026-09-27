//! Per-block range: max-abs (symmetric) and min/max (asymmetric).
//!
//! NaN is skipped and infinity is kept. The NEON loops use the "number" forms
//! (`vmaxnmq_f32`, `vminnmq_f32`), which skip NaN the same way `f32::max` and
//! `f32::min` do, so every architecture measures the same range.

#[inline]
pub(crate) fn abs_max(xs: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        abs_max_neon(xs)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut m = 0.0_f32;
        for &x in xs {
            m = m.max(x.abs());
        }
        m
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
fn abs_max_neon(xs: &[f32]) -> f32 {
    use core::arch::aarch64::*;
    let n = xs.len();
    let mut i = 0;
    // SAFETY: loads stay inside `xs`.
    unsafe {
        let mut acc = vdupq_n_f32(0.0);
        while i + 16 <= n {
            let p = xs.as_ptr().add(i);
            acc = vmaxnmq_f32(acc, vabsq_f32(vld1q_f32(p)));
            acc = vmaxnmq_f32(acc, vabsq_f32(vld1q_f32(p.add(4))));
            acc = vmaxnmq_f32(acc, vabsq_f32(vld1q_f32(p.add(8))));
            acc = vmaxnmq_f32(acc, vabsq_f32(vld1q_f32(p.add(12))));
            i += 16;
        }
        let mut m = vmaxnmvq_f32(acc);
        while i < n {
            m = m.max(xs[i].abs());
            i += 1;
        }
        m
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
    fn abs_max_matches_iterator() {
        assert_eq!(abs_max(&[0.1, -3.5, 2.0, -0.25]), 3.5);
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
        assert_eq!(abs_max(&values), 2.0);
        assert_eq!(min_max(&values), (-2.0, 0.5));
    }

    #[test]
    fn range_keeps_infinity() {
        let mut values = [0.5_f32; 20];
        values[3] = f32::NEG_INFINITY;
        assert_eq!(abs_max(&values), f32::INFINITY);
        assert_eq!(min_max(&values), (f32::NEG_INFINITY, 0.5));
    }
}
