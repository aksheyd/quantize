//! Integer grids, scale selection, and mixed-precision bit choice.

/// Panic unless `bits` is in `2..=16`, the widths codes can have. Without
/// this, a release build would wrap an overlong shift around, so 40 bits
/// would get the 8-bit answer.
pub(crate) const fn assert_bits_in_range(bits: u32) {
    assert!(2 <= bits && bits <= 16, "bits must be in 2..=16");
}

/// Largest integer code for this bit width. 8-bit → 127, 4-bit → 7.
///
/// Panics if `bits` is outside `2..=16`.
#[inline]
pub const fn largest_code(bits: u32) -> i32 {
    assert_bits_in_range(bits);
    (1_i32 << (bits - 1)) - 1
}

/// Smallest integer code for this bit width. 8-bit → -128, 4-bit → -8.
///
/// Panics if `bits` is outside `2..=16`.
#[inline]
pub const fn smallest_code(bits: u32) -> i32 {
    assert_bits_in_range(bits);
    -(1_i32 << (bits - 1))
}

/// Tick size so the value farthest from zero lands on [`smallest_code`].
///
/// Codes run from -8 to 7 at 4 bits, one more below zero than above. Putting
/// the largest magnitude on 7 would never use -8; putting it on -8 uses all 16
/// codes, so each tick is 1/8 of it instead of 1/7. A positive extreme gets a
/// negative scale, so that -8 still decodes to it, and a value on the other
/// side that would need code 8 gets 7. GGML's Q4_0 picks its scale this way.
///
/// Panics if `bits` is outside `2..=16`.
#[inline]
pub fn symmetric_scale(extreme: f32, bits: u32) -> f32 {
    assert_bits_in_range(bits);
    if extreme != 0.0 {
        extreme / smallest_code(bits) as f32
    } else {
        1.0
    }
}

/// Scale and zero-point that stretch `[lowest, highest]` onto the integer grid.
///
/// A flat block (`lowest == highest`) has no range to stretch, so it falls back
/// to [`symmetric_scale`] with a zero-point of 0. An all-NaN block measures no
/// range at all (`lowest > highest`) and keeps scale 1, so its codes of 0
/// decode to 0.
///
/// Panics if `bits` is outside `2..=16`.
#[inline]
pub fn asymmetric_params(lowest: f32, highest: f32, bits: u32) -> (f32, f32) {
    assert_bits_in_range(bits);
    if lowest > highest {
        return (1.0, 0.0);
    }
    if lowest == highest {
        return (symmetric_scale(highest, bits), 0.0);
    }
    let code_min = smallest_code(bits) as f32;
    let code_max = largest_code(bits) as f32;
    let scale = (highest - lowest) / (code_max - code_min);
    let zero_point = code_min - lowest / scale;
    (scale, zero_point)
}

/// Half a step of `bits`-wide codes stretched over `range`: the farthest any
/// value can be from its nearest code.
pub(crate) fn half_step(range: f32, bits: u32) -> f32 {
    let tick_count = ((1u32 << bits) - 1) as f32;
    range / tick_count / 2.0
}

/// Smallest bit width in `2..=8` whose half-step is `<= tolerance`, or `None`
/// if even 8 bits step wider than that.
///
/// A flat block (range `0`) always returns 2.
pub fn choose_bits(range: f32, tolerance: f32) -> Option<u32> {
    if range <= 0.0 {
        return Some(2);
    }
    (2..=8).find(|&bits| half_step(range, bits) <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_bit_codes_run_from_minus_eight_to_seven() {
        assert_eq!(largest_code(4), 7);
        assert_eq!(smallest_code(4), -8);
    }

    #[test]
    #[should_panic(expected = "bits must be in 2..=16")]
    fn zero_bits_panic_instead_of_wrapping_around() {
        largest_code(0);
    }

    #[test]
    #[should_panic(expected = "bits must be in 2..=16")]
    fn forty_bits_panic_instead_of_giving_the_eight_bit_scale() {
        symmetric_scale(1.0, 40);
    }

    #[test]
    fn symmetric_scale_puts_the_extreme_on_minus_eight() {
        assert_eq!(symmetric_scale(-2.0, 4), 0.25);
        assert_eq!(symmetric_scale(2.0, 4), -0.25);
        assert_eq!(symmetric_scale(0.0, 4), 1.0);
    }

    #[test]
    fn choose_bits_picks_two_for_tiny_range() {
        assert_eq!(choose_bits(0.001, 0.001), Some(2));
    }

    #[test]
    fn choose_bits_stops_at_eight() {
        // 8 bits round a range of 10 to within 10 / 255 / 2, about 0.0196.
        assert_eq!(choose_bits(10.0, 0.02), Some(8));
        assert_eq!(choose_bits(10.0, 0.0001), None);
    }
}
