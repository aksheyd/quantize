//! Integer grids, scale selection, and mixed-precision bit choice.

/// Largest integer code for this bit width. 8-bit → 127, 4-bit → 7.
#[inline]
pub const fn largest_code(bits: u32) -> i32 {
    (1_i32 << (bits - 1)) - 1
}

/// Smallest integer code for this bit width. 8-bit → -128, 4-bit → -8.
#[inline]
pub const fn smallest_code(bits: u32) -> i32 {
    -(1_i32 << (bits - 1))
}

/// Tick size so the biggest absolute value lands on [`largest_code`].
#[inline]
pub fn symmetric_scale(max_abs: f32, bits: u32) -> f32 {
    if max_abs > 0.0 {
        max_abs / largest_code(bits) as f32
    } else {
        1.0
    }
}

/// Scale and zero-point that stretch `[lowest, highest]` onto the integer grid.
#[inline]
pub fn asymmetric_params(lowest: f32, highest: f32, bits: u32) -> (f32, f32) {
    if lowest >= highest {
        return (1.0, 0.0);
    }
    let code_min = smallest_code(bits) as f32;
    let code_max = largest_code(bits) as f32;
    let scale = (highest - lowest) / (code_max - code_min);
    let zero_point = code_min - lowest / scale;
    (scale, zero_point)
}

/// Smallest bit width in `2..=8` whose half-step is `<= tolerance`.
///
/// A flat block (range `0`) always returns 2.
pub fn choose_bits(range: f32, tolerance: f32) -> u32 {
    if range <= 0.0 {
        return 2;
    }
    for bits in 2..=8 {
        let tick_count = ((1u32 << bits) - 1) as f32;
        let half_step = range / tick_count / 2.0;
        if half_step <= tolerance {
            return bits;
        }
    }
    8
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
    fn choose_bits_picks_two_for_tiny_range() {
        assert_eq!(choose_bits(0.001, 0.001), 2);
    }

    #[test]
    fn choose_bits_saturates_at_eight() {
        assert_eq!(choose_bits(10.0, 0.0001), 8);
    }
}
