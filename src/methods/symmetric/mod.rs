//! Symmetric quantization: one scale per group, codes centered on zero.

use crate::error::{Result, check_bits, check_block};
use crate::kernels::quantize_sym_packed;
use crate::packed::Packed;
use crate::scale::Scale;
use crate::tensor::Quantized;

/// Quantize `values` into fixed-size blocks of `BLOCK` with `BITS`-wide codes.
///
/// `BITS` must be from 2 to 16, and `BLOCK` at least 1. Anything else stops
/// the build, though `cargo check` doesn't catch it:
///
/// ```compile_fail
/// let q = quantize::quantize::<f32, 1, 32>(&[0.5]);
/// ```
///
/// # Errors
///
/// [`crate::Error::ScaleOutOfRange`] if `S` can't hold a block's scale.
pub fn quantize<S: Scale, const BITS: u32, const BLOCK: usize>(
    values: &[f32],
) -> Result<Quantized<S>> {
    const {
        assert!(2 <= BITS && BITS <= 16, "BITS must be from 2 to 16");
        assert!(BLOCK >= 1, "BLOCK must be at least 1");
    }
    quantize_with::<S>(values, BITS, BLOCK)
}

/// [`quantize`] with the bit width and block size chosen at run time.
///
/// # Errors
///
/// [`crate::Error::InvalidBits`] or [`crate::Error::InvalidBlock`], and
/// [`crate::Error::ScaleOutOfRange`] if `S` can't hold a block's scale.
pub fn quantize_with<S: Scale>(values: &[f32], bits: u32, block: usize) -> Result<Quantized<S>> {
    check_bits(bits)?;
    check_block(block)?;
    if values.is_empty() {
        return Ok(Quantized::Symmetric {
            scales: Vec::new(),
            codes: Packed::from_raw(Vec::new(), bits, 0),
            block,
            len: 0,
            columns: None,
        });
    }
    let (scales, codes) = quantize_sym_packed::<S>(values, bits, block)?;
    Ok(Quantized::Symmetric {
        scales,
        codes,
        block,
        len: values.len(),
        columns: None,
    })
}

/// Quantize the entire tensor with a single scale. `BITS` must be from 2 to
/// 16, as in [`quantize`].
pub fn quantize_tensor<S: Scale, const BITS: u32>(values: &[f32]) -> Result<Quantized<S>> {
    const { assert!(2 <= BITS && BITS <= 16, "BITS must be from 2 to 16") };
    quantize_with::<S>(values, BITS, values.len().max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eight_bit_roundtrip_stays_within_half_step() {
        let w = [0.42_f32, -0.10, 0.70, -0.50];
        let q = quantize::<f32, 8, 4>(&w).unwrap();
        let back = q.dequantize();
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn farthest_value_lands_on_the_most_negative_code() {
        // With 0.8, 1.6, or 12.8 on code -8, -16, or -128, each tick is 0.1,
        // so every multiple of 0.1 decodes exactly, on either side of zero.
        for (bits, extreme) in [(4, 0.8_f32), (5, 1.6), (8, 12.8)] {
            for sign in [1.0, -1.0] {
                let w = [extreme, -0.4, 0.1, 0.1 - extreme].map(|value| value * sign);
                let back = quantize_with::<f32>(&w, bits, 4).unwrap().dequantize();
                for (a, b) in w.iter().zip(&back) {
                    assert!((a - b).abs() < 1e-5, "{bits} bits: {a} vs {b}");
                }
            }
        }
    }

    #[test]
    fn codes_are_picked_against_the_stored_scale() {
        // bf16 rounds each scale, yet every value decodes within half a tick
        // of it. All positive, so no value is clamped on the far side of zero.
        let w: Vec<f32> = (0..1024).map(|i| 0.6 + (i as f32).sin() * 0.5).collect();
        let q = quantize_with::<half::bf16>(&w, 8, 32).unwrap();
        let back = q.dequantize();
        let blocks = w.chunks(32).zip(back.chunks(32)).zip(q.scales());
        for ((values, decoded), scale) in blocks {
            let half_tick = scale.to_f32().abs() / 2.0;
            for (a, b) in values.iter().zip(decoded) {
                assert!((a - b).abs() <= half_tick * 1.001, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn packed_four_bit_uses_half_byte_per_code() {
        let w = [0.1_f32; 32];
        let q = quantize::<f32, 4, 32>(&w).unwrap();
        assert_eq!(q.codes().len(), 16);
    }

    #[test]
    fn remainder_block_roundtrips() {
        let w: Vec<f32> = (0..40).map(|i| (i as f32) * 0.01 - 0.2).collect();
        let q = quantize::<f32, 8, 32>(&w).unwrap();
        assert_eq!(q.len(), 40);
        let back = q.dequantize();
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.01, "{a} vs {b}");
        }
    }

    #[test]
    fn nan_quantizes_like_zero() {
        let mut with_nan: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.15).collect();
        let mut with_zero = with_nan.clone();
        with_nan[3] = f32::NAN;
        with_zero[3] = 0.0;
        for bits in [4, 5, 8] {
            let nan = quantize_with::<f32>(&with_nan, bits, 32).unwrap();
            let zero = quantize_with::<f32>(&with_zero, bits, 32).unwrap();
            assert_eq!(nan.dequantize(), zero.dequantize(), "{bits} bits");
        }
    }

    #[test]
    fn infinity_turns_its_block_into_nan() {
        let mut w = vec![0.1_f32; 64];
        w[3] = f32::INFINITY;
        for bits in [4, 5, 8] {
            let back = quantize_with::<f32>(&w, bits, 32).unwrap().dequantize();
            assert!(back[..32].iter().all(|v| v.is_nan()), "{bits} bits");
            assert!(
                back[32..].iter().all(|v| (v - 0.1).abs() < 0.01),
                "{bits} bits"
            );
        }
    }

    #[test]
    fn a_scale_that_f16_cannot_hold_is_an_error() {
        // 1e6 on code -8, 1e7 on code -128, and 1e10 on code -32768 all need
        // scales past f16's 65504.
        for (bits, extreme) in [(4, 1e6_f32), (8, 1e7), (16, 1e10)] {
            let values = [0.5, -0.5, extreme, 0.0];
            assert_eq!(
                quantize_with::<half::f16>(&values, bits, 2),
                Err(crate::Error::ScaleOutOfRange {
                    block_index: 1,
                    scale_type: "f16"
                })
            );
            assert!(quantize_with::<f32>(&values, bits, 2).is_ok());
        }
    }

    #[test]
    fn dequantize_into_rejects_wrong_length() {
        let w = [0.1_f32; 8];
        let q = quantize::<f32, 8, 8>(&w).unwrap();
        let mut out = [0.0f32; 3];
        assert!(matches!(
            q.dequantize_into(&mut out),
            Err(crate::Error::LengthMismatch {
                expected: 8,
                got: 3
            })
        ));
    }

    #[test]
    fn four_bit_remainder_roundtrips() {
        let w: Vec<f32> = (0..40).map(|i| (i as f32) * 0.02 - 0.4).collect();
        let q = quantize::<f32, 4, 32>(&w).unwrap();
        let back = q.dequantize();
        for (a, b) in w.iter().zip(&back) {
            assert!((a - b).abs() < 0.08, "{a} vs {b}");
        }
    }

    #[test]
    #[should_panic]
    fn four_bit_dequantize_panics_on_short_codes() {
        let short = Quantized::<f32>::Symmetric {
            scales: vec![1.0; 2],
            codes: Packed::from_raw(vec![0; 16], 4, 64),
            block: 32,
            len: 64,
            columns: None,
        };
        short.dequantize();
    }

    #[test]
    fn fused_dot_matches_dequant_then_dot() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let q = quantize::<f32, 8, 32>(&w).unwrap();
        let recon = q.dequantize();
        let naive: f32 = recon.iter().zip(&w).map(|(a, b)| a * b).sum();
        let fused = q.dot(&w).unwrap();
        assert!((naive - fused).abs() < 1e-4, "{naive} vs {fused}");
    }

    #[test]
    fn fused_dot_four_bit_matches_dequant_then_dot() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 - 0.4).collect();
        let q = quantize::<f32, 4, 32>(&w).unwrap();
        let recon = q.dequantize();
        let naive: f32 = recon.iter().zip(&w).map(|(a, b)| a * b).sum();
        let fused = q.dot(&w).unwrap();
        assert!((naive - fused).abs() < 1e-4, "{naive} vs {fused}");
    }

    #[test]
    fn fused_matmul_matches_dequant_then_multiply() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let mut q = quantize::<f32, 8, 32>(&w).unwrap();
        q.set_shape(2, 32).unwrap();
        let recon = q.dequantize();
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.02 - 0.1).collect();
        let fused = q.matmul(&input).unwrap();
        for row in 0..2 {
            let naive: f32 = recon[row * 32..(row + 1) * 32]
                .iter()
                .zip(&input)
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (naive - fused[row]).abs() < 1e-4,
                "{naive} vs {}",
                fused[row]
            );
        }
    }

    #[test]
    fn fused_matmul_four_bit_matches_dequant_then_multiply() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 - 0.4).collect();
        let mut q = quantize::<f32, 4, 32>(&w).unwrap();
        q.set_shape(2, 32).unwrap();
        let recon = q.dequantize();
        let input: Vec<f32> = (0..32).map(|i| (i as f32) * 0.03 - 0.2).collect();
        let fused = q.matmul(&input).unwrap();
        for row in 0..2 {
            let naive: f32 = recon[row * 32..(row + 1) * 32]
                .iter()
                .zip(&input)
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                (naive - fused[row]).abs() < 1e-4,
                "{naive} vs {}",
                fused[row]
            );
        }
    }

    #[test]
    fn matmul_batch_is_row_major() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let mut q = quantize::<f32, 8, 32>(&w).unwrap();
        q.set_shape(2, 32).unwrap();
        let recon = q.dequantize();
        let inputs: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02 - 0.15).collect();
        let fused = q.matmul(&inputs).unwrap();
        assert_eq!(fused.len(), 4);
        for vector in 0..2 {
            for row in 0..2 {
                let naive: f32 = recon[row * 32..(row + 1) * 32]
                    .iter()
                    .zip(&inputs[vector * 32..(vector + 1) * 32])
                    .map(|(a, b)| a * b)
                    .sum();
                let got = fused[vector * 2 + row];
                assert!((naive - got).abs() < 1e-4, "{naive} vs {got}");
            }
        }
    }

    #[test]
    fn matmul_into_writes_every_value_of_out() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let mut q = quantize::<f32, 8, 32>(&w).unwrap();
        q.set_shape(2, 32).unwrap();
        let inputs: Vec<f32> = (0..96).map(|i| (i as f32) * 0.02 - 0.15).collect();
        let mut out = [f32::NAN; 3 * 2];
        q.matmul_into(&inputs, &mut out).unwrap();
        assert_eq!(out.to_vec(), q.matmul(&inputs).unwrap());
    }

    #[test]
    fn matmul_into_catches_a_shape_recorded_the_wrong_way_round() {
        // A layer with 2 outputs and 32 inputs, recorded as 32 rows of 2. A
        // batch of 4 inputs also splits into 64 vectors of 2, so matmul
        // returns 64 × 32 values where 4 × 2 were expected.
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01 - 0.3).collect();
        let mut q = quantize::<f32, 8, 32>(&w).unwrap();
        q.set_shape(32, 2).unwrap();
        let inputs = [0.1_f32; 4 * 32];
        assert_eq!(q.matmul(&inputs).unwrap().len(), 64 * 32);
        assert_eq!(
            q.matmul_into(&inputs, &mut [0.0; 4 * 2]),
            Err(crate::Error::OutputMismatch {
                batch: 64,
                rows: 32,
                got: 4 * 2
            })
        );
    }

    #[test]
    fn set_shape_rejects_zero_columns() {
        let mut q = quantize::<f32, 8, 8>(&[0.1; 8]).unwrap();
        assert_eq!(
            q.set_shape(8, 0),
            Err(crate::Error::ShapeMismatch { len: 8, columns: 0 })
        );
    }

    #[test]
    fn set_shape_rejects_a_shape_that_does_not_hold_every_value() {
        let w: Vec<f32> = (0..40).map(|i| (i as f32) * 0.01).collect();
        let mut q = quantize::<f32, 8, 32>(&w).unwrap();
        let unchanged = q.clone();
        for (rows, columns) in [(1, 32), (usize::MAX, 2)] {
            assert_eq!(
                q.set_shape(rows, columns),
                Err(crate::Error::MatrixMismatch {
                    rows,
                    columns,
                    len: 40
                })
            );
        }
        assert_eq!(q, unchanged);
    }

    #[test]
    fn set_shape_records_rows_and_columns() {
        let mut q = quantize::<f32, 8, 32>(&[0.1; 64]).unwrap();
        assert_eq!(q.shape(), None);
        q.set_shape(2, 32).unwrap();
        assert_eq!(q.shape(), Some((2, 32)));
    }

    #[test]
    fn matmul_rejects_a_flat_vector() {
        let q = quantize::<f32, 8, 32>(&[0.1; 64]).unwrap();
        assert!(matches!(
            q.matmul(&[0.0; 32]),
            Err(crate::Error::NotAMatrix { len: 64 })
        ));
    }

    #[test]
    fn dequantize_row_into_rejects_a_flat_vector_a_row_past_the_end_and_a_wrong_output() {
        let mut q = quantize::<f32, 8, 32>(&[0.1; 64]).unwrap();
        let mut out = [0.0; 32];
        assert_eq!(
            q.dequantize_row_into(0, &mut out),
            Err(crate::Error::NotAMatrix { len: 64 })
        );
        q.set_shape(2, 32).unwrap();
        assert_eq!(
            q.dequantize_row_into(2, &mut out),
            Err(crate::Error::RowOutOfRange { row: 2, rows: 2 })
        );
        assert_eq!(
            q.dequantize_row_into(1, &mut [0.0; 16]),
            Err(crate::Error::LengthMismatch {
                expected: 32,
                got: 16
            })
        );
    }

    #[test]
    fn matmul_rejects_inputs_that_do_not_split_into_vectors() {
        let w: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let mut q = quantize::<f32, 8, 32>(&w).unwrap();
        q.set_shape(2, 32).unwrap();
        let inputs = [0.0_f32; 48];
        let mismatch = crate::Error::InputMismatch {
            columns: 32,
            got: 48,
        };
        assert_eq!(q.matmul(&inputs), Err(mismatch.clone()));
        assert_eq!(q.matmul_into(&inputs, &mut [0.0; 2]), Err(mismatch));
    }

    #[test]
    fn matmul_rejects_an_output_too_large_to_allocate() {
        // matmul reads only the shape before it sizes the output, so a tensor
        // that claims more values than it stores is enough to reach that check.
        // `usize::MAX` rows overflow the multiplication; `usize::MAX / 2` rows
        // pass it but need more bytes than any allocation can hold.
        for len in [usize::MAX, usize::MAX / 2] {
            let huge = Quantized::<f32>::Symmetric {
                scales: Vec::new(),
                codes: Packed::from_raw(Vec::new(), 8, 0),
                block: 1,
                len,
                columns: Some(1),
            };
            assert!(matches!(
                huge.matmul(&[0.0; 2]),
                Err(crate::Error::OutputTooLarge { batch: 2, rows }) if rows == len
            ));
        }
    }

    #[test]
    fn matmul_into_rejects_a_result_too_large_to_count() {
        // As above, 2 inputs times `usize::MAX` rows overflow a `usize`.
        let huge = Quantized::<f32>::Symmetric {
            scales: Vec::new(),
            codes: Packed::from_raw(Vec::new(), 8, 0),
            block: 1,
            len: usize::MAX,
            columns: Some(1),
        };
        assert_eq!(
            huge.matmul_into(&[0.0; 2], &mut []),
            Err(crate::Error::OutputTooLarge {
                batch: 2,
                rows: usize::MAX
            })
        );
    }
}
