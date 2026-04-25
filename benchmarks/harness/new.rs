//! Build random matrices and precompute the f32 ground truth and the
//! Candle Q8_0 reference result.

use super::Harness;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, Result, Tensor,
};
use rand::Rng;

impl Harness {
    /// `matrix_size` must be a multiple of 32 (Candle Q8_0 block size).
    pub fn new(matrix_size: usize) -> Result<Self> {
        assert!(
            matrix_size % 32 == 0,
            "matrix_size must be a multiple of 32 for Candle Q8_0"
        );
        let device = Device::Cpu;
        let shape = (matrix_size, matrix_size);
        let count = matrix_size * matrix_size;

        let mut rng = rand::thread_rng();
        let matrix_a: Vec<f32> = (0..count).map(|_| rng.gen_range(-0.5..=0.5)).collect();
        let matrix_b: Vec<f32> = (0..count).map(|_| rng.gen_range(-0.5..=0.5)).collect();

        let tensor_a = Tensor::from_vec(matrix_a.clone(), shape, &device)?;
        let tensor_b = Tensor::from_vec(matrix_b.clone(), shape, &device)?;

        // f32 ground truth
        let correct_matmul = tensor_a
            .matmul(&tensor_b)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        // Candle Q8_0: round-trip A and B through Q8_0, then matmul
        let candle_a_dequantized =
            QTensor::quantize(&tensor_a, GgmlDType::Q8_0)?.dequantize(&device)?;
        let candle_b_dequantized =
            QTensor::quantize(&tensor_b, GgmlDType::Q8_0)?.dequantize(&device)?;
        let candle_matmul = candle_a_dequantized
            .matmul(&candle_b_dequantized)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        Ok(Self {
            matrix_size,
            device,
            matrix_a,
            matrix_b,
            correct_matmul,
            candle_matmul,
        })
    }
}
