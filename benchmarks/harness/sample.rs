//! Generate one fresh random sample (matrices A and B + their f32 matmul).
//! Called once per run so each run measures a different random instance.

use super::{Harness, Sample};
use candle_core::{Result, Tensor};
use rand::Rng;

impl Harness {
    pub(super) fn sample(&self) -> Result<Sample> {
        let shape = (self.matrix_size, self.matrix_size);
        let count = self.matrix_size * self.matrix_size;

        let mut rng = rand::thread_rng();
        let matrix_a: Vec<f32> = (0..count).map(|_| rng.gen_range(-0.5..=0.5)).collect();
        let matrix_b: Vec<f32> = (0..count).map(|_| rng.gen_range(-0.5..=0.5)).collect();

        let tensor_a = Tensor::from_vec(matrix_a.clone(), shape, &self.device)?;
        let tensor_b = Tensor::from_vec(matrix_b.clone(), shape, &self.device)?;
        let ground_truth = tensor_a
            .matmul(&tensor_b)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        Ok(Sample {
            matrix_size: self.matrix_size,
            matrix_a,
            matrix_b,
            tensor_a,
            tensor_b,
            ground_truth,
        })
    }
}
