//! Round-trip A and B through your `quantize`/`dequantize`, matmul, and
//! return a `RunReport` so the caller can decide how to present it.

use super::{
    metrics::{cosine, mse},
    quant::roundtrip,
    Comparison, Harness, Stats,
};
use candle_core::{Result, Tensor};

impl Harness {
    pub fn run(&self) -> Result<Comparison> {
        let shape = (self.matrix_size, self.matrix_size);
        let your_a = Tensor::from_vec(roundtrip(&self.matrix_a), shape, &self.device)?;
        let your_b = Tensor::from_vec(roundtrip(&self.matrix_b), shape, &self.device)?;
        let your_matmul = your_a.matmul(&your_b)?.flatten_all()?.to_vec1::<f32>()?;

        Ok(Comparison {
            matrix_size: self.matrix_size,
            quantize: Stats {
                mse: mse(&your_matmul, &self.correct_matmul),
                cosine: cosine(&your_matmul, &self.correct_matmul),
            },
            candle_q8_0: Stats {
                mse: mse(&self.candle_matmul, &self.correct_matmul),
                cosine: cosine(&self.candle_matmul, &self.correct_matmul),
            },
        })
    }
}
