//! Outer loop: for each of `runs` fresh samples, evaluate every method,
//! accumulate (mse, cosine), then collapse each method's samples to mean ± std.

use super::{
    methods::methods,
    metrics::{cosine, mse},
    stats::mean_std,
    Comparison, Harness, MethodReport, Stats,
};
use candle_core::Result;

impl Harness {
    pub fn run(&self) -> Result<Comparison> {
        let methods = methods();
        let elements = self.matrix_size * self.matrix_size;
        let mut mses: Vec<Vec<f32>> = vec![Vec::with_capacity(self.runs); methods.len()];
        let mut coss: Vec<Vec<f32>> = vec![Vec::with_capacity(self.runs); methods.len()];

        for _ in 0..self.runs {
            let s = self.sample()?;
            for (i, m) in methods.iter().enumerate() {
                let predicted = (m.eval)(&s, &self.device)?;
                mses[i].push(mse(&predicted, &s.ground_truth));
                coss[i].push(cosine(&predicted, &s.ground_truth));
            }
        }

        #[rustfmt::skip]
        let methods = methods.iter().enumerate().map(|(i, m)| {
            let (mse_mean, mse_std) = mean_std(&mses[i]);
            let (cosine_mean, cosine_std) = mean_std(&coss[i]);
            MethodReport {
                name: m.name,
                bits_per_element: m.bits_per_element.evaluate(elements),
                stats: Stats { mse_mean, mse_std, cosine_mean, cosine_std },
            }
        }).collect();

        Ok(Comparison {
            matrix_size: self.matrix_size,
            runs: self.runs,
            methods,
        })
    }
}
