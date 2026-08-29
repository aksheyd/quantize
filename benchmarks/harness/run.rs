//! Outer loop: for each of `runs` fresh samples, evaluate every method,
//! then collapse each method's MSE samples to a mean.

use super::{methods::methods, metrics::mse, Comparison, Harness, MethodReport};
use candle_core::Result;

impl Harness {
    pub fn run(&self) -> Result<Comparison> {
        let methods = methods();
        let elements = self.matrix_size * self.matrix_size;
        let mut mses: Vec<Vec<f32>> = vec![Vec::with_capacity(self.runs); methods.len()];

        for _ in 0..self.runs {
            let s = self.sample()?;
            for (i, m) in methods.iter().enumerate() {
                let predicted = (m.eval)(&s, &self.device)?;
                mses[i].push(mse(&predicted, &s.ground_truth));
            }
        }

        let methods = methods
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let n = mses[i].len() as f32;
                MethodReport {
                    bits_per_element: m.bits_per_element.evaluate(elements),
                    mse: mses[i].iter().sum::<f32>() / n,
                }
            })
            .collect();

        Ok(Comparison { methods })
    }
}
