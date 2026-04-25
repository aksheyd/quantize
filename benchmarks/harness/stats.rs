//! Sample mean and (population) standard deviation of a slice of floats.

pub(super) fn mean_std(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}
