//! Error metrics comparing a predicted vector against the expected one.

pub(super) fn mse(predicted: &[f32], expected: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(expected)
        .map(|(p, e)| (p - e).powi(2))
        .sum::<f32>()
        / predicted.len() as f32
}
