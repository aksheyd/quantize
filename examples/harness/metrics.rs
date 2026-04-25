//! Error metrics comparing a predicted vector against the expected one.
//! The smaller the error, the better the prediction.
//!
//! predicted refers to the output of the model,
//! while expected is the ground truth.

pub(super) fn mse(predicted: &[f32], expected: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(expected)
        .map(|(p, e)| (p - e).powi(2))
        .sum::<f32>()
        / predicted.len() as f32
}

pub(super) fn cosine(predicted: &[f32], expected: &[f32]) -> f32 {
    let dot: f32 = predicted.iter().zip(expected).map(|(p, e)| p * e).sum();
    let norm_predicted: f32 = predicted.iter().map(|p| p * p).sum::<f32>().sqrt();
    let norm_expected: f32 = expected.iter().map(|e| e * e).sum::<f32>().sqrt();
    if norm_predicted == 0.0 || norm_expected == 0.0 {
        0.0
    } else {
        dot / (norm_predicted * norm_expected)
    }
}
