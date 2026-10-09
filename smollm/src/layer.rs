//! One layer of the model: attention, then the mlp. Each reads a normalized
//! copy of the hidden state and adds what it computes back onto it, so the
//! state passes through every layer and each one only adjusts it.

use crate::attention::{Attention, KeyValueCache};
use crate::linear::Linear;

pub struct Layer {
    pub attention_norm: RmsNorm,
    pub attention: Attention,
    pub mlp_norm: RmsNorm,
    pub mlp: Mlp,
}

impl Layer {
    pub fn forward(&self, hidden: &mut [f32], cache: &mut KeyValueCache) {
        let normalized = self.attention_norm.apply(hidden);
        add(hidden, &self.attention.forward(&normalized, cache));
        let normalized = self.mlp_norm.apply(hidden);
        add(hidden, &self.mlp.forward(&normalized));
    }
}

/// Rms norm: divide the values by their root mean square, so every layer
/// reads values of about the same size, then scale each by its own weight.
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub epsilon: f32,
}

impl RmsNorm {
    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        let mean_square = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
        let scale = 1.0 / (mean_square + self.epsilon).sqrt();
        input
            .iter()
            .zip(&self.weight)
            .map(|(x, weight)| x * scale * weight)
            .collect()
    }
}

/// The mlp, a SwiGLU: two projections widen the input, one passes through
/// silu and gates the other value by value, and a third projects the result
/// back down to the hidden size.
pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl Mlp {
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let gate = self.gate.multiply(input);
        let up = self.up.multiply(input);
        let gated: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| silu(*gate) * up)
            .collect();
        self.down.multiply(&gated)
    }
}

/// `x × sigmoid(x)`: close to `x` for large `x`, and close to 0 for very
/// negative `x`, with a smooth bend in between.
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn add(hidden: &mut [f32], update: &[f32]) {
    for (value, change) in hidden.iter_mut().zip(update) {
        *value += change;
    }
}
