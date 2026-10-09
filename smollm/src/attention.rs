//! Attention: how each token gathers what it needs from the tokens before it.
//!
//! Each head compares the token's query with the key of every token so far,
//! itself included, and takes a weighted average of their values: the better
//! a key matches, the more of its value goes in. Keys and values don't change
//! once computed, so a cache keeps them, and each new token computes only its
//! own.

use crate::config::Config;
use crate::linear::{Linear, dot};

/// The keys and values of every token so far, for one layer: one row of
/// `key_value_head_count × head_dim` values per token, row after row.
#[derive(Default)]
pub struct KeyValueCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}

pub struct Attention {
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub output: Linear,
    pub config: Config,
}

impl Attention {
    pub fn forward(&self, input: &[f32], cache: &mut KeyValueCache) -> Vec<f32> {
        let config = &self.config;
        let head_dim = config.head_dim();
        let row_width = config.key_value_head_count * head_dim;
        // The token's position is how many tokens came before it.
        let position = cache.keys.len() / row_width;
        let mut queries = self.query.multiply(input);
        let mut keys = self.key.multiply(input);
        rotate(&mut queries, position, head_dim, config.rope_theta);
        rotate(&mut keys, position, head_dim, config.rope_theta);
        cache.keys.extend(keys);
        cache.values.extend(self.value.multiply(input));

        // Heads share keys and values in groups: in SmolLM-135M, heads 0 to 2
        // read key-value head 0, heads 3 to 5 read head 1, and so on.
        let group_size = config.head_count / config.key_value_head_count;
        // A dot product of wider heads adds up more terms and spreads wider.
        // Dividing by √head_dim keeps the scores on one scale for any width,
        // so softmax doesn't put all its weight on a single token.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut mixed = vec![0.0; queries.len()];
        for head in 0..config.head_count {
            // Where this head's values start in `queries` and `mixed`, and
            // where its key-value head's start in each row of the cache.
            let own = head * head_dim;
            let shared = head / group_size * head_dim;
            let query = &queries[own..own + head_dim];
            let past_keys = cache.keys.chunks_exact(row_width);
            let scores = past_keys.map(|keys| dot(query, &keys[shared..shared + head_dim]) * scale);
            let mut weights: Vec<f32> = scores.collect();
            softmax(&mut weights);
            let past_values = cache.values.chunks_exact(row_width);
            for (weight, values) in weights.iter().zip(past_values) {
                let head_values = &values[shared..shared + head_dim];
                for (mixed, value) in mixed[own..own + head_dim].iter_mut().zip(head_values) {
                    *mixed += weight * value;
                }
            }
        }
        self.output.multiply(&mixed)
    }
}

/// Rope: rotate pairs of each head's values by angles that grow with the
/// token's position, so how well a query matches a key depends on how far
/// apart their tokens are. Pair `i` turns `position / theta^(2i / head_dim)`
/// radians, fast for the first pairs and slow for the last. Hugging Face's
/// llama pairs value `i` with value `i + head_dim / 2`.
fn rotate(heads: &mut [f32], position: usize, head_dim: usize, theta: f32) {
    for head in heads.chunks_exact_mut(head_dim) {
        let (first_half, second_half) = head.split_at_mut(head_dim / 2);
        for (i, (x, y)) in first_half.iter_mut().zip(second_half).enumerate() {
            let frequency = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let (sin, cos) = (position as f32 * frequency).sin_cos();
            (*x, *y) = (*x * cos - *y * sin, *y * cos + *x * sin);
        }
    }
}

/// Turn scores into weights that are positive and add up to 1. Subtracting
/// the largest score first keeps `exp` from overflowing.
fn softmax(scores: &mut [f32]) {
    let largest = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for score in scores.iter_mut() {
        *score = (*score - largest).exp();
    }
    let total: f32 = scores.iter().sum();
    for score in scores {
        *score /= total;
    }
}
