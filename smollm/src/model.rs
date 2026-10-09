//! The whole model: look up the token's row of the embedding table, run it
//! through every layer, normalize it, and score every token in the vocabulary
//! as the next one. `weights.rs` builds it from a model file.

use crate::attention::KeyValueCache;
use crate::layer::{Layer, RmsNorm};
use crate::linear::Linear;

pub struct Model {
    /// One row of `hidden_size` values for each token. It's the output head
    /// too: SmolLM scores each token by the dot product of its row with the
    /// last hidden state, so one table maps tokens in and out.
    pub embedding: Linear,
    pub layers: Vec<Layer>,
    pub final_norm: RmsNorm,
}

impl Model {
    /// Feed one token, at the position after those already in `caches`, and
    /// return a score, or logit, for each token that could come next.
    pub fn forward(&self, token: u32, caches: &mut [KeyValueCache]) -> Vec<f32> {
        let mut hidden = self.embedding.row(token as usize);
        for (layer, cache) in self.layers.iter().zip(caches) {
            layer.forward(&mut hidden, cache);
        }
        self.embedding.multiply(&self.final_norm.apply(&hidden))
    }

    /// An empty cache for each layer, to start a new sequence of tokens.
    pub fn new_caches(&self) -> Vec<KeyValueCache> {
        self.layers
            .iter()
            .map(|_| KeyValueCache::default())
            .collect()
    }
}
