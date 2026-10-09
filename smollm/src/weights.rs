//! Build the model from a model file's tensors, which Hugging Face's llama
//! names like `model.layers.0.self_attn.q_proj.weight`.

use std::collections::BTreeMap;

use quantize::f16;
use quantize_files::Tensor;

use crate::attention::Attention;
use crate::config::Config;
use crate::layer::{Layer, Mlp, RmsNorm};
use crate::linear::Linear;
use crate::model::Model;

impl Model {
    /// Build the model from a model file's tensors. Each matrix becomes a
    /// float or a quantized [`Linear`], as its tensor is, so this builds the
    /// f32 model and the quantized one alike.
    pub fn new(config: &Config, tensors: BTreeMap<String, Tensor<f16>>) -> Result<Self, String> {
        let mut weights = Weights {
            tensors,
            config: *config,
        };
        let (hidden, intermediate) = (config.hidden_size, config.intermediate_size);
        let query_width = config.head_count * config.head_dim();
        let key_value_width = config.key_value_head_count * config.head_dim();
        let mut layers = Vec::new();
        for index in 0..config.layer_count {
            let name = |part: &str| format!("model.layers.{index}.{part}.weight");
            let attention = Attention {
                query: weights.matrix(&name("self_attn.q_proj"), query_width, hidden)?,
                key: weights.matrix(&name("self_attn.k_proj"), key_value_width, hidden)?,
                value: weights.matrix(&name("self_attn.v_proj"), key_value_width, hidden)?,
                output: weights.matrix(&name("self_attn.o_proj"), hidden, query_width)?,
                config: *config,
            };
            let mlp = Mlp {
                gate: weights.matrix(&name("mlp.gate_proj"), intermediate, hidden)?,
                up: weights.matrix(&name("mlp.up_proj"), intermediate, hidden)?,
                down: weights.matrix(&name("mlp.down_proj"), hidden, intermediate)?,
            };
            layers.push(Layer {
                attention_norm: weights.norm(&name("input_layernorm"))?,
                attention,
                mlp_norm: weights.norm(&name("post_attention_layernorm"))?,
                mlp,
            });
        }
        Ok(Self {
            embedding: weights.matrix("model.embed_tokens.weight", config.vocab_size, hidden)?,
            layers,
            final_norm: weights.norm("model.norm.weight")?,
        })
    }
}

/// A model file's tensors, taken out by name as the model is built, each
/// checked against the sizes in the config.
struct Weights {
    tensors: BTreeMap<String, Tensor<f16>>,
    config: Config,
}

impl Weights {
    fn take(&mut self, name: &str) -> Result<Tensor<f16>, String> {
        self.tensors
            .remove(name)
            .ok_or(format!("the model file has no {name}"))
    }

    fn matrix(&mut self, name: &str, rows: usize, columns: usize) -> Result<Linear, String> {
        Linear::new(name, self.take(name)?, rows, columns)
    }

    fn norm(&mut self, name: &str) -> Result<RmsNorm, String> {
        let (hidden, epsilon) = (self.config.hidden_size, self.config.rms_norm_epsilon);
        match self.take(name)? {
            Tensor::Float { shape, values } if shape == [hidden] => Ok(RmsNorm {
                weight: values,
                epsilon,
            }),
            _ => Err(format!("{name} should hold {hidden} floats")),
        }
    }
}
