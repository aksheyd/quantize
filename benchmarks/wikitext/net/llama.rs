use crate::wikitext::data::load::Loaded;
use crate::wikitext::data::weights::Extracted;
use crate::wikitext::net::attn::Attention;
use crate::wikitext::net::block::{Block, Mlp};
use crate::wikitext::net::cache::KvCache;
use crate::wikitext::net::linear::Linear;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, RmsNorm};
use candle_transformers::models::llama::Config;
use half::f16;
use quantize::quantize;
use std::collections::HashMap;

pub struct LlamaNet {
    pub config: Config,
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    device: candle_core::Device,
}

impl LlamaNet {
    pub fn packed(loaded: &Loaded, extracted: &[Extracted]) -> Result<Self> {
        let mut linears = HashMap::new();
        for layer in extracted {
            let packed =
                quantize::<f16, 4, 32>(&layer.values).map_err(crate::wikitext::candle_msg)?;
            linears.insert(
                layer.name.clone(),
                Linear::packed(layer.rows, layer.columns, packed),
            );
        }
        Self::assemble(loaded, linears)
    }

    pub fn candle_q4(loaded: &Loaded, extracted: &[Extracted]) -> Result<Self> {
        let mut linears = HashMap::new();
        for layer in extracted {
            let tensor = Tensor::from_vec(
                layer.values.clone(),
                (layer.rows, layer.columns),
                &loaded.device,
            )?;
            let quantized = QTensor::quantize(&tensor, GgmlDType::Q4_0)?;
            linears.insert(
                layer.name.clone(),
                Linear::candle_packed(layer.rows, layer.columns, QMatMul::from_qtensor(quantized)?),
            );
        }
        Self::assemble(loaded, linears)
    }

    pub fn dense(loaded: &Loaded, extracted: &[Extracted]) -> Result<Self> {
        let mut linears = HashMap::new();
        for layer in extracted {
            let weight = Tensor::from_vec(
                layer.values.clone(),
                (layer.rows, layer.columns),
                &loaded.device,
            )?;
            linears.insert(
                layer.name.clone(),
                Linear::dense(layer.rows, layer.columns, weight),
            );
        }
        Self::assemble(loaded, linears)
    }

    fn assemble(loaded: &Loaded, mut linears: HashMap<String, Linear>) -> Result<Self> {
        let config = &loaded.config;
        let vb = &loaded.vb;
        let wte = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;
        let ln_f =
            candle_nn::rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let mut take = |name: &str| {
                linears
                    .remove(&format!("model.layers.{layer}.{name}"))
                    .ok_or_else(|| {
                        crate::wikitext::candle_msg(format!(
                            "missing linear {name} in layer {layer}"
                        ))
                    })
            };
            let attn = Attention::new(
                take("self_attn.q_proj")?,
                take("self_attn.k_proj")?,
                take("self_attn.v_proj")?,
                take("self_attn.o_proj")?,
                config,
            );
            let mlp = Mlp {
                gate_proj: take("mlp.gate_proj")?,
                up_proj: take("mlp.up_proj")?,
                down_proj: take("mlp.down_proj")?,
            };
            let prefix = format!("model.layers.{layer}");
            let rms_1 = candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp(format!("{prefix}.input_layernorm")),
            )?;
            let rms_2 = candle_nn::rms_norm(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp(format!("{prefix}.post_attention_layernorm")),
            )?;
            blocks.push(Block::new(attn, mlp, rms_1, rms_2));
        }
        let lm_head = linears
            .remove("lm_head")
            .ok_or_else(|| crate::wikitext::candle_msg("missing lm_head"))?;
        Ok(Self {
            config: config.clone(),
            wte,
            blocks,
            ln_f,
            lm_head,
            device: loaded.device.clone(),
        })
    }

    pub fn linears(&self) -> Vec<&Linear> {
        let mut out = Vec::new();
        for block in &self.blocks {
            out.extend([
                &block.attn.q_proj,
                &block.attn.k_proj,
                &block.attn.v_proj,
                &block.attn.o_proj,
                &block.mlp.gate_proj,
                &block.mlp.up_proj,
                &block.mlp.down_proj,
            ]);
        }
        out.push(&self.lm_head);
        out
    }

    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }

    pub fn set_store_input(&self, store: bool) {
        for linear in self.linears() {
            linear.store_input.set(store);
        }
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        index_pos: usize,
        cache: &mut KvCache,
        all_logits: bool,
    ) -> Result<Tensor> {
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let mut hidden = self.wte.forward(&input)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            hidden = block.forward(&hidden, index_pos, block_idx, cache)?;
        }
        let hidden = self.ln_f.forward(&hidden)?;
        let hidden = if all_logits {
            hidden
        } else {
            hidden.i((.., tokens.len() - 1, ..))?.contiguous()?
        };
        self.lm_head.forward(&hidden)?.to_dtype(DType::F32)
    }
}
