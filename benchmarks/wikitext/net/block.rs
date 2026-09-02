use crate::wikitext::net::attn::Attention;
use crate::wikitext::net::cache::KvCache;
use crate::wikitext::net::linear::Linear;
use candle_core::{Result, Tensor};
use candle_nn::{Module, RmsNorm};

pub struct Mlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl Mlp {
    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gated = candle_nn::ops::silu(&self.gate_proj.forward(hidden)?)?;
        self.down_proj
            .forward(&(&gated * self.up_proj.forward(hidden)?)?)
    }
}

pub struct Block {
    pub attn: Attention,
    pub mlp: Mlp,
    rms_1: RmsNorm,
    rms_2: RmsNorm,
}

impl Block {
    pub fn new(attn: Attention, mlp: Mlp, rms_1: RmsNorm, rms_2: RmsNorm) -> Self {
        Self {
            attn,
            mlp,
            rms_1,
            rms_2,
        }
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let residual = hidden;
        let hidden =
            (self
                .attn
                .forward(&self.rms_1.forward(hidden)?, index_pos, block_idx, cache)?
                + residual)?;
        let residual = &hidden;
        let hidden = (self.mlp.forward(&self.rms_2.forward(&hidden)?)? + residual)?;
        Ok(hidden)
    }
}
