use crate::wikitext::net::cache::{causal_mask, masked_fill, KvCache};
use crate::wikitext::net::linear::Linear;
use candle_core::{DType, Result, Tensor};
use candle_transformers::models::llama::Config;
use candle_transformers::utils::repeat_kv;

pub struct Attention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(q: Linear, k: Linear, v: Linear, o: Linear, config: &Config) -> Self {
        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
        }
    }

    pub fn forward(
        &self,
        hidden: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, hidden_size) = hidden.dims3()?;
        let query =
            self.reshape_heads(self.q_proj.forward(hidden)?, batch, seq_len, self.n_heads)?;
        let key = self.reshape_heads(
            self.k_proj.forward(hidden)?,
            batch,
            seq_len,
            self.n_kv_heads,
        )?;
        let mut value = self.reshape_heads(
            self.v_proj.forward(hidden)?,
            batch,
            seq_len,
            self.n_kv_heads,
        )?;
        let query = apply_rope(&query, index_pos, seq_len, cache)?;
        let mut key = apply_rope(&key, index_pos, seq_len, cache)?;
        if cache.use_kv_cache {
            if let Some((past_key, past_value)) = &cache.kvs[block_idx] {
                key = Tensor::cat(&[past_key, &key], 2)?.contiguous()?;
                value = Tensor::cat(&[past_value, &value], 2)?.contiguous()?;
            }
            cache.kvs[block_idx] = Some((key.clone(), value.clone()));
        }
        let repeats = self.n_heads / self.n_kv_heads;
        let key = repeat_kv(key, repeats)?;
        let value = repeat_kv(value, repeats)?;
        let scale = (self.head_dim as f64).sqrt();
        let mut scores = query
            .to_dtype(DType::F32)?
            .matmul(&key.to_dtype(DType::F32)?.t()?)?;
        scores = (scores / scale)?;
        if seq_len > 1 {
            let mask = causal_mask(seq_len, hidden.device())?.broadcast_as(scores.shape())?;
            scores = masked_fill(&scores, &mask, f32::NEG_INFINITY)?;
        }
        let context = candle_nn::ops::softmax_last_dim(&scores)?
            .matmul(&value.to_dtype(DType::F32)?.contiguous()?)?;
        let context = context
            .transpose(1, 2)?
            .reshape((batch, seq_len, hidden_size))?;
        self.o_proj.forward(&context)
    }

    fn reshape_heads(
        &self,
        projected: Tensor,
        batch: usize,
        seq_len: usize,
        heads: usize,
    ) -> Result<Tensor> {
        projected
            .reshape((batch, seq_len, heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }
}

fn apply_rope(
    states: &Tensor,
    index_pos: usize,
    seq_len: usize,
    cache: &KvCache,
) -> Result<Tensor> {
    let cos = cache.cos.narrow(0, index_pos, seq_len)?;
    let sin = cache.sin.narrow(0, index_pos, seq_len)?;
    candle_nn::rotary_emb::rope(states, &cos, &sin)
}
