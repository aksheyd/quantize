use candle_core::{DType, Device, Result, Tensor};
use candle_transformers::models::llama::Config;

pub struct KvCache {
    pub use_kv_cache: bool,
    pub kvs: Vec<Option<(Tensor, Tensor)>>,
    pub cos: Tensor,
    pub sin: Tensor,
}

impl KvCache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let (cos, sin) = rope_tables(config, dtype, device)?;
        Ok(Self {
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos,
            sin,
        })
    }
}

fn rope_tables(config: &Config, dtype: DType, device: &Device) -> Result<(Tensor, Tensor)> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    let theta: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / config.rope_theta.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta, device)?;
    let length = config.max_position_embeddings;
    let index_theta = Tensor::arange(0, length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    Ok((
        index_theta.cos()?.to_dtype(dtype)?,
        index_theta.sin()?.to_dtype(dtype)?,
    ))
}

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let flags: Vec<u8> = (0..seq_len)
        .flat_map(|row| (0..seq_len).map(move |col| u8::from(col > row)))
        .collect();
    Tensor::from_slice(&flags, (seq_len, seq_len), device)
}

pub fn masked_fill(values: &Tensor, mask: &Tensor, fill: f32) -> Result<Tensor> {
    let fill = Tensor::new(fill, values.device())?.broadcast_as(mask.shape().dims())?;
    mask.where_cond(&fill, values)
}
