use crate::wikitext::net::cache::KvCache;
use crate::wikitext::net::LlamaNet;
use crate::wikitext::{DECODE_NEW, DECODE_PROMPT};
use candle_core::{DType, Result, Tensor};
use std::time::Instant;

pub fn tokens(model: &LlamaNet, tokens: &[u32]) -> Result<f64> {
    let prompt = prompt_tokens(tokens);
    if prompt.is_empty() {
        return Ok(0.0);
    }
    let mut cache = KvCache::new(true, DType::F32, &model.config, model.device())?;
    let mut logits = model.forward(&prompt, 0, &mut cache, false)?;
    let mut next = argmax_tensor(&logits)?;
    let started = Instant::now();
    for step in 0..DECODE_NEW {
        logits = model.forward(&[next], prompt.len() + step, &mut cache, false)?;
        next = argmax_tensor(&logits)?;
    }
    Ok(DECODE_NEW as f64 / started.elapsed().as_secs_f64())
}

fn prompt_tokens(tokens: &[u32]) -> Vec<u32> {
    let length = tokens.len().min(DECODE_PROMPT);
    tokens[..length].to_vec()
}

fn argmax_tensor(logits: &Tensor) -> Result<u32> {
    let values = logits.flatten_all()?.to_vec1::<f32>()?;
    let index = values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .map(|(index, _)| index)
        .unwrap_or(0);
    Ok(index as u32)
}
