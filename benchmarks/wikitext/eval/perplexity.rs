use crate::wikitext::net::cache::KvCache;
use crate::wikitext::net::LlamaNet;
use crate::wikitext::{CONTEXT, STRIDE};
use candle_core::{DType, Result};

pub fn rolling(model: &LlamaNet, tokens: &[u32]) -> Result<f64> {
    if tokens.len() < 2 {
        return Ok(f64::NAN);
    }
    let mut nll = 0.0;
    let mut counted = 0usize;
    let mut start = 0;
    let mut first_window = true;
    let windows = if tokens.len() <= CONTEXT {
        1
    } else {
        1 + (tokens.len() - CONTEXT).div_ceil(STRIDE)
    };
    let mut window_index = 0usize;
    while start < tokens.len() {
        let end = (start + CONTEXT).min(tokens.len());
        if end - start < 2 {
            break;
        }
        let window = &tokens[start..end];
        let mut cache = KvCache::new(false, DType::F32, &model.config, model.device())?;
        let logits = model.forward(window, 0, &mut cache, true)?;
        // [1, seq, vocab] — keep the batch axis in LlamaNet::forward.
        let rows = logits.squeeze(0)?.to_vec2::<f32>()?;
        let score_from = if first_window {
            0
        } else {
            window.len().saturating_sub(STRIDE)
        };
        for (index, row) in rows.iter().enumerate() {
            if index < score_from || index + 1 >= window.len() {
                continue;
            }
            nll += log_softmax_nll(row, window[index + 1] as usize);
            counted += 1;
        }
        first_window = false;
        window_index += 1;
        if window_index == 1 || window_index.is_multiple_of(10) || window_index == windows {
            eprintln!("  ppl window {window_index}/{windows} ({counted} tokens scored)");
        }
        if end == tokens.len() {
            break;
        }
        start += STRIDE;
    }
    if counted == 0 {
        Ok(f64::NAN)
    } else {
        Ok((nll / counted as f64).exp())
    }
}

fn log_softmax_nll(logits: &[f32], target: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut sum = 0.0;
    for &logit in logits {
        sum += (logit as f64 - max).exp();
    }
    let log_z = max + sum.ln();
    log_z - logits[target] as f64
}
