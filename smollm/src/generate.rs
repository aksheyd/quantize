//! Greedy generation: feed the prompt, then keep picking the token the model
//! scores highest and feeding it back in.

use std::time::{Duration, Instant};

use crate::model::Model;

/// Feed `prompt` to `model`, then pick `count` new tokens greedily, handing
/// each one to `on_token` as soon as it's picked. Returns how long picking
/// them took, not counting the prompt.
pub fn generate(
    model: &Model,
    prompt: &[u32],
    count: usize,
    mut on_token: impl FnMut(u32),
) -> Result<Duration, String> {
    let Some((&last, earlier)) = prompt.split_last() else {
        return Err("the prompt has no tokens".into());
    };
    let mut caches = model.new_caches();
    for &token in earlier {
        model.forward(token, &mut caches);
    }

    // Each step feeds one token and picks the next, starting from the
    // prompt's last token, so `count` steps pick `count` tokens.
    let started = Instant::now();
    let mut token = last;
    for _ in 0..count {
        token = highest(&model.forward(token, &mut caches));
        on_token(token);
    }
    Ok(started.elapsed())
}

/// The token with the highest score.
fn highest(logits: &[f32]) -> u32 {
    let scored = logits.iter().enumerate();
    let best = scored.max_by(|(_, left), (_, right)| left.total_cmp(right));
    best.map_or(0, |(token, _)| token as u32)
}
