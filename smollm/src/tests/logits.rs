//! Checks that the quantized model's logits stay close to the f32 model's.

use quantize::Scheme;

use super::{encode, tiny_llama};
use crate::checkpoint::quantize_matrices;
use crate::model::Model;

/// The model's logits at every position of `tokens`.
fn logits(model: &Model, tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut caches = model.new_caches();
    tokens
        .iter()
        .map(|&token| model.forward(token, &mut caches))
        .collect()
}

#[test]
fn quantized_logits_stay_close_to_f32s() {
    let tiny = tiny_llama("logits");
    let text = "the city was one of the first in the world and it has been a school since";
    let tokens = encode(&tiny.tokenizer, text);
    let float_logits = logits(
        &Model::new(&tiny.config, tiny.tensors.clone()).unwrap(),
        &tokens,
    );

    // The f32 logits reach about 2.4. Q8_32's stay within 0.014 of them, and
    // Q4_32's within 0.29. A transposed matrix moves them by 0.96 or more,
    // and q's and k's rows in llama.cpp's order by 0.05 to 0.3, so Q8_32's
    // tolerance catches both. A wrong q or k shows only from the second token
    // on: a lone token's attention returns its own value, whatever the scores.
    for (scheme, tolerance) in [(Scheme::Q8_32, 0.05), (Scheme::Q4_32, 0.75)] {
        let quantized_tensors = quantize_matrices(&tiny.tensors, scheme).unwrap();
        let model = Model::new(&tiny.config, quantized_tensors).unwrap();
        let positions = float_logits.iter().zip(logits(&model, &tokens));
        for (position, (float, quantized)) in positions.enumerate() {
            let differences = float.iter().zip(&quantized).map(|(a, b)| (a - b).abs());
            let largest = differences.fold(0.0, f32::max);
            assert!(
                largest < tolerance,
                "{scheme} at position {position}: {largest}"
            );
        }
    }
}
