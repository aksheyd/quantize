//! WikiText-2: fp32 vs packed Q4_32 vs packed Candle Q4_0.
//!
//! `just wikitext`

#[path = "wikitext/mod.rs"]
mod wikitext;

use candle_core::Device;
use std::error::Error;
use wikitext::data::{corpus, load, weights};
use wikitext::eval::{args, decode, layers, perplexity, report};
use wikitext::net::LlamaNet;
use wikitext::{CONTEXT, DECODE_NEW, DECODE_PROMPT, STRIDE};

fn main() -> Result<(), Box<dyn Error>> {
    let max_tokens = args::max_tokens();
    let device = Device::Cpu;

    println!("loading HuggingFaceTB/SmolLM-135M and WikiText-2 test");
    let loaded = load::load(&device)?;
    let mut tokens = corpus::tokenize(&loaded.tokenizer)?;
    if let Some(limit) = max_tokens {
        tokens.truncate(limit);
        println!("truncated corpus to {limit} tokens (--max-tokens)");
    }
    println!(
        "setup: WikiText-2 raw test, {n} tokens, context {CONTEXT}, stride {STRIDE}, teacher-forced\n\
         decode: greedy argmax stopwatch, {DECODE_PROMPT}-token prompt, {DECODE_NEW} new tokens, KV cache on\n\
         same LlamaNet; only the Linear multiply changes\n\
         quantize Q4_32 = Quantized::matmul; candle Q4_0 = QMatMul (packed)",
        n = tokens.len()
    );

    let extracted = weights::extract_linears(&loaded.vb, &loaded.config)?;
    let packed = LlamaNet::packed(&loaded, &extracted)?;
    layers::write_dump(&packed, &extracted, tokens.first().copied().unwrap_or(0))?;

    let fp32 = LlamaNet::dense(&loaded, &extracted)?;
    let candle = LlamaNet::candle_q4(&loaded, &extracted)?;

    let ppl_fp32 = perplexity::rolling(&fp32, &tokens)?;
    let tps_fp32 = decode::tokens(&fp32, &tokens)?;
    drop(fp32);

    let ppl_q = perplexity::rolling(&packed, &tokens)?;
    let tps_q = decode::tokens(&packed, &tokens)?;
    drop(packed);

    let ppl_c = perplexity::rolling(&candle, &tokens)?;
    let tps_c = decode::tokens(&candle, &tokens)?;

    report::print(&[
        report::Row {
            method: "fp32",
            perplexity: ppl_fp32,
            tokens_per_second: tps_fp32,
        },
        report::Row {
            method: "quantize Q4_32",
            perplexity: ppl_q,
            tokens_per_second: tps_q,
        },
        report::Row {
            method: "candle Q4_0",
            perplexity: ppl_c,
            tokens_per_second: tps_c,
        },
    ]);
    Ok(())
}
