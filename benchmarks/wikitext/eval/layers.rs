use crate::wikitext::data::weights::Extracted;
use crate::wikitext::net::cache::KvCache;
use crate::wikitext::net::LlamaNet;
use candle_core::{DType, Result};
use serde::Serialize;
use std::fs::File;

#[derive(Serialize)]
struct LayerRow {
    name: String,
    rows: usize,
    columns: usize,
    bits_per_value: f32,
    relative_output_error: f32,
    code_saturation: f32,
}

pub fn write_dump(model: &LlamaNet, extracted: &[Extracted], first_token: u32) -> Result<()> {
    model.set_store_input(true);
    let mut cache = KvCache::new(false, DType::F32, &model.config, model.device())?;
    let _ = model.forward(&[first_token], 0, &mut cache, true)?;
    model.set_store_input(false);

    let mut rows = Vec::new();
    for (linear, spec) in model.linears().into_iter().zip(extracted) {
        let Some(weight) = linear.packed_weight() else {
            continue;
        };
        let activation = linear.last_input.borrow();
        if activation.len() != spec.columns {
            continue;
        }
        let reference = dense_mul(&spec.values, spec.rows, spec.columns, &activation);
        let quantized = weight
            .matmul(&activation, spec.columns)
            .map_err(crate::wikitext::candle_msg)?;
        rows.push(LayerRow {
            name: spec.name.clone(),
            rows: spec.rows,
            columns: spec.columns,
            bits_per_value: weight.bits_per_element(),
            relative_output_error: relative_error(&reference, &quantized),
            code_saturation: code_saturation(weight.codes(), weight.len()),
        });
    }
    rows.sort_by(|left, right| {
        right
            .relative_output_error
            .total_cmp(&left.relative_output_error)
    });
    serde_json::to_writer_pretty(File::create("wikitext-layers.json")?, &rows)
        .map_err(crate::wikitext::candle_msg)?;
    println!("wrote wikitext-layers.json ({} linears)", rows.len());
    Ok(())
}

fn dense_mul(weight: &[f32], rows: usize, columns: usize, activation: &[f32]) -> Vec<f32> {
    weight
        .chunks_exact(columns)
        .take(rows)
        .map(|row| row.iter().zip(activation).map(|(w, x)| w * x).sum())
        .collect()
}

fn relative_error(reference: &[f32], quantized: &[f32]) -> f32 {
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for (left, right) in reference.iter().zip(quantized) {
        let delta = left - right;
        numerator += delta * delta;
        denominator += left * left;
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator.sqrt() / denominator.sqrt()
    }
}

fn code_saturation(bytes: &[u8], values: usize) -> f32 {
    if values == 0 {
        return 0.0;
    }
    // Symmetric 4-bit: nibble 0x8 is -8, 0x7 is +7.
    let mut saturated = 0usize;
    for index in 0..values {
        let byte = bytes[index / 2];
        let nibble = if index.is_multiple_of(2) {
            byte & 0x0F
        } else {
            byte >> 4
        };
        if nibble == 0x7 || nibble == 0x8 {
            saturated += 1;
        }
    }
    saturated as f32 / values as f32
}
