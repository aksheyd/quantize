use crate::wikitext::{MODEL_ID, candle_msg};
use candle_core::{DType, Device, Result};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config, LlamaConfig};
use hf_hub::api::sync::ApiBuilder;
use std::fs;
use tokenizers::Tokenizer;

pub struct Loaded {
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub vb: VarBuilder<'static>,
    pub device: Device,
}

pub fn load(device: &Device) -> Result<Loaded> {
    // Unlike Api::new(), from_env() reads HF_HOME (where the cache lives) and
    // HF_ENDPOINT (which server to download from).
    let api = ApiBuilder::from_env().build().map_err(candle_msg)?;
    let repo = api.model(MODEL_ID.to_string());
    let config_path = repo.get("config.json").map_err(candle_msg)?;
    let tokenizer_path = repo.get("tokenizer.json").map_err(candle_msg)?;
    let weights_path = repo.get("model.safetensors").map_err(candle_msg)?;

    let llama_config: LlamaConfig =
        serde_json::from_slice(&fs::read(config_path)?).map_err(candle_msg)?;
    let config = llama_config.into_config(false);
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(candle_msg)?;
    // SAFETY: this process only reads the mmaped safetensors through `vb`.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };

    Ok(Loaded {
        config,
        tokenizer,
        vb,
        device: device.clone(),
    })
}
