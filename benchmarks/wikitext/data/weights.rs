use candle_core::Result;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::Config;

pub struct Extracted {
    pub name: String,
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<f32>,
}

const PROJECTIONS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

pub fn extract_linears(vb: &VarBuilder, config: &Config) -> Result<Vec<Extracted>> {
    let mut out = Vec::new();
    let head_dim = config.hidden_size / config.num_attention_heads;
    let query_rows = head_dim * config.num_attention_heads;
    let key_rows = head_dim * config.num_key_value_heads;
    for layer in 0..config.num_hidden_layers {
        for projection in PROJECTIONS {
            let (rows, columns) = match projection {
                "self_attn.q_proj" | "self_attn.o_proj" => (query_rows, config.hidden_size),
                "self_attn.k_proj" | "self_attn.v_proj" => (key_rows, config.hidden_size),
                "mlp.gate_proj" | "mlp.up_proj" => (config.intermediate_size, config.hidden_size),
                "mlp.down_proj" => (config.hidden_size, config.intermediate_size),
                _ => unreachable!(),
            };
            let name = format!("model.layers.{layer}.{projection}");
            out.push(load_matrix(vb, &name, rows, columns)?);
        }
    }
    out.push(load_matrix(
        vb,
        "lm_head",
        config.vocab_size,
        config.hidden_size,
    )?);
    Ok(out)
}

fn load_matrix(vb: &VarBuilder, name: &str, rows: usize, columns: usize) -> Result<Extracted> {
    let path = if name == "lm_head" && !vb.contains_tensor("lm_head.weight") {
        "model.embed_tokens"
    } else {
        name
    };
    let tensor = vb
        .pp(path)
        .get((rows, columns), "weight")?
        .to_dtype(candle_core::DType::F32)?;
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    Ok(Extracted {
        name: name.to_string(),
        rows,
        columns,
        values,
    })
}
