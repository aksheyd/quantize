//! The Rust half of `just files-check` for gguf, which checks quantize-files
//! against the gguf Python package both ways, with `check_gguf.py`:
//!
//! 1. `write DIR` saves `DIR/from_rust.gguf`, with float tensors and metadata
//!    of every type, which the Python half reads with `gguf.GGUFReader` and
//!    checks.
//! 2. The Python half saves `DIR/from_python.gguf` with `gguf.GGUFWriter`.
//! 3. `read DIR` reads that file back and checks every value.
//!
//! Both halves make the same values, `(i - 60) / 16` for value `i`, which f32
//! and f16 both hold exactly.

use std::collections::BTreeMap;
use std::path::Path;
use std::{env, fs};

use quantize::f16;
use quantize_files::Tensor;
use quantize_files::gguf::{self, Metadata, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let [step, directory] = &arguments[..] else {
        return Err("usage: check_gguf write|read DIR".into());
    };
    let directory = Path::new(directory);
    match step.as_str() {
        "write" => {
            let metadata = metadata_of(&[
                ("general.architecture", text("quantize-check")),
                ("check.u8", Value::U8(u8::MAX)),
                ("check.i8", Value::I8(i8::MIN)),
                ("check.u16", Value::U16(u16::MAX)),
                ("check.i16", Value::I16(i16::MIN)),
                ("check.u32", Value::U32(u32::MAX)),
                ("check.i32", Value::I32(i32::MIN)),
                ("check.u64", Value::U64(u64::MAX)),
                ("check.i64", Value::I64(i64::MIN)),
                ("check.f32", Value::F32(-1.5)),
                ("check.f64", Value::F64(0.1)),
                ("check.bool", Value::Bool(true)),
                ("check.strings", strings(&["a", "", "naïve"])),
                (
                    "check.floats",
                    Value::Array(vec![Value::F32(0.5), Value::F32(-2.0)]),
                ),
            ]);
            let tensors = BTreeMap::from([
                ("matrix".to_string(), float(&[6, 20])),
                ("vector".to_string(), float(&[20])),
                ("cube".to_string(), float(&[2, 3, 4])),
            ]);
            fs::create_dir_all(directory)?;
            gguf::write(directory.join("from_rust.gguf"), &metadata, &tensors)?;
            println!("quantize-files wrote from_rust.gguf");
        }
        "read" => {
            let (metadata, tensors) = gguf::read(directory.join("from_python.gguf"))?;
            // The Python half saved its values as F32 and F16, aligned to 64
            // bytes.
            let expected_metadata = metadata_of(&[
                ("general.architecture", text("quantize-check")),
                ("general.alignment", Value::U32(64)),
                ("check.u8s", Value::Array(vec![Value::U8(1), Value::U8(2)])),
                ("check.i64", Value::I64(-7)),
                ("check.f64", Value::F64(0.1)),
                ("check.bool", Value::Bool(false)),
                ("check.strings", strings(&["x", "yz"])),
                (
                    "check.nested",
                    Value::Array(vec![
                        Value::Array(vec![Value::I32(1), Value::I32(2)]),
                        Value::Array(vec![Value::I32(3)]),
                    ]),
                ),
            ]);
            let expected_tensors = BTreeMap::from([
                ("f32_matrix".to_string(), float(&[6, 20])),
                ("f16_matrix".to_string(), float(&[6, 20])),
                ("f16_cube".to_string(), float(&[2, 3, 4])),
            ]);
            assert_eq!(metadata, expected_metadata);
            assert_eq!(tensors, expected_tensors);
            println!(
                "quantize-files read from_python.gguf: {} metadata entries and {} tensors with the values the gguf package wrote",
                metadata.len(),
                tensors.len()
            );
        }
        _ => return Err(format!("unknown step {step:?}; use write or read").into()),
    }
    Ok(())
}

fn metadata_of(entries: &[(&str, Value)]) -> Metadata {
    let entries = entries.iter().cloned();
    entries
        .map(|(key, value)| (key.to_string(), value))
        .collect()
}

fn text(text: &str) -> Value {
    Value::String(text.to_string())
}

fn strings(texts: &[&str]) -> Value {
    Value::Array(texts.iter().map(|word| text(word)).collect())
}

/// Value `i` is `(i - 60) / 16`, as `check_gguf.py` makes them.
fn float(shape: &[usize]) -> Tensor<f16> {
    let count = shape.iter().product();
    let values = (0..count).map(|i| (i as f32 - 60.0) / 16.0).collect();
    let shape = shape.to_vec();
    Tensor::Float { shape, values }
}
