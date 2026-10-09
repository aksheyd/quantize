//! The Rust half of `just files-check` for gguf, which checks quantize-files
//! against the gguf Python package both ways, with `check_gguf.py`:
//!
//! 1. `write DIR` saves `DIR/from_rust.gguf`, with float tensors and metadata
//!    of every type, which the Python half reads with `gguf.GGUFReader` and
//!    checks. It also saves `DIR/quantized_from_rust.gguf`, with a `Q4_32`
//!    and a `Q8_32` matrix, which the Python half decodes with
//!    `gguf.quants.dequantize`.
//! 2. The Python half saves `DIR/from_python.gguf` with `gguf.GGUFWriter`,
//!    and `DIR/quantized_from_python.gguf` with blocks that
//!    `gguf.quants.quantize` made.
//! 3. `read DIR` reads both files back and checks every value.
//!
//! Both halves make the same float values, `(i - 60) / 16` for value `i`,
//! which f32 and f16 both hold exactly. Next to each quantized tensor `NAME`,
//! each half saves `NAME.decoded`: the values it decodes to, which the other
//! half must decode it to as well, bit for bit.

use std::collections::BTreeMap;
use std::path::Path;
use std::{env, fs};

use quantize::{Scheme, f16};
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
            // The quantized matrices go in a file of their own.
            let metadata = metadata_of(&[("general.architecture", text("quantize-check"))]);
            let path = directory.join("quantized_from_rust.gguf");
            gguf::write(path, &metadata, &quantized()?)?;
            println!("quantize-files wrote from_rust.gguf and quantized_from_rust.gguf");
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

            let (_, tensors) = gguf::read(directory.join("quantized_from_python.gguf"))?;
            for name in ["q4_0", "q8_0"] {
                let decoded = &tensors[&format!("{name}.decoded")];
                let (Tensor::Quantized(matrix), Tensor::Float { shape, values }) =
                    (&tensors[name], decoded)
                else {
                    return Err(format!("{name} isn't quantized, next to floats").into());
                };
                let same_bits = bits(&matrix.dequantize()) == bits(values);
                assert!(
                    same_bits,
                    "{name} decodes to other values than the gguf package's"
                );
                assert_eq!(matrix.shape(), Some((shape[0], shape[1])), "{name}");
            }
            println!(
                "quantize-files read quantized_from_python.gguf: q4_0 and q8_0 decode to the gguf package's values, bit for bit"
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

/// A `Q4_32` and a `Q8_32` matrix of 64 rows of 256 [`weights`], as `q4_0`
/// and `q8_0`, each next to `NAME.decoded`.
fn quantized() -> Result<BTreeMap<String, Tensor<f16>>, quantize::Error> {
    let (rows, columns) = (64, 256);
    let mut tensors = BTreeMap::new();
    for (name, scheme) in [("q4_0", Scheme::Q4_32), ("q8_0", Scheme::Q8_32)] {
        let mut matrix = scheme.quantize::<f16>(&weights(rows, columns))?;
        matrix.set_shape(rows, columns)?;
        let shape = vec![rows, columns];
        let decoded = Tensor::Float {
            shape,
            values: matrix.dequantize(),
        };
        tensors.insert(format!("{name}.decoded"), decoded);
        tensors.insert(name.to_string(), Tensor::Quantized(matrix));
    }
    Ok(tensors)
}

/// Weights spread like a layer's, with standard deviation 0.02, and every
/// 97th 20 times larger, as outliers are. Row 0 is all zeros, and row 1 so
/// small that its f16 scales are subnormal.
fn weights(rows: usize, columns: usize) -> Vec<f32> {
    let mut weights = normal(rows * columns, 1, 0.02);
    for weight in weights.iter_mut().step_by(97) {
        *weight *= 20.0;
    }
    weights[..columns].fill(0.0);
    for weight in &mut weights[columns..2 * columns] {
        *weight *= 5e-5;
    }
    weights
}

/// `count` values from a normal distribution with standard deviation
/// `deviation`, from `seed`, so that every run makes the same ones.
fn normal(count: usize, seed: u64, deviation: f32) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 + 0.5) / (1 << 24) as f32
    };
    let mut normal = move || {
        let radius = (-2.0 * uniform().ln()).sqrt();
        radius * (std::f32::consts::TAU * uniform()).cos()
    };
    (0..count).map(|_| deviation * normal()).collect()
}

/// Each value's bit pattern, so that -0 differs from 0.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}
