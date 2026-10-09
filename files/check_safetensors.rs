//! The Rust half of `just files-check`, which checks quantize-files against
//! the safetensors Python package both ways, with `check_safetensors.py`:
//!
//! 1. `write DIR` saves `DIR/from_rust.safetensors`, which the Python half
//!    reads with `safetensors.numpy.load_file` and checks.
//! 2. The Python half saves `DIR/from_python.safetensors` with `save_file`.
//! 3. `read DIR` reads that file back and checks every value.
//!
//! Both halves make the same values, `(i - 60) / 16` for value `i`, which f32
//! and f16 both hold exactly.

use std::collections::BTreeMap;
use std::path::Path;
use std::{env, fs};

use quantize::{Quantized, Scheme, f16};
use quantize_files::{Tensor, safetensors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    let [step, directory] = &arguments[..] else {
        return Err("usage: check_safetensors write|read DIR".into());
    };
    let directory = Path::new(directory);
    match step.as_str() {
        "write" => {
            let tensors = BTreeMap::from([
                ("matrix".to_string(), float(&[6, 20])),
                ("vector".to_string(), float(&[20])),
                ("quantized".to_string(), Tensor::Quantized(quantized())),
            ]);
            fs::create_dir_all(directory)?;
            safetensors::write(directory.join("from_rust.safetensors"), &tensors)?;
            println!("quantize-files wrote from_rust.safetensors");
        }
        "read" => {
            let tensors = safetensors::read::<f16>(directory.join("from_python.safetensors"))?;
            // The Python half saved its values as F32 and F16, and copied the
            // quantized tensor from from_rust.safetensors.
            let expected = BTreeMap::from([
                ("f32_matrix".to_string(), float(&[6, 20])),
                ("f16_matrix".to_string(), float(&[6, 20])),
                ("f16_vector".to_string(), float(&[20])),
                ("quantized".to_string(), Tensor::Quantized(quantized())),
            ]);
            assert_eq!(tensors, expected);
            let Tensor::Quantized(loaded) = &tensors["quantized"] else {
                unreachable!("the maps are equal");
            };
            assert_eq!(loaded.dequantize(), quantized().dequantize());
            println!(
                "quantize-files read from_python.safetensors: {} tensors with the values the safetensors package wrote",
                tensors.len()
            );
        }
        _ => return Err(format!("unknown step {step:?}; use write or read").into()),
    }
    Ok(())
}

/// Value `i` is `(i - 60) / 16`, as `check_safetensors.py` makes them.
fn values(count: usize) -> Vec<f32> {
    (0..count).map(|i| (i as f32 - 60.0) / 16.0).collect()
}

fn float(shape: &[usize]) -> Tensor<f16> {
    let values = values(shape.iter().product());
    let shape = shape.to_vec();
    Tensor::Float { shape, values }
}

/// A 6 × 20 matrix of 4-bit codes with f16 scales.
fn quantized() -> Quantized<f16> {
    let mut quantized = Scheme::Q4_32.quantize(&values(120)).unwrap();
    quantized.set_shape(6, 20).unwrap();
    quantized
}
