use std::collections::BTreeMap;
use std::fmt::Debug;

use half::{bf16, f16};
use quantize::{Scale, Scheme};
use serde_json::{Value, json};

use super::{file, float, temporary_path, values};
use crate::Tensor;
use crate::safetensors::read::from_bytes;
use crate::safetensors::write::to_bytes;
use crate::safetensors::{read, write};

#[test]
fn floats_of_every_shape_round_trip_through_a_file() {
    // A matrix, a vector, a single value, an empty tensor, and three dimensions.
    let shapes: [&[usize]; 5] = [&[3, 4], &[7], &[], &[2, 0], &[2, 3, 2]];
    let tensors: BTreeMap<String, Tensor<f32>> = shapes
        .iter()
        .map(|shape| (format!("{shape:?}"), float(shape)))
        .collect();
    let path = temporary_path("floats");
    write(&path, &tensors).unwrap();
    assert_eq!(read(&path).unwrap(), tensors);
    std::fs::remove_file(path).unwrap();
}

fn assert_quantized_tensors_round_trip<S: Scale + PartialEq + Debug>() {
    let schemes = [
        Scheme::Q4_32,
        Scheme::Q8_32,
        Scheme::Asymmetric { bits: 3, block: 7 },
        Scheme::Adaptive {
            block: 32,
            tolerance: 0.01,
        },
    ];
    let mut tensors = BTreeMap::from([("norm".to_string(), float(&[24]))]);
    for scheme in schemes {
        let mut matrix = scheme.quantize::<S>(&values(96)).unwrap();
        matrix.set_shape(4, 24).unwrap();
        let vector = scheme.quantize::<S>(&values(50)).unwrap();
        tensors.insert(format!("{scheme} matrix"), Tensor::Quantized(matrix));
        tensors.insert(format!("{scheme} vector"), Tensor::Quantized(vector));
    }
    let loaded = from_bytes::<S>(&to_bytes(&tensors).unwrap()).unwrap();
    assert_eq!(loaded, tensors);
    for (name, tensor) in &tensors {
        if let (Tensor::Quantized(saved), Tensor::Quantized(loaded)) = (tensor, &loaded[name]) {
            assert_eq!(loaded.shape(), saved.shape(), "{name}");
            assert_eq!(loaded.dequantize(), saved.dequantize(), "{name}");
        }
    }
}

#[test]
fn quantized_tensors_of_every_scheme_and_scale_type_round_trip() {
    assert_quantized_tensors_round_trip::<f32>();
    assert_quantized_tensors_round_trip::<f16>();
    assert_quantized_tensors_round_trip::<bf16>();
}

#[test]
fn f16_and_bf16_tensors_read_as_the_f32_values_they_hold() {
    // 1, -2, about a third, the largest value, the smallest above zero, and
    // negative zero.
    let f16_bits: [u16; 6] = [0x3C00, 0xC000, 0x3555, 0x7BFF, 0x0001, 0x8000];
    let f16_values = [1.0, -2.0, 0.333_251_95, 65504.0, 2.0_f32.powi(-24), -0.0];
    let bf16_bits: [u16; 6] = [0x3F80, 0xC000, 0x3EAB, 0x7F7F, 0x0001, 0x8000];
    // A bf16 is the first 16 bits of an f32.
    let bf16_values = bf16_bits.map(|bits| f32::from_bits(u32::from(bits) << 16));

    let data: Vec<u8> = f16_bits
        .iter()
        .chain(&bf16_bits)
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    let header = r#"{"f16": {"dtype": "F16", "shape": [2, 3], "data_offsets": [0, 12]},
                     "bf16": {"dtype": "BF16", "shape": [6], "data_offsets": [12, 24]}}"#;
    let tensors = from_bytes::<f32>(&file(header, &data)).unwrap();
    let bits = |name: &str| match &tensors[name] {
        Tensor::Float { values, .. } => values.iter().map(|value| value.to_bits()).collect(),
        Tensor::Quantized(_) => Vec::new(),
    };
    assert_eq!(bits("f16"), f16_values.map(f32::to_bits));
    assert_eq!(bits("bf16"), bf16_values.map(f32::to_bits));
    assert!(matches!(&tensors["f16"], Tensor::Float { shape, .. } if *shape == [2, 3]));
}

#[test]
fn metadata_and_the_spaces_after_the_header_are_skipped() {
    let header = r#"{"__metadata__": {"format": "pt"},
                     "one": {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}}    "#;
    let tensors = from_bytes::<f32>(&file(header, &1.5_f32.to_le_bytes())).unwrap();
    let one = Tensor::Float {
        shape: vec![],
        values: vec![1.5],
    };
    assert_eq!(tensors, BTreeMap::from([("one".to_string(), one)]));
}

#[test]
fn floats_come_first_after_a_header_padded_to_8_bytes() {
    let quantized = Scheme::Q4_32.quantize::<f32>(&values(33)).unwrap();
    let tensors = BTreeMap::from([
        ("a quantized".to_string(), Tensor::Quantized(quantized)),
        ("b float".to_string(), float(&[5])),
    ]);
    let bytes = to_bytes(&tensors).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    assert!(header_length.is_multiple_of(8));
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_length]).unwrap();
    assert_eq!(header["b float"]["data_offsets"], json!([0, 20]));
    assert_eq!(header["a quantized"]["dtype"], "U8");
}
