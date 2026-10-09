use std::collections::BTreeMap;

use half::f16;

use super::{entry, f32_bytes, file, float, string, temporary_path, tensor_info};
use crate::Tensor;
use crate::gguf::read::from_bytes;
use crate::gguf::write::to_bytes;
use crate::gguf::{Metadata, Value, read, write};

/// One entry of each type, at the edges of its range where it has one.
fn every_type() -> Metadata {
    let words = ["hello", "", "naïve", "日本"].map(|word| Value::String(word.to_string()));
    let values = [
        ("u8", Value::U8(u8::MAX)),
        ("i8", Value::I8(i8::MIN)),
        ("u16", Value::U16(u16::MAX)),
        ("i16", Value::I16(i16::MIN)),
        ("u32", Value::U32(u32::MAX)),
        ("i32", Value::I32(i32::MIN)),
        ("u64", Value::U64(u64::MAX)),
        ("i64", Value::I64(i64::MIN)),
        ("f32", Value::F32(-f32::MIN_POSITIVE)),
        ("f64", Value::F64(f64::MAX)),
        ("bool", Value::Bool(true)),
        ("string", Value::String("llama".to_string())),
        ("strings", Value::Array(words.to_vec())),
        (
            "bools",
            Value::Array(vec![Value::Bool(false), Value::Bool(true)]),
        ),
        (
            "floats",
            Value::Array(vec![Value::F32(0.5), Value::F32(-2.0)]),
        ),
        (
            "nested",
            Value::Array(vec![
                Value::Array(vec![Value::I32(1), Value::I32(2)]),
                Value::Array(vec![Value::String("three".to_string())]),
            ]),
        ),
    ];
    values
        .into_iter()
        .map(|(key, value)| (format!("check.{key}"), value))
        .collect()
}

#[test]
fn every_metadata_type_and_float_shape_round_trips_through_a_file() {
    // A matrix, a vector, a single value, an empty tensor, and three dimensions.
    let shapes: [&[usize]; 5] = [&[3, 4], &[7], &[], &[2, 0], &[2, 3, 4]];
    let tensors: BTreeMap<String, Tensor<f16>> = shapes
        .iter()
        .map(|shape| (format!("{shape:?}"), float(shape)))
        .collect();
    let metadata = every_type();
    let path = temporary_path("round-trip");
    write(&path, &metadata, &tensors).unwrap();
    assert_eq!(read(&path).unwrap(), (metadata, tensors));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn the_writer_lays_out_a_file_as_the_module_docs_show() {
    let metadata = Metadata::from([(
        "general.architecture".to_string(),
        Value::String("llama".to_string()),
    )]);
    let matrix = Tensor::Float {
        shape: vec![2, 3],
        values: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    };
    let tensors = BTreeMap::from([("w".to_string(), matrix)]);
    let mut data = f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    data.resize(32, 0);
    let expected = file(
        &[entry("general.architecture", 8, &string("llama"))],
        // 3 columns, then 2 rows, of ggml type 0, F32.
        &[tensor_info("w", &[3, 2], 0, 0)],
        &data,
    );
    let bytes = to_bytes(&metadata, &tensors).unwrap();
    assert_eq!(bytes, expected);
    assert_eq!(bytes.len(), 160);
    assert_eq!(bytes[128..152], f32_bytes(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
}

#[test]
fn dimensions_read_innermost_first() {
    let counting: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let bytes = file(
        &[],
        &[tensor_info("cube", &[4, 3, 2], 0, 0)],
        &f32_bytes(&counting),
    );
    let (_, tensors) = from_bytes(&bytes).unwrap();
    let cube = Tensor::Float {
        shape: vec![2, 3, 4],
        values: counting,
    };
    assert_eq!(tensors["cube"], cube);
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

    let mut data: Vec<u8> = f16_bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect();
    data.resize(32, 0);
    data.extend(bf16_bits.iter().flat_map(|bits| bits.to_le_bytes()));
    let infos = [
        tensor_info("f16", &[3, 2], 1, 0),
        tensor_info("bf16", &[6], 30, 32),
    ];
    let (_, tensors) = from_bytes(&file(&[], &infos, &data)).unwrap();
    let bits = |name: &str| match &tensors[name] {
        Tensor::Float { values, .. } => values.iter().map(|value| value.to_bits()).collect(),
        Tensor::Quantized(_) => Vec::new(),
    };
    assert_eq!(bits("f16"), f16_values.map(f32::to_bits));
    assert_eq!(bits("bf16"), bf16_values.map(f32::to_bits));
    assert!(matches!(&tensors["f16"], Tensor::Float { shape, .. } if *shape == [2, 3]));
}

#[test]
fn general_alignment_sets_where_each_tensor_starts() {
    let metadata = Metadata::from([("general.alignment".to_string(), Value::U32(64))]);
    let three = float(&[3]);
    let last = Tensor::Float {
        shape: vec![1],
        values: vec![7.5],
    };
    let tensors = BTreeMap::from([("a".to_string(), three), ("b".to_string(), last)]);
    let bytes = to_bytes(&metadata, &tensors).unwrap();
    // "a" takes 12 bytes, padded to 64, so "b" starts 64 bytes into the data,
    // and the file ends after its 4 bytes, padded to 64.
    assert!(bytes.len().is_multiple_of(64));
    assert_eq!(bytes[bytes.len() - 64..][..4], 7.5_f32.to_le_bytes());
    assert_eq!(from_bytes(&bytes).unwrap(), (metadata, tensors));
}
