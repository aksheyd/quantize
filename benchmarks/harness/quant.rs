//! Per-method evaluators: round-trip A and B through the method, matmul,
//! return the result vector. Each `eval_*` matches the `EvalFn` signature
//! used in `methods.rs`.

use super::Sample;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, Result, Tensor,
};
use quantize::{choose_scale_bits, dequantize_bits, quantize_bits};

fn roundtrip_bits<const BITS: u32>(values: &[f32]) -> Vec<f32> {
    let scale = choose_scale_bits::<BITS>(values);
    values
        .iter()
        .map(|&x| dequantize_bits(quantize_bits::<BITS>(x, scale), scale))
        .collect()
}

fn matmul(a: Vec<f32>, b: Vec<f32>, n: usize, d: &Device) -> Result<Vec<f32>> {
    let a = Tensor::from_vec(a, (n, n), d)?;
    let b = Tensor::from_vec(b, (n, n), d)?;
    a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()
}

pub(super) fn eval_yours<const BITS: u32>(s: &Sample, d: &Device) -> Result<Vec<f32>> {
    let a = roundtrip_bits::<BITS>(&s.matrix_a);
    let b = roundtrip_bits::<BITS>(&s.matrix_b);
    matmul(a, b, s.matrix_size, d)
}

fn eval_candle(s: &Sample, d: &Device, dtype: GgmlDType) -> Result<Vec<f32>> {
    let a = QTensor::quantize(&s.tensor_a, dtype)?.dequantize(d)?;
    let b = QTensor::quantize(&s.tensor_b, dtype)?.dequantize(d)?;
    a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()
}

pub(super) fn eval_q4_0(s: &Sample, d: &Device) -> Result<Vec<f32>> {
    eval_candle(s, d, GgmlDType::Q4_0)
}
pub(super) fn eval_q5_0(s: &Sample, d: &Device) -> Result<Vec<f32>> {
    eval_candle(s, d, GgmlDType::Q5_0)
}
pub(super) fn eval_q8_0(s: &Sample, d: &Device) -> Result<Vec<f32>> {
    eval_candle(s, d, GgmlDType::Q8_0)
}
