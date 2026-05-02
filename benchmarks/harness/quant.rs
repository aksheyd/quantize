//! Per-method evaluators: round-trip A and B through the method, matmul,
//! return the result vector. Each `eval_*` matches the `EvalFn` signature
//! used in `methods.rs`.

use super::Sample;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, Result, Tensor,
};
use half::f16;
use quantize::{dequantize, quantize};

fn roundtrip_block<const BITS: u32>(values: &[f32]) -> Vec<f32> {
    let (scales, codes) = quantize::<f16, BITS, 32>(values);
    dequantize::<_, 32>(&scales, &codes)
}

fn matmul(a: Vec<f32>, b: Vec<f32>, n: usize, d: &Device) -> Result<Vec<f32>> {
    let a = Tensor::from_vec(a, (n, n), d)?;
    let b = Tensor::from_vec(b, (n, n), d)?;
    a.matmul(&b)?.flatten_all()?.to_vec1::<f32>()
}

pub(super) fn eval_quantize<const BITS: u32>(s: &Sample, d: &Device) -> Result<Vec<f32>> {
    let a = roundtrip_block::<BITS>(&s.matrix_a);
    let b = roundtrip_block::<BITS>(&s.matrix_b);
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
