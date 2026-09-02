use crate::wikitext::candle_msg;
use candle_core::quantized::QMatMul;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear as DenseLinear, Module};
use half::f16;
use quantize::Quantized;
use std::cell::{Cell, RefCell};

pub enum Kind {
    Dense(DenseLinear),
    Packed(Quantized<f16>),
    CandlePacked(QMatMul),
}

pub struct Linear {
    pub rows: usize,
    pub columns: usize,
    pub kind: Kind,
    pub store_input: Cell<bool>,
    pub last_input: RefCell<Vec<f32>>,
}

impl Linear {
    pub fn dense(rows: usize, columns: usize, weight: Tensor) -> Self {
        Self {
            rows,
            columns,
            kind: Kind::Dense(DenseLinear::new(weight, None)),
            store_input: Cell::new(false),
            last_input: RefCell::new(Vec::new()),
        }
    }

    pub fn packed(rows: usize, columns: usize, weight: Quantized<f16>) -> Self {
        Self {
            rows,
            columns,
            kind: Kind::Packed(weight),
            store_input: Cell::new(false),
            last_input: RefCell::new(Vec::new()),
        }
    }

    pub fn candle_packed(rows: usize, columns: usize, weight: QMatMul) -> Self {
        Self {
            rows,
            columns,
            kind: Kind::CandlePacked(weight),
            store_input: Cell::new(false),
            last_input: RefCell::new(Vec::new()),
        }
    }

    pub fn packed_weight(&self) -> Option<&Quantized<f16>> {
        match &self.kind {
            Kind::Packed(weight) => Some(weight),
            Kind::Dense(_) | Kind::CandlePacked(_) => None,
        }
    }

    pub fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        if self.store_input.get() {
            let flat = hidden
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let width = flat.len().min(self.columns);
            self.last_input.replace(flat[..width].to_vec());
        }
        match &self.kind {
            Kind::Dense(inner) => inner.forward(hidden),
            Kind::Packed(weight) => packed_matmul(weight, hidden, self.rows, self.columns),
            Kind::CandlePacked(weight) => weight.forward(hidden),
        }
    }
}

fn packed_matmul(
    weight: &Quantized<f16>,
    hidden: &Tensor,
    rows: usize,
    columns: usize,
) -> Result<Tensor> {
    let dtype = hidden.dtype();
    let device = hidden.device().clone();
    let mut shape = hidden.dims().to_vec();
    let values = hidden
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let product = weight.matmul(&values, columns).map_err(candle_msg)?;
    if let Some(last) = shape.last_mut() {
        *last = rows;
    }
    Tensor::from_vec(product, shape, &device)?.to_dtype(dtype)
}
