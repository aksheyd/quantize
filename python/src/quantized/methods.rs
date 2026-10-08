use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple, PyType};

use quantize::{Error, Quantized, Scale};

use super::inner::{PyQuantized, QuantizedInner, with_inner};
use super::parts::Parts;
use crate::error::from_quantize;
use crate::gil::detach_if_large;
use crate::input::{
    as_block_bits, as_bytes, as_f32_array, as_f32_matmul_values, as_f32_values, as_packed_codes,
    as_writable_f32_out, check_shape,
};
use crate::scale::PyScale;

const PICKLED_BY_0_2: &str = "this tensor was pickled by quantize-py 0.2, which 0.3 can't load. \
    quantize the original weights again, or rebuild the tensor with Quantized.from_parts \
    from what its getters return under 0.2";

fn f32_array<'py>(py: Python<'py>, values: Vec<f32>) -> Bound<'py, PyAny> {
    values.into_pyarray(py).into_any()
}

fn kind<S: Scale>(quantized: &Quantized<S>) -> &'static str {
    match quantized {
        Quantized::Symmetric { .. } => "symmetric",
        Quantized::Asymmetric { .. } => "asymmetric",
        Quantized::Adaptive { .. } => "adaptive",
    }
}

fn bits<S: Scale>(quantized: &Quantized<S>) -> Option<u32> {
    match quantized {
        Quantized::Symmetric { codes, .. } | Quantized::Asymmetric { codes, .. } => {
            Some(codes.bits())
        }
        Quantized::Adaptive { .. } => None,
    }
}

// Each method works on a snapshot of the tensor, so a refit that another
// thread stores meanwhile doesn't change what it reads.
#[pymethods]
impl PyQuantized {
    /// Decode the values into an array of the tensor's `shape`. `out`, if
    /// given, must be a float32 NumPy array or CPU PyTorch tensor of that
    /// shape, and is returned.
    #[pyo3(signature = (out = None))]
    fn dequantize<'py>(
        &self,
        py: Python<'py>,
        out: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.snapshot();
        let shape = inner.shape();
        match out {
            Some(out) => {
                let mut writable = as_writable_f32_out(&out, &shape, &[])?;
                let output = writable.as_slice_mut()?;
                detach_if_large(py, inner.len(), || {
                    with_inner!(&inner, |quantized| quantized.dequantize_into(output))
                })
                .map_err(from_quantize)?;
                Ok(out)
            }
            None => {
                let values = detach_if_large(py, inner.len(), || {
                    with_inner!(&inner, |quantized| quantized.dequantize())
                });
                Ok(values.into_pyarray(py).reshape(shape)?.into_any())
            }
        }
    }

    /// The dot product of the decoded values with `values`, an array of the
    /// tensor's `shape`. For a matrix times a vector, use `matmul`.
    ///
    /// Each code is read straight from the packed bytes as it's multiplied,
    /// so the decoded values are never stored, whatever the scheme.
    fn dot(&self, py: Python<'_>, values: Bound<'_, PyAny>) -> PyResult<f32> {
        let (array, values_shape) = as_f32_array(&values)?;
        let values = array.as_slice()?;
        let inner = self.snapshot();
        check_shape(py, "values", &inner.shape(), &values_shape)?;
        detach_if_large(py, values.len(), || {
            with_inner!(&inner, |quantized| quantized
                .dot(values)
                .map_err(from_quantize))
        })
    }

    /// Multiply `inputs` by this tensor's matrix `W`, of shape
    /// `(rows, columns)`, which the tensor was quantized from.
    ///
    /// `inputs` is one vector of shape `(columns,)` or a batch of shape
    /// `(batch, columns)`. The result is `inputs @ W.T`, of shape `(rows,)`
    /// or `(batch, rows)`. `out`, if given, must be a float32 NumPy array or
    /// CPU PyTorch tensor of that shape that shares no memory with `inputs`,
    /// and is returned, so a loop can reuse one.
    ///
    /// Each call decodes the matrix one row at a time, straight from the
    /// packed codes, and multiplies each row by every input before moving
    /// on, so the whole matrix is never decoded at once.
    #[pyo3(signature = (inputs, out = None))]
    fn matmul<'py>(
        &self,
        py: Python<'py>,
        inputs: Bound<'_, PyAny>,
        out: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.snapshot();
        let Some((rows, columns)) = with_inner!(&inner, |quantized| quantized.shape()) else {
            return Err(from_quantize(Error::NotAMatrix { len: inner.len() }));
        };
        let (array, batch) = as_f32_matmul_values(&inputs, columns)?;
        let inputs = array.as_slice()?;
        let products = inputs.len().saturating_mul(rows);
        if let Some(out) = out {
            let shape = match batch {
                Some(batch) => vec![batch, rows],
                None => vec![rows],
            };
            let mut writable = as_writable_f32_out(&out, &shape, inputs)?;
            let output = writable.as_slice_mut()?;
            detach_if_large(py, products, || {
                with_inner!(&inner, |quantized| quantized.matmul_into(inputs, output))
            })
            .map_err(from_quantize)?;
            return Ok(out);
        }
        let output = detach_if_large(py, products, || {
            with_inner!(&inner, |quantized| quantized
                .matmul(inputs)
                .map_err(from_quantize))
        })?;
        let output = output.into_pyarray(py);
        match batch {
            Some(batch) => Ok(output.reshape([batch, rows])?.into_any()),
            None => Ok(output.into_any()),
        }
    }

    /// An independent copy of the tensor, such as the original to keep before
    /// `learned.refine` changes it.
    fn copy(&self) -> Self {
        Self::from(self.snapshot())
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    /// Whether the tensor holds no values.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(rows, columns)` for a tensor quantized from a 2-D array, or `(len,)`.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.snapshot().shape())
    }

    /// `'symmetric'`, `'asymmetric'`, or `'adaptive'`.
    #[getter]
    fn kind(&self) -> &'static str {
        with_inner!(&self.snapshot(), |quantized| kind(quantized))
    }

    /// How the scales and zero-points are stored, as a `Scale`.
    #[getter]
    fn scale(&self) -> PyScale {
        self.snapshot().scale()
    }

    /// The width of every code, or `None` for an adaptive tensor, whose
    /// widths are in `block_bits`.
    #[getter]
    fn bits(&self) -> Option<u32> {
        with_inner!(&self.snapshot(), |quantized| bits(quantized))
    }

    /// How many values share each scale.
    #[getter]
    fn block(&self) -> usize {
        with_inner!(&self.snapshot(), |quantized| quantized.block())
    }

    /// One scale per block, as float32. A symmetric block whose value farthest
    /// from zero is positive gets a negative scale.
    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let values = with_inner!(&self.snapshot(), |quantized| quantized
            .scales()
            .iter()
            .copied()
            .map(Scale::to_f32)
            .collect());
        f32_array(py, values)
    }

    /// One zero-point per block, as float32, or none for a symmetric tensor.
    #[getter]
    fn zero_points<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let values = with_inner!(&self.snapshot(), |quantized| quantized
            .zero_points()
            .iter()
            .copied()
            .map(Scale::to_f32)
            .collect());
        f32_array(py, values)
    }

    /// The codes packed into bytes, low bits first, as uint8.
    #[getter]
    fn codes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let bytes = with_inner!(&self.snapshot(), |quantized| quantized.codes().to_vec());
        bytes.into_pyarray(py)
    }

    /// One code per value, as int32, row after row for a matrix.
    #[getter]
    fn unpacked_codes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        let codes = with_inner!(&self.snapshot(), |quantized| quantized.unpacked_codes());
        codes.into_pyarray(py)
    }

    /// Each block's code width for an adaptive tensor, as uint8, or `None`.
    #[getter]
    fn block_bits<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u8>>> {
        with_inner!(&self.snapshot(), |quantized| quantized
            .block_bits()
            .map(|bits| bits.to_vec().into_pyarray(py)))
    }

    /// Bytes held by the codes, scales, zero-points, and block widths. An
    /// adaptive matrix also keeps 8 bytes a row in memory, to find where each
    /// row starts, which this doesn't count.
    #[getter]
    fn nbytes(&self) -> usize {
        with_inner!(&self.snapshot(), |quantized| quantized.nbytes())
    }

    /// Bits per value, counting the scales: 4-bit codes with one f16 scale
    /// per 32 values cost 4.5. An adaptive matrix also keeps 8 bytes a row in
    /// memory, to find where each row starts, which this doesn't count: at
    /// 64 columns, that's 1 more bit per value.
    #[getter]
    fn bits_per_element(&self) -> f32 {
        with_inner!(&self.snapshot(), |quantized| quantized.bits_per_element())
    }

    /// Rebuild a tensor from the values its getters return, such as parts
    /// saved with `numpy.savez`. `scale` is a `Scale` or its name, like
    /// `q.scale.name`. `shape` is `(len,)` or `(rows, columns)`. Symmetric and
    /// asymmetric tensors take `bits`, adaptive ones take `block_bits`, and
    /// symmetric ones take no `zero_points`. `kind` and `scale` can be the 0-d
    /// arrays that `numpy.load` returns for saved strings. Parts that don't fit
    /// together, or a field out of range, raise `QuantizeError`.
    #[staticmethod]
    #[pyo3(signature = (
        *, kind, shape, block, codes, scales, scale, zero_points = None, bits = None,
        block_bits = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        kind: Bound<'_, PyAny>,
        shape: Vec<usize>,
        block: usize,
        codes: Bound<'_, PyAny>,
        scales: Bound<'_, PyAny>,
        scale: PyScale,
        zero_points: Option<Bound<'_, PyAny>>,
        bits: Option<u32>,
        block_bits: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let block_bits = match block_bits {
            Some(block_bits) => Some(as_block_bits(&block_bits)?),
            None => None,
        };
        let zero_points = match zero_points {
            Some(zero_points) => as_f32_values(&zero_points)?.to_vec()?,
            None => Vec::new(),
        };
        let parts = Parts {
            kind: kind.str()?.to_string(),
            shape,
            block,
            codes: as_packed_codes(&codes)?,
            scales: as_f32_values(&scales)?.to_vec()?,
            zero_points,
            bits,
            block_bits,
        };
        QuantizedInner::from_parts(parts, scale).map(Self::from)
    }

    /// Save the tensor as bytes that `from_bytes` loads back. They hold every
    /// part, shape and scale type included, in the format of the Rust crate's
    /// `Quantized::to_bytes`, so either language can load them.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.snapshot().to_bytes())
    }

    /// Load a tensor that `to_bytes` saved, in Python or in Rust. `data` is
    /// `bytes` or another bytes-like object, or a 1-D uint8 array, such as a
    /// NumPy array or the PyTorch tensor that safetensors loads. Bytes that
    /// don't hold a valid tensor raise `QuantizeError`.
    #[staticmethod]
    fn from_bytes(data: Bound<'_, PyAny>) -> PyResult<Self> {
        QuantizedInner::from_bytes(&as_bytes(&data)?)
            .map(Self::from)
            .map_err(from_quantize)
    }

    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_bytes(data)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let shape = self.shape(py)?.repr()?;
        Ok(match self.bits() {
            Some(bits) => format!(
                "Quantized(kind='{}', bits={bits}, block={}, shape={shape}, scale={})",
                self.kind(),
                self.block(),
                self.scale()
            ),
            None => format!(
                "Quantized(kind='adaptive', block={}, shape={shape}, scale={})",
                self.block(),
                self.scale()
            ),
        })
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    // Pickles rebuild the tensor by calling the class, so that `torch.load`
    // loads them once `add_safe_globals([Quantized])` allows it. A static
    // method would pickle as a call to `getattr`, which `torch.load` refuses.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> (Bound<'py, PyType>, (Bound<'py, PyBytes>,)) {
        (slf.get_type(), (slf.get().to_bytes(slf.py()),))
    }

    /// Pickles saved by quantize-py 0.2 call this with one tuple, which starts
    /// with their format number, 1. That format doesn't load anymore, so say
    /// how to move the tensor over instead.
    #[staticmethod]
    fn _from_pickle(state: Bound<'_, PyAny>) -> PyResult<Self> {
        let format: Option<i64> = state.get_item(0).and_then(|item| item.extract()).ok();
        let reason = match format {
            Some(1) => PICKLED_BY_0_2,
            _ => "malformed pickle state",
        };
        Err(PyValueError::new_err(reason))
    }
}
