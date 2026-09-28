use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use quantize::{Error, Packed, Quantized, Scale};

use super::inner::{PyQuantized, QuantizedInner, with_inner};
use super::parts::Parts;
use crate::error::{from_quantize, length_mismatch};
use crate::input::{as_f32_matmul_values, as_f32_values, as_packed_codes, as_writable_f32_out};
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

fn unpacked_codes<S: Scale>(quantized: &Quantized<S>) -> Vec<i32> {
    let mut unpacked = vec![0; quantized.len()];
    match quantized {
        Quantized::Symmetric { codes, .. } | Quantized::Asymmetric { codes, .. } => {
            codes.unpack_into(&mut unpacked);
        }
        Quantized::Adaptive {
            bytes,
            bits,
            block,
            len,
            ..
        } => {
            let mut byte_offset = 0;
            let mut value_offset = 0;
            for &bit_width in bits {
                let count = (*len - value_offset).min(*block);
                let byte_count = (count * bit_width as usize).div_ceil(8);
                Packed::unpack_slice(
                    &bytes[byte_offset..byte_offset + byte_count],
                    bit_width.into(),
                    &mut unpacked[value_offset..value_offset + count],
                    count,
                );
                byte_offset += byte_count;
                value_offset += count;
            }
        }
    }
    unpacked
}

// Methods that release the GIL take `slf` instead of `&self`, and clone the
// tensor out of a short borrow first. A borrow held while the GIL is released
// would keep `learned.refine` on another thread from borrowing it mutably.
#[pymethods]
impl PyQuantized {
    /// Decode the values into an array of the tensor's `shape`. `out`, if
    /// given, must be a float32 array of that shape, and is returned.
    #[pyo3(signature = (out = None))]
    fn dequantize<'py>(
        slf: &Bound<'py, Self>,
        out: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let shape = slf.borrow().inner.shape();
        match out {
            Some(out) => {
                let mut output = as_writable_f32_out(&out, &shape)?;
                slf.borrow()
                    .dequantize_into(output.as_slice_mut()?)
                    .map_err(from_quantize)?;
                Ok(out)
            }
            None => {
                let inner = slf.borrow().inner.clone();
                let values = py.detach(|| with_inner!(&inner, |quantized| quantized.dequantize()));
                Ok(values.into_pyarray(py).reshape(shape)?.into_any())
            }
        }
    }

    /// The dot product of the decoded values with `values`, a 1-D array of
    /// `len(q)` numbers, without storing the decoded values. For a matrix
    /// times a vector, use `matmul`.
    fn dot(slf: &Bound<'_, Self>, values: Bound<'_, PyAny>) -> PyResult<f32> {
        let array = as_f32_values(&values)?;
        let values = array.as_slice()?;
        let inner = slf.borrow().inner.clone();
        if values.len() != inner.len() {
            return Err(length_mismatch("values", inner.len(), values.len()));
        }
        slf.py().detach(|| {
            with_inner!(&inner, |quantized| quantized
                .dot(values)
                .map_err(from_quantize))
        })
    }

    /// Multiply `values` by this tensor's matrix `W`, of shape
    /// `(rows, columns)`, which the tensor was quantized from.
    ///
    /// `values` is one vector of shape `(columns,)` or a batch of shape
    /// `(batch, columns)`. The result is `values @ W.T`, of shape `(rows,)`
    /// or `(batch, rows)`.
    fn matmul<'py>(
        slf: &Bound<'py, Self>,
        values: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let inner = slf.borrow().inner.clone();
        let Some((rows, columns)) = with_inner!(&inner, |quantized| quantized.shape()) else {
            return Err(from_quantize(Error::NotAMatrix { len: inner.len() }));
        };
        let (array, batch) = as_f32_matmul_values(&values, columns)?;
        let values = array.as_slice()?;
        let output = py.detach(|| {
            with_inner!(&inner, |quantized| quantized
                .matmul(values)
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
        Self {
            inner: self.inner.clone(),
        }
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
        PyTuple::new(py, self.inner.shape())
    }

    /// `'symmetric'`, `'asymmetric'`, or `'adaptive'`.
    #[getter]
    fn kind(&self) -> &'static str {
        with_inner!(&self.inner, |quantized| kind(quantized))
    }

    /// How the scales and zero-points are stored, as a `Scale`.
    #[getter]
    fn scale(&self) -> PyScale {
        self.inner.scale()
    }

    /// The width of every code, or `None` for an adaptive tensor, whose
    /// widths are in `block_bits`.
    #[getter]
    fn bits(&self) -> Option<u32> {
        with_inner!(&self.inner, |quantized| bits(quantized))
    }

    /// How many values share each scale.
    #[getter]
    fn block(&self) -> usize {
        with_inner!(&self.inner, |quantized| quantized.block())
    }

    /// One scale per block, as float32. A symmetric block whose value farthest
    /// from zero is positive gets a negative scale.
    #[getter]
    fn scales<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let values = with_inner!(&self.inner, |quantized| quantized
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
        let values = with_inner!(&self.inner, |quantized| quantized
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
        let bytes = with_inner!(&self.inner, |quantized| quantized.codes().to_vec());
        bytes.into_pyarray(py)
    }

    /// One code per value, as int32, row after row for a matrix.
    #[getter]
    fn unpacked_codes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        let codes = with_inner!(&self.inner, |quantized| unpacked_codes(quantized));
        codes.into_pyarray(py)
    }

    /// Each block's code width for an adaptive tensor, as uint8, or `None`.
    #[getter]
    fn block_bits<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u8>>> {
        with_inner!(&self.inner, |quantized| quantized
            .block_bits()
            .map(|bits| bits.to_vec().into_pyarray(py)))
    }

    /// Bytes held by the codes, scales, zero-points, and block widths.
    #[getter]
    fn nbytes(&self) -> usize {
        with_inner!(&self.inner, |quantized| quantized.nbytes())
    }

    /// Bits per value, counting the scales: 4-bit codes with one f16 scale
    /// per 32 values cost 4.5.
    #[getter]
    fn bits_per_element(&self) -> f32 {
        with_inner!(&self.inner, |quantized| quantized.bits_per_element())
    }

    /// Rebuild a tensor from the values its getters return, such as parts
    /// saved with `numpy.savez`. `scale` is a `Scale` or its name, like
    /// `q.scale.name`. `shape` is `(len,)` or `(rows, columns)`. Symmetric and
    /// asymmetric tensors take `bits`, adaptive ones take `block_bits`, and
    /// symmetric ones take no `zero_points`. `kind` and `scale` can be the 0-d
    /// arrays that `numpy.load` returns for saved strings. Parts that don't fit
    /// together raise `ValueError`, or a `QuantizeError` for a field out of
    /// range.
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
        block_bits: Option<Vec<u32>>,
    ) -> PyResult<Self> {
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
        let inner = QuantizedInner::from_parts(parts, scale)?;
        Ok(Self { inner })
    }

    /// Save the tensor as bytes that `from_bytes` loads back. They hold every
    /// part, shape and scale type included, in the format of the Rust crate's
    /// `Quantized::to_bytes`, so either language can load them.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    /// Load a tensor that `to_bytes` saved, in Python or in Rust. `data` is
    /// `bytes` or another bytes-like object, such as a uint8 NumPy array.
    /// Bytes that don't hold a valid tensor raise `ValueError`, or a
    /// `QuantizeError` for a field out of range.
    #[staticmethod]
    fn from_bytes(py: Python<'_>, data: PyBuffer<u8>) -> PyResult<Self> {
        let inner = QuantizedInner::from_bytes(&data.to_vec(py)?).map_err(from_quantize)?;
        Ok(Self { inner })
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

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyBytes>,))> {
        let from_bytes = slf.getattr("from_bytes")?;
        Ok((from_bytes, (slf.borrow().to_bytes(slf.py()),)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::adaptive;

    #[test]
    fn unpacked_adaptive_codes_repack_to_the_original_bytes() {
        let values: Vec<f32> = (0..40).map(|index| index as f32 * 0.02 - 0.4).collect();
        let quantized = adaptive::quantize::<f32, 32>(&values, 0.002).unwrap();
        let unpacked = unpacked_codes(&quantized);

        let Quantized::Adaptive {
            bytes, bits, block, ..
        } = quantized
        else {
            unreachable!()
        };
        let mut repacked = Vec::new();
        for (block_index, &bit_width) in bits.iter().enumerate() {
            let start = block_index * block;
            let end = (start + block).min(unpacked.len());
            repacked.extend_from_slice(
                Packed::from_i32s(&unpacked[start..end], bit_width.into()).as_bytes(),
            );
        }
        assert_eq!(repacked, bytes);
    }
}
