use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};

use quantize::{Error, Packed, Quantized, Scale};

use super::inner::{with_inner, PyQuantized, QuantizedInner};
use crate::error::{from_quantize, length_mismatch};
use crate::input::{as_f32_matmul_values, as_f32_values, as_writable_f32_out};
use crate::scale::PyScale;

/// A pickled tensor: its scale type and [`quantize::Quantized::to_bytes`].
type Pickled<'py> = (PyScale, Bound<'py, PyBytes>);

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
                    bit_width,
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

    fn dot(&self, py: Python<'_>, values: Bound<'_, PyAny>) -> PyResult<f32> {
        let values = as_f32_values(&values)?;
        if values.len() != self.len() {
            return Err(length_mismatch(self.len(), values.len()));
        }
        let inner = self.inner.clone();
        py.detach(|| {
            with_inner!(&inner, |quantized| quantized
                .dot(&values)
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
        &self,
        py: Python<'py>,
        values: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let Some((rows, columns)) = with_inner!(&inner, |quantized| quantized.shape()) else {
            return Err(from_quantize(Error::NotAMatrix { len: self.len() }));
        };
        let (values, batch) = as_f32_matmul_values(&values, columns)?;
        let output = py.detach(|| {
            with_inner!(&inner, |quantized| quantized
                .matmul(&values)
                .map_err(from_quantize))
        })?;
        let output = output.into_pyarray(py);
        match batch {
            Some(batch) => Ok(output.reshape([batch, rows])?.into_any()),
            None => Ok(output.into_any()),
        }
    }

    fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(rows, columns)` for a tensor quantized from a 2-D array, or `(len,)`.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape())
    }

    #[getter]
    fn kind(&self) -> &'static str {
        with_inner!(&self.inner, |quantized| kind(quantized))
    }

    #[getter]
    fn scale(&self) -> PyScale {
        self.inner.scale()
    }

    #[getter]
    fn bits(&self) -> Option<u32> {
        with_inner!(&self.inner, |quantized| bits(quantized))
    }

    #[getter]
    fn block(&self) -> usize {
        with_inner!(&self.inner, |quantized| quantized.block())
    }

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

    #[getter]
    fn codes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let bytes = with_inner!(&self.inner, |quantized| quantized.codes().to_vec());
        bytes.into_pyarray(py)
    }

    #[getter]
    fn unpacked_codes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        let codes = with_inner!(&self.inner, |quantized| unpacked_codes(quantized));
        codes.into_pyarray(py)
    }

    #[getter]
    fn block_bits<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<u32>>> {
        with_inner!(&self.inner, |quantized| quantized
            .block_bits()
            .map(|bits| bits.to_vec().into_pyarray(py)))
    }

    #[getter]
    fn nbytes(&self) -> usize {
        with_inner!(&self.inner, |quantized| quantized.nbytes())
    }

    #[getter]
    fn bits_per_element(&self) -> f32 {
        with_inner!(&self.inner, |quantized| quantized.bits_per_element())
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

    fn __hash__(&self) -> PyResult<isize> {
        Err(PyTypeError::new_err("unhashable type: 'Quantized'"))
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Pickled<'py>)> {
        let callable = slf.getattr("_from_pickle")?;
        let quantized = slf.borrow();
        let bytes = PyBytes::new(slf.py(), &quantized.inner.to_bytes());
        Ok((callable, (quantized.inner.scale(), bytes)))
    }

    #[staticmethod]
    fn _from_pickle(scale: PyScale, bytes: &[u8]) -> PyResult<Self> {
        let inner = QuantizedInner::from_bytes(scale, bytes).map_err(from_quantize)?;
        Ok(Self { inner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantize::adaptive;

    #[test]
    fn unpacked_adaptive_codes_repack_to_the_original_bytes() {
        let values: Vec<f32> = (0..40).map(|index| index as f32 * 0.02 - 0.4).collect();
        let quantized = adaptive::quantize::<f32, 32>(&values, 0.001).unwrap();
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
            repacked
                .extend_from_slice(Packed::from_i32s(&unpacked[start..end], bit_width).as_bytes());
        }
        assert_eq!(repacked, bytes);
    }
}
