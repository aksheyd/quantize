use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

use quantize::{Packed, Quantized, Scale};

use super::inner::{with_inner, PyQuantized};
use super::pickle::{from_pickle, pickle_state};
use crate::error::{from_quantize, length_mismatch};
use crate::input::{as_f32_values, as_writable_f32_out};
use crate::scale::PyScale;

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
    #[pyo3(signature = (out = None))]
    fn dequantize<'py>(
        slf: &Bound<'py, Self>,
        out: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let len = slf.borrow().len();
        match out {
            Some(out) => {
                let mut output = as_writable_f32_out(&out, len)?;
                slf.borrow()
                    .dequantize_into(output.as_slice_mut()?)
                    .map_err(from_quantize)?;
                Ok(out)
            }
            None => {
                let inner = slf.borrow().inner.clone();
                let values = py.detach(|| with_inner!(&inner, |quantized| quantized.dequantize()));
                Ok(f32_array(py, values))
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

    fn __repr__(&self) -> String {
        match self.bits() {
            Some(bits) => format!(
                "Quantized(kind='{}', bits={bits}, block={}, len={}, scale={})",
                self.kind(),
                self.block(),
                self.len(),
                self.scale()
            ),
            None => format!(
                "Quantized(kind='adaptive', block={}, len={}, scale={})",
                self.block(),
                self.len(),
                self.scale()
            ),
        }
    }

    fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pickle_state(py, &self.inner)
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, (Bound<'py, PyAny>,))> {
        let callable = slf.getattr("_from_pickle")?;
        let state = slf.borrow().__getstate__(slf.py())?;
        Ok((callable, (state,)))
    }

    #[staticmethod]
    fn _from_pickle(state: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: from_pickle(state)?,
        })
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
