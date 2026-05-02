//! The list of methods evaluated each run. Add a row here to add a column
//! to the comparison table. Rows are listed in printing order — sorted by
//! `bits_per_element` so comparable methods sit on adjacent rows.
//!
//! `bits_per_element` accounts for both data bits and amortized scale
//! overhead, computed from actual storage facts (no magic numbers).

use super::{
    quant::{eval_q4_0, eval_q5_0, eval_q8_0, eval_quantize},
    Sample,
};
use candle_core::{quantized::GgmlDType, Device, Result};

pub(super) type EvalFn = fn(&Sample, &Device) -> Result<Vec<f32>>;

/// How `bits_per_element` is computed for a given method.
pub(super) enum Bits {
    Ours(u32),         // BITS data bits + one f16 scale per 32-element block
    Candle(GgmlDType), // type_size bytes per block_size-element block
}

impl Bits {
    pub fn evaluate(&self, _elements: usize) -> f32 {
        match self {
            Self::Ours(b) => *b as f32 + 16.0 / 32.0,
            Self::Candle(d) => d.type_size() as f32 * 8.0 / d.block_size() as f32,
        }
    }
}

pub(super) struct Method {
    pub name: &'static str,
    pub bits_per_element: Bits,
    pub eval: EvalFn,
}

#[rustfmt::skip]
pub(super) fn methods() -> Vec<Method> {
    use Bits::*;
    use GgmlDType::{Q4_0, Q5_0, Q8_0};
    vec![
        Method { name: "ours 4b×32",  bits_per_element: Ours(4),      eval: eval_quantize::<4> },
        Method { name: "candle Q4_0",  bits_per_element: Candle(Q4_0), eval: eval_q4_0       },
        Method { name: "ours 5b×32",  bits_per_element: Ours(5),      eval: eval_quantize::<5> },
        Method { name: "candle Q5_0",  bits_per_element: Candle(Q5_0), eval: eval_q5_0       },
        Method { name: "ours 8b×32",  bits_per_element: Ours(8),      eval: eval_quantize::<8> },
        Method { name: "candle Q8_0",  bits_per_element: Candle(Q8_0), eval: eval_q8_0       },
    ]
}
