//! The list of methods evaluated each run. Add a row here to add a column
//! to the comparison table. Each `eval` is a function pointer with the
//! shared `EvalFn` signature.

use super::{
    quant::{eval_q4_0, eval_q5_0, eval_q8_0, eval_yours},
    Sample,
};
use candle_core::{Device, Result};

pub(super) type EvalFn = fn(&Sample, &Device) -> Result<Vec<f32>>;

pub(super) struct Method {
    pub name: &'static str,
    pub bits_per_element: f32,
    pub eval: EvalFn,
}

pub(super) fn methods() -> Vec<Method> {
    vec![
        Method {
            name: "yours 4-bit",
            bits_per_element: 4.0,
            eval: eval_yours::<4>,
        },
        Method {
            name: "yours 8-bit",
            bits_per_element: 8.0,
            eval: eval_yours::<8>,
        },
        Method {
            name: "yours 16-bit",
            bits_per_element: 16.0,
            eval: eval_yours::<16>,
        },
        Method {
            name: "candle Q4_0",
            bits_per_element: 4.5,
            eval: eval_q4_0,
        },
        Method {
            name: "candle Q5_0",
            bits_per_element: 5.5,
            eval: eval_q5_0,
        },
        Method {
            name: "candle Q8_0",
            bits_per_element: 8.5,
            eval: eval_q8_0,
        },
    ]
}
