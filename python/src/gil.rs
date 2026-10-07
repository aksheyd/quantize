//! When calls let other threads run.

use pyo3::marker::Ungil;
use pyo3::prelude::*;

/// Calls that read a tensor, like `dot` and `matmul`, keep the GIL when they
/// go through at most this many values, like one vector times a 256 × 256
/// matrix. They take a tenth of a millisecond or less, so holding it barely
/// delays other threads. Giving it up would make them wait to get it back
/// while another thread keeps running Python, for up to the switch interval,
/// `sys.getswitchinterval()`, which is 5 ms by default.
const MOST_VALUES_READ_KEEPING_THE_GIL: usize = 65_536;

/// Quantizing or refitting a value takes 10 to 20 times as long as reading
/// it, so those calls keep the GIL only up to this many values, like a
/// 64 × 64 matrix, which take about as long as the largest reads that keep
/// it. Larger ones give it up, so that threads quantizing a model's layers
/// run in parallel instead of taking turns.
const MOST_VALUES_QUANTIZED_KEEPING_THE_GIL: usize = 4_096;

/// Run `work`, which reads a tensor, letting other threads run meanwhile
/// unless it's small. `values` counts each value it goes through, once for
/// each vector it's multiplied by.
pub fn detach_if_large<T, F>(py: Python<'_>, values: usize, work: F) -> T
where
    F: Ungil + FnOnce() -> T,
    T: Ungil,
{
    if values <= MOST_VALUES_READ_KEEPING_THE_GIL {
        work()
    } else {
        py.detach(work)
    }
}

/// Run `work`, which quantizes or refits a tensor, letting other threads run
/// meanwhile unless it's tiny. `values` counts each value it goes through,
/// once for each pass over it.
pub fn detach_unless_tiny<T, F>(py: Python<'_>, values: usize, work: F) -> T
where
    F: Ungil + FnOnce() -> T,
    T: Ungil,
{
    if values <= MOST_VALUES_QUANTIZED_KEEPING_THE_GIL {
        work()
    } else {
        py.detach(work)
    }
}
