//! When calls let other threads run.

use pyo3::marker::Ungil;
use pyo3::prelude::*;

/// Calls on at most this many values, like a 256 × 256 matrix, keep the GIL.
/// They take half a millisecond or less, so holding it barely delays other
/// threads. Giving it up would make them wait to get it back while another
/// thread keeps running Python, for up to the switch interval,
/// `sys.getswitchinterval()`, which is 5 ms by default.
const MOST_VALUES_KEEPING_THE_GIL: usize = 65_536;

/// Run `work`, letting other threads run meanwhile unless it's small.
/// `values` counts each value it goes through, once for each pass over it or
/// each vector it's multiplied by.
pub fn detach_if_large<T, F>(py: Python<'_>, values: usize, work: F) -> T
where
    F: Ungil + FnOnce() -> T,
    T: Ungil,
{
    if values <= MOST_VALUES_KEEPING_THE_GIL {
        work()
    } else {
        py.detach(work)
    }
}
