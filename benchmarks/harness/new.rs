//! Build a `Harness` config. No data is generated yet — `sample.rs` does
//! that fresh on each call to `run()`.

use super::Harness;
use candle_core::{Device, Result};

impl Harness {
    /// `matrix_size` must be a multiple of 32 (Candle Q4/Q5/Q8 block size).
    pub fn new(matrix_size: usize, runs: usize) -> Result<Self> {
        assert!(
            matrix_size % 32 == 0,
            "matrix_size must be a multiple of 32 for Candle Q*_0"
        );
        Ok(Self {
            matrix_size,
            runs,
            device: Device::Cpu,
        })
    }
}
