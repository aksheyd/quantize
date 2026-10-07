//! One layout, built the same way with `main` and with this branch.

pub trait BothScales: quantize::Scale + quantize_main::Scale {}
impl<S: quantize::Scale + quantize_main::Scale> BothScales for S {}

#[derive(Clone, Copy, Debug)]
pub enum Layout {
    Symmetric { bits: u32, block: usize },
    Asymmetric { bits: u32, block: usize },
    Adaptive { block: usize, tolerance: f32 },
}

impl Layout {
    #[allow(dead_code)]
    pub fn name(self) -> String {
        match self {
            Layout::Symmetric { bits: 4, block: 32 } => "Q4_32".into(),
            Layout::Symmetric { bits: 8, block: 32 } => "Q8_32".into(),
            Layout::Symmetric { bits, block } => format!("sym{bits}/{block}"),
            Layout::Asymmetric { bits, block } => format!("asym{bits}/{block}"),
            Layout::Adaptive { block, .. } => format!("adaptive/{block}"),
        }
    }

    pub fn branch<S: BothScales>(self, values: &[f32]) -> Option<quantize::Quantized<S>> {
        use quantize::Scheme;
        let scheme = match self {
            Layout::Symmetric { bits, block } => Scheme::Symmetric { bits, block },
            Layout::Asymmetric { bits, block } => Scheme::Asymmetric { bits, block },
            Layout::Adaptive { block, tolerance } => Scheme::Adaptive { block, tolerance },
        };
        scheme.quantize(values).ok()
    }

    pub fn main<S: BothScales>(self, values: &[f32]) -> Option<quantize_main::Quantized<S>> {
        use quantize_main::Scheme;
        let scheme = match self {
            Layout::Symmetric { bits, block } => Scheme::Symmetric { bits, block },
            Layout::Asymmetric { bits, block } => Scheme::Asymmetric { bits, block },
            Layout::Adaptive { block, tolerance } => Scheme::Adaptive { block, tolerance },
        };
        scheme.quantize(values).ok()
    }
}

pub fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}
