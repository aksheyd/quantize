//! Checks that `matmul` gives the same bits with `main` and with this branch,
//! on random matrices in every layout, at batch sizes on both sides of each
//! group boundary, with f32, f16, and bf16 scales.
//!
//! Run: `cargo run --release --manifest-path timing/Cargo.toml --bin bits`

mod layout;

use layout::{BothScales, Layout, values};
use quantize::{bf16, f16};

struct Random(u64);

impl Random {
    fn below(&mut self, limit: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % limit as u64) as usize
    }
}

#[derive(Default)]
struct Counts {
    matrices: usize,
    products: usize,
    grouped: usize,
}

fn check<S: BothScales>(random: &mut Random, counts: &mut Counts, seed: u64) {
    let layout = match random.below(8) {
        0 | 1 => Layout::Symmetric {
            bits: 4,
            block: [32, 64, 96, 128][random.below(4)],
        },
        2 | 3 => Layout::Symmetric {
            bits: 8,
            block: [32, 64, 96, 128][random.below(4)],
        },
        4 => Layout::Symmetric {
            bits: [4, 8][random.below(2)],
            block: [1, 7, 8, 16, 24, 40][random.below(6)],
        },
        5 => Layout::Symmetric {
            bits: 2 + random.below(15) as u32,
            block: 1 + random.below(70),
        },
        6 => Layout::Asymmetric {
            bits: 2 + random.below(15) as u32,
            block: 1 + random.below(70),
        },
        _ => Layout::Adaptive {
            block: 1 + random.below(70),
            tolerance: [0.001, 0.01, 0.1][random.below(3)],
        },
    };
    let block = match layout {
        Layout::Symmetric { block, .. }
        | Layout::Asymmetric { block, .. }
        | Layout::Adaptive { block, .. } => block,
    };
    // Mostly rows of whole blocks, sometimes rows that end partway through one.
    let columns = if random.below(4) == 0 {
        1 + random.below(300)
    } else {
        block * (1 + random.below(6))
    };
    let rows = 1 + random.below(12);
    let mut weights = values(rows * columns, seed);
    for weight in &mut weights {
        if random.below(50) == 0 {
            *weight *= 1000.0;
        }
    }
    let (Some(mut branch), Some(mut main)) =
        (layout.branch::<S>(&weights), layout.main::<S>(&weights))
    else {
        return;
    };
    branch.set_shape(rows, columns).unwrap();
    main.set_shape(rows, columns).unwrap();
    counts.matrices += 1;
    for _ in 0..3 {
        let batch = [1, 2, 3, 255, 256, 257, 300, 511, 512, 513, 769][random.below(11)];
        let mut inputs = values(batch * columns, seed ^ 0xABCD ^ batch as u64);
        for input in &mut inputs {
            match random.below(400) {
                0 => *input = f32::NAN,
                1 => *input = f32::INFINITY,
                2 => *input = -0.0,
                3 => *input *= 1e30,
                _ => {}
            }
        }
        let expected = main.matmul(&inputs).unwrap();
        let got = branch.matmul(&inputs).unwrap();
        let same = expected.len() == got.len()
            && expected
                .iter()
                .zip(&got)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let scale = <S as quantize::Scale>::NAME;
        assert!(
            same,
            "{layout:?}, {scale} scales, {rows} x {columns}, batch {batch}: bits differ"
        );
        counts.products += 1;
        if batch > 256
            && matches!(layout, Layout::Symmetric { bits: 4 | 8, block } if block % 32 == 0 && columns % block == 0)
        {
            counts.grouped += 1;
        }
    }
}

fn main() {
    let mut random = Random(0x5EED_1234_ABCD_0001);
    let mut counts = Counts::default();
    for seed in 0..20_000 {
        match seed % 3 {
            0 => check::<f32>(&mut random, &mut counts, seed),
            1 => check::<f16>(&mut random, &mut counts, seed),
            _ => check::<bf16>(&mut random, &mut counts, seed),
        }
    }
    println!(
        "{} matrices, {} products, {} of them batches of more than 256 in the grouped layouts: every bit the same as main",
        counts.matrices, counts.products, counts.grouped
    );
}
