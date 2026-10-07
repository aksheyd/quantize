//! Times `matmul` with `main` and with this branch, alternating calls, and
//! checks that both give the same bits.
//!
//! Run: `cargo run --release --manifest-path timing/Cargo.toml --bin compare`

mod layout;

use layout::{BothScales, Layout, values};
use quantize::{bf16, f16};
use std::hint::black_box;
use std::time::{Duration, Instant};

const Q4_32: Layout = Layout::Symmetric { bits: 4, block: 32 };
const Q8_32: Layout = Layout::Symmetric { bits: 8, block: 32 };
const Q4_64: Layout = Layout::Symmetric { bits: 4, block: 64 };
const Q4_16: Layout = Layout::Symmetric { bits: 4, block: 16 };
const FIVE_BIT: Layout = Layout::Symmetric { bits: 5, block: 32 };
const ASYM4: Layout = Layout::Asymmetric { bits: 4, block: 32 };
const ADAPTIVE: Layout = Layout::Adaptive {
    block: 32,
    tolerance: 0.1,
};

fn compare<S: BothScales>(label: &str, layout: Layout, rows: usize, columns: usize, batch: usize) {
    let weights = values(rows * columns, 1);
    let mut branch = layout.branch::<S>(&weights).unwrap();
    let mut main = layout.main::<S>(&weights).unwrap();
    branch.set_shape(rows, columns).unwrap();
    main.set_shape(rows, columns).unwrap();
    let inputs = values(batch * columns, 2);
    let mut branch_out = vec![0.0; batch * rows];
    let mut main_out = vec![0.0; batch * rows];

    let start = Instant::now();
    main.matmul_into(&inputs, &mut main_out).unwrap();
    branch.matmul_into(&inputs, &mut branch_out).unwrap();
    let per_round = start.elapsed();
    let same = main_out
        .iter()
        .zip(&branch_out)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(same, "{label}: main and this branch give different bits");

    let budget = if cfg!(debug_assertions) {
        Duration::from_secs(6)
    } else {
        Duration::from_secs(8)
    };
    let rounds = (budget.as_secs_f64() / per_round.as_secs_f64()).clamp(5.0, 2000.0) as usize;
    let (mut main_best, mut branch_best) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..rounds {
        let start = Instant::now();
        main.matmul_into(black_box(&inputs), &mut main_out).unwrap();
        main_best = main_best.min(start.elapsed().as_secs_f64());
        black_box(&main_out);
        let start = Instant::now();
        branch
            .matmul_into(black_box(&inputs), &mut branch_out)
            .unwrap();
        branch_best = branch_best.min(start.elapsed().as_secs_f64());
        black_box(&branch_out);
    }
    let case = format!(
        "{label} {} {} @{batch}",
        layout.name(),
        <S as quantize::Scale>::NAME
    );
    println!(
        "{case:<40}{:>11.3}{:>11.3}{:>9.2}x   ({rounds} rounds)",
        main_best * 1e3,
        branch_best * 1e3,
        main_best / branch_best
    );
}

fn main() {
    let build = if cfg!(debug_assertions) {
        "debug, dependencies optimized"
    } else {
        "release"
    };
    println!("{build}: best ms of alternating calls, and how many times faster this branch is");
    println!(
        "{:<40}{:>11}{:>11}{:>10}",
        "case", "main", "branch", "speedup"
    );
    if cfg!(debug_assertions) {
        compare::<f16>("1024²", Q4_32, 1024, 1024, 1);
        compare::<f16>("1024²", Q8_32, 1024, 1024, 1);
        compare::<f16>("1024²", Q4_32, 1024, 1024, 16);
        compare::<f16>("1024²", Q4_32, 1024, 1024, 256);
        compare::<f16>("1024²", Q4_32, 1024, 1024, 384);
        compare::<f16>("1024²", Q4_32, 1024, 1024, 512);
        compare::<f16>("1024²", Q8_32, 1024, 1024, 512);
        compare::<f32>("1024²", Q4_32, 1024, 1024, 512);
        compare::<f16>("768²", Q8_32, 768, 768, 394);
        compare::<f16>("1024²", ASYM4, 1024, 1024, 384);
        compare::<f16>("SmolLM down", Q4_32, 576, 1536, 512);
        compare::<f16>("4096²", Q4_32, 4096, 4096, 512);
        compare::<f16>("ViT qkv", Q4_32, 2304, 768, 6304);
        compare::<f16>("ViT qkv", Q8_32, 2304, 768, 6304);
        return;
    }
    // One vector, as when a language model generates a token.
    compare::<f16>("2048²", Q4_32, 2048, 2048, 1);
    compare::<f16>("2048²", Q8_32, 2048, 2048, 1);
    compare::<f32>("2048²", Q4_32, 2048, 2048, 1);
    // Small batches, which stay one group.
    compare::<f16>("1024²", Q4_32, 1024, 1024, 2);
    compare::<f16>("1024²", Q4_32, 1024, 1024, 16);
    compare::<f16>("1024²", Q8_32, 1024, 1024, 16);
    compare::<f16>("1024²", Q4_32, 1024, 1024, 64);
    compare::<f16>("1024²", Q4_32, 1024, 1024, 256);
    compare::<f16>("1024²", Q8_32, 1024, 1024, 256);
    // Batches just past one group.
    compare::<f16>("1024²", Q4_32, 1024, 1024, 257);
    compare::<f16>("1024²", Q4_32, 1024, 1024, 384);
    compare::<f16>("1024²", Q4_32, 1024, 1024, 512);
    compare::<f16>("1024²", Q8_32, 1024, 1024, 512);
    compare::<f32>("1024²", Q4_32, 1024, 1024, 512);
    compare::<bf16>("1024²", Q4_32, 1024, 1024, 512);
    compare::<f16>("1024²", Q4_64, 1024, 1024, 512);
    compare::<f16>("1024²", Q4_32, 1024, 1024, 1024);
    compare::<f16>("768²", Q4_32, 768, 768, 394);
    // Large batches: SmolLM-135M's layers at the WikiText window of 512, a
    // 4096-wide layer, and ViT-B/16's layers on 32 images.
    compare::<f16>("SmolLM q", Q4_32, 576, 576, 512);
    compare::<f16>("SmolLM up", Q4_32, 1536, 576, 512);
    compare::<f16>("SmolLM down", Q4_32, 576, 1536, 512);
    compare::<f16>("SmolLM lm_head", Q4_32, 49152, 576, 512);
    compare::<f16>("4096²", Q4_32, 4096, 4096, 512);
    compare::<f16>("4096²", Q4_32, 4096, 4096, 2048);
    compare::<f16>("ViT qkv", Q4_32, 2304, 768, 6304);
    compare::<f16>("ViT qkv", Q8_32, 2304, 768, 6304);
    compare::<f32>("ViT qkv", Q4_32, 2304, 768, 6304);
    compare::<f16>("ViT MLP in", Q4_32, 3072, 768, 6304);
    compare::<f16>("ViT MLP out", Q4_32, 768, 3072, 6304);
    // Layouts that keep the whole batch as one group.
    compare::<f16>("1024²", ASYM4, 1024, 1024, 384);
    compare::<f16>("1024²", ADAPTIVE, 1024, 1024, 384);
    compare::<f16>("1024²", FIVE_BIT, 1024, 1024, 384);
    compare::<f16>("1024²", Q4_16, 1024, 1024, 384);
    compare::<f16>("ViT qkv", ASYM4, 2304, 768, 6304);
}
