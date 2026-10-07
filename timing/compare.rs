//! Times `matmul` with `main`, a second copy of `main`, and this branch,
//! called in turn, and checks that `main` and this branch give the same bits.
//! The two copies of `main` show how far apart identical code can land.
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
    let mut main_again = layout.main_again::<S>(&weights).unwrap();
    branch.set_shape(rows, columns).unwrap();
    main.set_shape(rows, columns).unwrap();
    main_again.set_shape(rows, columns).unwrap();
    let inputs = values(batch * columns, 2);
    let mut branch_out = vec![0.0; batch * rows];
    let mut main_out = vec![0.0; batch * rows];

    let start = Instant::now();
    main.matmul_into(&inputs, &mut main_out).unwrap();
    branch.matmul_into(&inputs, &mut branch_out).unwrap();
    main_again.matmul_into(&inputs, &mut main_out).unwrap();
    let per_round = start.elapsed();
    let same = main_out
        .iter()
        .zip(&branch_out)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    assert!(same, "{label}: main and this branch give different bits");

    let budget = if cfg!(debug_assertions) {
        Duration::from_secs(6)
    } else {
        Duration::from_secs(9)
    };
    let rounds = (budget.as_secs_f64() / per_round.as_secs_f64()).clamp(6.0, 3000.0) as usize;
    // Best seconds of main, main again, and this branch, called in turn,
    // starting with a different one each round.
    let mut best = [f64::INFINITY; 3];
    for round in 0..rounds {
        for turn in 0..3 {
            let which = (round + turn) % 3;
            let start = Instant::now();
            match which {
                0 => main.matmul_into(black_box(&inputs), &mut main_out).unwrap(),
                1 => main_again
                    .matmul_into(black_box(&inputs), &mut main_out)
                    .unwrap(),
                _ => branch
                    .matmul_into(black_box(&inputs), &mut branch_out)
                    .unwrap(),
            }
            best[which] = best[which].min(start.elapsed().as_secs_f64());
            black_box((&main_out, &branch_out));
        }
    }
    let case = format!(
        "{label} {} {} @{batch}",
        layout.name(),
        <S as quantize::Scale>::NAME
    );
    let main_best = best[0].min(best[1]);
    println!(
        "{case:<36}{:>10.3}{:>11.3}{:>10.3}{:>8.3}{:>8.2}x   ({rounds} rounds)",
        best[0] * 1e3,
        best[1] * 1e3,
        best[2] * 1e3,
        best[0].max(best[1]) / main_best,
        main_best / best[2],
    );
}

fn main() {
    let build = if cfg!(debug_assertions) {
        "debug, dependencies optimized"
    } else {
        "release"
    };
    println!("{build}: best ms of calls in turn. main again is the same commit as main;");
    println!("spread is how many times slower the slower copy of main is, and speedup");
    println!("is how many times faster this branch is than the faster copy");
    println!(
        "{:<36}{:>10}{:>11}{:>10}{:>8}{:>9}",
        "case", "main", "main again", "branch", "spread", "speedup"
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
        compare::<f16>("1024²", ASYM4, 1024, 1024, 2);
        compare::<f16>("1024²", FIVE_BIT, 1024, 1024, 2);
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
    compare::<f16>("1024²", ASYM4, 1024, 1024, 2);
    compare::<f16>("1024²", FIVE_BIT, 1024, 1024, 2);
    compare::<f16>("1024²", ADAPTIVE, 1024, 1024, 2);
    compare::<f16>("1024²", ASYM4, 1024, 1024, 384);
    compare::<f16>("1024²", ADAPTIVE, 1024, 1024, 384);
    compare::<f16>("1024²", FIVE_BIT, 1024, 1024, 384);
    compare::<f16>("1024²", Q4_16, 1024, 1024, 384);
    compare::<f16>("ViT qkv", ASYM4, 2304, 768, 6304);
}
