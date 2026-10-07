//! Times `matmul_into` on one thread with the batch split into groups of
//! vectors, one call per group, for several group sizes. Every group size
//! must give the same bits as one call over the whole batch.
//!
//! Run: `cargo run --release --manifest-path timing/Cargo.toml -- [quick]`

use quantize::{Quantized, Scheme, f16};
use std::hint::black_box;
use std::time::{Duration, Instant};

const WHOLE_BATCH: usize = usize::MAX;

fn values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn multiply(matrix: &Quantized<f16>, inputs: &[f32], out: &mut [f32], group: usize) {
    let (rows, columns) = matrix.shape().unwrap();
    if group == WHOLE_BATCH {
        matrix.matmul_into(inputs, out).unwrap();
        return;
    }
    let groups = inputs
        .chunks(group * columns)
        .zip(out.chunks_mut(group * rows));
    for (group_inputs, group_out) in groups {
        matrix.matmul_into(group_inputs, group_out).unwrap();
    }
}

fn scheme_named(name: &str) -> Scheme {
    match name {
        "Q4_32" => Scheme::Q4_32,
        "Q8_32" => Scheme::Q8_32,
        "4-bit, block 64" => Scheme::Symmetric { bits: 4, block: 64 },
        "5-bit" => Scheme::Symmetric { bits: 5, block: 32 },
        "asymmetric 4-bit" => Scheme::Asymmetric { bits: 4, block: 32 },
        "adaptive" => Scheme::Adaptive {
            block: 32,
            tolerance: 0.1,
        },
        _ => unreachable!("{name}"),
    }
}

fn main() {
    let quick = std::env::args().any(|argument| argument == "quick");
    let group_sizes: &[usize] = if quick {
        &[16, 32, 64, 128, 256, 512, 1024, WHOLE_BATCH]
    } else {
        &[8, 16, 32, 48, 64, 96, 128, 256, 512, WHOLE_BATCH]
    };
    // (label, scheme, rows, columns, batch)
    let cases: &[(&str, &str, usize, usize, usize)] = if quick {
        &[
            ("ViT qkv", "Q4_32", 2304, 768, 6304),
            ("ViT MLP out", "Q4_32", 768, 3072, 6304),
            ("4096 square", "Q4_32", 4096, 4096, 512),
            ("SmolLM down", "Q4_32", 576, 1536, 512),
            ("SmolLM up", "Q4_32", 1536, 576, 512),
            ("SmolLM q", "Q4_32", 576, 576, 512),
            ("ViT qkv", "Q8_32", 2304, 768, 6304),
            ("ViT qkv", "asymmetric 4-bit", 2304, 768, 6304),
            ("1024 square", "Q4_32", 1024, 1024, 128),
            ("1024 square", "Q4_32", 1024, 1024, 256),
            ("1024 square", "Q8_32", 1024, 1024, 256),
            ("1024 square", "asymmetric 4-bit", 1024, 1024, 128),
            ("1024 square", "asymmetric 4-bit", 1024, 1024, 256),
            ("768 square", "Q4_32", 768, 768, 197),
            ("768 square", "asymmetric 4-bit", 768, 768, 197),
        ]
    } else {
        &[
            ("ViT qkv", "Q4_32", 2304, 768, 6304),
            ("ViT MLP in", "Q4_32", 3072, 768, 6304),
            ("ViT MLP out", "Q4_32", 768, 3072, 6304),
            ("4096 square", "Q4_32", 4096, 4096, 512),
            ("SmolLM down", "Q4_32", 576, 1536, 512),
            ("SmolLM q", "Q4_32", 576, 576, 512),
            ("ViT qkv", "Q8_32", 2304, 768, 6304),
            ("ViT qkv", "asymmetric 4-bit", 2304, 768, 6304),
            ("ViT qkv", "adaptive", 2304, 768, 6304),
            ("ViT qkv", "5-bit", 2304, 768, 6304),
            ("1024 square", "Q4_32", 1024, 1024, 128),
            ("1024 square", "Q4_32", 1024, 1024, 256),
            ("1024 square", "Q8_32", 1024, 1024, 256),
            ("1024 square", "asymmetric 4-bit", 1024, 1024, 128),
            ("1024 square", "asymmetric 4-bit", 1024, 1024, 256),
            ("1024 square", "adaptive", 1024, 1024, 128),
            ("1024 square", "adaptive", 1024, 1024, 256),
            ("1024 square", "5-bit", 1024, 1024, 128),
            ("768 square", "Q4_32", 768, 768, 197),
            ("768 square", "asymmetric 4-bit", 768, 768, 197),
        ]
    };
    print!("{:<34}", "case, ms (best of rounds)");
    for &group in group_sizes {
        let label = if group == WHOLE_BATCH {
            "one call".to_string()
        } else {
            group.to_string()
        };
        print!("{label:>10}");
    }
    println!();
    for &(label, scheme_name, rows, columns, batch) in cases {
        let weights = values(rows * columns, 1);
        let mut matrix = scheme_named(scheme_name).quantize::<f16>(&weights).unwrap();
        matrix.set_shape(rows, columns).unwrap();
        let inputs = values(batch * columns, 2);
        let mut expected = vec![0.0; batch * rows];
        multiply(&matrix, &inputs, &mut expected, WHOLE_BATCH);
        let expected_bits: Vec<u32> = expected.iter().map(|value| value.to_bits()).collect();

        let mut out = vec![0.0; batch * rows];
        let start = Instant::now();
        multiply(&matrix, &inputs, &mut out, WHOLE_BATCH);
        let one_call = start.elapsed();
        let budget = if quick {
            Duration::from_secs(4)
        } else {
            Duration::from_secs(8)
        };
        let per_round = one_call * group_sizes.len() as u32;
        let rounds = (budget.as_secs_f64() / per_round.as_secs_f64()).clamp(3.0, 200.0) as usize;

        let mut best = vec![f64::INFINITY; group_sizes.len()];
        for _ in 0..rounds {
            for (slot, &group) in best.iter_mut().zip(group_sizes) {
                let start = Instant::now();
                multiply(&matrix, black_box(&inputs), &mut out, group);
                *slot = slot.min(start.elapsed().as_secs_f64() * 1e3);
                black_box(&out);
                assert!(
                    out.iter()
                        .map(|value| value.to_bits())
                        .eq(expected_bits.iter().copied()),
                    "{label} {scheme_name}: groups of {group} changed the result"
                );
            }
        }
        let case = format!("{label} {scheme_name} @{batch}");
        print!("{case:<34}");
        for time in &best {
            print!("{time:>10.2}");
        }
        println!("   ({rounds} rounds)");
    }
}
