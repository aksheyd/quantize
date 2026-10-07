//! Times `matmul_into` on one thread with the batch split into groups of
//! vectors, one call per group, for several ways of sizing a group. Every
//! way must give the same bits as one call over the whole batch.
//!
//! Run: `cargo run --release --manifest-path timing/Cargo.toml`

use quantize::{Quantized, Scheme, f16};
use std::hint::black_box;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
enum Rule {
    Vectors(usize),
    Bytes(usize),
    WholeBatch,
}

impl Rule {
    fn label(self) -> String {
        match self {
            Rule::Vectors(count) => format!("{count}"),
            Rule::Bytes(bytes) if bytes >= 1 << 20 => {
                format!("{}M", bytes as f64 / (1 << 20) as f64)
            }
            Rule::Bytes(bytes) => format!("{}K", bytes >> 10),
            Rule::WholeBatch => "one call".to_string(),
        }
    }

    fn vectors_per_group(self, columns: usize) -> usize {
        match self {
            Rule::Vectors(count) => count,
            Rule::Bytes(bytes) => (bytes / (columns * 4)).max(1),
            Rule::WholeBatch => usize::MAX,
        }
    }
}

fn values(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn multiply(matrix: &Quantized<f16>, inputs: &[f32], out: &mut [f32], rule: Rule) {
    let (rows, columns) = matrix.shape().unwrap();
    let group = rule.vectors_per_group(columns);
    if group >= inputs.len() / columns {
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
        "5-bit" => Scheme::Symmetric { bits: 5, block: 32 },
        "asym4" => Scheme::Asymmetric { bits: 4, block: 32 },
        "adaptive" => Scheme::Adaptive {
            block: 32,
            tolerance: 0.1,
        },
        _ => unreachable!("{name}"),
    }
}

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;

fn main() {
    let debug = cfg!(debug_assertions);
    let rules = [
        Rule::Vectors(64),
        Rule::Vectors(128),
        Rule::Vectors(256),
        Rule::Bytes(512 * KIB),
        Rule::Bytes(MIB),
        Rule::Bytes(2 * MIB),
        Rule::Bytes(4 * MIB),
        Rule::WholeBatch,
    ];
    // (label, scheme, rows, columns, batch)
    let cases: &[(&str, &str, usize, usize, usize)] = if debug {
        &[
            ("ViT qkv", "Q4_32", 2304, 768, 6304),
            ("ViT qkv", "Q8_32", 2304, 768, 6304),
            ("4096 square", "Q4_32", 4096, 4096, 512),
            ("1024 square", "Q4_32", 1024, 1024, 384),
            ("1024 square", "Q8_32", 1024, 1024, 384),
            ("1024 square", "asym4", 1024, 1024, 384),
            ("1024 square", "asym4", 1024, 1024, 1024),
            ("768 square", "Q8_32", 768, 768, 394),
        ]
    } else {
        &[
            ("ViT qkv", "Q4_32", 2304, 768, 6304),
            ("ViT MLP out", "Q4_32", 768, 3072, 6304),
            ("4096 square", "Q4_32", 4096, 4096, 512),
            ("4096 square", "Q4_32", 4096, 4096, 2048),
            ("SmolLM down", "Q4_32", 576, 1536, 512),
            ("SmolLM q", "Q4_32", 576, 576, 512),
            ("ViT qkv", "Q8_32", 2304, 768, 6304),
            ("ViT qkv", "asym4", 2304, 768, 6304),
            ("ViT qkv", "adaptive", 2304, 768, 6304),
            ("1024 square", "Q4_32", 1024, 1024, 384),
            ("1024 square", "asym4", 1024, 1024, 384),
            ("1024 square", "asym4", 1024, 1024, 512),
            ("1024 square", "asym4", 1024, 1024, 1024),
            ("1024 square", "5-bit", 1024, 1024, 512),
            ("768 square", "Q4_32", 768, 768, 394),
            ("768 square", "asym4", 768, 768, 394),
            ("768 square", "asym4", 768, 768, 788),
        ]
    };
    print!("{:<30}", "case, ms (best of rounds)");
    for rule in rules {
        print!("{:>9}", rule.label());
    }
    println!();
    for &(label, scheme_name, rows, columns, batch) in cases {
        let weights = values(rows * columns, 1);
        let mut matrix = scheme_named(scheme_name).quantize::<f16>(&weights).unwrap();
        matrix.set_shape(rows, columns).unwrap();
        let inputs = values(batch * columns, 2);
        let mut expected = vec![0.0; batch * rows];
        multiply(&matrix, &inputs, &mut expected, Rule::WholeBatch);
        let expected_bits: Vec<u32> = expected.iter().map(|value| value.to_bits()).collect();

        let mut out = vec![0.0; batch * rows];
        let start = Instant::now();
        multiply(&matrix, &inputs, &mut out, Rule::WholeBatch);
        let one_call = start.elapsed();
        let budget = Duration::from_secs(12);
        let per_round = one_call * rules.len() as u32;
        let rounds = (budget.as_secs_f64() / per_round.as_secs_f64()).clamp(3.0, 300.0) as usize;

        let mut best = vec![f64::INFINITY; rules.len()];
        for _ in 0..rounds {
            for (slot, &rule) in best.iter_mut().zip(&rules) {
                let start = Instant::now();
                multiply(&matrix, black_box(&inputs), &mut out, rule);
                *slot = slot.min(start.elapsed().as_secs_f64() * 1e3);
                black_box(&out);
                assert!(
                    out.iter()
                        .map(|value| value.to_bits())
                        .eq(expected_bits.iter().copied()),
                    "{label} {scheme_name}: groups of {} changed the result",
                    rule.label()
                );
            }
        }
        let case = format!("{label} {scheme_name} @{batch}");
        print!("{case:<30}");
        for time in &best {
            print!("{time:>9.2}");
        }
        println!("   ({rounds} rounds)");
    }
}
