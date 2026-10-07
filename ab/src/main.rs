//! One SmolLM-135M token's linears at batch 1 (Q8_32 by default, f16 scales),
//! with `main`, the inline branch, and the fused branch linked side by side
//! and timed in interleaved rounds. Also a 1024 x 1024 matrix at batch 1 and
//! 16, and `dequantize_into` on it.

use half::f16;
use std::hint::black_box;
use std::time::Instant;

const HIDDEN: usize = 576;
const KV: usize = 192;
const INTERMEDIATE: usize = 1536;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut seed = seed;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed as f32 / u32::MAX as f32) * 2.0 - 1.0) * 0.05
        })
        .collect()
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

macro_rules! variant {
    ($krate:ident, $bits:expr, $layers:expr, $vocab:expr, $shapes:expr, $square:expr, $side:expr) => {{
        let scheme = $krate::Scheme::Symmetric { bits: $bits, block: 32 };
        let mut matrices = Vec::new();
        for layer in 0..$layers {
            for (index, &(rows, columns)) in $shapes.iter().enumerate() {
                let weights = values(rows * columns, (layer * 7 + index) as u32 + 1);
                let mut q = scheme.quantize::<f16>(&weights).unwrap();
                q.set_shape(rows, columns).unwrap();
                matrices.push(q);
            }
        }
        let mut lm = scheme.quantize::<f16>(&values($vocab * HIDDEN, 999)).unwrap();
        lm.set_shape($vocab, HIDDEN).unwrap();
        matrices.push(lm);
        let mut square = scheme.quantize::<f16>($square).unwrap();
        square.set_shape($side, $side).unwrap();
        (matrices, square)
    }};
}

#[derive(Default)]
struct Times {
    token: Vec<f64>,
    one: Vec<f64>,
    sixteen: Vec<f64>,
    dequantize: Vec<f64>,
}

fn time(mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64()
}

macro_rules! measure_round {
    ($model:expr, $square:expr, $input_for:expr, $one:expr, $sixteen:expr, $decoded:expr, $times:expr, $keep:expr) => {{
        let (model, square) = (&$model, &$square);
        let token = time(|| {
            for q in model.iter() {
                black_box(q.matmul($input_for(q.shape().unwrap().1)).unwrap());
            }
        });
        let one = time(|| {
            for _ in 0..10 {
                black_box(square.matmul($one).unwrap());
            }
        }) / 10.0;
        let sixteen = time(|| {
            black_box(square.matmul($sixteen).unwrap());
        });
        let dequantize = time(|| {
            for _ in 0..10 {
                square.dequantize_into(&mut $decoded).unwrap();
                black_box(&$decoded);
            }
        }) / 10.0;
        if $keep {
            $times.token.push(token);
            $times.one.push(one);
            $times.sixteen.push(sixteen);
            $times.dequantize.push(dequantize);
        }
    }};
}

fn main() {
    let bits = env("BITS", 8) as u32;
    let layers = env("LAYERS", 30);
    let vocab = env("VOCAB", 49152);
    let rounds = env("ROUNDS", 15);
    let shapes = [
        (HIDDEN, HIDDEN),
        (KV, HIDDEN),
        (KV, HIDDEN),
        (HIDDEN, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (HIDDEN, INTERMEDIATE),
    ];
    let side = 1024;
    let square_values = values(side * side, 7);
    let (main_model, main_square) =
        variant!(q_main, bits, layers, vocab, shapes, &square_values, side);
    let (inline_model, inline_square) =
        variant!(q_inline, bits, layers, vocab, shapes, &square_values, side);
    let (fused_model, fused_square) =
        variant!(q_fused, bits, layers, vocab, shapes, &square_values, side);
    let input_hidden = values(HIDDEN, 4242);
    let input_intermediate = values(INTERMEDIATE, 4343);
    let input_for = |columns: usize| -> &[f32] {
        if columns == HIDDEN { &input_hidden } else { &input_intermediate }
    };
    let one = &square_values[..side];
    let sixteen = &square_values[..16 * side];

    // Same answers, bit for bit, before timing anything.
    let bits_of = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
    for ((a, b), c) in main_model.iter().zip(&inline_model).zip(&fused_model) {
        let input = input_for(a.shape().unwrap().1);
        let expected = bits_of(&a.matmul(input).unwrap());
        assert_eq!(expected, bits_of(&b.matmul(input).unwrap()));
        assert_eq!(expected, bits_of(&c.matmul(input).unwrap()));
    }
    for batch in [one, sixteen] {
        let expected = bits_of(&main_square.matmul(batch).unwrap());
        assert_eq!(expected, bits_of(&inline_square.matmul(batch).unwrap()));
        assert_eq!(expected, bits_of(&fused_square.matmul(batch).unwrap()));
    }
    let expected = bits_of(&main_square.dequantize());
    assert_eq!(expected, bits_of(&inline_square.dequantize()));
    assert_eq!(expected, bits_of(&fused_square.dequantize()));

    let names = ["main", "inline", "fused"];
    let mut times: [Times; 3] = Default::default();
    let mut decoded = vec![0.0f32; side * side];
    for round in 0..rounds + 2 {
        let keep = round >= 2;
        measure_round!(main_model, main_square, input_for, one, sixteen, decoded, times[0], keep);
        measure_round!(inline_model, inline_square, input_for, one, sixteen, decoded, times[1], keep);
        measure_round!(fused_model, fused_square, input_for, one, sixteen, decoded, times[2], keep);
    }
    let min = |v: &Vec<f64>| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let median = |v: &Vec<f64>| {
        let mut v = v.clone();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let per_value = (side * side) as f64;
    println!(
        "{bits}-bit, blocks of 32, f16 scales; {rounds} interleaved rounds, min (median); all give the same bits"
    );
    println!(
        "{:<8}{:>22}{:>24}{:>24}{:>24}",
        "", "SmolLM token, ms", "1024² ×1, ns/value", "1024² ×16, ns/value", "dequantize, ns/value"
    );
    for (name, t) in names.iter().zip(&times) {
        println!(
            "{:<8}{:>14.2} ({:>6.2}){:>15.3} ({:>6.3}){:>15.3} ({:>6.3}){:>15.3} ({:>6.3})",
            name,
            min(&t.token) * 1e3,
            median(&t.token) * 1e3,
            min(&t.one) * 1e9 / per_value,
            median(&t.one) * 1e9 / per_value,
            min(&t.sixteen) * 1e9 / per_value,
            median(&t.sixteen) * 1e9 / per_value,
            min(&t.dequantize) * 1e9 / per_value,
            median(&t.dequantize) * 1e9 / per_value,
        );
    }
}
