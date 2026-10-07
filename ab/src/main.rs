//! A/B/C: one SmolLM-135M token's linears at batch 1 (Q4_32, f16 scales), with
//! `main`, the decode PR, and the working tree linked side by side, timed in
//! interleaved rounds. Also a 1024 × 1024 matrix at batch 1 and 16.

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

macro_rules! model {
    ($krate:ident, $shapes:expr, $layers:expr, $vocab:expr) => {{
        let mut matrices = Vec::new();
        for layer in 0..$layers {
            for (index, &(rows, columns)) in $shapes.iter().enumerate() {
                let weights = values(rows * columns, (layer * 7 + index) as u32 + 1);
                let mut q = $krate::quantize::<f16, 4, 32>(&weights).unwrap();
                q.set_shape(rows, columns).unwrap();
                matrices.push(q);
            }
        }
        let mut lm = $krate::quantize::<f16, 4, 32>(&values($vocab * HIDDEN, 999)).unwrap();
        lm.set_shape($vocab, HIDDEN).unwrap();
        matrices.push(lm);
        matrices
    }};
}

fn main() {
    let layers: usize = std::env::var("LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let vocab: usize = std::env::var("VOCAB").ok().and_then(|s| s.parse().ok()).unwrap_or(49152);
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let shapes = [
        (HIDDEN, HIDDEN),
        (KV, HIDDEN),
        (KV, HIDDEN),
        (HIDDEN, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (INTERMEDIATE, HIDDEN),
        (HIDDEN, INTERMEDIATE),
    ];
    let main_model = model!(quantize_old, shapes, layers, vocab);
    let decode_model = model!(quantize_b, shapes, layers, vocab);
    let tree_model = model!(quantize, shapes, layers, vocab);
    let input_hidden = values(HIDDEN, 4242);
    let input_intermediate = values(INTERMEDIATE, 4343);
    let input_for = |columns: usize| -> &[f32] {
        if columns == HIDDEN { &input_hidden } else { &input_intermediate }
    };

    // Same answers, bit for bit, before timing anything.
    for ((a, b), c) in main_model.iter().zip(&decode_model).zip(&tree_model) {
        let input = input_for(a.shape().unwrap().1);
        let (x, y, z) = (a.matmul(input).unwrap(), b.matmul(input).unwrap(), c.matmul(input).unwrap());
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&x), bits(&y));
        assert_eq!(bits(&x), bits(&z));
    }

    let side = 1024;
    let square = values(side * side, 7);
    let mut square_main = quantize_old::quantize::<f16, 4, 32>(&square).unwrap();
    let mut square_decode = quantize_b::quantize::<f16, 4, 32>(&square).unwrap();
    let mut square_tree = quantize::quantize::<f16, 4, 32>(&square).unwrap();
    square_main.set_shape(side, side).unwrap();
    square_decode.set_shape(side, side).unwrap();
    square_tree.set_shape(side, side).unwrap();
    let one = &square[..side];
    let sixteen = &square[..16 * side];

    let names = ["main", "decode PR", "this tree"];
    let mut token = [Vec::new(), Vec::new(), Vec::new()];
    let mut square_one = [Vec::new(), Vec::new(), Vec::new()];
    let mut square_sixteen = [Vec::new(), Vec::new(), Vec::new()];
    let mut square_decode_all = [Vec::new(), Vec::new(), Vec::new()];
    let mut decoded = vec![0.0f32; side * side];
    let time = |f: &mut dyn FnMut()| {
        let start = Instant::now();
        f();
        start.elapsed().as_secs_f64()
    };
    for round in 0..rounds + 2 {
        let t = [
            time(&mut || for q in &main_model { black_box(q.matmul(input_for(q.shape().unwrap().1)).unwrap()); }),
            time(&mut || for q in &decode_model { black_box(q.matmul(input_for(q.shape().unwrap().1)).unwrap()); }),
            time(&mut || for q in &tree_model { black_box(q.matmul(input_for(q.shape().unwrap().1)).unwrap()); }),
        ];
        let s1 = [
            time(&mut || { for _ in 0..10 { black_box(square_main.matmul(one).unwrap()); } }),
            time(&mut || { for _ in 0..10 { black_box(square_decode.matmul(one).unwrap()); } }),
            time(&mut || { for _ in 0..10 { black_box(square_tree.matmul(one).unwrap()); } }),
        ];
        let s16 = [
            time(&mut || { black_box(square_main.matmul(sixteen).unwrap()); }),
            time(&mut || { black_box(square_decode.matmul(sixteen).unwrap()); }),
            time(&mut || { black_box(square_tree.matmul(sixteen).unwrap()); }),
        ];
        let d = [
            time(&mut || { for _ in 0..10 { square_main.dequantize_into(&mut decoded).unwrap(); black_box(&decoded); } }),
            time(&mut || { for _ in 0..10 { square_decode.dequantize_into(&mut decoded).unwrap(); black_box(&decoded); } }),
            time(&mut || { for _ in 0..10 { square_tree.dequantize_into(&mut decoded).unwrap(); black_box(&decoded); } }),
        ];
        if round >= 2 {
            for i in 0..3 {
                square_decode_all[i].push(d[i] / 10.0);
                token[i].push(t[i]);
                square_one[i].push(s1[i] / 10.0);
                square_sixteen[i].push(s16[i]);
            }
        }
    }
    let min = |v: &Vec<f64>| v.iter().cloned().fold(f64::INFINITY, f64::min);
    let median = |v: &Vec<f64>| {
        let mut v = v.clone();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let per_value = (side * side) as f64;
    println!("{rounds} interleaved rounds, min (median); all three give the same bits");
    println!("{:<11}{:>22}{:>24}{:>24}{:>24}", "", "SmolLM token, ms", "1024² ×1, ns/value", "1024² ×16, ns/value", "dequantize, ns/value");
    for i in 0..3 {
        println!(
            "{:<11}{:>14.2} ({:>6.2}){:>15.3} ({:>6.3}){:>15.3} ({:>6.3}){:>15.3} ({:>6.3})",
            names[i],
            min(&token[i]) * 1e3,
            median(&token[i]) * 1e3,
            min(&square_one[i]) * 1e9 / per_value,
            median(&square_one[i]) * 1e9 / per_value,
            min(&square_sixteen[i]) * 1e9 / per_value,
            median(&square_sixteen[i]) * 1e9 / per_value,
            min(&square_decode_all[i]) * 1e9 / per_value,
            median(&square_decode_all[i]) * 1e9 / per_value,
        );
    }
}
