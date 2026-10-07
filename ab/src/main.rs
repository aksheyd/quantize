//! Compare quantize 0.3.1, main, and the fix branch, linked into one binary.
//!
//! `qbench bench [rows] [columns] [rounds]` times the decode operations.
//! `qbench check [cases] [seed]` compares every output bit for bit.
//! `qbench fuzz [cases] [seed]` loads corrupted bytes in every version.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Symmetric,
    Asymmetric,
    Adaptive,
}

#[derive(Clone, Copy, Debug)]
pub enum ScaleKind {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug)]
pub struct Spec {
    pub kind: Kind,
    pub bits: u32,
    pub block: usize,
    pub tolerance: f32,
    pub scale: ScaleKind,
}

pub trait Ops {
    fn matmul_into(&self, inputs: &[f32], out: &mut [f32]) -> Result<(), String>;
    fn matmul(&self, inputs: &[f32]) -> Result<Vec<f32>, String>;
    fn dequantize_into(&self, out: &mut [f32]) -> Result<(), String>;
    fn row_into(&self, row: usize, out: &mut [f32]) -> Result<(), String>;
    fn dot(&self, rhs: &[f32]) -> Result<f32, String>;
    fn to_bytes(&self) -> Vec<u8>;
    fn shape(&self) -> Option<(usize, usize)>;
    fn len(&self) -> usize;
}

macro_rules! each {
    ($value:expr, $t:ident => $body:expr) => {
        match $value {
            Tensor::F32($t) => $body,
            Tensor::F16($t) => $body,
            Tensor::Bf16($t) => $body,
        }
    };
}

macro_rules! version {
    ($module:ident, $krate:ident) => {
        pub mod $module {
            use crate::{Kind, Ops, ScaleKind, Spec};
            use $krate as q;
            use q::{bf16, f16, Quantized};

            pub enum Tensor {
                F32(Quantized<f32>),
                F16(Quantized<f16>),
                Bf16(Quantized<bf16>),
            }

            fn make<S: q::Scale>(spec: &Spec, values: &[f32]) -> Result<Quantized<S>, String> {
                let made = match spec.kind {
                    Kind::Symmetric => q::symmetric::quantize_with::<S>(values, spec.bits, spec.block),
                    Kind::Asymmetric => {
                        q::asymmetric::quantize_with::<S>(values, spec.bits, spec.block)
                    }
                    Kind::Adaptive => {
                        q::adaptive::quantize_with::<S>(values, spec.block, spec.tolerance)
                    }
                };
                made.map_err(|e| e.to_string())
            }

            pub fn build(
                spec: &Spec,
                values: &[f32],
                rows: usize,
                columns: usize,
            ) -> Result<Box<dyn Ops>, String> {
                let mut tensor = match spec.scale {
                    ScaleKind::F32 => Tensor::F32(make(spec, values)?),
                    ScaleKind::F16 => Tensor::F16(make(spec, values)?),
                    ScaleKind::Bf16 => Tensor::Bf16(make(spec, values)?),
                };
                each!(&mut tensor, t => t.set_shape(rows, columns)).map_err(|e| e.to_string())?;
                Ok(Box::new(tensor))
            }

            pub fn load(scale: ScaleKind, bytes: &[u8]) -> Result<Box<dyn Ops>, String> {
                let tensor = match scale {
                    ScaleKind::F32 => Tensor::F32(Quantized::from_bytes(bytes).map_err(|e| e.to_string())?),
                    ScaleKind::F16 => Tensor::F16(Quantized::from_bytes(bytes).map_err(|e| e.to_string())?),
                    ScaleKind::Bf16 => Tensor::Bf16(Quantized::from_bytes(bytes).map_err(|e| e.to_string())?),
                };
                Ok(Box::new(tensor))
            }

            impl Ops for Tensor {
                fn matmul_into(&self, inputs: &[f32], out: &mut [f32]) -> Result<(), String> {
                    each!(self, t => t.matmul_into(inputs, out)).map_err(|e| e.to_string())
                }
                fn matmul(&self, inputs: &[f32]) -> Result<Vec<f32>, String> {
                    each!(self, t => t.matmul(inputs)).map_err(|e| e.to_string())
                }
                fn dequantize_into(&self, out: &mut [f32]) -> Result<(), String> {
                    each!(self, t => t.dequantize_into(out)).map_err(|e| e.to_string())
                }
                fn row_into(&self, row: usize, out: &mut [f32]) -> Result<(), String> {
                    each!(self, t => t.dequantize_row_into(row, out)).map_err(|e| e.to_string())
                }
                fn dot(&self, rhs: &[f32]) -> Result<f32, String> {
                    each!(self, t => t.dot(rhs)).map_err(|e| e.to_string())
                }
                fn to_bytes(&self) -> Vec<u8> {
                    each!(self, t => t.to_bytes())
                }
                fn shape(&self) -> Option<(usize, usize)> {
                    each!(self, t => t.shape())
                }
                fn len(&self) -> usize {
                    each!(self, t => t.len())
                }
            }
        }
    };
}

version!(v031, q031);
version!(vmain, qmain);
version!(vfix, qfix);
version!(vsame, qsame);
version!(vh1, qh1);
version!(vh3dot, qh3dot);
version!(va4, qa4);

type Builder = fn(&Spec, &[f32], usize, usize) -> Result<Box<dyn Ops>, String>;
type Loader = fn(ScaleKind, &[u8]) -> Result<Box<dyn Ops>, String>;

const VERSIONS: [(&str, Builder, Loader); 7] = [
    ("0.3.1", v031::build, v031::load),
    ("main", vmain::build, vmain::load),
    ("fix", vfix::build, vfix::load),
    ("main again", vsame::build, vsame::load),
    ("H1", vh1::build, vh1::load),
    ("H3+dot", vh3dot::build, vh3dot::load),
    ("A4", va4::build, va4::load),
];

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

fn spec_name(spec: &Spec) -> String {
    format!(
        "{:?} bits={} block={} scale={:?}",
        spec.kind, spec.bits, spec.block, spec.scale
    )
}

fn time_once(op: &mut dyn FnMut()) -> f64 {
    let start = Instant::now();
    op();
    start.elapsed().as_secs_f64() * 1e3
}

fn summarize(mut times: Vec<f64>) -> (f64, f64) {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[0], times[times.len() / 2])
}

fn bench(args: &[String]) {
    let rows: usize = args.first().map_or(2048, |a| a.parse().unwrap());
    let columns: usize = args.get(1).map_or(2048, |a| a.parse().unwrap());
    let rounds: usize = args.get(2).map_or(5, |a| a.parse().unwrap());
    let only: Option<&str> = args.get(3).map(|s| s.as_str());
    let mut random = Random(7);
    let values: Vec<f32> = (0..rows * columns).map(|_| random.normal() * 0.02).collect();
    let input: Vec<f32> = (0..4 * columns).map(|_| random.normal()).collect();
    let flat_rhs: Vec<f32> = (0..rows * columns).map(|_| random.normal()).collect();
    let scale = match std::env::var("SCALE").as_deref() {
        Ok("f32") => ScaleKind::F32,
        Ok("bf16") => ScaleKind::Bf16,
        _ => ScaleKind::F16,
    };
    println!("{rows} x {columns}, {rounds} rounds, {scale:?} scales, ms as best / median");
    let schemes = [
        ("Q4_32", Spec { kind: Kind::Symmetric, bits: 4, block: 32, tolerance: 0.0, scale }),
        ("Q8_32", Spec { kind: Kind::Symmetric, bits: 8, block: 32, tolerance: 0.0, scale }),
    ];
    for (scheme_name, spec) in schemes {
        let tensors: Vec<(&str, Box<dyn Ops>)> = VERSIONS
            .iter()
            .map(|(name, build, _)| (*name, build(&spec, &values, rows, columns).unwrap()))
            .collect();
        let operations: [&str; 5] = ["matmul x1", "matmul x4", "dequantize_into", "every row", "dot"];
        for operation in operations {
            if let Some(only) = only {
                if !operation.starts_with(only) {
                    continue;
                }
            }
            let mut times = vec![Vec::new(); tensors.len()];
            let mut out = vec![0.0; rows * columns.max(4)];
            for _ in 0..rounds {
                for (index, (_, tensor)) in tensors.iter().enumerate() {
                    let mut op = || match operation {
                        "matmul x1" => tensor.matmul_into(&input[..columns], &mut out[..rows]).unwrap(),
                        "matmul x4" => tensor.matmul_into(&input[..4 * columns], &mut out[..4 * rows]).unwrap(),
                        "dequantize_into" => tensor.dequantize_into(&mut out[..rows * columns]).unwrap(),
                        "every row" => {
                            for row in 0..rows {
                                tensor.row_into(row, &mut out[..columns]).unwrap();
                            }
                        }
                        _ => {
                            std::hint::black_box(tensor.dot(&flat_rhs).unwrap());
                        }
                    };
                    times[index].push(time_once(&mut op));
                }
            }
            let cells: Vec<String> = tensors
                .iter()
                .zip(times)
                .map(|((name, _), times)| {
                    let (best, median) = summarize(times);
                    format!("{name} {best:.3} / {median:.3}")
                })
                .collect();
            println!("{scheme_name} {operation:16} {}", cells.join("   "));
        }
    }
}

fn random_spec(random: &mut Random) -> Spec {
    let kind = random.pick(&[
        Kind::Symmetric,
        Kind::Symmetric,
        Kind::Symmetric,
        Kind::Asymmetric,
        Kind::Adaptive,
    ]);
    let bits = random.pick(&[4, 4, 4, 8, 8, 8, 2, 3, 5, 6, 7, 12, 16]);
    let block = random.pick(&[
        1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 32, 32, 33, 48, 63, 64, 64, 65, 96, 128, 128, 160, 256,
        512, 1024,
    ]);
    let scale = random.pick(&[ScaleKind::F32, ScaleKind::F16, ScaleKind::Bf16]);
    let tolerance = random.pick(&[0.002, 0.01, 0.05, 0.2]);
    Spec { kind, bits, block, tolerance, scale }
}

fn random_shape(random: &mut Random, block: usize) -> (usize, usize) {
    let rows = 1 + random.below(6);
    let columns = match random.below(4) {
        0 => block * (1 + random.below(140)),
        1 => block * (1 + random.below(8)),
        2 => 1 + random.below(700),
        _ => 32 * (1 + random.below(150)),
    };
    let columns = columns.min(9000);
    (rows, columns)
}

fn random_weight(random: &mut Random, style: usize) -> f32 {
    match style {
        0 => random.normal() * 0.02,
        1 => random.normal() + 3.0,
        2 => random.unit() * 10.0 - 1.0,
        _ => match random.below(200) {
            0 => f32::NAN,
            1 => 0.0,
            2 => -0.0,
            3 => 1e-40,
            _ => random.normal(),
        },
    }
}

fn random_input(random: &mut Random) -> f32 {
    match random.below(300) {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => f32::NEG_INFINITY,
        3 => 0.0,
        4 => -0.0,
        5 => 1e30,
        6 => -1e30,
        7 => 1e-40,
        _ => random.normal(),
    }
}

#[derive(Default)]
struct Tally {
    fused: usize,
    multi_run_tensor: usize,
    multi_run_row: usize,
    cases: usize,
    compared: usize,
    exact_mismatch: usize,
    nan_only_mismatch: usize,
    errors: usize,
}

fn bits_equal(a: &[f32], b: &[f32]) -> (bool, bool) {
    let exact = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let loose = a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
    (exact, loose)
}

/// Every output of one tensor, flattened, plus a label for each section.
fn outputs(tensor: &dyn Ops, inputs: &[f32], rhs: &[f32]) -> Vec<(String, Result<Vec<f32>, String>)> {
    let mut results = Vec::new();
    let len = tensor.len();
    let (rows, columns) = tensor.shape().unwrap_or((1, len.max(1)));
    let mut whole = vec![0.0; len];
    results.push(("dequantize".into(), tensor.dequantize_into(&mut whole).map(|_| whole.clone())));
    let mut every_row = Vec::new();
    let mut row_error = None;
    let mut row = vec![0.0; columns];
    for index in 0..rows {
        match tensor.row_into(index, &mut row) {
            Ok(()) => every_row.extend_from_slice(&row),
            Err(error) => {
                row_error = Some(error);
                break;
            }
        }
    }
    results.push(("rows".into(), row_error.map_or(Ok(every_row), Err)));
    for batch in 1..=3 {
        let batch_inputs = &inputs[..batch * columns];
        let mut out = vec![0.0; batch * rows];
        results.push((
            format!("matmul_into x{batch}"),
            tensor.matmul_into(batch_inputs, &mut out).map(|_| out.clone()),
        ));
        results.push((format!("matmul x{batch}"), tensor.matmul(batch_inputs)));
    }
    results.push(("dot".into(), tensor.dot(&rhs[..len]).map(|value| vec![value])));
    results
}

fn compare(
    label: &str,
    spec_text: &str,
    tensors: &[(&str, Box<dyn Ops>)],
    inputs: &[f32],
    rhs: &[f32],
    tally: &mut Tally,
) {
    let all: Vec<_> = tensors
        .iter()
        .map(|(name, tensor)| {
            let got = catch_unwind(AssertUnwindSafe(|| outputs(tensor.as_ref(), inputs, rhs)));
            (*name, got)
        })
        .collect();
    let reference_index: usize = std::env::var("REFERENCE").ok().map_or(1, |r| r.parse().unwrap());
    let reference = match &all[reference_index].1 {
        Ok(reference) => reference,
        Err(_) => {
            for (name, got) in &all {
                if got.is_ok() {
                    println!("PANIC MISMATCH {label} {spec_text}: main panicked, {name} didn't");
                    tally.exact_mismatch += 1;
                }
            }
            return;
        }
    };
    for (name, got) in &all {
        let Ok(got) = got else {
            println!("PANIC {label} {spec_text}: {name} panicked, main didn't");
            tally.exact_mismatch += 1;
            continue;
        };
        for ((section, expected), (_, value)) in reference.iter().zip(got) {
            if section == "dot" && std::env::var("SKIP_DOT").is_ok() {
                continue;
            }
            tally.compared += 1;
            match (expected, value) {
                (Ok(expected), Ok(value)) => {
                    let (exact, loose) = bits_equal(expected, value);
                    if !exact {
                        if loose {
                            tally.nan_only_mismatch += 1;
                        } else {
                            tally.exact_mismatch += 1;
                        }
                        println!(
                            "MISMATCH {label} {spec_text} {section}: {name} vs main{}",
                            if loose { " (NaN bits only)" } else { "" }
                        );
                    }
                }
                (Err(expected), Err(value)) if expected == value => {}
                (Err(expected), Err(value)) if *name == "0.3.1" => {
                    let _ = (expected, value);
                }
                (expected, value) => {
                    tally.errors += 1;
                    println!(
                        "RESULT MISMATCH {label} {spec_text} {section}: main {:?} vs {name} {:?}",
                        expected.as_ref().map(|v| v.len()),
                        value.as_ref().map(|v| v.len())
                    );
                }
            }
        }
    }
}

fn check(args: &[String]) {
    let cases: usize = args.first().map_or(2000, |a| a.parse().unwrap());
    let seed: u64 = args.get(1).map_or(1, |a| a.parse().unwrap());
    let mut random = Random(seed);
    let mut tally = Tally::default();
    for case in 0..cases {
        let spec = random_spec(&mut random);
        let (rows, columns) = random_shape(&mut random, spec.block);
        let style = random.below(4);
        let values: Vec<f32> = (0..rows * columns).map(|_| random_weight(&mut random, style)).collect();
        let inputs: Vec<f32> = (0..3 * columns).map(|_| random_input(&mut random)).collect();
        let rhs: Vec<f32> = (0..rows * columns).map(|_| random_input(&mut random)).collect();
        let built: Vec<_> = VERSIONS
            .iter()
            .map(|(name, build, _)| (*name, build(&spec, &values, rows, columns)))
            .collect();
        if built.iter().any(|(_, tensor)| tensor.is_err()) {
            let errors: Vec<_> = built.iter().map(|(_, t)| t.as_ref().err().cloned()).collect();
            if errors.iter().any(|e| e != &errors[1]) {
                println!("BUILD MISMATCH {case} {}: {errors:?}", spec_name(&spec));
                tally.errors += 1;
            }
            continue;
        }
        let tensors: Vec<(&str, Box<dyn Ops>)> =
            built.into_iter().map(|(name, tensor)| (name, tensor.unwrap())).collect();
        let bytes: Vec<Vec<u8>> = tensors.iter().map(|(_, tensor)| tensor.to_bytes()).collect();
        if bytes.iter().any(|b| b != &bytes[1]) {
            println!("BYTES MISMATCH {case} {}", spec_name(&spec));
            tally.exact_mismatch += 1;
        }
        tally.cases += 1;
        if matches!(spec.kind, Kind::Symmetric) && matches!(spec.bits, 4 | 8) {
            let len = rows * columns;
            let rows_ok = columns % spec.block == 0 && (columns * spec.bits as usize) % 8 == 0;
            if rows_ok && spec.block % 32 == 0 {
                tally.fused += 1;
            }
            if len.div_ceil(spec.block) > 64 {
                tally.multi_run_tensor += 1;
            }
            if rows_ok && columns / spec.block > 64 {
                tally.multi_run_row += 1;
            }
        }
        let text = format!("{} {rows}x{columns}", spec_name(&spec));
        compare(&format!("case {case}"), &text, &tensors, &inputs, &rhs, &mut tally);
    }
    println!(
        "check: {} tensors ({} take the single-vector path, {} decode in several runs, {} decode rows in several runs), {} comparisons, {} exact mismatches, {} NaN-bit-only mismatches, {} result mismatches",
        tally.cases, tally.fused, tally.multi_run_tensor, tally.multi_run_row, tally.compared, tally.exact_mismatch, tally.nan_only_mismatch, tally.errors
    );
}

fn fuzz(args: &[String]) {
    let cases: usize = args.first().map_or(2000, |a| a.parse().unwrap());
    let seed: u64 = args.get(1).map_or(1, |a| a.parse().unwrap());
    let mut random = Random(seed);
    let mut tally = Tally::default();
    let mut loaded = 0;
    std::panic::set_hook(Box::new(|_| {}));
    for case in 0..cases {
        let spec = random_spec(&mut random);
        let (rows, columns) = random_shape(&mut random, spec.block);
        let (rows, columns) = (rows.min(3), columns.min(600));
        let values: Vec<f32> = (0..rows * columns).map(|_| random.normal()).collect();
        let Ok(tensor) = vmain::build(&spec, &values, rows, columns) else {
            continue;
        };
        let mut bytes = tensor.to_bytes();
        for _ in 0..1 + random.below(4) {
            let at = random.below(bytes.len());
            match random.below(3) {
                0 => bytes[at] = random.next() as u8,
                1 => bytes[at] ^= 1 << random.below(8),
                _ => bytes.truncate(at.max(1)),
            }
        }
        let loads: Vec<_> = VERSIONS
            .iter()
            .map(|(name, _, load)| (*name, load(spec.scale, &bytes)))
            .collect();
        let errors: Vec<_> = loads.iter().map(|(_, t)| t.as_ref().err().cloned()).collect();
        if errors[1] != errors[2] {
            println!("LOAD MISMATCH {case}: {errors:?}");
            tally.errors += 1;
            continue;
        }
        if loads.iter().any(|(_, t)| t.is_err()) {
            continue;
        }
        loaded += 1;
        let tensors: Vec<(&str, Box<dyn Ops>)> =
            loads.into_iter().map(|(name, tensor)| (name, tensor.unwrap())).collect();
        let len = tensors[1].1.len();
        let columns = tensors[1].1.shape().map_or(len.max(1), |(_, c)| c);
        if len > 1 << 22 || columns > 1 << 20 {
            continue;
        }
        let inputs: Vec<f32> = (0..3 * columns).map(|_| random.normal()).collect();
        let rhs: Vec<f32> = (0..len).map(|_| random.normal()).collect();
        tally.cases += 1;
        compare(&format!("fuzz {case}"), &spec_name(&spec), &tensors, &inputs, &rhs, &mut tally);
    }
    println!(
        "fuzz: {loaded} loaded, {} compared tensors, {} comparisons, {} exact mismatches, {} NaN-bit-only, {} result mismatches",
        tally.cases, tally.compared, tally.exact_mismatch, tally.nan_only_mismatch, tally.errors
    );
}

/// One token through TinyLlama-shaped linears at batch 1: per layer q, k, v,
/// o, then gate, up, and down, at hidden size 2048, 256 for keys and values,
/// and 5632 in the MLP.
fn token(args: &[String]) {
    let layers: usize = args.first().map_or(2, |a| a.parse().unwrap());
    let rounds: usize = args.get(1).map_or(3, |a| a.parse().unwrap());
    let batch: usize = args.get(2).map_or(1, |a| a.parse().unwrap());
    let scale = match std::env::var("SCALE").as_deref() {
        Ok("f32") => ScaleKind::F32,
        Ok("bf16") => ScaleKind::Bf16,
        _ => ScaleKind::F16,
    };
    let shapes = [
        (2048, 2048),
        (256, 2048),
        (256, 2048),
        (2048, 2048),
        (5632, 2048),
        (5632, 2048),
        (2048, 5632),
    ];
    let mut random = Random(11);
    for bits in [4, 8] {
        let spec = Spec { kind: Kind::Symmetric, bits, block: 32, tolerance: 0.0, scale };
        let mut per_version: Vec<Vec<Box<dyn Ops>>> = VERSIONS.iter().map(|_| Vec::new()).collect();
        for _ in 0..layers {
            for &(rows, columns) in &shapes {
                let values: Vec<f32> = (0..rows * columns).map(|_| random.normal() * 0.02).collect();
                for (index, (_, build, _)) in VERSIONS.iter().enumerate() {
                    per_version[index].push(build(&spec, &values, rows, columns).unwrap());
                }
            }
        }
        let input: Vec<f32> = (0..batch * 5632).map(|_| random.normal()).collect();
        let mut out = vec![0.0; batch * 5632];
        let mut times = vec![Vec::new(); VERSIONS.len()];
        for _ in 0..rounds {
            for (index, linears) in per_version.iter().enumerate() {
                let mut op = || {
                    for linear in linears {
                        let (rows, columns) = linear.shape().unwrap();
                        linear.matmul_into(&input[..batch * columns], &mut out[..batch * rows]).unwrap();
                    }
                };
                times[index].push(time_once(&mut op));
            }
        }
        let cells: Vec<String> = VERSIONS
            .iter()
            .zip(times)
            .map(|((name, _, _), times)| {
                let (best, median) = summarize(times);
                format!("{name} {best:.1} / {median:.1}")
            })
            .collect();
        println!("Q{bits}_32 {scale:?}, {layers} layers, batch {batch}, ms best / median: {}", cells.join("   "));
    }
}

/// One vector through a matrix in every layout, with `main`'s code.
fn layouts(args: &[String]) {
    let rows: usize = args.first().map_or(4096, |a| a.parse().unwrap());
    let columns: usize = args.get(1).map_or(4096, |a| a.parse().unwrap());
    let rounds: usize = args.get(2).map_or(9, |a| a.parse().unwrap());
    let mut random = Random(5);
    let values: Vec<f32> = (0..rows * columns).map(|_| random.normal() * 0.02).collect();
    let input: Vec<f32> = (0..columns).map(|_| random.normal()).collect();
    let mut out = vec![0.0; rows];
    let symmetric = |bits, block| Spec { kind: Kind::Symmetric, bits, block, tolerance: 0.0, scale: ScaleKind::F16 };
    let asymmetric = |bits, block| Spec { kind: Kind::Asymmetric, bits, block, tolerance: 0.0, scale: ScaleKind::F16 };
    let specs = [
        ("symmetric 4-bit, blocks of 32", symmetric(4, 32)),
        ("symmetric 8-bit, blocks of 32", symmetric(8, 32)),
        ("symmetric 4-bit, blocks of 128", symmetric(4, 128)),
        ("symmetric 4-bit, blocks of 16", symmetric(4, 16)),
        ("symmetric 8-bit, blocks of 16", symmetric(8, 16)),
        ("symmetric 3-bit, blocks of 32", symmetric(3, 32)),
        ("symmetric 5-bit, blocks of 32", symmetric(5, 32)),
        ("asymmetric 4-bit, blocks of 32", asymmetric(4, 32)),
        ("asymmetric 8-bit, blocks of 32", asymmetric(8, 32)),
        ("adaptive, tolerance 0.002", Spec { kind: Kind::Adaptive, bits: 0, block: 32, tolerance: 0.002, scale: ScaleKind::F16 }),
    ];
    for (name, spec) in specs {
        let tensors: Vec<(&str, Box<dyn Ops>)> = VERSIONS[1..3]
            .iter()
            .map(|(version, build, _)| (*version, build(&spec, &values, rows, columns).unwrap()))
            .collect();
        let mut times = vec![Vec::new(); tensors.len()];
        for _ in 0..rounds {
            for (index, (_, tensor)) in tensors.iter().enumerate() {
                times[index].push(time_once(&mut || tensor.matmul_into(&input, &mut out).unwrap()));
            }
        }
        let cells: Vec<String> = tensors
            .iter()
            .zip(times)
            .map(|((version, _), times)| format!("{version} {:.2}", summarize(times).1))
            .collect();
        println!("{name:32} {}", cells.join("   "));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("layouts") => layouts(&args[1..]),
        Some("dotprec") => dot_precision(&args[1..]),
        Some("dotcheck") => dot_check(&args[1..]),
        Some("check") => check(&args[1..]),
        Some("fuzz") => fuzz(&args[1..]),
        Some("token") => token(&args[1..]),
        _ => bench(args.get(1..).unwrap_or(&[])),
    }
}

/// Relative error of `dot` and of a one-row `matmul` against f64, and how long
/// `dot` takes, over one long vector.
pub fn dot_precision(args: &[String]) {
    let len: usize = args.first().map_or(1 << 24, |a| a.parse().unwrap());
    let rounds: usize = args.get(1).map_or(5, |a| a.parse().unwrap());
    let mut random = Random(9);
    let cases: [(&str, Vec<f32>, Vec<f32>); 2] = [
        (
            "every product positive",
            (0..len).map(|i| (i as f32 * 0.37).sin().abs()).collect(),
            vec![1.0; len],
        ),
        (
            "normal values",
            (0..len).map(|_| random.normal() * 0.02).collect(),
            (0..len).map(|_| random.normal()).collect(),
        ),
    ];
    for (case, values, rhs) in &cases {
        for bits in [4, 8] {
            let spec = Spec { kind: Kind::Symmetric, bits, block: 32, tolerance: 0.0, scale: ScaleKind::F16 };
            let mut cells = Vec::new();
            for (name, build, _) in &VERSIONS[1..3] {
                let tensor = build(&spec, values, 1, len).unwrap();
                let mut decoded = vec![0.0; len];
                tensor.dequantize_into(&mut decoded).unwrap();
                let exact: f64 = decoded.iter().zip(rhs).map(|(&w, &x)| w as f64 * x as f64).sum();
                let dot = tensor.dot(rhs).unwrap() as f64;
                let matmul = tensor.matmul(rhs).unwrap()[0] as f64;
                let times: Vec<f64> = (0..rounds)
                    .map(|_| time_once(&mut || {
                        std::hint::black_box(tensor.dot(rhs).unwrap());
                    }))
                    .collect();
                let matmul_times: Vec<f64> = (0..rounds)
                    .map(|_| time_once(&mut || {
                        std::hint::black_box(tensor.matmul(rhs).unwrap());
                    }))
                    .collect();
                cells.push(format!(
                    "{name}: dot error {:.1e} in {:.2} ms, matmul error {:.1e} in {:.2} ms",
                    ((dot - exact) / exact).abs(),
                    summarize(times).1,
                    ((matmul - exact) / exact).abs(),
                    summarize(matmul_times).1
                ));
            }
            println!("Q{bits}_32, {len} values, {case}: {}", cells.join("; "));
        }
    }
}

/// `dot` on random flat tensors, against f64: the worst error of `main` and of
/// the fix, as a fraction of the sum of the products' sizes.
pub fn dot_check(args: &[String]) {
    let cases: usize = args.first().map_or(2000, |a| a.parse().unwrap());
    let seed: u64 = args.get(1).map_or(1, |a| a.parse().unwrap());
    let mut random = Random(seed);
    let (mut fused, mut compared, mut nan_differs) = (0, 0, 0);
    let (mut worst_main, mut worst_fix) = (0.0_f64, 0.0_f64);
    for _ in 0..cases {
        let mut spec = random_spec(&mut random);
        if random.below(2) == 0 {
            spec.kind = Kind::Symmetric;
            spec.bits = random.pick(&[4, 8]);
            spec.block = random.pick(&[32, 64, 96, 128, 256]);
        }
        let blocks = 1 + random.below(300);
        let len = if random.below(4) == 0 { 1 + random.below(20000) } else { blocks * spec.block };
        let style = random.below(4);
        let values: Vec<f32> = (0..len).map(|_| random_weight(&mut random, style)).collect();
        let specials = random.below(3) == 0;
        let rhs: Vec<f32> = (0..len)
            .map(|_| if specials { random_input(&mut random) } else { random.normal() })
            .collect();
        let (Ok(main), Ok(fix)) = (vmain::build(&spec, &values, 1, len), vfix::build(&spec, &values, 1, len)) else {
            continue;
        };
        compared += 1;
        if matches!(spec.kind, Kind::Symmetric) && matches!(spec.bits, 4 | 8) && spec.block % 32 == 0 && len % spec.block == 0 {
            fused += 1;
        }
        let mut decoded = vec![0.0; len];
        main.dequantize_into(&mut decoded).unwrap();
        let exact: f64 = decoded.iter().zip(&rhs).map(|(&w, &x)| w as f64 * x as f64).sum();
        let size: f64 = decoded.iter().zip(&rhs).map(|(&w, &x)| (w as f64 * x as f64).abs()).sum();
        let (a, b) = (main.dot(&rhs).unwrap(), fix.dot(&rhs).unwrap());
        if a.is_nan() != b.is_nan() || (a.is_infinite() && a != b) || (b.is_infinite() && a != b) {
            nan_differs += 1;
            println!("NAN OR INFINITY DIFFERS {}: main {a} fix {b}", spec_name(&spec));
            continue;
        }
        if a.is_finite() && size > 0.0 && size.is_finite() {
            worst_main = worst_main.max((a as f64 - exact).abs() / size);
            worst_fix = worst_fix.max((b as f64 - exact).abs() / size);
        }
    }
    println!("dot: {compared} tensors ({fused} take the fused path), worst error main {worst_main:.1e}, fix {worst_fix:.1e}, NaN or infinity differs in {nan_differs}");
}
