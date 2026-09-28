# quantize

a simple, fast quantization library usable as a [rust crate](https://crates.io/crates/quantize) or [python package](https://pypi.org/project/quantize-py/).

quantization stores numbers in fewer bits, trading a little accuracy for a lot less memory.

## learn

feel free to peruse the [chapters](https://github.com/aksheyd/quantize/tree/main/chapters), which derive quantization from first principles, one step at a time. run them in order from a clone of the repo:

```
git clone https://github.com/aksheyd/quantize
cd quantize
cargo run --release --example ch01_simple
```

then `ch02_naive`, `ch03_bits`, `ch04_block`, `ch05_asymmetric`, `ch06_adaptive`, `ch07_learned`, and `ch08_alternating`.

requires rust 1.88 or newer. with rustup, the first build installs the repo's pinned toolchain automatically.

## use as a library

```
cargo add quantize
```

requires rust 1.88 or newer. older toolchains may get version 0.1.0 instead, which has a different api.

```rust
use quantize::{f16, quantize};

let weights = [0.42_f32, -0.10, 0.70, -0.50];

// f16 scales, 8-bit codes, blocks of 32 values
let q = quantize::<f16, 8, 32>(&weights).unwrap();

let back = q.dequantize(); // [0.421, -0.098, 0.700, -0.498]
let dot = q.dot(&weights).unwrap(); // 0.926
```

`quantize` is symmetric: each block gets one scale. everything below uses the same `Quantized` type:

- `asymmetric::quantize` adds a zero-point per block, for values that aren't centered on zero
- `adaptive::quantize` picks each block's bit width from an error tolerance in the values' own units, like a tenth of their standard deviation
- `learned::refine` refits each block's scale, and its zero-point if it has one, to lower the mean squared error
- `learned::alternate` refits too, then rounds each value to the nearest code on its block's new line, and repeats until no code moves. both can raise the worst error past an adaptive tensor's tolerance
- `Scheme` picks one at run time, like `Scheme::Q4_32.quantize::<f16>(&weights)`

for a weight matrix, `q.set_shape(rows, columns)` records its shape, so `q.matmul(&inputs)` can multiply a batch of inputs by it, like a linear layer, and `q.dequantize_row_into(row, &mut out)` can decode one row, like an embedding lookup. `q.to_bytes()` saves a tensor, shape included, and `Quantized::from_bytes` loads it back.

codes can be 2 to 16 bits, and scales `f32`, `f16`, or `bf16`. the rest is in the [api docs](https://docs.rs/quantize). for python, see [python/](https://github.com/aksheyd/quantize/tree/main/python).

---

## comparison

against [candle](https://github.com/huggingface/candle)'s `Q4_0`, `Q5_0`, and `Q8_0` formats. 1024x1024 matrix, 50 iterations. like the chapters, these run from a clone of the repo.

### quality

```
cargo run --release --example compare
```

quantize two random matrices, reconstruct them, then matmul. mse is the mean squared error against the exact f32 result: smaller is better, and it varies a little between runs. bits/value counts the scales too: 4-bit codes plus one f16 scale per 32 values is 4 + 16/32 = 4.5.

<!-- comparison:start -->

| bits/value | quantize mse | candle mse |
| ---: | ---: | ---: |
| 4.5 | 0.060212 | 0.060252 |
| 5.5 | 0.013844 | 0.013854 |
| 8.5 | 0.000200 | 0.000201 |

<!-- comparison:end -->

on WikiText-2 with SmolLM-135M, 4-bit perplexity is 26.44 against candle `Q4_0`'s 26.46, and fp32 is 18.93. lower is better, and `cargo run --release --example wikitext --features workload` reproduces them.

### speed

```
cargo run --release --example throughput
```

quantize and dequantize with f16 scales. both libraries allocate their output on every call. aarch64 is an apple M5 Max and x86_64 an intel xeon. the hand-written simd only targets aarch64, so x86_64 runs plain loops and is slower.

<!-- speed:start -->

| ns/value | 4-bit quant | 8-bit quant | 4-bit dequant | 8-bit dequant |
| --- | ---: | ---: | ---: | ---: |
| quantize, aarch64 | 0.30 | 0.30 | 0.08 | 0.07 |
| candle, aarch64 | 0.41 | 0.43 | 0.20 | 0.25 |
| quantize, x86_64 | 3.90 | 3.74 | 1.06 | 0.36 |
| candle, x86_64 | 2.22 | 4.29 | 0.45 | 0.45 |

<!-- speed:end -->

_ns = nanosecond, a billionth of a second. smaller is faster._
