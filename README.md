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

then `ch02_naive`, `ch03_bits`, `ch04_block`, `ch05_asymmetric`, `ch06_adaptive`, and `ch07_learned`.

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

let back = q.dequantize(); // [0.419, -0.099, 0.700, -0.502]
let dot = q.dot(&weights).unwrap(); // 0.927
```

`quantize` is symmetric: each block gets one scale. everything below uses the same `Quantized` type:

- `asymmetric::quantize` adds a zero-point per block, for values that aren't centered on zero
- `adaptive::quantize` picks each block's bit width from an error tolerance
- `learned::refine` refits each block's scale and zero-point to lower the error
- `Scheme` picks one at run time, like `Scheme::Q4_32.quantize::<f16>(&weights)`

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
| 4.5 | 0.066669 | 0.060276 |
| 5.5 | 0.014429 | 0.013859 |
| 8.5 | 0.000201 | 0.000201 |

<!-- comparison:end -->

### speed

```
cargo run --release --example throughput
```

quantize and dequantize with f16 scales, on an apple M4. the hand-written simd only targets 64-bit arm, so x86 runs plain loops and is slower.

| kernel | quantize ns/value | candle ns/value |
| --- | ---: | ---: |
| 4-bit quant | 0.29 | 0.46 |
| 8-bit quant | 0.24 | 0.30 |
| 4-bit dequant | 0.09 | 0.29 |
| 8-bit dequant | 0.07 | 0.26 |

_ns = nanosecond, a billionth of a second. smaller is faster._
