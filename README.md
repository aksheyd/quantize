# quantize

a simple, fast quantization library usable as a [rust crate](https://crates.io/crates/quantize) or [python package](https://pypi.org/project/quantize-py/).

to learn more, feel free to peruse the [chapters](https://github.com/aksheyd/quantize/tree/main/chapters), which show how the repo's algorithms progressed over time.

## use as a library

```rust
use quantize::{adaptive, asymmetric, quantize, Scheme};

let weights = [0.42_f32, -0.10, 0.70, -0.50];

let q = quantize::<f32, 8, 32>(&weights).unwrap();
let _ = asymmetric::quantize::<f32, 8, 32>(&weights).unwrap();
let _ = adaptive::quantize::<f32, 32>(&weights, 0.001).unwrap();
let _ = Scheme::Q4_32.quantize::<f32>(&weights).unwrap();

let back = q.dequantize();
let _ = q.dot(&weights);
```

---

## comparison

1024x1024 matrix, 50 iterations.

### quality

```
cargo run --release --example compare
```

reconstruct, then matmul.

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

pack and unpack. apple M4.

| kernel | quantize ns/value | candle ns/value |
| --- | ---: | ---: |
| 4-bit quant | 0.29 | 0.46 |
| 8-bit quant | 0.24 | 0.30 |
| 4-bit dequant | 0.09 | 0.29 |
| 8-bit dequant | 0.07 | 0.26 |

_ns = nanosecond, a billionth of a second. smaller is faster._
