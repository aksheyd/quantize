# quantize

[![Crates.io](https://img.shields.io/crates/v/quantize.svg)](https://crates.io/crates/quantize)
[![Docs.rs](https://docs.rs/quantize/badge.svg)](https://docs.rs/quantize)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple quantization library

## use as a library

```rust
use quantize::{quantize, dequantize};

let weights = [0.42_f32, -0.10, 0.70, -0.50];
let (scales, codes) = quantize::<f32, 8, 32>(&weights);
let back = dequantize::<_, 32>(&scales, &codes);
```

---

### comparison

```
cargo run --release --example compare
```

---

<!-- comparison:start -->

| method | bits/elt | mse (mean) | cosine (mean) |
| --- | ---: | ---: | ---: |
| ours 4b×32 | 4.5 | 0.008385 | 0.995404 |
| candle Q4_0 | 4.5 | 0.007587 | 0.995787 |
| ours 5b×32 | 5.5 | 0.001804 | 0.998995 |
| candle Q5_0 | 5.5 | 0.001737 | 0.999032 |
| ours 8b×32 | 8.5 | 0.000025 | 0.999986 |
| candle Q8_0 | 8.5 | 0.000025 | 0.999985 |

_matrix size: 128x128, runs: 10_

<!-- comparison:end -->
