# quantize

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

<!-- comparison:start -->

| method | bits/elt | mse (mean) | cosine (mean) |
| --- | ---: | ---: | ---: |
| quantize 4b×32 | 4.5 | 0.008339 | 0.995434 |
| candle Q4_0 | 4.5 | 0.007543 | 0.995825 |
| quantize 5b×32 | 5.5 | 0.001808 | 0.998995 |
| candle Q5_0 | 5.5 | 0.001730 | 0.999037 |
| quantize 8b×32 | 8.5 | 0.000025 | 0.999986 |
| candle Q8_0 | 8.5 | 0.000025 | 0.999986 |

_matrix size: 128x128, runs: 10_

<!-- comparison:end -->
