# quantize

A simple quantization library

## use as a library

```rust
use quantize::{quantize, dequantize};

let q = quantize(0.42, 0.01);
let back = dequantize(q, 0.01);
```

---

### comparison

```
cargo run --release --example compare
```

---

<!-- comparison:start -->

| method | mse | cosine |
| --- | ---: | ---: |
| quantize | 0.876830 | 0.000000 |
| [candle q8_0](https://github.com/huggingface/candle) | 0.000025 | 0.999985 |

_matrix size: 128x128_

<!-- comparison:end -->
