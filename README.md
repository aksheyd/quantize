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




| method                                               | mse      | cosine   |
| ---------------------------------------------------- | -------- | -------- |
| quantize                                             | 0.000028 | 0.999985 |
| [candle q8_0](https://github.com/huggingface/candle) | 0.000025 | 0.999986 |


*matrix size: 128x128*

