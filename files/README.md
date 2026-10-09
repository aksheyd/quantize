# quantize-files

read and write the files models ship in, with [quantize](https://crates.io/crates/quantize)'s tensors inside: [safetensors](https://github.com/huggingface/safetensors), hugging face's format, and [gguf](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), llama.cpp's.

```
cargo add quantize-files
```

requires rust 1.88 or newer. each version of quantize-files comes out with the quantize of the same version, and requires exactly that one.

- both formats read `F32`, `F16`, and `BF16` tensors as f32, and write floats as `F32`
- safetensors keeps a quantized tensor of any scheme, as the bytes `to_bytes` writes
- gguf keeps `Q4_32` and `Q8_32` matrices with f16 scales, as ggml's `Q4_0` and `Q8_0` blocks, which hold the same values

the [api docs](https://docs.rs/quantize-files) lay out each format byte by byte.
