# changelog

## 0.3.1

0.3.1 is a patch release of the rust crate `quantize` and the python package `quantize-py`. it decodes with less memory, and adds `matmul_into` and `Scheme` text.

### faster, with less memory

- `dot`, `dequantize_into`, and `dequantize_row_into` allocate nothing, whatever the scheme. on a 5-bit, asymmetric, or adaptive tensor of 16 million values, `dot` used to allocate about 70 MB, and `dequantize_into` is about twice as fast.
- `matmul` decodes one row at a time. it used to decode an adaptive matrix whole on every call, and a 4096 × 4096 one is now over twice as fast.
- python's `dot`, `matmul`, and `dequantize()` no longer copy the tensor on every call.

### new

- `matmul_into` writes into a buffer you pass, so a loop can reuse it, and its `Error::OutputMismatch` catches a `set_shape` with rows and columns swapped, at any batch size.
- `Scheme` prints as text, like `symmetric(bits=4, block=32)`, and `parse` reads it back, so a scheme can come from config or a cli flag.
- python: `Scale(name)` and `Scheme(text)`, like `Scale("f16")` and `Scheme("Q4_32")`.

### fixes

- `fit_scale_and_zero_point` fits the closest line through zero when the values don't rise or fall with your codes. it returned scale 1, which decoded each code to itself.
- python: `torch.load` reads `Scale` and `Scheme` pickles once you allow their classes, as with `Quantized`.

### changes you might notice

- free-threaded python turns the gil back on when it imports `quantize`, with a `RuntimeWarning`, so a refit can't panic when another thread uses the tensor.
- 0.3.0 can't load `Scale` or `Scheme` pickles made by 0.3.1, though 0.3.1 loads 0.3.0's, and tensors pickle as before.
- `Scheme` text rejects values that `quantize` would, like `bits=99`, so in python, a scheme like `Scheme.symmetric(bits=99)` raises when it's unpickled or copied.
- the `no_std` claim is gone: the `std` feature does nothing, and `Error` implements `std::error::Error` either way.

### chapters and contributors

- chapters 3 to 8 fill small gaps, like what `1 << n` means, and chapter 3's worst error no longer hides nan.
- the benchmark examples need `-p benchmarks`, so a first chapter run no longer downloads candle.

## 0.3.0

0.3.0 is a breaking release of the rust crate `quantize` and the python package `quantize-py`. it adds matrices, saving and loading, and `learned::alternate`, and its 4-bit accuracy now matches candle's.

### highlights

- matrices: `set_shape` records a weight matrix's shape, `matmul` multiplies a batch of inputs by it, like a linear layer, and `dequantize_row_into` decodes one row, like an embedding lookup. in python, a 2-d array keeps its shape, and input can be anything `np.asarray` reads, like a pytorch tensor.
- saving: `to_bytes` and `from_bytes` share one format across rust and python. python also has `Quantized.from_parts`, and its pickles load in `torch.load` once you allow the class.
- `learned::alternate` refits like `refine`, then lets the codes move too, until none does. new chapter 8 derives it.
- 4-bit mean squared error is 0.060212 against candle's 0.060252, and WikiText-2 perplexity is 26.44 against candle `Q4_0`'s 26.46.

### breaking changes

rust:
- the crate needs rust 1.88 or newer.
- `learned::refine` returns `Result`: add `?`.
- `Error` is `#[non_exhaustive]`, has new variants, and is no longer `Eq`: add a `_ =>` arm to matches.
- every `Quantized` variant has a `columns` field: add `columns: None`, or `..` in patterns. the adaptive variant's `bytes` and `bits` are now `codes` and `block_bits`, a `Vec<u8>`.
- `quantize::error`, `packed`, `scale`, `scheme`, and `tensor` are gone: import from the crate root.
- `Packed::unpack_slice` is private: use `q.unpacked_codes()`. `params::choose_bits` returns an `Option`. a custom `Scale` needs `NAME`, `from_f32_away_from_zero`, `write_le_bytes`, and `read_le_bytes`.

python:
- `adaptive.quantize` and `Scheme.adaptive` need `tolerance=`, like `tolerance=0.1 * np.std(w)`, and `Scale.Bf16` is now `Scale.BF16`.
- `QuantizeError` is a `ValueError`: put `except QuantizeError` before `except ValueError`.
- tensors pickled by 0.2 don't load: quantize again, or pass what 0.2's getters return to `Quantized.from_parts`, with `shape=(len(q),)` and the scale's name, like `scale="f16"`.
- `Quantized` is unhashable, now that `==` compares values. negative `bits` or `block` raise `InvalidBitsError` or `InvalidBlockError` instead of `OverflowError`, 3-d input raises `ValueError`, and masked arrays raise `TypeError`.
- building the sdist needs rust 1.88, and fails with `--locked`; a plain build pins the same versions.

results that change, in both languages:
- the same values get different codes and scales. a block's value farthest from zero lands on the most negative code, as in ggml's `Q4_0`, so scales can be negative, and f16 and bf16 scales round away from zero. `params::symmetric_scale` takes that value with its sign.
- `refine` keeps symmetric tensors symmetric, and keeps a block's new fit only if it lowers the error. `fit_scale_and_zero_point` fits just the offset when every code is equal.
- where 0.2 returned wrong values, these now error or panic: `adaptive::quantize` when 8 bits miss the tolerance (retry with the error's `smallest_tolerance`), any scheme when a scale overflows f16 or bf16 (use f32 scales), the `params` and `Packed` helpers given bit widths outside 2 to 16, and decoding a tensor with too few code bytes, which used to read past its buffer on arm.
- also: flat asymmetric blocks decode to their value, nan on arm no longer zeroes its block, `dot` stays precise on long tensors, adaptive sizes count one byte per block width (python's `block_bits` is `uint8`), `{:?}` prints a one-line summary, and `repr` shows `shape=` instead of `len=`.
