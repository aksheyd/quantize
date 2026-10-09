# changelog

## 0.4.0

0.4.0 is a breaking release of the rust crate `quantize` and the python package `quantize-py`. it adds a `rayon` feature that splits large `matmul` calls across the cores, and finds any row of an adaptive matrix equally fast. python users see no api change, though `quantize-py` moves to 0.4.0 with the crate, since they share a version. results are the same as 0.3.2's, bit for bit, on x86 and apple silicon, and the two load each other's files and pickles.

### breaking changes

- the `std` and `default` features are gone, since they did nothing: delete them from `quantize`'s `features`.
- `Scale` requires `Sync`, so every core can read one tensor: make a custom scale type `Sync`, which a plain number type already is.
- `Quantized::Adaptive` has a new field, `row_starts`: add `..` to patterns that list every field. to build one by hand, give it `columns: None` and `row_starts: Vec::new()`, then call `validate()` and `set_shape(rows, columns)`, which fills it in. code that sets `columns` itself, in a struct literal or through a pattern, calls `set_shape` instead: until it does, `validate`, `dequantize_row_into`, and `matmul` return an error.
- a `BITS` outside 2 to 16 or a `BLOCK` of 0, like `quantize::<f32, 1, 32>`, stops the build instead of returning `InvalidBits` or `InvalidBlock`, though `cargo check` misses it: for the error at run time, call the scheme's `quantize_with`, like `symmetric::quantize_with`.
- `matmul` and `matmul_into` return `Error::InputMismatch` instead of `ShapeMismatch` for `inputs` that don't split into vectors of `columns` values, and the error says `set_shape` may have rows and columns swapped. a match on `ShapeMismatch` there still compiles, but reaches its `_` arm: match `Error::InputMismatch { .. }` instead.

### new and faster

- the `rayon` feature, off by default, splits a large `matmul` across the cores, with the same results, bit for bit: on an 8-core intel xeon, 512 tokens through SmolLM-135M's linear layers take 0.7 s instead of 5.1, and 64 vectors through a 4096 × 4096 layer take 10 ms instead of 80. small calls stay on one core. a program that already keeps every core busy calling `matmul` loses up to about a quarter, and one with cores to spare gains. python's wheels leave the feature off.
- `dequantize_row_into` finds any row of an adaptive matrix equally fast: the last row of a 32,000 × 4,096 embedding table takes 0.006 ms instead of 0.52, or 2.4 with blocks of 30.
- python: `from_parts` copies the array that `q.block_bits` returns at once, as it does `codes`, so it rebuilds a 4096 × 4096 adaptive tensor in 5 ms instead of 19.

### changes you might notice

- an adaptive matrix keeps where each row starts, 8 bytes a row, which `to_bytes` doesn't save, and `nbytes` and `bits_per_element` don't count: 0.25% more memory with 4,096 columns, and 16% with 64.
- `set_shape` checks an adaptive tensor's buffers first, as `validate` does, so one built by hand that's short of widths or codes gets `validate`'s error instead of `Ok`.

### contributors

- `just lint`, `just test`, and `just doc` run with and without the `rayon` feature.
- the WikiText run multiplies on every core, as candle does, with the same perplexities.

## 0.3.2

0.3.2 is a patch release of the rust crate `quantize` and the python package `quantize-py`. it multiplies one vector up to 3 times as fast and big batches up to 2.7 times, and adds wheels for free-threaded python.

### faster

- `matmul` on one vector, as when a model generates a token, multiplies `Q4_32`, `Q8_32`, and similar matrices as it decodes them. with 4-bit weights, one token through SmolLM-135M's linear layers takes 45 ms instead of 140 on an intel xeon, and 19 instead of 27 on an M1. 8-bit weights gain about 1.4 times.
- `dot` takes the same path: 2 to 3 times as fast on the xeon, and 5 on the M1.
- batches of more than 256 vectors go through those matrices in groups of 256 that stay in cache, so 32 images through ViT-B/16's qkv layer run 2.2 to 2.7 times as fast on the xeon, and 1.3 on the M1.
- on x86, symmetric 4-bit codes decode 2.8 times as fast, and 8-bit codes 1.4 times.
- python: `dot`, `matmul`, and `dequantize` on 65,536 values or fewer keep the gil, so while another thread keeps running python, a 256 × 256 `dot` takes under 0.1 ms instead of 5. quantizing and refitting more than 4,096 values let other threads run, so four threads refit 3.2 to 3.5 times as fast as one.

### new

- free-threaded python 3.14 gets its own wheels and keeps the gil off, where 0.3.1 turned it back on, so threads call `quantize` in parallel.
- python: `matmul` takes `out=`, like `matmul_into`, and `out=` can be a float32 pytorch tensor on the cpu.
- `Scheme` text can leave out `block`, for blocks of 32, so `symmetric(bits=4)` reads as `Q4_32`.

### fixes

- `Error::OutputMismatch` says how many values `out` should hold, and when `inputs` is empty, instead of blaming `set_shape`.
- python: an `out` that another call is using says so, instead of claiming it isn't writable.

### changes you might notice

- python: two threads refitting one tensor at once both start from it as it was, and it keeps the last result. the gil used to run them one after the other.
- python: `print(scheme)` shows the text `Scheme(text)` reads, instead of the repr.
- `Scheme` text without `block` and with a value out of range, like `symmetric(bits=1)`, gets `InvalidBits` or `InvalidTolerance` instead of `InvalidScheme`.
- `dot` results can change in the last bits, and are more precise on long tensors.
- on x86, each f16 scale converts through a function call, costing one-vector `matmul` 3 to 6%. f32 or bf16 scales, or a build for f16c, like `-C target-cpu=x86-64-v3`, avoid it.
- an amd epyc runs batches of more than 256 vectors 1 to 3% slower while they fit in its L3 cache, and up to 2 times faster beyond it.
- python: while another thread keeps running python, `refine` on more than 4,096 values and `alternate` on more than 40 wait about 5 ms a call for the gil, where 0.3.1 kept it.
- python: with a gil, `dequantize()` calls on 65,536 values or fewer take turns across threads, so 64 of them on 256 × 256 over 8 threads take 6.8 ms against 0.3.1's 2.7, though on one thread they run 3 times as fast.

### chapters and contributors

- chapter 3 stops on 0 or more than 31 bits and says why 1 bit gives nan, and chapter 7 gives a typical block's gain.
- the WikiText run stops on an argument it doesn't know, instead of quietly scoring the whole set for hours.

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
