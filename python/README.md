# quantize-py

python bindings for [quantize](https://github.com/aksheyd/quantize), a simple, fast quantization library written in rust. quantization stores numbers in fewer bits, trading a little accuracy for a lot less memory.

```
pip install quantize-py
```

requires python 3.12 or newer. numpy is installed with it.

```python
from quantize import Scale, quantize

weights = [0.42, -0.10, 0.70, -0.50]

q = quantize(weights, bits=8, block=32, scale=Scale.F16)
back = q.dequantize()  # [0.421, -0.098, 0.700, -0.498]
dot = q.dot(weights)  # 0.926
```

`bits` is the width of each code, from 2 to 16. `block` is how many values share one scale, and `scale` is how that scale is stored: `Scale.F32` (the default), `Scale.F16`, or `Scale.BF16`. values can be a list, a numpy array, or anything else `np.asarray` reads, like a pytorch tensor. a 2-d array keeps its shape, so `q.dequantize()` gives back a matrix and `q.matmul(x)` computes `x @ W.T`, like a linear layer.

the scales count toward the size: 4-bit codes with one f16 scale per 32 values cost 4.5 bits per value, or 5 with the default f32 scale. `q.bits_per_element` reports it.

the other schemes return the same `Quantized` type:

- `asymmetric.quantize(weights, bits=8, block=32)` adds a zero-point per block, for values that aren't centered on zero
- `adaptive.quantize(weights, tolerance=0.1 * weights.std())` gives each block the fewest bits, from 2 to 8, that round every weight within `tolerance`, in the weights' own units. a tenth of their standard deviation gives about 5 bits a block. for a list, use `np.std(weights)`
- `learned.refine(q, weights)` refits each block's scale, and its zero-point if it has one, to lower the mean squared error. it changes `q` in place, so call `q.copy()` first to keep the original
- `learned.alternate(q, weights)` refits too, then rounds each value to the nearest code on its block's new line, and repeats until no code moves. it also changes `q` in place. both can raise the worst error past an adaptive tensor's tolerance
- `Scheme.Q4_32.quantize(weights)` picks a scheme at run time

a block with outliers can need more than 8 bits, which raises `ToleranceTooTightError`. retrying with its `smallest_tolerance` works, but loosens every block, not just that one:

```python
try:
    q = adaptive.quantize(weights, tolerance=0.1 * weights.std())
except ToleranceTooTightError as error:
    q = adaptive.quantize(weights, tolerance=error.smallest_tolerance)
```

quantized values can be pickled, and compared with `==`. `q.to_bytes()` saves one as bytes, in the same format as the rust crate, and `Quantized.from_bytes(data)` loads it back. to keep it in an `np.savez` or safetensors file, store `np.frombuffer(q.to_bytes(), np.uint8)`.

to save its parts as plain arrays instead, like with `np.savez`, pass them back by name to `Quantized.from_parts`. leave out the one that's `None`: `bits` for an adaptive tensor, or `block_bits` for the others:

```python
parts = dict(kind=q.kind, shape=q.shape, block=q.block, bits=q.bits, block_bits=q.block_bits,
             codes=q.codes, scales=q.scales, zero_points=q.zero_points, scale=q.scale.name)
np.savez("layer.npz", **{name: part for name, part in parts.items() if part is not None})
q = Quantized.from_parts(**np.load("layer.npz"))
```

to load quantized values from a `torch.save` checkpoint, call `torch.serialization.add_safe_globals([Quantized])` before `torch.load`.

each value decodes as `code * scale`, or `(code - zero_point) * scale` with zero-points, using the scale and zero-point of its block. codes are signed and `bits` wide, and `q.codes` packs them low bits first. scales can be negative, since a symmetric block puts its value farthest from zero on the most negative code. `help(Quantized)` has the details.

to build and test from a clone of the repo, with rust 1.88 or newer and [just](https://github.com/casey/just):

```
just setup
just python
```

on debian or ubuntu, run `sudo apt install python3-venv` first.
