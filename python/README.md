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

`bits` is the width of each code, from 2 to 16. `block` is how many values share one scale, and `scale` is how that scale is stored: `Scale.F32` (the default), `Scale.F16`, or `Scale.Bf16`. values can be a list of floats or a 1-d numpy array. a 2-d array keeps its shape, so `q.dequantize()` gives back a matrix and `q.matmul(x)` computes `x @ W.T`, like a linear layer.

the scales count toward the size: 4-bit codes with one f16 scale per 32 values cost 4.5 bits per value, or 5 with the default f32 scale. `q.bits_per_element` reports it.

the other schemes return the same `Quantized` type:

- `asymmetric.quantize(weights, bits=8, block=32)` adds a zero-point per block, for values that aren't centered on zero
- `adaptive.quantize(weights, block=32, tolerance=0.001)` picks each block's bit width from `tolerance`, the rounding error to aim for, in the same units as the weights
- `learned.refine(q, weights)` refits each block's scale and zero-point to lower the error
- `Scheme.Q4_32.quantize(weights)` picks a scheme at run time

quantized values can be pickled.

to build and test from a clone of the repo, with rust 1.88 or newer and [just](https://github.com/casey/just):

```
just setup
just python
```

on debian or ubuntu, run `sudo apt install python3-venv` first.
