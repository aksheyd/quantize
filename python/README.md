# Python

```bash
pip install quantize-rs
```

```python
from quantize import Scale, quantize

weights = [0.42, -0.10, 0.70, -0.50]
q = quantize(weights, bits=8, block=32, scale=Scale.F32)
back = q.dequantize()
```

From this repository:

```bash
pip install maturin numpy
maturin develop
```
