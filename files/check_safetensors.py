"""The Python half of `just files-check`, with check_safetensors.rs.

Reads DIR/from_rust.safetensors, which quantize-files wrote, with the
safetensors package, and checks every value. Then saves
DIR/from_python.safetensors, with F32 and F16 tensors and a copy of the
quantized one, for the Rust half to read back. numpy has no bf16, so the
Rust tests cover BF16 instead.

Both halves make the same values, (i - 60) / 16 for value i, which f32 and
f16 both hold exactly.

Usage: python check_safetensors.py DIR
"""

import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

directory = Path(sys.argv[1])


def values(*shape, dtype=np.float32):
    """Value i is (i - 60) / 16, as check_safetensors.rs makes them."""
    count = int(np.prod(shape))
    return ((np.arange(count) - 60) / 16).astype(dtype).reshape(shape)


from_rust = load_file(directory / "from_rust.safetensors")
assert sorted(from_rust) == ["matrix", "quantized", "vector"], sorted(from_rust)
for name, expected in [("matrix", values(6, 20)), ("vector", values(20))]:
    tensor = from_rust[name]
    assert tensor.dtype == np.float32, (name, tensor.dtype)
    assert tensor.shape == expected.shape, (name, tensor.shape)
    assert np.array_equal(tensor, expected), name
quantized = from_rust["quantized"]
assert quantized.dtype == np.uint8 and quantized.ndim == 1, (quantized.dtype, quantized.shape)
assert quantized[:4].tobytes() == b"QNTZ", quantized[:4].tobytes()
print(f"safetensors read from_rust.safetensors: {len(from_rust)} tensors with the values quantize-files wrote")

save_file(
    {
        "f32_matrix": values(6, 20),
        "f16_matrix": values(6, 20, dtype=np.float16),
        "f16_vector": values(20, dtype=np.float16),
        "quantized": quantized,
    },
    directory / "from_python.safetensors",
    metadata={"written_by": "check_safetensors.py"},
)
print("safetensors wrote from_python.safetensors")
