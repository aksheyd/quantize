"""The Python half of `just files-check` for gguf, with check_gguf.rs.

Reads DIR/from_rust.gguf, which quantize-files wrote, with gguf.GGUFReader,
and checks every metadata entry's type and value, and every tensor's shape
and values. Then saves DIR/from_python.gguf with gguf.GGUFWriter: F32 and
F16 tensors, aligned to 64 bytes, and assorted metadata, for the Rust half to
read back.

Both halves make the same values, (i - 60) / 16 for value i, which f32 and
f16 both hold exactly.

Usage: python check_gguf.py DIR
"""

import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter

directory = Path(sys.argv[1])
T = GGUFValueType


def values(*shape, dtype=np.float32):
    """Value i is (i - 60) / 16, as check_gguf.rs makes them."""
    count = int(np.prod(shape))
    return ((np.arange(count) - 60) / 16).astype(dtype).reshape(shape)


reader = GGUFReader(directory / "from_rust.gguf")
expected_metadata = {
    "general.architecture": ([T.STRING], "quantize-check"),
    "check.u8": ([T.UINT8], 2**8 - 1),
    "check.i8": ([T.INT8], -(2**7)),
    "check.u16": ([T.UINT16], 2**16 - 1),
    "check.i16": ([T.INT16], -(2**15)),
    "check.u32": ([T.UINT32], 2**32 - 1),
    "check.i32": ([T.INT32], -(2**31)),
    "check.u64": ([T.UINT64], 2**64 - 1),
    "check.i64": ([T.INT64], -(2**63)),
    "check.f32": ([T.FLOAT32], -1.5),
    "check.f64": ([T.FLOAT64], 0.1),
    "check.bool": ([T.BOOL], True),
    "check.strings": ([T.ARRAY, T.STRING], ["a", "", "naïve"]),
    "check.floats": ([T.ARRAY, T.FLOAT32], [0.5, -2.0]),
    # GGUFReader gives a nested array's innermost elements, one after another.
    "check.nested": ([T.ARRAY, T.ARRAY, T.STRING], ["b", "c", "d"]),
}
# GGUFReader also lists the header's numbers, as fields named GGUF.*.
fields = {key: field for key, field in reader.fields.items() if not key.startswith("GGUF.")}
assert sorted(fields) == sorted(expected_metadata), sorted(fields)
for key, (types, contents) in expected_metadata.items():
    assert fields[key].types == types, (key, fields[key].types)
    assert fields[key].contents() == contents, (key, fields[key].contents())

tensors = {tensor.name: tensor for tensor in reader.tensors}
expected_tensors = {"matrix": values(6, 20), "vector": values(20), "cube": values(2, 3, 4)}
assert sorted(tensors) == sorted(expected_tensors), sorted(tensors)
for name, expected in expected_tensors.items():
    tensor = tensors[name]
    assert tensor.tensor_type == GGMLQuantizationType.F32, (name, tensor.tensor_type)
    # GGUFReader lists dimensions innermost first, and shapes data outermost first.
    assert list(tensor.shape) == list(reversed(expected.shape)), (name, tensor.shape)
    assert tensor.data.shape == expected.shape, (name, tensor.data.shape)
    assert np.array_equal(tensor.data, expected), name
print(f"gguf read from_rust.gguf: {len(fields)} metadata entries and {len(tensors)} tensors with the values quantize-files wrote")

writer = GGUFWriter(directory / "from_python.gguf", arch="quantize-check")
writer.add_custom_alignment(64)
writer.add_key_value("check.u8s", [1, 2], T.ARRAY, sub_type=T.UINT8)
writer.add_int64("check.i64", -7)
writer.add_float64("check.f64", 0.1)
writer.add_bool("check.bool", False)
writer.add_array("check.strings", ["x", "yz"])
writer.add_key_value("check.nested", [[1, 2], [3]], T.ARRAY)
writer.add_tensor("f32_matrix", values(6, 20))
writer.add_tensor("f16_matrix", values(6, 20, dtype=np.float16))
writer.add_tensor("f16_cube", values(2, 3, 4, dtype=np.float16))
writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_tensors_to_file()
writer.close()
print("gguf wrote from_python.gguf")
