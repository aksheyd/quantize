system_python := if os_family() == "windows" { "python" } else { "python3" }
venv := if os_family() == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }

format:
    cargo fmt --all

lint:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace
    cargo test -p quantize --features rayon

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p quantize --features rayon
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p quantize-files

minimum-rust:
    rustup toolchain install 1.88 --profile minimal
    cargo +1.88 check --workspace --all-targets --all-features

setup:
    {{system_python}} -m venv .venv
    {{venv}} -m pip install maturin numpy pytest gguf==0.19.0

python:
    cargo clippy -p quantize-py --all-targets -- -D warnings
    VIRTUAL_ENV="{{justfile_directory()}}/.venv" {{venv}} -m maturin develop
    {{venv}} -m pytest python/tests

python-test:
    {{system_python}} -m pytest python/tests

# Checks quantize-files against the safetensors and gguf Python packages,
# both ways. Run `just setup` first.
files-check:
    {{venv}} -m pip install --quiet safetensors gguf==0.19.0
    cargo run -p quantize-files --example check_safetensors -- write target/files-check
    {{venv}} files/check_safetensors.py target/files-check
    cargo run -p quantize-files --example check_safetensors -- read target/files-check
    cargo run -p quantize-files --example check_gguf -- write target/files-check
    {{venv}} files/check_gguf.py target/files-check
    cargo run -p quantize-files --example check_gguf -- read target/files-check

llama_cpp_commit := "a11f57ba93797579a5d1855ee216a31f10242676"
# The compilers that build ggml and its check: gcc on Linux, and Apple's
# clang on macOS.
ggml_cc := if os() == "macos" { "cc" } else { "gcc" }
ggml_cxx := if os() == "macos" { "c++" } else { "g++" }
ggml_libraries := "target/ggml-build/ggml/src/libggml.a target/ggml-build/ggml/src/libggml-cpu.a target/ggml-build/ggml/src/libggml-base.a"

# Checks that ggml, the library under llama.cpp, loads the Q4_0 and Q8_0
# tensors quantize-files writes, and multiplies by them as quantize does.
# Builds only ggml's CPU backend, from llama.cpp at a pinned commit, for this
# machine's CPU, as llama.cpp builds by default.
ggml-check:
    git init --quiet target/llama.cpp
    git -C target/llama.cpp cat-file -e {{llama_cpp_commit}} 2>/dev/null || git -C target/llama.cpp fetch --quiet --depth 1 https://github.com/ggml-org/llama.cpp {{llama_cpp_commit}}
    git -C target/llama.cpp checkout --quiet --detach {{llama_cpp_commit}}
    cmake -S target/llama.cpp -B target/ggml-build --log-level=WARNING \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER={{ggml_cc}} -DCMAKE_CXX_COMPILER={{ggml_cxx}} \
        -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF \
        -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
    cmake --build target/ggml-build --target ggml --parallel {{num_cpus()}}
    mkdir -p target/ggml-check
    {{ggml_cc}} -O2 -Wall -Wextra -Werror -I target/llama.cpp/ggml/include -c files/check_ggml.c -o target/ggml-check/check_ggml.o
    {{ggml_cxx}} target/ggml-check/check_ggml.o {{ggml_libraries}} -lpthread -o target/ggml-check/check_ggml
    cargo run -p quantize-files --example check_gguf -- write target/ggml-check
    target/ggml-check/check_ggml target/ggml-check/quantized_from_rust.gguf

wheels:
    maturin build --release --out dist

# CI runs this on a version tag, like 0.3.0, before publishing anything.
release-check tag:
    grep -qxF 'version = "{{tag}}"' Cargo.toml
    grep -qxF '## {{tag}}' CHANGELOG.md
    cargo publish -p quantize --dry-run

# Prints a version's section of CHANGELOG.md: the notes of its GitHub release.
release-notes version:
    @awk '/^## /{p = ($2 == "{{version}}"); next} p' CHANGELOG.md

chapters:
    cargo run --release --example ch01_simple
    cargo run --release --example ch02_naive
    cargo run --release --example ch03_bits
    cargo run --release --example ch04_block
    cargo run --release --example ch05_asymmetric
    cargo run --release --example ch06_adaptive
    cargo run --release --example ch07_learned
    cargo run --release --example ch08_alternating

compare:
    cargo run --release -p benchmarks --example compare

throughput:
    cargo run --release -p benchmarks --example throughput

wikitext *args:
    cargo run --release -p benchmarks --example wikitext --features workload -- {{args}}

update-readme:
    cargo run --release -p benchmarks --example update_readme

# Downloads SmolLM-135M, quantizes it, and generates text, like
# `just smollm --scheme Q8_32 the capital of france is`.
[positional-arguments]
smollm *args:
    cargo run --release -p smollm -- "$@"
