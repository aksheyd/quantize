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
    {{venv}} -m pip install maturin numpy pytest

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
