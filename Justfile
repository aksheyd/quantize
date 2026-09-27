venv := if os_family() == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }

format:
    cargo fmt --all

lint:
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace

doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

setup:
    python -m venv .venv
    {{venv}} -m pip install maturin numpy pytest

python:
    cargo clippy -p quantize-py --all-targets -- -D warnings
    VIRTUAL_ENV="{{justfile_directory()}}/.venv" {{venv}} -m maturin develop
    {{venv}} -m pytest python/tests

python-test:
    python -m pytest python/tests

wheels:
    maturin build --release --out dist

chapters:
    cargo run --example ch01_simple
    cargo run --example ch02_naive
    cargo run --example ch03_bits
    cargo run --example ch04_block
    cargo run --example ch05_asymmetric
    cargo run --example ch06_adaptive
    cargo run --example ch07_learned

compare:
    cargo run --release --example compare

throughput:
    cargo run --release --example throughput

wikitext:
    cargo run --release --example wikitext --features workload

update-readme:
    cargo run --release --example update_readme
