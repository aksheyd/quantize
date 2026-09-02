venv := if os_family() == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }

format:
    cargo fmt --all

lint:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings

test:
    cargo test

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

compare:
    cargo run --release --example compare

throughput:
    cargo run --release --example throughput

wikitext:
    cargo run --release --example wikitext --features workload

update-readme:
    cargo run --release --example update_readme
