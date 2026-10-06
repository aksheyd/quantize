system_python := if os_family() == "windows" { "python" } else { "python3" }
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
    cargo run --release --example compare

throughput:
    cargo run --release --example throughput

wikitext *args:
    cargo run --release --example wikitext --features benchmarks/workload -- {{args}}

update-readme:
    cargo run --release --example update_readme
