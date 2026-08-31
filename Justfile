format:
    cargo fmt --all

lint:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings

test:
    cargo test

python:
    cargo clippy -p quantize-py --all-targets -- -D warnings
    maturin develop
    just python-test

python-test:
    python -m pytest python/tests

wheels:
    maturin build --release --out dist

compare:
    cargo run --release --example compare

throughput:
    cargo run --release --example throughput

update-readme:
    cargo run --release --example update_readme
