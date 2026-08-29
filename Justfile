c:
    cargo run --release --example compare

l:
    cargo fmt && cargo clippy --all-targets -- -D warnings

t:
    cargo test

b:
    cargo run --release --example throughput

ur:
    cargo run --release --example update_readme
