c:
    cargo run --release --example compare

l:
    cargo fmt && cargo clippy -- -D warnings

ur:
    cargo run --release --example update_readme