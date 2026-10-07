#!/bin/bash
# Temporary: times 0.3.1, main, this branch, and main again side by side on
# Apple silicon, each built as a git dependency, the way a user's crate gets it.
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
git -C "$repo" branch -f timing-main HEAD^1
git -C "$repo" branch -f timing-fix HEAD^2
git -C "$repo" branch -f timing-main-again HEAD^1
echo "main $(git -C "$repo" rev-parse --short HEAD^1), this branch $(git -C "$repo" rev-parse --short HEAD^2)"
cd "$(dirname "$0")"
cat > Cargo.toml <<TOML
[package]
name = "qbench"
version = "0.1.0"
edition = "2021"

[dependencies]
q031 = { version = "=0.3.1", package = "quantize" }
qmain = { package = "quantize", git = "file://$repo", branch = "timing-main" }
qfix = { package = "quantize", git = "file://$repo", branch = "timing-fix" }
qsame = { package = "quantize", git = "file://$repo", branch = "timing-main-again" }

[profile.dev.package."*"]
opt-level = 3

[workspace]
TOML
cargo build --release -q
cargo build -q
echo "== release"
./target/release/qbench bench 2048 2048 15
./target/release/qbench bench 2048 2048 15
./target/release/qbench token 8 9 1
./target/release/qbench token 8 9 1
./target/release/qbench token 8 5 4
echo "== dev, dependencies at opt-level 3"
./target/debug/qbench bench 1024 1024 5
./target/debug/qbench token 2 3 1
