#!/bin/bash
# Temporary: times 0.3.1, main, this branch, and variants of it side by side
# on Apple silicon, each built as a git dependency, the way a user's crate gets it.
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
# Each version is a crate of its own, in a git repository of its own.
make_version() {
  local dir="/tmp/versions/$1"
  rm -rf "$dir" && mkdir -p "$dir"
  git -C "$repo" archive "$2" src | tar -x -C "$dir"
  if [ -n "${3:-}" ]; then perl -pi -e "$3" "$dir/src/shared/decode.rs"; fi
  printf '[package]\nname = "quantize"\nversion = "0.3.2-dev"\nedition = "2024"\n\n[dependencies]\nhalf = "2"\n' > "$dir/Cargo.toml"
  git -C "$dir" init -q -b timing
  git -C "$dir" add -A
  git -C "$dir" -c user.name=timing -c user.email=timing@example.com commit -q -m "$1"
  echo "$1: $(git -C "$repo" rev-parse --short "$2") $(grep -c BLOCKS_PER_RUN "$dir/src/shared/decode.rs") $(grep 'const BLOCKS_PER_RUN' "$dir/src/shared/decode.rs" || true)"
}
make_version main HEAD^1
make_version main-again HEAD^1
make_version fix HEAD^2
make_version part-a d1d7572
make_version runs-1024 HEAD^2 's/^const BLOCKS_PER_RUN: usize = 64;/const BLOCKS_PER_RUN: usize = 1024;/'
make_version runs-16384 HEAD^2 's/^const BLOCKS_PER_RUN: usize = 64;/const BLOCKS_PER_RUN: usize = 16384;/'
cd "$(dirname "$0")"
cat > Cargo.toml <<TOML
[package]
name = "qbench"
version = "0.1.0"
edition = "2021"

[dependencies]
q031 = { version = "=0.3.1", package = "quantize" }
qmain = { package = "quantize", git = "file:///tmp/versions/main", branch = "timing" }
qfix = { package = "quantize", git = "file:///tmp/versions/fix", branch = "timing" }
qsame = { package = "quantize", git = "file:///tmp/versions/main-again", branch = "timing" }
qparta = { package = "quantize", git = "file:///tmp/versions/part-a", branch = "timing" }
qruns1024 = { package = "quantize", git = "file:///tmp/versions/runs-1024", branch = "timing" }
qruns16384 = { package = "quantize", git = "file:///tmp/versions/runs-16384", branch = "timing" }

[profile.dev.package."*"]
opt-level = 3

[workspace]
TOML
cargo build --release -q
echo "== release, f16 scales"
./target/release/qbench bench 2048 2048 21
./target/release/qbench bench 2048 2048 21
echo "== release, f32 scales"
SCALE=f32 ./target/release/qbench bench 2048 2048 21
echo "== release, bf16 scales"
SCALE=bf16 ./target/release/qbench bench 2048 2048 21 matmul
echo "== release, whole tokens"
./target/release/qbench token 8 15 1
./target/release/qbench token 8 15 1
SCALE=f32 ./target/release/qbench token 8 15 1
