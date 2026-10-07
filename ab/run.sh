#!/bin/bash
# Builds main, the decode PR, and the fused branch as three crates, then times them side by side.
set -euo pipefail
git fetch -q --depth=1 origin \
  +refs/heads/main:refs/remotes/origin/main \
  +refs/heads/akshey/branch-free-4bit-decode-d178:refs/remotes/origin/decode \
  +refs/heads/akshey/fused-batch-one-matmul-d178:refs/remotes/origin/fused
make_crate() {
  rm -rf "/tmp/$1" && mkdir -p "/tmp/$1"
  git archive "$2" src | tar -x -C "/tmp/$1"
  printf '[package]\nname = "%s"\nversion = "0.0.0"\nedition = "2024"\n\n[dependencies]\nhalf = "2"\n' "$1" > "/tmp/$1/Cargo.toml"
  echo "$1: $(git rev-parse --short "$2")"
}
make_crate quantize_old refs/remotes/origin/main
make_crate quantize_b refs/remotes/origin/decode
make_crate quantize refs/remotes/origin/fused
cd "$(dirname "$0")"
cargo build --release -q
./target/release/abbench
./target/release/abbench
