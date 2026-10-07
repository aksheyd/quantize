#!/bin/bash
# Builds main, the inline branch, and the fused branch as three crates, then times them side by side.
set -euo pipefail
git fetch -q --depth=1 origin \
  +refs/heads/main:refs/remotes/origin/main \
  +refs/heads/akshey/inline-8bit-decode-15ca:refs/remotes/origin/inline \
  +refs/heads/akshey/fused-8bit-batch-one-15ca:refs/remotes/origin/fused
make_crate() {
  rm -rf "/tmp/$1" && mkdir -p "/tmp/$1"
  git archive "$2" src | tar -x -C "/tmp/$1"
  printf '[package]\nname = "%s"\nversion = "0.0.0"\nedition = "2024"\n\n[dependencies]\nhalf = "2"\n' "$1" > "/tmp/$1/Cargo.toml"
  echo "$1: $(git rev-parse --short "$2")"
}
make_crate q_main refs/remotes/origin/main
make_crate q_inline refs/remotes/origin/inline
make_crate q_fused refs/remotes/origin/fused
cd "$(dirname "$0")"
cargo build --release -q
./target/release/ab8
./target/release/ab8
BITS=4 ./target/release/ab8
