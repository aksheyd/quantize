#!/bin/bash
# Temporary: times 0.3.1, main, and variants of this branch side by side on
# Apple silicon, each built as a git dependency, the way a user's crate gets it.
#   fix: this branch
#   check per block: an earlier version that checked bytes once per 4-bit block (h1.patch)
#   fix+dot: this branch, and dot through the single-vector kernels (final-dot.patch)
set -euo pipefail
repo="$(git rev-parse --show-toplevel)"
here="$(cd "$(dirname "$0")" && pwd)"
make_version() {
  local name="$1" ref="$2"
  shift 2
  local dir="/tmp/versions/$name"
  rm -rf "$dir" && mkdir -p "$dir"
  git -C "$repo" archive "$ref" src | tar -x -C "$dir"
  for patch in "$@"; do patch -s -p1 -d "$dir" < "$here/$patch"; done
  printf '[package]\nname = "quantize"\nversion = "0.3.2-dev"\nedition = "2024"\n\n[dependencies]\nhalf = "2"\n' > "$dir/Cargo.toml"
  git -C "$dir" init -q -b timing
  git -C "$dir" add -A
  git -C "$dir" -c user.name=timing -c user.email=timing@example.com commit -q -m "$name"
  echo "$name: $(git -C "$repo" rev-parse --short "$ref") $*"
}
make_version main HEAD^1
make_version main-again HEAD^1
make_version fix HEAD^2
make_version h1 HEAD^1 h1.patch
make_version dot HEAD^2 final-dot.patch
cd "$here"
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
qh1 = { package = "quantize", git = "file:///tmp/versions/h1", branch = "timing" }
qdot = { package = "quantize", git = "file:///tmp/versions/dot", branch = "timing" }

[profile.dev.package."*"]
opt-level = 3

[workspace]
TOML
cargo build --release -q
cargo build -q
echo "== release, f16 scales"
./target/release/qbench bench 1024 1024 31
./target/release/qbench bench 1024 1024 31
./target/release/qbench bench 2048 2048 21
echo "== release, f32 scales"
SCALE=f32 ./target/release/qbench bench 1024 1024 31
echo "== release, whole tokens"
./target/release/qbench token 8 15 1
./target/release/qbench token 8 15 1
echo "== dev, dependencies at opt-level 3"
./target/debug/qbench bench 1024 1024 5
./target/debug/qbench bench 1024 1024 5
./target/debug/qbench token 2 3 1
