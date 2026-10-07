#!/bin/bash
# WikiText on its first 512 tokens with main and the fused branch, alternating, twice each.
set -euo pipefail
git fetch -q --depth=1 origin \
  +refs/heads/main:refs/remotes/origin/main \
  +refs/heads/akshey/fused-batch-one-matmul-d178:refs/remotes/origin/fused
export CARGO_TARGET_DIR=/tmp/wikitext-target
for name in main fused; do
  rm -rf "/tmp/tree-$name"
  git worktree add -q --detach "/tmp/tree-$name" "refs/remotes/origin/$name"
  echo "$name: $(git -C "/tmp/tree-$name" rev-parse --short HEAD)"
  (cd "/tmp/tree-$name" && cargo build --release -q -p benchmarks --example wikitext --features workload)
  cp "$CARGO_TARGET_DIR/release/examples/wikitext" "/tmp/wikitext-$name"
done
for round in 1 2; do
  for name in main fused; do
    echo "== $name, round $round"
    (cd "/tmp/tree-$name" && "/tmp/wikitext-$name" --max-tokens 512 | tail -7)
  done
done
