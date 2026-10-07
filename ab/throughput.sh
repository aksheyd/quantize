#!/bin/bash
# Runs `just throughput` with main's src, the inline branch's, and the fused branch's, in turn, twice.
set -euo pipefail
git fetch -q --depth=1 origin \
  +refs/heads/akshey/inline-8bit-decode-15ca:refs/remotes/origin/inline \
  +refs/heads/akshey/fused-8bit-batch-one-15ca:refs/remotes/origin/fused
for round in 1 2; do
  for tree in HEAD refs/remotes/origin/inline refs/remotes/origin/fused; do
    git checkout -q "$tree" -- src
    echo "== src from $tree ($(git rev-parse --short "$tree")), round $round"
    just throughput 2>&1 | grep -E 'matmul|dequant|dot|kernel'
  done
done
git checkout -q HEAD -- src
