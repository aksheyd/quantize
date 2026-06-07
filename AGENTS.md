# quantize

A simple Rust library for learning and experimenting with quantization techniques.

## Overview

The `chapters/` directory contains progressive, tutorial-style examples deriving quantization from first principles.

The `benchmarks/` directory contains a harness to compare different quantization methods.

The `src/` directory contains library code for the crate's public quantization API on cargo.

## Guidelines

1. Human readability and the ability for code to teach quantization concepts is the highest priority.

Structure, naming, ordering, and comments must allow readers (including even those unfamiliar with Rust) to understand the ideas by reading the relevant files in sequence.

2. Comments and documentation must be fully self-contained. They must never reference external conversations, design discussions, previous sessions, LLMs, or any non-persistent context.

3. Try to keep all files (especially in `chapters/` and `benchmarks/`) under 50-100 lines when practical and reasonable.

The preference for keeping chapters/ and benchmarks/ line count low supports simplicity and is not absolute. It may be exceeded when doing so improves clarity, linearity, or the reader’s ability to derive the concepts directly from the code.

The existing code should showcase the quality bar for explainability without being overly verbose.

4. When choosing between abstraction and explicitness, favor code and comments that make the progression of ideas (e.g. symmetric vs. asymmetric, per-tensor vs. per-block, fixed vs. variable precision) visible and learnable without external explanation.

5. Use Conventional Commits for all commit messages.
