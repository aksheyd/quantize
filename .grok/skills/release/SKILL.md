---
name: release
description: Helps with releasing new versions of this Rust crate
---

Bump `version` in Cargo.toml.

Then:

1. Commit
2. `cargo publish`
3. `git tag -a <ver> -m "Release <ver>"` (plain version, e.g. `0.1.1`)
4. `git push && git push --tags`

Clean tree + tests first.
