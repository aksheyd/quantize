---
name: release
description: Helps with releasing new versions of this Rust crate
---

The version lives only in `[workspace.package]` in the root `Cargo.toml`. The crate, `python/Cargo.toml`, and `pyproject.toml` (through maturin) all read it.

To release:

1. Set the release version, run `just test` so `Cargo.lock` picks it up, and commit
2. `cargo publish -p quantize` (a bare `cargo publish` also selects the unpublished `chapters` package)
3. `git tag -a <ver> -m "Release <ver>"` (plain version, e.g. `0.1.1`)
4. `git push && git push --tags`

A version tag also publishes the Python package `quantize-py`: `.github/workflows/wheels.yaml` builds wheels and uploads to PyPI via trusted publishing. The GitHub `pypi` environment must match a PyPI trusted publisher for this repository.

Never `cargo publish --allow-dirty`. `include` paths must be rooted (`/LICENSE`, `/README.md`, `/src/**`) so files named `LICENSE` under `.venv` are not packed.

Clean tree + tests first.
