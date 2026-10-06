---
name: release
description: Helps with releasing new versions of this Rust crate
---

The version lives only in `[workspace.package]` in the root `Cargo.toml`. The crate, `python/Cargo.toml`, and `pyproject.toml` (through maturin) all read it. Between releases it names the next release with a `-dev` suffix, e.g. `0.3.0-dev`.

CI publishes each release from its tag. Never run `cargo publish`, `uv publish`, or `gh release create` by hand.

Every commit and tag uses `Akshey D <131929364+aksheyd@users.noreply.github.com>`, never another of the owner's emails, whatever the local git config says.

To release:

1. Set the release version (`0.3.0-dev` becomes `0.3.0`), run `just test` so `Cargo.lock` picks it up, add a `## 0.3.0` section to `CHANGELOG.md`, and merge that commit into `main`
2. On an up-to-date `main`, `just release-check <ver>` runs the checks CI runs before publishing
3. `git -c user.name="Akshey D" -c user.email=131929364+aksheyd@users.noreply.github.com tag -a <ver> -m "Release <ver>"` (plain version, e.g. `0.3.0`)
4. `git push origin <ver>` publishes the release
5. Set the next dev version the same way (e.g. `0.3.1-dev`), commit, and push

On a version tag, `.github/workflows/wheels.yaml` runs `just release-check`, builds and tests the wheels, then publishes in order: `quantize-py` to PyPI, `quantize` to crates.io, and a GitHub release titled `quantize <ver>` with `just release-notes <ver>` as its notes. PyPI and crates.io use trusted publishing with no stored token, so the GitHub `pypi` and `crates-io` environments must match the trusted publishers on PyPI and crates.io.

If a publishing job fails, fix the cause and re-run the failed jobs. Don't push the tag again: rebuilt wheels can differ, and PyPI rejects a changed file under a name it already has.

`include` paths must be rooted (`/LICENSE`, `/README.md`, `/src/**`) so files named `LICENSE` under `.venv` are not packed.

Clean tree + tests first.
