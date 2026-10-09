---
name: release
description: Helps with releasing new versions of this Rust crate
---

The version lives in `[workspace.package]` in the root `Cargo.toml`. The crate, `quantize-files`, `python/Cargo.toml`, and `pyproject.toml` (through maturin) all read it. Between releases it names the next release with a `-dev` suffix, e.g. `0.3.0-dev`. Just below, the `quantize` entry in `[workspace.dependencies]` repeats it with an `=` in front, e.g. `=0.3.0-dev`, because `quantize-files` requires exactly the `quantize` it's released with.

CI publishes each release from its tag. Never run `cargo publish`, `uv publish`, or `gh release create` by hand.

Every commit and tag uses `Akshey D <131929364+aksheyd@users.noreply.github.com>`, never another of the owner's emails, whatever the local git config says.

To release:

1. Set the release version in both places (`0.3.0-dev` becomes `0.3.0`, and `=0.3.0-dev` becomes `=0.3.0`), run `just test` so `Cargo.lock` picks it up, add a `## 0.3.0` section to `CHANGELOG.md`, and merge that commit into `main`
2. On an up-to-date `main`, `just release-check <ver>` runs the checks CI runs before publishing
3. `git -c user.name="Akshey D" -c user.email=131929364+aksheyd@users.noreply.github.com tag -a <ver> -m "Release <ver>"` (plain version, e.g. `0.3.0`)
4. `git push origin <ver>` publishes the release
5. Set the next dev version the same way, in both places (e.g. `0.3.1-dev` and `=0.3.1-dev`), commit, and push

On a version tag, `.github/workflows/wheels.yaml` runs `just release-check`, builds and tests the wheels, then publishes in order: `quantize-py` to PyPI, `quantize` and then `quantize-files` to crates.io, and a GitHub release titled `quantize <ver>` with `just release-notes <ver>` as its notes. PyPI and crates.io use trusted publishing with no stored token, so the GitHub `pypi` and `crates-io` environments must match the trusted publishers on PyPI and on each crate's crates.io settings.

To hold `quantize-files` back from a release, add `publish = false` to `files/Cargo.toml`. `just release-check` and the `crates-io` job then skip it and publish `quantize` alone, with nothing else to change; the `=` requirement still moves with the version.

A trusted-publishing token can't create a crate, so `quantize-files`'s first version needs a one-time token:

1. Before the tag, make a crates.io API token limited to `quantize-files`, with the `publish-new` and `publish-update` scopes, and add it as the `CRATES_IO_BOOTSTRAP_TOKEN` secret of the GitHub `crates-io` environment. While the secret is set, the `crates-io` job publishes `quantize-files` with it, so one left in place by mistake keeps working until the token expires.
2. After the tag publishes `quantize-files`, add a trusted publisher in its crates.io settings, the same as `quantize`'s: owner `aksheyd`, repository `quantize`, workflow `wheels.yaml`, environment `crates-io`.
3. Then delete the secret and revoke the token.

If a publishing job fails, fix the cause and re-run the failed jobs. Don't push the tag again: rebuilt wheels can differ, and PyPI rejects a changed file under a name it already has. A re-run of the `crates-io` job skips a crate that crates.io already has at that version, and publishes the rest.

`include` paths must be rooted (`/LICENSE`, `/README.md`, `/src/**`) so files named `LICENSE` under `.venv` are not packed. `files/LICENSE` links to the root `LICENSE`, so `quantize-files` ships it too.

Clean tree + tests first.
