# Online research — issue #171

External references gathered for R6.

## Node.js test runner

- Docs: <https://nodejs.org/api/test.html>
- Confirms `node --test` discovers `*.test.{js,mjs,cjs}` by default and
  supports the `describe(...)` / `it(...)` style we already use across
  `js/tests/`.
- Stable as of Node.js 20 LTS (which we pin in `tests.yml`).

## GitHub Actions concurrency

- Docs: <https://docs.github.com/en/actions/using-jobs/using-concurrency>
- Describes the
  `concurrency: { group: …, cancel-in-progress: … }` pattern. Both
  pipeline templates use this with a branch-aware `cancel-in-progress`
  expression to avoid cancelling `main` pushes; `tests.yml` adopts the
  same.

## Rust toolchain action

- Repo: <https://github.com/dtolnay/rust-toolchain>
- The Rust template uses `@stable`; pinning to `@stable` keeps the
  CI on the latest stable channel without a date-locked toolchain.

## Rust build cache

- Repo: <https://github.com/Swatinem/rust-cache>
- v2 is current. The Rust template uses it to cache `~/.cargo` and
  `target/` keyed on `Cargo.lock`. We adopt it in `tests.yml`.

## Cross-language parity patterns

Searched for projects that ship both a Rust implementation and a JS
implementation of the same library and run both suites in CI:

- `napi-rs` (<https://github.com/napi-rs/napi-rs>) — uses
  GitHub Actions matrix with separate jobs per language;
  `cargo test` for Rust, `yarn test` for JS, no shared test runner.
- `swc` (<https://github.com/swc-project/swc>) — Rust core + JS API;
  separate `cargo test` and `yarn test` jobs.
- `denoland/std` (<https://github.com/denoland/std>) — runs the same
  test files under Node and Deno via a matrix; not a parallel
  reimplementation but the matrix pattern is reused here as a future
  recommendation.

None of these projects use a single tool to "diff" tests between
languages; they all rely on the conventional approach of running both
native test suites and treating each as a required check. That is the
pattern this PR adopts.

## Why no off-the-shelf "parity linter"?

We searched for tools that compare a Rust and a JS test suite for
parity and surface missing tests. We found none. The closest is
`scripts/check-corpus-parity.mjs` in this repository, which compares
the *outputs* of the two implementations on a shared corpus — not the
*tests themselves*. A future tool that diffs test names and surfaces
"present in JS, missing in Rust" (and vice versa) would mechanise
R1 going forward; recorded in `recommendations.md`.
