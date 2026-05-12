# Recommendations / follow-up work — issue #171

Captured during the work on PR #172. Each item is sized so it can be its
own issue + PR.

## High-value follow-ups (R1 — large feature gaps)

1. **Port the LSP server to Rust.** `js/src/rml-lsp.mjs` implements a
   Language Server Protocol server (`rml-lsp` binary). The Rust crate
   ships `rml`, `rml-check`, `rml-meta` but no `rml-lsp`. Need a Rust
   `rml-lsp` binary plus a Rust mirror of `js/tests/lsp.test.mjs`.
2. **Port the Isabelle exporter to Rust.** `js/src/rml-links.mjs`
   exposes `exportIsabelle()`. Mirror as `rust/src/isabelle_export.rs`
   with `rust/tests/isabelle_export_tests.rs` covering the four cases
   in `js/tests/isabelle-export.test.mjs`.
3. **Close the `rml-links` ↔ `rml` 66-test gap.** Audit the JS suite
   case-by-case and decide which tests belong on the Rust side. This
   is the largest single chunk of R1 and benefits from being split
   into a few focused PRs (e.g. one per feature area).

## CI/CD follow-ups (R2/R3)

4. **Add a 3-OS matrix to `tests.yml`.** Run `npm test` and
   `cargo test` on ubuntu / macos / windows like both templates do.
   Deferred from this PR for diff size; the workflow is structured so
   adding `strategy.matrix.os` is a few lines per job.
5. **Add a Node × Bun × Deno runtime matrix** to the JS half of
   `tests.yml`, mirroring the JS template's pattern.
6. **Bump existing workflows to v6 actions.** `bootstrap.yml`,
   `parity.yml`, `docker.yml`, `lint-english.yml` still pin
   `actions/checkout@v4` and (where used) `actions/setup-node@v4`.
7. **Add code coverage to CI.** `cargo-llvm-cov` + `codecov-action@v4`
   for Rust; `c8` or `node:test --experimental-test-coverage` for JS.
8. **Add lint/format gates to CI.** `cargo fmt --check`, `cargo
   clippy -- -D warnings`, ESLint, Prettier, `secretlint`, and the
   templates' 1500-line file-size check.
9. **Adopt the concurrency pattern across all workflows.** Pre-PR,
   only the new `tests.yml` has a `concurrency:` block. Apply the
   same group/cancel-in-progress pattern to the other five workflows.

## Process improvements (R1 long-term)

10. **Write a parity linter** (`scripts/check-test-parity.mjs`) that
    diffs test names between `js/tests/*.mjs` and `rust/tests/*.rs`
    and surfaces missing tests. Wire it into `tests.yml` as a soft
    check first, then promote to a required check once the corpus is
    clean. This mechanises R1 going forward so drift can't reappear.
11. **Add a CONTRIBUTING note** explaining that any change to one
    language requires a mirror change to the other, with the parity
    linter as the enforcement mechanism.
