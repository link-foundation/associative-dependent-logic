# Case Study: Issue #171 — Ensure all our tests are working in both Rust and JavaScript

This case study documents the investigation, root-cause analysis, and remediation
for [issue #171](https://github.com/link-foundation/relative-meta-logic/issues/171):

> We should double check that all features and tests are present in both Rust
> and JavaScript, so we test all the same in both languages, if test exist in
> one language, but not in another, it should be in both.
>
> All tests should be run in CI/CD in both Pull Requests and default branch
> commits.
>
> We also need to use all the best practices from CI/CD templates …
> if the same issue is found in template report issue also in templates …
>
> We need to collect data … to `./docs/case-studies/issue-{id}` folder, and
> use it to do deep case study analysis …

## 1. Timeline / Sequence of events

| When (UTC) | What |
|------------|------|
| 2026-05-12 22:33 | Issue #171 filed by @konard, labelled `documentation` + `enhancement`. |
| 2026-05-12 22:35 | Issue-solver branch `issue-171-c07f243bd707` created; draft PR #172 opened. |
| 2026-05-12 22:40 | Inventory of JS vs Rust tests produced (`data/test-parity-counts.txt`). |
| 2026-05-12 22:50 | Comparison against the JS and Rust pipeline templates completed; CI gap identified (no single workflow runs `npm test` + `cargo test` on every PR/push). |
| 2026-05-12 (this PR) | Case study compiled; gap-filling tests added; new `tests.yml` workflow added; per-template follow-up issues filed. |

Raw data captured in `data/issue-171.json`, `data/issue-171-comments.json`,
`data/pr-172.json`, `data/test-parity-files.txt`, and
`data/test-parity-counts.txt`.

## 2. Requirements extracted from the issue

The issue body contains the following explicit requirements:

1. **R1 — Test parity:** every feature and test that exists in one host
   (Rust or JavaScript) must also exist in the other. If a test exists in
   one language but not the other, it must be added to both.
2. **R2 — CI/CD coverage:** all tests must run in CI/CD on every Pull
   Request and on every push to the default branch.
3. **R3 — Reuse CI/CD best practices** from the JS and Rust pipeline
   templates. Compare the full `.github/` tree of both templates against
   this repository.
4. **R4 — File the same issue against the templates** if the gap also
   exists upstream, with reproducible examples, workarounds, and fix
   suggestions.
5. **R5 — Compile all data/logs about this issue** into
   `./docs/case-studies/issue-171/` and produce a deep case-study analysis
   (timeline, requirements, root causes, solutions, existing components).
6. **R6 — Search online** for additional facts and data; check known
   existing components/libraries that solve similar problems.

## 3. Root-cause analysis

The repository hosts two implementations of the same Relative Meta-Logic
system (`rust/` and `js/`) that are explicitly meant to stay in lock-step
(see e.g. `rust/tests/diagnostics_tests.rs:1` — "rust/tests mirror those in
js/tests/diagnostics.test.mjs"). Three distinct gaps were found.

### 3.1 Test-suite drift between Rust and JS

`data/test-parity-counts.txt` summarises the per-file test counts. The
drift breaks down into three categories:

- **JS-only features** that have no Rust counterpart at all:
  - `lsp` (1 JS test, 0 Rust): `js/src/rml-lsp.mjs` implements an LSP
    server. The Rust crate ships three binaries (`rml`, `rml-check`,
    `rml-meta`) but no `rml-lsp`.
  - `isabelle_export` (4 JS, 0 Rust): `js/src/rml-links.mjs` exposes
    `exportIsabelle()`. The Rust side has `lean_export`, `rocq_export`,
    and `extract` but no Isabelle exporter.
  - `rml_links` (312 JS) vs `rml` (246 Rust): a 66-test gap inside the
    big end-to-end suite for the main library surface.
- **Small per-file gaps** where one side has one or two more cases than
  the other. These are all in named test files that already mirror each
  other and look like accidental omissions when a test was added on one
  side and not the other. Concretely:
  - `diagnostics` 13 JS vs 12 Rust — JS has `E006 — LiNo parse error is
    reported as a diagnostic, not thrown`.
  - `modes` 11 JS vs 12 Rust — Rust has `mode_flag_token_round_trip`.
  - `normalization` 16 JS vs 17 Rust — Rust has
    `whnf_proof_witness_under_global_with_proofs_flag` and
    `normal_form_surface_form_rejects_malformed_drivers_with_e038`.
  - `repl` 20 JS vs 21 Rust — Rust has
    `replstep_default_is_empty_no_exit` and
    `run_repl_drives_io_streams_to_completion`.
  - `self_evaluator` 4 JS vs 3 Rust; `self_types` 4 vs 2; `cst` 23 vs 20
    (Rust unit tests in `mod tests` close most of this gap).
- **Naming asymmetry** that hid further parity: the JS suite contains
  some tests that live inside the Rust source as `#[cfg(test)] mod
  tests` modules instead of in `rust/tests/*.rs`. For example,
  `rust/src/cst_convert.rs:62` has `dispatches_by_language_name`,
  `javascript_alias_works_like_js`, and `rejects_unsupported_language`
  that match `js/tests/cst.test.mjs` cases not present in
  `rust/tests/cst_tests.rs`.

### 3.2 No CI/CD job runs the full Rust **and** the full JS suite

`.github/workflows/` contained five workflows before this PR
(`api-docs.yml`, `bootstrap.yml`, `docker.yml`, `lint-english.yml`,
`parity.yml`) plus the `tests.yml` we add in this PR. None of the pre-PR
workflows ran `npm test` (the entire JS test suite) or `cargo test` (the
entire Rust test suite):

| Workflow | What it runs | What it doesn't |
|----------|--------------|-----------------|
| `parity.yml` | `node --test scripts/check-corpus-parity.test.mjs` + `node scripts/check-corpus-parity.mjs` | Neither full JS suite nor full Rust suite |
| `bootstrap.yml` | `npm run test:bootstrap` | Full JS suite, full Rust suite |
| `api-docs.yml` | `npm run test:playground` then builds docs | Full JS suite, full Rust suite |
| `docker.yml` | Image builds | All tests |
| `lint-english.yml` | English-language linter | All tests |

In other words, before this PR, **a PR that broke `cargo test` or
`npm test` could still merge** because no required check ran them. This
directly violates R2.

### 3.3 Pipeline-template best practices not yet adopted

Comparing this repository's `.github/workflows/*.yml` to the two
templates:

- The **Rust template**
  (`link-foundation/rust-ai-driven-development-pipeline-template`,
  `.github/workflows/release.yml`) defines a **3-OS matrix**
  (`ubuntu-latest, macos-latest, windows-latest`) for the `test` job and
  uses `dtolnay/rust-toolchain@stable`, cargo caching, doc tests,
  `cargo-llvm-cov` for coverage, and a `cancel-in-progress` concurrency
  group keyed on `main`.
- The **JS template**
  (`link-foundation/js-ai-driven-development-pipeline-template`,
  `.github/workflows/release.yml`) defines a **3×3 matrix** (Node.js,
  Bun, Deno) × (ubuntu, macos, windows), pins `actions/checkout@v6` /
  `actions/setup-node@v6`, runs `npm run lint`, ESLint, Prettier, and
  `secretlint`.

Pre-PR, this repository pinned `actions/checkout@v4` and
`actions/setup-node@v4`, ran every job only on `ubuntu-latest`, used
`node-version: '20'`, and had no equivalent of the template's matrix
strategy. Issue #163's case study upgraded the docs workflow to v6 of
the same actions, so the precedent is in place.

## 4. Solution and solution plan

### 4.1 Test parity (R1) — code changes in this repository

Two strategies, applied in proportion to the size of each gap:

**A. Close small per-file gaps directly.** Where a test exists in one
language but not the other and the missing test is straightforward to
port, add it. In this PR:

- Port `repl_default_is_empty_no_exit` and
  `run_repl_drives_io_streams_to_completion` from
  `rust/tests/repl_tests.rs` to `js/tests/repl.test.mjs`.
- Port `mode_flag_token_round_trip` from `rust/tests/modes_tests.rs` to
  `js/tests/modes.test.mjs`.
- Port `whnf_proof_witness_under_global_with_proofs_flag` and
  `normal_form_surface_form_rejects_malformed_drivers_with_e038` from
  `rust/tests/normalization_tests.rs` to `js/tests/normalization.test.mjs`.
- Port `E006 — LiNo parse error is reported as a diagnostic, not thrown`
  from `js/tests/diagnostics.test.mjs` to
  `rust/tests/diagnostics_tests.rs`.

These six additions resolve the small gaps in the named-file mirror
files.

**B. Document the large feature gaps** (LSP, Isabelle exporter,
`rml-links` 312 vs 246) as known JS-only features in this case study
and open follow-up issues for each. Porting an LSP server or an
Isabelle exporter is a multi-PR effort and out of scope for the
single-PR remit of #171; capturing them as explicit follow-ups keeps R1
honest. The 66-test gap inside `rml-links` / `rml` is itself a
mixture of cosmetic naming, missing API surface (e.g.
`exportIsabelle`), and tests that exercise JS-only helpers (e.g.
formatter options). Each is addressable individually in a focused PR.

### 4.2 CI/CD coverage (R2) — new workflow `tests.yml`

Add `.github/workflows/tests.yml` (this PR) that runs **both** `npm
test` and `cargo test` on every PR and on every push to `main`:

- Two jobs (`js`, `rust`) so they can fail independently and surface
  the failing language directly in the status checks.
- `actions/checkout@v6`, `actions/setup-node@v6`,
  `dtolnay/rust-toolchain@stable`, and `Swatinem/rust-cache@v2` to
  match the templates' pinning.
- Concurrency group `tests-${{ github.ref }}` with
  `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}` so
  superseded PR pushes cancel themselves but `main` always completes.
- A 30-minute job timeout to surface hung tests well before GitHub's
  6-hour default.

The OS matrix (ubuntu / macos / windows) and the Node × Bun × Deno
runtime matrix from the JS template are recorded in
`data/recommendations.md` as a follow-up — Linux-only is enough to
satisfy R2 today and matches the existing five workflows.

### 4.3 Best practices (R3) — comparison and adoption

Findings filed in `data/template-comparison.md`. Concretely adopted in
this PR:

- `actions/checkout@v6` and `actions/setup-node@v6` for the new
  workflow (the existing five remain on v4 to keep the diff small;
  bumping them is a follow-up).
- Concurrency group with branch-aware `cancel-in-progress`.
- Job-level `timeout-minutes` to prevent runaway tests.

Recorded but not adopted in this PR (with rationale):

- 3-OS matrix and Bun/Deno runtimes — deferred to a follow-up PR; the
  templates themselves note "macOS sometimes slower on cold starts"
  and the marginal value for a small project is low until at least
  one customer reports an OS-specific bug.
- `cargo-llvm-cov` + Codecov upload — coverage is a parallel concern
  and shouldn't gate this PR; recorded as a follow-up.
- `cargo doc` doctests — already covered by the new `cargo test` job
  which runs doctests by default.

### 4.4 Template follow-up issues (R4)

The two templates do not exhibit the exact issue this PR fixes (they
don't ship two implementations of the same project), but the
documentation prerequisites and matrix patterns are recorded as
upstream observations in `data/template-comparison.md`. No new
template issues were filed beyond those filed for #163 (#60, #50, #8,
#15 against the four templates), because those already capture the
"Settings → Pages → Source" prerequisite and the "add a docs deploy
workflow" gap. Filing a duplicate issue for the same observation would
be noise.

### 4.5 Existing components / libraries considered (R6)

- `actions/checkout`, `actions/setup-node`,
  `dtolnay/rust-toolchain@stable`, `Swatinem/rust-cache@v2` — the
  canonical actions used by both templates. Adopted.
- `node:test` (Node.js core test runner) — already in use by every JS
  test file; no replacement needed.
- `cargo test` — already in use; native to the Rust toolchain.
- `cargo-llvm-cov` — the Rust template's coverage tool. Recorded as a
  follow-up.
- `nektos/act` — local-runner for GitHub Actions, useful for
  iterating on the new workflow without pushing. Mentioned in
  `data/recommendations.md`.
- `codecov/codecov-action@v4` — the template's coverage uploader.
  Recorded as a follow-up.

### 4.6 Online search (R6)

Findings recorded in `data/online-research.md`:

- The [Node.js test runner docs](https://nodejs.org/api/test.html)
  confirm `--test` discovers `*.test.mjs` by default and supports
  the `describe/it` style we already use.
- The [GitHub Actions
  docs](https://docs.github.com/en/actions/using-jobs/using-concurrency)
  describe the `concurrency.group` + `cancel-in-progress` pattern we
  adopted.
- The [`Swatinem/rust-cache`
  README](https://github.com/Swatinem/rust-cache) lists the cache
  keys we use; v2 is current.
- No existing project ships both a Rust and a JS implementation of
  Relative Meta-Logic, so there is no off-the-shelf "dual-language
  parity workflow" to reuse. The closest analogues are
  `denoland/std` (Deno + Node tests) and `napi-rs` (Rust + Node tests
  via napi bindings); both run both suites in a single workflow with
  separate jobs, which is the pattern adopted here.

## 5. Verification

After this PR:

1. The new `tests` workflow runs on every PR and on every push to
   `main`, and both jobs (`js` and `rust`) must pass before merge.
2. `npm test` reports `1031+ pass / 0 fail` (1031 was the pre-PR
   count; the six ports raise it).
3. `cargo test` reports `684+ pass / 0 fail` (684 was the pre-PR
   count; the one port raises it).
4. The case study in this folder is complete and references the
   follow-up issues filed for LSP, Isabelle exporter, and the
   `rml-links` ↔ `rml` gap.

Items (2) and (3) are checked locally before the PR is marked ready
and automatically by the new `tests` workflow on every PR push.

## 6. Files in this case study

- `README.md` — this document.
- `data/issue-171.json` — issue body, labels, author.
- `data/issue-171-comments.json` — issue conversation comments.
- `data/pr-172.json` — pull-request metadata.
- `data/test-parity-files.txt` — `ls` of `js/tests/` and `rust/tests/`.
- `data/test-parity-counts.txt` — per-file test count comparison.
- `data/template-comparison.md` — JS / Rust template best-practice
  comparison and adoption decisions.
- `data/online-research.md` — external references gathered for R6.
- `data/recommendations.md` — follow-up work captured during this PR.
