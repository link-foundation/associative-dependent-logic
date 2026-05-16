# Evidence — CI/CD template review for issue #97

@konard's comment on issue #97 carries forward the practice
established in #171:

> Make sure everything covered with unit, integration and e2e tests,
> and executed in CI/CD. Use all the best practices from CI/CD
> templates (check full file tree to compare for all GitHub workflow
> and CI/CD scripts file), if the same issue is found in template
> report issue also in templates.

This file records the comparison between this repo's
`.github/workflows/` and the four
`link-foundation/*-ai-driven-development-pipeline-template` repos
*as of branch `issue-97-bbe597194dee`*. The bulk of the cross-template
comparison work was already done by
`docs/case-studies/issue-171/data/template-comparison.md`
and `docs/case-studies/issue-171/data/recommendations.md` (PR #172).
This document is the foundation-PR-specific delta on top of that.

## 1. Templates inspected

| Repo | Default branch | Last updated (UTC, observed 2026-05-15) | Workflows |
|------|----------------|-----------------------------------------|-----------|
| `link-foundation/js-ai-driven-development-pipeline-template` | `main` | 2026-05-15 | `release.yml`, `example-app.yml`, `links.yml` |
| `link-foundation/rust-ai-driven-development-pipeline-template` | `main` | (recent) | `release.yml` |
| `link-foundation/python-ai-driven-development-pipeline-template` | `main` | (recent) | `release.yml` |
| `link-foundation/csharp-ai-driven-development-pipeline-template` | `main` | (recent) | `release.yml`, `docs.yml` |

(Both Python and C# templates were added after #171 was filed; their
review is new in this PR.)

## 2. Workflow inventory (delta since issue #171)

| File | This repo | JS template | Rust template | Python template | C# template |
|------|-----------|-------------|---------------|-----------------|-------------|
| `release.yml` | (none — split across 5 files) | yes | yes | yes | yes |
| `tests.yml` | yes (issue #171) | (folded into `release.yml`) | (folded) | (folded) | (folded) |
| `links.yml` | (none) | yes | (none) | (none) | (none) |
| `docs.yml` / `api-docs.yml` / `example-app.yml` | `api-docs.yml` | `example-app.yml` | (in `release.yml`) | (in `release.yml`) | `docs.yml` |
| `bootstrap.yml` | yes (repo-specific) | (n/a) | (n/a) | (n/a) | (n/a) |
| `parity.yml` | yes (repo-specific) | (n/a) | (n/a) | (n/a) | (n/a) |
| `docker.yml` | yes (repo-specific) | (n/a) | (n/a) | (n/a) | (n/a) |
| `lint-english.yml` | yes (repo-specific) | (n/a) | (n/a) | (n/a) | (n/a) |

No new workflow is required by issue #97: the existing `tests.yml`
runs `npm test` and `cargo test` on every push / PR, which picks up
`js/tests/foundations.test.mjs`, `rust/tests/foundations_tests.rs`,
the extended `self-evaluator.test.mjs`, the extended
`self_evaluator_tests.rs`, and the `examples/foundation-*.lino`
replays automatically.

## 3. Action pinning audit

Action versions observed in the four templates and in this repo's
six workflow files:

| Action | JS template | Rust template | Python template | C# template | This repo (`tests.yml`) | This repo (other workflows) |
|--------|-------------|---------------|-----------------|-------------|-------------------------|------------------------------|
| `actions/checkout` | `@v6` | `@v6` | `@v4` | `@v4` | `@v6` | `@v6` (api-docs), `@v4` (bootstrap, docker, lint-english, parity) |
| `actions/setup-node` | `@v6` | (n/a) | (n/a) | (n/a) | `@v6` | `@v6` (api-docs), `@v4` (bootstrap, lint-english, parity) |
| `actions/setup-python` | (n/a) | (n/a) | `@v5` | (n/a) | (n/a) | (n/a) |
| `actions/setup-dotnet` | (n/a) | (n/a) | (n/a) | `@v4` | (n/a) | (n/a) |
| `dtolnay/rust-toolchain` | (n/a) | `@stable` | (n/a) | (n/a) | `@stable` | (n/a) |
| `Swatinem/rust-cache` | (n/a) | (none observed) | (n/a) | (n/a) | `@v2` | (n/a) |
| `actions/configure-pages` | (n/a) | `@v6` | (n/a) | (n/a) | (n/a) | `@v6` (api-docs) |
| `actions/upload-pages-artifact` | (n/a) | `@v5` | (n/a) | (n/a) | (n/a) | `@v5` (api-docs) |
| `actions/deploy-pages` | (n/a) | `@v5` | (n/a) | (n/a) | (n/a) | `@v5` (api-docs) |

### Reportable findings (templates)

**Finding T-1 — Python template lags on `actions/checkout` and `actions/setup-python`.**
The Python template pins `actions/checkout@v4` and
`actions/setup-python@v5` across all seven jobs in
`release.yml`, while the JS and Rust sibling templates have moved
to `actions/checkout@v6`. The two version families work, but the
templates should be aligned so end-users get a consistent
"newest stable" experience.
Recommended template PR: bump `actions/checkout` to `@v6` (or to
whatever the JS/Rust templates land on) across the Python template.
Suggested issue title for `link-foundation/python-ai-driven-development-pipeline-template`:
"Bump `actions/checkout` and `actions/setup-python` to align with
JS/Rust templates".

**Finding T-2 — C# template lags on `actions/checkout` and `actions/setup-dotnet`.**
Same as T-1 but for the C# template (`actions/checkout@v4`,
`actions/setup-dotnet@v4`). Recommended template PR: align with the
JS/Rust templates.
Suggested issue title for `link-foundation/csharp-ai-driven-development-pipeline-template`:
"Bump `actions/checkout` and `actions/setup-dotnet` to align with
JS/Rust templates".

These two findings are *not* introduced by this PR — they were
already present in the template repos at the start of this work. They
are recorded here so we can file them upstream once the foundation PR
lands, satisfying the "report issue also in templates" instruction.

### Reportable findings (this repo)

**Finding R-1 — Action-version drift across this repo's workflows.**
`bootstrap.yml`, `docker.yml`, `lint-english.yml`, and `parity.yml`
still use `actions/checkout@v4` and (where applicable)
`actions/setup-node@v4`. `tests.yml` and `api-docs.yml` use `@v6`.
This is the same gap recorded as recommendation #6 in
`docs/case-studies/issue-171/data/recommendations.md` and is
explicitly out of scope for the foundation PR (no behavioural
relationship to issue #97). The recommendation stands.

## 4. Concurrency pattern

| Source | Pattern |
|--------|---------|
| JS template `release.yml` | `cancel-in-progress: ${{ github.ref == 'refs/heads/main' }}` |
| Rust template `release.yml` | `cancel-in-progress: ${{ github.ref == 'refs/heads/main' }}` |
| Python template `release.yml` | `cancel-in-progress: true` |
| C# template `release.yml` | `cancel-in-progress: true` |
| This repo `tests.yml` | `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}` |
| This repo `api-docs.yml` | `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}` |
| This repo other workflows | (none) |

**Finding T-3 — Templates disagree on concurrency semantics.**
The JS and Rust templates set `cancel-in-progress: ${{ github.ref ==
'refs/heads/main' }}` — i.e., cancel on `main` pushes, queue on PR
pushes. The Python and C# templates set
`cancel-in-progress: true` — i.e., always cancel.

Neither matches the more typical developer pattern of "always finish
main runs, cancel duplicate PR runs", which is what this repo's
`tests.yml` and `api-docs.yml` adopt
(`cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`).

The four templates should converge on one of these three patterns.
The "cancel PR duplicates, never cancel main" pattern this repo uses
is the safest default — the others can lead to surprising cancelled
main builds (JS/Rust templates) or surprising lost main-build work
(Python/C# templates).

Suggested upstream issue title (file in all four templates once the
foundation PR lands): "Standardise `concurrency.cancel-in-progress`
to `${{ github.ref != 'refs/heads/main' }}`".

## 5. OS / runtime matrix

| Aspect | JS template | Rust template | Python template | C# template | This repo `tests.yml` |
|--------|-------------|---------------|-----------------|-------------|------------------------|
| OS matrix | `[ubuntu, macos, windows]` | `[ubuntu, macos, windows]` | `[ubuntu, macos, windows]` | `[ubuntu, macos, windows]` | `ubuntu-latest` only |
| Node.js | `'24.x'` | (n/a) | (n/a) | (n/a) | `'20'` |
| Python | (n/a) | (n/a) | `'3.13'` | (n/a) | (n/a) |
| .NET | (n/a) | (n/a) | (n/a) | `'8.0.x'` | (n/a) |
| Rust toolchain | (n/a) | stable + clippy + rustfmt | (n/a) | (n/a) | stable |

The OS matrix and Node version bump are tracked in
`docs/case-studies/issue-171/data/recommendations.md` (items #4
and the Node-bump implication of #6) as deferred follow-ups. They
have no behavioural relationship to issue #97 and stay deferred.

## 6. Quality gates (templates → this repo)

The four templates collectively introduce these quality gates which
this repo does not yet run in CI:

| Gate | JS | Rust | Python | C# | This repo (foundation-PR scope?) |
|------|----|------|--------|-----|----------------------------------|
| `eslint --max-warnings 0` | yes | (n/a) | (n/a) | (n/a) | deferred (issue-171 #8) |
| `prettier --check` | yes | (n/a) | (n/a) | (n/a) | deferred (issue-171 #8) |
| `secretlint --maskedTokens` | yes | (n/a) | (n/a) | (n/a) | deferred (issue-171 #8) |
| `cargo fmt --check` | (n/a) | yes | (n/a) | (n/a) | deferred (issue-171 #8) |
| `cargo clippy -- -D warnings` | (n/a) | yes | (n/a) | (n/a) | deferred (issue-171 #8) |
| `ruff check` / `black --check` | (n/a) | (n/a) | yes | (n/a) | n/a (no Python here) |
| `dotnet format --verify-no-changes` | (n/a) | (n/a) | (n/a) | yes | n/a (no .NET here) |
| File-size check (line limit 1500) | yes | yes | yes | yes | deferred (issue-171 #8) |
| Coverage upload (Codecov) | yes (c8) | yes (`cargo-llvm-cov`) | yes | yes | deferred (issue-171 #7) |
| Changeset / changelog gate | yes | yes | yes | yes | deferred (issue-171 #7-9) |

None of these are blockers for issue #97 — the foundation surface is
covered by the existing `tests.yml` job pair. The deferred items
remain captured in `docs/case-studies/issue-171/data/recommendations.md`.

## 7. Decisions for this PR

- **Adopt now:** Nothing new — the existing `tests.yml` from PR #172
  already runs both engines' full test suites on every push and PR,
  which picks up all the foundation tests (`foundations.test.mjs`,
  `foundations_tests.rs`, extended `self-evaluator.test.mjs`, extended
  `self_evaluator_tests.rs`) without workflow changes.
- **Report upstream after this PR lands:**
  - T-1 (Python template — bump `actions/checkout` + `actions/setup-python`).
  - T-2 (C# template — bump `actions/checkout` + `actions/setup-dotnet`).
  - T-3 (all four templates — converge on
    `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`).
- **Defer (still tracked in `docs/case-studies/issue-171/data/recommendations.md`):**
  3-OS matrix, Node bump to 24.x, ESLint/Prettier/secretlint, rustfmt,
  clippy, file-size check, coverage upload, changeset workflow,
  parity linter, CONTRIBUTING note.
- **Out of scope for issue #97:** rewriting the existing
  `api-docs.yml` / `parity.yml` / `bootstrap.yml` / `docker.yml` /
  `lint-english.yml`. They are unaffected by the foundation surface.

## 8. Verification

The CI workflow already runs `npm test` and `cargo test` on every push
and PR; this can be confirmed by viewing recent runs of the `tests`
workflow on the `issue-97-2fa46f510db9` branch. The eight new JS
tests in `foundations.test.mjs` and the eight new Rust tests in
`foundations_tests.rs`, plus the extended self-evaluator coverage,
all execute under the existing `tests` workflow. No additional
workflow changes are needed for issue #97.

### 8.1 Phase 2–9 follow-up coverage (PR #175)

The subsequent phases that landed on PR #175 add the following test
files to the same suite — no workflow changes required:

| Phase | JS test file | Rust test file |
|-------|--------------|----------------|
| 2 — equality provenance | `js/tests/foundations.test.mjs` (extended) | `rust/tests/foundations_tests.rs` (extended) |
| 3 — proof-object replay | `js/tests/proof-substrate.test.mjs` | `rust/tests/proof_substrate_tests.rs` |
| 4 — links-defined truth tables | `js/tests/foundations.test.mjs` (extended) | `rust/tests/foundations_tests.rs` (extended) |
| 6 — pure-links strict mode | `js/tests/pure-links-strict.test.mjs` | `rust/tests/pure_links_strict_tests.rs` |
| 7 — dependency-graph traversal | `js/tests/dependency-graph.test.mjs` | `rust/tests/dependency_graph_tests.rs` |
| 8 — carrier enforcement | `js/tests/foundations.test.mjs` (extended) | `rust/tests/foundations_tests.rs` (extended) |
| 9 — `mtc-anum` experimental profile | `js/tests/mtc-anum.test.mjs` | `rust/tests/mtc_anum_tests.rs` |

All of these are picked up automatically by `npm test` and
`cargo test --all-targets`, which the existing `tests.yml` workflow
already runs on every push and PR.

### 8.2 Re-audit findings (2026-05-16)

A second pass against the four templates (described in this file
plus a fresh independent review documented in this PR) confirms no
new gaps were introduced by the Phase 2–9 work. The reportable
findings remain T-1, T-2, T-3, and the pre-existing CI-debt items
tracked in `docs/case-studies/issue-171/data/recommendations.md`.
The pattern of *path-filtered triggers + safer concurrency
(`!= 'refs/heads/main'`)* this repo has on its `tests.yml` and
`api-docs.yml` is still ahead of all four templates and is worth
contributing upstream once the foundation PR has merged.
