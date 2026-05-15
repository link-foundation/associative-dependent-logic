# Pipeline-template comparison — issue #171

Compared the `.github/` tree of this repository against the two templates
referenced in the issue:

- `link-foundation/js-ai-driven-development-pipeline-template`
- `link-foundation/rust-ai-driven-development-pipeline-template`

## Workflow inventory

| File | This repo | JS template | Rust template |
|------|-----------|-------------|---------------|
| `release.yml` (full pipeline) | (none — split across 5 files) | yes | yes |
| `tests.yml` (full suite gate) | **added by this PR** | (folded into `release.yml`) | (folded into `release.yml`) |
| `links.yml` (broken-link scan) | (none) | yes | (none) |
| `example-app.yml` / `api-docs.yml` (docs deploy) | `api-docs.yml` | `example-app.yml` | (in `release.yml`) |
| `bootstrap.yml` | yes (repo-specific) | (n/a) | (n/a) |
| `parity.yml` | yes (repo-specific) | (n/a) | (n/a) |
| `docker.yml` | yes (repo-specific) | (n/a) | (n/a) |
| `lint-english.yml` | yes (repo-specific) | (n/a) | (n/a) |

## Action-version pinning

| Action | This repo (pre-PR) | This repo (post-PR `tests.yml`) | JS template | Rust template |
|--------|--------------------|---------------------------------|-------------|----------------|
| `actions/checkout` | `@v4` | `@v6` | `@v6` | `@v6` |
| `actions/setup-node` | `@v4` | `@v6` | `@v6` | (n/a) |
| `dtolnay/rust-toolchain` | (none) | `stable` | (n/a) | `stable` |
| `Swatinem/rust-cache` | (none) | `@v2` | (n/a) | `@v2` |
| `actions/configure-pages` | `@v6` (fixed in #163) | (n/a) | `@v6` | `@v6` |
| `actions/upload-pages-artifact` | `@v5` (fixed in #163) | (n/a) | `@v5` | `@v5` |
| `actions/deploy-pages` | `@v5` (fixed in #163) | (n/a) | `@v5` | `@v5` |

## OS / runtime matrix

| Aspect | This repo | JS template | Rust template |
|--------|-----------|-------------|----------------|
| OS matrix (tests) | `ubuntu-latest` only | `[ubuntu, macos, windows]` | `[ubuntu, macos, windows]` |
| Node.js | `'20'` | `'24.x'` | (n/a) |
| Runtime matrix | (single: Node) | `[node, bun, deno]` | (n/a) |
| Rust toolchain | (n/a in pre-PR) | (n/a) | stable + clippy + rustfmt |

## Concurrency / cancellation

- Both templates use:
  ```yaml
  concurrency:
    group: ${{ github.workflow }}-${{ github.ref }}
    cancel-in-progress: ${{ github.ref == 'refs/heads/main' }}
  ```
  (Inverted from the default — main pushes always finish; PR pushes
  cancel the previous run.)
- This repo (pre-PR) had no `concurrency:` block in any workflow.
- This repo (post-PR `tests.yml`) adopts the same pattern with a
  branch-aware `cancel-in-progress`.

## Linting and quality gates present in templates but not here

| Gate | JS template | Rust template | This repo |
|------|-------------|---------------|-----------|
| `eslint` | yes | (n/a) | not in CI |
| `prettier --check` | yes | (n/a) | not in CI |
| `secretlint` | yes | (n/a) | not in CI |
| `rustfmt --check` | (n/a) | yes | not in CI |
| `cargo clippy -- -D warnings` | (n/a) | yes | not in CI |
| File-size check | yes (line limit 1500) | yes (line limit 1500) | not in CI |
| Changeset / changelog fragment check | yes | yes | not in CI |
| Coverage upload | yes | yes (`cargo-llvm-cov`) | not in CI |
| Doc tests | (n/a) | yes | covered by `cargo test` in new `tests.yml` |

## Decisions for this PR

- **Adopt now:** `tests.yml` with both `npm test` and `cargo test`,
  v6 pinning of `actions/checkout` and `actions/setup-node`, the
  concurrency pattern, job-level `timeout-minutes`.
- **Defer (recorded in `recommendations.md`):** OS matrix, Bun/Deno
  matrix, ESLint/Prettier/secretlint/rustfmt/clippy/file-size in CI,
  coverage upload, changeset workflow.
- **Out of scope:** rewriting the existing `api-docs.yml` /
  `parity.yml` / `bootstrap.yml` / `docker.yml` /
  `lint-english.yml` — `tests.yml` lives alongside them and a
  consolidation pass is a separate concern.
