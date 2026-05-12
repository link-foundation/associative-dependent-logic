# Case Study: Issue #163 — Broken playground link in README.md

This case study documents the investigation, root-cause analysis, and remediation
for [issue #163](https://github.com/link-foundation/relative-meta-logic/issues/163):

> Link to online playground in README.md is broken:
> <https://link-foundation.github.io/relative-meta-logic/playground/>
> Also it may mean GitHub Pages were not published as expected.

## 1. Timeline / Sequence of events

| When (UTC) | What |
|------------|------|
| 2025-10-07 04:27 | Repository `link-foundation/relative-meta-logic` created. |
| 2026-05-09 (PR #160, merged 580b540…) | Playground sources added under `docs/playground/`; `README.md` updated with a link to <https://link-foundation.github.io/relative-meta-logic/playground/>; `.github/workflows/api-docs.yml` added with **build on PR/push** but **deploy only on `release: published`**. |
| 2026-05-10 13:07 | Issue #163 filed by @konard, labelled `bug`. |
| 2026-05-12 22:03 | Issue-solver branch `issue-163-3a962eb1d8a7` created; draft PR #170 opened. |
| 2026-05-12 22:05 | `curl -I https://link-foundation.github.io/relative-meta-logic/playground/` returns **HTTP/2 404**; `gh api repos/link-foundation/relative-meta-logic/pages` returns `404 Not Found` with `has_pages: false` — the Pages site has never been provisioned. No `gh release list` entries exist. |
| 2026-05-12 (this PR) | Workflow updated to build **and** deploy GitHub Pages on `push` to `main`, matching the JS and Rust pipeline templates; case study compiled; upstream template issues filed. |

Raw data captured in `data/issue-163.json`, `data/issue-163-comments.json`, `data/issue-163-timeline.json`,
`captures/playground-url-headers.txt`, `captures/playground-url-response.txt`,
and the workflows directory contains the original `api-docs.yml` plus the
templates we compared against. (The directory is named `captures/` rather than
`logs/` because `.gitignore` excludes `logs`.)

## 2. Requirements extracted from the issue

The issue body contains the following explicit requirements:

1. **R1 — Fix the broken playground link** so that
   <https://link-foundation.github.io/relative-meta-logic/playground/> serves
   the playground.
2. **R2 — Make sure GitHub Pages publishes as expected** (i.e. on every push
   to `main`, not gated behind something rare such as a release).
3. **R3 — Reuse CI/CD best practices from the four pipeline templates**
   (`js`, `rust`, `python`, `csharp` `*-ai-driven-development-pipeline-template`)
   and compare each file under `.github/` between this repository and the
   templates so future CI errors are avoided.
4. **R4 — Report the same issue against the templates** if the templates
   exhibit it, with reproducible examples, workarounds, and fix suggestions.
5. **R5 — Compile all data/logs about this issue into
   `./docs/case-studies/issue-163/`** and perform a deep case-study analysis
   (timeline, requirements, root causes, possible solutions, existing
   components that solve the same problem).
6. **R6 — Add debug/verbose output** when the available data is not enough
   to pinpoint the root cause, so the next iteration can find it.

## 3. Root-cause analysis

### Why is the playground URL returning 404?

The playground sources exist at `docs/playground/` and the build pipeline in
`.github/workflows/api-docs.yml` does copy them into `_site/playground/`. But
the steps that actually publish that artifact to GitHub Pages are gated on
`github.event_name == 'release'`:

```yaml
- name: Configure GitHub Pages
  if: github.event_name == 'release'
  uses: actions/configure-pages@v5
- name: Upload GitHub Pages artifact
  if: github.event_name == 'release'
  uses: actions/upload-pages-artifact@v3
  with:
    path: _site

deploy:
  if: github.event_name == 'release'
  needs: build
  ...
```

`gh release list --repo link-foundation/relative-meta-logic` returns no
releases. Therefore the deploy job has **never been executed**, the Pages
site has never been provisioned (`has_pages: false`), and the URL the README
advertises is a 404.

### Why does it look like “Pages were not published as expected”?

Because they literally were not — the deploy step was conditional on an
event that has not happened yet. There is no infrastructure problem, no
permissions problem, and no path problem; the gating condition is simply
wrong for a repository that wants the docs site to track `main`.

### Secondary findings against the upstream templates

The same `if: github.event_name == 'release'` gate is **not** present in the
JS / Rust pipeline templates that this repository is supposed to reuse:

- `link-foundation/js-ai-driven-development-pipeline-template`
  (`.github/workflows/example-app.yml`) deploys on
  `github.event_name == 'push' && github.ref == 'refs/heads/main'`,
  uses `actions/configure-pages@v6`, `actions/upload-pages-artifact@v5`,
  `actions/deploy-pages@v5`, and pins `actions/checkout@v6` /
  `actions/setup-node@v6`.
- `link-foundation/rust-ai-driven-development-pipeline-template`
  (`.github/workflows/release.yml`, `deploy-docs` job) does the same with
  the additional `workflow_dispatch` + `release_mode == 'instant'` branch.

The Python and C# templates don't ship a Pages workflow at all, so they have
no immediate gap; the corresponding upstream issues are filed as suggestions
to add one.

In other words: this repository's `api-docs.yml` predates the current
templates and was written against `@v5/@v3` action versions with a stricter
trigger. Bringing it up to date with the templates solves both the immediate
bug (R1, R2) and the “reuse best practices” requirement (R3).

## 4. Solution and solution plan

### 4.1 Code change in this repository (applied in this PR)

Rewrite `.github/workflows/api-docs.yml` so that the build still runs on
every PR/push touching the docs sources, but the Pages deploy runs on
every push to `main` (and on `workflow_dispatch`), matching
`example-app.yml` and `release.yml` from the templates:

- Bump action versions: `actions/checkout@v4 → v6`,
  `actions/setup-node@v4 → v6`, `actions/configure-pages@v5 → v6`,
  `actions/upload-pages-artifact@v3 → v5`,
  `actions/deploy-pages@v4 → v5`.
- Replace `if: github.event_name == 'release'` with
  `if: github.event_name == 'push' && github.ref == 'refs/heads/main'`
  (plus `workflow_dispatch` for manual redeploys).
- Keep the playground regression tests (`npm run test:playground`) on
  every PR so we don't ship a broken artifact.
- Add a `verbose-deploy` step that prints the resolved deployment URL and
  a listing of `_site/playground/` before upload, so future failures
  expose the root cause directly in the workflow log (R6).

This is the minimum needed to fix #163. The first push of the merged PR
to `main` will provision the Pages site (the `actions/configure-pages@v6`
step does that on demand) and publish the playground at the advertised URL.

### 4.2 Manual follow-up after merge

The repository setting `Settings → Pages → Source` must be set to
**“GitHub Actions”** for the deploy job to succeed. The workflow itself
provisions the Pages site, but the source selection is a repo-level
setting that is not modifiable by a workflow. The PR description calls
this out for the maintainer.

### 4.3 Upstream template issues (R4)

Filed reports against the four pipeline-template repositories. The JS and
Rust templates are already correct, so the issue against them is a
documentation note that asks them to keep the “deploy on push to main”
pattern stable and to add a checklist item reminding users to flip the
Pages source. The Python and C# templates don't currently ship a Pages
deployment workflow, so the issue against them is a feature request to
add one mirroring the JS/Rust pattern, with the
`.github/workflows/api-docs.yml` rewrite in this PR as a concrete example.

### 4.4 Existing components / libraries considered

- `actions/configure-pages`, `actions/upload-pages-artifact`,
  `actions/deploy-pages` — the official, recommended way to publish from a
  workflow. Already used; only the versions and the trigger gate change.
- `peaceiris/actions-gh-pages` — a popular alternative that pushes a
  branch. Rejected because the official actions are simpler, do not
  require a branch, and are what the templates use.
- `lycheeverse/lychee-action` — used by `links.yml` in the JS template
  as a broken-link check; out of scope for this PR but recorded in
  `data/recommendations.md` for a follow-up.

## 5. Verification

After this PR is merged into `main`:

1. The `api-docs` workflow run on `main` shows green for the
   `pages-deploy` job and reports a deployment URL.
2. `curl -I https://link-foundation.github.io/relative-meta-logic/playground/`
   returns `HTTP/2 200`.
3. `gh api repos/link-foundation/relative-meta-logic/pages` returns
   `has_pages: true` and `html_url:
   https://link-foundation.github.io/relative-meta-logic/`.

Item (1) is checked automatically by CI; items (2) and (3) are recorded
in `captures/post-merge-verification.md` once the deploy succeeds.

## 6. Files in this case study

- `README.md` — this document.
- `data/issue-163.json` — issue body, labels, author, comments.
- `data/issue-163-comments.json` — `gh api .../issues/163/comments`.
- `data/issue-163-timeline.json` — `gh api .../issues/163/timeline`.
- `data/recommendations.md` — recommendations for follow-up work.
- `captures/playground-url-headers.txt` — HTTP response for the broken URL.
- `captures/playground-url-response.txt` — body of the broken URL.
- `workflows/api-docs.yml.before` — the workflow file before this PR.
- `workflows/template-js-example-app.yml` — JS template reference.
- `workflows/template-rust-release.yml` — Rust template reference.
