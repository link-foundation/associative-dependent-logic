# Follow-up recommendations

These are observations that came out of the issue #163 investigation but
are intentionally out of scope for the bug-fix PR. Each is a candidate
for its own issue / PR.

1. **Add a broken-link check workflow.** The JS pipeline template ships
   `.github/workflows/links.yml` using `lycheeverse/lychee-action@v2`
   with a Web Archive fallback. Adding it here would have caught
   issue #163 automatically the moment the README link was introduced.
2. **Add a `release.yml` for `js/` and `rust/` packages.** All four
   templates have one; this repository currently has none. Without it,
   `release: published` is never fired, which is part of why the
   release-gated deploy in the old `api-docs.yml` never ran.
3. **Bump action versions consistently.** The templates already moved
   to `@v6` (`checkout`, `setup-node`, `configure-pages`) and `@v5`
   (`upload-pages-artifact`, `deploy-pages`). The other workflow files
   in this repo (`bootstrap.yml`, `docker.yml`, `lint-english.yml`,
   `parity.yml`) still use `@v4`; updating them in a follow-up keeps
   the repository aligned with the templates.
4. **Set `Settings → Pages → Source = "GitHub Actions"` once.** This
   is a one-time manual step required for the new deploy job to
   succeed; it cannot be automated from inside a workflow.
5. **Document the deployment contract in `README.md`.** A short
   “How the playground is deployed” section linking to the workflow
   would make this kind of regression easier to debug next time.
