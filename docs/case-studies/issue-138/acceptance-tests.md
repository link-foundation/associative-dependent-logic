# Acceptance tests — Issue #138

This artefact lists the test categories every converter must pass before the corresponding phase issue closes. Phase numbering follows [`README.md` § Phased roadmap](./README.md#phased-roadmap-and-recommended-issue-split).

The five categories are intentionally a thin generalisation of [Karl Palmskog's round-trip suite for Coq SerAPI](https://github.com/rocq-archive/coq-serapi/blob/main/CHANGES.md) and rust-analyzer's syntax round-trip tests.

## Category 1 — Trivia round-trip

For every fixture `s`:

```text
assert print_cst(parse_lino(host_to_lino(s), { mode: 'cst' })) == s
```

byte-for-byte, modulo the canonicalisation list documented in [`cst-model.md` § 8](./cst-model.md#8-canonicalisations-documented).

Concrete fixtures per phase:

- **Phase K** (`.lino` itself): every `.lino` file under `examples/` and `lib/`.
- **Phase L** (Rust): every `.rs` file under `rust/src/` plus a curated 10-file subset of the rust-analyzer parser test corpus.
- **Phase M** (JavaScript): every `.mjs` file under `js/src/` plus a 10-file subset of test262.
- **Phase N** (Lean): `examples/lean-export-basic.lean` plus 5 small Mathlib snippets.
- **Phase O** (Rocq): the compiled `.v` form of `examples/rocq-export.lino` plus 5 small Rocq stdlib snippets.

The bootstrap corpus job ([issue #91](https://github.com/link-foundation/relative-meta-logic/issues/91)) is extended to include every fixture in this category.

## Category 2 — Idempotent canonicalisation

For every fixture `s` and every documented canonicalisation `c`:

```text
let s' = print_cst(parse_lino(host_to_lino(s), { mode: 'cst' }), { canonicalise: c })
assert print_cst(parse_lino(host_to_lino(s'), { mode: 'cst' }), { canonicalise: c }) == s'
```

This is the contract that canonicalisations are themselves stable: once the source has been canonicalised, applying the same canonicalisation again is a no-op.

## Category 3 — Cross-language identity (shared dialect)

For every fixture in the shared semantic fragment (typed λ + ADT + Pi + records + match):

```text
let shared = host_to_lino(s) -> lino_cst_shared
let s'     = lino_cst_shared -> host_to_lino -> print
assert ast(s) ≡ ast(s')      # semantic equality, not byte equality
```

That is: when we round-trip through the shared dialect, the result is semantically equivalent to the input, even though the printer may pick different formatting.

This is the test that backs the issue's example "convert lean to .lino and after that convert it to JavaScript": the Lean source need not survive byte-for-byte through the JS leg, but the typed kernel it encodes must.

## Category 4 — Negative `unrepresentable` tests

For every construct outside the shared dialect (probabilistic operators, fuzzy aggregators, range/valence config, `Notation`, Lean macros, Rust macros …):

```text
let lino = host_to_lino(s, dialect)
let shared = try_to_shared(lino)
assert shared.is_unrepresentable()
assert shared.diagnostic.matches('E0XX: <construct> has no shared encoding')
```

The point is that every host-specific construct produces a **structured** diagnostic, not a silent loss. The error code list is appended to the existing diagnostics catalogue in [`docs/DIAGNOSTICS.md`](../../DIAGNOSTICS.md).

## Category 5 — Bootstrap corpus inclusion

The case-study folder itself (`docs/case-studies/issue-138/`) and all per-language round-trip fixtures are added to the bootstrap corpus gate (see [issue #91](https://github.com/link-foundation/relative-meta-logic/issues/91) and [issue #92](https://github.com/link-foundation/relative-meta-logic/issues/92)) so the converters cannot regress without CI catching it.

## CI integration

We add one workflow per phase, named after the phase letter:

| Workflow | Trigger | Jobs |
|----------|---------|------|
| `.github/workflows/round-trip-lino.yml` (Phase K) | PR + cron | Category 1, 2 on `.lino` corpus |
| `.github/workflows/round-trip-rust.yml` (Phase L) | PR + cron | Categories 1, 2, 3, 4 on Rust corpus |
| `.github/workflows/round-trip-js.yml` (Phase M) | PR + cron | Same for JS |
| `.github/workflows/round-trip-lean.yml` (Phase N) | PR + cron | Same for Lean |
| `.github/workflows/round-trip-rocq.yml` (Phase O) | PR + cron | Same for Rocq |
| `.github/workflows/bootstrap.yml` (existing, extended) | PR + cron | Category 5 |

Each workflow runs both the JS and Rust implementations, matching the existing parity-CI shape ([PR #168](https://github.com/link-foundation/relative-meta-logic/pull/168)).

## Coverage targets

| Phase | Fixtures (initial) | Fixtures (steady-state) |
|-------|--------------------|--------------------------|
| K | 30 (existing `examples/`, `lib/`) | 50+ |
| L | 10 (rust-analyzer subset) | 100+ |
| M | 10 (test262 subset) | 100+ |
| N | 5 (Mathlib snippets) | 30+ |
| O | 5 (Rocq stdlib snippets) | 30+ |
| P | 10 (cross-language demos) | 50+ |

The steady-state numbers are not blockers for the first PR of each phase — they are the targets the per-phase epic closes against.

## What "passes" looks like for issue #138

The case-study deliverable itself (this PR) is reviewed against the rubric in [`requirements.md`](./requirements.md). Implementation issues track their own acceptance criteria.

Issue #138 itself is considered closed when all of:

- All five Phase letters K–O have green CI.
- Phase P round-trip identity (Category 3) is green on at least one example per cross-language pair.
- The tutorial in Phase Q is published and linked from `README.md`.

This matches the closure pattern used for [issue #95 (the parity epic)](https://github.com/link-foundation/relative-meta-logic/issues/95), recorded in [`docs/case-studies/issue-95/README.md`](../issue-95/README.md).
