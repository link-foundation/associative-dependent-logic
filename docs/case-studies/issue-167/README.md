# Case Study: Issue #167 — Make the comparison matrix technically precise

This case study documents the investigation, root-cause analysis, and remediation
for [issue #167](https://github.com/link-foundation/relative-meta-logic/issues/167):

> `docs/CONCEPTS-COMPARISION.md` is useful as a positioning document, but the
> current comparison matrix mixes several different levels of meaning … As a
> result, several `Yes` / `No` / `Part` entries are ambiguous or outdated,
> especially for RML itself. This issue proposes revising the document so that
> it remains useful as a high-level comparison while being more technically
> precise and harder to misread.

A follow-up comment by @konard widened the remit:

> We need to update our comparison with all other similar systems, by supported
> concepts, by supported features and so on. **Make sure comparison is verified
> with tests.** We need to collect data … to `./docs/case-studies/issue-{id}`
> folder, and use it to do deep case study analysis (also make sure to search
> online for additional facts and data) … and propose possible solutions and
> solution plans for each requirement.

## 1. Timeline / Sequence of events

| When (UTC) | What |
|------------|------|
| 2026-05-11 10:53 | Issue #167 filed by @netkeep80, labelled `documentation` + `enhancement`. |
| 2026-05-12 23:07 | @konard adds the "verify with tests + compile a case study" comment, widening the scope from "documentation edit" to "documentation + tests + case study". |
| 2026-05-12 23:10 | Issue-solver branch `issue-167-b75ac8a2555c` created; placeholder PR #173 opened. |
| 2026-05-12 (this PR) | Feature audit produced (`evidence/rml-feature-audit.md`); the two `COMPARISION` files rewritten and renamed to `COMPARISON`; old filenames retained as compatibility stubs; cross-doc links updated; JS and Rust test suites added so the matrix cannot quietly drift away from the implementations. |

Raw data captured in `data/issue-167.json`, `data/issue-167-comments.json`,
and `data/pr-173.json`. Code-level evidence in
`evidence/rml-feature-audit.md`.

## 2. Requirements extracted from the issue

The issue body, the in-issue acceptance checklist, and @konard's comment
together yield the following explicit requirements:

1. **R1 — Expand the legend** with `Kernel` / `Library` / `Encoding` /
   `Runtime` / `Host` / `External` / `Prototype` / `Theory` markers, and
   allow qualified `Yes (Kernel)`, `Yes (Library)`, `Part (Prototype)`,
   `Yes (Runtime + Host)` cells.
2. **R2 — Separate provers/frameworks/languages from libraries/archives.**
   Foundation and AFP must live in a libraries/archives section rather than
   alongside Lean, Rocq, Isabelle, etc.
3. **R3 — Clarify kernel vs object logic vs library support** for Lean,
   Rocq, and Isabelle, so that `Yes` for "Classical logic" or "Redefinable
   logical operators" does not imply kernel-level support that those systems
   do not have.
4. **R4 — Update stale RML rows.** Full normalization, inductive families,
   coinductive types/predicates, totality / termination / coverage, tactic
   construction, proof-producing evaluator, independent proof replay, and
   external certification bridge must reflect what current RML actually
   implements (with `Part / Prototype` qualifiers where appropriate).
5. **R5 — Clarify RML's numeric / many-valued semantics** as runtime
   configuration that is host-implemented. Numeric truth values,
   configurable semantic range, configurable valence, fuzzy logic, and
   probabilistic operators must be marked `Yes (Runtime + Host)` with text
   explaining what is host-defined.
6. **R6 — Add an equality-layers row** distinguishing structural / assigned
   / numeric / definitional (convertibility) equality.
7. **R7 — Avoid overclaiming about self-reference.** Paradox-tolerant
   evaluation is runtime behaviour, not a mechanised soundness claim.
8. **R8 — Refine Lambda Prolog and Twelf rows.** Lambda Prolog must be
   described as higher-order hereditary Harrop logic programming "not HOL
   in the Isabelle/HOL sense". Twelf must not be advertised as having
   tactic-level interactive proof construction.
9. **R9 — Add an RML status note** explaining that "available" in the
   matrix usually means "available in the current evaluator/runtime", not
   "defined inside `.lino` with a mechanised metatheory".
10. **R10 — Update the RML positioning summary.** Remove "no ATP bridge"
    and "no independent proof replay" claims; the ATP bridge (with trusted
    external nodes) and the independent proof-replay checker both exist.
11. **R11 — Rename `COMPARISION` → `COMPARISON`** (fix the spelling) for
    both files, with compatibility stubs at the old filenames so external
    links keep working.
12. **R12 — Verify the comparison with tests** (added by @konard). The
    matrix must be checked against the live implementations so that any
    drift between the matrix and the code (or between Rust and JS) breaks
    a test suite.
13. **R13 — Compile data and a deep case-study analysis** under
    `./docs/case-studies/issue-167/`, with timeline, requirements, root
    causes, solution plan, and a survey of existing components and online
    references.

## 3. Root-cause analysis

### 3.1 Legend coarseness lets disparate concepts collapse into the same cell

The pre-PR legend was:

```text
Yes / Part / Host / Theory / N/A / No
```

Six labels could not separate (a) trusted kernel features, (b) library-level
features, (c) object-logic encodings, (d) runtime configuration,
(e) host-language implementation details, (f) external trusted procedures,
(g) theoretical capabilities, (h) prototype / partial features. A single
`Yes` therefore carried different meanings for different systems and (more
importantly) for different rows within the same system.

The downstream effect is that the cells in the matrix become non-comparable.
A row like "Redefinable logical operators" reading `Yes` for Lean and `Yes`
for RML hides the fundamental difference: Lean lets users define new
notations and connectives in a library, but the kernel rules are fixed;
RML lets the runtime semantics of `(and ...)`, `(or ...)`, implication,
etc. be reselected via configuration.

### 3.2 Stale RML rows because the matrix is older than the code

The issue lists nine specific RML cells whose `No` or `Part: encodable only`
labels were correct at one point but no longer reflect the current
codebase. The feature audit in `evidence/rml-feature-audit.md` cites
line-numbered code references for each one. Concretely:

| Row | Old label | What current code actually has |
|-----|-----------|--------------------------------|
| Full normalization | `No: evaluator is not a normalizer` | `whnf` / `nf` for the typed lambda fragment (Rust `lib.rs:2178/2350/2364`, JS `rml-links.mjs`). |
| `(normal-form …)` surface | implicit | Exercised by `rust/tests/self_evaluator_tests.rs:55` (`"(eval (normal-form expression))"`) and JS mirror. |
| Inductive families | `Part: encodable only` | `(inductive ...)` declarations + generated recursors (`rust/src/lib.rs:7400+`, `rust/tests/inductive_tests.rs`, JS mirror). |
| Coinductive types | `Part: encodable only` | `(coinductive ...)` with productivity checks (`rust/src/lib.rs` `parse_coinductive_form`, `rust/tests/coinductive_tests.rs:37`). |
| Totality / termination / coverage | `No` | `is_total` (`rust/src/lib.rs:5588`), `is_terminating` (`:5878`), `is_covered` (`:6020`); JS mirrors. |
| Tactic-level proof construction | `No` | `reflexivity` (`:4952`), `symmetry` (`:4969`), `transitivity` (`:4986`), `rewrite` (`:4683`), `simplify` (`:4712`), `exact` (`:5181`), `induction` (`:5198`); JS mirrors. |
| Proof-producing evaluator | `No` | Selected queries produce proof links; `build_proof` at `rust/src/lib.rs:3247`. |
| Independent proof replay | `No` | `rust/src/check.rs` is a dedicated module; `js/src/rml-check.mjs` mirrors it; tests in `*/check*.test.*` and `rust/tests/check_tests.rs`. |
| External certification bridge | `No` / "no ATP bridge" | `(by smt ...)` SMT-LIB bridge and `(by atp ...)` TPTP bridge with trusted-external recording; `AtpOptions` at `rust/src/lib.rs:3638`, `parse_atp_status` at `:4483`. |

In short: the matrix had drifted away from the code, and there was no
test gating that drift.

### 3.3 No automated verification linking the doc to the code

Before this PR, nothing in CI or `npm test` / `cargo test` connected the
words in the comparison matrix to the actual public surface of the JS and
Rust implementations. Any time the codebase gained `whnf` / `nf` / `(by atp
...)` / `is_covered`, the doc had to be updated by hand or it silently
got out of sync.

This is the root cause of R12: the matrix needs structural tests against
the doc files **and** smoke tests against the JS module exports and the
Rust public surface, so future drift breaks the build.

### 3.4 Filename mis-spelling

`COMPARISION` is misspelled. The right spelling is `COMPARISON`. Because
the misspelled filenames already appear in `README.md`,
`docs/CONFIGURABILITY.md`, `docs/case-studies/issue-22/README.md`,
`docs/tutorials/README.md`, and possibly in external links, a flat rename
risks breaking inbound links.

## 4. Solution and solution plan

### 4.1 Rewrite the matrix with the expanded legend (R1–R11)

`docs/CONCEPTS-COMPARISON.md` (218 lines) and `docs/FEATURE-COMPARISON.md`
(141 lines) are new canonical files. They:

- Define the expanded legend: `Kernel`, `Library`, `Encoding`, `Runtime`,
  `Host`, `External`, `Prototype`, `Theory`, `Archive`, plus the existing
  `Yes` / `Part` / `N/A` / `No` (R1).
- Split the systems list into "Provers, frameworks, and languages" and
  "Libraries and archives", with Foundation and AFP in the latter (R2).
- Replace bare `Yes` for Lean/Rocq/Isabelle on "Classical logic" and
  "Redefinable logical operators" with `Library`/`Encoding`/`Notation`
  qualifiers that say what is library-level vs kernel-level (R3).
- Update every stale RML cell using the evidence catalogued in
  `evidence/rml-feature-audit.md` (R4), with `Part / Prototype` qualifiers
  where the host implementation is in place but mature surface tooling is
  still limited.
- Mark "Numeric truth values in the core", "Configurable semantic range",
  "Configurable valence", "Fuzzy logic", and "Probabilistic operators" as
  `Yes (Runtime + Host)` with text explaining what is host-defined (R5).
- Add an "Equality layers distinguished" row that calls out structural,
  assigned, numeric, and definitional/convertibility equality (R6).
- Rewrite the self-reference row to say "Yes (Runtime semantics)" and
  explicitly disclaim a classical-consistency or mechanised-soundness
  reading (R7).
- Rewrite the Lambda Prolog row to disclaim "HOL in the Isabelle/HOL
  sense", and the Twelf row to disclaim tactic-level interactive proof
  construction (R8).
- Include a short "RML status note" paragraph (R9) and a positioning
  summary that explicitly mentions both the ATP bridge and the
  independent proof-replay checker (R10).

### 4.2 Keep the old filenames as compatibility stubs (R11)

A flat rename would break any inbound link to
`docs/CONCEPTS-COMPARISION.md` or `docs/FEATURE-COMPARISION.md`. The old
files are therefore retained as one-paragraph compatibility stubs that
point to the renamed files. Internal references in
`README.md`, `docs/CONFIGURABILITY.md`,
`docs/case-studies/issue-22/README.md`, and
`docs/tutorials/README.md` are updated to the new spelling.

### 4.3 Verify the matrix with tests (R12)

Two new test files were added so that drift between the doc and the code
breaks both implementations' suites:

- `js/tests/concepts-comparison-doc.test.mjs` — 17 tests across 2
  describe blocks. The first block reads the Markdown files and asserts
  on document structure (the new files exist and are non-trivial; the
  old files remain as compatibility stubs; the legend defines each
  qualifier mark; Provers/Libraries sections are split with Foundation +
  AFP in the libraries section; the stale "no ATP bridge" string is
  gone; the "independent proof-replay checker" phrase is present; the
  RML status note exists; the equality-layers row exists; the Lambda
  Prolog and Twelf rows have been rewritten; the five numeric/many-valued
  rows are marked `Yes (Runtime + Host)`). The second block imports the
  JS module and asserts that every capability the matrix advertises
  (`whnf`, `nf`, `(normal-form ...)` surface form, `(inductive ...)`,
  `(coinductive ...)`, `(total ...)`, termination, coverage, structural
  + definitional equality, tactic links, `(by smt ...)`, `(by atp ...)`,
  independent proof-replay checker) is actually exported.

- `rust/tests/concepts_comparison_doc_tests.rs` — 14 tests mirroring the
  same checks for the Rust crate, using `std::fs::read_to_string` for
  the structural checks and the `evaluate(...)` surface for the
  capability smoke tests, so that any drift between Rust and JS breaks
  one of the two suites.

Both suites are picked up by the new `tests.yml` workflow added under
issue #171, so every PR and every push to `main` runs them.

### 4.4 Compile the case study (R13)

This document, the captured GitHub data under `data/`, and the
line-numbered audit under `evidence/` are the case study. Following the
style established by `docs/case-studies/issue-171/README.md` and
`docs/case-studies/issue-163/README.md`, the case study is organised as
timeline → requirements → root causes → solution plan → verification →
file index.

## 5. Existing components / libraries considered

- **`node:test`** (Node.js core test runner). The new
  `js/tests/concepts-comparison-doc.test.mjs` uses the existing
  `describe` / `it` style, which is already used by every other
  `js/tests/*.test.mjs` file. No new dependency added.
- **`cargo test`** (native to the Rust toolchain). The new
  `rust/tests/concepts_comparison_doc_tests.rs` uses
  `std::fs::read_to_string` and `CARGO_MANIFEST_DIR` for path resolution,
  matching the pattern used in `rust/tests/imports_tests.rs`. No new
  dependency added.
- **`tests.yml`** workflow (introduced under issue #171). It already runs
  `npm test` and `cargo test` on every PR and every push to `main`, so
  the new tests gain CI coverage without further workflow changes.
- **Existing JS module exports** in `js/src/rml-links.mjs` — `whnf`,
  `nf`, `isStructurallySame`, `isConvertible`, `isTotal`, `isTerminating`,
  `isCovered`, `parseInductiveForm`, `parseCoinductiveForm`, `rewrite`,
  `simplify`, `runTactics`, `buildProof`. These are the symbols the JS
  smoke tests reference; nothing new had to be exported.
- **Existing Rust public surface** in `rust/src/lib.rs` — `evaluate`,
  `whnf`, `nf`, `is_convertible`, `is_total`, `is_terminating`,
  `is_covered`, `parse_inductive_form`, `parse_coinductive_form`,
  `build_proof`, `run_tactics`, `rewrite`, `simplify`. Again, nothing new
  had to be exported.
- **`rust/src/check.rs` and `js/src/rml-check.mjs`** — the independent
  proof-replay checker introduced by issue #36. The tests assert that
  these files exist, which is the cheapest way to verify the "independent
  proof replay" cell against the code.

## 6. Online search (R13)

Findings recorded for cross-referencing:

- **Lean / Mathlib** — `Mathlib.Logic.Basic` and `Mathlib.Init.Classical`
  ship `Classical.em`, `Classical.choice`, etc. as theorems and axioms in
  a library, not as kernel rules. Lean's kernel (CIC + a few additions)
  is documented in the `lean4` repository (`src/library/kernel/`) and
  remains fixed across these libraries. This justifies the
  `Library/Axiom` qualifier on the "Classical logic" row for Lean.
- **Rocq (Coq)** — the standard library's `Coq.Logic.Classical*` modules
  similarly add classical reasoning as an axiom set on top of CIC. The
  CIC kernel is described in the `coq/coq` repository
  (`kernel/`). Same `Library/Axiom` qualifier applies.
- **Isabelle/HOL vs Isabelle/Pure** — Isabelle's
  `src/Pure/` is the meta-logic (intuitionistic higher-order logic);
  `src/HOL/HOL.thy` is the classical higher-order object logic. The
  matrix now records "Isabelle/HOL is classical HOL; Isabelle/Pure
  remains the meta-logic".
- **Twelf / LF** — the Twelf manual ("Proof and metatheorem checking")
  describes mode, world, totality, and coverage checking, plus proof
  search, but does not provide a tactic system in the Lean/Rocq/Isabelle
  sense. The matrix's new wording for the Twelf cell reflects this.
- **Lambda Prolog** — Miller and Nadathur's "Programming with
  Higher-Order Logic" describes Lambda Prolog as higher-order
  hereditary Harrop logic programming with HOAS. It is not HOL theorem
  proving. The matrix now disclaims that reading.
- **Foundation (`leanprover-community/mathlib4`)** and **AFP**
  (`isa-afp.org`) — Foundation is a Lean 4 library and AFP is an archive
  of Isabelle developments; both ride on top of an existing prover and
  do not ship their own kernels. The matrix now places both in the
  "Libraries and archives" section.

These references are quoted from publicly available documentation pages
and source repositories of the respective systems. They are listed here
so that future revisions of the matrix have a starting point for
follow-up research.

## 7. Verification

After this PR:

1. `node --test tests/` (run from `js/`) reports 17 new tests for
   `concepts-comparison-doc.test.mjs` and 0 regressions across the full
   suite (996 tests, all passing at time of writing).
2. `cargo test --manifest-path rust/Cargo.toml` reports 14 new tests in
   `concepts_comparison_doc_tests` and 0 regressions across the full
   suite.
3. Every RML capability the matrix now advertises is either (a) backed
   by a line-numbered citation in `evidence/rml-feature-audit.md`, or
   (b) exercised by the new doc-smoke tests above.
4. The old `COMPARISION` filenames keep working: they are short stubs
   that link to the new `COMPARISON` files, so inbound links elsewhere
   do not break.
5. Internal references to the renamed files (`README.md`,
   `docs/CONFIGURABILITY.md`, `docs/case-studies/issue-22/README.md`,
   `docs/tutorials/README.md`) have been updated to the new spelling.

Items (1) and (2) are also checked automatically on every PR push by the
`tests.yml` workflow.

## 8. Files in this case study

- `README.md` — this document.
- `data/issue-167.json` — issue body, labels, author.
- `data/issue-167-comments.json` — issue conversation comments.
- `data/pr-173.json` — pull-request metadata.
- `evidence/rml-feature-audit.md` — line-numbered code-level evidence
  for each RML row in the matrix.
