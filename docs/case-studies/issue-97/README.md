# Case Study: Issue #97 — Build root constructs from links/references

This case study documents the investigation, root-cause analysis, and
remediation for
[issue #97](https://github.com/link-foundation/relative-meta-logic/issues/97):

> We need to find a way to build everything need to reason about types
> and so on based on idea of
> [Software Foundations / `Logic.v`](https://softwarefoundations.cis.upenn.edu/sf-3.2/Logic.html)
> expressed in pure links/references (and fully defined), that in
> addition to already supported features. So everything that is working
> continue to work, we just add a way to `override` / `redefine in other
> terms` these root constructs.

Two later comments from @netkeep80 widened the remit into a structured
"root construct registry" proposal with eight trust statuses, a
foundation scope, and acceptance tests. A third comment from @konard
fixed the contract:

> We need to make sure all described is alternative way to do logic. We
> just will be able to replace the foundations of logic apparatus, but
> it should behave essentially the same way. **Traditional foundations
> should work by default.** … Make sure everything covered with unit,
> integration and e2e tests, and executed in CI/CD. … Compile that data
> to `./docs/case-studies/issue-{id}` folder.

## 1. Timeline / Sequence of events

| When (UTC) | What |
|------------|------|
| 2025-09-26 | Original two-line issue filed by @konard. |
| 2025-10-04 | @netkeep80's "Implementation comment" expands the issue into a structured root-construct-registry proposal with eight trust statuses, a foundation scope, six phases, and seven acceptance tests. |
| 2025-10-08 | @netkeep80's "Additional clarification" adds the explicit acceptance contract — four conflated statuses, dependency-graph requirement, scoped overrides, layered equality, minimal milestone, four strict modes, relation to Software Foundations. |
| 2026-05-15 | @konard's comment fixes backward-compatibility and CI/CD requirements. Issue-solver branch `issue-97-bbe597194dee` created; placeholder PR #174 opened. |
| 2026-05-15 (this PR) | Root-construct registry + foundation scope implemented in both engines; JS and Rust unit/integration/e2e tests added; `examples/foundation-boolean-kleene.lino` + `examples/foundation-with-min.lino` added and replayed through both engines; self-evaluator parity test taught to recognise the new forms; `lib/self/foundations.lino` and `lib/self/evaluator.lino` extended; `docs/FOUNDATIONS.md` written; `docs/DIAGNOSTICS.md` extended with `E060`/`E061`/`E062`; this case study compiled. |
| 2026-05-16 (Phases 2–9) | Phase 2 equality-layer separation, Phase 3 proof-object replay (`E064`), Phase 4 truth-tables, Phase 6 pure-links strict mode (`E065`), Phase 7 dependency-graph traversal, Phase 8 carrier enforcement (`E063`), and Phase 9 experimental `mtc-anum` profile with `encodeAnum`/`decodeAnum` helpers (`E066`) all landed on PR #175 with parallel JS/Rust tests and parity replay. `docs/FOUNDATIONS.md` rewritten to cover the full programme. |

Raw data captured in `data/issue-97.json`, `data/issue-97-comments.json`,
`data/pr-174.json`, and `data/pr-175.json`. Code-level evidence in
`evidence/foundation-surface.md`.

## 2. Requirements extracted from the issue

The issue body, the two implementation comments, and @konard's final
backward-compatibility comment together yield the following explicit
requirements:

1. **R1 — Add a root-construct registry.** Every primitive used by the
   parser, evaluator, type checker, proof-replay checker, tactic engine,
   or metatheorem checker must have a descriptor with at minimum a
   `kind`, a `status`, and (where applicable) a `depends-on` list. No
   silent primitives.
2. **R2 — Eight trust statuses.** Each descriptor's `status` is one of
   `host-primitive`, `host-derived`, `external-trusted`,
   `user-configurable`, `links-encoded`, `links-defined`,
   `user-overridden`, or `planned`. The eight statuses must not collapse
   into one.
3. **R3 — Construct dependency graph.** Descriptors form a graph via
   `depends-on`, so a `(foundation-report)` can answer "which trusted
   primitives were used".
4. **R4 — Foundation scope.** A `(foundation <name> …)` declaration
   bundles a coherent set of root-construct interpretations; a
   `(with-foundation <name> …)` form activates that bundle for the
   duration of its body. Overrides are scoped, not global mutation.
5. **R5 — Backward compatibility.** Every `.lino` file that ran before
   foundations existed must run identically afterwards. The default
   foundation `default-rml` is pre-registered, mirrors the current host
   semantics, and is the only foundation active until a
   `(with-foundation …)` switches.
6. **R6 — `(foundation-report)` form.** The host can emit a structured
   trust/foundation report listing the active foundation, all known
   foundations, and every root construct bucketed by status. The
   printed layout is byte-identical between JavaScript and Rust.
7. **R7 — Structured diagnostics.** Bad `(root-construct …)` /
   `(foundation …)` / `(with-foundation …)` forms emit `E060` / `E061` /
   `E062` diagnostics instead of panics, and never break unrelated
   downstream queries.
8. **R8 — Self-bootstrap encoding.** The registry of root constructs and
   the default + alternative foundations are encoded as data in
   `lib/self/foundations.lino`, and the evaluator's data-encoded rule
   list in `lib/self/evaluator.lino` covers the new surface forms so
   that the future self-hosted evaluator can consume them.
9. **R9 — Links-defined finite logics.** At least one nontrivial
   alternative foundation (Boolean / Kleene) is demonstrated as a
   replay-tested example.
10. **R10 — Twin engine parity.** JS and Rust implementations agree on
    the registry shape, the foundation-report layout, the diagnostics
    codes, and the example outputs. A self-evaluator replay test
    enforces drift detection across the data-encoded evaluator.
11. **R11 — Tests at every level.** Unit, integration, and end-to-end
    coverage for both engines, executed in CI/CD on every push.
12. **R12 — Documentation.** `docs/FOUNDATIONS.md` documents the
    surface, the eight statuses, the report shape, the diagnostics, and
    the bundled foundations. The README links to it.
13. **R13 — Case-study compilation.** Issue/PR data captured under
    `docs/case-studies/issue-97/data/`, line-numbered code evidence
    under `evidence/`, and a deep analysis at `README.md`.

The broader programme proposed in @netkeep80's two comments (Phases
2–9: equality-layer provenance, links-defined proof-object substrate,
links-defined finite logics, links-defined type kernel, pure-links
strict mode, dependency-graph traversal, carrier enforcement, and
experimental `mtc-anum` profile) was originally framed as out of scope
for the first PR; subsequent commits on PR #175 brought all of these
in under the same backward-compatibility guarantee. See §10 for the
phase-by-phase status.

## 3. Root-cause analysis

### 3.1 No explicit boundary between host primitives and user-configurable knobs

Before this PR, RML already supported per-operator aggregator selection
(`(and: min)`), range/valence configuration, and operator composition.
But there was no machine-readable record of which primitives the kernel
silently depended on. A user who asked "which numeric domain is my
proof relying on?" or "is structural equality redefinable?" had to read
JS or Rust source code. The configurability story was implicit.

This is the underlying gap @netkeep80's comment identifies: RML was
configurable at the *object-logic* level but not foundationally honest
at the *meta-logic* level. Configuration and override were conflated.

### 3.2 No scoped override mechanism

Per-operator selection (`(and: min)`) is permanent: once executed it
rebinds `and` for the rest of the file. There was no
"swap-and-restore" form, so a file could not say "evaluate this body
under classical semantics, then restore the previous semantics for the
rest of the file". Bundled overrides (swap `and`, `or`, `both`,
`neither` together) were also impossible — each had to be reselected
individually.

This forced users to either fork a file or carefully reselect every
operator after each scope. Both approaches scale poorly and silently
drop bindings on file imports.

### 3.3 Self-bootstrap surface had no foundation forms

`lib/self/evaluator.lino` already encoded the evaluator's rule list as
links (one `(rule (eval <pattern>) <action>)` per surface form). The
self-evaluator parity test
(`js/tests/self-evaluator.test.mjs`,
`rust/tests/self_evaluator_tests.rs`) replays examples through that
data table and compares the result to the host evaluator. Without
foundation rules in the table, any example that used
`(with-foundation …)` would silently return zero results and the parity
test would fail.

### 3.4 No CI/CD coverage for the surface

Per @konard's comment: unit/integration/e2e in CI/CD is a hard
requirement. The existing `tests.yml` workflow (added under issue
#171) already runs both engines' suites on every push, so the only
remaining work was to *add the tests* that exercise the new surface.

## 4. Solution and solution plan

### 4.1 Add a root-construct registry seeded from `lib/self/foundations.lino` (R1, R2, R3, R5, R8)

Both engines now own a `rootConstructs: Map<name → descriptor>` and a
`foundations: Map<name → foundation>` on `Env`. The constructor seeds
the registries with the same set of descriptors that
`lib/self/foundations.lino` records as data — Parsing / LiNo layer,
Numeric layer, Truth / aggregator layer, Equality layer, Typed kernel
layer, Inductive / coinductive layer, Proof / tactics layer,
Metatheorem layer, and the self-bootstrap entries themselves.

The default foundation `default-rml` is pre-registered with the current
operator semantics so legacy programs are unaffected.

Top-level `(root-construct …)` and `(foundation …)` forms in user files
extend the registry by merging into the descriptor stored under that
name — explicit fields overwrite, omitted fields preserve, repeated
declarations accumulate.

### 4.2 Add `(with-foundation <name> …)` with snapshot/restore (R4, R7)

Both engines implement:

1. **Enter**: `env.enterFoundation(name)` / `env.enter_foundation(&name)`
   snapshots the previous binding of every operator the named
   foundation `(defines …)`, then applies the new bindings.
2. **Body**: forms inside the body evaluate normally. Nested
   `(with-foundation …)` recurse with their own frame.
3. **Exit**: `env.exitFoundation()` / `env.exit_foundation()` restores
   the snapshot in reverse, including the previously-active foundation
   tag.

`E062` is emitted when the named foundation is unknown; evaluation
continues with the surrounding forms intact. `E060` and `E061` cover
malformed `(root-construct …)` / `(foundation …)` declarations.

### 4.3 Add `(foundation-report)` and its formatter (R6, R10)

`env.foundationReport()` / `env.foundation_report()` returns a
structured snapshot with the shape documented in
[`docs/FOUNDATIONS.md`](../../FOUNDATIONS.md). `formatFoundationReport`
(JS) and `format_foundation_report` (Rust) render the snapshot as a
deterministic text block whose every line is byte-identical between the
two engines. The shared test corpus
(`rust/tests/shared_test_corpus.rs`, `rust/tests/shared_examples.rs`)
asserts this.

### 4.4 Bundled foundations and finite-logic examples (R9)

`lib/self/foundations.lino` declares two named alternatives:

- `boolean-classical` — two-valued logic, `and=min`, `or=max`,
  `both=min`, `neither=product`.
- `kleene-three-valued` — Strong Kleene, same operator shape over the
  real unit interval with `0.5` as `unknown`.

`examples/foundation-boolean-kleene.lino` exercises both inside the
same file, restoring the default after each scope.
`examples/foundation-with-min.lino` is a minimal one-operator demo.
Both examples are replayed by `rust/tests/shared_examples.rs` /
`js/tests/shared-examples.test.mjs` and by the self-evaluator parity
test, so any drift between the host and the data-encoded evaluator
breaks the build.

### 4.5 Self-bootstrap surface (R8)

`lib/self/evaluator.lino` now contains four new rules:

```lino
(rule (eval (root-construct name details))
  (register-root-construct name details))

(rule (eval (foundation name details))
  (register-foundation name details))

(rule (eval (with-foundation name body))
  (evaluate-body-under-foundation name body))

(rule (eval (foundation-report))
  (query (foundation-report-snapshot)))
```

The `EncodedEvaluator` in `js/tests/self-evaluator.test.mjs` was
taught to recognise these rules so the data-encoded evaluator replays
the new surface as faithfully as the host. The Rust counterpart in
`rust/tests/self_evaluator_tests.rs` checks the same patterns are
present.

### 4.6 Tests at every level (R11)

| Layer | JS | Rust |
|-------|----|------|
| **Unit** — `Env` lifecycle (preregister default, register, enter/exit, snapshot/restore, report shape) | `js/tests/foundations.test.mjs` (8 tests) | `rust/tests/foundations_tests.rs` (8 tests) |
| **Integration** — operator swap inside `(with-foundation …)`, nesting, `E062` no-abort behaviour | `js/tests/foundations.test.mjs` | `rust/tests/foundations_tests.rs` |
| **End-to-end** — full `.lino` files (`foundation-boolean-kleene.lino`, `foundation-with-min.lino`) evaluated by the host, the data-encoded evaluator, and the Rust shared-example replay | `js/tests/self-evaluator.test.mjs`, `js/tests/shared-examples.test.mjs` | `rust/tests/self_evaluator_tests.rs`, `rust/tests/shared_examples.rs`, `rust/tests/shared_test_corpus.rs` |

All three layers are picked up by the existing `tests.yml` workflow,
so every push runs `npm test` (JS) and `cargo test` (Rust) and fails
fast on any regression.

### 4.7 Documentation (R12)

- `docs/FOUNDATIONS.md` — surface reference, registry concept, status
  table, scope semantics, report shape, error codes, bundled
  foundations, programme roadmap.
- `docs/DIAGNOSTICS.md` — extended with `E060`/`E061`/`E062` rows.
- `README.md` — short pointer to `FOUNDATIONS.md` in the *Comparisons*
  section, alongside `CONFIGURABILITY.md`.

### 4.8 Case study (R13)

This document, the captured GitHub data under `data/`, and the
line-numbered audit under `evidence/foundation-surface.md` are the
case study. Following the style established by
`docs/case-studies/issue-167/README.md` and
`docs/case-studies/issue-171/README.md`, the case study is organised as
timeline → requirements → root causes → solution plan → verification →
file index.

## 5. Existing components / libraries considered

- **`Env.ops` table.** Already a `Map<string → operator fn>` in JS and
  `HashMap<String, Op>` in Rust. Foundation scope reuses this table —
  `enterFoundation` mutates entries in place and snapshots the previous
  ones; `exitFoundation` reverses the mutation. No new operator
  abstraction layer was needed.
- **`Aggregator` enum (Rust) / aggregator-name string (JS).** The
  `(defines and min)` clause is interpreted by the same aggregator
  resolver as the existing `(and: min)` runtime form. The two surfaces
  share their dispatch path, so a bug in one is a bug in both.
- **Self-evaluator parity replay**
  (`js/tests/self-evaluator.test.mjs`,
  `rust/tests/self_evaluator_tests.rs`). The data-encoded evaluator
  rule list in `lib/self/evaluator.lino` is already validated against
  the host evaluator by running shared examples through both. Adding
  four `(rule (eval …))` entries for the foundation forms and teaching
  the replay's `EncodedEvaluator` to interpret them put the new surface
  inside the same parity gate.
- **Shared-example replay**
  (`rust/tests/shared_examples.rs`, `rust/tests/shared_test_corpus.rs`,
  `js/tests/shared-examples.test.mjs`). Already exercises every file
  in `examples/` by both engines. Adding
  `examples/foundation-boolean-kleene.lino` and
  `examples/foundation-with-min.lino` automatically gave end-to-end
  coverage without writing per-example tests.
- **Existing `tests.yml` GitHub Actions workflow.** Already runs
  `npm test` and `cargo test` on every PR and every push to `main`,
  picking up all new test files without workflow changes.
- **Existing diagnostics dispatch** (`E000`–`E050`). Foundation forms
  reuse the same `RmlError(code, message)` pattern in JS and the same
  `panic!`-based dispatch in Rust, so the new `E060`/`E061`/`E062`
  codes appear in `evaluate(...).diagnostics` like any other code.
- **`lib/self/foundations.lino` framework.** The file existed as a
  draft inventory of root constructs; this PR turns the descriptors
  into the actual seed data that both engine constructors load.

## 6. Online search

Findings recorded for cross-referencing:

- **Software Foundations vol. 1 — `Logic.v`** — the issue's
  starting point. The Coq sources at
  <https://softwarefoundations.cis.upenn.edu/lf-current/Logic.v> define
  conjunction, disjunction, negation, implication, biconditional, and
  existential quantification *on top of* Coq's `Prop` universe and
  inductive definitions. The lesson the RML PR carries over is the
  organising idea ("many logical forms can be reduced to a smaller
  explicit substrate and then manipulated as ordinary formal objects")
  rather than the specific encoding; `Logic.v` still rides on Coq's
  fixed kernel.
- **Twelf foundation analysis** —
  Pfenning & Schürmann's "System Description: Twelf" (CADE 1999)
  documents Twelf's metatheorem checker for LF signatures (mode,
  totality, coverage, termination). RML's registry borrows the same
  status-tagging discipline for primitives, with the addition of an
  *intermediate* `links-encoded` status (data exists but the host still
  interprets it) versus `links-defined` (host derives behaviour from
  the data).
- **MMT Foundations** — Florian Rabe's *A Logical Framework Combining
  Model and Proof Theory* (Mathematical Structures in Computer
  Science, 2013) describes a meta-language explicitly designed to host
  multiple foundations side-by-side (HOL, ZFC, FOL, …) with
  inter-foundation morphisms. The eight-status registry here is closer
  in spirit to MMT's "what is a meta-language" question than to Coq's
  fixed kernel.
- **Isabelle/Pure vs Isabelle/HOL** — `src/Pure/` is the
  meta-logic; `src/HOL/HOL.thy` is the classical object logic
  installed on top of it. RML's `default-rml` plays the
  `Isabelle/Pure` role and a registered alternative foundation plays
  the `Isabelle/HOL` role; the distinction the registry records is
  exactly the kernel-vs-object-logic distinction Isabelle has had
  since its earliest releases.
- **Lean 4's `Std.Tactic.Open` / `MathlibBootstrap`** — show that
  user-level scope constructs over re-bindable notations are
  industrially viable. RML's `(with-foundation …)` is the same idea
  with a different keyword.

These references are listed so future revisions can extend the registry
toward links-defined kernels and proof-object substrates with prior art
in hand.

## 7. Verification

After this PR:

1. `node --test tests/` (run from `js/`) reports 8 new tests in
   `foundations.test.mjs` plus extended `self-evaluator.test.mjs`
   coverage of the new surface, and 0 regressions across the full
   suite.
2. `cargo test --manifest-path rust/Cargo.toml` reports 8 new tests in
   `foundations_tests` and 0 regressions across the full suite,
   including extended `self_evaluator_tests` and `shared_examples`
   replay coverage.
3. Every requirement R1–R13 has at least one test or doc-level check.
   The mapping is recorded in `evidence/foundation-surface.md`.
4. The default foundation `default-rml` is loaded on every fresh
   `Env`, so every `.lino` file that ran before this PR runs
   identically afterwards. This is asserted by
   `js/tests/foundations.test.mjs:35-46` ("does not change baseline
   semantics when no foundation is declared") and
   `rust/tests/foundations_tests.rs:60-72` ("baseline_semantics_unchanged_when_no_foundation_is_declared").
5. The data-encoded evaluator in `lib/self/evaluator.lino` covers the
   new surface; the self-evaluator parity test asserts that the
   data-encoded evaluator returns the same query results as the host
   evaluator for `examples/foundation-boolean-kleene.lino` and
   `examples/foundation-with-min.lino`.

Items (1) and (2) are also checked automatically on every PR push by
the `tests.yml` workflow.

## 8. CI/CD template review

Per @konard's comment, the CI/CD templates were reviewed:

- <https://github.com/link-foundation/js-ai-driven-development-pipeline-template>
- <https://github.com/link-foundation/rust-ai-driven-development-pipeline-template>
- <https://github.com/link-foundation/python-ai-driven-development-pipeline-template>
- <https://github.com/link-foundation/csharp-ai-driven-development-pipeline-template>

The relevant template-vs-repo comparison and any reportable
discrepancies are recorded in `evidence/cicd-template-review.md`. No
foundation-specific CI changes were needed for this PR — the existing
`tests.yml` workflow already runs `npm test` and `cargo test` and
therefore picks up the new tests automatically.

## 9. Roadmap progress (Phases 1–9)

The original issue thread laid out a six-phase programme; @netkeep80's
follow-up comment widened it to eight; the experimental `mtc-anum`
profile added a ninth. The status of each on PR #175:

| Phase | Topic | Status | Where |
|-------|-------|--------|-------|
| 1 | Inventory + reporting + scoped overrides | done | §4.1–§4.3; `docs/FOUNDATIONS.md` §2–§4 |
| 2 | Equality and numeric-domain separation | done (registry-level) | `lib/self/foundations.lino` equality entries; trace-layer follow-up tracked |
| 3 | Proof-object substrate (`(check-proof …)`) | done | `E064`; `js/tests/check-proof.test.mjs` + `rust/tests/check_proof_tests.rs` |
| 4 | Links-defined finite logics (Boolean, Kleene, truth-tables) | done | §4.4; `examples/foundation-boolean-kleene.lino`, `(truth-table …)` |
| 5 | Links-defined type/proof kernel fragment | done (descriptor + dep-graph) | `lib/self/foundations.lino` typed-kernel entries |
| 6 | Pure-links strict mode | done | `E065`; `(strict-foundation pure-links)` + `(allow-host-primitive …)` |
| 7 | Dependency-graph traversal | done | Rendered in `formatFoundationReport`; powers strict mode |
| 8 | Carrier enforcement | done | `E063`; `(carrier …)` + `(strict-carrier)` |
| 9 | Experimental `mtc-anum` profile + `encodeAnum`/`decodeAnum` | done | `E066`; pre-seeded but opt-in; `(experimental)`, `(root …)`, `(abit …)` clauses |

Every phase landed with **parallel JS and Rust tests** plus the
existing self-evaluator parity replay (`lib/self/evaluator.lino` plus
the test suites in `js/tests/self-evaluator.test.mjs` and
`rust/tests/self_evaluator_tests.rs`), so the data-encoded evaluator
recognises the new surface forms on both engines.

### 9.1 Phase 3 — `(check-proof …)`

`(check-proof <conclusion> <object>)` replays a previously emitted
proof object against the active foundation. The replay verifies that
every premise listed in the object holds under the current operator
table and that the conclusion follows; mismatches raise `E064`.

### 9.2 Phase 6 — pure-links strict mode

`(strict-foundation pure-links)` flips a per-`Env` flag that causes any
subsequent query referencing a `host-primitive` or `host-derived`
construct to raise `E065`, unless the construct (or one of its
ancestors in the dependency graph) is whitelisted by
`(allow-host-primitive <name>...)`. The whitelist is additive across
declarations; clearing requires exiting the strict-foundation scope.

### 9.3 Phase 7 — dependency-graph traversal

The trust audit (`formatFoundationReport`) renders the transitive
closure of `depends-on` so users can see, before flipping the strict
switch, exactly which constructs would need to be whitelisted. The
traversal is also what powers Phase 6's enforcement check.

### 9.4 Phase 8 — `(carrier …)` and `(strict-carrier)`

A foundation can declare its carrier set (the values it considers
legal) and opt into runtime enforcement. Out-of-carrier query results
or probability assignments raise `E063`. Symbolic carrier members
(`true`, `false`, `unknown`) resolve through `env.symbol_prob` at
activation time. Numeric literals stay literal.

### 9.5 Phase 9 — experimental `mtc-anum` profile

A pre-seeded experimental foundation that is **never activated
implicitly**. It carries an `[experimental]` tag, a root symbol `∞`,
and four "abits" `[ ] 0 1` published on the trust report. Companion
helpers `encodeAnum` / `decodeAnum` (JS) and `encode_anum` /
`decode_anum` (Rust) round-trip arbitrary `Node` values through the
four-abit alphabet. Errors raise `E066`.

## 10. Files in this case study

- `README.md` — this document.
- `data/issue-97.json` — issue body, labels, author.
- `data/issue-97-comments.json` — full conversation comments
  (@netkeep80's two implementation comments and @konard's final
  contract comment).
- `data/pr-174.json` — pull-request metadata.
- `evidence/foundation-surface.md` — line-numbered code-level
  evidence for each requirement R1–R13.
- `evidence/cicd-template-review.md` — review of the four CI/CD
  templates against this repository's `.github/workflows/`.
