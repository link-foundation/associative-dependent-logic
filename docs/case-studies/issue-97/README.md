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
| 2026-05-16 (Phases 2–9) | Phase 2 equality-layer separation, Phase 3 proof-object replay (`E064`), Phase 4 truth-tables, Phase 5 links-defined typed-kernel fragment (`pi-formation`, `lambda-introduction`, `application-elimination`, `beta-conversion` replayed through `(check-proof …)` from `examples/typed-kernel-links.lino`), Phase 6 pure-links strict mode (`E065`), Phase 7 dependency-graph traversal, Phase 8 carrier enforcement (`E063`), and Phase 9 experimental `mtc-anum` profile with `encodeAnum`/`decodeAnum` helpers (`E066`) advanced on PR #175 with parallel JS/Rust tests and parity replay. |
| 2026-05-17 (PR #176) | Latest issue feedback was incorporated: `semantic-status` / `semanticStatus` separates host-executed lookup and structural matching from links-level data; truth-table implementations report `links-checked`; substitution/freshness/alpha-renaming remain explicit `host-trusted` boundaries; and `examples/proof-checking-relation.lino` adds a nontrivial proof-checking relation represented as links-level data and replayed by both engines. |
| 2026-05-17 (Phase 12 on PR #177) | Phase 12 adds the inductive data layer the issue originally pointed at: five links-defined Peano rules (`nat-zero-formation`, `nat-succ-formation`, `nat-add-zero`, `nat-add-succ`, `nat-induction`) in `examples/nat-links.lino`, registered as `(foundation nat-links …)`, replayed by both engines, and pinned by `js/tests/nat-links.test.mjs` + `rust/tests/nat_links_tests.rs`. |
| 2026-05-17 (PR #178) | Explicit Nat equality. PR 178 adds the dedicated `nat-equality` equality layer plus the proof rules `nat-refl` and `nat-cong-succ`, switches `examples/nat-links.lino` from the bare literal `equals` to `nat-equals`, registers the layer + rules in `lib/self/foundations.lino`, and extends `(foundation nat-links …)`'s `uses` list accordingly. The host's `=`/`numeric-equality` layer is left untouched (backward compatibility) and there is no evaluator dispatch on `nat-equals` yet — the operator stays a bare literal that the structural matcher replays. Both engines pin the new behaviour with parity tests (14 + 14). |
| 2026-05-17 (PR #178 continued) | The same PR extended the Nat fragment with the recursor and dependent eliminator `nat-recursion` / `nat-eliminator`, multiplication `nat-mul-zero` / `nat-mul-succ`, the per-proof dependency/trust reporter `(proof-report <name>)`, the links-evaluated `(eval-nat <term>)` normalizer, strict pure-links audit coverage, typed/proof-kernel boundary reporting, and the dedicated meta-theory mapping document `docs/META-THEORY-MAPPING.md`. The latest revision makes `eval-nat` dispatch through the active Peano computation rules and splits host rendering into `nat-normal-form-to-host-number`. |

Raw data captured in `data/issue-97.json`, `data/issue-97-comments.json`,
`data/pr-174.json`, `data/pr-175.json`, and `data/pr-176.json`.
Code-level evidence in `evidence/foundation-surface.md`.

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
14. **R14 — Execution-boundary classification.** New mathematical
    constructs must distinguish `links-described`, `links-checked`,
    `links-evaluated`, `self-hosted`, and `host-trusted` rather than
    relying only on the legacy trust `status`.
15. **R15 — Links-level proof-checking relation.** At least one
    nontrivial proof-checking relation must be represented as data and
    replay-tested, with the host matching boundary visible.
16. **R16 — Host boundaries remain explicit.** Substitution,
    alpha-renaming, freshness, definitional equality, normalization,
    conversion, truth-table lookup, and structural proof matching must
    either be represented explicitly or marked as trusted host
    boundaries.

The broader programme proposed in @netkeep80's two comments (Phases
2–9: equality-layer provenance, proof-object substrate, links-defined
finite logics, links-defined type kernel, pure-links strict mode,
dependency-graph traversal, carrier enforcement, and experimental
`mtc-anum` profile) was originally framed as out of scope for the first
PR; subsequent commits on PR #175 implemented the backward-compatible
parts and left the still-hosted type/proof kernel status explicit. PR
#176 adds the semantic execution-boundary layer requested by the latest
feedback. See §10 for the phase-by-phase status.

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

PR #176 extends the snapshot with `semanticStatus` /
`semantic_status` on root constructs and active implementations, plus a
`bySemanticStatus` / `by_semantic_status` bucket. This layer records
whether an entry is `host-trusted`, `links-described`, `links-checked`,
`links-evaluated`, or `self-hosted`, so a `links-defined` truth-table row
set is no longer confused with a self-hosted evaluator.

### 4.4 Bundled foundations and finite-logic examples (R9)

`lib/self/foundations.lino` declares three named alternatives:

- `boolean-links` — strict `{0,1}` Boolean logic whose `and`, `or`, and
  `not` implementations are finite truth-table rows recorded as
  `links-defined` and `links-checked` while the foundation is active.
- `boolean-classical` — two-valued logic, `and=min`, `or=max`,
  `both=min`, `neither=product`; these are host aggregator bindings.
- `kleene-three-valued` — Strong Kleene, same operator shape over the
  real unit interval with `0.5` as `unknown`; these are also host
  aggregator bindings.

`examples/foundation-boolean-kleene.lino` exercises the host-backed
Boolean/Kleene profiles and the bundled `boolean-links` truth-table
profile in the same file, restoring the default after each scope.
`examples/foundation-with-min.lino` is a minimal one-operator demo.
Both examples are replayed by `rust/tests/shared_examples.rs` /
`js/tests/shared-examples.test.mjs` and by the self-evaluator parity
test, so any drift between the host and the data-encoded evaluator
breaks the build.

### 4.5 Self-bootstrap surface (R8)

`lib/self/evaluator.lino` now contains data-only rules for the new
foundation, strict-mode, proof-substrate, and MTC/anum surface:

```lino
(rule (eval (root-construct name details))
  (record-root-construct name details))

(rule (eval (foundation name details))
  (record-foundation name details))

(rule (eval (with-foundation name body))
  (evaluate-body-under-foundation name body))

(rule (eval (foundation-report))
  (emit-foundation-report))

(rule (eval (strict-foundation pure-links))
  (enable-strict-foundation pure-links))

(rule (eval (allow-host-primitive names))
  (record-allowed-host-primitives names))

(rule (eval (assumption name (judgement judgement)))
  (record-proof-assumption name judgement))

(rule (eval (axiom name (judgement judgement)))
  (record-proof-axiom name judgement))

(rule (eval (proof-object name clauses))
  (record-proof-object name clauses))

(rule (eval (check-proof name))
  (check-proof-object name))

(rule (eval (encodeAnum node))
  (encode-anum node))

(rule (eval (decodeAnum payload))
  (decode-anum payload))
```

The `EncodedEvaluator` in `js/tests/self-evaluator.test.mjs` was
taught to recognise these rules so the data-encoded evaluator replays
the new surface as faithfully as the host. The Rust counterpart in
`rust/tests/self_evaluator_tests.rs` checks the same patterns are
present.

### 4.6 Tests at every level (R11)

| Layer | JS | Rust |
|-------|----|------|
| **Unit** — `Env` lifecycle (preregister defaults, register, enter/exit, snapshot/restore, report shape) | `js/tests/foundations.test.mjs` | `rust/tests/foundations_tests.rs` |
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

1. `npm test` (run from `js/`) exercises the extended foundation,
   proof-substrate, pure-links strict-mode, self-evaluator, and shared
   example coverage with 0 regressions across the full JS suite.
2. `cargo test --all-targets` (run from `rust/`) exercises the mirrored
   Rust coverage, including `self_evaluator_tests` and
   `shared_examples` replay coverage, with 0 regressions across the
   full Rust suite.
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
profile added a ninth. PR #176 adds the explicit execution-boundary
classification requested by the latest issue feedback. Current status:

| Phase | Topic | Status | Where |
|-------|-------|--------|-------|
| 1 | Inventory + reporting + scoped overrides | done | §4.1–§4.3; `docs/FOUNDATIONS.md` §2–§4 |
| 2 | Equality and numeric-domain separation | done (registry-level) | `lib/self/foundations.lino` equality entries; trace-layer follow-up tracked |
| 3 | Proof-object substrate (`(check-proof …)`) | done | `E064`; `js/tests/proof-substrate.test.mjs` + `rust/tests/proof_substrate_tests.rs` |
| 4 | Links-defined finite logics (Boolean, truth-tables) | done for finite truth tables | §4.4; `boolean-links`, `examples/foundation-boolean-kleene.lino`, `(truth-table …)` |
| 5 | Links-defined type/proof kernel fragment | done | §9.6; `examples/typed-kernel-links.lino`; `js/tests/typed-kernel-links.test.mjs` + `rust/tests/typed_kernel_links_tests.rs`; `lib/self/foundations.lino` typed-kernel entries |
| 6 | Pure-links strict mode | done | `E065`; `(strict-foundation pure-links)` + `(allow-host-primitive …)` |
| 7 | Dependency-graph traversal | done | Rendered in `formatFoundationReport`; powers strict-mode paths |
| 8 | Carrier enforcement | done | `E063`; `(carrier …)` + `(strict-carrier)` |
| 9 | Experimental `mtc-anum` profile + `encodeAnum`/`decodeAnum` | done as a serialization profile **and** as a links-defined MTC theory fragment | §9.7; `E066`; pre-seeded but opt-in; `(experimental)`, `(root …)`, `(abit …)` clauses; `examples/mtc-anum-theory.lino`; canonicality/injectivity tests in both engines |
| 10 | Execution-boundary semantic statuses | done | `semantic-status`; `semanticStatus` / `semantic_status`; `bySemanticStatus` / `by_semantic_status`; docs §4.3 |
| 11 | Links-level proof-checking relation | done | §9.8; `examples/proof-checking-relation.lino`; proof-substrate tests in both engines |
| 12 | Links-defined Peano naturals (`zero`, `succ`, `add`, induction, `nat-equality`) | done | §9.9; `examples/nat-links.lino` (PR 178: explicit `nat-equality` layer plus `nat-refl` / `nat-cong-succ`, host `=`/`numeric-equality` untouched); `js/tests/nat-links.test.mjs` + `rust/tests/nat_links_tests.rs`; `(foundation nat-links …)` in `lib/self/foundations.lino` |
| 13 | `nat-recursion`, `nat-eliminator`, and multiplication (`nat-mul-zero`, `nat-mul-succ`) | done | §9.10; PR 178 commit `b5e9db4`; `examples/nat-links.lino` carries the recursor / eliminator / `mul` derivations; `js/tests/nat-links.test.mjs` + `rust/tests/nat_links_tests.rs` extend the parity tests to the twelve-rule fragment |
| 14 | Links-evaluated `(eval-nat <term>)` normalizer | done | §9.11; `docs/FOUNDATIONS.md` §9b; the normalizer dispatches through the active `nat-add-zero` / `nat-add-succ` / `nat-mul-zero` / `nat-mul-succ` proof rules, so removing or replacing one affects evaluation; host-number rendering is split into `nat-normal-form-to-host-number`; pinned by JS + Rust parity tests |
| 15 | Per-proof dependency / trust reporter `(proof-report <name>)` | done | §9.12; PR 178 commit `c6f5a14`; `docs/FOUNDATIONS.md` §9c; reports the proof object, the applied rule, the active foundation, and the transitive dependency edges as a small associative network |
| 16 | Strict pure-links audit pinned on the full Nat fragment | done | §9.13; PR 178 commit `172b8ec`; `(strict-foundation pure-links)` over `examples/nat-links.lino` emits zero `E065` diagnostics with the host-number renderer explicitly allowlisted; the audit is the regression gate for any future Peano rule |
| 17 | Typed/proof-kernel boundary report | done | §9.14; PR 178 commit `351c577`; `(foundation-report)` lists every typed-kernel root construct (`pi-formation`, `lambda-introduction`, `application-elimination`, `beta-conversion`, plus the still-host-trusted `substitution`, `freshness`, `alpha-renaming`, `beta-reduction`, `whnf`, `normalization`, `conversion`) so the trust boundary is visible without reading source |
| 18 | Meta-theory mapping document | done | §9.15; PR 178 commit `244524e`; `docs/META-THEORY-MAPPING.md` bridges every issue-#97 construct to the meta-theory's references / links / doublets / triplets vocabulary, fulfilling @netkeep80's "meta-theory mapping" deliverable |

Every implemented or partially implemented phase has **parallel JS and
Rust tests** plus the existing self-evaluator parity replay
(`lib/self/evaluator.lino` plus the test suites in
`js/tests/self-evaluator.test.mjs` and
`rust/tests/self_evaluator_tests.rs`), so the data-encoded evaluator
recognises the new surface forms on both engines.

### 9.1 Phase 3 — `(check-proof …)`

`(check-proof <name>)` replays a registered proof object against the
active proof-rule table. Each proof-object premise must be justified by
an `(assumption ...)`, `(axiom ...)`, or earlier proof object cited with
`(premise-by ...)` / `(uses ...)`; raw premises without a dependency
raise `E064`. The replay also checks premise counts, premise shapes,
conclusions, and cyclic proof dependencies.

### 9.2 Phase 6 — pure-links strict mode

`(strict-foundation pure-links)` flips a per-`Env` flag that causes any
subsequent query whose transitive dependency path reaches a
`host-primitive` or `host-derived` construct to raise `E065`, unless
that exact construct or dependency name is whitelisted by
`(allow-host-primitive <name>...)`. Active foundation implementations
are consulted first, so `and -> avg -> host-primitive` fails under the
default foundation while `and` under the bundled `boolean-links`
truth-table foundation is accepted as `links-defined`. The whitelist is
additive across declarations.

Strict mode accepts a truth-table implementation as fallback-free only
when the row set is total over the active strict carrier. Partial tables
remain supported for normal evaluation, but their active implementation
records a `truth-table-fallback` dependency so pure-links mode still
reports the host-backed path.

### 9.3 Phase 7 — dependency-graph traversal

The trust audit (`formatFoundationReport`) renders the active
implementation map and the transitive closure of `depends-on` so users
can see, before flipping the strict switch, exactly which paths would
need to be links-defined or explicitly whitelisted. The same traversal
powers Phase 6's enforcement check.

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
four-abit alphabet. Errors raise `E066`. The profile remains
descriptive and serialization-only at the host level; the companion
*theory* fragment expressed as links-defined rules is documented in
§9.7 below.

### 9.6 Phase 5 — links-defined typed-kernel fragment

The four typing rules of a dependent kernel (`pi-formation`,
`lambda-introduction`, `application-elimination`, `beta-conversion`)
are expressed as proof-substrate rules in
`examples/typed-kernel-links.lino`. They use `has-type` and
`turnstile` as literal identifiers, so `(empty turnstile (term has-type
T))` plays the role of the usual `Γ ⊢ t : T` judgement and matches
against `?metavariables` exactly like any other Phase 3 proof rule.

The example derives `(Pi (x : Nat) Nat)` from `Nat : Type0`, types the
identity function on `Nat` via `lambda-introduction`, applies it to
`zero` via `application-elimination`, and uses `beta-conversion` to
move the resulting redex back to `zero` at the substituted type. All
four `(check-proof …)` calls return `1` and the host engine emits no
diagnostics; the matching entry in `examples/expected.lino`
(`(typed-kernel-links.lino: 1 1 1 1)`) is checked by both the JS and
Rust expected-output harnesses. The companion `(foundation
typed-kernel-links …)` registration in `lib/self/foundations.lino`
records the four rules as `(root-construct … links-defined …)` so the
trust report and `foundation-report` API stay consistent with the
substrate.

Each rule is also pinned down individually by
`js/tests/typed-kernel-links.test.mjs` (8 tests) and
`rust/tests/typed_kernel_links_tests.rs` (8 tests), including a
negative case where swapping the domain of the conclusion's Pi type
makes `(check-proof …)` return `0` and raises `E064`. The
self-bootstrap evaluator (`EncodedEvaluator` in
`js/tests/self-evaluator.test.mjs`) replays the same example through
the data-encoded rules, completing the parity loop required for every
phase.

### 9.7 Phase 9 — MTC theory fragment and serialization invariants

The pre-seeded `mtc-anum` foundation was originally a serialization
skeleton: `encodeAnum`/`decodeAnum` round-trip arbitrary `Node` values
through a four-abit alphabet (`[`, `]`, `0`, `1`). The PR review
("Blocking issue 7") asked the foundation to also publish axioms and
rules expressed as links and to replay at least one non-trivial MTC
theorem. `examples/mtc-anum-theory.lino` supplies that fragment: three
theory rules (`root-is-link`, `frame-makes-link`, `pair-makes-link`)
plus three proof-objects that build a composite link from the root
`∞`. All three `(check-proof …)` calls return `1` and the expected
output is pinned in `examples/expected.lino` so both engines replay
the fragment in their shared-examples test suite.

The example also makes the **theory / serialization boundary**
explicit in prose: the rules speak about links themselves (the theory
domain), while the abits `[`, `]`, `0`, `1` are the alphabet of the
serialization domain. `decodeAnum(encodeAnum(x)) == x` is a
serialization invariant, not an MTC theorem.

`js/tests/mtc-anum.test.mjs` and `rust/tests/mtc_anum_tests.rs` pin
down three serialization invariants:

- **canonicality** — calling `encodeAnum`/`encode_anum` on the same
  input twice produces the same string.
- **injectivity** — distinct inputs encode to distinct strings (no
  collisions across a representative sample).
- **totality** — every encoding round-trips: `decodeAnum(encodeAnum(x))
  == x` for the same sample.

Both test files additionally cover the theory fragment: an end-to-end
example replay, an individual `frame-makes-link` derivation, a
negative case where swapping the conclusion's head raises `E064`, and
a structural check that the example file documents the theory /
serialization boundary.

### 9.8 PR #176 — semantic statuses and proof-checking relation

PR #175 used `links-defined` for both finite truth-table rows and proof
rules consumed by the host. The latest issue feedback called out that
this can hide the actual trust boundary: the rows and rules are links
data, but lookup, substitution, alpha-renaming, freshness checks,
normalization/conversion, and structural proof matching still execute
in JS/Rust.

PR #176 adds an execution-boundary layer:

- `semantic-status` in `.lino` descriptors.
- `semanticStatus` / `bySemanticStatus` in JavaScript reports.
- `semantic_status` / `by_semantic_status` in Rust reports.
- active implementations that print, for example,
  `and: links-defined; semantic links-checked; via truth-table:boolean-links/and`.

The new shared example
`examples/proof-checking-relation.lino` represents a data-level
proof-checking judgement:

```lino
(?proof checks-as ?conclusion)
```

Its `proof-checks-modus-ponens` rule checks that a proof object has an
implication proof, an antecedent proof, an applied rule tag, and two
dependency edges. The final `(check-proof mp-rain-checks-wet)` returns
`1`; both shared-example harnesses pin this in
`examples/expected.lino`.

### 9.9 Phase 12 — links-defined Peano naturals

The issue's original reference (Software Foundations / `Logic.v`) sits
on top of Coq's inductive `nat` definition. PR #176 covered the typed
kernel side (`pi-formation`, `lambda-introduction`,
`application-elimination`, `beta-conversion`) and PR #176 added a
proof-checking relation; the obvious missing piece was the **inductive
data layer itself**: a links-level account of the natural numbers so
that "Nat", "zero", "succ", "add", and induction are no longer silent
host primitives but proof-substrate rules consumed by `(check-proof
…)`.

`examples/nat-links.lino` supplies that fragment with seven
proof-substrate rules (PR 178 added the last three — `nat-refl`,
`nat-cong-succ`, and the dedicated equality layer `nat-equality` — on
top of PR 177's original five):

- `nat-zero-formation` — `(zero has-type Nat)` with no premises.
- `nat-succ-formation` — `(?n has-type Nat) ⇒ ((succ ?n) has-type Nat)`.
- `nat-add-zero` — `(?n has-type Nat) ⇒ ((add zero ?n) nat-equals ?n)`.
- `nat-add-succ` — `((add ?m ?n) nat-equals ?k) ⇒ ((add (succ ?m) ?n)
  nat-equals (succ ?k))`.
- `nat-induction` — `(?P at zero)` and `(forall ?n (implies (?P at ?n)
  (?P at (succ ?n))))` together imply `(forall ?n (?P at ?n))`.
- `nat-refl` — `(?n has-type Nat) ⇒ (?n nat-equals ?n)`. Reflexivity of
  the links-defined equality layer.
- `nat-cong-succ` — `(?m nat-equals ?n) ⇒ ((succ ?m) nat-equals (succ
  ?n))`. Successor respects `nat-equals`.

The example then builds `zero`, `(succ zero)`, and `(succ (succ
zero))` as inhabitants of `Nat`, computes `0+0`, `1+0`, `0+1`, and
`1+1` through `nat-add-zero` / `nat-add-succ`, witnesses `(zero
nat-equals zero)` through `nat-refl`, lifts `1+1` through
`nat-cong-succ` to derive `(succ (add (succ zero) (succ zero)))
nat-equals (succ (succ (succ zero)))`, and discharges a universal
claim via `nat-induction`. All ten `(check-proof …)` calls return
`1` and the engines emit no diagnostics. The matching entry in
`examples/expected.lino` (`(nat-links.lino: 1 1 1 1 1 1 1 1 1 1)`) is
checked by both the JS and Rust expected-output harnesses, so any
drift between engines is caught at shared-examples replay time.

The example deliberately uses `nat-equals` rather than the bare
literal `equals` so the trust audit can distinguish the
links-defined equality layer from `=`/`numeric-equality`. Programs
that never opt into `nat-links` keep the host's decimal-12 `=`
semantics unchanged — both engines pin this with the
`leaves the host =/numeric-equality layer unchanged when nat-links is
not selected` test in `js/tests/nat-links.test.mjs` and the matching
`leaves_the_host_equality_layer_unchanged_when_nat_links_is_not_selected`
test in `rust/tests/nat_links_tests.rs`.

The companion `(foundation nat-links …)` registration in
`lib/self/foundations.lino` records the Nat proof rules, the
`nat-equality` equality-layer root construct, and the `eval-nat`
normalizer/renderer split as root constructs so the trust audit lists
the links-checked and links-evaluated derivations rather than silent
host primitives. The host's decimal numeric domain and
`=`/`numeric-equality` are unaffected — `Nat` here is purely
structural and `nat-equals` lives in a separate equality layer; the
foundation just gives the trust report something concrete to point at
when a user asks "where do
`Nat`, `succ`, addition, and their equality come from in this proof?".

Each rule is also pinned down individually by
`js/tests/nat-links.test.mjs` (14 tests) and
`rust/tests/nat_links_tests.rs` (14 tests), including three negative
cases (a mistyped `(succ zero)`, an `add-succ` claim that would
require `(succ ?k)` to unify with `zero`, and a `nat-cong-succ`
derivation that drops one of the `succ` wrappers) where
`(check-proof …)` returns `0` and raises `E064`. The pre-registration
test asserts that `(foundation-report)` lists `nat-links` with
`uses = [nat-add-succ, nat-add-zero, nat-cong-succ, nat-equality,
nat-induction, nat-refl, nat-succ-formation, nat-zero-formation]`
(sorted) and `extends = default-rml`, matching the
`_registerDefaultFoundation` / `register_default_foundation` seeds in
both engines.

This phase deliberately does **not** implement `Nat` by host numeric
conversion: there is no `succ → +1` shortcut, and there is no
evaluator dispatch on `nat-equals` — the operator stays a bare
literal that the proof substrate's structural matcher compares for
syntactic identity. Successor is a literal constructor, so the only
host operation involved is the same pattern matcher that backs every
other Phase 3 proof rule. Substitution, alpha-renaming, and freshness
remain explicit `host-trusted` boundaries documented in the
foundation report.

### 9.10 Phase 13 — `nat-recursion`, `nat-eliminator`, and multiplication

PR 178 commit `b5e9db4` extends the Peano fragment from the original
five rules (plus the equality layer) to the twelve-rule, fully
Software-Foundations-aligned layer. The added rules are:

- `nat-recursion` — `(?P at zero)` and a step
  `(?P at ?n) ⇒ (?P at (succ ?n))` derive `(?P at ?n)` at any
  specific `?n` reachable from `zero`. The recursor is the
  computational counterpart of `nat-induction`: where induction
  delivers `(forall ?n (?P at ?n))`, recursion delivers a single
  derived inhabitant for the witnessing `?n`.
- `nat-eliminator` — the dependent eliminator. The premises are
  `(?P at zero)` and `(forall ?n (implies (?P at ?n) (?P at (succ
  ?n))))`; the conclusion is the same as `nat-induction` but tagged
  as an *eliminator* in the registry so the trust audit can tell the
  inductive presentation apart from the eliminator presentation.
- `nat-mul-zero` — `((mul zero ?n) nat-equals zero)`; no premises.
- `nat-mul-succ` — `((mul ?m ?n) nat-equals ?k) ⇒ ((mul (succ ?m)
  ?n) nat-equals (add ?n ?k))`. Multiplication is a *links-defined*
  recursive reduction layered on top of `nat-add-*`; there is no
  `succ → ×k` shortcut.

`examples/nat-links.lino` now exercises each of the four new rules
through `(check-proof …)`. The shared expected-output harness pins
the additional `1`s and the `(foundation-report)` snapshot picks up
the extended `uses` list. The Software-Foundations-style theorems
(`plus_O_n`, `plus_n_O`, `plus_n_Sm`, `plus_comm`, `mult_0_r`,
`mult_n_Sm`) are encoded as a small sequence of proof objects that
each cite an earlier proof object as their `premise-by`, so the
proof-substrate's dependency check is the same engine that gates the
typed kernel.

### 9.11 Phase 14 — `(eval-nat <term>)` normalizer

PR 178 adds the **links-evaluated** Peano normalizer documented in
`docs/FOUNDATIONS.md` §9b. `(eval-nat <term>)` recognises the grammar
`zero | (succ T) | (add T T) | (mul T T)` and normalizes it by
dispatching through the active computation rules `nat-add-zero`,
`nat-add-succ`, `nat-mul-zero`, and `nat-mul-succ`. Removing a needed
rule from the active foundation makes evaluation fail with `E067`, and
replacing a rule changes the result. The semantic result is the Peano
normal form; the host number in the result stream is the separate
`nat-normal-form-to-host-number` renderer.

Trust-report integration:

- The root construct `eval-nat-normalize` carries the links-evaluated
  dependency on `nat-add-zero`, `nat-add-succ`, `nat-mul-zero`,
  `nat-mul-succ`, and `structural-matcher`. The surface construct
  `eval-nat` depends on that normalizer and on the separate
  host-derived `nat-normal-form-to-host-number` renderer.
- The `eval-nat` normalizer family gives the
  `links-evaluated` semantic status a concrete occupant, so the
  `bySemanticStatus` / `by_semantic_status` view is no longer empty
  under that key.
- Anything outside the recognised grammar raises `E067`, leaving
  surrounding queries intact.

Both engines pin the normalizer with parity tests
(`js/tests/nat-links.test.mjs`, `rust/tests/nat_links_tests.rs`),
including a negative case where `(eval-nat (succ apple))` raises
`E067` and the surrounding queries continue to evaluate.

### 9.12 Phase 15 — `(proof-report <name>)`

PR 178 commit `c6f5a14` adds the per-proof analogue of
`(foundation-report)`. Given an already-registered proof object,
`(proof-report <name>)` returns a structured snapshot containing:

- the **proof object's name** and **conclusion**;
- the **rule** it applies (the `(applies …)` clause);
- the **transitive dependency edges** — every `(premise-by …)` /
  `(uses …)` reference resolved to the named assumption, axiom, or
  earlier proof object;
- the **active foundation** at the time of the call;
- and the **trust status** of every cited rule, so a reader can see
  whether the proof rests on `links-defined` or `host-trusted`
  primitives.

The report is a small associative network: a root node (the proof
object), an edge to the applied rule, and a flat list of edges to
dependencies by name. `docs/META-THEORY-MAPPING.md` §3 documents how
that network maps onto the meta-theory's doublets and triplets.

`examples/nat-links.lino` calls `(proof-report …)` on representative
proofs (the zero/succ inhabitants and one Software-Foundations-style
theorem) so the parity replay covers the new form. Both engines
emit byte-identical reports; this is enforced by the shared
expected-output harness.

### 9.13 Phase 16 — strict pure-links audit on the full Nat fragment

PR 178 commit `172b8ec` pins `(strict-foundation pure-links)` over
the entire `examples/nat-links.lino` file: the audit emits **zero**
`E065` diagnostics. Concretely:

- Every proof rule in the twelve-rule fragment is registered with
  `(root-construct … (status links-defined) (semantic-status
  links-checked) …)` and a `(depends-on …)` list that bottoms out at
  the small set of explicitly-whitelisted host primitives
  (`structural-matcher`, `substitution`, `freshness`, `alpha-
  renaming`) that the strict-mode `(allow-host-primitive …)` clause
  in the example file enumerates.
- The `(eval-nat …)` normalizer's `(depends-on …)` chain is bounded by
  the same explicit boundary set: it dispatches through `nat-add-*` and
  `nat-mul-*`, which are themselves links-defined, and reports the
  structural matcher / renderer boundary separately. The example
  allowlists `nat-normal-form-to-host-number` so strict mode can still
  reject accidental host arithmetic while accepting the legacy numeric
  result renderer.
- Any future regression — say, a new Peano rule that quietly depends
  on `host-derived` `numeric-equality` — fails the audit at
  shared-examples replay time. The strict-mode audit is therefore
  the **regression gate** for the entire Nat fragment, not a
  one-time check.

`js/tests/nat-links.test.mjs` and `rust/tests/nat_links_tests.rs`
encode the gate as a parity test ("strict pure-links accepts the
full nat-links fragment with zero E065"), and the negative test
("strict pure-links rejects a synthetic rule that depends on
numeric-equality") confirms the gate fails as expected when
violated.

### 9.14 Phase 17 — typed/proof-kernel boundary pinning report

PR 178 commit `351c577` makes the kernel boundary explicit in the
trust report. `(foundation-report)` now lists, in a stable order,
every root construct that the typed kernel and proof substrate
depend on, together with its `status`, its `semanticStatus` /
`semantic_status`, and its `depends-on` edges:

- **`links-defined` / `links-checked`** — `pi-formation`,
  `lambda-introduction`, `application-elimination`,
  `beta-conversion`, and every Phase 12/13 Peano rule.
- **`host-primitive` / `host-trusted`** — `substitution`,
  `freshness`, `alpha-renaming`, `beta-reduction`, `whnf`,
  `normalization`, `conversion`. These cannot be reduced to a finite
  associative network without losing decidability and remain
  trusted; the report records them explicitly so a user does not
  have to read JS or Rust source to find them.

The pinning test
`js/tests/typed-kernel-links.test.mjs` ("reports the complete
typed-kernel boundary in foundation-report") and its Rust mirror
`rust/tests/typed_kernel_links_tests.rs` enforce the boundary
mechanically: adding, removing, or re-tagging any kernel root
construct without updating the test fails the build.

### 9.15 Phase 18 — meta-theory mapping document

PR 178 commit `244524e` adds `docs/META-THEORY-MAPPING.md`, the
deliverable that @netkeep80's "meta-theory mapping" review item
asked for. The document bridges every issue-#97 construct — the
twelve-rule Nat fragment, the proof-substrate, rule application,
inductive closure, and the trust report — to the universal
vocabulary of [Links Theory
(meta-theory)](https://github.com/link-foundation/meta-theory):
**references**, **links**, **doublets** (2-tuples), and **triplets**
(3-tuples).

Highlights:

- The inductive constructors `Nat`, `zero`, `succ`, `add`, and `mul`
  factor through references, doublets, and triplets at the leaves —
  no construct exceeds arity 3 once the rule body is unfolded.
- A proof object is a small associative network whose root edges
  are doublets (`applies`, `premise-by`, `conclusion`) and whose
  leaves are triplet judgements.
- "Closure under a rule" is **not** a primitive in the meta-theory;
  it is the result of replaying a rule application zero or more
  times. `nat-induction` is therefore expressible as a finite
  network of triplets that, when instantiated at a predicate, lets
  `(check-proof …)` produce the closure on demand.
- The trust report itself reads as a labelled graph (`foundation`,
  `uses`, `depends-on` are all doublets), so `(foundation-report)`
  is literally an instance of the meta-theory in action.

The document closes with the reading guide and the explicit list of
host-trusted operations and the reasons each is irreducible to a
finite link network, completing the chain from
issue → constructs → tests → trust audit → meta-theory.

## 10. Files in this case study

- `README.md` — this document.
- `data/issue-97.json` — issue body, labels, author.
- `data/issue-97-comments.json` — full conversation comments
  (@netkeep80's two implementation comments and @konard's final
  contract comment).
- `data/pr-174.json` — pull-request metadata.
- `data/pr-175.json` — PR #175 metadata for the completed phase work.
- `data/pr-176.json` — PR #176 metadata for the semantic-status follow-up.
- `evidence/foundation-surface.md` — line-numbered code-level
  evidence for requirements R1–R16.
- `evidence/cicd-template-review.md` — review of the four CI/CD
  templates against this repository's `.github/workflows/`.
- [`../../META-THEORY-MAPPING.md`](../../META-THEORY-MAPPING.md) —
  documents how the inductive Peano fragment, the proof-substrate,
  rule application, inductive closure, and the trust report map onto
  the references / links / doublets vocabulary of the
  [meta-theory](https://github.com/link-foundation/meta-theory). The
  bridge required by netkeep80's "meta-theory mapping" deliverable.
