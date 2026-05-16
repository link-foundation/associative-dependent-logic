# Evidence — Foundation surface, requirement by requirement

Line numbers below refer to file state at branch
`issue-97-bbe597194dee`, head commit at the time this case study was
compiled. The mapping resolves every requirement R1–R13 from
`docs/case-studies/issue-97/README.md` to concrete code, data, or
documentation.

## R1 — Root-construct registry

A `Map<name → descriptor>` lives on `Env` in both engines, populated
on construction:

- JS — `js/src/rml-links.mjs:298` calls `this._registerDefaultFoundation()`
  inside the `Env` constructor. The seed routine is at
  `js/src/rml-links.mjs:504-516` and the public surface is
  `registerRootConstruct(descriptor)` at
  `js/src/rml-links.mjs:517-534`.
- Rust — `rust/src/lib.rs:1110` calls `env.register_default_foundation()`
  from `Env::new`. The seed routine is
  `register_default_foundation` at `rust/src/lib.rs:1163-1178`
  and the public surface is
  `register_root_construct` at `rust/src/lib.rs:1179-1202`.

The same set of descriptors is recorded as data in
`lib/self/foundations.lino` (Parsing / LiNo layer, Numeric layer,
Truth / aggregator layer, Equality layer, Typed kernel layer,
Self-bootstrap layer). The host-seeded set and the data-encoded set
are kept in sync deliberately so the future self-hosted evaluator can
reconstruct the registry from data without behavioural drift.

## R2 — Eight trust statuses

The eight statuses are declared as data in
`lib/self/foundations.lino:24-31`:

```lino
(trust-status host-primitive reads implemented-by-host-kernel)
(trust-status host-derived reads defined-in-host-from-other-host-primitives)
(trust-status external-trusted reads relies-on-external-binary-such-as-smt-or-atp)
(trust-status user-configurable reads runtime-knobs-the-user-can-tune)
(trust-status links-encoded reads description-lives-in-self-bootstrap-as-data)
(trust-status links-defined reads behaviour-derived-by-evaluating-self-rules)
(trust-status user-overridden reads replaced-by-user-foundation)
(trust-status planned reads not-yet-implemented)
```

`foundationReport()` returns the `byStatus` bucket containing every
known status; descriptors that do not declare a status default to
`host-primitive` (JS `js/src/rml-links.mjs:593-651`, Rust
`rust/src/lib.rs:1264-1300`). The doc surface for the statuses lives
at `docs/FOUNDATIONS.md`.

## R3 — Dependency graph via `depends-on`

Every descriptor accepts an optional `(depends-on …)` clause. The
arithmetic and comparison operators record their dependency on
`decimal-12-arithmetic`:

- `lib/self/foundations.lino:65-93` — `+`, `-`, `*`, `/`, `<`, `<=`
  declare `(depends-on decimal-12-arithmetic)`.

The parser in `js/src/rml-links.mjs:813-883` and
`rust/src/lib.rs:9952-9963` accumulates `depends-on` items into the
descriptor's array, and `foundationReport()` surfaces them under
each entry's `dependsOn` field.

## R4 — Foundation scope (`(foundation …)`, `(with-foundation …)`)

Three surface forms cover the lifecycle:

| Form | JS | Rust |
|------|----|------|
| `(foundation <name> …)` | parser at `js/src/rml-links.mjs:885-934`; register at `js/src/rml-links.mjs:536-548` | parser/dispatch at `rust/src/lib.rs:9925-9947` / `10222-10247`; register at `rust/src/lib.rs:1203-1219` |
| `(with-foundation <name> body)` | dispatch at `js/src/rml-links.mjs:5541-5604`; enter/exit at `js/src/rml-links.mjs:550-580` | dispatch at `rust/src/lib.rs:9879-9920` / `10254-10300`; enter/exit at `rust/src/lib.rs:1220-1263` |
| `(foundation-report)` | dispatch at `js/src/rml-links.mjs:5716-5733`; report at `js/src/rml-links.mjs:593-651` | dispatch at `rust/src/lib.rs:9969-9990` / `10311-...`; report at `rust/src/lib.rs:1264-1300` |

`enterFoundation` (JS) and `enter_foundation` (Rust) snapshot every
operator the foundation rebinds, then apply the new bindings; the
snapshot is stored on a stack so nested scopes restore correctly.
`exitFoundation` / `exit_foundation` pop the most recent frame and
restore the operators verbatim, including the previously-active
foundation tag.

Nesting is covered by:
- JS — `js/tests/foundations.test.mjs:66-82`
  (`it('nests \`(with-foundation ...)\` scopes correctly', …)`).
- Rust — `rust/tests/foundations_tests.rs:94-114`
  (`fn nests_with_foundation_scopes_correctly()`).

## R5 — Backward compatibility (`default-rml` pre-registered)

The default foundation is loaded automatically by `Env::new` /
`new Env`. It mirrors the legacy operator semantics:

- JS — `_registerDefaultFoundation` at
  `js/src/rml-links.mjs:504-516` seeds `default-rml` with
  `(defines and avg) (defines or max) (defines both avg) (defines neither product)`.
- Rust — `register_default_foundation` at
  `rust/src/lib.rs:1163-1178` does the same.

The contract is asserted by:
- JS — `js/tests/foundations.test.mjs:25-46`
  ("preregisters `default-rml`" and "does not change baseline
  semantics when no foundation is declared").
- Rust — `rust/tests/foundations_tests.rs:46-73`
  (`preregisters_default_rml_so_legacy_programs_need_no_migration`,
  `baseline_semantics_unchanged_when_no_foundation_is_declared`).

Every other `.lino` file in `examples/` and `test-corpus/` is also
replayed through both engines unchanged
(`rust/tests/shared_examples.rs`, `js/tests/shared-examples.test.mjs`).

## R6 — `(foundation-report)` form, byte-identical output

The structured snapshot is produced by:

- JS — `foundationReport()` at
  `js/src/rml-links.mjs:593-651`; rendered by
  `formatFoundationReport` at `js/src/rml-links.mjs:939-1010`.
- Rust — `foundation_report` at `rust/src/lib.rs:1264-1300`;
  rendered by `format_foundation_report` at
  `rust/src/lib.rs:2519-...`.

The two text renderers emit byte-identical lines for the same input;
this is checked by replaying the foundation examples through both
engines in the shared-example test
(`rust/tests/shared_examples.rs`, `js/tests/shared-examples.test.mjs`).

Snapshot-shape tests:
- JS — `js/tests/foundations.test.mjs:121-145`
  ("builds a structured `foundation-report` snapshot").
- Rust — `rust/tests/foundations_tests.rs:172-211`
  (`builds_a_structured_foundation_report_snapshot`).

## R7 — Structured diagnostics E060 / E061 / E062 and follow-up foundation diagnostics

The codes are emitted at:

- JS — E060 first at `js/src/rml-links.mjs:519`, parser checks
  `js/src/rml-links.mjs:816-867`, dispatch fallback
  `js/src/rml-links.mjs:5682-5695`.
- JS — E061 first at `js/src/rml-links.mjs:538`, parser checks
  `js/src/rml-links.mjs:885-934`, dispatch fallback
  `js/src/rml-links.mjs:5694-5715`.
- JS — E062 first at `js/src/rml-links.mjs:553`, dispatch fallback
  `js/src/rml-links.mjs:5542-5602`.
- Rust — E060 at `rust/src/lib.rs:9953`, `9963`, `10211`, `10224`.
- Rust — E061 at `rust/src/lib.rs:9933`, `9943`, `10235`, `10248`.
- Rust — E062 at `rust/src/lib.rs:9884`, `9894`, `9902`, `10256`, `10266`, `10275`.

Unknown-foundation tests:
- JS — `js/tests/foundations.test.mjs:84-97`
  ("reports an unknown foundation as E062 without aborting evaluation").
- Rust — `rust/tests/foundations_tests.rs:116-132`
  (`reports_unknown_foundation_as_e062_without_aborting`).

The doc surface — `docs/DIAGNOSTICS.md` — lists E060 through E066 in
its error code table. E063 covers strict-carrier failures, E064 covers
proof-substrate failures, E065 covers pure-links strict-mode failures,
and E066 covers MTC/anum encode/decode failures.

## R8 — Self-bootstrap encoding

`lib/self/foundations.lino` declares the registry as data and records
the bundled foundations, including the finite truth-table
`boolean-links` profile. `lib/self/evaluator.lino` encodes matching
data-only evaluator rules for the foundation surface, strict mode,
proof substrate, and MTC/anum helpers:

```lino
(rule (eval (root-construct name details))
  (record-root-construct name details))

(rule (eval (foundation name details))
  (record-foundation name details))

(rule (foundation-clause (carrier values))
  (record-foundation-carrier values))

(rule (foundation-clause strict-carrier)
  (record-foundation-strict-carrier))

(rule (foundation-clause (truth-table operator rows))
  (record-foundation-truth-table operator rows))

(rule (foundation-clause experimental)
  (record-foundation-experimental))

(rule (foundation-clause (root symbol))
  (record-foundation-root symbol))

(rule (foundation-clause (abit symbol bits))
  (record-foundation-abit symbol bits))

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

(rule (proof-object-clause (premise-by name))
  (record-proof-dependency name))

(rule (proof-object-clause (uses names))
  (record-proof-dependencies names))

(rule (eval (check-proof name))
  (check-proof-object name))

(rule (eval (encodeAnum node))
  (encode-anum node))

(rule (eval (decodeAnum payload))
  (decode-anum payload))
```

The Rust self-evaluator parity test requires these rules to be
present in the data file:
`rust/tests/self_evaluator_tests.rs` enumerates `REQUIRED_EVAL_RULES`,
including the foundation, strict-mode, proof, and MTC/anum entries, and
`REQUIRED_SURFACE_RULES`, including foundation clauses, proof dependency
clauses, and equality provenance. The JS parity test covers the same
forms with the `EncodedEvaluator` class, including the bundled
`boolean-links` truth-table foundation.

## R9 — Links-defined finite-logic example

Three named alternative foundations ship as data in
`lib/self/foundations.lino`:

- `boolean-links` — strict `{0,1}` Boolean logic with finite
  truth-table rows for `and`, `or`, and `not`. Activating this
  foundation records those operators as `links-defined`.
- `boolean-classical` — two-valued logic, `and=min`, `or=max`,
  `both=min`, `neither=product`, `(extends default-rml)`. These are
  host aggregator bindings.
- `kleene-three-valued` — Strong Kleene over the real unit interval,
  same `and`/`or` shape. These are also host aggregator bindings.

A complete end-to-end example uses all three profiles inside one file:
`examples/foundation-boolean-kleene.lino` (replayed through both
engines). A minimal sanity-check example is at
`examples/foundation-with-min.lino` (16 lines). Both are picked up
automatically by the shared-example replay
(`rust/tests/shared_examples.rs`, `js/tests/shared-examples.test.mjs`).

## R10 — Twin engine parity

The shared-example replay
(`rust/tests/shared_examples.rs`,
`rust/tests/shared_test_corpus.rs`, `js/tests/shared-examples.test.mjs`)
runs every file in `examples/` and `test-corpus/` through both
engines and asserts that the printed output is byte-identical.
`formatFoundationReport` (JS, `js/src/rml-links.mjs:939-1010`)
and `format_foundation_report` (Rust, `rust/src/lib.rs:2519-…`) are
designed for exactly this byte-identity contract.

The self-evaluator parity replays the foundation examples through the
data-encoded `EncodedEvaluator` and compares the result to the host
evaluator. JS — `js/tests/self-evaluator.test.mjs:141-300` covers the
foundation surface explicitly; Rust —
`rust/tests/self_evaluator_tests.rs:11-65` enforces that the same set
of rule patterns exists in the data file.

## R11 — Unit, integration, end-to-end coverage

| Layer | JS | Rust |
|-------|----|------|
| Unit (Env lifecycle) | `js/tests/foundations.test.mjs:25-46`, `121-163` | `rust/tests/foundations_tests.rs:46-74`, `172-249` |
| Integration (operator swap, nesting, E062) | `js/tests/foundations.test.mjs:48-97` | `rust/tests/foundations_tests.rs:75-132` |
| End-to-end (`.lino` files) | `js/tests/shared-examples.test.mjs`, `js/tests/self-evaluator.test.mjs` | `rust/tests/shared_examples.rs`, `rust/tests/shared_test_corpus.rs`, `rust/tests/self_evaluator_tests.rs` |

The CI workflow `.github/workflows/tests.yml` already runs `npm test`
and `cargo test` on every push and PR, so the new tests are picked
up without workflow changes. See
`evidence/cicd-template-review.md` for the cross-template review.

## R12 — Documentation

- `docs/FOUNDATIONS.md` — surface reference (registry concept, eight
  trust statuses, foundation scope, report shape, diagnostic codes,
  bundled foundations, programme roadmap).
- `docs/DIAGNOSTICS.md` — E060/E061/E062 rows added to the error code
  table, each linking back to `docs/FOUNDATIONS.md`.
- `README.md` — *Comparisons* section now lists a pointer to
  `docs/FOUNDATIONS.md` immediately after `CONFIGURABILITY.md`.

## R13 — Case study compilation

- `docs/case-studies/issue-97/README.md` — narrative.
- `docs/case-studies/issue-97/data/issue-97.json` — issue payload.
- `docs/case-studies/issue-97/data/issue-97-comments.json` — all
  conversation comments (the two implementation comments and
  the @konard backward-compatibility / CI/CD contract).
- `docs/case-studies/issue-97/data/pr-174.json` — PR metadata.
- `docs/case-studies/issue-97/evidence/foundation-surface.md` —
  this file.
- `docs/case-studies/issue-97/evidence/cicd-template-review.md` —
  CI/CD template comparison.
