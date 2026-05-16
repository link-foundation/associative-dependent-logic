# Foundations

This document describes RML's **root-construct registry** and **foundation
scope** (issue [#97](https://github.com/link-foundation/relative-meta-logic/issues/97)).
The two together let users replace the meaning of operators such as `and`,
`or`, `both`, `neither` without touching the evaluator, and inspect what the
prover is actually trusting at any point in time.

The headline guarantee is backward compatibility:

> Every `.lino` source file that ran before this surface existed runs
> identically afterwards. The default foundation `default-rml` is
> pre-registered automatically, mirrors the current host-implemented
> semantics, and is the only foundation active until a `(with-foundation
> ...)` form says otherwise.

If you are looking for **what** is reconfigurable inside the default
foundation (range, valence, aggregator selection, operator composition),
see [`CONFIGURABILITY.md`](./CONFIGURABILITY.md). This page is the broader
story: how every primitive the kernel uses is catalogued, given a trust
status, and how an alternative foundation can re-bind those primitives
inside a scoped region.

## 1. Why foundations

RML is a meta-logic; its job is to host *other* logic systems. The
configurability story in `CONFIGURABILITY.md` already lets a user pick a
range, valence, aggregator, or operator composition by writing ordinary
top-level forms. That is enough to swap, say, fuzzy `and = avg` for
classical `and = min`, but it leaves the bookkeeping implicit:

- Which root constructs are *host primitives* — implemented directly in
  JS/Rust and trusted unconditionally?
- Which are *links-encoded* (described as data in `lib/self/*.lino`) vs.
  *links-defined* (their meaning is derived by evaluating those data
  rules)?
- Which are *user-configurable* runtime knobs?
- Which are *external trusted* oracles (SMT, ATP)?
- Which are *planned* but not yet implemented?

A registry that records this status for every primitive — plus a scope
construct that swaps a coherent bundle of those primitives in one move —
is the difference between a configurable prover and a foundationally
honest one.

## 2. The root-construct registry

Every primitive the evaluator, type checker, proof-replay checker,
tactic engine, or metatheorem checker depends on has a descriptor in the
registry. Descriptors are *data only* — they never change evaluator
behaviour. They feed the `(foundation-report)` form and the trust audit
the CLI emits.

A descriptor field is one of:

| Field | Meaning |
|-------|---------|
| `kind` | What category the construct belongs to (e.g. `truth-operator`, `aggregator`, `binder`, `equality-layer`). |
| `status` | One of the trust statuses below. |
| `depends-on` | Names of constructs this one builds on. Forms a dependency graph the report can traverse. |
| `encoded-as` | The host symbol that currently implements the construct (for documentation only). |
| `pure-links-ready` | `yes` if the construct's meaning is derivable from links alone, `no` otherwise. |
| `planned-as` | The status the construct should reach in a future milestone. |

Trust statuses, mirroring the categories proposed in the issue thread:

| Status | When to use it |
|--------|----------------|
| `host-primitive`     | Implemented directly in JS/Rust and trusted unconditionally. |
| `host-derived`       | Defined in host code from other host primitives. |
| `external-trusted`   | Relies on an external binary (SMT, ATP) or system service. |
| `user-configurable`  | A runtime knob the user can tune (range, valence, operator aggregator). |
| `links-encoded`      | Description lives in `lib/self/*.lino` as data; the host still interprets it. |
| `links-defined`      | Behaviour is derived by evaluating the encoded self-rules. |
| `user-overridden`    | Replaced by an active user foundation inside the current scope. |
| `planned`            | Not yet implemented. |

The canonical inventory lives in [`lib/self/foundations.lino`](../lib/self/foundations.lino).
That file is the single source of truth — both the JS and Rust hosts pre-seed
the same descriptors on construction so that a freshly constructed `Env`
already reports the default trust base. Top-level `(root-construct ...)`
forms in user files extend or refine that base.

### Declaring a root construct

```lino
(root-construct my-and
  (kind truth-operator)
  (status links-defined)
  (depends-on truth-range))
```

Repeated declarations merge non-destructively: an explicit field
overrides the prior value, and an omitted field preserves the prior one,
so partial descriptors can be accumulated.

### Diagnostics

| Code | Cause |
|------|-------|
| `E060` | Malformed `(root-construct ...)` declaration (missing name, unknown child clause shape, non-symbolic field value). |
| `E061` | Malformed `(foundation ...)` declaration (missing name, unknown child clause shape, malformed `(defines ...)` clause, malformed `(root ...)` / `(abit ...)` clause). |
| `E062` | `(with-foundation <name> ...)` references a foundation that has not been registered. The diagnostic does not abort evaluation — forms after the bad scope still run. |
| `E063` | Carrier violation under `(strict-carrier)` — a query result or probability assignment falls outside the active foundation's declared carrier. |
| `E064` | Malformed proof rule / assumption / proof object / `(check-proof ...)` form, premise/conclusion mismatch, unjustified raw premise, or cyclic proof dependency. |
| `E065` | Pure-links strict mode rejected a query whose transitive dependency path reaches an unallowed `host-primitive` / `host-derived` construct; also raised for malformed `(strict-foundation ...)` / `(allow-host-primitive ...)` forms. |
| `E066` | MTC/anum encode/decode error (input outside the four-abit alphabet, unbalanced frame, leaf payload not byte-aligned, or non-Node value passed to `encodeAnum`). |

## 3. Foundations

A foundation bundles a coherent set of root-construct interpretations.
Declarations look like:

```lino
(foundation classical-min
  (description two-valued-classical-boolean-logic)
  (defines and min)
  (defines or max))
```

Field reference:

| Clause | Meaning |
|--------|---------|
| `(description <text>)` | Free-form summary; shown by `(foundation-report)`. |
| `(numeric-domain <name>)` | Name of the numeric domain this foundation expects. |
| `(truth-domain <name>)` | Name of the truth domain. |
| `(extends <name>)` | Inherits fields from a previously registered foundation. |
| `(defines <op> <aggregator>)` | Re-binds operator `<op>` to aggregator `<aggregator>` while the foundation is active. Repeats are allowed; each new `(defines ...)` replaces the binding for that operator. |
| `(carrier <v1> <v2> ...)` | Declares the set of values the foundation considers legal. Symbolic constants (`true`, `false`, `unknown`) resolve through `env.symbol_prob` on activation; numeric literals stay literal. Informational unless `(strict-carrier)` is also present (see §6). |
| `(strict-carrier)` | Opts the foundation into runtime carrier enforcement. Out-of-carrier query results and probability assignments raise `E063` instead of being silently clamped. |
| `(truth-table <op> (in1 in2 -> out) ...)` | Rebinds `<op>` to a finite truth table for the duration of `(with-foundation ...)` and records that active implementation as `links-defined`. The host still executes the table lookup; the selected behaviour comes from the rows. Partial tables are allowed — rows that don't match fall through to the previously installed op. Symbolic truth constants resolve through `env.symbol_prob` on activation. |
| `(experimental)` | Flags the foundation as experimental so the trust audit prints an `[experimental]` tag next to its name. Carries no behavioural guarantees. |
| `(root <symbol>)` | Records the foundation's root concept (e.g. `∞` for `mtc-anum`). Informational; surfaced on the report. |
| `(abit <symbol> <meaning>)` | Records one atomic bit of the foundation's alphabet. Used by experimental profiles like `mtc-anum` to publish their four-abit (`[`, `]`, `0`, `1`) serialization alphabet. Informational; surfaced on the report. |

### The `default-rml` foundation

`default-rml` is pre-registered on every fresh `Env`. It records the
current host-implemented operator semantics (`and=avg`, `or=max`,
`both=avg`, `neither=product`) so that any program written before
foundations existed keeps the same observable behaviour. There is no
`(use-default-rml ...)` form to call — the default is simply always
active until a `(with-foundation ...)` enters another foundation.

`active foundation: default-rml` is therefore the canonical state for
any program that has not declared its own foundation.

### Entering a foundation: `(with-foundation ...)`

```lino
(foundation classical-min
  (defines and min)
  (defines or max))

(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)

; avg → 0.5
(? ((a = true) and (b = true)))

(with-foundation classical-min
  ; min → 0.4
  (? ((a = true) and (b = true)))
  ; max → 0.6
  (? ((a = true) or (b = true))))

; back to avg → 0.5
(? ((a = true) and (b = true)))
```

Semantics:

1. On entry, the host snapshots the current operator table for every
   construct the foundation `(defines ...)`. The snapshot includes the
   previously-bound aggregator (or its absence).
2. The foundation's `(defines ...)` bindings are then applied: each named
   operator gets the named aggregator. Unknown aggregators trigger
   `E061`; unknown operators are tolerated for forward-compatibility.
3. The body forms evaluate normally. Inside the body, `(foundation ...)`
   declarations are still allowed and are recorded just like at the top
   level. Nested `(with-foundation ...)` forms recurse with their own
   snapshot/restore frame.
4. On exit, the snapshot is restored verbatim — the previously-bound
   operators come back, and the previously-active foundation tag is
   restored.

Scoping is purely lexical: only forms that appear inside the body of a
`(with-foundation ...)` see the swapped operators. Forms after the scope
see the previous bindings restored.

### Errors that do not abort

If `(with-foundation <name> ...)` names a foundation that has not been
registered, the host emits an `E062` diagnostic for the form and skips
the body, then continues evaluating the rest of the file. The
surrounding queries still run with the previously-active foundation.

This matches the project-wide diagnostics philosophy described in
[`docs/DIAGNOSTICS.md`](./DIAGNOSTICS.md): bad forms produce structured
errors, not panics, and never break unrelated downstream queries.

## 4. `(foundation-report)`

`(foundation-report)` is a top-level form that emits a structured
snapshot of the active foundation. The snapshot has the shape:

```text
{
  activeFoundation: "<name>",
  description:      "<text> | null",
  numericDomain:    "<name> | null",
  truthDomain:      "<name> | null",
  rootConstructs:   [ { name, kind, status, dependsOn, encodedAs, ... } ],
  byStatus:         { "<status>": ["<name>", ...] },
  foundations:      [ { name, description, defines, ... } ],
  activeImplementations: [
    { construct, foundation, implementation, status, dependsOn }
  ],
  proofRules:       [ { name, premises, conclusion } ],
  proofAssumptions: [ { name, kind, judgement } ],
  proofObjects:     [ { name, rule, premises, premiseRefs, conclusion } ],
}
```

The CLI prints the report using `formatFoundationReport`. The printed
layout is byte-identical between the JavaScript and Rust hosts so that
parity tests can compare them line-for-line. A typical output:

```text
Foundation report:
  active foundation: default-rml
  description: Default RML foundation: host-implemented configurable kernel
  numeric domain: decimal-12
  truth domain: default-truth

host-primitive:
  - +
  - -
  - *
  - /
  - <
  - <=
  - =
  - !=
  - Pi
  - Prop
  - Type
  - apply
  - avg
  - beta-reduction
  - canonical-printer
  - lambda
  - max
  - min
  - product
  - probabilistic_sum
  - ...

user-configurable:
  - and
  - both
  - false
  - neither
  - not
  - or
  - true
  - truth-range
  - undefined
  - unknown
  - valence

external-trusted:
  - atp-trusted
  - lino-parser
  - smt-trusted

foundations:
  - boolean-links — links-defined two-valued Boolean logic via finite truth tables
      numeric domain: boolean-zero-one
      truth domain: boolean-two-valued
      truth tables: and(4 rows), not(2 rows), or(4 rows)
  - boolean-classical — two-valued-classical-boolean-logic
      numeric domain: boolean-zero-one
      truth domain: two-valued
      defines: and=min, or=max, both=min, neither=product
  - default-rml — Default RML foundation: host-implemented configurable kernel
      numeric domain: decimal-12
      truth domain: default-truth
  - kleene-three-valued — strong-kleene-three-valued-logic
      numeric domain: real-unit-interval
      truth domain: three-valued
      defines: and=min, or=max, both=min, neither=product
```

The structured form is also useful in programs:

```js
import { Env } from 'relative-meta-logic';

const env = new Env();
env.enterFoundation('boolean-classical');
try {
  const report = env.foundationReport();
  console.log(report.activeFoundation); // 'boolean-classical'
} finally {
  env.exitFoundation();
}
```

```rust
use rml::{Env, format_foundation_report};

let mut env = Env::new(None);
env.enter_foundation("boolean-classical").unwrap();
let report = env.foundation_report();
assert_eq!(report.active_foundation, "boolean-classical");
println!("{}", format_foundation_report(&report));
env.exit_foundation();
```

## 5. Bundled foundations

Three alternative foundations ship in [`lib/self/foundations.lino`](../lib/self/foundations.lino)
and are pre-seeded by the JS and Rust hosts:

- **`boolean-links`** — two-valued Boolean logic over the strict carrier
  `{0,1}`. `and`, `or`, and `not` are selected from finite truth-table
  rows, so the active implementation descriptors for those operators
  are `links-defined` while the foundation is active.
- **`boolean-classical`** — two-valued classical Boolean logic. `and`
  becomes `min`, `or` becomes `max`, `both` collapses to `min`, and
  `neither` to `product`. The numeric domain field is set to
  `boolean-zero-one` so the trust report records the intended carrier.
  These bindings still use host aggregator implementations.
- **`kleene-three-valued`** — Strong Kleene three-valued logic.
  Same `min`/`max` operator shape as classical, but the truth domain is
  `three-valued` and the numeric domain is the real unit interval; the
  midpoint `0.5` represents `unknown`. These bindings also use host
  aggregators.

The example [`examples/foundation-boolean-kleene.lino`](../examples/foundation-boolean-kleene.lino)
exercises both in one file and shows the default semantics being
restored after each scope. A minimal single-operator showcase lives in
[`examples/foundation-with-min.lino`](../examples/foundation-with-min.lino).

## 6. Carrier enforcement: `(carrier ...)` and `(strict-carrier)`

A foundation can declare which values it considers legal with
`(carrier ...)`. By itself this is informational — the trust audit
surfaces the carrier, but evaluation is not changed. Adding
`(strict-carrier)` opts the foundation into runtime enforcement: a
query result or probability assignment that falls outside the carrier
emits an `E063` diagnostic rather than being silently clamped or
returned.

```lino
(foundation bool-strict
  (description "two-valued classical, enforced")
  (carrier true false)
  (strict-carrier)
  (defines and min)
  (defines or max))

(a: a is a)
((a = true) has probability 0.5)

(with-foundation bool-strict
  ; E063: 0.5 is not in {true=1.0, false=0.0}
  (? (a = true)))
```

Symbolic carrier members (`true`, `false`, `unknown`) resolve through
`env.symbol_prob` at activation time, so the same foundation works
whether `true`/`false` are bound to `{0,1}` or `{0.0, 1.0}` or `{0, 0.5,
1}`. Numeric literals stay literal.

## 7. Truth tables: `(truth-table ...)`

Operators can also be rebound to a finite truth table written in
`.lino`. This is the smallest implemented path to a `links-defined`
operator: the rows are links data, and matching rows do not consult a
host aggregator while the foundation is active. The evaluator still
performs the table lookup, so this is not a self-hosted proof kernel.

```lino
(foundation xor3
  (description "three-valued exclusive-or")
  (carrier true false unknown)
  (truth-table xor
    (true  false   -> true)
    (false true    -> true)
    (true  true    -> false)
    (false false   -> false)
    (unknown _     -> unknown)
    (_       unknown -> unknown)))

(with-foundation xor3
  (? (a xor b)))
```

Partial tables are allowed: rows that don't match fall through to the
previously installed operator, which may be host-backed. The symbolic
constants are resolved through `env.symbol_prob` on activation, just
like `(carrier ...)`.

## 8. Pure-links strict mode

The `(strict-foundation pure-links)` form (paired with optional
`(allow-host-primitive ...)` whitelists) is the strict checking mode
described in the original roadmap. While active, any query that
transitively depends on a `host-primitive` or `host-derived` construct
not on the whitelist raises `E065`. The dependency graph in the trust
audit drives the check. The scanner reports concrete paths through the
active implementation map and the root-construct `depends-on` graph,
for example `and -> avg -> host-primitive`.

```lino
(strict-foundation pure-links)

(a: a is a)
(b: b is b)
((a = true) has probability 1)
((b = true) has probability 0)

; E065: default `and` depends on the host `avg` aggregator:
;       and -> avg -> host-primitive
(? ((a = true) and (b = true)))

; OK: the active implementation of `and` is a truth-table row set
; recorded as links-defined.
(with-foundation boolean-links
  (? ((a = true) and (b = true))))
```

`(allow-host-primitive <name>...)` whitelists exact construct or
dependency names. For example, allowing `avg` permits the default
`and -> avg -> host-primitive` path, while allowing an unrelated
ancestor does not.

The strict mode is *opt-in*; nothing about its presence in the host
changes the behaviour of a file that does not use it. The dependency
graph is also rendered in `formatFoundationReport` so the user can see,
before flipping the strict switch, exactly which constructs would need
to be whitelisted.

## 9. Experimental profiles: `mtc-anum`

A pre-seeded experimental foundation `mtc-anum` ships with both hosts.
It is **never** activated implicitly — `default-rml` stays the active
foundation on every fresh `Env`. The profile carries an `[experimental]`
tag, a root symbol `∞`, and four "abits" (atomic bits): `[`, `]`, `0`,
`1`. Together those four characters form a self-contained serialization
alphabet for arbitrary `Node` values. It is descriptive metadata plus
encode/decode helpers, not a replacement evaluator, proof checker, or
minimal trusted kernel.

```text
mtc-anum [experimental] — minimal-trust-core experimental anum profile
    root: ∞
    abits: [=start-of-meaning, ]=end-of-meaning, 0=leaf-tag, 1=list-tag
```

The companion helpers `encodeAnum` / `decodeAnum` (JS) and
`encode_anum` / `decode_anum` (Rust) round-trip `Node` values through
that four-character alphabet:

```js
import { encodeAnum, decodeAnum, parseLino } from 'relative-meta-logic';

const node = parseLino('(? (1 + 2))')[0];
const wire = encodeAnum(node);
// wire is a string drawn only from [ ] 0 1
const back = decodeAnum(wire);
// back deepStrictEqual node
```

Encoding rules:

- Leaf: `[0` + UTF-8 bytes as MSB-first 8-bit groups + `]`
- List: `[1` + concatenation of child encodings + `]`

Decoding rejects characters outside the four-abit alphabet, unbalanced
frames, and leaf payloads that are not byte-aligned, all with `E066`.

## 10. Status of the broader programme

The issue #97 roadmap is implemented where the work could stay
backward-compatible and inspectable. The remaining self-hosting target
is still deliberately explicit: Phase 5 has descriptor and dependency
coverage, but not a links-defined type/proof kernel.

- **Phase 1 — inventory + reporting + scoped overrides.** Implemented.
  See §2 (registry), §4 (`foundation-report`), §3 (`(with-foundation ...)`).
- **Phase 2 — equality and numeric-domain separation.** Implemented as
  registry descriptors: `structural-equality`, `numeric-equality`,
  `assigned-equality`, `definitional-equality` are distinct entries
  with their own `depends-on`. Proof-trace layering remains an open
  follow-up.
- **Phase 3 — proof-object substrate.** Implemented via proof rules,
  `(assumption ...)` / `(axiom ...)`, `(proof-object ...)`, and
  `(check-proof ...)`. Premises must cite an assumption, axiom, or
  earlier proof object using `(premise-by ...)` / `(uses ...)`;
  unjustified raw premises raise `E064`.
- **Phase 4 — links-defined finite logics.** Implemented. See the
  bundled `boolean-links` truth-table foundation (§5), the
  `(truth-table ...)` clause (§7), and `(carrier ...)` +
  `(strict-carrier)` (§6). The older `boolean-classical` /
  `kleene-three-valued` foundations remain host-aggregator examples.
- **Phase 5 — links-defined type/proof kernel fragment.** Partial:
  the registry's `Prop`, `Pi`, `lambda`, `apply`, `beta-reduction`, and
  equality layers are catalogued with `depends-on` links, and the
  trust audit can walk them via the dependency graph. The actual
  type/proof kernel remains host-implemented.
- **Phase 6 — pure-links checking mode.** Implemented. See §8.
- **Phase 7 — dependency-graph traversal.** Implemented for trust
  reporting and strict-mode enforcement paths.
- **Phase 8 — bundled `(carrier ...)` / `(strict-carrier)` /
  `(truth-table ...)`.** Implemented (§6, §7).
- **Phase 9 — experimental profiles (`mtc-anum`).** Implemented as an
  opt-in serialization profile (§9), with `encodeAnum` / `decodeAnum`
  helpers and the `(experimental)` / `(root ...)` / `(abit ...)`
  clauses. It does not implement a minimal trusted core evaluator.

Everything in this document is the *backward-compatible* surface. The
strict modes (`(strict-carrier)`, `(strict-foundation pure-links)`) are
opt-in; nothing about their existence changes the behaviour of files
that do not use them.

## 11. Relationship to other documents

- [`CONFIGURABILITY.md`](./CONFIGURABILITY.md) — what is reconfigurable
  *inside* a single foundation (range, valence, aggregator selection,
  operator composition).
- [`SOUNDNESS.md`](./SOUNDNESS.md) — the trusted kernel guarantee. The
  status fields on the registry are the granular version of the
  "trusted base" listed there.
- [`KERNEL.md`](./KERNEL.md) — typed kernel rules. The registry's
  `Type`, `Prop`, `Pi`, `lambda`, `apply`, `beta-reduction`,
  `substitution`, and `freshness` entries correspond one-to-one to the
  rules in that file.
- [`DIAGNOSTICS.md`](./DIAGNOSTICS.md) — full error-code table,
  including `E060`–`E066` for foundation forms, carrier enforcement,
  proof-object replay, pure-links strict mode, and MTC/anum
  encode/decode.
- [`tutorials/self-bootstrap.md`](./tutorials/self-bootstrap.md) — the
  capstone walkthrough; foundations slot in beside the encoded grammar,
  evaluator, types, operators, and metatheorem checker.
- [`case-studies/issue-97/`](./case-studies/issue-97/) — the deep
  case-study analysis behind this surface, including the requirements
  extracted from issue #97, the root-cause analysis, the test plan, and
  links to the captured GitHub data.
