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
| `E061` | Malformed `(foundation ...)` declaration (missing name, unknown child clause shape, malformed `(defines ...)` clause). |
| `E062` | `(with-foundation <name> ...)` references a foundation that has not been registered. The diagnostic does not abort evaluation — forms after the bad scope still run. |

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
| `(numeric-domain <name>)` | Name of the numeric domain this foundation expects (data only — see §6 for the planned strict-checking mode). |
| `(truth-domain <name>)` | Name of the truth domain (data only). |
| `(extends <name>)` | Inherits fields from a previously registered foundation. |
| `(defines <op> <aggregator>)` | Re-binds operator `<op>` to aggregator `<aggregator>` while the foundation is active. Repeats are allowed; each new `(defines ...)` replaces the binding for that operator. |

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

Two alternative foundations ship in [`lib/self/foundations.lino`](../lib/self/foundations.lino):

- **`boolean-classical`** — two-valued classical Boolean logic. `and`
  becomes `min`, `or` becomes `max`, `both` collapses to `min`, and
  `neither` to `product`. The numeric domain field is set to
  `boolean-zero-one` so the trust report records the intended carrier;
  numeric-domain enforcement is planned (§6).
- **`kleene-three-valued`** — Strong Kleene three-valued logic.
  Same `min`/`max` operator shape as classical, but the truth domain is
  `three-valued` and the numeric domain is the real unit interval; the
  midpoint `0.5` represents `unknown`.

The example [`examples/foundation-boolean-kleene.lino`](../examples/foundation-boolean-kleene.lino)
exercises both in one file and shows the default semantics being
restored after each scope. A minimal single-operator showcase lives in
[`examples/foundation-with-min.lino`](../examples/foundation-with-min.lino).

## 6. Status of the broader programme

This document captures Phase 1 (inventory + reporting + scoped overrides)
and the start of Phase 4 (links-defined finite logics — Boolean and
Kleene). The roadmap from the issue thread continues:

- **Phase 2 — equality and numeric-domain separation.** The registry
  already names `structural-equality`, `numeric-equality`,
  `assigned-equality`, and `definitional-equality` as four distinct
  layers (see [`lib/self/foundations.lino`](../lib/self/foundations.lino)),
  but the proof object does not yet record which layer fired for each
  step. Future work will surface that in trace/proof output.
- **Phase 3 — proof-object substrate.** Proof rules become first-class
  links; replay consults the declared rule set rather than only the
  host's classifier table.
- **Phase 5 — links-defined type/proof kernel fragment.** A small
  propositions-as-types fragment defined entirely in `.lino`, similar in
  spirit to [Software Foundations'](https://softwarefoundations.cis.upenn.edu/sf-3.2/Logic.html)
  `Logic.v`, with `Prop`, implication, conjunction, disjunction, equality.
- **Phase 6 — pure-links checking mode.** A strict mode that rejects
  proof steps depending on a construct whose status is still
  `host-primitive`. Useful for measuring how much of the kernel has
  actually moved into links.

Everything in this document is the *backward-compatible* surface. The
strict modes above are opt-in.

## 7. Relationship to other documents

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
  including `E060`/`E061`/`E062` for foundation forms.
- [`tutorials/self-bootstrap.md`](./tutorials/self-bootstrap.md) — the
  capstone walkthrough; foundations slot in beside the encoded grammar,
  evaluator, types, operators, and metatheorem checker.
- [`case-studies/issue-97/`](./case-studies/issue-97/) — the deep
  case-study analysis behind this surface, including the requirements
  extracted from issue #97, the root-cause analysis, the test plan, and
  links to the captured GitHub data.
