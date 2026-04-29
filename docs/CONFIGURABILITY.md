# Configurability and Operator Redefinition

This document explains the design rationale behind one of the most surprising
properties of Relative Meta-Logic (RML): **every operator, truth constant,
range, and valence is redefinable at runtime, by ordinary `.lino` source.**
There is no privileged "kernel" of fixed semantics. A user who reads only
classical logic textbooks may file bugs about behaviour that is by design;
this page is the canonical place to point them at.

> Short version: RML is a meta-logic. Its job is to host *other* logic
> systems, not to enforce a particular one. So the connectives `and`, `or`,
> `not`, the constants `true`, `false`, `unknown`, the range of truth values,
> the valence (number of truth values), and the aggregator used by a Belnap
> operator are all configuration knobs, not laws.

If you came here looking for **what** is configurable rather than **why**,
jump straight to:

- [Range configuration](#1-range-configuration-range-lo-hi)
- [Valence configuration](#2-valence-configuration-valence-n)
- [Operator redefinition (composition)](#3-operator-redefinition-by-composition)
- [Aggregator selection](#4-aggregator-selection-for-and-or-both-neither)
- [Truth-constant redefinition](#5-truth-constant-redefinition-true-false-unknown-undefined)
- [Precedence at runtime](#precedence-and-runtime-semantics)

## Why everything is redefinable

Traditional proof assistants like
[Lean 4](https://lean-lang.org/),
[Rocq (Coq)](https://rocq-prover.org/),
[Isabelle](https://isabelle.in.tum.de/),
[Twelf](http://twelf.org/), and the LF family fix the meaning of their
logical connectives in the kernel. The kernel of Lean knows what `And` and
`Or` are; the kernel of Rocq knows the rules of the Calculus of Inductive
Constructions; Isabelle/Pure fixes meta-implication and meta-quantification.
Object logics are then encoded *on top* of those fixed primitives.

RML reverses this priority. Its design goals demand:

1. **Many-valued by default.** RML supports unary, Boolean, ternary
   ([Kleene](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)),
   N-valued, [Belnap four-valued](https://en.wikipedia.org/wiki/Four-valued_logic#Belnap),
   and continuous [fuzzy](https://en.wikipedia.org/wiki/Fuzzy_logic) /
   [Łukasiewicz ∞-valued](https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic) logics.
   No single fixed `and`/`or` works across all of these; min, max, average,
   product, and probabilistic-sum are all the right answer somewhere.
2. **Probabilistic and Bayesian use cases.** When `and` means
   `P(A ∩ B) = P(A) · P(B)` and `or` means
   `P(A ∪ B) = 1 - (1-P(A))·(1-P(B))`, the user must be able to *say so* in
   the source.
3. **Paradox tolerance.** RML resolves the
   [liar paradox](https://en.wikipedia.org/wiki/Liar_paradox) to the
   midpoint of the truth range rather than rejecting it. That requires
   midpoints, ranges, and Belnap "both/neither" operators to be first-class
   and tunable.
4. **Self-reasoning.** A meta-logic that wants to encode and reason about
   *other* logic systems must be able to *become* those systems by
   reconfiguration, not only model them indirectly.

Locking semantics into a kernel would defeat goals (1)–(4). So RML treats
every operator and constant as data. The same `.lino` evaluator is
classical Boolean logic, fuzzy logic, Bayesian inference, or Belnap's
four-valued logic depending on what the source file says.

This is an [axiomatic-system constructor](#related-systems), not a fixed
logic.

## How RML differs from Lean and Rocq

| Aspect | Lean / Rocq (CIC) | RML |
|--------|-------------------|-----|
| Where `and`, `or`, `not` live | In the kernel / Prelude; semantics fixed by inductive definition or kernel rules | In a runtime operator table; redefinable from source |
| Truth values | `Prop` (proof-relevant), `Bool` (two-valued); other valences encoded as data | Numeric truth value drawn from a configurable range; valence configured per file |
| Range of truth values | N/A — propositions are not numeric | `(range: 0 1)` (default) or `(range: -1 1)` (balanced) — configurable |
| Truth constants | `True`, `False` are inductive types with fixed semantics | `true`, `false`, `unknown`, `undefined` are entries in a symbol table; redefinable |
| `Type : Type` | Forbidden ([Russell's paradox](https://en.wikipedia.org/wiki/Russell%27s_paradox)); stratified universes only | Allowed via `(Type: Type Type)`; paradoxes resolve to the midpoint |
| Adding a new connective | Define a function, prove its theorems | Write `(myop: <unary> <binary>)` or `(and: product)`; takes effect from the next form |
| Aggregator choice for `and`/`or` | Fixed by definition | Run-time choice: `avg`, `min`, `max`, `product`, `probabilistic_sum` |
| Belnap `both`/`neither` | Not built-in; encode if needed | Built-in composite operators with redefinable aggregators |

In Lean and Rocq, **the user adapts to the kernel**. In RML, **the kernel
adapts to the user**. The trade-off is real: RML deliberately gives up the
kind of soundness guarantee that comes from a small fixed kernel, in
exchange for the ability to reconfigure into whichever logic system the
problem demands.

For the longer feature-by-feature comparison, see
[CONCEPTS-COMPARISION.md](./CONCEPTS-COMPARISION.md) and
[FEATURE-COMPARISION.md](./FEATURE-COMPARISION.md).

## What is configurable

### 1. Range configuration: `(range: lo hi)`

The truth value range is a closed interval of real numbers. The default is
`[0, 1]` (standard probabilistic). Use `(range: -1 1)` for the balanced /
symmetric range, where the midpoint is `0` — convenient for
[balanced ternary](https://en.wikipedia.org/wiki/Balanced_ternary) and for
expressing positive vs. negative evidence symmetrically.

```lino
(range: -1 1)
(? true)             # -> 1   (max of range)
(? false)            # -> -1  (min of range)
(? unknown)          # -> 0   (mid of range)
(? (not 0.5))        # -> -0.5
```

Changing the range automatically re-derives the default truth constants
(`true`, `false`, `unknown`, `undefined`), so a `(range: ...)` form is in
effect a **bulk redefinition** of all constants whose defaults depend on
`min`, `max`, and `mid` of the range.

Examples that exercise this:
- [`examples/liar-paradox-balanced.lino`](../examples/liar-paradox-balanced.lino) — paradox resolution in `[-1, 1]`.

### 2. Valence configuration: `(valence: N)`

Valence is the *number* of admissible truth values. `N=0` (the default)
means continuous — any real in the range is allowed. `N=2` means Boolean
(`{0, 1}` in `[0,1]`, or `{-1, 1}` in `[-1,1]`). `N=3` is Kleene's
strong three-valued logic. Higher `N` partitions the range into `N`
evenly spaced levels; raw values are quantized to the nearest level.

```lino
(valence: 2)         # Boolean
(valence: 3)         # ternary (Kleene)
(valence: 7)         # 7-valued Łukasiewicz fragment
(valence: 0)         # continuous (default)
```

Examples:
- [`examples/classical-logic.lino`](../examples/classical-logic.lino) — `(valence: 2)`.
- [`examples/ternary-kleene.lino`](../examples/ternary-kleene.lino) — `(valence: 3)`.
- [`examples/fuzzy-logic.lino`](../examples/fuzzy-logic.lino) — continuous (default valence).

### 3. Operator redefinition by composition

Operators are entries in a runtime operator table. The composition form
defines a new operator as the composition of two existing ones:

```lino
(<op>: <unary_op> <binary_op>)
```

The canonical example is the default definition of inequality, written
explicitly in `examples/demo.lino`:

```lino
(!=: not =)          # != is "not =" — the negation of equality
```

Any operator name can be redefined this way: `=`, `!=`, `and`, `or`, `not`,
`is`, `?:`, `both`, `neither`, plus user-coined symbols containing `=` or
`!`. The composition takes effect on the next evaluated form.

Examples:
- [`examples/demo.lino`](../examples/demo.lino) — `(!=: not =)`.

### 4. Aggregator selection for `and`, `or`, `both`, `neither`

For the connectives that combine truth values, you do not pick a logical
formula — you pick the **aggregator**, the function used to fold a list of
truth values into one. RML ships five:

| Aggregator | Formula | Used by |
|------------|---------|---------|
| `avg` | `(x1 + x2 + ... + xn) / n` | Default `and`; default `both` |
| `min` | `min(x1, ..., xn)` | [Kleene/Łukasiewicz](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics)/[Zadeh](https://en.wikipedia.org/wiki/Fuzzy_logic) `and` |
| `max` | `max(x1, ..., xn)` | Default `or`; Zadeh / Kleene `or` |
| `product` (`prod`) | `x1 · x2 · ... · xn` | [Bayesian joint probability](https://en.wikipedia.org/wiki/Bayesian_network); default `neither` |
| `probabilistic_sum` (`ps`) | `1 - (1-x1)(1-x2)...(1-xn)` | Independent-event union ([inclusion-exclusion](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle)) |

Selecting an aggregator is a one-liner:

```lino
(and: min)               # Kleene / Łukasiewicz / Zadeh AND
(or:  max)               # Kleene / Łukasiewicz / Zadeh OR
(and: product)           # Bayesian joint
(or:  probabilistic_sum) # Bayesian union
(both: min)              # tighten Belnap "both" to the stricter side
(neither: max)           # loosen Belnap "neither" to the more permissive side
```

This is what makes RML a meta-logic in practice: a single `.lino` file is
classical Boolean, fuzzy, Bayesian, or Belnap depending on which two lines
appear at the top.

Examples that exercise different aggregator choices:
- [`examples/classical-logic.lino`](../examples/classical-logic.lino) — `(and: min)`, `(or: max)`.
- [`examples/fuzzy-logic.lino`](../examples/fuzzy-logic.lino) — `(and: min)`, `(or: max)` over continuous values.
- [`examples/propositional-logic.lino`](../examples/propositional-logic.lino) — `(and: product)`, `(or: probabilistic_sum)`.
- [`examples/bayesian-network.lino`](../examples/bayesian-network.lino) — `product` and `probabilistic_sum` for a DAG.
- [`examples/belnap-four-valued.lino`](../examples/belnap-four-valued.lino) — default `both`/`neither` aggregators.
- [`examples/demo.lino`](../examples/demo.lino) — non-standard `(and: avg)`.

### 5. Truth-constant redefinition: `true`, `false`, `unknown`, `undefined`

Truth constants are *not* keywords. They are entries in a symbol-probability
table whose defaults are derived from the current range:

| Constant    | Default in `[0, 1]` | Default in `[-1, 1]` | Definition   |
|-------------|---------------------|----------------------|--------------|
| `true`      | `1`                 | `1`                  | `max(range)` |
| `false`     | `0`                 | `-1`                 | `min(range)` |
| `unknown`   | `0.5`               | `0`                  | `mid(range)` |
| `undefined` | `0.5`               | `0`                  | `mid(range)` |

Any of them can be redefined by ordinary symbol-probability syntax:

```lino
(true: 0.8)          # Redefine true to 0.8 — useful when modelling
(false: 0.2)         # high-confidence-but-not-perfect oracles.
(? true)             # -> 0.8
(? false)             # -> 0.2
```

A subsequent `(range: ...)` form re-initializes the constants to the new
range's defaults, undoing previous redefinitions. This is intentional:
changing the range typically means starting over with a different logic.

Examples that touch truth constants:
- [`examples/classical-logic.lino`](../examples/classical-logic.lino) — `true`, `false` at their defaults.
- [`examples/belnap-four-valued.lino`](../examples/belnap-four-valued.lino) — uses `true` / `false` plus the `both` / `neither` derived "values".

## Precedence and runtime semantics

Configuration in RML is **textual order, last writer wins, scope-global**.
There is no separate "configuration phase" or import system. Every form
runs sequentially and updates the same environment.

The rules are:

1. **Forms are evaluated top-to-bottom.** A `(range: 0 1)` on line 5
   affects every form on line 6 and below; nothing above it.
2. **Redefinitions take effect immediately and persist.** Once you write
   `(and: product)`, every subsequent occurrence of `and` — including
   inside queries, `both`/`neither` composites, and lambda bodies — uses
   `product`, until and unless another `(and: ...)` form replaces it.
3. **`(range: ...)` re-initializes the operator table and truth
   constants** to defaults for the new range. Range changes are therefore
   "heavy" — treat them as a logic-mode switch, not a tweak. Place
   `(range: ...)` once, near the top of a file, before other configuration.
4. **Valence is just quantization.** `(valence: N)` does not change which
   operator is bound to `and`. It changes how *result* values are clamped
   on the way out. `(and: avg)` followed by `(valence: 2)` gives Boolean
   results computed from an averaging aggregator — typically what you want
   when modelling "majority vote" semantics.
5. **Truth-constant redefinitions are symbol assignments.** `(true: 0.8)`
   is exactly the same kind of form as `(p: 0.8)` for an arbitrary symbol
   `p`. They share one symbol table.
6. **Equality is special.** `=` first checks explicit
   `((expr) has probability v)` assignments, then structural equality of
   the two sides, then numeric comparison (with decimal-precision
   rounding) of the evaluated values. You can redefine `=` by composition,
   but most users keep the default and override `!=` instead.
7. **Operator redefinitions do not retroactively change earlier
   evaluations.** If line 10 evaluates `(? (a and b))` with the default
   aggregator, and line 11 changes `(and: product)`, the line-10 result
   is already printed and stays as it was.

In practice, the idiomatic structure of a `.lino` file is:

```lino
# 1. Range and valence (logic mode)
(range: 0 1)
(valence: 2)

# 2. Operator and aggregator choice
(and: min)
(or:  max)
(!=: not =)

# 3. Optional truth-constant overrides (rare)
# (true: 0.95)

# 4. Term definitions
(p: p is p)
(q: q is q)

# 5. Probability assignments / axioms
((p = true) has probability 1)
((q = true) has probability 0)

# 6. Queries
(? ((p = true) and (q = true)))
```

This ordering is a convention, not a rule. The evaluator will accept any
order; the convention exists so that human readers can scan a file and
know what logic system they are reading.

## Why this is not a bug

If you expect classical fixed semantics, the following will all surprise
you:

- `(? (true and false))` returning `0.5` (not `0`). The default `and` is
  `avg`, not `min`. Add `(and: min)` for classical behaviour, or pick
  whichever aggregator matches your intended logic.
- The law of excluded middle `(? (A or (not A)))` failing in
  `(valence: 3)`. That is Kleene logic, not classical logic. Use
  `(valence: 2)` for the classical answer.
- `(Type: Type Type)` not raising
  [Russell's paradox](https://en.wikipedia.org/wiki/Russell%27s_paradox).
  RML resolves it to the midpoint of the truth range. See
  [README.md § Paradox Resolution in the Type System](../README.md#paradox-resolution-in-the-type-system).
- A `.lino` file behaving differently after another `.lino` file is
  prepended to it. RML configuration is global to the run.

These are deliberate consequences of RML's role as a **configurable
meta-logic** rather than a fixed object logic.

## Related systems

| System | What it fixes | What RML lets you change instead |
|--------|---------------|----------------------------------|
| [Lean 4](https://lean-lang.org/) | CIC kernel; `Prop` / `Type` universes; classical-by-`Classical.em` | Range, valence, all connectives; allow `Type : Type` |
| [Rocq (Coq)](https://rocq-prover.org/) | CIC kernel; stratified universes | Range, valence, all connectives; allow `Type : Type` |
| [Isabelle](https://isabelle.in.tum.de/) | `Pure` meta-logic; HOL on top | Both meta and object connectives, in one place |
| [Twelf / LF](http://twelf.org/) | LF kernel; logical framework with fixed types | Object logics by reconfiguration, not encoding |
| [lambda Prolog](https://www.lix.polytechnique.fr/Labo/Dale.Miller/lProlog/) | Hereditary Harrop logic | Range, valence, aggregators chosen per file |
| [Pecan](https://github.com/ReedOei/Pecan) | Buchi automata over fixed numeration systems | Many-valued/probabilistic semantics |

For the comprehensive matrix, see
[CONCEPTS-COMPARISION.md](./CONCEPTS-COMPARISION.md) and
[FEATURE-COMPARISION.md](./FEATURE-COMPARISION.md).

## See also

- [README.md](../README.md) — language tour with worked examples for each
  configurable knob.
- [ARCHITECTURE.md](../ARCHITECTURE.md) — how the JS and Rust evaluators
  implement the operator table and symbol-probability map.
- [DIAGNOSTICS.md](./DIAGNOSTICS.md) — error codes you may see when an
  unknown aggregator name or operator definition is used (`E003`, `E004`).
- [`examples/`](../examples/) — the canonical, language-agnostic corpus
  that demonstrates every configurable feature on real `.lino` files.
