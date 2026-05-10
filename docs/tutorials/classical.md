# Classical Logic Tutorial

Classical logic is the first RML tutorial because it uses the smallest truth
space: every proposition is either false (`0`) or true (`1`). The same query
surface will be reused by the fuzzy, probabilistic, typed, and metatheory
tutorials.

Run the full example:

```bash
node js/src/rml-links.mjs examples/classical-logic.lino
```

The source is [`../../examples/classical-logic.lino`](../../examples/classical-logic.lino).

## 1. Select Boolean Semantics

The example starts by selecting two-valued truth and familiar Boolean
connectives:

```lino
(valence: 2)
(and: min)
(or: max)
```

`(valence: 2)` quantizes truth values to `0` and `1`. With that choice:

- `and` as `min` means a conjunction is true only when both sides are true.
- `or` as `max` means a disjunction is true when either side is true.
- `not` is available by default and flips truth across the current range.

These settings make the rest of the file behave like ordinary Boolean logic.

## 2. Declare Propositions

RML source is made of links. A simple proposition is declared with the same
shape used throughout the repository:

```lino
(p: p is p)
(q: q is q)
```

This records `p` and `q` as terms that can be used in later expressions. The
declaration does not say whether either proposition is true; it only introduces
the names.

## 3. Assign Truth Values

Truth assignments attach a probability to an expression:

```lino
((p = true) has probability 1)
((q = true) has probability 0)
```

In Boolean mode, these are ordinary truth facts:

- `p` is true.
- `q` is false.

The word `probability` is still used because the same evaluator also supports
continuous and probabilistic values in later tutorials.

## 4. Ask Queries

A query is written with `?`:

```lino
(? (p = true))
(? (q = true))
```

The evaluator prints one result per query. For this example, those two lines
produce `1` and `0`.

The same syntax works for compound expressions:

```lino
(? ((p = true) and (q = true)))
(? ((p = true) or (q = true)))
```

Because `p` is true and `q` is false, the conjunction evaluates to `0` and the
disjunction evaluates to `1`.

## 5. Check Classical Laws

The example closes with three standard laws:

```lino
(? ((p = true) or (not (p = true))))
(? ((p = true) and (not (p = true))))
(? (not (not (p = true))))
```

Read these as:

- Excluded middle: `p` or not `p`.
- Non-contradiction: not both `p` and not `p`.
- Double negation: not not `p` is equivalent to `p`.

With `valence: 2`, these behave like the classical laws they model.

## 6. Reuse the Classical Library

After the surface syntax is clear, read
[`../../lib/classical/core.lino`](../../lib/classical/core.lino). It installs the
same Boolean settings and exports named templates under the `classical`
namespace:

```lino
(import "lib/classical/core.lino" as cl)

(p: p is p)
((p = true) has probability 1)

(? (cl.excluded-middle (p = true)))
(? (cl.non-contradiction (p = true)))
```

Use the library when you want named law schemas instead of spelling each
connective by hand.

## What To Remember

Classical RML programs have three moving parts:

1. Configure the truth space with `(valence: 2)`.
2. Declare terms and assign truth values.
3. Query expressions with `(? ...)`.

The next tutorial removes the Boolean restriction and lets truth values live
anywhere in `[0, 1]`.
