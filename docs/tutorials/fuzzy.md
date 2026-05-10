# Fuzzy Logic Tutorial

Fuzzy logic keeps the same declaration, assignment, and query forms from the
classical tutorial, but it allows intermediate truth values. A proposition can
be true to degree `0.8`, false to degree `0.2`, or anywhere between.

Run the full example:

```bash
node js/src/rml-links.mjs examples/fuzzy-logic.lino
```

The source is [`../../examples/fuzzy-logic.lino`](../../examples/fuzzy-logic.lino).

## 1. Keep Continuous Truth Values

The fuzzy example does not set `(valence: 2)`. RML's default valence is
continuous, so values are not quantized to `0` or `1`.

It does choose the standard Zadeh-style connectives:

```lino
(and: min)
(or: max)
```

With these settings:

- `and` returns the lower degree.
- `or` returns the higher degree.
- `not` maps a degree `x` to `1 - x`.

## 2. Assign Degrees

The example declares three subjects and assigns a degree to the predicate
`tall`:

```lino
(a: a is a)
(b: b is b)
(c: c is c)

((a = tall) has probability 0.8)
((b = tall) has probability 0.3)
((c = tall) has probability 0.6)
```

These assignments are better read as membership degrees:

- `a` is tall to degree `0.8`.
- `b` is tall to degree `0.3`.
- `c` is tall to degree `0.6`.

The syntax still says `has probability` because RML uses one numeric truth
channel for Boolean, fuzzy, probabilistic, and many-valued logic.

## 3. Query Membership

Direct queries return the assigned degrees:

```lino
(? (a = tall))
(? (b = tall))
```

The first prints `0.8`; the second prints `0.3`.

## 4. Combine Fuzzy Predicates

Fuzzy conjunction takes the smaller degree:

```lino
(? ((a = tall) and (b = tall)))
```

The result is `0.3`, because the weaker side limits the conjunction.

Fuzzy disjunction takes the larger degree:

```lino
(? ((a = tall) or (b = tall)))
```

The result is `0.8`.

Negation complements the degree:

```lino
(? (not (a = tall)))
```

The result is `0.2`.

## 5. Read Nested Queries Inside Out

The last query combines all three subjects:

```lino
(? ((a = tall) and ((b = tall) or (c = tall))))
```

Read it inside out:

1. `(b = tall) or (c = tall)` is `max(0.3, 0.6)`, so it is `0.6`.
2. `(a = tall) and 0.6` is `min(0.8, 0.6)`, so it is `0.6`.

This is the same query shape as classical logic, but the intermediate values
are now meaningful rather than rounded away.

## 6. Reuse the Fuzzy Library

The reusable fuzzy helpers live in
[`../../lib/probabilistic/fuzzy.lino`](../../lib/probabilistic/fuzzy.lino). They
provide named templates for membership, rule activation, and a small centroid
calculation:

```lino
(import "lib/probabilistic/fuzzy.lino" as fz)

(fz.membership temperature hot 0.8)
(fz.membership humidity wet 0.6)

(? (fz.rule (fz.degree temperature hot) (fz.degree humidity wet)))
(? (fz.two-point-centroid 0.6 0.8 0.4 0.3))
```

Use this library when the source reads more naturally as fuzzy-control data
than as raw equality assignments.

## What To Remember

Fuzzy RML keeps the classical workflow but changes the meaning of truth values:

1. Leave valence continuous.
2. Assign degrees between `0` and `1`.
3. Choose aggregators such as `min` and `max`.
4. Read compound query results as degrees, not just pass/fail answers.

The next tutorial keeps continuous values and changes the aggregators to model
probabilistic events.
