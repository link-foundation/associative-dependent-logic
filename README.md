# associative-dependent-logic

A prototype for logic framework that can reason about anything relative to given probability of input statements.

## Implementations

This project provides two equivalent implementations:

- **[JavaScript](./js/)** — Node.js implementation using the official [links-notation](https://github.com/link-foundation/links-notation) parser
- **[Rust](./rust/)** — Rust implementation using the official [links-notation](https://github.com/link-foundation/links-notation) crate

Both implementations pass the same 122 tests and produce identical results.

For implementation details, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Overview

ADL (Associative-Dependent Logic) is a minimal probabilistic logic system built on top of [LiNo (Links Notation)](https://github.com/link-foundation/links-notation). It supports [many-valued logics](https://en.wikipedia.org/wiki/Many-valued_logic) from unary (1-valued) through continuous probabilistic ([fuzzy](https://en.wikipedia.org/wiki/Fuzzy_logic)), allowing you to:

- Define terms
- Assign probabilities (truth values) to logical expressions
- Redefine logical operators with different semantics
- Configure truth value ranges: `[0, 1]` or `[-1, 1]` (balanced/symmetric)
- Configure logic valence: 2-valued ([Boolean](https://en.wikipedia.org/wiki/Boolean_algebra)), 3-valued ([ternary/Kleene](https://en.wikipedia.org/wiki/Three-valued_logic)), N-valued, or continuous
- Use and redefine truth constants: `true`, `false`, `unknown`, `undefined`
- Resolve paradoxical statements (e.g. the [liar paradox](https://en.wikipedia.org/wiki/Liar_paradox))
- Perform decimal-precision arithmetic (`+`, `-`, `*`, `/`) — `0.1 + 0.2 = 0.3`, not `0.30000000000000004`
- Query the truth value of complex expressions

## Supported Logic Types

| Valence | Name | Truth Values (in `[0, 1]`) | Truth Values (in `[-1, 1]`) | Reference |
|---------|------|---------------------------|----------------------------|-----------|
| 1 | Unary (trivial) | `{any}` (no quantization) | `{any}` (no quantization) | [Many-valued logic](https://en.wikipedia.org/wiki/Many-valued_logic) |
| 2 | Binary / [Boolean](https://en.wikipedia.org/wiki/Boolean_algebra) | `{0, 1}` (false, true) | `{-1, 1}` (false, true) | [Classical logic](https://en.wikipedia.org/wiki/Classical_logic) |
| 3 | Ternary / [Three-valued](https://en.wikipedia.org/wiki/Three-valued_logic) | `{0, 0.5, 1}` (false, unknown, true) | `{-1, 0, 1}` (false, unknown, true) | [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics), [Łukasiewicz logic](https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic), [Balanced ternary](https://en.wikipedia.org/wiki/Balanced_ternary) |
| 4 | Quaternary | `{0, ⅓, ⅔, 1}` | `{-1, -⅓, ⅓, 1}` | [Belnap's four-valued logic](https://en.wikipedia.org/wiki/Many-valued_logic) |
| 5 | Quinary | `{0, 0.25, 0.5, 0.75, 1}` | `{-1, -0.5, 0, 0.5, 1}` | [Many-valued logic](https://en.wikipedia.org/wiki/Many-valued_logic) |
| N | N-valued | N evenly-spaced levels | N evenly-spaced levels | [Many-valued logic](https://en.wikipedia.org/wiki/Many-valued_logic) |
| 0/∞ | Continuous / [Fuzzy](https://en.wikipedia.org/wiki/Fuzzy_logic) | Any value in `[0, 1]` | Any value in `[-1, 1]` | [Fuzzy logic](https://en.wikipedia.org/wiki/Fuzzy_logic), [Łukasiewicz ∞-valued](https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic) |

## Quick Start

### JavaScript

```bash
cd js
npm install
node src/adl-links.mjs demo.lino
```

### Rust

```bash
cd rust
cargo run -- demo.lino
```

### Example

Create a file `example.lino`:

```lino
# Define a term
(a: a is a)

# Set up operators
(!=: not =)
(and: avg)
(or: max)

# Assign probabilities to axioms
((a = a) has probability 1)
((a != a) has probability 0)

# Query probabilities
(? ((a = a) and (a != a)))   # -> 0.5
(? ((a = a) or  (a != a)))   # -> 1
```

Output:
```
0.5
1
```

## Syntax

### Term Definitions

```lino
(term_name: term_name is term_name)
```

Example: `(a: a is a)` declares `a` as a term.

### Probability Assignments

```lino
((<expression>) has probability <value>)
```

Example: `((a = a) has probability 1)` assigns probability 1 to the expression `a = a`.

### Range Configuration

```lino
(range: <lo> <hi>)
```

Sets the truth value range. Default is `[0, 1]` (standard probabilistic). Use `(range: -1 1)` for balanced/symmetric range where the midpoint is 0.

See: [Balanced ternary](https://en.wikipedia.org/wiki/Balanced_ternary)

### Valence Configuration

```lino
(valence: <N>)
```

Sets the number of discrete truth values. Default is `0` (continuous, no quantization).

- `(valence: 2)` — [Boolean logic](https://en.wikipedia.org/wiki/Boolean_algebra): truth values are quantized to `{0, 1}`
- `(valence: 3)` — [Ternary logic](https://en.wikipedia.org/wiki/Three-valued_logic): truth values are quantized to `{0, 0.5, 1}`
- `(valence: N)` — [N-valued logic](https://en.wikipedia.org/wiki/Many-valued_logic): truth values are quantized to N evenly-spaced levels

### Truth Constants

The symbols `true`, `false`, `unknown`, and `undefined` are predefined with values based on the current range:

| Constant | Default in `[0, 1]` | Default in `[-1, 1]` | Definition |
|----------|---------------------|----------------------|------------|
| `true`   | `1`                 | `1`                  | `max(range)` |
| `false`  | `0`                 | `-1`                 | `min(range)` |
| `unknown` | `0.5`              | `0`                  | `(max(range) - min(range)) / 2 + min(range)` |
| `undefined` | `0.5`            | `0`                  | `(max(range) - min(range)) / 2 + min(range)` |

These constants can be used directly in expressions:

```lino
(? true)              # -> 1 in [0,1], 1 in [-1,1]
(? false)             # -> 0 in [0,1], -1 in [-1,1]
(? unknown)           # -> 0.5 in [0,1], 0 in [-1,1]
(? (not true))        # -> 0 in [0,1], -1 in [-1,1]
(? (true and false))  # -> 0.5 (avg), 0 (avg in [-1,1])
```

Truth constants can be **redefined** to custom values:

```lino
(true: 0.8)           # Redefine true to 0.8
(false: 0.2)          # Redefine false to 0.2
(? true)              # -> 0.8
(? false)             # -> 0.2
```

When the range changes (via `(range: ...)`), truth constants are automatically re-initialized to their defaults for the new range.

### Operator Redefinitions

#### Binary operator composition

```lino
(operator: unary_op binary_op)
```

Example: `(!=: not =)` defines `!=` as the negation of `=`.

#### Aggregator selection

For `and` and `or` operators, you can choose different aggregators:

```lino
(and: avg)   # Average (default)
(and: min)   # Minimum (Kleene/Łukasiewicz AND)
(and: max)   # Maximum
(and: prod)  # Product
(and: ps)    # Probabilistic sum: 1 - (1-p1)*(1-p2)*...

(or: max)    # Maximum (default, Kleene/Łukasiewicz OR)
(or: avg)    # Average
(or: min)    # Minimum
(or: prod)   # Product
(or: ps)     # Probabilistic sum
```

### Arithmetic

```lino
(<A> + <B>)   # Addition
(<A> - <B>)   # Subtraction
(<A> * <B>)   # Multiplication
(<A> / <B>)   # Division
```

All arithmetic uses **decimal-precision rounding** (12 significant decimal places) to eliminate IEEE-754 floating-point artefacts:

```lino
(? (0.1 + 0.2))             # -> 0.3  (not 0.30000000000000004)
(? ((0.1 + 0.2) = 0.3))     # -> 1    (true)
(? ((0.3 - 0.1) = 0.2))     # -> 1    (true)
```

Arithmetic operands are not clamped to the logic range, allowing natural numeric computation. Clamping occurs only when results are used in a logical context (queries, `and`, `or`, etc.).

### Queries

```lino
(? <expression>)
```

Queries are evaluated and their truth value is printed to stdout.

### Comments

```lino
# Line comments start with #
(a: a is a)  # Inline comments are also supported
```

## Built-in Operators

- `=`: Equality (checks assigned probability, then structural equality, then numeric comparison with decimal precision)
- `!=`: Inequality (defined as `not =` by default)
- `not`: Logical negation — mirrors around the midpoint of the range (`1 - x` in `[0,1]`; `-x` in `[-1,1]`)
- `and`: Conjunction (average by default, configurable)
- `or`: Disjunction (maximum by default, configurable)
- `+`: Addition (decimal-precision)
- `-`: Subtraction (decimal-precision)
- `*`: Multiplication (decimal-precision)
- `/`: Division (decimal-precision, returns 0 on division by zero)

## Examples

Example `.lino` files are available in both `js/` and `rust/` directories.

### Standard Logic (with avg semantics)

See `demo.lino`:

```lino
(a: a is a)
(!=: not =)
(and: avg)
(or: max)

((a = a) has probability 1)
((a != a) has probability 0)

(? ((a = a) and (a != a)))   # -> 0.5
(? ((a = a) or  (a != a)))   # -> 1
```

### Flipped Axioms

See `flipped-axioms.lino` — demonstrates that the system can handle arbitrary probability assignments:

```lino
(a: a is a)
(!=: not =)
(and: avg)
(or: max)

((a = a) has probability 0)
((a != a) has probability 1)

(? ((a = a) and (a != a)))   # -> 0.5
(? ((a = a) or  (a != a)))   # -> 1
```

### Liar Paradox Resolution

The [liar paradox](https://en.wikipedia.org/wiki/Liar_paradox) ("this statement is false") is irresolvable in classical 2-valued logic. In many-valued logics (ternary and above), it resolves to the **midpoint** of the range — the fixed point of negation.

See `examples/liar-paradox.lino` — resolution in `[0, 1]` range:

```lino
(s: s is s)
(!=: not =)
(and: avg)
(or: max)

((s = false) has probability 0.5)
(? (s = false))          # -> 0.5  (50% from 0% to 100%)
(? (not (s = false)))    # -> 0.5  (fixed point: not(0.5) = 0.5)
```

See `examples/liar-paradox-balanced.lino` — resolution in `[-1, 1]` range:

```lino
(range: -1 1)

(s: s is s)
(!=: not =)
(and: avg)
(or: max)

((s = false) has probability 0)
(? (s = false))          # -> 0   (0% from -100% to 100%)
(? (not (s = false)))    # -> 0   (fixed point: not(0) = 0)
```

### Ternary Kleene Logic

See `examples/ternary-kleene.lino` — demonstrates [Kleene's strong three-valued logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics) where AND=min, OR=max:

```lino
(valence: 3)
(and: min)
(or: max)

(? (0.5 and 1))          # -> 0.5  (unknown AND true = unknown)
(? (0.5 and 0))          # -> 0    (unknown AND false = false)
(? (0.5 or 1))           # -> 1    (unknown OR true = true)
(? (0.5 or 0))           # -> 0.5  (unknown OR false = unknown)
(? (not 0.5))            # -> 0.5  (NOT unknown = unknown)
(? (0.5 or (not 0.5)))   # -> 0.5  (law of excluded middle FAILS!)
```

In [Kleene logic](https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics), the law of excluded middle (`A ∨ ¬A`) is **not** a tautology — this is the key difference from [classical logic](https://en.wikipedia.org/wiki/Classical_logic).

## Testing

Both implementations have 122 matching tests:

```bash
# JavaScript
cd js && npm test

# Rust
cd rust && cargo test
```

The test suites cover:
- Tokenization, parsing, and quantization
- Evaluation logic and operator aggregators
- Many-valued logics: unary, binary (Boolean), ternary (Kleene), quaternary, quinary, higher N-valued, and continuous (fuzzy)
- Both `[0, 1]` and `[-1, 1]` ranges
- Truth constants (`true`, `false`, `unknown`, `undefined`): defaults, redefinition, range changes, use in expressions, quantization
- Liar paradox resolution across logic types
- Decimal-precision arithmetic (`+`, `-`, `*`, `/`) and numeric equality

## API

See language-specific documentation:
- [JavaScript API](./js/README.md#api)
- [Rust API](./rust/README.md#api)

## References

- [Many-valued logic](https://en.wikipedia.org/wiki/Many-valued_logic) — overview of logics with more than two truth values
- [Boolean algebra](https://en.wikipedia.org/wiki/Boolean_algebra) — classical 2-valued logic
- [Three-valued logic](https://en.wikipedia.org/wiki/Three-valued_logic) — ternary logics (Kleene, Łukasiewicz, Bochvar)
- [Łukasiewicz logic](https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic) — N-valued and infinite-valued extensions
- [Fuzzy logic](https://en.wikipedia.org/wiki/Fuzzy_logic) — continuous-valued logic with degrees of truth
- [Balanced ternary](https://en.wikipedia.org/wiki/Balanced_ternary) — ternary system using {-1, 0, 1}
- [Liar paradox](https://en.wikipedia.org/wiki/Liar_paradox) — "this statement is false" and its resolution in many-valued logics

## License

See [LICENSE](LICENSE) file.
