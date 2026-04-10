# Case Study: Issue #20 — Standard Examples First, Belnap's Four-Valued Logic

## Timeline

1. **Issue opened**: User (via ChatGPT dialog) noticed that RML's `(and: avg)` semantics produce `0.5` for `(a = a) AND (a != a)`, which is non-standard — no mainstream logic assigns 0.5 to a conjunction of a tautology and contradiction.
2. **ChatGPT analysis**: Identified that 0.5 for paradoxes is closest to [Belnap's four-valued logic](https://en.wikipedia.org/wiki/Four-valued_logic#Belnap), where "both true and false" (contradiction) is a legitimate truth value.
3. **Requirements identified**: Add standard examples first, define `both`/`neither` keywords, ensure everything is tested and configurable.
4. **Feedback (PR #21)**: `both` and `neither` should be **operators** altering the AND operation, not truth constants. This aligns with Belnap's original framework where they represent different *conjunction strategies*.

## Root Cause Analysis

### Problem 1: Examples surprise users
The existing examples started with `(and: avg)` (the non-standard averaging aggregator), which produces `0.5` for `(true AND false)`. Users familiar with standard logic expect `0` (using min) or specific probabilistic rules (using product). The `avg` result is unintuitive without first showing standard alternatives.

**Root cause**: Examples were added chronologically as features were developed, not ordered pedagogically.

### Problem 2: No standard Belnap support
The liar paradox resolving to `0.5` is actually well-grounded in [Belnap's four-valued logic](https://en.wikipedia.org/wiki/Four-valued_logic#Belnap), where the four truth values are:
- **True** (T) — known to be true
- **False** (F) — known to be false
- **Both** (B) — both true and false (contradiction/paradox)
- **Neither** (N) — neither true nor false (unknown/gap)

However, RML lacked the `both` and `neither` keywords, so users couldn't express these concepts in standard terminology.

**Root cause**: The system supported the numeric values (0.5 maps to "both" in Belnap's encoding) but lacked the operators that make the connection to standard theory explicit.

### Problem 3: Constants vs Operators
Initial implementation made `both` and `neither` truth constants (symbol probabilities). However, in Belnap's framework they represent *different ways of combining truth values* — they alter how conjunction works:

- **`both`** (gullibility): combines info even when contradictory → `avg` semantics → `(true both false) = 0.5`
- **`neither`** (consensus): only propagates agreed info → `product` semantics → `(true neither false) = 0`

**Root cause**: Modeling Belnap's concepts as static values missed their dynamic, operational nature.

## Requirements (from issue and PR feedback)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Add standard/common examples first, so users aren't surprised | Done |
| 2 | After standard examples, show non-standard ones (avg semantics, etc.) | Done |
| 3 | Add `both` as an operator (not constant) for Belnap's contradiction | Done |
| 4 | Add `neither` as an operator (not constant) for Belnap's gap | Done |
| 5 | Support paradox resolution to 0.5 within standard theory | Done |
| 6 | Show how everything is configurable and redefinable | Done |
| 7 | Each example tested, new tests added, code updated | Done |
| 8 | Case study analysis | Done |

## Solution

### 1. Belnap Operators: `both` and `neither`

Added two new operators to both JS and Rust implementations:

```javascript
// In constructor ops:
'both': (...xs) => xs.length ? decRound(xs.reduce((a,b)=>a+b,0)/xs.length) : this.lo,  // avg
'neither': (...xs) => xs.length ? decRound(xs.reduce((a,b)=>a*b,1)) : this.lo,          // product
```

These operators:
- Use composite natural language syntax: `(both A and B)`, `(neither A nor B)`
- Also support prefix form: `(both true false)` and infix form: `(true both false)` for backward compatibility
- Support variadic form: `(both A and B and C)`, `(neither A nor B nor C)`
- Default to `avg` (both) and `product` (neither) aggregators
- Are redefinable like `and` and `or`: `(both: min)`, `(neither: max)`
- Auto-update when the range changes via `reinit_ops`
- Support all aggregator types: avg, min, max, product, probabilistic_sum

Key semantic difference from the AND operator:
- `(both true and false) = 0.5` — contradiction (avg: (1+0)/2)
- `(neither true nor false) = 0` — gap (product: 1*0)
- `(true and false) = 0.5` — default AND uses avg too, but `both`/`neither` provide explicit Belnap semantics

### 2. New Standard Examples

Four new example files added (in order of increasing non-standardness):

1. **`classical-logic.lino`** — Boolean 2-valued logic with standard laws (excluded middle, non-contradiction, double negation)
2. **`propositional-logic.lino`** — Standard probabilistic propositional logic with product/probabilistic_sum
3. **`fuzzy-logic.lino`** — Zadeh fuzzy logic with min/max connectives
4. **`belnap-four-valued.lino`** — Belnap's 4-valued logic demonstrating `both`/`neither` operators

### 3. Reordered Documentation

The README Examples section was restructured to present standard examples first:
1. Classical Logic (Boolean) — familiar to everyone
2. Propositional Logic — standard probabilistic
3. Fuzzy Logic — standard Zadeh
4. Ternary Kleene — standard 3-valued
5. Belnap Four-Valued — standard 4-valued with both/neither operators
6. Liar Paradox — natural resolution via midpoint
7. Custom Operators — non-standard avg semantics (moved after standard)
8. Bayesian/Markov/Self-Reasoning — advanced applications

### 4. Inline Comment Fix

Fixed a bug where inline `#` comments containing colons (e.g., `# -> 0.5 disagree: contradiction`) caused the LiNo parser to fail. The `run()` function now strips inline comments after closing parentheses before passing text to the parser.

## Related Work

### Belnap's Four-Valued Logic (FOUR)

Proposed by Nuel Belnap in 1977 for reasoning with potentially contradictory information from multiple sources. The four values form a **bilattice** with two orderings:
- **Truth ordering**: F < {N,B} < T
- **Knowledge/Information ordering**: N < {T,F} < B

The two orderings give rise to different operations:
- **Truth lattice ops**: standard AND (∧) = glb, OR (∨) = lub
- **Knowledge lattice ops**: consensus (⊗) = meet, gullibility (⊕) = join

In RML, `both` maps to gullibility (avg) and `neither` maps to consensus (product) in the numeric [0,1] encoding.

References:
- Belnap, N.D. (1977). "A useful four-valued logic." In Modern Uses of Multiple-Valued Logic.
- [Wikipedia: Four-valued logic](https://en.wikipedia.org/wiki/Four-valued_logic#Belnap)

### Numeric Encoding

The standard numeric encoding maps Belnap's values to [0,1]:
- T = 1, F = 0, B = 0.5, N = 0.5

Note: B and N have the same numeric value (0.5) in a simple [0,1] mapping, but the *operators* that produce them differ — `both` uses avg (creating contradiction at midpoint) while `neither` uses product (creating gap at zero).

### Existing Libraries

| Library | Language | Belnap Support |
|---------|----------|----------------|
| [logic-ts](https://github.com/nicholasgasior/logic-ts) | TypeScript | 4-valued Belnap |
| [belern](https://crates.io/crates/belern) | Rust | Belnap + evidence theory |
| [four-valued](https://hackage.haskell.org/package/four-valued) | Haskell | Belnap bilattice |

RML's approach differs: rather than hardcoding Belnap semantics, it provides configurable operators that can model Belnap's logic among many others.

## Testing

- **JS**: 297 tests, all passing
- **Rust**: 230 tests, all passing
- Test categories for both/neither:
  - Default aggregator behavior (avg for both, product for neither)
  - Aggregator redefinition (both: min, neither: max)
  - Prefix and infix form support
  - Fuzzy value computation
  - Range change behavior
  - Issue scenario reproduction: `(a=a) both (a!=a)` → 0.5
