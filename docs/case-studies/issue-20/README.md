# Case Study: Issue #20 — Standard Examples First, Belnap's Four-Valued Logic

## Timeline

1. **Issue opened**: User (via ChatGPT dialog) noticed that RML's `(and: avg)` semantics produce `0.5` for `(a = a) AND (a != a)`, which is non-standard — no mainstream logic assigns 0.5 to a conjunction of a tautology and contradiction.
2. **ChatGPT analysis**: Identified that 0.5 for paradoxes is closest to [Belnap's four-valued logic](https://en.wikipedia.org/wiki/Four-valued_logic#Belnap), where "both true and false" (contradiction) is a legitimate truth value.
3. **Requirements identified**: Add standard examples first, define `both`/`neither` keywords, ensure everything is tested and configurable.

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

**Root cause**: The system supported the numeric values (0.5 maps to "both" in Belnap's encoding) but lacked the named constants that make the connection to standard theory explicit.

## Requirements (from issue)

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Add standard/common examples first, so users aren't surprised | Done |
| 2 | After standard examples, show non-standard ones (avg semantics, etc.) | Done |
| 3 | Add `both` keyword for Belnap's "both true and false" | Done |
| 4 | Add `neither` keyword for Belnap's "neither true nor false" | Done |
| 5 | Support paradox resolution to 0.5 within standard theory | Done |
| 6 | Show how everything is configurable and redefinable | Done |
| 7 | Each example tested, new tests added, code updated | Done |
| 8 | Case study analysis | Done |

## Solution

### 1. New Truth Constants: `both` and `neither`

Added two new predefined truth constants to both JS and Rust implementations:

```javascript
// In _initTruthConstants():
this.symbolProb.set('both', this.mid);    // midpoint of range
this.symbolProb.set('neither', this.mid); // midpoint of range
```

These constants:
- Default to the midpoint of the current range (0.5 in [0,1], 0 in [-1,1])
- Are redefinable like all other truth constants: `(both: 0.7)`
- Auto-update when the range changes
- Are fixed points of negation: `not(both) = both`

### 2. New Standard Examples

Four new example files added (in order of increasing non-standardness):

1. **`classical-logic.lino`** — Boolean 2-valued logic with standard laws (excluded middle, non-contradiction, double negation)
2. **`propositional-logic.lino`** — Standard probabilistic propositional logic with product/probabilistic_sum
3. **`fuzzy-logic.lino`** — Zadeh fuzzy logic with min/max connectives
4. **`belnap-four-valued.lino`** — Belnap's 4-valued logic demonstrating `both`/`neither` constants

### 3. Reordered Documentation

The README Examples section was restructured to present standard examples first:
1. Classical Logic (Boolean) — familiar to everyone
2. Propositional Logic — standard probabilistic
3. Fuzzy Logic — standard Zadeh
4. Ternary Kleene — standard 3-valued
5. Belnap Four-Valued — standard 4-valued with both/neither
6. Liar Paradox — natural resolution via midpoint
7. Custom Operators — non-standard avg semantics (moved after standard)
8. Bayesian/Markov/Self-Reasoning — advanced applications

## Related Work

### Belnap's Four-Valued Logic (FOUR)

Proposed by Nuel Belnap in 1977 for reasoning with potentially contradictory information from multiple sources. The four values form a **bilattice** with two orderings:
- **Truth ordering**: F < N < B < T (or F < {N,B} < T)
- **Knowledge ordering**: N < {T,F} < B

References:
- Belnap, N.D. (1977). "A useful four-valued logic." In Modern Uses of Multiple-Valued Logic.
- [Wikipedia: Four-valued logic](https://en.wikipedia.org/wiki/Four-valued_logic#Belnap)

### Numeric Encoding

The standard numeric encoding maps Belnap's values to [0,1]:
- T = 1, F = 0, B = 0.5, N = 0.5

Note: B and N have the same numeric value (0.5) but different semantic meaning. In a richer implementation, they could be distinguished as separate symbols with different behaviors in certain operations. RML keeps them numerically equal for simplicity while providing distinct keywords for expressiveness.

### Existing Libraries

| Library | Language | Belnap Support |
|---------|----------|----------------|
| [logic-ts](https://github.com/nicholasgasior/logic-ts) | TypeScript | 4-valued Belnap |
| [belern](https://crates.io/crates/belern) | Rust | Belnap + evidence theory |
| [four-valued](https://hackage.haskell.org/package/four-valued) | Haskell | Belnap bilattice |

RML's approach differs: rather than hardcoding Belnap semantics, it provides configurable truth constants that can model Belnap's logic among many others.

## Testing

- **JS**: 294 tests (199 original + 95 new), all passing
- **Rust**: 226 tests (199 original + 27 new), all passing
- New test categories:
  - `both`/`neither` defaults, redefinition, range changes, expressions, fixed points
  - Classical logic example validation
  - Propositional logic example validation
  - Fuzzy logic example validation
  - Belnap four-valued logic example validation
