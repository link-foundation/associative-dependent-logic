# Existing Tools and Libraries Analysis

This document catalogs existing implementations, libraries, and academic work that can help implement dependent types in the ADL/LiNo system.

## Tier 1: Directly Usable (JavaScript + Rust)

### calculus-of-constructions (JavaScript)

- **Repository:** [VictorTaelin/calculus-of-constructions](https://github.com/VictorTaelin/calculus-of-constructions)
- **npm:** [calculus-of-constructions](https://www.npmjs.com/package/calculus-of-constructions)
- **Size:** ~400 lines, ~2.3kb minified
- **Features:** Full CoC with type checking, normalization, parsing
- **API:** `CoC.read()`, `CoC.type()`, `CoC.norm()`, `CoC.show()`
- **Approach:** HOAS (Higher-Order Abstract Syntax) — no explicit substitution
- **Why relevant:** Could be used as the type-checking backend for ADL's JavaScript implementation. The LiNo AST could be translated to CoC terms, type-checked, and results mapped back.

### FormCoreJS (JavaScript)

- **Repository:** [HigherOrderCO/FormCoreJS](https://github.com/HigherOrderCO/FormCoreJS)
- **npm:** [formcore-js](https://www.npmjs.com/package/formcore-js)
- **Size:** ~700 lines, zero dependencies
- **Features:** Self-dependent types, inductive reasoning, compilation to efficient JS
- **Why relevant:** More powerful than basic CoC (has self types for inductive reasoning). This is the kernel of Kind, a full proof assistant. Very small and auditable — follows the de Bruijn criterion.

### Typechecker Zoo — CoC (Rust)

- **Website:** [sdiehl.github.io/typechecker-zoo/coc](https://sdiehl.github.io/typechecker-zoo/coc/calculus-of-constructions.html)
- **Features:** Full CoC implementation in Rust with bidirectional type checking, universe hierarchy, dependent products, Σ-types, constraint solving
- **Why relevant:** Reference implementation for the Rust side of ADL. Uses standard Rust enum for term representation.

### links-notation (JavaScript + Rust)

- **npm:** [links-notation](https://www.npmjs.com/package/links-notation) (v0.13.0)
- **crate:** [links-notation](https://crates.io/crates/links-notation) (v0.13.0)
- **Status:** Already integrated in ADL
- **Why relevant:** The parser is already there. No new parsing infrastructure needed — new constructs like `(Pi ...)`, `(lam ...)` are just new LiNo expressions with new evaluation rules.

## Tier 2: Reference Implementations (Other Languages)

### elaboration-zoo (Haskell)

- **Repository:** [AndrasKovacs/elaboration-zoo](https://github.com/AndrasKovacs/elaboration-zoo)
- **Features:** Progressive series of dependent type elaboration implementations
- **Coverage:** Basics of unification, inference, implicit argument handling
- **Why relevant:** Best pedagogical resource for understanding how elaboration (the process of turning user-friendly syntax into core type theory) works. Each package adds one feature.

### nano-Agda (Haskell)

- **Repository:** [jyp/nano-Agda](https://github.com/jyp/nano-Agda)
- **Size:** ~200 lines
- **Based on:** "A Tutorial Implementation of a Dependently Typed Lambda Calculus" (Löh, McBride, Swierstra)
- **Why relevant:** Minimal working example showing exactly what's needed.

### LaTTe Kernel (Clojure/ClojureScript)

- **Repository:** [latte-central/latte-kernel](https://github.com/latte-central/latte-kernel)
- **Features:** Small trusted kernel for a proof assistant
- **Principle:** Follows the de Bruijn criterion — small code base that can be independently verified
- **Why relevant:** Shows how to build a minimal kernel that's correct by design. Could inform the architecture of ADL's type checker.

### tt (OCaml, by Andrej Bauer)

- **Tutorial:** [How to implement dependent type theory I](https://math.andrej.com/2012/11/08/how-to-implement-dependent-type-theory-i/)
- **Tutorial Part II:** [How to implement dependent type theory II](https://math.andrej.com/2012/11/11/how-to-implement-dependent-type-theory-ii/)
- **Size:** ~92 lines of core logic
- **Features:** Variables, universes, Π-types, λ, application, substitution, normalization, type checking
- **Why relevant:** The gold standard for "how small can a dependent type checker be?" Clear code, excellent documentation.

### Interaction-Type-Theory (Rust)

- **Repository:** [VictorTaelin/Interaction-Type-Theory](https://github.com/VictorTaelin/Interaction-Type-Theory)
- **Features:** CoC via interaction combinators + a "Decay" rule
- **Why relevant:** Novel approach that represents type theory as a graph rewriting system. This is conceptually close to the associative model where everything is links/nodes.

## Tier 3: Academic Papers

### A Type Theory for Probabilistic and Bayesian Reasoning

- **Authors:** Robin Adams, Bart Jacobs
- **Year:** 2015
- **arXiv:** [1511.09230](https://arxiv.org/abs/1511.09230)
- **Published:** LIPIcs, TYPES 2015
- **Key contribution:** COMET system — type theory with fuzzy predicates, normalization, and conditioning of probabilistic states. Shows how to do Bayesian inference as type-theoretic operations.
- **Why relevant:** Provides the theoretical foundation for Option C (Probabilistic Dependent Type Theory). This is the closest academic work to what ADL would become.

### A Probabilistic Dependent Type System based on Non-Deterministic Beta Reduction

- **Year:** 2016
- **arXiv:** [1602.06420](https://arxiv.org/abs/1602.06420)
- **Key contribution:** Extends intuitionistic type theory (dependent sums and products) with stochastic functions via non-deterministic β-reduction.
- **Why relevant:** Shows how to add randomness/probability directly into the reduction rules of type theory, rather than as an external layer.

### Probabilistic Type Theory and Natural Language Semantics

- **Author:** Robin Cooper
- **Published:** LILT 2015
- **Key contribution:** Uses TTR (Type Theory with Records) with probabilistic judgments for natural language semantics.
- **Why relevant:** Shows real-world application of probabilistic type theory, specifically for modeling gradience in meaning (not just true/false).

### Propositions as Types

- **Author:** Philip Wadler
- **Year:** 2015
- **PDF:** [propositions-as-types.pdf](https://homepages.inf.ed.ac.uk/wadler/papers/propositions-as-types/propositions-as-types.pdf)
- **Why relevant:** Definitive accessible overview of the Curry-Howard correspondence, which underpins the entire connection between logic (ADL) and type theory (Lean/Rocq).

## Integration Strategy

### For JavaScript Implementation

1. **Quick start:** Use `calculus-of-constructions` npm package as backend
2. **Translation layer:** Convert LiNo AST ↔ CoC terms
3. **Extended features:** Port additional features from `formcore-js` as needed

```javascript
import { parse as parseLiNo } from 'links-notation';
import CoC from 'calculus-of-constructions';

function typeCheck(linoSource) {
  const links = parseLiNo(linoSource);
  const cocTerms = translateToCoC(links);  // LiNo AST → CoC terms
  return CoC.type(cocTerms);               // Type inference
}
```

### For Rust Implementation

1. **Reference:** Use Typechecker Zoo CoC implementation as starting point
2. **Custom implementation:** Write a minimal type checker (~300-500 lines) that operates on the existing ADL AST
3. **Integration:** Add type checking as an optional mode alongside probabilistic evaluation

```rust
use links_notation::parse_lino_to_links;

fn type_check(source: &str) -> Result<Type, TypeError> {
    let links = parse_lino_to_links(source);
    let ast = parse_to_typed_ast(links)?;
    infer_type(&Context::new(), &ast)
}
```

### Incremental Path

1. **v0.7.0:** Add `(Type N)`, `(Pi ...)`, `(lam ...)`, `(app ...)` — basic CoC
2. **v0.8.0:** Add inductive types, pattern matching
3. **v0.9.0:** Add probabilistic propositions (PProp)
4. **v1.0.0:** Self-describing meta-theory (the system can describe its own type rules)
