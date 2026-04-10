# Case Study: Cyclic Markov Networks vs. Acyclic Bayesian Networks

**Issue:** [#18 — Add example of cyclic Markov network as opposite to acyclic Bayesian network](https://github.com/link-foundation/relative-meta-logic/issues/18)

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background: Bayesian Networks vs. Markov Networks](#background-bayesian-networks-vs-markov-networks)
3. [Requirements Analysis](#requirements-analysis)
4. [Current State](#current-state)
5. [Solution Plan](#solution-plan)
6. [Existing Tools and Libraries](#existing-tools-and-libraries)
7. [References](#references)

---

## Executive Summary

The issue requests adding a **cyclic Markov network** example as a structural opposite to the existing **acyclic Bayesian network** example. Both are fundamental probabilistic graphical models, but they differ in a key structural property:

- **Bayesian networks** are **directed acyclic graphs (DAGs)** — edges have direction (cause → effect) and no cycles are permitted.
- **Markov networks** (Markov random fields) are **undirected graphs** — edges are symmetric and **cycles are allowed**.

This case study analyzes the requirements, proposes a solution plan, and documents the implementation of cyclic Markov network examples alongside improvements to existing Bayesian examples in RML notation.

---

## Background: Bayesian Networks vs. Markov Networks

### Bayesian Networks (Directed Acyclic Graphs)

A [Bayesian network](https://en.wikipedia.org/wiki/Bayesian_network) is a probabilistic graphical model that represents a set of variables and their conditional dependencies via a **directed acyclic graph (DAG)**. Each node represents a random variable, and each directed edge represents a conditional dependency.

**Key properties:**
- **Directed**: Edges have a direction (parent → child), representing causality or conditional dependence
- **Acyclic**: No directed path from a node back to itself
- **Factorization**: Joint probability factors as a product of conditional probabilities: `P(X₁,...,Xₙ) = ∏ P(Xᵢ | Parents(Xᵢ))`
- **Example**: The classic "sprinkler" network: Cloudy → Sprinkler, Cloudy → Rain, Sprinkler → Wet Grass, Rain → Wet Grass

### Markov Networks (Undirected Graphical Models)

A [Markov random field (MRF)](https://en.wikipedia.org/wiki/Markov_random_field), also known as a Markov network, is a probabilistic graphical model that represents variables and their dependencies via an **undirected graph**. Unlike Bayesian networks, edges are symmetric and cycles are permitted.

**Key properties:**
- **Undirected**: Edges are symmetric — if A relates to B, then B relates to A
- **Cycles allowed**: The graph can contain loops (e.g., A—B—C—A)
- **Factorization**: Joint probability factors over **cliques** (fully connected subgraphs): `P(X₁,...,Xₙ) = (1/Z) ∏ φ(Xc)` where Z is a normalization constant
- **Example**: The [Ising model](https://en.wikipedia.org/wiki/Ising_model) — a lattice where each node interacts with its neighbors

### Structural Comparison

| Property | Bayesian Network | Markov Network |
|----------|-----------------|----------------|
| Edge type | Directed (→) | Undirected (—) |
| Cycles | Not allowed (acyclic) | Allowed (cyclic) |
| Factorization | Conditional probabilities | Potential functions over cliques |
| Causality | Can represent | Cannot represent |
| Symmetry | Asymmetric dependencies | Symmetric dependencies |
| Normalization | Implicit (conditional probabilities sum to 1) | Explicit (partition function Z) |

### Links Theory Perspective

In [Links Theory](https://github.com/link-foundation/meta-theory), links are inherently **directional** — each link has a source and a target. This makes links a natural fit for both:
- **Bayesian networks**: directed edges map directly to links
- **Markov networks**: undirected edges are represented as **bidirectional link pairs** (A→B and B→A), which naturally creates cycles

This directional foundation means that even undirected relationships are expressed through directed primitives, preserving the fundamental directionality of links while supporting cyclic structures.

---

## Requirements Analysis

From the issue description, the following requirements are identified:

### R1: Directionality
> "Both should be directional as links theory is directional by default."

All examples must use directional link notation. For Markov networks, undirected edges should be represented as bidirectional link pairs.

### R2: Separate Example Files
> "Also make multiple examples based on already existing in separate files for Bayesian inference/networks, for Markov chains/networks."

Create separate `.lino` files:
- `bayesian-inference.lino` — Bayesian inference (Bayes' theorem) *(already exists)*
- `bayesian-network.lino` — Acyclic Bayesian network *(already exists, needs directionality update)*
- `markov-chain.lino` — Markov chain (sequential transitions) *(already exists)*
- `markov-network.lino` — Cyclic Markov network **(new)**

### R3: Correctness Verification
> "And double check the correctness of all of them."

All examples must produce mathematically correct results, verified by automated tests.

### R4: Self-Descriptive Notation
> "Try to support fully self descriptive notation, meaning we use as much of existing notation, but support as much of math features as possible needed to fully represent these examples."

Use RML's native features (terms, probability assignments, operators, queries) to make examples readable and self-documenting.

### R5: Beginner-Friendly Naming
> "Also rename all short names like `prod` and `ps` into their full versions, the notation should be beginner friendly by default. So we should prefer full english words."

Rename:
- `prod` → `product` — clearer meaning for probabilistic product (AND for independent events)
- `ps` → `probabilistic_sum` — fully descriptive name for P(A∪B) = 1-(1-P(A))*(1-P(B))

Maintain backward compatibility by keeping the short names as aliases.

### R6: Case Study Documentation
> "We need to collect data related about the issue to this repository, make sure we compile that data to `./docs/case-studies/issue-{id}` folder."

This document fulfills this requirement.

---

## Current State

### Existing Examples (before this issue)
1. **bayesian-inference.lino**: Medical diagnosis using Bayes' theorem with pure arithmetic
2. **bayesian-network.lino**: Sprinkler network with `prod` and `ps` aggregators
3. **markov-chain.lino**: Weather system with transition matrix and stationary distribution

### Gaps Identified
1. No **cyclic Markov network** example (the primary request)
2. Bayesian network example doesn't emphasize **directionality** of edges
3. Short aggregator names (`prod`, `ps`) are not beginner-friendly
4. No example showing the structural contrast between acyclic and cyclic networks

---

## Solution Plan

### Phase 1: Source Code — Add Full Aggregator Names
1. Add `product` as an alias for `prod` in both JS and Rust implementations
2. Add `probabilistic_sum` as an alias for `ps` in both JS and Rust implementations
3. Keep `prod` and `ps` as backward-compatible aliases
4. Update all comments and documentation to prefer full names

### Phase 2: Examples — Update and Create
1. Update `bayesian-network.lino` to emphasize directional edges and use full aggregator names
2. Update `markov-chain.lino` to use full aggregator names
3. Create `markov-network.lino` with a cyclic undirected network using bidirectional links
4. Create `bayesian-network-diagnosis.lino` with a more elaborate directed Bayesian network

### Phase 3: Tests — Verify Correctness
1. Add tests for new `product` and `probabilistic_sum` aggregator names
2. Add tests for cyclic Markov network computations
3. Verify all existing tests still pass

### Phase 4: Documentation
1. Update README.md to reference full aggregator names
2. Update ARCHITECTURE.md operator table
3. Create this case study document

---

## Existing Tools and Libraries

### Probabilistic Graphical Model Libraries

| Library | Language | Features |
|---------|----------|----------|
| [pgmpy](https://pgmpy.org/) | Python | BNs, MRFs, inference, learning |
| [bnlearn](https://www.bnlearn.com/) | R | BN structure learning and inference |
| [pyAgrum](https://pyagrum.readthedocs.io/) | Python | BNs, MRFs, influence diagrams |
| [libDAI](https://staff.fnwi.uva.nl/j.m.mooij/libDAI/) | C++ | Factor graphs, MRFs, exact/approximate inference |
| [Stan](https://mc-stan.org/) | Multi | Bayesian inference, MCMC |

### Key Insight for RML

Unlike these specialized libraries, RML represents probabilistic graphical models using its **general-purpose link notation**. This means:
- No separate graph construction API — the network *is* the notation
- Bayesian networks and Markov networks use the same primitives (terms, links, operators)
- The structural difference (acyclic vs. cyclic) is visible in the link pattern itself

---

## References

1. [Bayesian network — Wikipedia](https://en.wikipedia.org/wiki/Bayesian_network)
2. [Markov random field — Wikipedia](https://en.wikipedia.org/wiki/Markov_random_field)
3. [Ising model — Wikipedia](https://en.wikipedia.org/wiki/Ising_model)
4. [Graphical model — Wikipedia](https://en.wikipedia.org/wiki/Graphical_model)
5. [Markov chain — Wikipedia](https://en.wikipedia.org/wiki/Markov_chain)
6. [Bayesian inference — Wikipedia](https://en.wikipedia.org/wiki/Bayesian_inference)
7. [CS228 Notes: Undirected Graphical Models — Stanford](https://ermongroup.github.io/cs228-notes/representation/undirected/)
8. [CS228 Notes: Directed Graphical Models — Stanford](https://ermongroup.github.io/cs228-notes/representation/directed/)
9. [Links Theory — Meta Theory](https://github.com/link-foundation/meta-theory)
10. [Cyclic Directed Probabilistic Graphical Model — arXiv](https://arxiv.org/pdf/2310.16525)
