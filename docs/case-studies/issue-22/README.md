# Case Study: Competitor Concepts and Feature Comparison

**Issue:** [#22 - Create a list of all core concepts and features that competitors support, with comparison tables by concepts and features separately](https://github.com/link-foundation/relative-meta-logic/issues/22)

## Executive Summary

Relative Meta-Logic (RML) is currently strongest as a small, executable, probabilistic meta-logic over LiNo links. Its distinctive capabilities are configurable truth ranges, N-valued and continuous truth values, probabilistic and fuzzy operators, paradox-tolerant midpoint semantics, and a uniform link notation that can encode terms, propositions, probabilities, and a prototype dependent type layer.

User-facing standalone comparison tables are available in:

- [Core Concept Comparison](../../CONCEPTS-COMPARISION.md)
- [Product Feature Comparison](../../FEATURE-COMPARISION.md)

The compared systems cluster into five groups:

1. **LF-family logical frameworks:** Edinburgh LF, Twelf, and HELF focus on representing deductive systems with dependent typed lambda terms. Twelf adds logic programming and metatheorem checking around totality, modes, worlds, termination, and coverage.
2. **Large proof assistants:** Isabelle, Coq/Rocq, and Lean provide mature kernels, libraries, automation, tactics, module systems, and proof engineering workflows.
3. **Formalized logic libraries:** FormalizedFormalLogic/Foundation and Isabelle AFP are not independent provers; they are large proof developments built on Lean and Isabelle respectively.
4. **Higher-order specification/proof systems:** Abella and lambda Prolog focus on lambda-tree syntax, higher-order abstract syntax, and relational specifications with binders.
5. **Domain-specific automated provers:** Pecan is specialized for automatic sequences, numeration systems, and Buchi automata.

The main competitive gap for RML is not expressiveness of its notation. It is the product layer around the notation: proof automation, robust type checking, inductive/coinductive definitions, tactics, libraries, editor integration, module/package systems, and large proof corpora.

## Scope

This document compares the systems explicitly named in issue #22:

| System | Role in this comparison |
|--------|-------------------------|
| RML | This repository: probabilistic, many-valued, link-based meta-logic |
| Twelf | LF implementation with logic programming and metatheorem checking |
| Edinburgh LF | Core logical framework theory behind Twelf and HELF |
| HELF | Haskell implementation of a Twelf/LF subset |
| Isabelle | Generic interactive theorem prover, especially Isabelle/Pure and Isabelle/HOL |
| Coq/Rocq | Interactive theorem prover based on the Calculus of Inductive Constructions |
| Lean | Interactive theorem prover and programming language based on dependent type theory |
| FormalizedFormalLogic/Foundation | Lean 4 library formalizing mathematical logic |
| Isabelle AFP | Archive of formal proof developments checked by Isabelle |
| Abella | Interactive theorem prover for lambda-tree syntax and two-level logic reasoning |
| lambda Prolog | Higher-order logic programming language |
| Pecan | Automated theorem prover based on Buchi automata |

## Legend

| Mark | Meaning |
|------|---------|
| Yes | First-class or central capability |
| Part | Partial, prototype, limited, or indirect support |
| Host | Inherited from the host proof assistant rather than provided by the project itself |
| N/A | Not an independent prover or not applicable |
| No | No evidence of support or outside the system's design goal |

## Core Concept Comparison

### Foundations and Representation

| Core concept | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|--------------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| General meta-logic for object logics | Yes | Yes | Yes | Part | Yes | Part | Part | Yes | Host | Yes | Part | No |
| Uniform representation of syntax and judgments | Yes: links | Yes: LF terms | Yes: LF terms | Yes: LF terms | Part: Pure/HOL terms | Yes: CIC terms | Yes: dependent terms | Host | Host | Yes: two-level terms | Yes: lambda terms | Part: formulas/automata |
| Dependent types | Part | Yes | Yes | Yes | No: HOL is simply typed; Pure is higher-order meta-logic | Yes | Yes | Host | Host | No: simply typed core | No: simply typed | No |
| Dependent products / Pi-types | Part | Yes | Yes | Yes | Part: meta-level universal quantification | Yes | Yes | Host | Host | No | No | No |
| Lambda abstraction and application | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Host | Host | Yes | Yes | No |
| Higher-order abstract syntax / lambda-tree syntax | Part | Yes | Yes | Yes | Part | Part | Part | Host | Host | Yes | Yes | No |
| Judgments-as-types | Part | Yes | Yes | Yes | Part | Yes via Curry-Howard | Yes via propositions-as-types | Host | Host | Part | Part | No |
| Propositions-as-types / proofs-as-terms | Part | Yes in LF style | Yes | Yes | Part: proof terms exist, HOL propositions are bool-like formulas | Yes | Yes | Host | Host | Part | Part | No |
| Universe hierarchy | Part | No: LF has kinds/types, not Lean-style universes | No | No | No | Yes | Yes | Host | Host | No | No | No |
| Self-reference / circularity tolerated | Yes | No | No | No | No | No | No | Host constraints | Host constraints | Part: coinductive reasoning | No | No |

### Logic and Semantics

| Core concept | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|--------------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Classical logic | Yes | Encodable | Encodable | Encodable | Yes | Yes | Yes | Yes | Yes | Encodable | Encodable | Part: decidable fragments |
| Intuitionistic / constructive logic | Part | Yes: LF foundation | Yes | Yes | Encodable | Yes | Yes | Yes | Yes | Yes | Yes | No |
| Higher-order logic | Part | Encodable | Encodable | Encodable | Yes | Encodable in CIC | Encodable in DTT | Host | Host | Part | Yes as programming logic fragment | No |
| First-order logic | Part | Encodable | Encodable | Encodable | Yes | Encodable | Encodable | Yes | Yes | Encodable | Encodable | Part: arithmetic formulas |
| Modal logic | Part: encodable as links | Encodable | Encodable | Encodable | Encodable | Encodable | Encodable | Yes | Yes | Encodable | Encodable | No |
| Many-valued logic | Yes | No | No | No | Encodable | Encodable | Encodable | Part | Part | Encodable | Encodable | No |
| Probabilistic truth values | Yes | No | No | No | Encodable | Encodable | Encodable | No | Part | Encodable | No | No |
| Fuzzy operators | Yes | No | No | No | Encodable | Encodable | Encodable | No | Part | Encodable | No | No |
| Paraconsistent / paradox-tolerant semantics | Yes | No | No | No | Encodable | Encodable with axioms/libraries | Encodable with axioms/libraries | Part: logic zoo focus | Part | Encodable | No | No |
| Automata-theoretic decision procedures | No | No | No | No | External/tools possible | Tactics/libraries possible | Tactics/libraries possible | No | Part | No | No | Yes |

### Metatheory and Proof Objects

| Core concept | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|--------------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Machine-checked proof terms | Part | Yes | Theory only | Yes: LF type checking | Yes | Yes | Yes | Host | Host | Yes | Part: proof search terms | No: automata decision result |
| Small trusted kernel | Part: small evaluator, prototype type layer | Yes: LF checker | Theory only | Yes: LF checker | Yes: LCF-style kernel | Yes | Yes | Host | Host | Yes | No: language implementation | Domain-specific checker |
| Inductive definitions | Part | Encoded as LF families | Encodable | Encodable | Yes | Yes | Yes | Host | Host | Yes | Encodable | No |
| Coinductive definitions | Part | Limited/encoded | No | No | Yes | Yes | Yes | Host | Host | Yes | Encodable | Automata over infinite words |
| Termination / totality checking | No | Yes | No | No | Part: function package and tools | Yes | Yes | Host | Host | Part: stratification/productivity constraints | No | N/A |
| Coverage checking | No | Yes | No | No | Part | Yes | Yes | Host | Host | Part | No | N/A |
| Definitional equality / normalization | Part | Yes | Yes | Yes | Yes | Yes | Yes | Host | Host | Part | Yes: beta conversion for lambda terms | No |
| Metatheorem checking about encoded systems | Part | Yes | Theory goal | No | Yes | Yes | Yes | Yes | Yes | Yes | Part | No |
| Executable specifications | Yes | Yes | No | Part: typecheck only | Part | Yes: programs in Gallina | Yes: functional programs | Host | Host | Yes: specification logic | Yes | Yes: automated proving scripts |
| Decidable complete domain fragment | No: general evaluator | No | No | No | Some procedures | Some tactics/procedures | Some tactics/procedures | Host | Host | No | No | Yes: Buchi/automatic structures |

## Product Feature Comparison

### Authoring and Checking Workflow

| Feature | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|---------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Batch file checking | Yes | Yes | N/A | Yes | Yes | Yes | Yes | Host | Yes | Yes | Yes | Yes |
| Interactive REPL / top-level | Part: CLI queries | Yes | N/A | No | Yes | Yes | Yes | Host | N/A | Yes | Yes | Yes |
| IDE/editor support | No dedicated IDE | Emacs mode | N/A | No | Isabelle/jEdit and PIDE | CoqIDE, Proof General, VS Code ecosystem | VS Code/LSP ecosystem | Host | Host | Emacs/Proof General style support | Teyjus tooling | Vim syntax; online demo |
| Module/import system | Part: file-level examples | Yes | N/A | Part | Yes | Yes | Yes | Host | Host | Yes | Yes | Yes |
| Package/build ecosystem | Part: npm/cargo package | Part | N/A | Cabal/Hackage | Isabelle sessions | Dune/opam/coq_makefile ecosystem | Lake | Lake | Isabelle sessions/releases | Source distribution | Teyjus/ELPI ecosystems | Python scripts/Docker |
| Documentation generation | Markdown docs | Part | N/A | Hackage README | Yes | Yes | Yes | Yes | Yes | Yes: abella_doc | Part | Manual/README |
| Stable large standard library | No | Examples/case studies | N/A | No | Yes | Yes | Yes plus mathlib | Yes: library itself | Yes: archive | Examples | Examples | Domain libraries |

### Proof Engineering

| Feature | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|---------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Tactic language | No | Part: theorem prover/proof search | N/A | No | Yes | Yes | Yes | Host | Host | Yes | Logic programming search | No tactics; automated proving |
| Simplifier / rewriting automation | Part: operators/evaluator | Part | N/A | No | Yes | Yes | Yes | Host | Host | Part | Part | No |
| External ATP/SMT integration | No | No | N/A | No | Yes: Sledgehammer/SMT/ATPs | Plugins/tools | Ecosystem tools | Host | Host | No | No | No |
| Built-in proof search | Part: query evaluation | Yes | N/A | No | Yes | Yes via tactics/plugins | Yes via tactics | Host | Host | Yes | Yes | Yes: decision procedure |
| Counterexample/model finding | No | No | N/A | No | Yes: Nitpick/Quickcheck | Plugins/tools | Ecosystem/tools | Host | Host | No | No | Yes/No by automata emptiness |
| Totality/termination automation | No | Yes | N/A | No | Yes for recursive definitions | Yes | Yes | Host | Host | Part | No | N/A |
| Coverage/productivity checks | No | Yes | N/A | No | Yes for datatypes/functions | Yes | Yes | Host | Host | Part | No | N/A |
| Program extraction / compilation | No | No | N/A | No | Yes: code generator | Yes: extraction | Yes: compiler/code gen | Host | Host | No | Logic programs execute | No general extraction |
| Reflection/metaprogramming | Part: links can encode rules | Part | N/A | No | Isabelle/ML | Ltac/ML/plugins | Yes: Lean metaprogramming/macros | Host | Host | Part | Program-level | No |
| Custom syntax/macros | Part: LiNo links are grammar-light | Fixity/operators | N/A | Limited Twelf subset | Yes | Yes | Yes | Host | Host | Part | Yes | Yes: language-specific syntax |

### Domain and Library Coverage

| Feature | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|---------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Programming language metatheory examples | Part | Yes | Encodable | Yes: examples | Yes | Yes | Yes | Part | Yes | Yes | Yes | No |
| Mathematics library | No | No | N/A | No | Yes | Yes | Yes: mathlib | Yes: mathematical logic | Yes | No | No | No |
| Formalized logic library | Part | Examples | N/A | Examples | Yes | Yes | Yes | Yes | Yes | Examples | Examples | Domain-specific formulas |
| Modal/provability/interpretablity logic corpus | Part | Encodable | Encodable | Encodable | Encodable | Encodable | Encodable | Yes | Yes | Encodable | Encodable | No |
| Probabilistic examples | Yes | No | No | No | Libraries possible | Libraries possible | Libraries possible | No | Some entries possible | No | No | No |
| Graphical model examples | Yes | No | No | No | Libraries possible | Libraries possible | Libraries possible | No | Some entries possible | No | No | No |
| Automatic sequences / numeration systems | No | No | No | No | Encodable with effort | Encodable with effort | Encodable with effort | No | Some entries possible | No | No | Yes |
| Browser/online use | No | No | N/A | No | Limited ecosystem demos | jsCoq exists | Web-based examples exist | Host | Website archive | No | No | Yes: online demo referenced |
| Multi-language implementation parity | Yes: JS and Rust | No | N/A | No: Haskell only | No | No | No | Host | Host | No | Multiple implementations exist | No |
| Human-readable research archive model | Part: case studies | Wiki/case studies | Papers | No | Yes | Yes | Yes | Book/docs | Yes: journal-like archive | Examples/publications | Examples/book | Paper/manual |

## Main RML Advantages

1. **Probabilistic and many-valued semantics are native.** RML supports configurable truth ranges, valence, continuous fuzzy values, probabilistic aggregators, and truth constants directly in the evaluator.
2. **Operators are redefinable at runtime.** The same file can switch between classical, fuzzy, probabilistic, Belnap-style, or custom semantics.
3. **Links provide a low-friction representation substrate.** RML can encode terms, propositions, probabilities, type declarations, graph structures, and self-reference with one parenthesized notation.
4. **Paradox tolerance is a deliberate semantic choice.** RML resolves several self-referential cases to the midpoint rather than rejecting them by stratification.
5. **Implementation surface is small.** The JavaScript and Rust implementations are compact enough to audit and keep behaviorally aligned.

## Main RML Gaps

1. **Proof assistant maturity.** Isabelle, Coq/Rocq, and Lean provide kernels, tactics, libraries, modules, and IDE workflows that RML does not yet have.
2. **LF-family metatheory support.** Twelf has mature support for modes, worlds, totality, termination, coverage, and logic programming around LF signatures.
3. **Inductive and coinductive infrastructure.** RML can encode many patterns as links, but it lacks first-class inductive families, eliminators, recursors, and productivity checks.
4. **Automation.** RML has query evaluation but no comparable Sledgehammer, simplifier, SMT bridge, tactic language, or domain decision procedure.
5. **Library ecosystem.** RML has examples and case studies, while Lean/mathlib, Isabelle/AFP, Rocq libraries, and Foundation provide large reusable formal corpora.
6. **Tooling.** RML lacks dedicated editor integration, language server support, structured diagnostics, proof state visualization, and generated reference docs.
7. **Explicit proof artifacts.** RML evaluates expressions and stores probabilities, but it does not yet produce independently checkable proof terms for derivations.

## Recommended Roadmap

### Near Term

1. **Make this comparison part of the docs index.** Link this case study from the root README or a docs index so users can find the positioning work.
2. **State RML's niche explicitly.** Position RML as a probabilistic many-valued meta-logic and executable link notation, not yet as a replacement for mature proof assistants.
3. **Add proof-state diagnostics before tactics.** Better explanations for failed type queries, missing assignments, and operator resolution would give immediate value.
4. **Add an explicit proof object experiment.** A small derivation trace for equality, probability assignment lookup, and operator application would move RML toward auditable proofs.

### Medium Term

1. **Inductive definitions as links.** Add a minimal representation for constructors, eliminators, and recursion over link-defined datatypes.
2. **Twelf-inspired totality checks.** Start with modes and termination for a small class of relation-like link families.
3. **A small tactic/query language.** Provide named proof/search steps before attempting full automation.
4. **A documentation generator for examples.** Convert `.lino` examples into rendered docs with inputs, outputs, and explanations.

### Long Term

1. **Proof-producing evaluator.** Make every query optionally return a derivation object that can be replayed by a smaller checker.
2. **Library organization.** Establish reusable libraries for classical logic, fuzzy logic, probabilistic reasoning, type theory, and graph models.
3. **Editor integration.** Provide an LSP or VS Code extension with parse errors, query outputs, and hover documentation.
4. **Bridge to mature provers.** Export selected RML fragments to Lean, Isabelle, or Rocq for external certification when a classical proof is desired.

## Source Notes

Primary sources consulted:

| System | Source notes |
|--------|--------------|
| RML | This repository's [README.md](../../../README.md) and [ARCHITECTURE.md](../../../ARCHITECTURE.md) describe the current syntax, evaluator, operators, type features, examples, and tests. |
| Twelf and LF | The Twelf LF page describes LF as a dependently typed lambda calculus for representing deductive systems, and lists Twelf's LF checker, logic programming language, and metatheorem checker: <https://twelf.org/wiki/lf/>. The Twelf logic programming page describes dependently typed higher-order logic programming and `%solve`/`%query`: <https://twelf.org/wiki/logic-programming/>. The Twelf guide lists syntax, reconstruction, logic programming, modes, termination, coverage, totality, theorem prover, ML interface, server, and Emacs interface chapters: <https://www.cs.cmu.edu/~twelf/guide-1-4/twelf_toc.html>. |
| Twelf totality | The Twelf totality tutorial describes `%mode`, `%worlds`, `%total`, and the checks for mode, worlds, termination, input coverage, and output coverage: <https://twelf.org/wiki/proving-metatheorems-proving-totality-assertions-about-the-natural-numbers/>. |
| HELF | The Hackage package states that HELF is a Haskell implementation of LF, parses and typechecks Twelf-style `.elf` files, and only implements a subset of Twelf without type reconstruction or unification: <https://hackage.haskell.org/package/helf>. |
| Isabelle | Official Isabelle documentation and generated docs cover Isabelle/Isar, locales, type classes, datatypes, Sledgehammer, code generation, and related manuals from the documentation index: <https://isabelle.in.tum.de/dist/library/Doc/>. |
| Coq/Rocq | The Coq/Rocq reference manual describes Gallina, the Calculus of Inductive Constructions, Curry-Howard, tactics, interactive and compiled modes: <https://docs.rocq-prover.org/V8.11.1/refman/>. The current Rocq docs cover inductive/coinductive reasoning and extraction: <https://docs.rocq-prover.org/master/refman/proofs/writing-proofs/reasoning-inductives.html> and <https://docs.rocq-prover.org/master/refman/addendum/extraction.html>. |
| Lean | The Lean language reference describes Lean as an interactive theorem prover based on dependent type theory with a minimal kernel, tactic language, functional programming, modules, type classes, macros, and build tools: <https://lean-lang.org/doc/reference/latest/>. The type system docs cover dependent terms, definitional equality, universes, and inductive types: <https://lean-lang.org/doc/reference/latest/The-Type-System/>. |
| FormalizedFormalLogic/Foundation | The Foundation README describes a Lean 4 library for propositional, first-order, modal, provability, interpretability, arithmetic, set theory, proof automation, and logic zoo developments: <https://github.com/FormalizedFormalLogic/Foundation>. |
| Isabelle AFP | The Archive of Formal Proofs describes itself as proof libraries, examples, and larger scientific developments mechanically checked by Isabelle and organized like a scientific journal: <https://www.isa-afp.org/>. |
| Abella | The Abella site describes an interactive theorem prover based on lambda-tree syntax, two-level logic, hereditary Harrop specifications, and reasoning over specifications: <https://abella-prover.org/index.html>. The reference guide documents simply typed specification and reasoning logics: <https://abella-prover.org/reference-guide.html>. |
| lambda Prolog | The Teyjus documentation describes lambda Prolog as a higher-order hereditary Harrop logic programming language with higher-order programming, polymorphic typing, scoping, modules, abstract data types, and lambda terms as data: <https://teyjus.cs.umn.edu/old/language/teyjus_1.html>. The lambda Prolog home page summarizes the same foundations and lambda-tree syntax focus: <https://www.lix.polytechnique.fr/Labo/Dale.Miller/lProlog/>. |
| Pecan | The Pecan repository describes it as an automated theorem prover for Buchi automata with support for numeration systems and automatic words: <https://github.com/ReedOei/Pecan>. The Pecan paper describes the automated theorem prover for automatic sequences using Buchi automata: <https://arxiv.org/abs/2102.01727>. |

## Acceptance Checklist

| Requirement | Status |
|-------------|--------|
| Include all competitors named in issue #22 | Done |
| Separate core concept comparison from feature comparison | Done |
| Compare RML against each alternative | Done |
| Capture RML advantages and gaps | Done |
| Include source links for follow-up research | Done |
