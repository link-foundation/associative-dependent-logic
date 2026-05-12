# Core Concept Comparison

This document compares Relative Meta-Logic (RML) with the systems named in
[issue #22](https://github.com/link-foundation/relative-meta-logic/issues/22)
by core logical and metatheoretic concepts. Product and workflow capabilities
are compared separately in [FEATURE-COMPARISON.md](./FEATURE-COMPARISON.md).

The goal is positioning, not a claim that every system has the same design
target. Several entries are logical frameworks, some are full proof
assistants, some are libraries inside a host prover, and Pecan is a
domain-specific automated prover.

> **How to read this matrix.** A `Yes` for RML usually means "available in
> the current evaluator/runtime", not necessarily "defined inside `.lino`
> with a mechanised metatheory". Wherever a row would otherwise be
> ambiguous, the cell uses the qualifiers from the legend below
> (`Kernel`, `Library`, `Encoding`, `Runtime`, `Host`, `External`,
> `Prototype`, `Theory`) so that readers can tell **where** and **in what
> sense** a concept is supported. The matrix is a positioning aid, not a
> formal metatheoretic equivalence claim.

## Legend

| Mark | Meaning |
|------|---------|
| Yes | First-class or central capability. |
| Part | Partial, prototype, limited, indirect, or available by encoding. |
| No | No evidence of support or outside the system's design goal. |
| N/A | Not applicable to the system's role. |
| Kernel | Built into the trusted kernel or meta-kernel of the system. |
| Library | Available through a checked library or standard development, not kernel-native. |
| Encoding | Can be represented as an object logic or encoded structure. |
| Runtime | Selected by runtime configuration rather than fixed kernel semantics. |
| Host | Implemented by the host implementation rather than defined in the object language. |
| External | Delegated to an external trusted tool or procedure. |
| Prototype | Implemented partially or experimentally; not mature. |
| Theory | Theoretical framework, not a standalone implementation. |
| Archive | Curated archive/library artifact rather than a standalone prover kernel. |

The qualifiers can be combined with `Yes` / `Part` to make the cell
self-describing, for example:

- `Yes (Kernel)` — supported and built into the trusted kernel.
- `Yes (Library)` — supported, but as a checked library on top of a
  fixed kernel.
- `Yes (Runtime + Host)` — supported, but the semantics come from a
  runtime configuration interpreted by the host implementation.
- `Part (Prototype)` — partially implemented or not yet mature.
- `Part (Encoding)` — available only by encoding inside the system.

## Systems and artifacts

The matrix mixes standalone systems with libraries and archives that ride
on top of a host system. The distinction is explicit below.

### Provers, frameworks, and languages

| System | Primary role |
|--------|--------------|
| RML | Link-based probabilistic and many-valued meta-logic with JS and Rust implementations. |
| Twelf | LF implementation with type checking, logic programming, and metatheorem checking. |
| Edinburgh LF | Dependently typed logical framework for representing deductive systems. |
| HELF | Haskell implementation of an LF/Twelf subset for parsing and type checking `.elf` files. |
| Isabelle | Generic interactive theorem prover, especially Isabelle/Pure and Isabelle/HOL. |
| Coq/Rocq | Interactive theorem prover based on the Calculus of Inductive Constructions. |
| Lean | Interactive theorem prover and programming language based on dependent type theory. |
| Abella | Interactive prover for lambda-tree syntax and two-level logic reasoning. |
| lambda Prolog | Higher-order hereditary Harrop logic programming language with lambda terms as data and higher-order abstract syntax. **Not HOL in the Isabelle/HOL sense.** |
| Pecan | Automated theorem prover for automatic sequences and Büchi automata. Strong on its decidable domain, intentionally not a general-purpose proof assistant. |

### Libraries and archives

| Artifact | Primary role | Host system |
|----------|--------------|-------------|
| Foundation | Lean 4 library formalizing mathematical logic (propositional, FOL, modal, provability, interpretability, arithmetic, set theory). | Lean |
| AFP | Archive of Formal Proofs — curated, mechanically-checked archive of Isabelle developments organized like a scientific journal. | Isabelle |

For Foundation and AFP rows, the matrix uses `Host`, `Library`, or
`Archive` instead of plain `Yes` whenever the capability is inherited
from Lean or Isabelle rather than defined by the artifact itself.

## RML status note

RML currently combines a host-implemented evaluator/kernel with
`.lino`-level encodings and runtime configuration. A `Yes` for RML often
means "available in the current evaluator/runtime", not necessarily
"defined inside `.lino` with a mechanised metatheory". The matrix uses
the `Runtime`, `Host`, `Prototype`, `Library`, and `External` qualifiers
to make this explicit.

RML also exposes several distinct equality layers — structural equality,
assigned equality (the rewrite/notation table), numeric equality,
beta/definitional convertibility, and object-level equality inside
encoded systems. Where a row would otherwise conflate them, the row
makes the intended layer explicit.

## Foundations and Representation

| Core concept | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|--------------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| General meta-logic for object logics | Yes (Runtime + Host): link substrate can encode many logics | Yes (Kernel) | Yes (Theory/Kernel) | Part (Prototype) | Yes (Pure framework) | Part (Encoding): can encode logics in CIC | Part (Encoding): can encode logics in DTT | Yes (Library): logic formalizations in Lean | Archive: Isabelle entries | Yes (Kernel) | Part: specification language | No |
| Uniform representation of syntax and judgments | Yes (Runtime): links | Yes (LF terms) | Yes (LF terms) | Yes (LF terms) | Part: Pure/HOL terms | Yes (CIC terms) | Yes (dependent terms) | Host | Archive | Yes (two-level terms) | Yes (lambda terms) | Part: formulas and automata |
| One syntactic category for terms/propositions/proofs | Part (Prototype): links are uniform; proof terms are prototype only | Yes (LF style) | Yes (LF style) | Yes (LF subset) | Part: terms/propositions distinct in HOL layer | Yes (Kernel) | Yes (Kernel) | Host | Archive | Part | Part | No |
| Explicit object-language encodings | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Library | Archive | Yes | Yes | Part |
| Adequacy-oriented encodings | Part | Yes | Yes | Part | Yes (by proof development) | Yes (by proof development) | Yes (by proof development) | Host | Archive | Yes | Part | No |
| Links or tuples as primitive notation | Yes | No | No | No | No | No | No | No | No | No | No | No |
| Named constants and declarations | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes |
| Definitions as first-class source entries | Yes | Yes | Theory | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes |
| Structural equality over expressions | Yes (Host): `is_structurally_same` over node graphs | Yes (type theory equality/convertibility) | Yes | Yes | Yes | Yes | Yes | Host | Archive | Part | Part | Domain-specific equality |
| Equality layers distinguished (structural / assigned / numeric / definitional / object-level) | Part: several equality mechanisms exist (structural, assigned/infix rewrite, numeric, beta/definitional convertibility); their proof-theoretic roles are being made explicit. | Yes: definitional equality / type equality are part of LF checking. | Yes | Yes | Yes (logic-dependent meta vs object equality) | Yes (definitional vs propositional equality) | Yes (definitional vs propositional equality) | Host | Archive | Part | Part | Domain-specific equality |
| Numeric truth values in the core | Yes (Runtime + Host): numeric truth values are native to the evaluator; numeric domains and arithmetic are currently host-implemented | No | No | No | No | No | No | No | No | No | No | Part: arithmetic domains |
| Configurable semantic range | Yes (Runtime + Host): range `[0,1]` or `[-1,1]` is configurable; interval endpoints, midpoint, order, clamping, and arithmetic are host-defined | No | No | No | No | No | No | No | No | No | No | No |
| Configurable valence | Yes (Runtime + Host): unary, Boolean, N-valued, continuous; quantization arithmetic is host-implemented | No | No | No | No | No | No | No | No | No | No | No |
| Self-reference accepted by default | Yes (Runtime semantics): self-referential/circular forms can be accepted and evaluated through many-valued/paradox-tolerant semantics. **Not** a classical consistency or mechanised soundness claim. | No | No | No | No | No | No | Host constraints | Archive | Part (coinductive reasoning only) | No | No |
| Circular definitions as ordinary data | Part | No | No | No | Guarded/fixed-point mechanisms | Guarded/termination rules | Guarded/termination rules | Host | Archive | Coinductive predicates | Recursive programs with restrictions | Automata fixed points |

## Type Theory and Binding

| Core concept | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|--------------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Dependent types | Part (Prototype): prototype type layer | Yes (Kernel) | Yes (Kernel) | Yes | No: HOL is simply typed; Pure is higher-order | Yes (Kernel) | Yes (Kernel) | Host | Archive | No: simply typed reasoning/spec logics | No: polymorphic/simple typed | No |
| Dependent products / Pi-types | Part (Prototype) | Yes | Yes | Yes | Part: meta-level universal quantification | Yes | Yes | Host | Archive | No | No | No |
| Lambda abstraction | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | No |
| Function application | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | No |
| Beta reduction | Part (Prototype): beta for link lambdas | Yes | Yes | Yes | Yes | Yes | Yes | Host | Archive | Part | Yes | No |
| Definitional equality / conversion | Yes: beta convertibility; eta opt-in API | Yes | Yes | Yes | Yes | Yes | Yes | Host | Archive | Part | Part: beta conversion | No |
| Full normalization | Part (Prototype): `whnf`, `nf`, and `(normal-form …)` exist for the typed lambda fragment; the whole evaluator is not a general normalizer for all RML semantics. | Yes for LF canonical forms | Yes (Theory) | Part | Yes where defined by logic/tools | Yes | Yes | Host | Archive | Part | Part | No |
| Universe hierarchy | Part (Prototype): `(Type 0)`, `(Type 1)` | No: LF has kinds/types | No | No | No in HOL; object theories possible | Yes (Kernel) | Yes (Kernel) | Host | Archive | No | No | No |
| Sorts / kinds | Part | Yes | Yes | Yes | Yes: types/classes/logics | Yes | Yes | Host | Archive | Yes: type declarations | Yes | Domain sorts |
| Type annotations | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes: domain declarations |
| Type inference | Part | Part: reconstruction | Theory | No reconstruction | Yes | Yes | Yes | Host | Archive | Part | Yes | Part |
| Type checking | Part (Prototype) | Yes | Theory | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes: domain/formula checking |
| Higher-order abstract syntax | Part (Encoding): encodable as links | Yes | Yes | Yes | Part | Part | Part | Host | Archive | Yes | Yes | No |
| Lambda-tree syntax | Part | Yes/related | Yes/related | Yes/related | Encoding | Encoding | Encoding | Host | Archive | Yes | Yes | No |
| Binding-aware substitution | Part | Yes | Yes | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | No |
| Variable freshness support | Part | Part | Part | Part | Yes (Library) | Yes (Library) | Yes (Library) | Host/Library | Archive/Library | Yes: nabla/generic judgments | Part | No |
| Polymorphism | No dedicated system | Part | Part | Part | Yes | Yes | Yes | Host | Archive | Schematic polymorphism | Yes | Domain-specific |
| Type classes | No | No | No | No | Yes (Library/notation) | Yes (Library/notation) | Yes (Library/notation) | Host | Archive | No | No | No |
| Inductive families | Part (Prototype): `(inductive …)` declarations and generated recursors exist; strict positivity and mature dependent pattern machinery remain limited. | Encoding (LF families) | Encoding | Encoding | Yes (Kernel) | Yes (Kernel) | Yes (Kernel) | Host | Archive | Yes: inductive definitions | Encoding | No |
| Coinductive types/predicates | Part (Prototype): `(coinductive …)` declarations and generated corecursors exist, with productivity checks; proof theory and tooling remain immature. | Limited/encoded | No | No | Yes (Kernel) | Yes (Kernel) | Yes (Kernel) | Host | Archive | Yes: coinductive predicates | Encoding | Automata over infinite words |

## Logic and Semantics

| Core concept | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|--------------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Classical logic | Yes (Runtime): aggregator profile selectable at runtime | Encoding | Encoding | Encoding | Yes — Isabelle/HOL is classical HOL; Isabelle/Pure remains the meta-logic | Library/Axiom — classical reasoning available; kernel remains CIC | Library/Axiom — classical reasoning available; kernel remains DTT | Yes (Library) | Archive | Encoding | Encoding | Part: decidable fragments |
| Intuitionistic logic | Part | Yes (LF foundation) | Yes | Yes | Encoding / FOL library | Yes (Kernel) | Yes (Kernel) | Yes (Library) | Archive | Yes | Yes | No |
| Constructive type theory | Part | LF-style | LF-style | LF-style | Object theories possible | Yes | Yes | Host | Archive | No | No | No |
| Higher-order logic | Part (Encoding) | Encoding | Encoding | Encoding | Yes | Encoding (in CIC) | Encoding (in DTT) | Host/Library | Archive | Part | Yes: higher-order programming logic (not Isabelle/HOL-style theorem proving) | No |
| First-order logic | Part | Encoding | Encoding | Encoding | Yes | Encoding | Encoding | Yes (Library) | Archive | Encoding | Encoding | Part: arithmetic/word formulas |
| Modal logic | Part (Encoding): link encoding | Encoding | Encoding | Encoding | Encoding | Encoding | Encoding | Yes (Library) | Archive | Encoding | Encoding | No |
| Provability logic | Part (Encoding) | Encoding | Encoding | Encoding | Encoding | Encoding | Encoding | Yes (Library) | Archive | Encoding | Encoding | No |
| Interpretability logic | Part (Encoding) | Encoding | Encoding | Encoding | Encoding | Encoding | Encoding | Yes (Library) | Archive | Encoding | Encoding | No |
| Set theory | Part (Encoding) | Encoding | Encoding | Encoding | Yes (Library): ZF object logic | Encoding/Library | Encoding/Library | Yes (Library) | Archive | Encoding | Encoding | No |
| Many-valued logic | Yes (Runtime + Host): native valence/range; quantization and aggregators are host-implemented | No | No | No | Encoding | Encoding | Encoding | Part | Archive (Part) | Encoding | Encoding | No |
| Fuzzy logic | Yes (Runtime + Host): continuous truth values and aggregators are supported; the numeric basis is host-defined | No | No | No | Encoding | Encoding | Encoding | No | Archive (Part) | Encoding | No | No |
| Probabilistic truth values | Yes (Runtime + Host): numeric probabilities are native to the evaluator; arithmetic is host-defined | No | No | No | Encoding | Encoding | Encoding | No | Archive (Part) | Encoding | No | No |
| Probabilistic operators | Yes (Runtime + Host): product and probabilistic-sum aggregators exist; their arithmetic semantics are host-defined | No | No | No | Encoding | Encoding | Encoding | No | Archive (Part) | Encoding | No | No |
| Redefinable logical operators | Yes (Runtime): operator table and aggregators are runtime-configurable — see [CONFIGURABILITY.md](./CONFIGURABILITY.md) | Encoding: object connectives are encoded as LF constants/families; LF core is fixed | Encoding | Limited | Encoding/Library — object logics and notation can be defined; Pure meta-logic is fixed | Library/Notation — new connectives and notation can be defined; CIC kernel is fixed | Library/Notation — new connectives, notation, typeclass instances, and macros can be defined; kernel rules are fixed | Host | Archive | Part | Yes (Library): higher-order programming connectives | Part |
| Paraconsistent semantics | Yes (Runtime semantics): paradox-tolerant midpoint behavior | No | No | No | Encoding | Encoding (with axioms/libraries) | Encoding (with axioms/libraries) | Part: logic zoo focus | Archive (Part) | Encoding | No | No |
| Liar/self-reference examples | Yes (Runtime semantics) | No | No | No | Encoding (with care) | Encoding (with restrictions) | Encoding (with restrictions) | Part | Archive (Part) | Part | No | No |
| Arithmetic reasoning | Part (Host): decimal arithmetic in evaluator | Constraint domains possible | Encoding | Encoding | Yes | Yes | Yes | Yes (Library) | Archive | Encoding | Encoding | Yes: numeration systems |
| Automated sequence reasoning | No | No | No | No | Encoding (with effort) | Encoding (with effort) | Encoding (with effort) | No | Archive entries possible | No | No | Yes |
| Automata-theoretic semantics | No | No | No | No | External/Library | External/Library | External/Library | No | Archive (Part) | No | No | Yes |
| Decidable complete target fragment | No: general meta-logic evaluator | No | No | No | Some procedures | Some procedures/tactics | Some procedures/tactics | Host | Archive | No | No | Yes |

## Metatheory and Proof Objects

| Core concept | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|--------------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Machine-checked proof terms | Part (Prototype): proof links are recorded and replayed, but they are not yet first-class derivation objects | Yes | Theory | Yes (LF type checking) | Yes | Yes | Yes | Host | Archive | Yes | Part: proof/search witnesses | No general proof terms |
| Small trusted kernel/checker | Part (Prototype): small evaluator plus an independent proof-replay checker (`rust/src/check.rs` / `js`) and prototype type layer | Yes (LF checker) | Theory | Yes (LF checker) | Yes (LCF-style kernel) | Yes | Yes | Host | Archive | Yes | No: language implementation | Domain-specific automata checker |
| Metatheorem checking about encoded systems | Part (Prototype): `check_metatheorems` in self/metatheorem.lino layer | Yes | Theory goal | No | Yes | Yes | Yes | Yes (Library) | Archive | Yes | Part | No |
| Totality checking | Part (Prototype): `(total …)` declarations are recognised and checked; clarify which checks are host-enforced and which are encoded as data | Yes | No | No | Part: function package/tools | Yes (Kernel) | Yes (Kernel) | Host | Archive | Part | No | N/A |
| Termination checking | Part (Prototype): `is_terminating` with `(measure …)` forms | Yes | No | No | Yes for recursive definitions/tools | Yes (Kernel) | Yes (Kernel) | Host | Archive | Part | No | N/A |
| Coverage checking | Part (Prototype): `(coverage …)` declarations checked against the slot's inductive constructors | Yes | No | No | Part | Yes (Kernel) | Yes (Kernel) | Host | Archive | Part | No | N/A |
| Mode checking | Part (Prototype): `+input`/`-output`/`*either` mode declarations are enforced | Yes | No | No | No direct equivalent | Tactic/program mechanisms | Tactic/program mechanisms | Host | Archive | Part | Modes in implementations/ecosystem | N/A |
| World declarations / regular worlds | No | Yes | No | No | No | No | No | No | No | No | No | N/A |
| Proof search | Part (Prototype): query evaluation | Yes | No | No | Yes | Yes (via tactics/plugins) | Yes (via tactics) | Host | Archive | Yes | Yes | Yes: decision procedure |
| Tactic-level proof construction | Part (Prototype): tactic links exist (`reflexivity`, `symmetry`, `transitivity`, `rewrite`, `simplify`, `smt`, `atp`, `exact`, `induction`) but the tactic layer is not mature like Lean/Rocq/Isabelle | No / N/A: proof search and metatheorem checking exist, but not tactic-level interactive proof construction in the Lean/Rocq/Isabelle style | No | No | Yes | Yes | Yes | Host | Archive | Yes | Logic-programming search | No |
| Rewriting as proof principle | Part: assigned-infix rewrite table and `rewrite` tactic | Encoding | Encoding | Encoding | Yes | Yes | Yes | Host | Archive | Part | Part | No |
| Countermodel/counterexample support | No | No | No | No | Yes (Library): Nitpick/Quickcheck | Plugins/tools | Ecosystem/tools | Host | Archive | No | No | Automata emptiness gives domain feedback |
| Executable specifications | Yes | Yes | No | Part: typecheck examples | Part | Yes (Gallina programs) | Yes (functional programs) | Host | Archive | Yes: executable specification logic | Yes | Yes |
| Proof-producing evaluator | Part (Prototype): selected queries can produce proof links; semantic computation still depends on trusted evaluator primitives | Yes for logic programming/type checking | Theory | Type-checking only | Yes | Yes | Yes | Host | Archive | Yes | Part | No general derivation object |
| Independent proof replay | Part (Prototype): the independent proof-replay checker (`rust/src/check.rs`, `js` counterpart) verifies recorded proof links without re-running the evaluator; not yet a fully mechanised metatheory | Yes (via LF checking) | Theory | Yes for LF terms | Yes | Yes | Yes | Host | Archive | Yes | No standard independent replay | Domain-specific rerun |
| Library-scale theorem reuse | No | Examples only | Theory | No | Yes | Yes | Yes | Yes (Library) | Archive | Examples | Examples | Domain libraries |
| Soundness story documented | Part: see [SOUNDNESS.md](./SOUNDNESS.md) | Yes (through LF/Twelf literature) | Yes | Part | Yes | Yes | Yes | Host | Archive | Yes | Yes | Domain-specific/paper |
| Proof irrelevance / propositions | No dedicated support | No general universe feature | No | No | Logic-dependent | Yes | Yes | Host | Archive | No | No | No |
| Reflection/metatheory inside the system | Part: links can encode rules; self-evaluator layer in `lib/self/` | Part | Theory | No | Isabelle/ML and object encodings | Ltac/MetaCoq/plugins | Lean metaprogramming | Host | Archive | Part | Program-level | No |
| External certification bridge | Part (External): SMT and ATP bridges exist (`(by smt …)` / `(by atp …)`), but successful decisions are recorded as **trusted external nodes** rather than independently replayed proof certificates | No | N/A | No | Can import/export through ecosystem | Plugins/tools | Ecosystem/tools | Host | Archive | No | No | No |

## RML Positioning From the Concept Matrix

| Area | RML advantage | RML gap |
|------|---------------|---------|
| Semantic flexibility | Native many-valued, probabilistic, fuzzy, and paradox-tolerant evaluation, all runtime-configurable. | Numeric semantics are currently host-implemented; no fully mechanised many-valued metatheory in `.lino`. |
| Representation | Links give one low-friction substrate for terms, propositions, probabilities, graph structures, and prototype type constructs. | No mature elaborator, module system, or binding discipline comparable to LF/DTT systems. |
| Logic diversity | Operators, aggregators, and truth ranges can be changed at runtime. | Encoded object logics lack a mature machine-checked metatheory infrastructure. |
| Type theory | Universe, Pi, lambda, application, type-query experiments; `(total …)`, `(coverage …)`, `(measure …)`/termination, and mode-checking forms exist; `whnf`/`nf`/`(normal-form …)` cover the typed lambda fragment; `(inductive …)`/`(coinductive …)` declarations with generated (co)recursors exist as prototype machinery. | No full normalization for the entire evaluator; strict positivity, dependent-pattern machinery, and proof-theory tooling for (co)inductive families remain limited. |
| Automation | Query evaluation, tactic links (`reflexivity`, `symmetry`, `transitivity`, `rewrite`, `simplify`, `smt`, `atp`, `exact`, `induction`), and an SMT/ATP bridge are small and easy to inspect. | SMT and ATP results are recorded as trusted external nodes rather than independently certified proofs; no complete domain decision procedure. |
| Proof artifacts | An independent proof-replay checker (`rust/src/check.rs` / `js` counterpart) re-verifies recorded proof links without re-running the evaluator. | Proof links are not yet first-class machine-checked derivation objects; the metatheory of replay rules is documented but not fully mechanised. |
| Ecosystem | JS/Rust parity, concise implementation surface, and a documented case-study trail. | No large library corpus comparable to AFP, mathlib, Rocq libraries, or Foundation. |

## Source Notes

| System | Source notes |
|--------|--------------|
| RML | This repository's [README.md](../README.md), [ARCHITECTURE.md](../ARCHITECTURE.md), [KERNEL.md](./KERNEL.md), [SOUNDNESS.md](./SOUNDNESS.md), [METATHEOREMS.md](./METATHEOREMS.md), [CONFIGURABILITY.md](./CONFIGURABILITY.md), and the `rust/`, `js/`, `lib/`, `examples/`, and test directories describe the current syntax, evaluator, truth ranges, valence, operators, type features, tactic links, ATP/SMT bridges, proof-replay checker, and self-evaluator metatheorem layer. Evidence for each RML row is collected in [`case-studies/issue-167/evidence/rml-feature-audit.md`](./case-studies/issue-167/evidence/rml-feature-audit.md). |
| Twelf and LF | The Twelf LF page describes LF as a dependently typed lambda calculus for representing deductive systems and lists Twelf's LF checker, logic programming language, and metatheorem checker: <https://twelf.org/wiki/lf/>. The Twelf logic programming page describes `%solve`, `%query`, tabled queries, and dependently typed higher-order logic programming: <https://twelf.org/wiki/logic-programming/>. The Twelf guide lists reconstruction, modes, termination, coverage, totality, theorem prover, ML interface, server, and Emacs interface chapters: <https://www.cs.cmu.edu/~twelf/guide-1-4/twelf_toc.html>. |
| HELF | Hackage describes HELF as a Haskell implementation of LF that parses and typechecks Twelf-style `.elf` files, implements a subset of Twelf, and omits type reconstruction/unification: <https://hackage.haskell.org/package/helf>. |
| Isabelle | The Isabelle documentation index lists Isabelle2025-2 manuals for locales, classes, datatypes, functions, code generation, Nitpick, Sledgehammer, Eisbach, Isabelle/Isar, implementation, system, and jEdit: <https://isabelle.in.tum.de/documentation.html>. |
| Coq/Rocq | The Rocq reference manual documents core language constructs, conversion, typing rules, inductive/coinductive types, modules, universes, proof mode, tactics, and extraction-related material: <https://docs.rocq-prover.org/master/refman/>. |
| Lean | The Lean reference describes Lean as an interactive theorem prover based on dependent type theory with a minimal kernel, tactics, simplifier, macros, modules, and build tools: <https://lean-lang.org/doc/reference/latest/>. |
| Foundation | The Foundation README describes a Lean 4 mathematical logic library covering propositional, first-order, modal, provability, interpretability, arithmetic, set theory, proof automation, and logic zoo material: <https://github.com/FormalizedFormalLogic/Foundation>. |
| AFP | The Archive of Formal Proofs describes itself as proof libraries, examples, and larger scientific developments mechanically checked by Isabelle and organized like a scientific journal: <https://www.isa-afp.org/>. |
| Abella | The Abella site describes an interactive theorem prover based on lambda-tree syntax and two-level logic for reasoning about specifications with binding: <https://abella-prover.org/index.html>. The reference guide documents induction, coinduction, search, apply, and other tactics: <https://abella-prover.org/reference-guide.html>. |
| lambda Prolog | The Teyjus documentation describes lambda Prolog as a higher-order hereditary Harrop logic programming language with higher-order programming, polymorphic typing, scoping, modules, abstract data types, and lambda terms as data: <https://teyjus.cs.umn.edu/old/language/teyjus_1.html>. It is not HOL in the Isabelle/HOL theorem-proving sense. |
| Pecan | The Pecan repository describes it as an automated theorem prover for Büchi automata, numeration systems, and automatic words, with batch and interactive modes: <https://github.com/ReedOei/Pecan>. The Pecan paper describes automated theorem proving for automatic sequences using Büchi automata: <https://arxiv.org/abs/2102.01727>. |
