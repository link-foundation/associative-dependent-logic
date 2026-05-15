# Product Feature Comparison

This document compares user-visible product, tooling, automation, and
library features for Relative Meta-Logic (RML) and the systems named in
[issue #22](https://github.com/link-foundation/relative-meta-logic/issues/22).
Core logical concepts are compared separately in
[CONCEPTS-COMPARISON.md](./CONCEPTS-COMPARISON.md), which also defines
the extended legend (`Kernel`, `Library`, `Encoding`, `Runtime`, `Host`,
`External`, `Prototype`, `Theory`, `Archive`). Foundation and AFP appear
as **libraries/archives** rather than as standalone prover kernels; see
the *Systems and artifacts* section in
[CONCEPTS-COMPARISON.md](./CONCEPTS-COMPARISON.md#systems-and-artifacts).

## Legend

| Mark | Meaning |
|------|---------|
| Yes | First-class or central capability. |
| Part | Partial, prototype, limited, indirect, or available by encoding. |
| Host | Inherited from the host proof assistant rather than implemented by the project itself. |
| Archive | Curated archive/library artifact rather than a standalone prover kernel. |
| N/A | Not applicable to the system's role. |
| No | No evidence of support or outside the system's design goal. |

The qualifiers from [CONCEPTS-COMPARISON.md](./CONCEPTS-COMPARISON.md)
(`Kernel`, `Library`, `Runtime`, `Encoding`, `External`, `Prototype`)
may appear as parenthetical refinements where they clarify whether a
feature is implemented in the host, exposed at runtime, or available as
a checked library.

## Authoring and Checking Workflow

| Feature | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|---------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Batch file checking | Yes: JS/Rust CLIs run `.lino` files | Yes | N/A | Yes: `helf file.elf` | Yes | Yes | Yes | Host | Yes: Isabelle sessions | Yes | Yes | Yes |
| Interactive REPL/top-level | Part: CLI queries only | Yes | N/A | No | Yes | Yes | Yes | Host | N/A | Yes | Yes | Yes: interactive mode |
| Query commands | Yes: `(? expr)` | Yes: `%query`, `%solve` | N/A | No | Yes | Yes | Yes | Host | Host | Yes: `search`, commands | Yes | Yes: theorem/prove commands |
| Incremental proof state | No | Part | N/A | No | Yes | Yes | Yes | Host | Archive | Yes | Logic-programming goal state | No |
| Dedicated IDE/editor support | No | Emacs mode | N/A | No | Isabelle/jEdit and PIDE | CoqIDE, Proof General, VS Code ecosystem | VS Code/LSP ecosystem | Host | Archive | Emacs/Proof General style support | Teyjus tooling | Vim syntax; online demo |
| Language server protocol support | No | No | N/A | No | PIDE rather than plain LSP | Ecosystem support | Yes: Lean language server | Host | Archive | No standard LSP | No standard LSP | No |
| Structured diagnostics | Part: parser/evaluator errors | Part | N/A | No: package notes poor errors | Yes | Yes | Yes | Host | Archive | Part | Part | Part |
| Module/import system | Part: file-level examples | Yes | N/A | Part | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes |
| Namespaces | No dedicated feature | Part | N/A | Part | Yes | Yes | Yes | Host | Archive | Part | Part | Part |
| Package/build ecosystem | Part: npm/cargo packages | Part | N/A | Cabal/Hackage | Isabelle sessions | Dune/opam/coq_makefile ecosystem | Lake | Lake | Isabelle releases/sessions | OPAM/source distribution | Teyjus/ELPI ecosystems | Python scripts/Docker |
| Versioned package distribution | Yes: npm/cargo project layout | Releases/source | N/A | Hackage | Isabelle releases | OPAM/packages | Lake/GitHub | GitHub/Lake | AFP releases | OPAM/source | Source/packages vary | Source/Docker |
| Documentation generation | Markdown docs | Part | N/A | Hackage README | Yes | Yes | Yes | Yes: generated docs/catalogue | Yes | Yes: `abella_doc` | Part | Manual/README |
| Literate proof documents | No | Part | N/A | No | Yes | Yes, via ecosystem | Yes, via ecosystem | Host | Archive | Part | No | No |
| Online/browser use | Yes: GitHub Pages playground (issue #163) | No | N/A | No | Limited ecosystem demos | jsCoq exists | Web examples exist | Host | Website docs | No | No | Yes: online demo referenced |
| Multi-implementation parity | Yes: JS and Rust | No | N/A | No: Haskell only | No | No | No | Host | Archive | No | Multiple implementations exist | No |
| Example command-line demos | Yes | Yes | N/A | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes |

## Proof Engineering and Automation

| Feature | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|---------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Tactic language | Part (Prototype): `reflexivity`, `symmetry`, `transitivity`, `rewrite`, `simplify`, `smt`, `atp`, `exact`, `induction` link tactics. Not as mature as Isar/Ltac/Lean. | Part: theorem prover/proof search | N/A | No | Yes: Isar/Eisbach/tools | Yes | Yes | Host | Archive | Yes | Logic-programming search | No tactics; automated proving |
| Simplifier | Part: evaluator and operators | Part | N/A | No | Yes | Yes | Yes | Host | Archive | Part | Part | No |
| Rewriting automation | Part: assigned-infix rewrite table and `rewrite` tactic | Part | N/A | No | Yes | Yes | Yes | Host | Archive | Part | Part | No |
| Built-in proof search | Part: query evaluation | Yes | N/A | No | Yes | Yes via tactics/plugins | Yes via tactics | Host | Archive | Yes | Yes | Yes: automata decision procedure |
| Search depth controls | No | Yes | N/A | No | Tool-dependent | Tactic-dependent | Tactic-dependent | Host | Archive | Yes | Yes | Domain-dependent |
| External ATP integration | Part (External): `(by atp …)` records results as trusted external nodes | No | N/A | No | Yes: Sledgehammer/ATPs/SMT | Plugins/tools | Ecosystem tools | Host | Archive | No | No | No |
| SMT integration | Part (External): `(by smt …)` SMT-LIB trusted bridge | No | N/A | No | Yes | Plugins/tools | Ecosystem tools | Host | Archive | No | No | No |
| Model/counterexample finding | No | No | N/A | No | Yes: Nitpick/Quickcheck | Plugins/tools | Ecosystem/tools | Host | Archive | No | No | Automata emptiness/domain feedback |
| Totality automation | Part (Prototype): `(total …)` declarations | Yes | N/A | No | Yes for recursive definitions | Yes | Yes | Host | Archive | Part | No | N/A |
| Termination automation | Part (Prototype): `is_terminating` / `(measure …)` | Yes | N/A | No | Yes | Yes | Yes | Host | Archive | Part | No | N/A |
| Coverage/productivity checks | Part: `(coverage <name>)` checks every `+input` slot against the slot's inductive constructors | Yes | N/A | No | Yes for datatypes/functions | Yes | Yes | Host | Archive | Part | No | N/A |
| Mode checking | Part (Prototype): `+input`/`-output`/`*either` mode declarations | Yes | N/A | No | No direct equivalent | Tactic/program mechanisms | Tactic/program mechanisms | Host | Archive | Part | Modes in implementations/ecosystem | N/A |
| Independent proof-replay checker | Part (Prototype): `rust/src/check.rs` and the JS counterpart re-verify recorded proof links without re-running the evaluator | Yes (via LF checking) | N/A | Yes for LF terms | Yes | Yes | Yes | Host | Archive | Yes | No standard independent replay | Domain-specific rerun |
| Program extraction | No | No | N/A | No | Yes: code generator | Yes: extraction | Yes: compiler/code generation | Host | Archive | No | Logic programs execute | No general extraction |
| Native executable programs | Yes: evaluator executes `.lino` queries | Logic programs | N/A | No | Generated code | Gallina extraction | Lean compiler | Host | Archive | Specification logic execution | Yes | Yes |
| Reflection/metaprogramming | Part: links can encode rules; self-evaluator layer | Part | N/A | No | Isabelle/ML | Ltac/ML/plugins | Lean metaprogramming/macros | Host | Archive | Part | Program-level | No |
| Custom syntax/macros | Part: grammar-light links | Fixity/operators | N/A | Limited Twelf subset | Yes | Yes | Yes | Host | Archive | Part | Yes | Yes: language syntax |
| Proof repair/refactoring tools | No | No | N/A | No | Ecosystem/editor support | Ecosystem support | Ecosystem support | Host | Archive | No | No | No |
| Generated proof/checking artifacts | Part (Prototype): proof links replayed by the proof-replay checker | Yes | N/A | LF checking result | Yes | Yes | Yes | Host | Archive | Yes | Part | Domain result |
| Replayable derivation trace | Part (Prototype): proof links | Yes for LF/proof terms | N/A | Part | Yes | Yes | Yes | Host | Archive | Yes | Part | Rerun script/procedure |

## Library and Domain Coverage

| Feature | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|---------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Stable large standard library | No | Examples/case studies | N/A | No | Yes | Yes | Yes plus mathlib | Yes: project library | Archive | Examples | Examples | Domain libraries |
| Mathematics library | No | No | N/A | No | Yes | Yes | Yes: mathlib | Part: mathematical logic | Archive | No | No | No |
| Formalized logic library | Part: examples/case studies | Examples | N/A | Examples | Yes | Yes | Yes | Yes (Library) | Archive | Examples | Examples | Domain-specific formulas |
| Programming language metatheory examples | Part | Yes | Encoding | Yes: examples | Yes | Yes | Yes | Part | Archive | Yes | Yes | No |
| Lambda calculus examples | Yes: dependent type examples | Yes | Encoding | Yes | Yes | Yes | Yes | Host/Library | Archive entries | Yes | Yes | No |
| Modal logic corpus | Part: encodable | Encoding | Encoding | Encoding | Encoding | Encoding | Encoding | Yes (Library) | Archive | Encoding | Encoding | No |
| Provability/interpretability logic corpus | Part: encodable | Encoding | Encoding | Encoding | Encoding | Encoding | Encoding | Yes (Library) | Archive | Encoding | Encoding | No |
| Set theory corpus | Part: encodable | Encoding | Encoding | Encoding | Yes | Library | Library | Yes (Library) | Archive | Encoding | Encoding | No |
| Arithmetic corpus | Part: evaluator arithmetic | Constraints/examples | Encoding | Encoding | Yes | Yes | Yes | Yes (Library) | Archive | Encoding | Encoding | Yes |
| Probabilistic examples | Yes (Runtime + Host) | No | No | No | Library possible | Library possible | Library possible | No | Archive entries possible | No | No | No |
| Fuzzy logic examples | Yes (Runtime + Host) | No | No | No | Library possible | Library possible | Library possible | No | Archive entries possible | No | No | No |
| Graphical model examples | Yes: Bayesian/Markov examples | No | No | No | Library possible | Library possible | Library possible | No | Archive entries possible | No | No | No |
| Paradox/self-reference examples | Yes (Runtime semantics): liar paradox examples | No | No | No | Encoding (with restrictions) | Encoding (with restrictions) | Encoding (with restrictions) | Part | Archive (Part) | Part | No | No |
| Automatic sequences/numeration systems | No | No | No | No | Encoding (with effort) | Encoding (with effort) | Encoding (with effort) | No | Archive entries possible | No | No | Yes |
| Automata libraries | No | No | No | No | Library possible | Library possible | Library possible | No | Archive entries possible | No | No | Yes |
| Scientific archive model | Part: case studies | Wiki/case studies | Papers | No | Yes | Yes | Yes | Docs/catalogue | Archive: journal-like archive | Examples/publications | Examples/book | Paper/manual |

## Distribution, Maintenance, and Integration

| Feature | RML | Twelf | Edinburgh LF | HELF | Isabelle | Coq/Rocq | Lean | Foundation | AFP | Abella | lambda Prolog | Pecan |
|---------|-----|-------|--------------|------|----------|----------|------|------------|-----|--------|---------------|-------|
| Public source repository | Yes | Yes | N/A/theory | Yes | Yes | Yes | Yes | Yes | Yes/downloads | Yes | Yes | Yes |
| Active language/runtime ecosystem | Node.js/Rust | Standard ML legacy | N/A | Haskell/Cabal | Isabelle/ML | OCaml/opam | Lean/Lake | Lean/Lake | Isabelle | OCaml/opam | Teyjus/OCaml ecosystem | Python/Docker |
| CI-friendly batch mode | Yes | Yes | N/A | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes |
| Docker support | Yes: `docker/Dockerfile.js`, `docker/Dockerfile.rust`, and `docker/docker-compose.yml` for both implementations, built in CI | No | N/A | No | Ecosystem possible | Ecosystem possible | Ecosystem possible | No | No | No | No | Yes |
| Package manager install | npm/cargo layout; package release dependent | Source/distribution | N/A | Hackage/Cabal | Isabelle distribution | OPAM/packages | Lake/elan | Lake/GitHub | AFP download/releases | OPAM | Depends on implementation | pip requirements/manual |
| Generated API/reference docs | Part | Part | N/A | Hackage page | Yes | Yes | Yes | Yes | Archive | Part | Part | Manual |
| Tutorials/walkthroughs | README/examples | Yes | Papers/docs | README/examples | Yes | Yes | Yes | Catalogue/docs | Archive entry docs | Yes | Manuals/examples | README/manual |
| Backward compatibility story | Part | Historical Twelf docs | Theory | Hackage package | Release-managed | Release-managed | Release-managed | Host constraints | Isabelle release coupling | Versioned releases | Implementation-dependent | Source-level |
| Human-readable examples | Yes: `.lino` examples | Yes | Papers | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes |
| Machine-readable examples/tests | Yes | Yes | N/A | Yes | Yes | Yes | Yes | Host | Archive | Yes | Yes | Yes |
| Cross-language implementation parity | Yes: JS/Rust implementations share behavior | No | N/A | No | No | No | No | Host | Archive | No | Multiple implementations differ | No |

## RML Product Roadmap Suggested by the Feature Matrix

| Priority | Gap | Why it matters | Candidate next step |
|----------|-----|----------------|---------------------|
| 1 | Discoverable user docs | Users should see positioning without reading issue case studies | Keep these top-level concept/feature comparison docs linked from README |
| 2 | Structured diagnostics | Failed queries and type checks need actionable explanations | Add opt-in trace output for parsing, assignment lookup, operator resolution, and type inference |
| 3 | First-class proof artifacts | RML currently records proof links and replays them, but they are not yet first-class machine-checked derivation objects | Promote proof links to derivation objects with documented replay rules |
| 4 | Module/import system | Larger examples need reusable files and namespaces | Add imports with cycle detection and explicit module names |
| 5 | Inductive definitions | Mature proof assistants rely on constructors, eliminators, recursion, and induction principles | Mature `(inductive …)`/`(coinductive …)` machinery: strict positivity, dependent patterns, productivity proofs |
| 6 | Automation | Competitors provide tactics, simplifiers, ATP/SMT bridges, or domain procedures | Mature the tactic link layer (`reflexivity`, `symmetry`, `transitivity`, `rewrite`, `simplify`, `smt`, `atp`, `exact`, `induction`) |
| 7 | Editor integration | Proof assistants win productivity through live feedback | Add JSON diagnostics first, then an LSP/VS Code extension |
| 8 | Library ecosystem | Users need reusable logic fragments, not only demos | Split examples into reusable libraries for classical, fuzzy, probabilistic, type-theoretic, and graph-model reasoning |

## Source Notes

| System | Source notes |
|--------|--------------|
| RML | This repository's [README.md](../README.md), [ARCHITECTURE.md](../ARCHITECTURE.md), JavaScript examples, Rust examples, and test suites describe the current user-facing behavior. Per-RML row evidence is collected in [`case-studies/issue-167/evidence/rml-feature-audit.md`](./case-studies/issue-167/evidence/rml-feature-audit.md). |
| Twelf and LF | Twelf LF, logic programming, and guide pages document LF representation, type checking, `%solve`, `%query`, modes, termination, coverage, totality, theorem proving, server, and Emacs support: <https://twelf.org/wiki/lf/>, <https://twelf.org/wiki/logic-programming/>, <https://www.cs.cmu.edu/~twelf/guide-1-4/twelf_toc.html>. |
| HELF | The Hackage package documents the `.elf` parser/typechecker, Twelf subset, Cabal installation, limitations, and examples: <https://hackage.haskell.org/package/helf>. |
| Isabelle | The official documentation index lists manuals and tooling for Isabelle2025-2, including datatypes, functions, code generation, Nitpick, Sledgehammer, Eisbach, Isabelle/Isar, system, and jEdit: <https://isabelle.in.tum.de/documentation.html>. |
| Coq/Rocq | The Rocq reference manual documents the core language, modules, universes, proof mode, tactics, inductives/coinductives, and related commands: <https://docs.rocq-prover.org/master/refman/>. |
| Lean | The Lean reference documents dependent type theory, kernel checking, tactics, simplifier, macros, modules, source files, and build tools: <https://lean-lang.org/doc/reference/latest/>. |
| Foundation | The Foundation README describes its Lean 4 mathematical logic library, generated catalogue/docs, logic zoo diagrams, and proof automation area: <https://github.com/FormalizedFormalLogic/Foundation>. |
| AFP | The AFP home page describes proof libraries, examples, larger developments, Isabelle checking, journal-style organization, and releases: <https://www.isa-afp.org/>. |
| Abella | The Abella site and reference guide document lambda-tree syntax, two-level logic, examples, OPAM/source installation, induction, coinduction, search, and tactics: <https://abella-prover.org/index.html>, <https://abella-prover.org/reference-guide.html>. |
| lambda Prolog | The Teyjus documentation describes lambda Prolog's higher-order hereditary Harrop foundation, higher-order programming, polymorphic typing, scoping, modules, abstract data types, and lambda terms as data: <https://teyjus.cs.umn.edu/old/language/teyjus_1.html>. It is not HOL in the Isabelle/HOL theorem-proving sense. |
| Pecan | The Pecan repository documents batch and interactive modes, Docker support, numeration systems, automatic words, Büchi automata, examples, and Vim syntax support: <https://github.com/ReedOei/Pecan>. The paper gives the automatic-sequence theorem-proving context: <https://arxiv.org/abs/2102.01727>. |
