# Case Study: Universal CST converters between RML/`.lino` and Rust, JavaScript, Lean, Rocq

**Issue:** [#138 — We need to make sure we have relative meta logic converters to and from Rust/JavaScript, and also Lean, Rocq](https://github.com/link-foundation/relative-meta-logic/issues/138)
**Pull request:** [#169](https://github.com/link-foundation/relative-meta-logic/pull/169)
**Sibling artefacts in this folder:** [`requirements.md`](./requirements.md), [`existing-tools.md`](./existing-tools.md), [`solution-plans.md`](./solution-plans.md), [`cst-model.md`](./cst-model.md), [`acceptance-tests.md`](./acceptance-tests.md), [`risks-and-open-questions.md`](./risks-and-open-questions.md).

## Table of contents

1. [Executive summary](#executive-summary)
2. [Goal restated and scoped](#goal-restated-and-scoped)
3. [Where RML stands today](#where-rml-stands-today)
4. [Why the four target languages are hard](#why-the-four-target-languages-are-hard)
5. [Key design decisions](#key-design-decisions)
6. [Proposed conversion architecture](#proposed-conversion-architecture)
7. [Per-language conversion plan](#per-language-conversion-plan)
8. [Phased roadmap and recommended issue split](#phased-roadmap-and-recommended-issue-split)
9. [Test strategy](#test-strategy)
10. [Existing components and libraries that help](#existing-components-and-libraries-that-help)
11. [Open questions](#open-questions)
12. [References](#references)

---

## Executive summary

Issue #138 asks for **lossless concrete-syntax-tree (CST) converters** between RML's `.lino` notation and four host languages — Rust, JavaScript, Lean 4 and Rocq — so that `.lino` can serve as a **universal intermediate CST**. The conversion must round-trip without data loss, preserve comments, names and (where it matters) whitespace, and cover enough of each language's concepts that translation is unambiguous.

This is a substantially larger ambition than the converters that already exist in RML today: `rml export lean`, `rml export rocq` and `rml extract js|rust` are one-way, narrow-fragment, *AST-level* exporters (see [`docs/LEAN_EXPORT.md`](../../LEAN_EXPORT.md), [`docs/ROCQ-EXPORT.md`](../../ROCQ-EXPORT.md), [`README.md`](../../../README.md#program-extraction-issue-f7)). Filling the gap to a true universal CST converter is roughly the size of the J-EPIC parity programme tracked under [issue #95](https://github.com/link-foundation/relative-meta-logic/issues/95): it is a multi-phase, multi-issue effort.

The recommended approach is to:

1. Adopt a **trivia-bearing concrete-syntax tree** for `.lino`, modelled on rust-analyzer's rowan CST and Roslyn/libsyntax (see [Easy Lossless Trees with Nom and Rowan](https://blog.kiranshila.com/post/easy_cst), [rust-analyzer syntax docs](https://rust-lang.github.io/rust-analyzer/syntax/index.html)). Whitespace, comments and original token spelling are first-class.
2. Encode each host language as a small set of **CST shapes in `.lino`** — one `.lino` link family per host-language production — so that round-trip is purely a tree mapping, not a guess.
3. Use **existing, vendor-blessed parsers** for each host language (`syn`/`proc-macro2` for Rust, `swc`/`recast` for JavaScript, `Lean.Syntax` via the Lean toolchain for Lean 4, `coq-lsp` `--record_comments` and the Rocq vernac parser for Rocq) rather than reimplementing them. RML owns only the `.lino` ↔ host-CST mapping.
4. Ship the work as an epic split into 8 phases (A–H), each landing a vertical slice (one direction, one language, one test set), with bootstrap and round-trip tests gating CI.

This case study captures the data we collected, lists every requirement extracted from issue #138, names existing libraries that solve each part, and proposes one or more solution plans per requirement. It does not itself land the converters; their implementation is broken out into the per-language phase issues recommended in [§ Phased roadmap](#phased-roadmap-and-recommended-issue-split).

---

## Goal restated and scoped

Quoting issue #138 directly:

> So it is possible to convert between lino notation and other languages, we also should use CST, so we should be able to encode comments, every variable and other name, and even whitespace if needed.
>
> So for example it should be possible to convert lean to .lino (relative meta logic) and after that convert it to JavaScript.
>
> And from JavaScript to .lino, to Rocq and so on. So .lino with relative meta logic dialect will become universal intermediate CST (not AST) language.
>
> Meaning conversion should not result in data loss, and should be as precise as possible.
>
> And relative meta logic should have enough concepts to strictly without ambiguity encode each concept of Rocq/Lean/Rust/JavaScript.

Concretely the goal is:

| Goal | Scope |
|------|-------|
| **G1** | A `.lino` CST dialect (the "relative meta-logic CST dialect") that can encode every syntactic concept of each target language. |
| **G2** | Bidirectional converters between `.lino` CST and the host language CST for Rust, JavaScript, Lean 4 and Rocq. |
| **G3** | Round-trip lossless: `host → .lino → host` equals the original source byte-for-byte (modulo a documented canonicalisation list). |
| **G4** | Cross-language transpilation: `host_A → .lino → host_B` produces compilable source in `host_B` for the **shared semantic fragment** of the two languages, and explicit, structured `unrepresentable` markers elsewhere. |
| **G5** | Test coverage proportional to the surface area: each production has at least one positive and one negative round-trip test. |
| **G6** | Documentation of how to add a fifth language, so the universal-CST claim is supported by the ease of extension. |

This case study scopes G1–G6 and produces a plan for each. Item G3 (byte-for-byte) is the hardest acceptance criterion; we discuss in [§ Key design decisions](#key-design-decisions) what minimal canonicalisation we recommend allowing.

---

## Where RML stands today

The current repository already has *parts* of this story:

| Capability | Status | Location |
|------------|--------|----------|
| LiNo as the surface notation for RML | Yes | [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) |
| AST-level export from typed `.lino` fragment to Lean 4 | Yes (one-way) | [`docs/LEAN_EXPORT.md`](../../LEAN_EXPORT.md), [`js/src/lean-export.mjs`](../../../js/src/lean-export.mjs), [`rust/src/lean_export.rs`](../../../rust/src/lean_export.rs) |
| AST-level export from typed `.lino` fragment to Rocq | Yes (one-way) | [`docs/ROCQ-EXPORT.md`](../../ROCQ-EXPORT.md), [`js/src/rml-rocq.mjs`](../../../js/src/rml-rocq.mjs), [`rust/src/rocq.rs`](../../../rust/src/rocq.rs) |
| AST-level extraction from typed `.lino` fragment to JavaScript and Rust | Yes (one-way, λ-only) | [`README.md`](../../../README.md#program-extraction-issue-f7), [`js/src/rml-links.mjs`](../../../js/src/rml-links.mjs) (`Program extraction (issue #66)`), [`rust/src/lib.rs`](../../../rust/src/lib.rs) (`Program extraction (issue #66)`) |
| Comment preservation in `.lino` | **No** — `#`-comments are stripped during tokenisation per [`ARCHITECTURE.md` § Stage 2](../../../ARCHITECTURE.md#stage-2-tokenization-and-ast-construction) |
| Whitespace preservation in `.lino` | **No** — only structural `(` and `)` survive |
| Identifier-collision detection on export | Yes | [`docs/ROCQ-EXPORT.md` § Supported subset](../../ROCQ-EXPORT.md#supported-subset) |
| Import direction (host → `.lino`) | **No** for all four targets |
| Cross-language transpilation through `.lino` | **No** |

So we have one-way **AST** exporters for a typed sub-fragment, no importers, and no CST infrastructure. The four importers plus the CST infrastructure are what issue #138 asks us to add.

The relative size of the new work, against the parity epic in [issue #95](https://github.com/link-foundation/relative-meta-logic/issues/95), is roughly:

- **Smaller** than D-phase (typed kernel maturation) — the type theory is reused.
- **Comparable** to F-phase (bridges to mature provers and ATP/SMT) — same shape (one issue per target).
- **Larger** than F1/F2 (Lean/Rocq export) alone, because importers and round-trip tests are harder than emitters.

---

## Why the four target languages are hard

Each target has its own pitfalls; treating them uniformly would be a mistake.

### Rust

- The official tokeniser, `proc_macro2`, and the parser `syn` operate on **proc-macro token streams** which strip `//` line comments and most whitespace. Doc-comments (`///`) survive as `#[doc = "..."]` attributes. See [syn](https://github.com/dtolnay/syn) and [proc_macro2 docs](https://docs.rs/proc-macro2). This means `syn` alone cannot give us a lossless CST.
- For a lossless tree we want **rust-analyzer's rowan-based syntax tree** (`ra_ap_syntax`, see [Lossless Syntax Trees](https://dev.to/cad97/lossless-syntax-trees-280c) and the [rust-analyzer syntax module docs](https://rust-lang.github.io/rust-analyzer/syntax/index.html)). Rowan stores **trivia** (whitespace and comments) as ordinary tokens attached to leaves.
- Macros are an open problem: rust-analyzer keeps the unexpanded form alongside an expanded view; we should mirror this by storing macro invocations as opaque `.lino` blocks until a separate expansion pass is requested.

### JavaScript / TypeScript

- The de-facto fast parser is **SWC** (`swc_ecma_parser`, [crate](https://lib.rs/crates/swc_ecma_parser)). SWC's `Parser::take_comments` API and `Comments` trait let us harvest a comments table alongside the AST; recast and Babel preserve comments by attaching them to surrounding nodes. See [SWC `swc_common::comments`](https://rustdoc.swc.rs/swc_common/comments/trait.Comments.html), [SWC #4079 — full-fidelity comments](https://github.com/swc-project/swc/discussions/4079).
- For round-trip pretty printing we want **`recast`** (used inside `jscodeshift`) — its philosophy of "mutate AST nodes rather than rebuild" matches the `.lino` strategy of round-tripping through the CST shape, not the token stream. See [jscodeshift docs](https://jscodeshift.com/overview/introduction).

### Lean 4

- Lean 4 ships its own surface CST type, `Lean.Syntax`, and a pretty-printer pipeline (delaborator → parenthesizer → formatter, see [Lean 4 Pretty Printing chapter](https://leanprover-community.github.io/lean4-metaprogramming-book/extra/03_pretty-printing.html)). With `pp.all := true`, the round-trip `Syntax → format → Syntax` is documented to be the inverse of macro expansion.
- Trivia (comments, whitespace) live inside `Lean.Syntax.SourceInfo` and are preserved by the parser.
- Because Lean's macro system is extensible, our `.lino` CST has to either (a) carry the unexpanded syntax kind verbatim or (b) re-expand on import. The same tension exists in Rocq.

### Rocq

- Rocq's parser (`pcoq.ml`, `clexer.ml`) is built on Camlp5 and can be extended **by user code** (`Notation`/`Tactic Notation`), so a stand-alone parser is not feasible in the general case — see [coq/dev/doc/parsing.md](https://github.com/coq/coq/blob/master/dev/doc/parsing.md). The advice from upstream is to delegate parsing to Rocq itself.
- The replacement for SerAPI is **`coq-lsp`**. As of v0.2.x, `coq-lsp` supports comment recording behind `--record_comments`; comments are returned in `doc.comments` (see [rocq-community/rocq-lsp](https://github.com/rocq-community/rocq-lsp)). This is the recommended channel for both directions.
- For tooling that does not need full elaboration, the small **`tree-sitter-rocq`** grammars by [`lamg/tree-sitter-rocq`](https://github.com/lamg/tree-sitter-rocq) and [`krathul/tree-sitter-rocq`](https://github.com/krathul/tree-sitter-rocq) cover a useful subset and are CST-shaped.

The takeaway: **we do not write any of these parsers ourselves**. We bind to the canonical one per language and own only the tree mapping.

---

## Key design decisions

### D1. `.lino` becomes a CST, not just a syntax for ASTs

Today `.lino` tokenisation strips `#` comments and discards whitespace. We extend the LiNo *evaluator* contract with a CST view:

- Every original byte of the source is reachable through a `lino-cst.*` link, with three categories of nodes:
  - **Structural** — the existing parenthesised list nodes.
  - **Token** — leaf nodes carrying the original lexeme.
  - **Trivia** — whitespace runs and comments, attached to the *following* token (the Roslyn/libsyntax convention) by default, with a configurable "leading vs trailing" policy.
- A round-trip is a tree walk that emits leaves in document order, concatenating trivia and tokens — exactly the rowan strategy described in [Easy Lossless Trees with Nom and Rowan](https://blog.kiranshila.com/post/easy_cst).

This preserves backward compatibility: the existing AST view is the same tree with trivia and tokens hidden. The diagnostic, evaluator and check tooling can opt-in to the CST view per request.

### D2. Each host language gets its own `.lino` CST dialect

Rather than try to find a "common AST" between Rust and Lean, we declare four sibling `.lino` dialects:

- `lino-cst.rust.*` — one link tag per `ra_ap_syntax` `SyntaxKind` (matches [Ungrammar](https://rust-analyzer.github.io//blog/2020/10/24/introducing-ungrammar.html) productions).
- `lino-cst.js.*` — one link tag per `estree` / `swc_ecma_ast` node kind.
- `lino-cst.lean.*` — one link tag per `Lean.SyntaxNodeKind`.
- `lino-cst.rocq.*` — one link tag per `Vernacexpr`/`constr_expr` constructor, as serialised by `coq-lsp`.

The result is **four faithful CST mirrors of the host languages**, all expressed inside `.lino`. Cross-language transpilation (`host_A → host_B`) is then a separate, named transformation that consumes one dialect and emits another, with the explicit semantic mapping documented and tested.

This pattern is what the **Spoofax** / **SDF3** community calls "concrete syntax in the IR" and is the same idea behind [CrossTL](https://arxiv.org/abs/2508.21256). It avoids the "lowest-common-denominator AST" trap that has historically killed universal transpiler projects.

### D3. The shared semantic dialect (`lino-cst.shared.*`) is opt-in

Issue #138 emphasises that conversion must be unambiguous. To achieve that, the **default** transformation is dialect-preserving: `Rust → lino-cst.rust → Rust` is byte-equal, and `Rust → lino-cst.rust → JavaScript` is **explicitly undefined** unless a translator from `lino-cst.rust` to `lino-cst.js` exists.

We additionally define a shared **semantic** dialect `lino-cst.shared.*` that encodes the intersection (typed λ-calculus + ADT/inductive types + Pi + records + match) and is the existing typed RML fragment. Translation through `lino-cst.shared` is the cross-language path and is the only one that may lose host-specific concepts. The four host dialects are the *lossless* path; the shared dialect is the *bridge*.

### D4. Round-trip is tested as a contract

Following the [coq-serapi round-trip infrastructure contributed by Karl Palmskog](https://github.com/rocq-archive/coq-serapi/blob/main/CHANGES.md) and rust-analyzer's "syntax round-trip" tests, we adopt:

- `host_source` → `.lino` CST → `host_source'` and assert byte equality, modulo a documented list of canonicalisations.
- A failure produces a diff and a structured diagnostic; CI gates on green.

### D5. The four converters all ship in both JS and Rust

To preserve the existing dual-implementation discipline of RML (see the parity-CI work in [issue #93/#168](https://github.com/link-foundation/relative-meta-logic/pull/168)), every converter has a JavaScript implementation and a Rust implementation. Where the upstream parser only exists for one of the two host runtimes (e.g. `syn` is Rust-only, `Lean.Syntax` is Lean-only), the other implementation either calls out via a subprocess or uses a JS port (e.g. for SWC the JS implementation can use `@swc/core`, the Rust one uses `swc_ecma_parser`).

---

## Proposed conversion architecture

```
              ┌─────────────────────────┐
   .rs ─────► │  syn + ra_ap_syntax     │ ───►  lino-cst.rust.*
   .mjs ────► │  swc_ecma_parser        │ ───►  lino-cst.js.*
   .lean ───► │  Lean.Syntax (subproc)  │ ───►  lino-cst.lean.*
   .v ──────► │  coq-lsp --record_       │ ───►  lino-cst.rocq.*
              │  comments                │
              └─────────────────────────┘
                            │
                            ▼
               ┌──────────────────────────┐
               │  lino-cst.* trees (RML)  │  ← lossless, comments, trivia
               └──────────────────────────┘
                  │            ▲          │
                  │ shared/    │ identity │
                  ▼            │          ▼
              lino-cst.       (per-       lino-cst.<other>.*
              shared.*         dialect)
                  │
                  ▼
                <host-B source>
```

The four upstream parser bindings, the four `.lino` dialect serialisers, the shared-dialect translator, and the four printers are the work items in [§ Phased roadmap](#phased-roadmap-and-recommended-issue-split).

---

## Per-language conversion plan

The detail for each language lives in dedicated artefacts; this section is a précis.

### Rust ↔ `.lino`

- **Parser:** `ra_ap_syntax` (rowan-based, lossless).
- **Trivia model:** Rowan tokens already carry trivia; map every `SyntaxToken` to a `.lino` token leaf.
- **Macros:** keep unexpanded form as opaque `lino-cst.rust.macro_call` with the raw token stream stored as a list of `.lino` token leaves.
- **Identifiers:** preserve raw spellings, including raw identifiers (`r#match`) and unicode.
- **Round-trip tests:** corpus drawn from rust-analyzer's own `tests/parser` fixtures plus the existing `lib/` and `rust/src/` files.

### JavaScript ↔ `.lino`

- **Parser:** `swc_ecma_parser` (Rust) and `@swc/core` (JS) for AST; comments via the `Comments` interface. For full-fidelity round-trip prefer `recast` (already used by `jscodeshift`).
- **Trivia model:** attach leading comments to nodes; preserve quote style and parenthesisation via a small "format" sidecar link (e.g. `(format quote double)`).
- **Module syntax:** support both ESM and CommonJS; default to ESM in the dialect.
- **Round-trip tests:** corpus drawn from `js/src/` and the test262 conformance suite (sampled).

### Lean 4 ↔ `.lino`

- **Parser:** `Lean.Syntax` via a small Lean tool we call as a subprocess (`lean --run rml/lean_to_lino.lean`). This avoids reimplementing Lean's extensible parser.
- **Trivia model:** read `Lean.Syntax.SourceInfo` for leading/trailing whitespace and comments.
- **Macros:** keep raw `Syntax` plus a separate "expanded" branch produced by Lean's elaborator if requested.
- **Round-trip tests:** corpus drawn from `examples/lean-export-basic.lean` plus a hand-curated set of small Mathlib snippets.

### Rocq ↔ `.lino`

- **Parser:** `coq-lsp` via JSON-RPC with `--record_comments` (subprocess from both JS and Rust).
- **Trivia model:** `coq-lsp` provides `doc.comments`; we attach them to the following vernac.
- **Notations:** when a `Notation` command is encountered, recurse and record both the notation declaration and the parser state delta so future vernacs are parsed correctly.
- **Round-trip tests:** corpus drawn from `examples/rocq-export.lino`'s compiled `.v` form and the `tree-sitter-rocq` test cases for sanity.

---

## Phased roadmap and recommended issue split

This work should be tracked as a new epic, sister to [#95](https://github.com/link-foundation/relative-meta-logic/issues/95). Suggested phases (one issue each unless noted):

### Phase K — `.lino` CST infrastructure

- **K1** Add trivia-aware tokeniser to the LiNo parser (preserve `#…` comments and whitespace runs as nodes; opt-in flag).
- **K2** Add the rowan-style `lino-cst.*` link family and a tree-printer that round-trips the original source.
- **K3** Add round-trip tests for every file in `examples/` and `lib/`.

### Phase L — Rust converter

- **L1** Rust → `.lino` importer (binds `ra_ap_syntax`).
- **L2** `.lino` → Rust printer.
- **L3** Round-trip CI gate.

### Phase M — JavaScript converter

- **M1** JS → `.lino` importer (binds `swc_ecma_parser` / `@swc/core`).
- **M2** `.lino` → JS printer (recast-style).
- **M3** Round-trip CI gate.

### Phase N — Lean 4 converter

- **N1** Lean → `.lino` importer (Lean subprocess).
- **N2** `.lino` → Lean printer (via `Lean.PrettyPrinter`).
- **N3** Round-trip CI gate.

### Phase O — Rocq converter

- **O1** Rocq → `.lino` importer (`coq-lsp` subprocess, `--record_comments`).
- **O2** `.lino` → Rocq printer (uses `coq-lsp`'s pretty-printer for safe outputs, falls back to a hand-written one for our own dialect).
- **O3** Round-trip CI gate.

### Phase P — Cross-language bridges

- **P1** Define `lino-cst.shared.*` (typed λ + ADT + Pi + records + match) and prove it embeds existing RML kernel terms.
- **P2** Rust ↔ shared, JS ↔ shared, Lean ↔ shared, Rocq ↔ shared translators (4 issues).
- **P3** End-to-end demos: Lean → `.lino` → JS, JS → `.lino` → Rocq, etc., each with at least one example and one regression test.

### Phase Q — Documentation and tutorial

- **Q1** Single-page tutorial in `docs/tutorials/universal-cst.md`.
- **Q2** "Add a fifth language" guide (Python suggested as the worked example).

This split is intentionally fine-grained so each issue can land in a self-contained PR, the way the parity epic was delivered.

---

## Test strategy

Cribbed from rust-analyzer, SWC and SerAPI's round-trip suites:

1. **Trivia round-trip** — `parse(source).print() == source` for every fixture, byte-for-byte.
2. **Idempotent canonicalisation** — for the documented canonicalisations (e.g. CRLF → LF), `parse(parse(source).print()).print() == parse(source).print()`.
3. **Cross-language identity** — when translating from `lino-cst.rust → lino-cst.shared → lino-cst.rust`, the result equals the input up to canonicalisation.
4. **Negative tests** — every "rejected" construct in [`docs/LEAN_EXPORT.md`](../../LEAN_EXPORT.md) / [`docs/ROCQ-EXPORT.md`](../../ROCQ-EXPORT.md) becomes a structured `unrepresentable` `.lino` node and is asserted to round-trip into the same node.
5. **Bootstrap** — the case-study tree (`docs/case-studies/issue-138/`) itself is included in the bootstrap corpus so the converters keep working over their own deliverable.

These map cleanly onto five new CI jobs, one per phase.

---

## Existing components and libraries that help

A full catalogue is in [`existing-tools.md`](./existing-tools.md). Top picks per language:

| Need | Library | Notes |
|------|---------|-------|
| Rust lossless CST | [`ra_ap_syntax`](https://docs.rs/ra_ap_syntax) | rust-analyzer's rowan-based CST; preserves all trivia. |
| Rust AST (proc-macro shape) | [`syn`](https://github.com/dtolnay/syn), [`proc-macro2`](https://docs.rs/proc-macro2) | Stable surface for codegen; **strips comments**. |
| JS lossless CST | [`swc_ecma_parser`](https://lib.rs/crates/swc_ecma_parser) + [`recast`](https://github.com/benjamn/recast) | SWC for parsing; recast for round-trip printing. |
| JS codemod ergonomics | [`jscodeshift`](https://jscodeshift.com/) | Demonstrates "mutate, don't rebuild" pattern. |
| Lean CST + delaborator | [`Lean.Syntax`](https://leanprover-community.github.io/lean4-metaprogramming-book/main/05_syntax.html), [Lean PrettyPrinter](https://leanprover-community.github.io/lean4-metaprogramming-book/extra/03_pretty-printing.html) | `pp.all := true` documents round-trip. |
| Rocq parsing | [`coq-lsp`](https://github.com/rocq-community/rocq-lsp) | Use `--record_comments` for trivia; SerAPI is deprecated. |
| Tree-sitter grammars (reference) | [`tree-sitter-lean`](https://github.com/Julian/tree-sitter-lean), [`tree-sitter-rocq`](https://github.com/lamg/tree-sitter-rocq), [`tree-sitter-rust`](https://github.com/tree-sitter/tree-sitter-rust) | Useful for CST sanity-checks; not authoritative. |
| Generic CST library | [`rowan`](https://github.com/rust-analyzer/rowan), [`cstree`](https://crates.io/crates/cstree) | Reusable rowan-style CST in Rust if we want our own. |
| Universal-IR research | [CrossTL (arXiv 2508.21256)](https://arxiv.org/pdf/2508.21256), [Spoofax](https://spoofax.dev/), [Rascal](https://www.rascal-mpl.org/) | Background on this design pattern. |

---

## Open questions

Tracked separately in [`risks-and-open-questions.md`](./risks-and-open-questions.md). The two largest:

- **Q1** How aggressively should the importers preserve formatting that is *not* part of the surface syntax (alignment, indentation widths)? Rowan-style trees can preserve it, but two host pretty-printers may not be able to reproduce it. Recommendation: preserve in `.lino`, accept that the printer may canonicalise.
- **Q2** How do we deal with **extensible parsers** in Lean and Rocq (`macro`, `Notation`)? Recommendation: store the un-elaborated `Syntax`/`constr_expr` plus the parser-state delta, and only attempt elaboration on the `lino-cst.shared.*` path.

---

## References

### Linked from this case study

- [Easy Lossless Trees with Nom and Rowan — Kiran Shila](https://blog.kiranshila.com/post/easy_cst)
- [Lossless Syntax Trees — Christopher Durham](https://dev.to/cad97/lossless-syntax-trees-280c)
- [rust-analyzer — syntax module docs](https://rust-lang.github.io/rust-analyzer/syntax/index.html)
- [rust-analyzer — Introducing Ungrammar](https://rust-analyzer.github.io//blog/2020/10/24/introducing-ungrammar.html)
- [rust-analyzer / rowan](https://github.com/rust-analyzer/rowan)
- [cstree crate](https://crates.io/crates/cstree)
- [syn crate](https://github.com/dtolnay/syn)
- [proc-macro2 crate](https://docs.rs/proc-macro2)
- [swc_ecma_parser crate](https://lib.rs/crates/swc_ecma_parser)
- [SWC `swc_common::comments`](https://rustdoc.swc.rs/swc_common/comments/trait.Comments.html)
- [SWC discussion #4079 — full-fidelity comments](https://github.com/swc-project/swc/discussions/4079)
- [SWC compilation docs — comments](https://swc.rs/docs/configuration/compilation)
- [recast](https://github.com/benjamn/recast)
- [jscodeshift](https://jscodeshift.com/overview/introduction)
- [Lean 4 — Pretty Printing chapter](https://leanprover-community.github.io/lean4-metaprogramming-book/extra/03_pretty-printing.html)
- [Lean 4 — Syntax chapter](https://leanprover-community.github.io/lean4-metaprogramming-book/main/05_syntax.html)
- [Lean 4 source — PrettyPrinter](https://github.com/leanprover/lean4/blob/master/src/Lean/PrettyPrinter.lean)
- [tree-sitter-lean (experimental)](https://github.com/Julian/tree-sitter-lean)
- [tree-sitter-rocq — lamg](https://github.com/lamg/tree-sitter-rocq)
- [tree-sitter-rocq — krathul](https://github.com/krathul/tree-sitter-rocq)
- [rocq-community/rocq-lsp](https://github.com/rocq-community/rocq-lsp)
- [coq-serapi (deprecated, superseded by coq-lsp)](https://github.com/rocq-archive/coq-serapi)
- [coq/dev/doc/parsing.md](https://github.com/coq/coq/blob/master/dev/doc/parsing.md)
- [Spoofax language workbench](https://spoofax.dev/)
- [Rascal metaprogramming language](https://www.rascal-mpl.org/)
- [CrossTL — Universal Programming Language Translator (arXiv 2508.21256)](https://arxiv.org/pdf/2508.21256)
- [c2rust — Migrate C code to Rust](https://github.com/immunant/c2rust)
- [awesome-transpilers — milahu](https://github.com/milahu/awesome-transpilers)

### Internal references

- [`ARCHITECTURE.md`](../../../ARCHITECTURE.md)
- [`README.md` — Program extraction](../../../README.md#program-extraction-issue-f7)
- [`docs/LEAN_EXPORT.md`](../../LEAN_EXPORT.md)
- [`docs/ROCQ-EXPORT.md`](../../ROCQ-EXPORT.md)
- [`docs/case-studies/issue-13/` — Dependent types case study](../issue-13/)
- [`docs/case-studies/issue-22/` — Competitor concepts and feature comparison](../issue-22/)
- [`docs/case-studies/issue-95/` — Feature-parity epic audit](../issue-95/)
