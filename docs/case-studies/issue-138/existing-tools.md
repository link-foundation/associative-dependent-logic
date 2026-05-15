# Existing Tools and Libraries — Issue #138

This document catalogues the libraries and tools we recommend reusing — and the ones we deliberately reject — for the universal CST converters proposed in [`README.md`](./README.md). The breakdown follows the same tiering as [`docs/case-studies/issue-13/existing-tools.md`](../issue-13/existing-tools.md): direct reuse, reference implementations, and academic background.

## Tier 1 — Directly reusable

### Rust

#### `ra_ap_syntax` (rust-analyzer)

- **Docs:** <https://docs.rs/ra_ap_syntax>
- **Background:** [rust-analyzer syntax module](https://rust-lang.github.io/rust-analyzer/syntax/index.html), [Introducing Ungrammar](https://rust-analyzer.github.io//blog/2020/10/24/introducing-ungrammar.html)
- **Why:** Produces a **lossless rowan-based CST** with whitespace and comments retained as trivia tokens. The same library is consumed by rust-analyzer itself, so its surface is stable enough for our needs.
- **Use:** Phase L1 (Rust → `.lino`) binds `ra_ap_syntax::SourceFile::parse` and walks every `SyntaxToken` to emit a matching `lino-cst.rust.*` leaf. Phase L2 reverses the walk.

#### `rowan` and `cstree`

- **Repos:** [`rust-analyzer/rowan`](https://github.com/rust-analyzer/rowan), [`cstree`](https://crates.io/crates/cstree)
- **Why:** If we ever want our **own** lossless CST inside RML (rather than going via `ra_ap_syntax`'s rowan tree), these are the reusable generics. `cstree` is the actively maintained fork.
- **Use:** Optional. Recommended only if we discover `ra_ap_syntax`'s shape is too coupled to rust-specific kinds.

#### `syn` + `proc-macro2`

- **Repos:** [`dtolnay/syn`](https://github.com/dtolnay/syn), [`proc-macro2`](https://docs.rs/proc-macro2)
- **Status:** Stable, but **strips `//` line comments** at tokenisation — only `///` doc-comments survive as `#[doc = "..."]` attributes. Confirmed by [SWC discussion noting the same proc-macro2 limitation](https://github.com/swc-project/swc/discussions/4079) and the [Rust reference on procedural macros](https://doc.rust-lang.org/reference/procedural-macros.html).
- **Why:** Useful for the *output* side (Phase L2) when we generate a syntactically valid Rust file from a `lino-cst.rust.*` tree — `syn`'s `quote!` macro is the lightest way to emit clean Rust.
- **Use:** Phase L2 as a printing helper; not the parser.

### JavaScript / TypeScript

#### `swc_ecma_parser` and `@swc/core`

- **Docs:** [`swc_ecma_parser`](https://lib.rs/crates/swc_ecma_parser), [`@swc/core`](https://swc.rs/docs/usage/core)
- **Comments interface:** [`swc_common::comments`](https://rustdoc.swc.rs/swc_common/comments/trait.Comments.html)
- **Why:** SWC is the de-facto fast JS/TS parser. Comments can be harvested through the `Comments` trait so the AST stays clean while we keep a side table of trivia. SWC ships in both Rust and Node bindings, which lets the JS and Rust sides of RML share the same parser version.
- **Caveats:** SWC's `preserveAllComments` option keeps comments in the AST but does not lock down their *exact* position; for byte-equal round-trips we layer `recast`/`magic-string` on top. See [SWC compilation docs](https://swc.rs/docs/configuration/compilation).
- **Use:** Phase M1 (JS → `.lino`).

#### `recast`

- **Repo:** [`benjamn/recast`](https://github.com/benjamn/recast)
- **Wrapper:** [`jscodeshift`](https://jscodeshift.com/overview/introduction)
- **Why:** Recast's philosophy is to **mutate the AST in place and re-print only the changed regions**, preserving original formatting elsewhere. This is exactly the contract the issue asks for ("conversion should not result in data loss, and should be as precise as possible"). It is the parser babel-codemod, jscodeshift and most JS refactor tools sit on.
- **Use:** Phase M2 (`.lino` → JS) uses recast's printer.

### Lean 4

#### `Lean.Syntax` and the Lean pretty-printer pipeline

- **Docs:** [Lean 4 Syntax chapter](https://leanprover-community.github.io/lean4-metaprogramming-book/main/05_syntax.html), [Lean 4 Pretty Printing chapter](https://leanprover-community.github.io/lean4-metaprogramming-book/extra/03_pretty-printing.html), [Lean 4 source — PrettyPrinter](https://github.com/leanprover/lean4/blob/master/src/Lean/PrettyPrinter.lean)
- **Why:** Lean ships its own parser, its own `Syntax` CST type (with `SourceInfo` carrying leading/trailing trivia), and a documented round-trip path via the delaborator. With `pp.all := true`, Lean's pretty printer is the inverse of macro expansion.
- **Use:** Phase N1 calls `Lean.Parser.parseHeader` + `Lean.Parser.parseCommand` in a tiny Lean program we invoke as a subprocess; Phase N2 calls `Lean.PrettyPrinter.format` to emit Lean.

#### `Lake` / `elan`

- **Why:** The Lean toolchain manager. Required to call Lean as a subprocess. Already used by `examples/lean-export-basic.lean` testing.

### Rocq

#### `coq-lsp`

- **Repo:** [`rocq-community/rocq-lsp`](https://github.com/rocq-community/rocq-lsp)
- **Flag:** `--record_comments` (experimental, documented in the README).
- **Why:** Rocq's parser is extensible at run-time via `Notation` / `Tactic Notation`, so no third-party tool can parse all Rocq files correctly without delegating to Rocq itself. `coq-lsp` is the supported way to do that and has explicit comment recording. Coq SerAPI is **deprecated** in favour of `coq-lsp` ([coq-serapi CHANGES](https://github.com/rocq-archive/coq-serapi/blob/main/CHANGES.md)).
- **Use:** Phases O1 and O2 talk to a `coq-lsp` subprocess via JSON-RPC.

#### Coq SerAPI (deprecated)

- **Repo:** [`rocq-archive/coq-serapi`](https://github.com/rocq-archive/coq-serapi)
- **Why we are *not* using it:** Development has stopped; upstream recommends `coq-lsp`. We mention it only because previous research on round-trip Rocq tooling (Karl Palmskog's round-trip infrastructure) lives here.

## Tier 2 — Reference implementations and grammars

### Reference CST grammars

| Grammar | Repo | Notes |
|---------|------|-------|
| Rust | [`tree-sitter/tree-sitter-rust`](https://github.com/tree-sitter/tree-sitter-rust) | Useful as a sanity check; not authoritative — rust-analyzer's grammar is. |
| JavaScript | [`tree-sitter/tree-sitter-javascript`](https://github.com/tree-sitter/tree-sitter-javascript) | Same; used for editor highlighting. |
| Lean 4 | [`Julian/tree-sitter-lean`](https://github.com/Julian/tree-sitter-lean) | Experimental; useful for our CST sanity checks. |
| Rocq | [`lamg/tree-sitter-rocq`](https://github.com/lamg/tree-sitter-rocq), [`krathul/tree-sitter-rocq`](https://github.com/krathul/tree-sitter-rocq) | Both grammars are incomplete by design; the second targets a minimal subset. |

### Source-to-source / language-workbench inspiration

| Project | Why it matters |
|---------|----------------|
| [Spoofax](https://spoofax.dev/) | Language workbench with SDF3 for parsing + pretty printing in one declarative spec, the closest historical analogue to what `.lino` aims to be. |
| [Rascal MPL](https://www.rascal-mpl.org/) | Source-to-source transformation language with a unified CST model. |
| [CrossTL (arXiv 2508.21256)](https://arxiv.org/pdf/2508.21256) | Recent (2025) paper on a universal IR for programming-language translation; supports the architectural choice in [`README.md` § D2](./README.md#d2-each-host-language-gets-its-own-lino-cst-dialect). |
| [c2rust](https://github.com/immunant/c2rust) | A real-world transpiler that explicitly punts on comment preservation; useful as a *what not to do* reference. |
| [awesome-transpilers (milahu)](https://github.com/milahu/awesome-transpilers) | Curated index. |

## Tier 3 — Academic and historical references

| Paper / post | Why it matters |
|--------------|----------------|
| [Easy Lossless Trees with Nom and Rowan — Kiran Shila](https://blog.kiranshila.com/post/easy_cst) | Hands-on explanation of how rowan attaches trivia. |
| [Lossless Syntax Trees — Christopher Durham](https://dev.to/cad97/lossless-syntax-trees-280c) | Why a CST is needed for refactoring tooling; we use the same arguments for cross-language transpilation. |
| [RFC: transition to Roslyn's model for trivia — rust-analyzer #6584](https://github.com/rust-lang/rust-analyzer/issues/6584) | Trade-offs between IntelliJ-style "trivia between tokens" and Roslyn-style "trivia attached to following token". We adopt Roslyn-style. |
| [coq/dev/doc/parsing.md](https://github.com/coq/coq/blob/master/dev/doc/parsing.md) | Authoritative description of Rocq's extensible parser; justifies delegating to `coq-lsp`. |
| [Karl Palmskog's round-trip testing infrastructure for SerAPI](https://github.com/rocq-archive/coq-serapi/blob/main/CHANGES.md) | Template for our round-trip tests. |
| [Lean 4 — The Lean 4 Theorem Prover and Programming Language (Moura, Ullrich, 2021)](https://link.springer.com/chapter/10.1007/978-3-030-79876-5_37) | Background on Lean's macro system, which informs Phase N. |

## What we deliberately do **not** use

| Tool | Why we skip it |
|------|----------------|
| Hand-written parsers for any of the four languages. | The upstream parsers are battle-tested; any divergence is a recurring source of bugs (see c2rust's history). |
| Tree-sitter grammars as the *authoritative* parser. | Incomplete for Lean and Rocq because of extensible syntax. Acceptable as a CST sanity check, not as the primary parser. |
| The deprecated Coq SerAPI. | Upstream points users at `coq-lsp` instead. |
| Translating directly between host AST pairs (Rust ↔ JS), bypassing `.lino`. | Loses the universal-IR property the issue asks for. |
