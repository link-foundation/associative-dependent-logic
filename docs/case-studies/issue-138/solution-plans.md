# Solution Plans — Issue #138

This document proposes one or more solution plans per requirement listed in [`requirements.md`](./requirements.md). The umbrella architecture is in [`README.md`](./README.md); the library inventory is in [`existing-tools.md`](./existing-tools.md); the CST data model is in [`cst-model.md`](./cst-model.md).

For each requirement we list:

- **Plan A** — the recommended approach.
- **Plan B / C** — alternatives we considered and rejected, with the reason.

## R1 — Bidirectional convert between LiNo and host languages

### Plan A (recommended): one importer + one printer per language, both share the `.lino` CST

- Importer reads host source via the canonical upstream parser (see [`existing-tools.md`](./existing-tools.md)), emits a `lino-cst.<host>.*` tree.
- Printer reverses the walk and produces host source.
- Both ship as a subcommand (`rml import <host>` and `rml export <host>`) per [R21](./requirements.md#derived-non-functional-requirements).

### Plan B (rejected): single bidirectional codec per language

- One function that takes either source string or `.lino` and returns the other.
- **Why rejected:** complicates testing and obscures asymmetric semantics (e.g. importing a Lean file requires a Lean subprocess, exporting one does not).

## R2 — Use a concrete syntax tree, not an AST

### Plan A: trivia-aware `.lino` CST modelled on rowan

Add three new node kinds to the LiNo parser (used only when the CST mode is requested):

- `lino-cst.token` — an opaque leaf with the original lexeme.
- `lino-cst.trivia.whitespace` — a run of whitespace characters.
- `lino-cst.trivia.comment` — a `#`-comment.

The existing structural-list node becomes the third CST node kind. Round-trip is a flat in-order walk of leaves.

Backward compatibility: the existing AST view is the same tree with trivia and tokens hidden, exactly how rust-analyzer projects an AST view onto its rowan CST.

### Plan B (rejected): keep `.lino` AST, store trivia in a sidecar file

- **Why rejected:** loses single-source-of-truth; tools that don't read the sidecar silently lose data on save.

## R3 — Encode comments

### Plan A: comments are CST trivia attached to the *following* token (Roslyn / libsyntax convention)

- Matches the trivia model used by rust-analyzer ([RFC #6584](https://github.com/rust-lang/rust-analyzer/issues/6584)) and Lean's `SourceInfo`.
- Comments at end-of-file are attached to a synthetic EOF token, so no comment is lost.

### Plan B (considered): "trivia between tokens" (IntelliJ-style)

- Trivia is its own peer node, not attached to a token.
- **Why not first choice:** harder to query ("find all comments preceding `fn foo`") without traversing trivia siblings; both libsyntax and Roslyn migrated away from this design.

## R4 — Encode every variable and other name

### Plan A: preserve raw spelling in `lino-cst.<host>.identifier` leaves

- Store the **original token text**, not a normalised string.
- Add a sanitisation pass on emit (Rust raw identifiers, Lean reserved keywords, Rocq punctuation) that is **distinct from the CST** so the CST itself is byte-faithful.
- Reuse the existing collision check in [`docs/ROCQ-EXPORT.md`](../../ROCQ-EXPORT.md#supported-subset) and generalise it.

### Plan B (rejected): canonicalise identifiers in the CST

- **Why rejected:** breaks the round-trip property (R9) and conflicts with the issue's "as precise as possible" wording.

## R5 — Encode whitespace where needed

### Plan A: whitespace runs are CST trivia, preserved verbatim

- Same node kind as comments, just `kind = whitespace`.
- The printer writes the bytes back unchanged, so indentation, blank lines and CRLF/LF are all preserved.

### Plan B (considered): only preserve "significant" whitespace

- Hard to define; rejected to keep the contract simple.

## R6 / R7 — Cross-language round-trip (Lean ↔ JS, JS ↔ Rocq, etc.)

### Plan A: two-step pipeline through `lino-cst.shared.*`

- `Lean → lino-cst.lean → lino-cst.shared → lino-cst.js → JavaScript`.
- The middle step `lino-cst.<src> → lino-cst.shared` is the *lossy* one (it drops host-specific concepts); the printers stay lossless.
- When the source has constructs outside the shared dialect, the translator emits an `unrepresentable` node and surfaces it as a diagnostic — the user can either rewrite the source or extend the shared dialect.

### Plan B (rejected): direct N×N translators (one per host pair)

- Twelve translators (4 hosts × 3 targets) instead of four, plus four shared-dialect translators.
- **Why rejected:** quadratic growth; loses the "universal IR" property the issue asks for.

## R8 — `.lino` becomes the universal intermediate CST

### Plan A: ship the four host dialects + the shared dialect, document the contract

- Treat dialect identifiers (`lino-cst.<host>.*`) as a typed namespace.
- Provide a tiny grammar file per dialect under `lib/lino-cst/<host>.lino` listing every node kind, so the dialect is self-describing inside RML (the same trick used by `lib/programming-language/core.lino`).

### Plan B (rejected): expose only the AST surface, hope for the best

- **Why rejected:** caller can never know whether a tree is faithful or normalised, which is the whole point of the issue.

## R9 — Conversion is lossless

### Plan A: round-trip test gate per language

- For every fixture, assert `host_to_lino(parse(s)).print() == s` byte-for-byte.
- Allow a small documented canonicalisation list (e.g. trailing newline, CRLF → LF). The list lives in `docs/case-studies/issue-138/canonicalisations.md` (to be added in Phase K).
- This mirrors [Karl Palmskog's SerAPI round-trip infrastructure](https://github.com/rocq-archive/coq-serapi/blob/main/CHANGES.md).

## R10 — As precise as possible

### Plan A: delegate parsing to the canonical upstream parser, do not write our own

- Rust: `ra_ap_syntax`.
- JS: `swc_ecma_parser` / `@swc/core`.
- Lean: `Lean.Parser` via a Lean subprocess.
- Rocq: `coq-lsp` JSON-RPC.
- Justified in [`existing-tools.md`](./existing-tools.md). c2rust's experience is the cautionary tale.

## R11 — Enough concepts to encode each host concept

### Plan A: one `lino-cst.<host>.*` tag per upstream production

- Generate the dialect from the upstream grammar where possible:
  - Rust: from rust-analyzer's `ungrammar` spec.
  - JS: from `swc_ecma_ast`'s `enum` definitions.
  - Lean: from `Lean.SyntaxNodeKind`.
  - Rocq: from `coq-lsp`'s `Vernacexpr` / `constr_expr` serialisation.
- The generated dialects are checked into `lib/lino-cst/<host>.lino` for inspection and bootstrap.

## R12 — Adding more languages later

### Plan A: ship a guide and one worked extra example

- `docs/tutorials/universal-cst-add-a-language.md` walks through wiring Python as the fifth language, since it is well-served by a permissive parser (`libcst`, which is itself a lossless CST) and is the obvious next target.
- Demonstrates that the case-study work in this PR generalises.

### Plan B (rejected): commit to N specific extra languages now

- **Why rejected:** not asked for in the issue; would balloon the epic.

## R13 — Extensive test coverage

### Plan A: five test categories, gated in CI

- See [`acceptance-tests.md`](./acceptance-tests.md). The five jobs are: trivia round-trip, idempotent canonicalisation, cross-language identity, negative `unrepresentable`, bootstrap corpus inclusion.

## R14, R15, R16, R17, R18 — Case-study deliverables

### Plan A: this PR

- `README.md`, `requirements.md`, `existing-tools.md`, `solution-plans.md`, `cst-model.md`, `acceptance-tests.md`, `risks-and-open-questions.md`.
- Mirrors [`docs/case-studies/issue-13/`](../issue-13/) and [`docs/case-studies/issue-26/`](../issue-26/).

## R19 — Single PR

### Plan A: case study in this PR, implementation in follow-up per-phase PRs

- Justified in [`requirements.md` § What this PR does not do](./requirements.md#what-this-pr-does-not-do).

### Plan B (considered): land the whole thing in one PR

- **Why rejected:** comparable in size to the parity epic ([issue #95](https://github.com/link-foundation/relative-meta-logic/issues/95)), which was delivered in 67 PRs over many phases. Doing the same here is the project convention.

## R20 — Parity across JS and Rust implementations

### Plan A: every CLI subcommand has a JS and a Rust implementation

- For Rust-only parser dependencies (`ra_ap_syntax`, `syn`), the JS implementation either:
  - Calls the Rust implementation as a subprocess (acceptable since the importer is already an out-of-process operation for Lean and Rocq), or
  - Uses a JS port (`@swc/core` for JS itself; no equivalent for Rust today).

## R21 — CLI surface

### Plan A: `rml import <host>` and `rml export <host>`

- `rml import lean file.lean -o file.lino`
- `rml import rust file.rs -o file.lino`
- `rml import js file.mjs -o file.lino`
- `rml import rocq file.v -o file.lino`
- `rml export <host> file.lino -o file.<ext>` (already exists for `lean` and `rocq`).

## R22 — CI gates

### Plan A: one CI job per converter, plus one cross-language matrix job

- Five new jobs in `.github/workflows/round-trip.yml`.
- Cross-language matrix is the 4×4 identity check for the shared dialect.

## R23 — Round-trip example per language

### Plan A: extend `examples/` with one fixture per language

- `examples/round-trip-rust.rs` ↔ `examples/round-trip-rust.lino`
- `examples/round-trip-js.mjs` ↔ `examples/round-trip-js.lino`
- `examples/round-trip-lean.lean` ↔ `examples/round-trip-lean.lino`
- `examples/round-trip-rocq.v` ↔ `examples/round-trip-rocq.lino`

## R24 — Opt-in CST infrastructure

### Plan A: CST mode is a parser flag, default off

- `parse_links(src, { mode: 'ast' })` (default) — existing behaviour, comments stripped.
- `parse_links(src, { mode: 'cst' })` — new behaviour, trivia preserved.
- The two modes share the underlying parser; the difference is whether trivia leaves are emitted.
