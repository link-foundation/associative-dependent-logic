# Risks and open questions — Issue #138

This artefact captures the design trade-offs we are not yet ready to commit to, and the risks the implementation phases need to plan around. They are listed in order of expected impact, highest first.

## Q1. Extensible parsers (Lean macros, Rocq notations)

Both Lean 4 and Rocq let user code extend the parser. A Rocq `Notation` declaration can change how subsequent tokens are read; a Lean `macro` does the same.

**Risk:** an importer that does not see the parser-state delta will mis-parse later vernacs / commands.

**Options:**

- **(A) Delegate to the upstream parser.** Always run `coq-lsp` / Lean's own parser. The dialect tree records the unexpanded `Syntax` / `constr_expr` plus the parser-state delta as opaque trivia. Round-trip is byte-faithful; cross-language translation through the shared dialect is gated behind explicit elaboration.
- **(B) Forbid extensible syntax in the input.** Reject `Notation` and `macro` declarations. Simpler, but rules out most real Lean and Rocq files.
- **(C) Run a separate elaboration pass.** Importer produces an elaborated tree; round-trip is no longer byte-faithful but cross-language translation is easier.

**Recommendation:** A for the lossless lossless path, optional C for the shared-dialect path. Document the trade-off in [`cst-model.md` § 8](./cst-model.md#8-canonicalisations-documented).

## Q2. Comment placement ambiguity

When a comment sits between two tokens, it is ambiguous whether it belongs to the preceding or following token. Three established conventions exist:

- **Roslyn / Swift libsyntax:** trivia attaches to the **following** token.
- **IntelliJ:** trivia is a peer node between tokens.
- **rust-analyzer (current):** trivia attaches to the **following** token, but with documented "leading vs trailing" exceptions for end-of-line comments.

**Risk:** any single rule will surprise some users; tools that round-trip through us must agree on the rule.

**Recommendation:** Roslyn-style, document the rule in [`cst-model.md`](./cst-model.md), and let callers query "leading comments of node X" and "trailing comments of node X" through helper APIs.

## Q3. Identifier normalisation

Rust raw identifiers (`r#match`), Lean's unicode-friendly names, and Rocq's escaped notations are not interchangeable. Naive translation will produce invalid code.

**Risk:** cross-language translation produces source that does not compile.

**Recommendation:** keep the *raw* spelling in the CST; add a per-language sanitisation pass on emit that is **distinct** from the CST. Generalise the existing collision check in [`docs/ROCQ-EXPORT.md`](../../ROCQ-EXPORT.md#supported-subset).

## Q4. How aggressively should we preserve formatting?

Indentation, blank-line counts, alignment, choice of quote characters are all part of the source. The rowan-style CST preserves them, but host pretty-printers may not be able to reproduce them.

**Risk:** byte-equal round-trip may be impossible for some host languages even if we preserve everything.

**Recommendation:** preserve in the CST. Where the host printer cannot reproduce a detail, surface a structured warning rather than silently dropping it. Document each known case in `docs/case-studies/issue-138/canonicalisations.md` when Phase K lands.

## Q5. Subprocess dependencies

Phases N (Lean) and O (Rocq) depend on subprocesses (`lean`, `coq-lsp`). The JS side of Phase L (Rust) also benefits from calling out to the Rust toolchain.

**Risk:** CI complexity, slower tests, harder to run locally.

**Recommendation:**

- Cache the toolchains in CI (already done for the bootstrap workflow).
- Provide a `rml import lean --no-subprocess` flag that errors cleanly if the toolchain is missing, rather than producing a bogus tree.
- Keep the subprocess invocation behind a small adapter so the rest of the converter does not need to know.

## Q6. Cross-language semantic equivalence

Category 3 of [`acceptance-tests.md`](./acceptance-tests.md) requires that we have a notion of "semantic equality" between host ASTs. For the typed shared fragment this is straightforward; for less-shared constructs it is an open research question.

**Risk:** we over-promise on cross-language transpilation and ship a tool that produces compilable but semantically wrong code.

**Recommendation:** narrow the cross-language guarantee to the typed shared fragment (see [`docs/case-studies/issue-13/`](../issue-13/)) and document, in each `unrepresentable` diagnostic, the exact construct that prevented translation.

## Q7. Documentation cost

Each new dialect needs a generated grammar file in `lib/lino-cst/<host>.lino` plus prose explaining the more surprising mappings. Together with five new CI jobs and five new round-trip suites, this is a substantial documentation effort.

**Risk:** docs go stale faster than they are updated.

**Recommendation:** generate as much as possible from the upstream grammar; gate CI on a `diff lib/lino-cst/rust.lino <(cargo run -p gen-rust-grammar)`-style check so docs cannot drift.

## Q8. Versioning the host languages

Rust, JS, Lean and Rocq all evolve. Lean 4 has a stable language version; Rocq has frequent releases; JS has yearly ECMAScript editions; Rust pins to an edition. A CST faithful to Lean 4.6 may not parse Lean 4.7.

**Risk:** importers fall behind upstream and silently mis-parse.

**Recommendation:** pin the upstream parser version per converter; bump in dedicated PRs; include the upstream version string in the `lino-cst.<host>.*` dialect header so old `.lino` files declare their provenance.

## Q9. Performance

A naive trivia-bearing CST roughly doubles the node count of a typical file. For the 10×–100× scale of cross-language refactoring, this can matter.

**Risk:** test runs slow down.

**Recommendation:** measure on the Phase K corpus first; only optimise if the measurements demand it. The rust-analyzer experience suggests rowan's representation is fast enough for entire crates; we should inherit that property.

## Q10. Scope creep

The cross-language space is a sink for ambition. Spoofax, Rascal and CrossTL all aim to be a "universal IR" and all carry decades of accumulated complexity.

**Risk:** the epic balloons and never lands.

**Recommendation:** treat each per-phase issue as independently shippable, modelled on the parity-epic discipline in [`docs/case-studies/issue-95/`](../issue-95/). Land the K–O lossless path first; defer P (cross-language) until the four importers are stable.
