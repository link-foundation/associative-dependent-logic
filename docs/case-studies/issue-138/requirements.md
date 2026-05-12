# Requirements — Issue #138

This document extracts every distinguishable requirement from the text of [issue #138](https://github.com/link-foundation/relative-meta-logic/issues/138) and pairs each one with a status flag and a proposed solution plan. Detail of the plans is in [`solution-plans.md`](./solution-plans.md). The umbrella analysis is in [`README.md`](./README.md).

## Legend

| Mark | Meaning |
|------|---------|
| Today | Already true in `main` (cite the artefact). |
| Partial | Partial support today; needs an extension. |
| Missing | Not implemented; new work. |
| Out of scope (OOS) | Mentioned but explicitly excluded by the issue text or by the recommended plan. |

## Requirements table

| ID | Requirement (paraphrased from #138) | Status | Where it lives today / where it needs to land |
|----|-------------------------------------|--------|------------------------------------------------|
| **R1** | Convert between LiNo notation and other host languages. | Partial | One-way AST exporters exist: [`docs/LEAN_EXPORT.md`](../../LEAN_EXPORT.md), [`docs/ROCQ-EXPORT.md`](../../ROCQ-EXPORT.md), [`README.md` § Program extraction](../../../README.md#program-extraction-issue-f7). Importers missing for all four targets. |
| **R2** | The conversion must use a **concrete syntax tree (CST)**, not just an AST. | Missing | `.lino` is currently tokenised into an AST that discards comments and whitespace ([`ARCHITECTURE.md` § Stage 2](../../../ARCHITECTURE.md#stage-2-tokenization-and-ast-construction)). Plan K introduces a trivia-aware CST. |
| **R3** | Encode comments. | Missing | `#`-comments are stripped during tokenisation. Plan K1 makes them first-class CST nodes; per-language dialects (Plans L–O) map host-language comments to `.lino` trivia. |
| **R4** | Encode every variable and every other name. | Today, on the typed subset | Existing exporters preserve identifiers with a sanitisation pass ([`docs/ROCQ-EXPORT.md` § Supported subset](../../ROCQ-EXPORT.md#supported-subset)). The CST must preserve **raw spellings** verbatim — including raw identifiers (`r#match`) and unicode. Plans L1, M1, N1, O1. |
| **R5** | Encode whitespace where needed. | Missing | Same plan as R3. Whitespace runs become trivia leaves attached to the following token, à la rust-analyzer's rowan trees. |
| **R6** | Round-trip: Lean → `.lino` → JS must compile. | Missing | Requires the four importers (R1) **and** a defined `lino-cst.shared.*` dialect (Phase P). Cross-language compilation is only meaningful on the shared semantic fragment. |
| **R7** | Round-trip in the other direction: JS → `.lino` → Rocq, etc. | Missing | Same as R6. |
| **R8** | `.lino` becomes the **universal intermediate CST**, not AST. | Missing | This is the architectural goal of the whole epic. Met when Phases K–O are green. |
| **R9** | Conversion is **lossless**: no data loss between source and CST. | Missing | Met by the CST round-trip contract in Phase K (`parse(s).print() == s`) and per-language round-trips in Phases L–O. |
| **R10** | Conversion is **as precise as possible**. | Missing | Met by binding to the canonical upstream parser per language (`ra_ap_syntax`, `swc_ecma_parser`, `Lean.Syntax`, `coq-lsp`) instead of writing our own; see [`existing-tools.md`](./existing-tools.md). |
| **R11** | RML has **enough concepts to strictly, unambiguously encode each concept** of Rocq, Lean, Rust, JavaScript. | Missing (per-language dialects), Today (shared fragment) | Achieved by declaring four sibling `.lino` dialects (`lino-cst.rust.*`, `lino-cst.js.*`, `lino-cst.lean.*`, `lino-cst.rocq.*`) — one tag per host-language syntactic production. The shared semantic dialect (`lino-cst.shared.*`) covers the typed fragment already present (see [`docs/case-studies/issue-13/`](../issue-13/)). |
| **R12** | Once successful with these four, more languages become a **formal, automated translation** target. | Missing | Phase Q2 ships a "Add a fifth language" guide (Python suggested), which is the concrete demonstration of this requirement. |
| **R13** | Everything is covered extensively with tests. | Missing for new code; the existing exporters are tested | Each new converter ships with the five test categories in [`acceptance-tests.md`](./acceptance-tests.md): trivia round-trip, idempotent canonicalisation, cross-language identity, negative `unrepresentable` tests, and inclusion in the bootstrap corpus. |
| **R14** | Collect data related to the issue into `docs/case-studies/issue-138/`. | Met by this PR | The folder structure follows the convention of [`docs/case-studies/issue-13/`](../issue-13/) and [`docs/case-studies/issue-95/`](../issue-95/). |
| **R15** | Do deep case-study analysis, search online for additional facts. | Met by this PR | [`README.md` § Why the four target languages are hard](./README.md#why-the-four-target-languages-are-hard) and [`existing-tools.md`](./existing-tools.md) cite external sources. |
| **R16** | List each and every requirement. | Met by this file. | — |
| **R17** | Propose possible solutions and a solution plan for each requirement. | Met by [`solution-plans.md`](./solution-plans.md). | — |
| **R18** | Check known existing components / libraries that solve similar problems. | Met by [`existing-tools.md`](./existing-tools.md). | — |
| **R19** | Plan and execute everything in a single pull request. | This PR delivers the case study, plan, and tracking issues. Implementation is then split into per-phase PRs as recommended in [`README.md` § Phased roadmap](./README.md#phased-roadmap-and-recommended-issue-split). | The issue explicitly notes "you have unlimited time and context, as context auto-compacts and you can continue indefinitely". We treat the deliverable as a planning-quality case study identical in shape to the parity epic ([issue #95](https://github.com/link-foundation/relative-meta-logic/issues/95)). |

## Derived non-functional requirements

Issue #138 implies, but does not state, the following. We surface them so the reviewer can confirm them.

| ID | Requirement | Justification |
|----|-------------|---------------|
| **R20** | Both the JavaScript and Rust implementations of RML implement every converter. | Mandated by the dual-implementation discipline tracked in [issue #93/#168](https://github.com/link-foundation/relative-meta-logic/pull/168). |
| **R21** | Each converter is exposed through the existing `rml` CLI (`rml import lean`, `rml export lean`, etc.). | Consistency with [`docs/LEAN_EXPORT.md` § CLI](../../LEAN_EXPORT.md) and [`docs/ROCQ-EXPORT.md` § CLI](../../ROCQ-EXPORT.md#cli). |
| **R22** | Each converter is gated by a dedicated CI job and a documented round-trip suite. | Consistency with the bootstrap and parity CI jobs (see [`docs/case-studies/issue-95/` § Phase J](../issue-95/)). |
| **R23** | Each converter ships with at least one example file in `examples/` that exercises the round-trip. | Consistency with the existing convention for `examples/lean-export-basic.lino` ↔ `examples/lean-export-basic.lean`. |
| **R24** | The CST infrastructure is opt-in for existing callers — current `rml run` and `rml check` behaviour is unchanged. | Preserves the property that all 122 existing tests keep passing (see [`ARCHITECTURE.md`](../../../ARCHITECTURE.md)). |

## What this PR does **not** do

For the avoidance of doubt:

- This PR does **not** implement any of Phases K–Q. It commits the case-study folder only.
- This PR does **not** open the per-phase tracking issues. We recommend doing that as a follow-up once the case study is reviewed, in the same way the parity epic was filed by [PR #27 / issue #26](https://github.com/link-foundation/relative-meta-logic/pull/27).
- This PR does **not** change any test or any source file outside `docs/case-studies/issue-138/`.

The reason is operational: issue #138 is a planning issue ("we need to make sure we have ..."), the existing related deliverables in this repository (`issue-13`, `issue-22`, `issue-26`/`#27`, `issue-95`) all use the same shape (case-study folder first, implementation tracked separately), and that shape lets reviewers concentrate on the design before code starts to land.
