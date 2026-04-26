# Case Study: Shared Examples Folder

**Issue:** [#68 - Make examples folder in the root, and these should be shared between both JavaScript and Rust and both implementation should fully support them](https://github.com/link-foundation/relative-meta-logic/issues/68)

## Executive Summary

Issue #68 asks the project to host a single, language-agnostic `examples/` folder at the repository root. Both the JavaScript and the Rust implementations must execute every example correctly, and every example must be covered by automated tests so that regressions in either implementation are caught immediately.

Before this change the examples were duplicated three times:

- `/js/examples/*.lino` (13 files)
- `/rust/examples/*.lino` (13 files, byte-identical to the JS copies)
- inlined verbatim into the JavaScript and Rust test suites under "Example:" `describe`/`mod` blocks

That layout violated the single-source-of-truth principle and meant that fixing or extending an example had to be done in 3+ places, making drift between the two implementations very easy.

## Collected Issue Data

The original issue body in full:

> Make examples folder in the root, and these should be shared between both JavaScript and Rust and both implementation should fully support them
>
> And every example should be carefully tested, and if anything wrong we should fix it.

The repository already had two separate `examples/` folders that were identical copies of each other, and a CLI in each implementation that could already run a `.lino` file from disk:

- JavaScript CLI: `node js/src/rml-links.mjs <file.lino>`
- Rust CLI: `cargo run -p relative-meta-logic -- <file.lino>` (binary `rml`)

Both CLIs accept the same `.lino` syntax (LiNo / Links Notation). LiNo is the shared serialization format that makes a single source-of-truth examples folder feasible — the same `.lino` file is parsed and evaluated by either implementation with identical results.

### Existing related work in the repo

- `js/examples/` and `rust/examples/` already contained 13 byte-identical examples (verified with `diff -r`).
- `js/demo.lino` and `rust/demo.lino` were also byte-identical, as were the two `flipped-axioms.lino` files.
- The JavaScript test suite (`js/tests/rml-links.test.mjs`) and Rust integration tests (`rust/tests/rml_tests.rs`) already had blocks of "Example: …" tests that re-typed the example contents inline and asserted on results.

## Requirements

| # | Requirement | Implementation |
|---|-------------|----------------|
| R1 | A single `examples/` folder lives at the repository root | `/examples/` now holds all `.lino` files; the per-language `js/examples/` and `rust/examples/` folders are removed |
| R2 | The JavaScript implementation fully supports every example | The JS CLI and tests load files from the root `examples/` folder. A new "load all root examples" test iterates every file and asserts the run completes without error and produces the same numeric outputs as the language-agnostic fixtures file |
| R3 | The Rust implementation fully supports every example | The Rust integration tests likewise iterate every file under `../examples/` and check the same fixtures file |
| R4 | Every example is carefully tested | Each example has explicit per-file expectations stored in `examples/expected.json`. Both languages assert against this single fixtures file, so any drift between implementations fails the test suite in both |
| R5 | Demo files are also shared | `demo.lino` and `flipped-axioms.lino` were also byte-identical between languages and have moved to the root `examples/` folder. `npm run demo` and `cargo run` continue to work via updated paths |
| R6 | If anything is wrong we should fix it | While moving, we ran every example in both implementations and recorded the actual outputs into `expected.json`. The recorded outputs were sanity-checked against the inline assertions that already lived in the test suite |

## Repository layout (after this change)

```
/examples/
  README.md                     # index + how-to-run for each language
  expected.json                 # canonical numeric outputs for every .lino file
  classical-logic.lino
  propositional-logic.lino
  fuzzy-logic.lino
  ternary-kleene.lino
  belnap-four-valued.lino
  liar-paradox.lino
  liar-paradox-balanced.lino
  bayesian-inference.lino
  bayesian-network.lino
  markov-chain.lino
  markov-network.lino
  self-reasoning.lino
  dependent-types.lino
  demo.lino
  flipped-axioms.lino
```

## Existing components / libraries considered

- **LiNo (Links Notation)** — already a dependency of both implementations (`links-notation` on npm and crates.io). LiNo is the only thing that makes a shared examples folder possible: both the JS parser and the Rust parser accept exactly the same surface syntax.
- **Cargo's built-in examples directory** — Rust's tooling has special support for `examples/` *inside a crate* (`cargo run --example foo`). We deliberately did **not** put the shared examples there because the issue asks for the folder to live at the *repository root*, not inside the crate. Each example is a `.lino` data file rather than a Rust source file, so Cargo's mechanism doesn't apply anyway.
- **npm `files`/workspaces** — npm has no equivalent concept for `.lino` data files, so no special configuration is needed there either.

The simplest and most language-agnostic solution is therefore: keep the files at the repository root, and have each implementation's CLI / test harness read them from `../examples/` (relative to its language root).

## Solution summary

1. Moved every `.lino` file from `js/examples/` and `rust/examples/` to a single root-level `examples/` folder. Verified the JS and Rust copies were byte-identical before deleting the duplicates.
2. Moved `demo.lino` and `flipped-axioms.lino` to `examples/` for the same reason.
3. Added `examples/expected.json` capturing the canonical numeric outputs every example must produce.
4. Added `examples/README.md` listing each example and showing how to run it from either implementation.
5. Updated the JS CLI script (`npm run demo`) and the Rust tests / docs to point at the new path.
6. Added tests in **both** implementations that walk the `examples/` directory, run every `.lino` file, and compare results to `expected.json`. Any divergence between JS and Rust now fails the build in both languages.

## Test plan

- `cd js && npm test` — runs all 307+ JS tests, including the new "shared examples" test that loads every file from `../examples/` and validates against `expected.json`.
- `cd rust && cargo test` — runs all Rust tests, including the new shared-examples integration test that does the same check from Rust.
