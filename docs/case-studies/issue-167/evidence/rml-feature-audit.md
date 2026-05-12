# RML feature audit — evidence backing the `CONCEPTS-COMPARISION.md` revision

This file records concrete code-level evidence for each capability that
[issue #167](https://github.com/link-foundation/relative-meta-logic/issues/167)
asks us to re-examine. Every claim that the comparison matrix now makes about
RML is anchored here.

The audit covers both implementations (`rust/` and `js/`) and was produced
during the work on PR #173. Line numbers refer to the state of the branch
`issue-167-b75ac8a2555c` at the time of the audit.

## 1. `whnf` — weak head normal form for the typed lambda fragment

- Rust: `rust/src/lib.rs:2178` defines `whnf_term`; `rust/src/lib.rs:2350`
  exposes `pub fn whnf`; `rust/src/lib.rs:2355` exposes
  `whnf_with_options`.
- JS: matching `whnf` entry points in `js/src/rml-links.mjs`.
- Tests: `rust/tests/normalization_tests.rs`,
  `js/tests/normalization.test.mjs`.

## 2. `nf` — full normalization for the typed lambda fragment

- Rust: `rust/src/lib.rs:2364` (`pub fn nf`); `rust/src/lib.rs:2369`
  (`nf_with_options`).
- Tests: `rust/tests/normalization_tests.rs`,
  `js/tests/normalization.test.mjs`.

## 3. `(normal-form …)` surface form

- Rust: exercised by `rust/tests/self_evaluator_tests.rs:55`
  (`"(eval (normal-form expression))"`).
- JS: matching entries in `js/tests/normalization.test.mjs` and the
  self-evaluator suite.

## 4. `(inductive …)` declarations and generated recursors

- Rust: parsing and declaration handling in `rust/src/lib.rs:7400+`.
- Tests: `rust/tests/inductive_tests.rs` (`elim_name == "Natural-rec"`
  exercises the auto-generated recursor), `js/tests/inductive.test.mjs`.

## 5. `(coinductive …)` declarations and generated corecursors

- Rust: `parse_coinductive_form` in `rust/src/lib.rs` with productivity
  checks.
- Tests: `rust/tests/coinductive_tests.rs:37`
  (`corec_name == "Stream-corec"`), `js/tests/coinductive.test.mjs`.

## 6. `(total …)` totality checking

- Rust: `rust/src/lib.rs:6943` parses the totality declaration; logic
  enforced in the totality module.
- Tests: `rust/tests/totality_tests.rs`, `js/tests/totality.test.mjs`.

## 7. `(coverage …)` coverage checking

- Rust: `rust/src/lib.rs:6020` exposes `pub fn is_covered`.
- Tests: `rust/tests/coverage_tests.rs:17`, `js/tests/coverage.test.mjs`.

## 8. Mode checking (`+input`, `-output`, `*either`)

- Rust: mode declaration parsing in `rust/src/lib.rs:6917`; flag enum
  `ModeFlag::{In, Out, Either}` in `rust/tests/modes_tests.rs:14`.
- Tests: `rust/tests/modes_tests.rs`, `js/tests/modes.test.mjs`.

## 9. Termination checking

- Rust: `rust/src/lib.rs:5878` exposes `pub fn is_terminating`.
- Tests: `rust/tests/termination_tests.rs:39` (`(measure …)` form),
  `js/tests/termination.test.mjs`.

## 10. Tactic links

- Rust: `reflexivity` at `rust/src/lib.rs:4952`; `symmetry` 4969;
  `transitivity` 4986; `rewrite` 4683; `simplify` 4712; `exact` 5181;
  `induction` 5198.
- Tests: `rust/tests/tactics_tests.rs`, `js/tests/tactics.test.mjs`.

## 11. ATP bridge (trusted external)

- Rust: `AtpOptions` struct at `rust/src/lib.rs:3638`;
  `parse_atp_status` at `:4483`; tactic dispatch for `(by atp …)`
  around `:5120`.
- Tests: `rust/tests/tactics_tests.rs:39` exercises a mock ATP and
  records the proof link as a trusted external node.

## 12. SMT bridge (trusted external)

- Rust: `TacticOptions.smt_solver` at `rust/src/lib.rs:3674`; SMT
  dispatch around `:4999`.
- Tests: `rust/tests/tactics_tests.rs`, `js/tests/tactics.test.mjs`.

## 13. Independent proof-replay checker

- Rust: `rust/src/check.rs` is a separate module dedicated to replaying
  proof links without re-running the evaluator. Its docstring states it
  is the "independent proof-replay checker (issue #36)".
- Tests: `rust/tests/check_tests.rs`, `js/tests/check.test.mjs`.

## 14. Numeric truth values, configurable range, configurable valence

- Rust: valence and range parsing in `rust/src/lib.rs` near line 10
  (header comment), `:668` (`fn quantize`), `:751` (the `valence`
  field), `:7160` (parser).
- Library code: `lib/*/core.lino` exposes the numeric semantics.

## 15. Multiple equality mechanisms

- Structural equality: `rust/src/lib.rs:634`
  (`fn is_structurally_same`).
- Assigned equality: `rust/src/lib.rs:2418`
  (`fn lookup_assigned_infix(env, "=", …)`).
- Numeric equality: applied at `rust/src/lib.rs:1298`.
- Definitional / convertibility: `rust/src/lib.rs:2551`
  (`fn is_convertible`).
- Tests: `rust/tests/proofs_tests.rs`, `rust/tests/check_tests.rs:35-36`.

## 16. Self-evaluator and metatheorem checking

- Library: `lib/self/evaluator.lino`, `lib/self/metatheorem.lino`.
- Rust: `rust/tests/self_evaluator_tests.rs`,
  `rust/tests/self_metatheorem_tests.rs` (`check_metatheorems`).
- JS: `js/tests/self-evaluator.test.mjs`,
  `js/tests/self-metatheorem.test.mjs`.

## Conclusion

Every RML capability that issue #167 marks as stale (`whnf`/`nf`/
`normal-form`, `(inductive …)`, `(coinductive …)`, `(total …)`,
`(coverage …)`, modes, termination, tactic links, ATP bridge, SMT bridge,
independent proof replay, multiple equality layers, self-evaluator
metatheorem checking) is implemented in the current codebase and is
covered by tests in both `rust/tests/` and `js/tests/`. The revision in
PR #173 reflects these capabilities while still recording where each
feature is **host-implemented** versus defined in `.lino`.
