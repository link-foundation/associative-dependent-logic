// Phase 12 — links-defined Peano naturals (issue #97).
//
// These tests pin down the behaviour of the seven proof-substrate rules
// (nat-zero-formation, nat-succ-formation, nat-add-zero, nat-add-succ,
// nat-induction, nat-refl, nat-cong-succ) that are expressed inside the
// proof substrate by `examples/nat-links.lino`, and of the dedicated
// equality layer `nat-equality` they inhabit. Each rule is exercised on
// its own, then composed end-to-end, and finally the corresponding
// `(foundation nat-links ...)` registration is verified so the data
// and the runtime stay in sync.
//
// PR 178 added the explicit `nat-equality` layer plus the rules
// `nat-refl` and `nat-cong-succ`, and switched the example from the
// bare literal `equals` to `nat-equals`. The tests below also check
// that programs which never opt into `nat-links` continue to use the
// host's `=`/`numeric-equality` layer unchanged (backward
// compatibility).

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import {
  Env,
  evaluate,
  evaluateFile,
} from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const examplePath = join(repoRoot, 'examples', 'nat-links.lino');
const foundationsPath = join(repoRoot, 'lib', 'self', 'foundations.lino');

describe('Phase 12 — links-defined Peano naturals', () => {
  it('nat-links foundation is pre-registered', () => {
    const env = new Env();
    const report = env.foundationReport();
    const found = report.foundations.find(f => f.name === 'nat-links');
    assert.ok(found, 'nat-links foundation must be registered');
    assert.deepStrictEqual(found.uses.slice().sort(), [
      'nat-add-succ',
      'nat-add-zero',
      'nat-cong-succ',
      'nat-equality',
      'nat-induction',
      'nat-refl',
      'nat-succ-formation',
      'nat-zero-formation',
    ]);
    assert.strictEqual(found.extends, 'default-rml');
  });

  it('lib/self/foundations.lino documents the nat-links foundation', () => {
    const source = readFileSync(foundationsPath, 'utf8');
    assert.match(source, /\(foundation nat-links/);
    for (const rule of [
      'nat-zero-formation',
      'nat-succ-formation',
      'nat-add-zero',
      'nat-add-succ',
      'nat-induction',
      'nat-refl',
      'nat-cong-succ',
    ]) {
      assert.match(source, new RegExp(`\\(uses ${rule}\\)`));
      assert.match(
        source,
        new RegExp(`\\(root-construct ${rule}[\\s\\S]*?links-defined`),
        `${rule} should appear as a root-construct with links-defined status`,
      );
    }
    // The equality layer that the new rules inhabit is registered as a
    // root construct, listed by the nat-links foundation, and marked
    // links-defined so the trust audit can distinguish it from the
    // host's `=`/`numeric-equality`.
    assert.match(source, /\(uses nat-equality\)/);
    assert.match(
      source,
      /\(root-construct nat-equality[\s\S]*?equality-layer[\s\S]*?links-defined/,
      'nat-equality should be a links-defined equality-layer root construct',
    );
  });

  it('nat-zero-formation derives `zero has-type Nat` without premises', () => {
    const out = evaluate(`
(rule nat-zero-formation
  (conclusion (zero has-type Nat)))

(proof-object zero-is-nat
  (applies nat-zero-formation)
  (conclusion (zero has-type Nat)))

(check-proof zero-is-nat)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('nat-succ-formation lifts a Nat to its successor', () => {
    const out = evaluate(`
(rule nat-zero-formation
  (conclusion (zero has-type Nat)))

(rule nat-succ-formation
  (premise (?n has-type Nat))
  (conclusion ((succ ?n) has-type Nat)))

(proof-object zero-is-nat
  (applies nat-zero-formation)
  (conclusion (zero has-type Nat)))

(proof-object one-is-nat
  (applies nat-succ-formation)
  (premise-by zero-is-nat)
  (conclusion ((succ zero) has-type Nat)))

(check-proof one-is-nat)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('nat-add-zero discharges the base case of addition', () => {
    const out = evaluate(`
(rule nat-add-zero
  (premise (?n has-type Nat))
  (conclusion ((add zero ?n) nat-equals ?n)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object zero-plus-zero
  (applies nat-add-zero)
  (premise-by zero-is-nat)
  (conclusion ((add zero zero) nat-equals zero)))

(check-proof zero-plus-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('nat-add-succ steps addition through the successor', () => {
    const out = evaluate(`
(rule nat-add-succ
  (premise ((add ?m ?n) nat-equals ?k))
  (conclusion ((add (succ ?m) ?n) nat-equals (succ ?k))))

(axiom zero-plus-zero
  (judgement ((add zero zero) nat-equals zero)))

(proof-object one-plus-zero
  (applies nat-add-succ)
  (premise-by zero-plus-zero)
  (conclusion ((add (succ zero) zero) nat-equals (succ zero))))

(check-proof one-plus-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('nat-refl derives (?n nat-equals ?n) for a well-typed Nat', () => {
    // Reflexivity of the links-defined equality layer: every
    // inhabitant of `Nat` is `nat-equals` to itself. The premise pins
    // the metavariable `?n` to `zero`, so the structural matcher
    // accepts `(zero nat-equals zero)` as the conclusion.
    const out = evaluate(`
(rule nat-refl
  (premise (?n has-type Nat))
  (conclusion (?n nat-equals ?n)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object zero-nat-equals-zero
  (applies nat-refl)
  (premise-by zero-is-nat)
  (conclusion (zero nat-equals zero)))

(check-proof zero-nat-equals-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('nat-cong-succ lifts a nat-equality through succ', () => {
    // Successor congruence: if `?m nat-equals ?n` is derived, then
    // applying `succ` to both sides yields `(succ ?m) nat-equals (succ
    // ?n)`. The axiom below stands in for the premise so the test
    // exercises `nat-cong-succ` in isolation.
    const out = evaluate(`
(rule nat-cong-succ
  (premise (?m nat-equals ?n))
  (conclusion ((succ ?m) nat-equals (succ ?n))))

(axiom zero-nat-equals-zero
  (judgement (zero nat-equals zero)))

(proof-object succ-zero-nat-equals-succ-zero
  (applies nat-cong-succ)
  (premise-by zero-nat-equals-zero)
  (conclusion ((succ zero) nat-equals (succ zero))))

(check-proof succ-zero-nat-equals-succ-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('rejects a nat-cong-succ derivation that drops one of the succ wrappers', () => {
    // The conclusion of `nat-cong-succ` must wrap both sides of the
    // premise in `succ`; deliberately dropping the wrapper on one side
    // breaks the structural match and `(check-proof ...)` must return
    // 0 with E064.
    const out = evaluate(`
(rule nat-cong-succ
  (premise (?m nat-equals ?n))
  (conclusion ((succ ?m) nat-equals (succ ?n))))

(axiom zero-nat-equals-zero
  (judgement (zero nat-equals zero)))

(proof-object bad-cong
  (applies nat-cong-succ)
  (premise-by zero-nat-equals-zero)
  (conclusion ((succ zero) nat-equals zero)))

(check-proof bad-cong)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.ok(out.diagnostics.some(d => d.code === 'E064'));
  });

  it('nat-induction folds a base case and a step into a universal claim', () => {
    const out = evaluate(`
(rule nat-induction
  (premise (?P at zero))
  (premise (forall ?n (implies (?P at ?n) (?P at (succ ?n)))))
  (conclusion (forall ?n (?P at ?n))))

(axiom is-nat-at-zero
  (judgement (is-nat at zero)))

(axiom is-nat-step
  (judgement (forall n (implies (is-nat at n) (is-nat at (succ n))))))

(proof-object every-nat-is-nat
  (applies nat-induction)
  (premise-by is-nat-at-zero)
  (premise-by is-nat-step)
  (conclusion (forall n (is-nat at n))))

(check-proof every-nat-is-nat)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('rejects a derivation that contradicts nat-succ-formation', () => {
    // The conclusion of `nat-succ-formation` must keep `Nat` as the
    // type of `(succ ?n)`; swapping the type to `Bool` therefore
    // breaks the match and `(check-proof ...)` must return 0 with E064.
    const out = evaluate(`
(rule nat-succ-formation
  (premise (?n has-type Nat))
  (conclusion ((succ ?n) has-type Nat)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object mistyped-succ
  (applies nat-succ-formation)
  (premise-by zero-is-nat)
  (conclusion ((succ zero) has-type Bool)))

(check-proof mistyped-succ)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.ok(out.diagnostics.some(d => d.code === 'E064'));
  });

  it('rejects an add-succ derivation that contradicts its arithmetic premise', () => {
    // Claiming that `(add (succ zero) zero) nat-equals zero` would
    // require `(succ ?k)` to unify with `zero`, which is impossible.
    const out = evaluate(`
(rule nat-add-succ
  (premise ((add ?m ?n) nat-equals ?k))
  (conclusion ((add (succ ?m) ?n) nat-equals (succ ?k))))

(axiom zero-plus-zero
  (judgement ((add zero zero) nat-equals zero)))

(proof-object wrong-add
  (applies nat-add-succ)
  (premise-by zero-plus-zero)
  (conclusion ((add (succ zero) zero) nat-equals zero)))

(check-proof wrong-add)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.ok(out.diagnostics.some(d => d.code === 'E064'));
  });

  it('leaves the host `=`/numeric-equality layer unchanged when nat-links is not selected', () => {
    // PR 178 (issue #97) introduces `nat-equality` as an additional
    // links-defined layer; programs that never opt into the nat-links
    // foundation must keep the host's decimal-12 `=` semantics.
    const out = evaluate(`
(? (= 1 1))
(? (= 1 2))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    // `(= 1 1)` ⇒ 1; `(= 1 2)` ⇒ 0.
    assert.deepStrictEqual(out.results, [1, 0]);
  });

  it('runs the full Phase 12 example end-to-end with no diagnostics', () => {
    const out = evaluateFile(examplePath);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
  });
});
