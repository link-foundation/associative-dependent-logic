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
      'forall',
      'implication',
      'mul',
      'nat-add-succ',
      'nat-add-zero',
      'nat-cong-succ',
      'nat-eliminator',
      'nat-equality',
      'nat-induction',
      'nat-mul-succ',
      'nat-mul-zero',
      'nat-rec-succ',
      'nat-rec-zero',
      'nat-recursion',
      'nat-refl',
      'nat-succ-formation',
      'nat-zero-formation',
      'predicate-application',
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
      'nat-recursion',
      'nat-eliminator',
      'nat-rec-zero',
      'nat-rec-succ',
      'mul',
      'nat-mul-zero',
      'nat-mul-succ',
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
    // Phase 13 promotes the logical glue (forall, implication,
    // predicate-application) to first-class root constructs so the
    // trust audit can list them by name.
    for (const glue of ['forall', 'implication', 'predicate-application']) {
      assert.match(source, new RegExp(`\\(uses ${glue}\\)`));
      assert.match(
        source,
        new RegExp(`\\(root-construct ${glue}[\\s\\S]*?links-defined`),
        `${glue} should appear as a links-defined root construct`,
      );
    }
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

  it('nat-mul-zero discharges the base case of multiplication', () => {
    const out = evaluate(`
(rule nat-mul-zero
  (premise (?n has-type Nat))
  (conclusion ((mul zero ?n) nat-equals zero)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object zero-mul-zero
  (applies nat-mul-zero)
  (premise-by zero-is-nat)
  (conclusion ((mul zero zero) nat-equals zero)))

(check-proof zero-mul-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('nat-mul-succ steps multiplication through the successor', () => {
    // Two premises: the recursive product and the helper addition.
    const out = evaluate(`
(rule nat-mul-succ
  (premise ((mul ?m ?n) nat-equals ?k))
  (premise ((add ?n ?k) nat-equals ?s))
  (conclusion ((mul (succ ?m) ?n) nat-equals ?s)))

(axiom zero-mul-one
  (judgement ((mul zero (succ zero)) nat-equals zero)))

(axiom one-plus-zero-fact
  (judgement ((add (succ zero) zero) nat-equals (succ zero))))

(proof-object one-mul-one
  (applies nat-mul-succ)
  (premise-by zero-mul-one)
  (premise-by one-plus-zero-fact)
  (conclusion ((mul (succ zero) (succ zero)) nat-equals (succ zero))))

(check-proof one-mul-one)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('rejects a nat-mul-succ derivation whose helper addition is inconsistent', () => {
    // Claiming `(mul (succ zero) zero) nat-equals (succ zero)` would
    // require the helper addition `(add zero zero) nat-equals (succ zero)`
    // — impossible. The structural matcher rejects it with E064.
    const out = evaluate(`
(rule nat-mul-succ
  (premise ((mul ?m ?n) nat-equals ?k))
  (premise ((add ?n ?k) nat-equals ?s))
  (conclusion ((mul (succ ?m) ?n) nat-equals ?s)))

(axiom zero-mul-zero
  (judgement ((mul zero zero) nat-equals zero)))

(axiom zero-plus-zero-fact
  (judgement ((add zero zero) nat-equals zero)))

(proof-object wrong-one-mul-zero
  (applies nat-mul-succ)
  (premise-by zero-mul-zero)
  (premise-by zero-plus-zero-fact)
  (conclusion ((mul (succ zero) zero) nat-equals (succ zero))))

(check-proof wrong-one-mul-zero)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.ok(out.diagnostics.some(d => d.code === 'E064'));
  });

  it('nat-rec-zero discharges the recursor at the base case', () => {
    const out = evaluate(`
(rule nat-rec-zero
  (premise (?base has-type Nat))
  (conclusion (((rec ?f ?base ?step) at zero) nat-equals ?base)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object rec-id-at-zero
  (applies nat-rec-zero)
  (premise-by zero-is-nat)
  (conclusion (((rec id zero step) at zero) nat-equals zero)))

(check-proof rec-id-at-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('nat-rec-succ steps the recursor through the successor', () => {
    const out = evaluate(`
(rule nat-rec-succ
  (premise (((rec ?f ?base ?step) at ?n) nat-equals ?prev))
  (premise (((?step ?n) at ?prev) nat-equals ?next))
  (conclusion (((rec ?f ?base ?step) at (succ ?n)) nat-equals ?next)))

(axiom rec-id-at-zero
  (judgement (((rec id zero step) at zero) nat-equals zero)))

(axiom step-zero-applied
  (judgement (((step zero) at zero) nat-equals (succ zero))))

(proof-object rec-id-at-one
  (applies nat-rec-succ)
  (premise-by rec-id-at-zero)
  (premise-by step-zero-applied)
  (conclusion (((rec id zero step) at (succ zero)) nat-equals (succ zero))))

(check-proof rec-id-at-one)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('rejects a nat-rec-succ derivation that drops the successor wrapper on the scrutinee', () => {
    const out = evaluate(`
(rule nat-rec-succ
  (premise (((rec ?f ?base ?step) at ?n) nat-equals ?prev))
  (premise (((?step ?n) at ?prev) nat-equals ?next))
  (conclusion (((rec ?f ?base ?step) at (succ ?n)) nat-equals ?next)))

(axiom rec-id-at-zero
  (judgement (((rec id zero step) at zero) nat-equals zero)))

(axiom step-zero-applied
  (judgement (((step zero) at zero) nat-equals (succ zero))))

(proof-object bad-rec
  (applies nat-rec-succ)
  (premise-by rec-id-at-zero)
  (premise-by step-zero-applied)
  (conclusion (((rec id zero step) at zero) nat-equals (succ zero))))

(check-proof bad-rec)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.ok(out.diagnostics.some(d => d.code === 'E064'));
  });

  it('(proof-report <name>) returns a per-proof dependency/trust summary', () => {
    // Phase 13 (issue #97): `(proof-report ...)` walks the proof-object
    // tree and reports the dependencies, rules, and root constructs the
    // proof touches — together with their semantic and trust statuses,
    // so the trust audit can be done per-proof instead of only
    // globally. We register `Nat`, `zero`, `succ` as links-defined root
    // constructs first so the report can cite them.
    const out = evaluate(`
(root-construct Nat
  (kind inductive-type)
  (status links-defined)
  (semantic-status links-checked))

(root-construct zero
  (kind constructor)
  (status links-defined)
  (semantic-status links-checked)
  (depends-on Nat))

(root-construct succ
  (kind constructor)
  (status links-defined)
  (semantic-status links-checked)
  (depends-on Nat))

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

(proof-report one-is-nat)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.strictEqual(out.results.length, 1);
    const report = out.results[0];
    assert.strictEqual(report.kind, 'proof-report');
    assert.strictEqual(report.name, 'one-is-nat');
    assert.strictEqual(report.rule, 'nat-succ-formation');
    assert.strictEqual(report.verdict.ok, true);
    assert.deepStrictEqual(report.premiseRefs, ['zero-is-nat']);
    assert.deepStrictEqual(report.rules.slice().sort(), [
      'nat-succ-formation',
      'nat-zero-formation',
    ]);
    const depNames = report.dependencies.map(d => d.name);
    assert.ok(depNames.includes('zero-is-nat'));
    assert.ok(report.rootConstructsUsed.includes('Nat'));
    assert.ok(report.rootConstructsUsed.includes('succ'));
    assert.ok(report.rootConstructsUsed.includes('zero'));
    assert.deepStrictEqual(
      report.bySemanticStatus['links-checked'].slice().sort(),
      ['Nat', 'succ', 'zero'],
    );
    assert.deepStrictEqual(
      report.byTrustStatus['links-defined'].slice().sort(),
      ['Nat', 'succ', 'zero'],
    );
  });

  it('(proof-report <unknown>) reports a failing verdict instead of throwing', () => {
    const out = evaluate(`(proof-report no-such-proof)`);
    assert.strictEqual(out.results.length, 1);
    const report = out.results[0];
    assert.strictEqual(report.kind, 'proof-report');
    assert.strictEqual(report.name, 'no-such-proof');
    assert.strictEqual(report.verdict.ok, false);
    assert.match(report.verdict.error, /unknown proof-object/);
  });

  it('runs the full Phase 12/13 example end-to-end with no diagnostics', () => {
    const out = evaluateFile(examplePath);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ]);
  });
});
