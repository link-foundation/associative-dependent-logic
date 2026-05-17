// Phase 12 — links-defined Peano naturals (issue #97).
//
// These tests pin down the behaviour of the five proof-substrate rules
// (nat-zero-formation, nat-succ-formation, nat-add-zero, nat-add-succ,
// nat-induction) that are expressed inside the proof substrate by
// `examples/nat-links.lino`. Each rule is exercised on its own, then
// composed end-to-end, and finally the corresponding
// `(foundation nat-links ...)` registration is verified so the data
// and the runtime stay in sync.

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
      'nat-induction',
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
    ]) {
      assert.match(source, new RegExp(`\\(uses ${rule}\\)`));
      assert.match(
        source,
        new RegExp(`\\(root-construct ${rule}[\\s\\S]*?links-defined`),
        `${rule} should appear as a root-construct with links-defined status`,
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
  (conclusion ((add zero ?n) equals ?n)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object zero-plus-zero
  (applies nat-add-zero)
  (premise-by zero-is-nat)
  (conclusion ((add zero zero) equals zero)))

(check-proof zero-plus-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('nat-add-succ steps addition through the successor', () => {
    const out = evaluate(`
(rule nat-add-succ
  (premise ((add ?m ?n) equals ?k))
  (conclusion ((add (succ ?m) ?n) equals (succ ?k))))

(axiom zero-plus-zero
  (judgement ((add zero zero) equals zero)))

(proof-object one-plus-zero
  (applies nat-add-succ)
  (premise-by zero-plus-zero)
  (conclusion ((add (succ zero) zero) equals (succ zero))))

(check-proof one-plus-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
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
    // Claiming that `(add (succ zero) zero)` equals `zero` would
    // require `(succ ?k)` to unify with `zero`, which is impossible.
    const out = evaluate(`
(rule nat-add-succ
  (premise ((add ?m ?n) equals ?k))
  (conclusion ((add (succ ?m) ?n) equals (succ ?k))))

(axiom zero-plus-zero
  (judgement ((add zero zero) equals zero)))

(proof-object wrong-add
  (applies nat-add-succ)
  (premise-by zero-plus-zero)
  (conclusion ((add (succ zero) zero) equals zero)))

(check-proof wrong-add)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.ok(out.diagnostics.some(d => d.code === 'E064'));
  });

  it('runs the full Phase 12 example end-to-end with no diagnostics', () => {
    const out = evaluateFile(examplePath);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1, 1, 1, 1]);
  });
});
