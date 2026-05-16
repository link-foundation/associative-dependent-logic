// Phase 5 — links-defined typed-kernel fragment (issue #97).
//
// These tests pin down the behaviour of the four typed-kernel rules
// (pi-formation, lambda-introduction, application-elimination,
// beta-conversion) that are expressed inside the proof substrate by
// `examples/typed-kernel-links.lino`. Each rule is exercised on its
// own, then composed end-to-end, and finally the corresponding
// `(foundation typed-kernel-links ...)` registration is verified so the
// data and runtime stay in sync.

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
const examplePath = join(repoRoot, 'examples', 'typed-kernel-links.lino');
const foundationsPath = join(repoRoot, 'lib', 'self', 'foundations.lino');

describe('Phase 5 — links-defined typed kernel', () => {
  it('typed-kernel-links foundation is pre-registered', () => {
    const env = new Env();
    const report = env.foundationReport();
    const found = report.foundations.find(f => f.name === 'typed-kernel-links');
    assert.ok(found, 'typed-kernel-links foundation must be registered');
    assert.deepStrictEqual(found.uses.slice().sort(), [
      'application-elimination',
      'beta-conversion',
      'lambda-introduction',
      'pi-formation',
    ]);
  });

  it('lib/self/foundations.lino documents the typed-kernel-links foundation', () => {
    const source = readFileSync(foundationsPath, 'utf8');
    assert.match(source, /\(foundation typed-kernel-links/);
    for (const rule of [
      'pi-formation',
      'lambda-introduction',
      'application-elimination',
      'beta-conversion',
    ]) {
      assert.match(source, new RegExp(`\\(uses ${rule}\\)`));
      assert.match(
        source,
        new RegExp(`\\(root-construct ${rule}[\\s\\S]*?links-defined`),
        `${rule} should appear as a root-construct with links-defined status`,
      );
    }
  });

  it('pi-formation derives a well-formed Pi type', () => {
    const out = evaluate(`
(rule pi-formation
  (premise (?Gamma turnstile (?A has-type Type0)))
  (premise ((?Gamma comma (?x has-type ?A)) turnstile (?B has-type Type0)))
  (conclusion (?Gamma turnstile ((Pi (?x has-type ?A) ?B) has-type Type0))))

(axiom nat-is-type
  (judgement (empty turnstile (Nat has-type Type0))))

(axiom nat-is-type-under-x
  (judgement ((empty comma (x has-type Nat)) turnstile (Nat has-type Type0))))

(proof-object pi-nat-nat
  (applies pi-formation)
  (premise-by nat-is-type)
  (premise-by nat-is-type-under-x)
  (conclusion (empty turnstile ((Pi (x has-type Nat) Nat) has-type Type0))))

(check-proof pi-nat-nat)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('lambda-introduction types the identity function on Nat', () => {
    const out = evaluate(`
(rule lambda-introduction
  (premise ((?Gamma comma (?x has-type ?A)) turnstile (?body has-type ?B)))
  (conclusion (?Gamma turnstile ((lambda (?x has-type ?A) ?body) has-type (Pi (?x has-type ?A) ?B)))))

(axiom x-is-nat
  (judgement ((empty comma (x has-type Nat)) turnstile (x has-type Nat))))

(proof-object id-nat-typed
  (applies lambda-introduction)
  (premise-by x-is-nat)
  (conclusion (empty turnstile ((lambda (x has-type Nat) x) has-type (Pi (x has-type Nat) Nat)))))

(check-proof id-nat-typed)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('application-elimination applies a typed function and substitutes the codomain', () => {
    const out = evaluate(`
(rule application-elimination
  (premise (?Gamma turnstile (?f has-type (Pi (?x has-type ?A) ?B))))
  (premise (?Gamma turnstile (?arg has-type ?A)))
  (conclusion (?Gamma turnstile ((apply ?f ?arg) has-type (subst ?B ?x ?arg)))))

(axiom id-typed
  (judgement (empty turnstile ((lambda (x has-type Nat) x) has-type (Pi (x has-type Nat) Nat)))))

(axiom zero-is-nat
  (judgement (empty turnstile (zero has-type Nat))))

(proof-object app-id-zero
  (applies application-elimination)
  (premise-by id-typed)
  (premise-by zero-is-nat)
  (conclusion (empty turnstile ((apply (lambda (x has-type Nat) x) zero) has-type (subst Nat x zero)))))

(check-proof app-id-zero)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('beta-conversion preserves typing through a redex reduction', () => {
    const out = evaluate(`
(rule beta-conversion
  (premise (?Gamma turnstile (?redex has-type ?B)))
  (premise (?redex beta-reduces-to ?reduct))
  (conclusion (?Gamma turnstile (?reduct has-type ?B))))

(axiom app-id-zero-typed
  (judgement (empty turnstile ((apply (lambda (x has-type Nat) x) zero) has-type (subst Nat x zero)))))

(axiom id-zero-beta
  (judgement ((apply (lambda (x has-type Nat) x) zero) beta-reduces-to zero)))

(proof-object zero-after-beta
  (applies beta-conversion)
  (premise-by app-id-zero-typed)
  (premise-by id-zero-beta)
  (conclusion (empty turnstile (zero has-type (subst Nat x zero)))))

(check-proof zero-after-beta)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('rejects a derivation that contradicts the typing rule', () => {
    // The conclusion of lambda-introduction must be a Pi type whose
    // domain matches the bound variable's annotation; a swapped domain
    // must therefore be rejected.
    const out = evaluate(`
(rule lambda-introduction
  (premise ((?Gamma comma (?x has-type ?A)) turnstile (?body has-type ?B)))
  (conclusion (?Gamma turnstile ((lambda (?x has-type ?A) ?body) has-type (Pi (?x has-type ?A) ?B)))))

(axiom x-is-nat
  (judgement ((empty comma (x has-type Nat)) turnstile (x has-type Nat))))

(proof-object id-mistyped
  (applies lambda-introduction)
  (premise-by x-is-nat)
  (conclusion (empty turnstile ((lambda (x has-type Nat) x) has-type (Pi (x has-type Bool) Nat)))))

(check-proof id-mistyped)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.ok(out.diagnostics.some(d => d.code === 'E064'));
  });

  it('runs the full Phase 5 example end-to-end with no diagnostics', () => {
    const out = evaluateFile(examplePath);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1]);
  });
});
