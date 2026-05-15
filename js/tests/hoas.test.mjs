// HOAS desugaring (issue #51).
//
// `forall` is surface sugar for `Pi`. The desugarer rewrites the head leaf at
// the AST level so every downstream pass (typing, beta, definitional equality,
// proofs) only ever sees `Pi`. These tests pin the rewrite at every entry
// point that consumes user-supplied terms.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { evaluate, synth, isConvertible, Env } from '../src/rml-links.mjs';

function evaluateClean(src) {
  const out = evaluate(src);
  assert.deepStrictEqual(out.diagnostics, []);
  return out.results;
}

describe('HOAS forall desugaring', () => {
  it('classifies a lambda under both Pi and forall spellings', () => {
    const results = evaluateClean(`
(Term: (Type 0) Term)
(identity: lambda (Term x) x)
(? (identity of (Pi     (Term x) Term)))
(? (identity of (forall (Term x) Term)))
`);
    assert.deepStrictEqual(results, [1, 1]);
  });

  it('records a forall-typed declaration as a Pi type', () => {
    const results = evaluateClean(`
(Natural: (Type 0) Natural)
(succ: (forall (Natural n) Natural))
(? (type of succ))
`);
    assert.deepStrictEqual(results, ['(Pi (Natural n) Natural)']);
  });

  it('treats forall as Pi in synth (host API)', () => {
    const env = new Env();
    evaluate(`(Term: (Type 0) Term)`, undefined, env);
    const piType = synth('(Pi (Term x) Term)', env);
    const forallType = synth('(forall (Term x) Term)', env);
    assert.deepStrictEqual(piType, forallType);
  });

  it('treats forall as Pi in isConvertible (host API)', () => {
    const env = new Env();
    evaluate(`(Term: (Type 0) Term)`, undefined, env);
    assert.strictEqual(
      isConvertible('(Pi (Term x) Term)', '(forall (Term x) Term)', env),
      true,
    );
  });

  it('desugars forall nested inside a lambda body', () => {
    const results = evaluateClean(`
(Natural: (Type 0) Natural)
(zero: Natural zero)
(? (apply (lambda (Natural x) (forall (Natural y) Natural)) zero))
`);
    // Result is a type, surfaced as a beta-reduced numeric query (1 = well-typed).
    // The forall must be desugared even when it sits underneath a binder.
    assert.deepStrictEqual(results, [1]);
  });

  it('does not rewrite forall when it is not the head leaf', () => {
    // A spurious 3-element list whose head is not the leaf "forall" must pass
    // through untouched. Guards against over-eager rewriting.
    const results = evaluateClean(`
(Natural: (Type 0) Natural)
(zero: Natural zero)
(? ((forall x) = (forall x)))
`);
    assert.deepStrictEqual(results, [1]);
  });
});
