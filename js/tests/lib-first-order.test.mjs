// Tests for the first-order standard library (issue #69).
// Mirrors rust/tests/lib_first_order_tests.rs so the LiNo library surface stays
// identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-first-order-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/first-order/core.lino', () => {
  it('exports forall as an alias-qualified template for Pi', () => {
    const out = evaluateFromRoot(`
(import "lib/first-order/core.lino" as fo)
(Term: (Type 0) Term)
(? ((fo.forall (Term x) (predicate x)) = (Pi (Term x) (predicate x))))
(? (fo.forall (Term x) Term))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1]);
  });

  it('exports exists as an alias-qualified first-order link shape', () => {
    const out = evaluateFromRoot(`
(import "lib/first-order/core.lino" as fo)
(Term: (Type 0) Term)
(? ((fo.exists (Term x) (predicate x)) = (exists (Term x) (predicate x))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('expands nested quantified formulas before evaluation', () => {
    const out = evaluateFromRoot(`
(import "lib/first-order/core.lino" as fo)
(Term: (Type 0) Term)
(? ((fo.forall (Term x)
       (fo.exists (Term y) ((pair x y) = (pair x y))))
     =
     (Pi (Term x)
       (exists (Term y) ((pair x y) = (pair x y))))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1]);
  });
});
