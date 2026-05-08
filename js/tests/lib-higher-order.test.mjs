// Tests for the higher-order standard library (issue #70).
// Mirrors rust/tests/lib_higher_order_tests.rs so the LiNo library surface stays
// identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-higher-order-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/higher-order/core.lino', () => {
  it('exports forall as an alias-qualified template for predicate binders', () => {
    const out = evaluateFromRoot(`
(import "lib/higher-order/core.lino" as ho)
(Natural: (Type 0) Natural)
(zero: Natural)
(succ: (Pi (Natural n) Natural))
(? ((ho.forall ((Pi (Natural n) Boolean) P)
       ((P zero) implies (ho.forall (Natural n) (P (succ n)))))
     =
     (Pi ((Pi (Natural n) Boolean) P)
       ((P zero) implies (Pi (Natural n) (P (succ n)))))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('exports exists as an alias-qualified template for predicate binders', () => {
    const out = evaluateFromRoot(`
(import "lib/higher-order/core.lino" as ho)
(Natural: (Type 0) Natural)
(zero: Natural)
(? ((ho.exists ((Pi (Natural n) Boolean) P) (P zero))
    =
    (exists ((Pi (Natural n) Boolean) P) (P zero))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1]);
  });
});
