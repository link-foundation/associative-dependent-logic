// Tests for the arithmetic standard library (issue #74).
// Mirrors rust/tests/lib_arithmetic_tests.rs so the LiNo library surface stays
// identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-arithmetic-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/arithmetic/core.lino', () => {
  it('exports Peano naturals and issue-surface arithmetic through an import alias', () => {
    const out = evaluateFromRoot(`
(import "lib/arithmetic/core.lino" as ar)
(? (zero of Natural))
(? (type of ar.succ))
(? ((plus zero zero) = zero))
(? (less-than zero (succ zero)))
(? (less-than-or-equal zero zero))
(? (ar.less-than zero (succ zero)))
(? (ar.less-than-or-equal zero zero))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [
      1,
      '(Pi (Natural n) Natural)',
      1,
      1,
      1,
      1,
      1,
    ]);
  });

  it('exports Peano axiom schemas as reusable templates', () => {
    const out = evaluateFromRoot(`
(import "lib/arithmetic/core.lino" as ar)
(? (ar.peano-zero-is-natural zero))
(? (ar.peano-successor-is-natural zero))
(? (ar.plus-zero-left zero))
(? (ar.plus-zero-right zero))
(? (ar.plus-successor-left zero zero))
(? (ar.less-than-successor zero))
(? (ar.less-than-or-equal-reflexive zero))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1, 1, 1]);
  });

  it('exports decimal-precision lemmas for built-in arithmetic', () => {
    const out = evaluateFromRoot(`
(import "lib/arithmetic/core.lino" as ar)
(? (ar.decimal-sum-equals 0.1 0.2 0.3))
(? (ar.decimal-difference-equals 0.3 0.1 0.2))
(? (ar.decimal-product-equals 0.1 0.2 0.02))
(? (ar.decimal-quotient-equals 1 3 0.333333333333))
(? (ar.less-than 0.1 0.2))
(? (ar.less-than-or-equal 0.3 (0.1 + 0.2)))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1, 1]);
  });
});
