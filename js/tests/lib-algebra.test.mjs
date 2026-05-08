// Tests for the algebra standard library (issue #75).
// Mirrors rust/tests/lib_algebra_tests.rs so the LiNo library surface stays
// identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-algebra-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/algebra/core.lino', () => {
  it('exports magma, monoid, and the issue-surface group through an import alias', () => {
    const out = evaluateFromRoot(`
(import "lib/algebra/core.lino" as al)
(? (al.and true true))
(? (al.and true false))
(? (al.group (carrier G) (op times) (identity e) (inverse inv)))
(? ((al.magma (carrier G) (op times)) =
     (al.closed-under (carrier G) (op times))))
(? ((al.monoid (carrier G) (op times) (identity e)) =
     (algebra.and
       (al.semigroup (carrier G) (op times))
       (al.identity-element (carrier G) (op times) (identity e)))))
(? ((al.group (carrier G) (op times) (identity e) (inverse inv)) =
     (algebra.and
       (al.monoid (carrier G) (op times) (identity e))
       (al.inverse-operation (carrier G) (op times) (identity e) (inverse inv)))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 0, 1, 1, 1, 1]);
  });

  it('exports reusable operation-law schemas', () => {
    const out = evaluateFromRoot(`
(import "lib/algebra/core.lino" as al)
(? ((al.closed-under G times) =
     (forall (G left)
       (forall (G right)
         ((times left right) of G)))))
(? ((al.associative G times) =
     (forall (G left)
       (forall (G middle)
         (forall (G right)
           (= (times (times left middle) right)
              (times left (times middle right))))))))
(? ((al.identity-element G times e) =
     (algebra.and
       (al.left-identity-law G times e)
       (al.right-identity-law G times e))))
(? ((al.inverse-operation G times e inv) =
     (algebra.and
       (al.left-inverse-law G times e inv)
       (al.right-inverse-law G times e inv))))
(? ((al.commutative G times) =
     (forall (G left)
       (forall (G right)
         (= (times left right) (times right left))))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1]);
  });

  it('exports ring schemas from additive group and multiplicative monoid pieces', () => {
    const out = evaluateFromRoot(`
(import "lib/algebra/core.lino" as al)
(? ((al.distributive R plus times) =
     (algebra.and
       (al.left-distributive R plus times)
       (al.right-distributive R plus times))))
(? ((al.ring R plus zero neg times one) =
     (algebra.and
       (al.abelian-group R plus zero neg)
       (al.monoid R times one)
       (al.distributive R plus times))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1]);
  });
});
