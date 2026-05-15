// Tests for the set-theory standard library (issue #73).
// Mirrors rust/tests/lib_set_theory_tests.rs so the LiNo library surface stays
// identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-set-theory-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/set-theory/core.lino', () => {
  it('exports membership and core set constructors through an import alias', () => {
    const out = evaluateFromRoot(`
(import "lib/set-theory/core.lino" as st)
(? ((st.member-of x (set-of-naturals)) = (member-of x (set-of-naturals))))
(? ((st.singleton x) = (singleton x)))
(? ((st.unordered-pair x y) = (unordered-pair x y)))
(? ((st.union family) = (union family)))
(? ((st.separation source predicate) = (separation source predicate)))
(? ((st.replacement source mapping) = (replacement source mapping)))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1, 1]);
  });

  it('exports extensionality and pairing schemas as reusable templates', () => {
    const out = evaluateFromRoot(`
(import "lib/set-theory/core.lino" as st)
(? ((st.subset-of left right) =
     (forall (Set element)
       (st.implies
         (st.member-of element left)
         (st.member-of element right)))))
(? ((st.same-set left right) =
     (forall (Set element)
       (st.iff
         (st.member-of element left)
         (st.member-of element right)))))
(? ((st.axiom-extensionality left right) =
     (st.implies
       (st.same-set left right)
       (= left right))))
(? ((st.axiom-pairing left right pair-set) =
     (forall (Set element)
       (st.iff
         (st.member-of element pair-set)
         (st.pair-membership element left right)))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1]);
  });

  it('exports union, separation, replacement, and infinity schemas', () => {
    const out = evaluateFromRoot(`
(import "lib/set-theory/core.lino" as st)
(? ((st.axiom-union collection union-set) =
     (forall (Set element)
       (st.iff
         (st.member-of element union-set)
         (exists (Set member-set)
           (set-theory.and
             (st.member-of element member-set)
             (st.member-of member-set collection)))))))
(? ((st.axiom-separation source predicate subset) =
     (forall (Set element)
       (st.iff
         (st.member-of element subset)
         (set-theory.and
           (st.member-of element source)
           (predicate element))))))
(? ((st.axiom-replacement source mapping image) =
     (forall (Set output)
       (st.iff
         (st.member-of output image)
         (exists (Set input)
           (set-theory.and
             (st.member-of input source)
             (mapping input output)))))))
(? ((st.successor x) =
     (st.union (st.unordered-pair x (st.singleton x)))))
(? ((st.inductive-set naturals) =
     (set-theory.and
       (st.member-of empty-set naturals)
       (forall (Set element)
         (st.implies
           (st.member-of element naturals)
           (st.member-of (st.successor element) naturals))))))
(? ((st.axiom-infinity (set-of-naturals)) =
     (st.inductive-set (set-of-naturals))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1, 1]);
  });
});
