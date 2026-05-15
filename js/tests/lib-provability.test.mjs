// Tests for the provability standard library (issue #72).
// Mirrors rust/tests/lib_provability_tests.rs so the LiNo library surface stays
// identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-provability-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/provability/core.lino', () => {
  it('exports provability operators through an import alias', () => {
    const out = evaluateFromRoot(`
(import "lib/provability/core.lino" as pr)
(? ((pr.provability-of p) = (provability-of p)))
(? ((pr.consistency-of p) =
     (provability.not
       (provability.provability-of (provability.not p)))))
(? (pr.implies true false))
(? (pr.implies false false))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 0, 1]);
  });

  it('exports GL axiom schemas as reusable templates', () => {
    const out = evaluateFromRoot(`
(import "lib/provability/core.lino" as pr)
(? ((pr.axiom-k p q) =
     (pr.implies
       (pr.provability-of (pr.implies p q))
       (pr.implies
         (pr.provability-of p)
         (pr.provability-of q)))))
(? ((pr.axiom-lob p) =
     (pr.implies
       (pr.provability-of
         (pr.implies (pr.provability-of p) p))
       (pr.provability-of p))))
(? ((pr.axiom-gl p) = (pr.axiom-lob p)))
(? ((pr.necessitation-rule p) =
     (pr.implies p (pr.provability-of p))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1]);
  });

  it('exports interpretability fragment schemas', () => {
    const out = evaluateFromRoot(`
(import "lib/provability/core.lino" as pr)
(? ((pr.interprets source target) = (interprets source target)))
(? ((pr.axiom-j1 source target) =
     (pr.implies
       (pr.provability-of (pr.implies source target))
       (pr.interprets source target))))
(? ((pr.axiom-j2 source middle target) =
     (pr.implies
       (provability.and
         (pr.interprets source middle)
         (pr.interprets middle target))
       (pr.interprets source target))))
(? ((pr.axiom-j3 left right target) =
     (pr.implies
       (provability.and
         (pr.interprets left target)
         (pr.interprets right target))
       (pr.interprets (provability.or left right) target))))
(? ((pr.axiom-j4 source target) =
     (pr.implies
       (pr.interprets source target)
       (pr.implies
         (pr.consistency-of source)
         (pr.consistency-of target)))))
(? ((pr.axiom-j5 source) =
     (pr.interprets (pr.consistency-of source) source)))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1, 1]);
  });
});
