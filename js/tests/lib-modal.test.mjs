// Tests for the modal standard library (issue #71).
// Mirrors rust/tests/lib_modal_tests.rs so the LiNo library surface stays
// identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-modal-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/modal/core.lino', () => {
  it('exports modal operators through an import alias', () => {
    const out = evaluateFromRoot(`
(import "lib/modal/core.lino" as ml)
(? ((ml.necessarily p) = (necessarily p)))
(? ((ml.possibly p) = (possibly p)))
(? (ml.implies true false))
(? (ml.implies false false))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 0, 1]);
  });

  it('exports K, T, S4, and S5 axiom schemas as reusable templates', () => {
    const out = evaluateFromRoot(`
(import "lib/modal/core.lino" as ml)
(? ((ml.axiom-k p q) =
     (ml.implies
       (ml.necessarily (ml.implies p q))
       (ml.implies (ml.necessarily p) (ml.necessarily q)))))
(? ((ml.axiom-t p) =
     (ml.implies (ml.necessarily p) p)))
(? ((ml.axiom-s4 p) =
     (ml.implies (ml.necessarily p)
       (ml.necessarily (ml.necessarily p)))))
(? ((ml.axiom-s5 p) =
     (ml.implies (ml.possibly p)
       (ml.necessarily (ml.possibly p)))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1]);
  });

  it('exports Kripke-frame interpretation templates', () => {
    const out = evaluateFromRoot(`
(import "lib/modal/core.lino" as ml)
(? ((ml.holds current p) = (holds current p)))
(? ((ml.accessible current next) = (accessible current next)))
(? ((ml.necessarily-at current p) =
     (forall (World accessible-world)
       (ml.implies
         (ml.accessible current accessible-world)
         (ml.holds accessible-world p)))))
(? ((ml.possibly-at current p) =
     (exists (World accessible-world)
       (modal.and
         (ml.accessible current accessible-world)
         (ml.holds accessible-world p)))))
(? ((ml.valid p) =
     (forall (World possible-world) (ml.holds possible-world p))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1]);
  });

  it('exports Kripke frame conditions for modal systems', () => {
    const out = evaluateFromRoot(`
(import "lib/modal/core.lino" as ml)
(? ((ml.frame-k modal.accessible) = true))
(? ((ml.frame-t modal.accessible) = (ml.reflexive-frame modal.accessible)))
(? ((ml.frame-s4 modal.accessible) =
     (modal.and
       (ml.reflexive-frame modal.accessible)
       (ml.transitive-frame modal.accessible))))
(? ((ml.frame-s5 modal.accessible) =
     (modal.and
       (ml.reflexive-frame modal.accessible)
       (modal.and
         (ml.symmetric-frame modal.accessible)
         (ml.transitive-frame modal.accessible)))))
(? ((ml.euclidean-frame modal.accessible) =
     (forall (World source)
       (forall (World left)
         (forall (World right)
           (ml.implies
             (modal.and
               (ml.accessible source left)
               (ml.accessible source right))
             (ml.accessible left right)))))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1]);
  });
});
