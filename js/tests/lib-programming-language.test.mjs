// Tests for the programming-language theory standard library (issue #76).
// Mirrors rust/tests/lib_programming_language_tests.rs so the LiNo library
// surface stays identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-programming-language-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/programming-language/core.lino', () => {
  it('exports untyped lambda-calculus syntax and beta-step schemas', () => {
    const out = evaluateFromRoot(`
(import "lib/programming-language/core.lino" as pl)
(? ((pl.variable x) = (programming-language.object-variable x)))
(? ((pl.abstraction x (pl.variable x)) =
     (programming-language.object-lambda x
       (programming-language.object-variable x))))
(? ((pl.application (pl.abstraction x (pl.variable x)) y) =
     (programming-language.object-apply
       (programming-language.object-lambda x
         (programming-language.object-variable x))
       y)))
(? ((pl.beta-reduction x (pl.variable x) y) =
     (programming-language.small-step
       (programming-language.object-apply
         (programming-language.object-lambda x
           (programming-language.object-variable x))
         y)
       (programming-language.object-substitution
         (programming-language.object-variable x) x y))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1]);
  });

  it('exports STLC typing rules as reusable schemas', () => {
    const out = evaluateFromRoot(`
(import "lib/programming-language/core.lino" as pl)
(? ((pl.function-type A B) =
     (programming-language.simple-function-type A B)))
(? ((pl.typing-abstraction gamma x A body B) =
     (programming-language.implies
       (pl.has-type (pl.extend-context gamma x A) body B)
       (pl.has-type gamma
         (pl.abstraction x body)
         (pl.function-type A B)))))
(? ((pl.typing-application gamma fn arg A B) =
     (programming-language.implies
       (programming-language.and
         (pl.has-type gamma fn (pl.function-type A B))
         (pl.has-type gamma arg A))
       (pl.has-type gamma (pl.application fn arg) B))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1]);
  });

  it('exports theorem progress and preservation through the issue surface', () => {
    const out = evaluateFromRoot(`
(import "lib/programming-language/core.lino" as pl)
(? (pl.theorem progress (pl.progress term T)))
(? (pl.theorem preservation (pl.preservation term next T)))
(? ((pl.progress term T) =
     (pl.progress-in programming-language.empty-context term T)))
(? ((pl.preservation term next T) =
     (pl.preservation-in programming-language.empty-context term next T)))
(? ((pl.type-safety term next T) =
     (programming-language.and
       (pl.progress term T)
       (pl.preservation term next T))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1]);
  });
});
