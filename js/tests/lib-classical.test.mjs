// Tests for the classical standard library (issue #67).
// Mirrors rust/tests/lib_classical_tests.rs so the LiNo library surface stays
// identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-classical-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/classical/core.lino', () => {
  it('exports Boolean operators through an import alias', () => {
    const out = evaluateFromRoot(`
(import "lib/classical/core.lino" as cl)
(? (cl.and true false))
(? (cl.or true false))
(? (cl.not true))
(? (cl.not false))
(? (cl.or p (cl.not p)))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [0, 1, 0, 1, 1]);
  });

  it('exports classical laws as reusable templates', () => {
    const out = evaluateFromRoot(`
(import "lib/classical/core.lino" as cl)
(? (cl.excluded-middle true))
(? (cl.excluded-middle false))
(? (cl.double-negation true))
(? (cl.double-negation false))
(? (cl.de-morgan-not-and true false))
(? (cl.de-morgan-not-or true false))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [1, 1, 1, 1, 1, 1]);
  });

  it('exports natural-deduction rule schemas as tautology templates', () => {
    const out = evaluateFromRoot(`
(import "lib/classical/core.lino" as cl)
(? (cl.implies true false))
(? (cl.implies false false))
(? (cl.and-introduction true false))
(? (cl.and-elimination-left true false))
(? (cl.and-elimination-right true false))
(? (cl.or-introduction-left false true))
(? (cl.or-introduction-right true false))
(? (cl.modus-ponens true false))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [0, 1, 1, 1, 1, 1, 1, 1]);
  });
});
