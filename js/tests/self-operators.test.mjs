// Tests for `lib/self/operators.lino` (issue #87).
// The file encodes host operator semantics as relation links and exposes
// executable templates whose outputs are checked to 12 decimal places.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import {
  decRound,
  Env,
  evaluate,
  evaluateFile,
  keyOf,
} from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const operatorsPath = join(repoRoot, 'lib', 'self', 'operators.lino');
const virtualRootFile = join(repoRoot, 'inline-self-operators-test.lino');

const REQUIRED_RELATIONS = new Map([
  ['avg', '(avg a b ((a + b) / 2))'],
  ['min', '(min a b (self.not (self.or (self.not a) (self.not b))))'],
  ['max', '(max a b (self.or a b))'],
  ['product', '(product a b (a * b))'],
  ['probabilistic_sum', '(probabilistic_sum a b (1 - ((1 - a) * (1 - b))))'],
  ['decimal-sum', '(decimal-sum left right (left + right))'],
  ['decimal-difference', '(decimal-difference left right (left - right))'],
  ['decimal-product', '(decimal-product left right (left * right))'],
  ['decimal-quotient', '(decimal-quotient left right (left / right))'],
]);

const OUTPUT_CASES = [
  {
    name: 'avg',
    encoded: '(? (ops.avg 0.1 0.2))',
    host: '(and: avg)\n(? (0.1 and 0.2))',
  },
  {
    name: 'min',
    encoded: '(? (ops.min 0.42 0.9))',
    host: '(and: min)\n(? (0.42 and 0.9))',
  },
  {
    name: 'max',
    encoded: '(? (ops.max 0.42 0.9))',
    host: '(or: max)\n(? (0.42 or 0.9))',
  },
  {
    name: 'product',
    encoded: '(? (ops.product 0.2 0.3))',
    host: '(and: product)\n(? (0.2 and 0.3))',
  },
  {
    name: 'probabilistic_sum',
    encoded: '(? (ops.probabilistic_sum 0.2 0.3))',
    host: '(or: probabilistic_sum)\n(? (0.2 or 0.3))',
  },
  {
    name: 'decimal-sum',
    encoded: '(? (ops.decimal-sum 0.1 0.2))',
    host: '(? (0.1 + 0.2))',
  },
  {
    name: 'decimal-difference',
    encoded: '(? (ops.decimal-difference 0.3 0.1))',
    host: '(? (0.3 - 0.1))',
  },
  {
    name: 'decimal-product',
    encoded: '(? (ops.decimal-product 0.1 0.2))',
    host: '(? (0.1 * 0.2))',
  },
  {
    name: 'decimal-quotient',
    encoded: '(? (ops.decimal-quotient 1 3))',
    host: '(? (1 / 3))',
  },
  {
    name: 'decimal-quotient-zero',
    encoded: '(? (ops.decimal-quotient 1 0))',
    host: '(? (1 / 0))',
  },
];

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

function singleNumber(out, label) {
  assertClean(out);
  assert.strictEqual(out.results.length, 1, `${label}: expected one result`);
  assert.strictEqual(typeof out.results[0], 'number', `${label}: expected numeric result`);
  return out.results[0];
}

describe('lib/self/operators.lino', () => {
  it('is importable as a standard library file', () => {
    const out = evaluateFile(operatorsPath);
    assertClean(out);
  });

  it('declares the operator relations using the issue surface', () => {
    const env = new Env();
    const source = readFileSync(operatorsPath, 'utf8');
    const out = evaluate(source, { env, file: operatorsPath });
    assertClean(out);

    for (const [name, expectedClause] of REQUIRED_RELATIONS) {
      const clauses = env.relations.get(name);
      assert.ok(clauses, `missing relation ${name}`);
      assert.deepStrictEqual(clauses.map(keyOf), [expectedClause]);
    }
  });

  for (const testCase of OUTPUT_CASES) {
    it(`matches host output for ${testCase.name} to 12 decimal places`, () => {
      const encoded = singleNumber(evaluateFromRoot(`
(import "lib/self/operators.lino" as ops)
${testCase.encoded}
`), testCase.name);
      const host = singleNumber(evaluateFromRoot(testCase.host), `${testCase.name} host`);

      assert.strictEqual(decRound(encoded), decRound(host));
    });
  }
});
