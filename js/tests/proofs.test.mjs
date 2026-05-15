// Tests for the proof-producing evaluator (issue #35).
// Covers the global `withProofs` option, the inline `(? expr with proof)`
// keyword pair, derivation shape per built-in operator, the
// `parse(print(proof)) == proof` round-trip, and backwards compatibility
// of the `{results, diagnostics}` shape when proofs are not requested.
// Mirrored by `rust/tests/proofs_tests.rs` so any drift between the two
// implementations fails both suites.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  evaluate,
  buildProof,
  keyOf,
  parseOne,
  tokenizeOne,
  isStructurallySame,
  Env,
} from '../src/rml-links.mjs';

// Round-trip a derivation through print -> tokenize -> parse and assert
// the resulting AST is structurally identical to the original. This is the
// acceptance criterion from the issue: "round-trip parse(print(D)) == D".
function assertRoundTrip(proof) {
  const printed = keyOf(proof);
  const reparsed = parseOne(tokenizeOne(printed));
  assert.ok(
    isStructurallySame(proof, reparsed),
    `proof did not round-trip: ${printed}`,
  );
}

describe('evaluate() returns proofs only when requested', () => {
  it('omits the proofs key when neither flag nor inline keyword is used', () => {
    const out = evaluate('(? 1)');
    assert.deepStrictEqual(Object.keys(out).sort(), ['diagnostics', 'results']);
    assert.strictEqual(out.proofs, undefined);
  });

  it('returns a proofs array when withProofs is true', () => {
    const out = evaluate('(? 1)', { withProofs: true });
    assert.ok(Array.isArray(out.proofs));
    assert.strictEqual(out.proofs.length, 1);
    assert.ok(out.proofs[0]);
  });

  it('returns a proofs array when an inline (? ... with proof) is present', () => {
    const out = evaluate('(? 1)\n(? 2 with proof)');
    assert.ok(Array.isArray(out.proofs));
    assert.strictEqual(out.proofs.length, 2);
    // First query did not opt in -> null; second one did.
    assert.strictEqual(out.proofs[0], null);
    assert.ok(out.proofs[1]);
  });
});

describe('issue example reproduction', () => {
  it('produces the canonical (by structural-equality (a a)) witness', () => {
    const out = evaluate('(a: a is a)\n(? (a = a) with proof)');
    assert.deepStrictEqual(out.results, [1]);
    assert.strictEqual(out.proofs.length, 1);
    assert.strictEqual(keyOf(out.proofs[0]), '(by structural-equality (a a))');
    assertRoundTrip(out.proofs[0]);
  });

  it('produces the same witness under the global flag', () => {
    const out = evaluate('(a: a is a)\n(? (a = a))', { withProofs: true });
    assert.deepStrictEqual(out.results, [1]);
    assert.strictEqual(keyOf(out.proofs[0]), '(by structural-equality (a a))');
  });
});

describe('per-rule witness shapes', () => {
  it('records assigned-equality when an equality has been assigned a probability', () => {
    const src = '((a = a) has probability 0.7)\n(? (a = a))';
    const out = evaluate(src, { withProofs: true });
    assert.deepStrictEqual(out.results, [0.7]);
    assert.strictEqual(keyOf(out.proofs[0]), '(by assigned-equality (a a))');
    assertRoundTrip(out.proofs[0]);
  });

  it('records assigned-inequality on (? (L != R)) when L=R has an assignment', () => {
    const src = '((a = a) has probability 0.7)\n(? (a != a))';
    const out = evaluate(src, { withProofs: true });
    assert.strictEqual(keyOf(out.proofs[0]), '(by assigned-inequality (a a))');
    assertRoundTrip(out.proofs[0]);
  });

  it('records numeric-equality when no assignment exists and operands differ', () => {
    // 1 = 2 is false (rule fires regardless of clamped truth value).
    const out = evaluate('(? (1 = 2))', { withProofs: true });
    assert.strictEqual(keyOf(out.proofs[0]), '(by numeric-equality (1 2))');
    assertRoundTrip(out.proofs[0]);
  });

  it('records sum / difference / product / quotient for arithmetic', () => {
    const src = '(? (1 + 2))\n(? (5 - 2))\n(? (3 * 4))\n(? (8 / 2))';
    const out = evaluate(src, { withProofs: true });
    assert.strictEqual(out.results.length, 4);
    const rules = out.proofs.map(p => p[1]);
    assert.deepStrictEqual(rules, ['sum', 'difference', 'product', 'quotient']);
    out.proofs.forEach(assertRoundTrip);
  });

  it('records and / or for binary infix logic', () => {
    const src = '(? (1 and 0))\n(? (1 or 0))';
    const out = evaluate(src, { withProofs: true });
    assert.strictEqual(out.proofs[0][1], 'and');
    assert.strictEqual(out.proofs[1][1], 'or');
    out.proofs.forEach(assertRoundTrip);
  });

  it('records both / neither for composite chains', () => {
    const src = '(? (both 1 and 0 and 1))\n(? (neither 0 nor 0))';
    const out = evaluate(src, { withProofs: true });
    assert.strictEqual(out.proofs[0][1], 'both');
    assert.strictEqual(out.proofs[0].length, 5); // (by both s1 s2 s3)
    assert.strictEqual(out.proofs[1][1], 'neither');
    out.proofs.forEach(assertRoundTrip);
  });

  it('records prefix operator names for (not X), (and X Y), (or X Y)', () => {
    const src = '(? (not 1))\n(? (and 1 1))\n(? (or 0 1))';
    const out = evaluate(src, { withProofs: true });
    assert.deepStrictEqual(
      out.proofs.map(p => p[1]),
      ['not', 'and', 'or'],
    );
    out.proofs.forEach(assertRoundTrip);
  });

  it('records assigned-probability for top-level ((expr) has probability p)', () => {
    const env = new Env();
    const proof = buildProof(
      [['a', '=', 'a'], 'has', 'probability', '0.5'],
      env,
    );
    assert.strictEqual(keyOf(proof), '(by assigned-probability (a = a) 0.5)');
    assertRoundTrip(proof);
  });

  it('records type-universe / prop / pi-formation / lambda-formation', () => {
    const env = new Env();
    const u = buildProof(['Type', '0'], env);
    assert.strictEqual(keyOf(u), '(by type-universe 0)');
    assertRoundTrip(u);

    const p = buildProof(['Prop'], env);
    assert.strictEqual(keyOf(p), '(by prop)');
    assertRoundTrip(p);

    const pi = buildProof(['Pi', ['x:', 'A'], 'B'], env);
    assert.strictEqual(keyOf(pi), '(by pi-formation (x: A) B)');
    assertRoundTrip(pi);

    const lam = buildProof(['lambda', ['x:', 'A'], 'x'], env);
    assert.strictEqual(keyOf(lam), '(by lambda-formation (x: A) x)');
    assertRoundTrip(lam);
  });
});

describe('proofs are index-aligned with results', () => {
  it('emits null for bare queries that did not opt in (inline mode only)', () => {
    const src = '(? 1)\n(? 0 with proof)\n(? 1)';
    const out = evaluate(src);
    assert.strictEqual(out.results.length, 3);
    assert.strictEqual(out.proofs.length, 3);
    assert.strictEqual(out.proofs[0], null);
    assert.ok(out.proofs[1]);
    assert.strictEqual(out.proofs[2], null);
  });

  it('produces a proof for every query when withProofs is true', () => {
    const src = '(a: a is a)\n(? 1)\n(? (1 + 0))\n(? (a = a))';
    const out = evaluate(src, { withProofs: true });
    assert.strictEqual(out.results.length, 3);
    assert.strictEqual(out.proofs.length, 3);
    out.proofs.forEach(p => {
      assert.ok(p);
      assertRoundTrip(p);
    });
  });
});

describe('proofs do not affect query results or diagnostics', () => {
  it('produces identical results and diagnostics with and without proofs', () => {
    const src = [
      '(a: a is a)',
      '(b: b is b)',
      '((a = a) has probability 1)',
      '((b = b) has probability 0)',
      '(? ((a = a) and (b = b)))',
    ].join('\n');
    const plain = evaluate(src);
    const proven = evaluate(src, { withProofs: true });
    assert.deepStrictEqual(plain.results, proven.results);
    assert.deepStrictEqual(plain.diagnostics, proven.diagnostics);
  });
});

describe('round-trip property holds for every produced proof', () => {
  it('survives parse(print(proof)) for a representative bundle of operators', () => {
    const src = [
      '(a: a is a)',
      '((a = a) has probability 0.7)',
      '(? (a = a))',
      '(? (1 + 2))',
      '(? (5 - 2))',
      '(? (3 * 4))',
      '(? (8 / 2))',
      '(? (not 1))',
      '(? (1 and 0))',
      '(? (0 or 1))',
      '(? (both 1 and 1 and 0))',
      '(? (neither 0 nor 0))',
      '(? (1 = 2))',
      '(? (1 != 2))',
    ].join('\n');
    const out = evaluate(src, { withProofs: true });
    assert.strictEqual(out.proofs.length, out.results.length);
    out.proofs.forEach(assertRoundTrip);
  });
});
