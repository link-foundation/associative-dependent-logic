// Tests for the Pecan-style automatic-sequence domain plugin (issue #63).
// Mirrored by `rust/tests/automatic_sequences_tests.rs` so JS and Rust stay
// aligned.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Env, evaluate, keyOf } from '../src/rml-links.mjs';

describe('(domain automatic-sequences ...) decides registered examples', () => {
  it('decides the classic Thue-Morse cube-free theorem', () => {
    const env = new Env();
    const out = evaluate(
      '(domain automatic-sequences\n' +
      '  (theorem thue-morse-cube-free))\n' +
      '(? thue-morse-cube-free)',
      { env },
    );

    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
    const decision = env.automaticSequenceDecisions.get('thue-morse-cube-free');
    assert.strictEqual(decision.value, true);
    assert.strictEqual(keyOf(decision.certificate), '(buchi-emptiness thue-morse cube-free)');
  });

  it('can be queried directly as a domain decision form', () => {
    const out = evaluate(
      '(? (domain automatic-sequences (theorem thue-morse-cube-free)))',
    );

    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });
});

describe('(domain automatic-sequences ...) reports unsupported requests', () => {
  it('rejects an unknown theorem with E041', () => {
    const out = evaluate(
      '(domain automatic-sequences (theorem thue-morse-square-free))',
    );

    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E041');
    assert.match(out.diagnostics[0].message, /unknown automatic-sequences theorem/);
  });

  it('rejects an unknown domain plugin with E041', () => {
    const out = evaluate('(domain imaginary (theorem thue-morse-cube-free))');

    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E041');
    assert.match(out.diagnostics[0].message, /Unknown domain plugin/);
  });
});
