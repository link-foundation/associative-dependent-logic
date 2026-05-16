// MTC/anum experimental foundation profile tests (issue #97, Phase 9).
//
// The `mtc-anum` foundation is pre-seeded but opt-in. Activating it does
// not rewire host arithmetic — it is metadata plus a four-abit
// serialization alphabet (`[`, `]`, `0`, `1`). The trust audit surfaces
// the profile with its `[experimental]` tag, root symbol, and abit list.
// `encodeAnum` / `decodeAnum` round-trip arbitrary Node values through
// strings written only in that alphabet.
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97
import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  Env,
  evaluate,
  formatFoundationReport,
  encodeAnum,
  decodeAnum,
  parseLino,
} from '../src/rml-links.mjs';

describe('mtc-anum experimental foundation', () => {
  it('is pre-seeded but never activated implicitly', () => {
    const env = new Env();
    assert.ok(env.foundations.has('mtc-anum'), 'mtc-anum should be pre-seeded');
    assert.strictEqual(env.activeFoundation, 'default-rml',
      'default-rml stays the active foundation');
  });

  it('does not perturb baseline semantics when not selected', () => {
    // The pre-seeded mtc-anum profile is opt-in. A clean session should
    // continue to report the default-rml foundation and produce the same
    // query results it always did.
    const plain = evaluate('(? (1 + 2))');
    assert.deepStrictEqual(plain.diagnostics, []);
    assert.strictEqual(plain.results.length, 1);
    const env = new Env();
    assert.strictEqual(env.activeFoundation, 'default-rml');
  });

  it('surfaces the experimental flag, root, and abits on foundation report', () => {
    const env = new Env();
    const report = env.foundationReport();
    const mtc = report.foundations.find(f => f.name === 'mtc-anum');
    assert.ok(mtc, 'mtc-anum should appear in foundations');
    assert.strictEqual(mtc.experimental, true);
    assert.strictEqual(mtc.root, '∞');
    assert.ok(Array.isArray(mtc.abits));
    const symbols = mtc.abits.map(a => a.symbol).sort();
    assert.deepStrictEqual(symbols, ['0', '1', '[', ']']);
  });

  it('renders the experimental tag and abits in the printed report', () => {
    const env = new Env();
    const printed = formatFoundationReport(env.foundationReport());
    assert.match(printed, /mtc-anum \[experimental\]/);
    assert.match(printed, /root: ∞/);
    assert.match(printed, /abits: .*\[=start-of-meaning/);
  });

  it('accepts `(experimental)`, `(root ...)`, and `(abit ...)` parser clauses', () => {
    const env = new Env();
    const out = evaluate(`
(foundation toy-mtc
  (description "a toy mtc-style profile")
  (experimental)
  (root ★)
  (abit ▲ up)
  (abit ▼ down))
`, { env });
    assert.deepStrictEqual(out.diagnostics, []);
    const f = env.foundations.get('toy-mtc');
    assert.ok(f, 'toy-mtc should be registered');
    assert.strictEqual(f.experimental, true);
    assert.strictEqual(f.root, '★');
    assert.deepStrictEqual(
      f.abits.map(a => a.symbol),
      ['▲', '▼'],
    );
  });

  it('encodes and decodes leaf strings round-trip', () => {
    const cases = ['x', 'hello', '+', '∞', ''];
    for (const c of cases) {
      const enc = encodeAnum(c);
      // Only the four abits should ever appear in the encoded form.
      assert.match(enc, /^[\[\]01]+$/, `encoding of ${JSON.stringify(c)} not in alphabet`);
      assert.deepStrictEqual(decodeAnum(enc), c);
    }
  });

  it('encodes and decodes lists round-trip', () => {
    const cases = [
      [],
      ['a', 'b'],
      ['+', '1', '2'],
      ['lambda', ['x'], ['+', 'x', '1']],
    ];
    for (const c of cases) {
      const enc = encodeAnum(c);
      assert.match(enc, /^[\[\]01]+$/);
      assert.deepStrictEqual(decodeAnum(enc), c);
    }
  });

  it('round-trips a real parsed link form', () => {
    const parsed = parseLino('(? (1 + 2))')[0];
    const enc = encodeAnum(parsed);
    assert.match(enc, /^[\[\]01]+$/);
    assert.deepStrictEqual(decodeAnum(enc), parsed);
  });

  it('decodeAnum rejects characters outside the four-abit alphabet', () => {
    assert.throws(() => decodeAnum('[0AB]'), err => err.code === 'E066');
    assert.throws(() => decodeAnum('[2]'), err => err.code === 'E066');
    assert.throws(() => decodeAnum('xyz'), err => err.code === 'E066');
  });

  it('decodeAnum rejects unbalanced frames', () => {
    assert.throws(() => decodeAnum('[0'), err => err.code === 'E066');
    assert.throws(() => decodeAnum('[1[0]'), err => err.code === 'E066');
    assert.throws(() => decodeAnum('[0]extra'), err => err.code === 'E066');
  });

  it('decodeAnum rejects misaligned leaf payloads', () => {
    // 7 bits — not byte-aligned.
    assert.throws(() => decodeAnum('[0' + '0101010' + ']'), err => err.code === 'E066');
  });

  it('encodeAnum rejects non-Node values', () => {
    assert.throws(() => encodeAnum(undefined), err => err.code === 'E066');
    assert.throws(() => encodeAnum(null), err => err.code === 'E066');
    assert.throws(() => encodeAnum({ a: 1 }), err => err.code === 'E066');
  });
});
