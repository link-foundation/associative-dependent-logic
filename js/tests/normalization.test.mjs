// Tests for the typed-fragment normalization API (issue #50, D4).
//
// Covers:
//   - the public `whnf` and `nf` functions,
//   - the surface-form drivers `(whnf ...)`, `(nf ...)`, and `(normal-form ...)`,
//   - termination and result equality on Church numerals (the acceptance
//     criterion from the issue),
//   - proof witnesses for normalization steps.
//
// The Rust suite mirrors these cases in rust/tests/normalization_tests.rs so
// any drift between the two implementations fails both suites.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  evaluate,
  Env,
  evalNode,
  whnf,
  nf,
  isStructurallySame,
  keyOf,
} from '../src/rml-links.mjs';

function evaluateClean(src) {
  const out = evaluate(src);
  assert.deepStrictEqual(out.diagnostics, [], `expected no diagnostics, got: ${JSON.stringify(out.diagnostics)}`);
  return out.results;
}

describe('whnf reduces only the spine', () => {
  it('beta-reduces an outer redex but leaves arguments unevaluated', () => {
    const env = new Env();
    const reduced = whnf(['apply', ['lambda', ['Natural', 'x'], 'x'],
                          ['apply', ['lambda', ['Natural', 'y'], 'y'], 'zero']], env);
    // The argument is not under whnf — it remains a redex.
    assert.deepStrictEqual(reduced, ['apply', ['lambda', ['Natural', 'y'], 'y'], 'zero']);
  });

  it('does not descend under a lambda binder', () => {
    const env = new Env();
    const term = ['lambda', ['Natural', 'x'],
                  ['apply', ['lambda', ['Natural', 'y'], 'y'], 'x']];
    // Already a lambda value; whnf returns it unchanged.
    assert.deepStrictEqual(whnf(term, env), term);
  });

  it('reduces a named lambda head and stops there', () => {
    const env = new Env();
    evalNode(['identity:', 'lambda', ['Natural', 'x'], 'x'], env);
    const reduced = whnf(['apply', 'identity', ['apply', 'identity', 'zero']], env);
    // The outer identity reduces, but the inner argument is not normalized.
    assert.deepStrictEqual(reduced, ['apply', 'identity', 'zero']);
  });
});

describe('nf reduces every redex', () => {
  it('fully normalizes a nested application', () => {
    const env = new Env();
    const term = ['apply', ['lambda', ['Natural', 'x'], 'x'],
                  ['apply', ['lambda', ['Natural', 'y'], 'y'], 'zero']];
    assert.strictEqual(nf(term, env), 'zero');
  });

  it('reduces redexes under a lambda binder', () => {
    const env = new Env();
    const term = ['lambda', ['Natural', 'x'],
                  ['apply', ['lambda', ['Natural', 'y'], 'y'], 'x']];
    assert.deepStrictEqual(nf(term, env), ['lambda', ['Natural', 'x'], 'x']);
  });

  it('accepts string surface input and returns a structural AST', () => {
    const env = new Env();
    evalNode(['Natural:', ['Type', '0'], 'Natural'], env);
    evalNode(['identity:', 'lambda', ['Natural', 'x'], 'x'], env);
    const reduced = nf('(apply identity zero)', env);
    assert.strictEqual(reduced, 'zero');
  });
});

describe('(whnf ...) surface form', () => {
  it('returns the weak-head reduct as a term result', () => {
    const out = evaluate(`
(Natural: (Type 0) Natural)
(zero: Natural zero)
(identity: lambda (Natural x) x)
(? (whnf (apply identity (apply identity zero))))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    // Outer identity reduces, inner application is left alone.
    assert.deepStrictEqual(out.results, ['(apply identity zero)']);
  });

  it('rejects malformed driver forms with E038', () => {
    const out = evaluate('(whnf)');
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E038');
  });
});

describe('(nf ...) and (normal-form ...) surface forms', () => {
  it('reduce to the same beta-normal form', () => {
    const out = evaluate(`
(Natural: (Type 0) Natural)
(zero: Natural zero)
(identity: lambda (Natural x) x)
(? (nf (apply identity (apply identity zero))))
(? (normal-form (apply identity (apply identity zero))))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, ['zero', 'zero']);
  });

  it('rejects malformed driver forms with E038', () => {
    const a = evaluate('(nf)');
    assert.strictEqual(a.diagnostics.length, 1);
    assert.strictEqual(a.diagnostics[0].code, 'E038');
    const b = evaluate('(normal-form)');
    assert.strictEqual(b.diagnostics.length, 1);
    assert.strictEqual(b.diagnostics[0].code, 'E038');
  });
});

// Church numerals encoded with the kernel's lambda. Acceptance test from
// the issue: `(normal-form (apply (compose succ succ) zero))` should produce
// `(succ (succ zero))`.
describe('Church numerals normalize as expected', () => {
  // The acceptance contract from issue #50 treats `succ` and `zero` as
  // constructors and `compose` as the standard `\f g x. f (g x)`. Two
  // applications of `compose succ succ` to `zero` should peel two outer
  // `succ`s onto the constructor stack — yielding `(succ (succ zero))`.
  const churchPreamble = `
(Term: (Type 0) Term)
(zero: Term zero)
(succ: (Pi (Term n) Term))
(compose: lambda (Term f) (lambda (Term g) (lambda (Term x) (apply f (apply g x)))))
`;

  it('normal-form of (apply (compose succ succ) zero) is (succ (succ zero))', () => {
    const out = evaluate(churchPreamble +
      '(? (normal-form (apply (apply (apply compose succ) succ) zero)))');
    assert.deepStrictEqual(out.diagnostics, []);
    // Pretty-printer drops the explicit `apply` keyword for neutral
    // applications (head is a free constructor symbol), matching the
    // surface shape `(succ (succ zero))` from the issue's acceptance
    // criterion.
    assert.deepStrictEqual(out.results, ['(succ (succ zero))']);
  });

  it('whnf reduces only the head — leaves the inner succ call alone', () => {
    const env = new Env();
    evalNode(['Term:', ['Type', '0'], 'Term'], env);
    evalNode(['zero:', 'Term', 'zero'], env);
    evalNode(['succ:', ['Pi', ['Term', 'n'], 'Term']], env);
    evalNode(['compose:', 'lambda', ['Term', 'f'],
              ['lambda', ['Term', 'g'],
               ['lambda', ['Term', 'x'], ['apply', 'f', ['apply', 'g', 'x']]]]], env);
    const term = ['apply', ['apply', ['apply', 'compose', 'succ'], 'succ'], 'zero'];
    const head = whnf(term, env);
    // Whnf unfolds `compose` and applies it to its arguments, exposing the
    // outer `succ` application of the result. The inner `(apply succ zero)`
    // remains a redex because whnf does not descend into argument positions.
    assert.deepStrictEqual(head, ['apply', 'succ', ['apply', 'succ', 'zero']]);
  });

  it('fully normalizes nested compositions', () => {
    const out = evaluate(churchPreamble +
      '(? (nf (apply (apply (apply compose succ) succ) (apply succ zero))))');
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, ['(succ (succ (succ zero)))']);
  });
});

describe('proof witness names for whnf and nf', () => {
  it('attaches a whnf-reduction witness when proofs are requested', () => {
    const out = evaluate('(? (whnf (apply (lambda (Natural x) x) zero)) with proof)');
    assert.deepStrictEqual(out.diagnostics, []);
    assert.ok(Array.isArray(out.proofs) && out.proofs[0]);
    // Proof witnesses are tagged ['by', <rule>, ...sub-witnesses].
    assert.strictEqual(out.proofs[0][0], 'by');
    assert.strictEqual(out.proofs[0][1], 'whnf-reduction');
  });

  it('attaches an nf-reduction witness for both nf and normal-form', () => {
    const out = evaluate(
      '(? (nf (apply (lambda (Natural x) x) zero)) with proof)\n' +
      '(? (normal-form (apply (lambda (Natural x) x) zero)) with proof)',
    );
    assert.deepStrictEqual(out.diagnostics, []);
    assert.ok(Array.isArray(out.proofs) && out.proofs[0] && out.proofs[1]);
    assert.strictEqual(out.proofs[0][0], 'by');
    assert.strictEqual(out.proofs[0][1], 'nf-reduction');
    assert.strictEqual(out.proofs[1][0], 'by');
    assert.strictEqual(out.proofs[1][1], 'nf-reduction');
  });

  it('attaches a whnf proof witness under the global withProofs flag', () => {
    // Mirror of rust/tests/normalization_tests.rs::
    // whnf_proof_witness_under_global_with_proofs_flag.
    const src = '(? (whnf (apply (lambda (Natural x) x) zero)))';
    const out = evaluate(src, { withProofs: true });
    assert.deepStrictEqual(out.diagnostics, []);
    assert.ok(Array.isArray(out.proofs) && out.proofs[0]);
    assert.strictEqual(out.proofs[0][1], 'whnf-reduction');
  });
});

describe('isConvertible already uses full normalization', () => {
  it('agrees with `nf` on beta-equal terms', () => {
    const env = new Env();
    evalNode(['Term:', ['Type', '0'], 'Term'], env);
    evalNode(['zero:', 'Term', 'zero'], env);
    // `succ` as a Pi-typed constructor — defining it as `lambda n. succ n`
    // would loop because `succ` would unfold itself.
    evalNode(['succ:', ['Pi', ['Term', 'n'], 'Term']], env);
    evalNode(['identity:', 'lambda', ['Term', 'x'], 'x'], env);
    // The `apply identity` redex inside should reduce away under nf, leaving
    // two stacked `succ` constructors (printed without the explicit `apply`).
    const lhs = ['apply', 'succ', ['apply', 'identity', ['apply', 'succ', 'zero']]];
    const expected = ['succ', ['succ', 'zero']];
    const rhs = nf(lhs, env);
    assert.ok(isStructurallySame(rhs, expected) || keyOf(rhs) === keyOf(expected));
  });
});
