// Foundation / root-construct registry tests (issue #97).
//
// The foundation surface is the user-facing mechanism for replacing
// kernel-level interpretations (`and`, `or`, ...) without touching the
// evaluator. These tests cover three layers:
//   1. The data registry (`(root-construct ...)`, `(foundation ...)`)
//      round-trips through `evaluate()` without losing fields.
//   2. `(with-foundation <name> ...)` swaps operator semantics inside
//      its body and restores them on exit, leaving outer scopes intact.
//   3. `(foundation-report)` returns a structured snapshot whose printed
//      form matches the canonical layout (kept byte-identical with Rust).
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97
import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Env, evaluate, formatFoundationReport } from '../src/rml-links.mjs';

function run(src, env) {
  const out = evaluate(src, env ? { env } : undefined);
  assert.deepStrictEqual(out.diagnostics, []);
  return out.results;
}

describe('foundation / root-construct registry', () => {
  it('preregisters `default-rml` so legacy programs need no migration', () => {
    const env = new Env();
    assert.strictEqual(env.activeFoundation, 'default-rml');
    assert.ok(env.foundations.has('default-rml'));
    const def = env.foundations.get('default-rml');
    assert.ok(def, 'default-rml descriptor is missing');
    assert.strictEqual(typeof def.description, 'string');
    assert.strictEqual(def.description.length > 0, true);
  });

  it('does not change baseline semantics when no foundation is declared', () => {
    const results = run(`
(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)
(? ((a = true) and (b = true)))
(? ((a = true) or (b = true)))
`);
    // Default `and` is avg, default `or` is max.
    assert.deepStrictEqual(results, [0.5, 0.6]);
  });

  it('switches `and`/`or` semantics inside `(with-foundation ...)`', () => {
    const results = run(`
(foundation classical-min (defines and min) (defines or max))
(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)
(? ((a = true) and (b = true)))
(with-foundation classical-min
  (? ((a = true) and (b = true)))
  (? ((a = true) or (b = true))))
(? ((a = true) and (b = true)))
`);
    // Outer default avg → 0.5; inside foundation min → 0.4 and max → 0.6;
    // outer scope restored → 0.5.
    assert.deepStrictEqual(results, [0.5, 0.4, 0.6, 0.5]);
  });

  it('nests `(with-foundation ...)` scopes correctly', () => {
    const results = run(`
(foundation use-min (defines and min))
(foundation use-prod (defines and product))
(a: a is a)
(b: b is b)
((a = true) has probability 0.5)
((b = true) has probability 0.4)
(with-foundation use-min
  (? ((a = true) and (b = true)))
  (with-foundation use-prod
    (? ((a = true) and (b = true))))
  (? ((a = true) and (b = true))))
`);
    // min(0.5,0.4)=0.4 ; product(0.5,0.4)=0.2 ; back to min=0.4
    assert.deepStrictEqual(results, [0.4, 0.2, 0.4]);
  });

  it('reports an unknown foundation as E062 without aborting evaluation', () => {
    const out = evaluate(`
(a: a is a)
((a = true) has probability 0.5)
(with-foundation does-not-exist
  (? ((a = true) and (a = true))))
(? ((a = true) and (a = true)))
`);
    // The bad `with-foundation` emits E062 but the trailing query still runs.
    assert.deepStrictEqual(out.results, [0.5]);
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E062');
  });

  it('records root constructs and foundations via the data registry', () => {
    const env = new Env();
    run(`
(root-construct my-and
  (kind truth-operator)
  (status links-defined)
  (depends-on truth-range))
(foundation my-foundation
  (description my-toy-foundation)
  (defines and min)
  (numeric-domain unit-interval))
`, env);
    const my = env.rootConstructs.get('my-and');
    assert.ok(my, 'my-and not registered');
    assert.strictEqual(my.kind, 'truth-operator');
    assert.strictEqual(my.status, 'links-defined');
    const foundation = env.foundations.get('my-foundation');
    assert.ok(foundation, 'my-foundation not registered');
    assert.strictEqual(foundation.description, 'my-toy-foundation');
    assert.strictEqual(foundation.numericDomain, 'unit-interval');
    assert.deepStrictEqual([...foundation.defines.entries()], [['and', 'min']]);
  });

  it('builds a structured `foundation-report` snapshot', () => {
    const env = new Env();
    run(`
(foundation tiny
  (description toy-foundation)
  (defines and min))
`, env);
    env.enterFoundation('tiny');
    try {
      const report = env.foundationReport();
      assert.strictEqual(report.activeFoundation, 'tiny');
      assert.strictEqual(report.description, 'toy-foundation');
      assert.ok(Array.isArray(report.rootConstructs));
      assert.ok(report.rootConstructs.length > 0,
        'root constructs should be seeded by default');
      const text = formatFoundationReport(report);
      assert.match(text, /active foundation: tiny/);
      assert.match(text, /description: toy-foundation/);
    } finally {
      env.exitFoundation();
    }
    // Exiting must restore the active foundation tag.
    assert.strictEqual(env.activeFoundation, 'default-rml');
  });

  it('`enterFoundation` snapshots ops so `exitFoundation` restores them', () => {
    const env = new Env();
    run(`
(foundation only-min (defines and min))
`, env);
    const beforeAnd = env.ops.get('and');
    env.enterFoundation('only-min');
    const insideAnd = env.ops.get('and');
    assert.notStrictEqual(beforeAnd, insideAnd,
      '`and` should be re-bound inside the foundation');
    // min(0.6, 0.4) = 0.4
    assert.strictEqual(insideAnd(0.6, 0.4), 0.4);
    env.exitFoundation();
    // After exit, the original op function reference is restored.
    assert.strictEqual(env.ops.get('and'), beforeAnd);
    // Original avg semantics: avg(0.6, 0.4) = 0.5
    assert.strictEqual(env.ops.get('and')(0.6, 0.4), 0.5);
  });
});
