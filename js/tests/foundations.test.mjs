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

// ---------------------------------------------------------------------------
// Equality-layer provenance (issue #97, Section 1 of netkeep80's punch-list).
//
// The evaluator must surface which of the four equality layers fired for
// every equality query so foundations can argue about identity without
// recomputing the proof tree. The layers, in precedence order, are:
//
//   assigned-equality     — `((L = R) has probability p)` was declared
//   structural-equality   — L and R are syntactically the same node
//   definitional-equality — L and R normalize to the same term via
//                           beta-reduction (one side contains a lambda/apply)
//   numeric-equality      — no other layer applied; classical numeric test
//
// JS and Rust must agree on the rule string for the same source. The
// per-query `out.provenance` array is index-aligned with `out.results`, and
// is only attached when at least one equality query was observed so legacy
// programs keep the original `{results, diagnostics}` shape.
// ---------------------------------------------------------------------------
describe('equality-layer provenance', () => {
  it('omits the provenance field when no equality query is present', () => {
    const out = evaluate(`
(a: a is a)
((a = true) has probability 0.5)
(? ((a = true) and (a = true)))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.strictEqual('provenance' in out, false,
      'provenance should stay unset when no top-level equality query fires');
  });

  it('reports structural-equality for `(? (a = a))`', () => {
    const out = evaluate('(a: a is a)\n(? (a = a))');
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
    assert.deepStrictEqual(out.provenance, ['structural-equality']);
  });

  it('reports assigned-equality once a `has probability` rule exists', () => {
    const out = evaluate('((a = a) has probability 0.7)\n(? (a = a))');
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0.7]);
    assert.deepStrictEqual(out.provenance, ['assigned-equality']);
  });

  it('reports numeric-equality for `(? ((0.1 + 0.2) = 0.3))`', () => {
    const out = evaluate('(? ((0.1 + 0.2) = 0.3))');
    assert.deepStrictEqual(out.diagnostics, []);
    // 0.1 + 0.2 == 0.3 holds under the dec-12 numeric domain.
    assert.deepStrictEqual(out.results, [1]);
    assert.deepStrictEqual(out.provenance, ['numeric-equality']);
  });

  it('reports definitional-equality when a beta-reduction connects both sides', () => {
    // (apply (lambda (Natural x) x) y) ≡ y via single beta step.
    const out = evaluate('(? ((apply (lambda (Natural x) x) y) = y))');
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.provenance, ['definitional-equality']);
  });

  it('reports assigned-inequality for `(? (a != a))` when an assignment exists', () => {
    const out = evaluate('((a = a) has probability 0.7)\n(? (a != a))');
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.provenance, ['assigned-inequality']);
  });

  it('aligns provenance with results when equality and non-equality queries mix', () => {
    const out = evaluate(`
(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)
(? ((a = true) and (b = true)))
(? (a = a))
(? (1 = 2))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0.5, 1, 0]);
    assert.deepStrictEqual(out.provenance,
      [null, 'structural-equality', 'numeric-equality']);
  });

  it('records provenance inside `(with-foundation ...)` bodies', () => {
    const out = evaluate(`
(foundation classical-min (defines and min) (defines or max))
(a: a is a)
(with-foundation classical-min
  (? (a = a)))
(? (a = a))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.provenance,
      ['structural-equality', 'structural-equality']);
  });

  it('emits an equality-layer trace event for each classified query', () => {
    const out = evaluate('(a: a is a)\n(? (a = a))', { trace: true });
    assert.deepStrictEqual(out.diagnostics, []);
    const equalityEvents = out.trace.filter(e => e.kind === 'equality-layer');
    assert.strictEqual(equalityEvents.length, 1);
    assert.strictEqual(equalityEvents[0].detail, 'structural-equality');
  });
});

// ---------------------------------------------------------------------------
// Carrier enforcement (issue #97, Section 2 of netkeep80's punch-list).
//
// Foundations may now declare an explicit `(carrier ...)` of legal values
// and opt into runtime enforcement with `(strict-carrier)`. The check is
// active only inside a `(with-foundation ...)` whose descriptor carries
// both clauses, so legacy programs and foundations that omit either clause
// stay backward-compatible.
//
// Violations emit `E063` diagnostics; they never silently coerce values.
// ---------------------------------------------------------------------------
describe('foundation carrier enforcement', () => {
  it('parses `(carrier ...)` and `(strict-carrier)` onto the descriptor', () => {
    const env = new Env();
    const out = evaluate(`
(foundation two-valued
  (carrier 0 1)
  (strict-carrier)
  (defines and min)
  (defines or max))
`, { env });
    assert.deepStrictEqual(out.diagnostics, []);
    const descriptor = env.foundations.get('two-valued');
    assert.ok(descriptor, 'foundation should be registered');
    assert.deepStrictEqual(descriptor.carrier, ['0', '1']);
    assert.strictEqual(descriptor.strictCarrier, true);
  });

  it('keeps `(carrier ...)` informational when `(strict-carrier)` is absent', () => {
    const out = evaluate(`
(foundation lax-two-valued (carrier 0 1) (defines and min) (defines or max))
(a: a is a)
((a = true) has probability 0.4)
(with-foundation lax-two-valued
  (? ((a = true) and (a = true))))
`);
    // No (strict-carrier) → backward compatible, no E063.
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0.4]);
  });

  it('flags an out-of-carrier query result with E063', () => {
    const out = evaluate(`
(foundation two-valued (carrier 0 1) (strict-carrier)
  (defines and min) (defines or max))
(a: a is a)
(b: b is b)
((a = true) has probability 1)
((b = true) has probability 1)
(with-foundation two-valued
  (? ((a = true) and (b = true)))
  (? ((a = true) or (b = false))))
`);
    // min(1,1)=1 and max(1,0)=1 → both legal → no diagnostics
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1, 1]);

    const bad = evaluate(`
(foundation two-valued (carrier 0 1) (strict-carrier))
(a: a is a)
((a = true) has probability 0.5)
(with-foundation two-valued
  (? (a = true)))
`);
    // The probability assignment runs OUTSIDE the with-foundation body so
    // it is allowed; the query inside returns 0.5 → E063.
    assert.strictEqual(bad.diagnostics.length, 1);
    assert.strictEqual(bad.diagnostics[0].code, 'E063');
    assert.deepStrictEqual(bad.results, [0.5]);
  });

  it('flags an out-of-carrier probability assignment with E063', () => {
    const out = evaluate(`
(foundation two-valued (carrier 0 1) (strict-carrier))
(a: a is a)
(with-foundation two-valued
  ((a = true) has probability 0.5)
  (? (a = true)))
`);
    // The probability assignment inside the strict foundation violates the
    // carrier; the diagnostic is E063 and the assignment is rejected so the
    // query falls back to the default symbol probability (0.5 = mid range).
    assert.ok(out.diagnostics.some(d => d.code === 'E063'),
      `expected an E063 diagnostic, got ${JSON.stringify(out.diagnostics)}`);
  });

  it('restores prior carrier on `exitFoundation` (nested scopes)', () => {
    const env = new Env();
    evaluate(`
(foundation outer (carrier 0 1) (strict-carrier))
(foundation inner (carrier 0 0.5 1) (strict-carrier))
`, { env });
    env.enterFoundation('outer');
    assert.strictEqual(env._strictCarrier, true);
    assert.deepStrictEqual([...env._carrier].sort(), [0, 1]);
    env.enterFoundation('inner');
    assert.deepStrictEqual([...env._carrier].sort(), [0, 0.5, 1]);
    env.exitFoundation();
    // Back to outer carrier.
    assert.deepStrictEqual([...env._carrier].sort(), [0, 1]);
    env.exitFoundation();
    // Back to default — no carrier enforcement.
    assert.strictEqual(env._strictCarrier, false);
    assert.strictEqual(env._carrier, null);
  });

  it('exposes carrier and strictCarrier on `(foundation-report)`', () => {
    const env = new Env();
    evaluate(`
(foundation two-valued (carrier 0 1) (strict-carrier))
`, { env });
    const report = env.foundationReport();
    const tv = report.foundations.find(f => f.name === 'two-valued');
    assert.ok(tv, 'two-valued should be reported');
    assert.deepStrictEqual(tv.carrier, ['0', '1']);
    assert.strictEqual(tv.strictCarrier, true);
  });

  it('does not enforce carrier at the top level (only inside `with-foundation`)', () => {
    const out = evaluate(`
(foundation two-valued (carrier 0 1) (strict-carrier))
(a: a is a)
((a = true) has probability 0.5)
(? (a = true))
`);
    // Carrier strictness lives inside the foundation; declaring the
    // foundation alone must not break ordinary programs.
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0.5]);
  });
});
