// Tests for evaluation trace mode (issue #30).
// Covers the evaluate() trace option, TraceEvent shape, format helper, and
// determinism across repeated runs. The Rust suite mirrors these cases in
// rust/tests/trace_tests.rs to keep the two implementations in lock-step.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  evaluate,
  TraceEvent,
  formatTraceEvent,
} from '../src/rml-links.mjs';

const DEMO = [
  '(a: a is a)',
  '(!=: not =)',
  '(and: avg)',
  '((a = a) has probability 1)',
  '((a != a) has probability 0)',
  '(? ((a = a) and (a != a)))',
].join('\n');

describe('evaluate() with trace: true returns trace events', () => {
  it('returns no trace array when trace is not enabled', () => {
    const out = evaluate('(? 1)', { file: 'q.lino' });
    assert.strictEqual(out.trace, undefined);
  });

  it('returns a deterministic, non-empty trace when trace is enabled', () => {
    const out = evaluate(DEMO, { file: 'demo.lino', trace: true });
    assert.ok(Array.isArray(out.trace));
    assert.ok(out.trace.length > 0);
    for (const ev of out.trace) {
      assert.ok(ev instanceof TraceEvent);
      assert.ok(['resolve', 'assign', 'lookup', 'eval'].includes(ev.kind));
      assert.strictEqual(ev.span.file, 'demo.lino');
      assert.ok(ev.span.line >= 1);
      assert.ok(ev.span.col >= 1);
    }
  });

  it('produces the same trace on repeated runs (determinism)', () => {
    const a = evaluate(DEMO, { file: 'demo.lino', trace: true });
    const b = evaluate(DEMO, { file: 'demo.lino', trace: true });
    const fmt = (out) => out.trace.map(formatTraceEvent).join('\n');
    assert.strictEqual(fmt(a), fmt(b));
  });

  it('does not affect query results when enabled', () => {
    const plain = evaluate(DEMO, { file: 'demo.lino' });
    const traced = evaluate(DEMO, { file: 'demo.lino', trace: true });
    assert.deepStrictEqual(plain.results, traced.results);
    assert.deepStrictEqual(plain.diagnostics, traced.diagnostics);
  });
});

describe('TraceEvent shape and span tracking', () => {
  it('records a resolve event for (and: avg) with the operator span', () => {
    const out = evaluate('(and: avg)', { file: 'op.lino', trace: true });
    const resolves = out.trace.filter((e) => e.kind === 'resolve');
    assert.strictEqual(resolves.length, 1);
    assert.strictEqual(resolves[0].detail, '(and: avg)');
    assert.strictEqual(resolves[0].span.file, 'op.lino');
    assert.strictEqual(resolves[0].span.line, 1);
    assert.strictEqual(resolves[0].span.col, 1);
  });

  it('records an assign event for ((p) has probability v)', () => {
    const out = evaluate('((a = a) has probability 1)', {
      file: 'p.lino',
      trace: true,
    });
    const assigns = out.trace.filter((e) => e.kind === 'assign');
    assert.strictEqual(assigns.length, 1);
    assert.strictEqual(assigns[0].detail, '(a = a) ← 1');
    assert.strictEqual(assigns[0].span.line, 1);
  });

  it('records a lookup event when an assigned equality fires', () => {
    const src = '((a = a) has probability 0.7)\n(? (a = a))';
    const out = evaluate(src, { file: 'lk.lino', trace: true });
    const lookups = out.trace.filter((e) => e.kind === 'lookup');
    assert.ok(lookups.length >= 1);
    assert.match(lookups[0].detail, /\(a = a\) → 0\.7/);
    assert.strictEqual(lookups[0].span.line, 2);
  });

  it('records an eval event per top-level form', () => {
    const out = evaluate(DEMO, { file: 'demo.lino', trace: true });
    const evals = out.trace.filter((e) => e.kind === 'eval');
    // One eval event per top-level form (6 forms in DEMO).
    assert.strictEqual(evals.length, 6);
    // Final form is the query and its summary uses the `query` tag.
    const lastEval = evals[evals.length - 1];
    assert.match(lastEval.detail, /→ query 0\.5$/);
    assert.strictEqual(lastEval.span.line, 6);
  });
});

describe('formatTraceEvent formats span and kind', () => {
  it('renders [span <file>:<line>:<col>] <kind> <detail>', () => {
    const out = evaluate('(and: avg)', { file: 'fmt.lino', trace: true });
    const line = formatTraceEvent(out.trace[0]);
    assert.strictEqual(line, '[span fmt.lino:1:1] resolve (and: avg)');
  });

  it('falls back to <input> when the span has no file', () => {
    const out = evaluate('(and: avg)', { trace: true });
    const line = formatTraceEvent(out.trace[0]);
    assert.strictEqual(line, '[span <input>:1:1] resolve (and: avg)');
  });
});
