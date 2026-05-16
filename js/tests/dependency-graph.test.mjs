// Dependency-graph traversal tests (issue #97, Phase 7).
//
// The root-construct registry records each construct's direct dependencies
// via `dependsOn`. The dependency-graph helpers expose the transitive
// closure deterministically — for the global graph (sorted construct list,
// each with its sorted transitive deps) and for a single construct
// (`dependencyClosure(name)`).
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97
import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  Env,
  evaluate,
  formatFoundationReport,
  buildDependencyGraph,
} from '../src/rml-links.mjs';

describe('dependency graph traversal', () => {
  it('exposes a `dependencyGraph` field on the foundation report', () => {
    const env = new Env();
    const report = env.foundationReport();
    assert.ok(report.dependencyGraph, 'expected dependencyGraph field');
    assert.strictEqual(typeof report.dependencyGraph, 'object');
  });

  it('includes every seeded root construct as a key', () => {
    const env = new Env();
    const graph = env.foundationReport().dependencyGraph;
    // Spot-check a few seeded constructs from the registry.
    assert.ok('+' in graph, '+ missing');
    assert.ok('-' in graph, '- missing');
    assert.ok('=' in graph, '= missing');
    assert.ok('!=' in graph, '!= missing');
    assert.ok('and' in graph, 'and missing');
    assert.ok('lambda' in graph, 'lambda missing');
  });

  it('returns sorted, deduplicated transitive deps for each construct', () => {
    const env = new Env();
    const graph = env.foundationReport().dependencyGraph;
    // `+` depends directly on `decimal-12-arithmetic`. Closure should be
    // that single construct, sorted.
    assert.deepStrictEqual(graph['+'], ['decimal-12-arithmetic']);
    // `!=` is host-derived, depends on `=` (which itself pulls in
    // `decimal-12-arithmetic` + `structural-equality`) and `not` (which
    // depends on `truth-range`). The transitive closure should include
    // them all, sorted.
    const ineq = graph['!='];
    assert.ok(ineq.includes('='));
    assert.ok(ineq.includes('not'));
    assert.ok(ineq.includes('decimal-12-arithmetic'));
    assert.ok(ineq.includes('structural-equality'));
    assert.ok(ineq.includes('truth-range'));
    // Sort property:
    assert.deepStrictEqual(ineq.slice().sort(), ineq);
  });

  it('returns [] for leaf constructs with no dependencies', () => {
    const env = new Env();
    const graph = env.foundationReport().dependencyGraph;
    assert.deepStrictEqual(graph['decimal-12-arithmetic'], []);
    assert.deepStrictEqual(graph['structural-equality'], []);
  });

  it('`dependencyClosure(name)` returns the same closure as the graph', () => {
    const env = new Env();
    const graph = env.foundationReport().dependencyGraph;
    for (const name of Object.keys(graph)) {
      assert.deepStrictEqual(
        env.dependencyClosure(name),
        graph[name],
        `mismatch for ${name}`,
      );
    }
  });

  it('`dependencyClosure(<unknown>)` returns null', () => {
    const env = new Env();
    assert.strictEqual(env.dependencyClosure('no-such-construct'), null);
  });

  it('`dependencyClosure(name)` does not include the construct itself', () => {
    const env = new Env();
    const closure = env.dependencyClosure('!=');
    assert.ok(closure !== null);
    assert.ok(!closure.includes('!='), 'closure should not include the root');
  });

  it('tolerates dangling deps — skips unknown names cleanly', () => {
    const env = new Env();
    evaluate(`
(root-construct my-op
  (status host-primitive)
  (kind arithmetic-operator)
  (depends-on ghost-construct decimal-12-arithmetic))
`, { env });
    const closure = env.dependencyClosure('my-op');
    // `ghost-construct` is never registered; the closure surfaces the
    // dangling name in the list because the descriptor declared the
    // dependency. Tools downstream can detect that it has no entry in
    // the graph itself.
    assert.deepStrictEqual(closure, ['decimal-12-arithmetic', 'ghost-construct']);
  });

  it('formatFoundationReport renders the dependency graph section', () => {
    const env = new Env();
    const printed = formatFoundationReport(env.foundationReport());
    assert.match(printed, /dependency graph \(transitive\):/);
    assert.match(printed, /\+ → decimal-12-arithmetic/);
  });

  it('only renders entries with at least one dep', () => {
    const env = new Env();
    const printed = formatFoundationReport(env.foundationReport());
    // `decimal-12-arithmetic` is a leaf; it should not appear with an
    // arrow on its own line in the dependency-graph section.
    assert.ok(!/dependency graph \(transitive\):[\s\S]*decimal-12-arithmetic →/.test(printed));
  });

  it('buildDependencyGraph helper is exported and idempotent', () => {
    const env = new Env();
    const a = buildDependencyGraph(env);
    const b = buildDependencyGraph(env);
    assert.deepStrictEqual(a, b);
  });

  it('user-registered constructs surface in the graph immediately', () => {
    const env = new Env();
    evaluate(`
(root-construct fancy-op
  (status user-overridden)
  (kind custom-operator)
  (depends-on + -))
`, { env });
    const graph = env.foundationReport().dependencyGraph;
    assert.ok('fancy-op' in graph);
    const closure = graph['fancy-op'];
    assert.ok(closure.includes('+'));
    assert.ok(closure.includes('-'));
    assert.ok(closure.includes('decimal-12-arithmetic'));
  });

  it('handles cycles without infinite-looping (defensive BFS guard)', () => {
    const env = new Env();
    evaluate(`
(root-construct cyc-a
  (status host-primitive)
  (depends-on cyc-b))
(root-construct cyc-b
  (status host-primitive)
  (depends-on cyc-a))
`, { env });
    const closureA = env.dependencyClosure('cyc-a');
    const closureB = env.dependencyClosure('cyc-b');
    assert.deepStrictEqual(closureA, ['cyc-b']);
    assert.deepStrictEqual(closureB, ['cyc-a']);
  });
});
