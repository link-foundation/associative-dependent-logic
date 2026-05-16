// Pure-links strict mode tests (issue #97, Phase 6).
//
// `(strict-foundation pure-links)` flips the strict audit on for every
// subsequent query; any operator inside the queried form whose registered
// root-construct status is `host-primitive` or `host-derived` triggers an
// E065 diagnostic. `(allow-host-primitive <name>...)` lets a program opt in
// to specific constructs while keeping everything else strict. The mode is
// surfaced on `foundationReport()` so the trust audit can prove the engine
// is running in pure-links territory.
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97
import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  Env,
  evaluate,
  formatFoundationReport,
  parseStrictFoundationForm,
  parseAllowHostPrimitiveForm,
  scanPureLinksOffenders,
} from '../src/rml-links.mjs';

describe('pure-links strict mode', () => {
  it('is off by default — legacy programs run unchanged', () => {
    const out = evaluate('(? (1 + 2))');
    assert.deepStrictEqual(out.diagnostics, []);
    assert.strictEqual(out.results.length, 1);
  });

  it('emits E065 when a query depends on a host-primitive arithmetic op', () => {
    const out = evaluate(`
(strict-foundation pure-links)
(? (1 + 2))
`);
    assert.strictEqual(out.results.length, 1);
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E065');
    assert.match(out.diagnostics[0].message, /pure-links strict mode/);
    assert.match(out.diagnostics[0].message, /\+/);
  });

  it('lists every offending construct in a single E065 diagnostic', () => {
    const out = evaluate(`
(strict-foundation pure-links)
(? ((1 + 2) - (3 * 4)))
`);
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E065');
    // The scanner reports offenders sorted/deduplicated.
    assert.match(out.diagnostics[0].message, /\*/);
    assert.match(out.diagnostics[0].message, /\+/);
    assert.match(out.diagnostics[0].message, /-/);
  });

  it('flags host-derived constructs too (e.g. `!=`)', () => {
    const out = evaluate(`
(strict-foundation pure-links)
(? (a != b))
`);
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E065');
    assert.match(out.diagnostics[0].message, /!=/);
  });

  it('flags user-configurable truth operators whose active implementation is still host-backed', () => {
    const out = evaluate(`
(strict-foundation pure-links)
(? (1 and 0))
(? (not 1))
`);
    assert.strictEqual(out.diagnostics.length, 2);
    assert.strictEqual(out.diagnostics[0].code, 'E065');
    assert.strictEqual(out.diagnostics[1].code, 'E065');
    assert.match(out.diagnostics[0].message, /and -> avg -> host-primitive/);
    assert.match(out.diagnostics[1].message, /not -> decimal-12-arithmetic -> host-primitive/);
    assert.strictEqual(out.results.length, 2);
  });

  it('accepts truth operators when the active foundation provides links-defined truth tables', () => {
    const out = evaluate(`
(strict-foundation pure-links)
(with-foundation boolean-links
  (? (1 and 0))
  (? (not 1)))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0, 0]);
  });

  it('honours `(allow-host-primitive ...)` to whitelist specific constructs', () => {
    const out = evaluate(`
(strict-foundation pure-links)
(allow-host-primitive + -)
(? (1 + 2))
(? (5 - 2))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.strictEqual(out.results.length, 2);
  });

  it('still flags constructs that are not in the allow list', () => {
    const out = evaluate(`
(strict-foundation pure-links)
(allow-host-primitive +)
(? (1 + 2))
(? (3 * 4))
`);
    assert.strictEqual(out.results.length, 2);
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E065');
    assert.match(out.diagnostics[0].message, /\*/);
  });

  it('does not affect query results — strict mode is observation-only', () => {
    // Strict mode should not change `results` even when it raises E065.
    // It is a diagnostic-only audit, never silently rewrites the truth
    // value of a query.
    const plain = evaluate('(? (1 + 2))');
    const strict = evaluate(`
(strict-foundation pure-links)
(? (1 + 2))
`);
    assert.deepStrictEqual(plain.results, strict.results);
  });

  it('rejects an unknown strict-foundation profile with E065', () => {
    const out = evaluate('(strict-foundation handwritten)');
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E065');
    assert.match(out.diagnostics[0].message, /unknown strict-foundation profile/);
  });

  it('rejects malformed strict-foundation forms with E065', () => {
    const out = evaluate('(strict-foundation)');
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E065');
    assert.match(out.diagnostics[0].message, /requires a single profile name/);
  });

  it('rejects malformed allow-host-primitive forms with E065', () => {
    const out = evaluate('(allow-host-primitive)');
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E065');
    assert.match(out.diagnostics[0].message, /requires at least one construct name/);
  });

  it('surfaces strict-pure-links state on foundationReport()', () => {
    const env = new Env();
    evaluate(`
(strict-foundation pure-links)
(allow-host-primitive + -)
`, { env });
    const report = env.foundationReport();
    assert.strictEqual(report.strictPureLinks, true);
    assert.deepStrictEqual(report.allowedHostPrimitives, ['+', '-']);
    const printed = formatFoundationReport(report);
    assert.match(printed, /pure-links strict mode: on/);
    assert.match(printed, /allowed host primitives: \+, -/);
  });

  it('parseStrictFoundationForm parses the profile name', () => {
    const decl = parseStrictFoundationForm(['strict-foundation', 'pure-links']);
    assert.strictEqual(decl.profile, 'pure-links');
  });

  it('parseAllowHostPrimitiveForm accepts multiple construct names', () => {
    const decl = parseAllowHostPrimitiveForm(['allow-host-primitive', '+', '-', '*']);
    assert.deepStrictEqual(decl.names, ['+', '-', '*']);
  });

  it('scanPureLinksOffenders returns [] when strict mode is off', () => {
    const env = new Env();
    assert.deepStrictEqual(scanPureLinksOffenders([['1', '+', '2']], env), []);
  });

  it('scanPureLinksOffenders surfaces every host-primitive operator', () => {
    const env = new Env();
    env.strictPureLinks = true;
    const offenders = scanPureLinksOffenders([['1', '+', '2'], '-', ['3', '*', '4']], env);
    // `+`, `-`, `*` are all host-backed through decimal-12 arithmetic; the
    // scanner returns transitive paths sorted/deduplicated.
    assert.deepStrictEqual(offenders, [
      '* -> decimal-12-arithmetic -> host-primitive',
      '+ -> decimal-12-arithmetic -> host-primitive',
      '- -> decimal-12-arithmetic -> host-primitive',
    ]);
  });

  it('lets links-encoded self-bootstrap files keep working under strict mode', () => {
    // The self-bootstrap files use only `links-encoded` constructs at the
    // top level. Strict mode must not reject them. We synthesise the
    // pattern from a (rule ...) data form (the kind self-grammar.lino uses)
    // and confirm it does not trigger an E065.
    const out = evaluate(`
(strict-foundation pure-links)
(rule source-for-evaluation
  (sequence parse normalize evaluate)
  (normalizes-to document))
`);
    assert.deepStrictEqual(out.diagnostics, []);
  });

  it('lets the proof substrate stay clean under strict mode', () => {
    // A `(check-proof ...)` query has nothing to do with host-primitive
    // arithmetic. Strict mode must not interfere with proof-checking.
    const out = evaluate(`
(strict-foundation pure-links)
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(assumption rain (judgement raining))
(proof-object mp-rain
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (premise-by rain)
  (conclusion wet))
(check-proof mp-rain)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('allows queries that only reference user constants under strict mode', () => {
    // No operator at all — `?` head, plain constants. Should pass.
    const out = evaluate(`
(strict-foundation pure-links)
(? 1)
(? 0)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1, 0]);
  });
});
