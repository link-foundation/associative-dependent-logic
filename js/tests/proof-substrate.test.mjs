// Proof-object substrate tests (issue #97, Phase 3).
//
// The proof-substrate lets `.lino` programs declare rules of inference and
// concrete derivations as data, then verify the derivations by structural
// pattern matching against the declared rule. The three surface forms are:
//
//   (rule <name> (premise <pat>)... (conclusion <pat>))
//   (assumption <name> (judgement <judgement>)) / (axiom <name> ...)
//   (proof-object <name> (applies <rule>) (premise-by <dependency>)... (conclusion <judgement>))
//   (check-proof <name>)
//
// `?meta` leaves inside patterns are metavariables; repeated metavariables
// must structurally match. Failures emit an `E064` diagnostic and push `0`
// into the result stream; successes push `1`. Rules and proof objects are
// surfaced on `foundationReport()` so the trust audit can inspect them.
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97
import { describe, it } from 'node:test';
import assert from 'node:assert';
import {
  Env,
  evaluate,
  formatFoundationReport,
  parseRuleForm,
  parseProofAssumptionForm,
  parseProofObjectForm,
  matchProofPattern,
  checkProofObject,
} from '../src/rml-links.mjs';

describe('proof-object substrate as links', () => {
  it('parses a rule form into a structured descriptor', () => {
    const rule = parseRuleForm([
      'rule',
      'modus-ponens',
      ['premise', [['?a'], 'implies', ['?b']]],
      ['premise', '?a'],
      ['conclusion', '?b'],
    ]);
    assert.strictEqual(rule.name, 'modus-ponens');
    assert.strictEqual(rule.premises.length, 2);
    assert.strictEqual(rule.conclusion, '?b');
  });

  it('parses a proof-object form into a structured descriptor', () => {
    const po = parseProofObjectForm([
      'proof-object',
      'mp-1',
      ['applies', 'modus-ponens'],
      ['premise', ['raining', 'implies', 'wet']],
      ['premise-by', 'rain-implies-wet'],
      ['premise', 'raining'],
      ['uses', 'rain'],
      ['conclusion', 'wet'],
    ]);
    assert.strictEqual(po.name, 'mp-1');
    assert.strictEqual(po.rule, 'modus-ponens');
    assert.strictEqual(po.premises.length, 2);
    assert.deepStrictEqual(po.premiseRefs, ['rain-implies-wet', 'rain']);
    assert.strictEqual(po.conclusion, 'wet');
  });

  it('parses assumptions and axioms as explicit proof leaves', () => {
    const assumption = parseProofAssumptionForm([
      'assumption',
      'rain',
      ['judgement', 'raining'],
    ]);
    assert.deepStrictEqual(assumption, {
      name: 'rain',
      kind: 'assumption',
      judgement: 'raining',
    });
  });

  it('matches `?meta` leaves and binds them into the substitution map', () => {
    const subs = {};
    const ok = matchProofPattern(['?a', 'implies', '?b'], ['raining', 'implies', 'wet'], subs);
    assert.strictEqual(ok, true);
    assert.strictEqual(subs['?a'], 'raining');
    assert.strictEqual(subs['?b'], 'wet');
  });

  it('rejects inconsistent bindings for the same metavariable', () => {
    const subs = {};
    const ok = matchProofPattern(
      [['?a'], 'implies', ['?a']],
      [['raining'], 'implies', ['snowing']],
      subs,
    );
    assert.strictEqual(ok, false);
  });

  it('verifies a valid modus-ponens derivation end-to-end', () => {
    const out = evaluate(`
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

  it('fails with E064 when a premise does not match the rule pattern', () => {
    const out = evaluate(`
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(assumption sunny-now (judgement sunny))
(proof-object mp-bad
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (premise-by sunny-now)
  (conclusion wet))
(check-proof mp-bad)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /premise 2 does not match rule modus-ponens/);
  });

  it('fails with E064 when premise count differs from the rule arity', () => {
    const out = evaluate(`
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(proof-object mp-short
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (conclusion wet))
(check-proof mp-short)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /expected 2 premise\(s\) for rule modus-ponens, got 1/);
  });

  it('fails with E064 when the proof-object references an unknown rule', () => {
    const out = evaluate(`
(proof-object orphan
  (applies nonexistent-rule)
  (premise foo)
  (conclusion bar))
(check-proof orphan)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /references unknown rule nonexistent-rule/);
  });

  it('fails with E064 when (check-proof ...) targets an unknown proof-object', () => {
    const out = evaluate('(check-proof never-declared)');
    assert.deepStrictEqual(out.results, [0]);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /unknown proof-object never-declared/);
  });

  it('enforces metavariable consistency across premises and the conclusion', () => {
    // The rule says the conclusion's `?b` must be the same `?b` that
    // appeared in the implication premise. Swapping it must fail.
    const out = evaluate(`
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(assumption rain (judgement raining))
(proof-object mp-skew
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (premise-by rain)
  (conclusion snowing))
(check-proof mp-skew)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /conclusion does not match rule modus-ponens/);
  });

  it('rejects malformed rule forms with E064', () => {
    // The routing guard requires at least one `conclusion` clause to
    // distinguish proof-substrate rules from data-only `(rule ...)` forms
    // used by self-bootstrap files. Once routed in, additional malformed
    // structure (here: a premise clause with the wrong arity) is caught
    // by `parseRuleForm` and surfaced as E064.
    const out = evaluate(`
(rule bad-arity
  (premise ?a ?extra)
  (conclusion ?b))
`);
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(
      out.diagnostics[0].message,
      /\(premise <pat>\) requires exactly one pattern/,
    );
  });

  it('rejects malformed proof-object forms with E064', () => {
    // Missing (applies <rule>) clause.
    const out = evaluate(`
(proof-object missing-applies
  (premise foo)
  (conclusion bar))
`);
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /\(applies <rule>\) clause is required/);
  });

  it('rejects (check-proof ...) without a name argument', () => {
    const out = evaluate('(check-proof)');
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /requires a proof-object name/);
  });

  it('surfaces rules and proof objects on foundationReport()', () => {
    const env = new Env();
    evaluate(
      `
(rule reflexivity
  (conclusion (?a = ?a)))
(proof-object refl-a
  (applies reflexivity)
  (conclusion (apple = apple)))
`,
      { env },
    );
    const report = env.foundationReport();
    assert.deepStrictEqual(report.proofRules, [
      {
        name: 'reflexivity',
        premises: [],
        conclusion: '(?a = ?a)',
      },
    ]);
    assert.deepStrictEqual(report.proofObjects, [
      {
        name: 'refl-a',
        rule: 'reflexivity',
        premises: [],
        premiseRefs: [],
        conclusion: '(apple = apple)',
      },
    ]);
    const printed = formatFoundationReport(report);
    assert.match(printed, /proof rules:/);
    assert.match(printed, /reflexivity \(0 premises → \(\?a = \?a\)\)/);
    assert.match(printed, /proof objects:/);
    assert.match(printed, /refl-a : applies reflexivity \(0 premises → \(apple = apple\)\)/);
  });

  it('surfaces proof assumptions and dependency refs on foundationReport()', () => {
    const env = new Env();
    evaluate(
      `
(rule identity
  (premise ?a)
  (conclusion ?a))
(assumption rain (judgement raining))
(proof-object id-rain
  (applies identity)
  (premise-by rain)
  (conclusion raining))
`,
      { env },
    );
    const report = env.foundationReport();
    assert.deepStrictEqual(report.proofAssumptions, [
      {
        name: 'rain',
        kind: 'assumption',
        judgement: 'raining',
      },
    ]);
    assert.deepStrictEqual(report.proofObjects[0].premiseRefs, ['rain']);
    const printed = formatFoundationReport(report);
    assert.match(printed, /proof assumptions:/);
    assert.match(printed, /rain \[assumption\] : raining/);
    assert.match(printed, /id-rain : applies identity \(0 premises using rain → raining\)/);
  });

  it('lets `checkProofObject()` return a substitution witness on success', () => {
    const env = new Env();
    evaluate(
      `
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
`,
      { env },
    );
    const verdict = checkProofObject(env, 'mp-rain');
    assert.strictEqual(verdict.ok, true);
    assert.strictEqual(verdict.substitution['?a'], 'raining');
    assert.strictEqual(verdict.substitution['?b'], 'wet');
    assert.deepStrictEqual(verdict.dependencies, ['rain-implies-wet', 'rain']);
  });

  it('rejects raw proof-object premises that cite no proof dependency', () => {
    const out = evaluate(`
(rule identity
  (premise ?a)
  (conclusion ?a))
(proof-object raw
  (applies identity)
  (premise raining)
  (conclusion raining))
(check-proof raw)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /premise 1 is unjustified/);
  });

  it('accepts proof-object dependencies produced by other proof objects', () => {
    const out = evaluate(`
(rule identity
  (premise ?a)
  (conclusion ?a))
(rule reflexivity
  (conclusion (?a = ?a)))
(proof-object refl-rain
  (applies reflexivity)
  (conclusion (raining = raining)))
(proof-object id-rain
  (applies identity)
  (premise-by refl-rain)
  (conclusion (raining = raining)))
(check-proof id-rain)
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('checks a links-level proof-checking relation as data', () => {
    const env = new Env();
    const out = evaluate(`
(root-construct proof-checking-relation
  (kind checking-relation)
  (status links-defined)
  (semantic-status links-checked)
  (depends-on proof-replay structural-equality proof-object))
(rule proof-checks-modus-ponens
  (premise (?implication-proof checks-as (?antecedent implies ?consequent)))
  (premise (?antecedent-proof checks-as ?antecedent))
  (premise (?proof applies-rule modus-ponens))
  (premise (?proof uses-proof ?implication-proof))
  (premise (?proof uses-proof ?antecedent-proof))
  (conclusion (?proof checks-as ?consequent)))
(axiom rain-implies-wet-proof
  (judgement (rain-implies-wet checks-as (raining implies wet))))
(axiom rain-proof
  (judgement (rain-proof checks-as raining)))
(axiom mp-rain-applies
  (judgement (mp-rain applies-rule modus-ponens)))
(axiom mp-rain-uses-implication
  (judgement (mp-rain uses-proof rain-implies-wet)))
(axiom mp-rain-uses-antecedent
  (judgement (mp-rain uses-proof rain-proof)))
(proof-object mp-rain-checks-wet
  (applies proof-checks-modus-ponens)
  (premise-by rain-implies-wet-proof)
  (premise-by rain-proof)
  (premise-by mp-rain-applies)
  (premise-by mp-rain-uses-implication)
  (premise-by mp-rain-uses-antecedent)
  (conclusion (mp-rain checks-as wet)))
(check-proof mp-rain-checks-wet)
`, { env });
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);

    const report = env.foundationReport();
    const proofChecking = report.rootConstructs.find(rc => rc.name === 'proof-checking-relation');
    assert.ok(proofChecking);
    assert.strictEqual(proofChecking.semanticStatus, 'links-checked');
    const relationRule = report.proofRules.find(r => r.name === 'proof-checks-modus-ponens');
    assert.ok(relationRule);
    assert.strictEqual(relationRule.premises.length, 5);
    assert.ok(report.bySemanticStatus['links-checked'].includes('proof-checking-relation'));
  });

  it('detects cyclic proof-object dependencies', () => {
    const out = evaluate(`
(rule identity
  (premise ?a)
  (conclusion ?a))
(proof-object a
  (applies identity)
  (premise-by b)
  (conclusion rain))
(proof-object b
  (applies identity)
  (premise-by a)
  (conclusion rain))
(check-proof a)
`);
    assert.deepStrictEqual(out.results, [0]);
    assert.strictEqual(out.diagnostics[0].code, 'E064');
    assert.match(out.diagnostics[0].message, /cyclic proof dependency: a -> b -> a/);
  });

  it('does not hijack existing `(rule ...)` data forms from self-bootstrap files', () => {
    // The self-grammar bootstrap uses `(rule <name> (sequence ...) ...)`.
    // The proof substrate must let these forms pass through to the
    // legacy data path with no E064 diagnostic and no result emission.
    const out = evaluate(`
(rule source-for-evaluation
  (sequence parse normalize evaluate)
  (normalizes-to document))
`);
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, []);
  });

  it('keeps proof rules separate from evaluator behaviour', () => {
    // Declaring a rule must not affect baseline query semantics: the
    // engine still resolves the arithmetic query exactly as it would
    // without the rule, the diagnostics list is empty, and the only
    // result emitted is the query's truth value.
    const withRule = evaluate(`
(rule unused
  (premise ?a)
  (conclusion ?a))
(? (1 + 2))
`);
    const withoutRule = evaluate('(? (1 + 2))');
    assert.deepStrictEqual(withRule.diagnostics, []);
    assert.deepStrictEqual(withRule.results, withoutRule.results);
  });
});
