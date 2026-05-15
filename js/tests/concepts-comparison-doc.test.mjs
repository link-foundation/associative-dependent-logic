// Tests for docs/CONCEPTS-COMPARISON.md and docs/FEATURE-COMPARISON.md
// (issue #167). These tests are deliberately deterministic and run against
// both the Markdown source and the public surface of the JS implementation,
// so that the matrix cannot quietly drift away from the code.
//
// Two things are checked:
//   1. Document structure: the new (correctly-spelled) files exist, contain
//      the expanded legend qualifiers, and the old `COMPARISION` filenames
//      remain as compatibility stubs that point to the new files.
//   2. RML claims: every RML capability that the matrix advertises as
//      available (whnf/nf/normal-form, (inductive ...), (coinductive ...),
//      (total ...), (coverage ...), modes, termination, tactic links,
//      ATP and SMT bridges, independent proof replay, structural and
//      definitional equality) is actually exposed by the JS module.
//
// The Rust suite mirrors this in rust/tests/concepts_comparison_doc_tests.rs
// so that drift between the two implementations fails both test suites.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  evaluate,
  whnf,
  nf,
  isStructurallySame,
  isConvertible,
  isTotal,
  isTerminating,
  isCovered,
  parseInductiveForm,
  parseCoinductiveForm,
  rewrite,
  simplify,
  runTactics,
  buildProof,
} from '../src/rml-links.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, '../..');
const CONCEPTS_NEW = path.join(REPO_ROOT, 'docs/CONCEPTS-COMPARISON.md');
const CONCEPTS_OLD = path.join(REPO_ROOT, 'docs/CONCEPTS-COMPARISION.md');
const FEATURE_NEW = path.join(REPO_ROOT, 'docs/FEATURE-COMPARISON.md');
const FEATURE_OLD = path.join(REPO_ROOT, 'docs/FEATURE-COMPARISION.md');

function read(file) {
  return fs.readFileSync(file, 'utf8');
}

describe('CONCEPTS-COMPARISON / FEATURE-COMPARISON: document structure', () => {
  it('the renamed (COMPARISON) files exist and are non-trivial', () => {
    const concepts = read(CONCEPTS_NEW);
    const feature = read(FEATURE_NEW);
    assert.ok(concepts.length > 5000, 'CONCEPTS-COMPARISON.md should be substantial');
    assert.ok(feature.length > 5000, 'FEATURE-COMPARISON.md should be substantial');
    assert.match(concepts, /^# Core Concept Comparison/m);
    assert.match(feature, /^# Product Feature Comparison/m);
  });

  it('the old (COMPARISION) filenames remain as compatibility stubs', () => {
    const concepts = read(CONCEPTS_OLD);
    const feature = read(FEATURE_OLD);
    assert.match(concepts, /compatibility stub/i);
    assert.match(feature, /compatibility stub/i);
    assert.match(concepts, /CONCEPTS-COMPARISON\.md/);
    assert.match(feature, /FEATURE-COMPARISON\.md/);
  });

  it('the expanded legend advertises Kernel/Library/Encoding/Runtime/Host/External/Prototype/Theory', () => {
    const concepts = read(CONCEPTS_NEW);
    for (const mark of [
      'Kernel',
      'Library',
      'Encoding',
      'Runtime',
      'Host',
      'External',
      'Prototype',
      'Theory',
      'Archive',
    ]) {
      assert.match(
        concepts,
        new RegExp(`\\| ${mark} \\|`),
        `legend should define the \`${mark}\` qualifier`,
      );
    }
  });

  it('the systems list separates provers/frameworks/languages from libraries/archives', () => {
    const concepts = read(CONCEPTS_NEW);
    assert.match(concepts, /### Provers, frameworks, and languages/);
    assert.match(concepts, /### Libraries and archives/);
    // Foundation and AFP must live in the libraries/archives section.
    const archiveIdx = concepts.indexOf('### Libraries and archives');
    const foundationIdx = concepts.indexOf('| Foundation |');
    const afpIdx = concepts.indexOf('| AFP |');
    assert.ok(archiveIdx > 0, 'libraries/archives section is missing');
    assert.ok(
      foundationIdx > archiveIdx,
      'Foundation should appear in the libraries/archives section',
    );
    assert.ok(
      afpIdx > archiveIdx,
      'AFP should appear in the libraries/archives section',
    );
  });

  it('drops the misleading "no ATP bridge" / "no independent proof replay" RML claims', () => {
    const concepts = read(CONCEPTS_NEW);
    assert.doesNotMatch(
      concepts,
      /no ATP bridge/,
      '"no ATP bridge" is stale — the (by atp ...) bridge exists',
    );
    // The positioning summary should explicitly say replay exists.
    assert.match(
      concepts,
      /independent proof-replay checker/i,
      'the matrix should advertise the independent proof-replay checker',
    );
  });

  it('includes the RML status note explaining host-implemented + runtime configuration', () => {
    const concepts = read(CONCEPTS_NEW);
    assert.match(concepts, /RML status note/);
    assert.match(concepts, /host-implemented/);
    assert.match(concepts, /runtime configuration/);
  });

  it('adds the equality-layers row distinguishing structural / assigned / numeric / definitional', () => {
    const concepts = read(CONCEPTS_NEW);
    assert.match(concepts, /Equality layers distinguished/);
    assert.match(concepts, /structural/);
    assert.match(concepts, /assigned/);
    assert.match(concepts, /numeric/);
    assert.match(concepts, /definitional|convertibility/);
  });

  it('rewrites Lambda Prolog / Twelf rows to avoid HOL-style theorem-proving / tactic claims', () => {
    const concepts = read(CONCEPTS_NEW);
    assert.match(concepts, /lambda Prolog.*Not HOL in the Isabelle\/HOL/i);
    assert.match(
      concepts,
      /No \/ N\/A: proof search and metatheorem checking exist, but not tactic-level/i,
    );
  });

  it('marks RML numeric/many-valued semantics as Yes (Runtime + Host)', () => {
    const concepts = read(CONCEPTS_NEW);
    for (const row of [
      'Numeric truth values in the core',
      'Configurable semantic range',
      'Configurable valence',
      'Fuzzy logic',
      'Probabilistic operators',
    ]) {
      // The row exists and the RML cell uses the "Runtime + Host" qualifier.
      const re = new RegExp(`\\| ${row} \\| Yes \\(Runtime \\+ Host\\):`);
      assert.match(concepts, re, `row "${row}" should be marked Yes (Runtime + Host)`);
    }
  });
});

describe('CONCEPTS-COMPARISON: RML claims match the JS implementation', () => {
  it('whnf and nf exist for the typed lambda fragment', () => {
    assert.strictEqual(typeof whnf, 'function', 'whnf must be exported');
    assert.strictEqual(typeof nf, 'function', 'nf must be exported');
  });

  it('(normal-form ...) is exercised by the self-evaluator', () => {
    // The self-evaluator surface form is (eval (normal-form expression)).
    // Smoke-test that the parser accepts it.
    const out = evaluate('(? (normal-form ((lambda (x) x) 1)))');
    assert.ok(Array.isArray(out.diagnostics));
    // We don't require a particular numeric result here; we only require
    // that the parser does not reject the surface form.
    assert.ok(out.diagnostics.every(d => d.code !== 'E001'));
  });

  it('(inductive ...) and (coinductive ...) declarations are parseable', () => {
    assert.strictEqual(typeof parseInductiveForm, 'function');
    assert.strictEqual(typeof parseCoinductiveForm, 'function');
  });

  it('(total ...), termination, and coverage forms are exposed', () => {
    assert.strictEqual(typeof isTotal, 'function');
    assert.strictEqual(typeof isTerminating, 'function');
    assert.strictEqual(typeof isCovered, 'function');
  });

  it('structural and definitional equality both exist', () => {
    assert.strictEqual(typeof isStructurallySame, 'function');
    assert.strictEqual(typeof isConvertible, 'function');
  });

  it('tactic links and proof-building primitives are exposed', () => {
    assert.strictEqual(typeof runTactics, 'function');
    assert.strictEqual(typeof rewrite, 'function');
    assert.strictEqual(typeof simplify, 'function');
    assert.strictEqual(typeof buildProof, 'function');
  });

  it('SMT and ATP bridges accept (by smt ...) / (by atp ...) tactic forms', () => {
    // The matrix says "Part (External): (by smt …) SMT-LIB trusted bridge"
    // and "(by atp …) records results as trusted external nodes". The
    // parser must accept those forms (full SMT/ATP execution depends on
    // external tools and is exercised in tactics.test.mjs with mocks).
    const out1 = evaluate('(? (by smt (= 1 1)))');
    const out2 = evaluate('(? (by atp (= 1 1)))');
    assert.ok(Array.isArray(out1.diagnostics));
    assert.ok(Array.isArray(out2.diagnostics));
    assert.ok(
      out1.diagnostics.every(d => d.code !== 'E001'),
      '(by smt ...) should parse',
    );
    assert.ok(
      out2.diagnostics.every(d => d.code !== 'E001'),
      '(by atp ...) should parse',
    );
  });

  it('independent proof-replay checker is shipped as a separate module', () => {
    const checkPath = path.join(REPO_ROOT, 'js/src/rml-check.mjs');
    assert.ok(fs.existsSync(checkPath), 'js/src/rml-check.mjs must exist');
  });
});
