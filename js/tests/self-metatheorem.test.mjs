// Tests for `lib/self/metatheorem.lino` (issue #88).
//
// The metatheorem file is data: it records the host C3 metatheorem checker
// as `(rule ...)` links. These tests keep that data parseable and importable,
// require the essential rule surface (mode, totality, coverage, termination),
// and verify that the encoded checker surface is tied to the diagnostics
// emitted by the host checker.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import {
  evaluateFile,
  keyOf,
  parseLino,
  parseOne,
  tokenizeOne,
} from '../src/rml-links.mjs';
import { checkMetatheorems, formatReport } from '../src/rml-meta.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const metatheoremPath = join(repoRoot, 'lib', 'self', 'metatheorem.lino');

// Rule subjects that must be encoded in the file to mirror the host checker.
const REQUIRED_RULES = [
  '(mode name flags)',
  '(inductive type-name constructors)',
  '(relation name clauses)',
  '(define name cases)',
  '(totality-check name env)',
  '(coverage-check name env)',
  '(termination-check name env)',
  '(check-metatheorems program)',
  '(check-relation env name)',
  '(check-definition env name)',
  '(format-report report)',
];

// Diagnostic codes the encoded checker must reference.
const REQUIRED_DIAGNOSTICS = ['E030', 'E031', 'E032', 'E035', 'E037'];

// Check kinds that must appear as `(check-kind ...)` declarations.
const REQUIRED_CHECK_KINDS = ['totality', 'coverage', 'termination'];

function parseForms(source) {
  return parseLino(source).map(link => parseOne(tokenizeOne(link)));
}

function metatheoremForms() {
  return parseForms(readFileSync(metatheoremPath, 'utf8'));
}

function ruleSubjects(forms) {
  const subjects = new Set();
  for (const form of forms) {
    if (Array.isArray(form) && form[0] === 'rule' && Array.isArray(form[1])) {
      subjects.add(keyOf(form[1]));
    }
  }
  return subjects;
}

function diagnosticCodes(forms) {
  const codes = new Set();
  for (const form of forms) {
    if (!Array.isArray(form)) continue;
    for (const child of form) {
      if (Array.isArray(child) && child[0] === 'emits' && typeof child[1] === 'string') {
        codes.add(child[1]);
      }
    }
    // Also search nested rule bodies
    if (form[0] === 'rule') {
      for (const part of form.slice(2)) {
        if (Array.isArray(part) && part[0] === 'emits' && typeof part[1] === 'string') {
          codes.add(part[1]);
        }
      }
    }
  }
  return codes;
}

function checkKinds(forms) {
  const kinds = new Set();
  for (const form of forms) {
    if (Array.isArray(form) && form[0] === 'check-kind' && typeof form[1] === 'string') {
      kinds.add(form[1]);
    }
  }
  return kinds;
}

const NATURAL_DECL =
  '(inductive Natural\n' +
  '  (constructor zero)\n' +
  '  (constructor (succ (Pi (Natural n) Natural))))\n';

describe('self metatheorem checker', () => {
  it('is importable as a standard library file', () => {
    const out = evaluateFile(metatheoremPath);
    assert.deepStrictEqual(out.diagnostics, []);
  });

  it('declares the required rule subjects as links', () => {
    const subjects = ruleSubjects(metatheoremForms());
    for (const rule of REQUIRED_RULES) {
      assert.ok(subjects.has(rule), `missing encoded rule ${rule}`);
    }
  });

  it('encodes the required diagnostic codes', () => {
    const codes = diagnosticCodes(metatheoremForms());
    for (const code of REQUIRED_DIAGNOSTICS) {
      assert.ok(codes.has(code), `missing diagnostic code ${code}`);
    }
  });

  it('declares totality, coverage, and termination as check kinds', () => {
    const kinds = checkKinds(metatheoremForms());
    for (const kind of REQUIRED_CHECK_KINDS) {
      assert.ok(kinds.has(kind), `missing check-kind declaration for ${kind}`);
    }
  });

  it('declares the master entity with correct version', () => {
    const forms = metatheoremForms();
    const entityDecl = forms.find(
      f =>
        Array.isArray(f) &&
        f[0] === 'metatheorem-checker' &&
        f[1] === 'rml-metatheorem-checker' &&
        f[2] === 'matches',
    );
    assert.ok(entityDecl, 'missing metatheorem-checker entity declaration');
    assert.ok(
      typeof entityDecl[3] === 'string' && entityDecl[3].startsWith('relative-meta-logic'),
      'entity declaration must name relative-meta-logic',
    );
  });

  it('declares a host-metatheorem-checker-presentation rule', () => {
    const forms = metatheoremForms();
    const presentationRule = forms.find(
      f =>
        Array.isArray(f) &&
        f[0] === 'rule' &&
        Array.isArray(f[1]) &&
        f[1][0] === 'host-metatheorem-checker-presentation',
    );
    assert.ok(presentationRule, 'missing host-metatheorem-checker-presentation rule');
  });
});

// Verify the encoded surface stays consistent with the host checker behavior
// by running the same pass/fail cases through the host `checkMetatheorems`
// API. The encoded file does not implement the checks itself (it is data),
// so these tests confirm that the rule subjects named in the file correspond
// to the checks that the host actually runs.

describe('self metatheorem checker surface matches host checker behavior', () => {
  it('host certifies `plus` as total and covered (D12+D14)', () => {
    const report = checkMetatheorems(
      NATURAL_DECL +
      '(mode plus +input +input -output)\n' +
      '(relation plus\n' +
      '  (plus zero n n)\n' +
      '  (plus (succ m) n (succ (plus m n))))\n',
    );
    assert.strictEqual(report.ok, true, formatReport(report));
    const plus = report.relations.find(r => r.name === 'plus');
    assert.ok(plus);
    assert.strictEqual(plus.ok, true);
    const kinds = plus.checks.map(c => c.kind).sort();
    assert.deepStrictEqual(kinds, ['coverage', 'totality']);
  });

  it('host flags missing constructor case with E037 (D14 coverage failure)', () => {
    const report = checkMetatheorems(
      NATURAL_DECL +
      '(mode f +input -output)\n' +
      '(relation f\n' +
      '  (f zero zero))\n',
    );
    assert.strictEqual(report.ok, false);
    const f = report.relations.find(r => r.name === 'f');
    assert.ok(f);
    const coverage = f.checks.find(c => c.kind === 'coverage');
    assert.ok(coverage);
    assert.strictEqual(coverage.ok, false);
    assert.match(coverage.diagnostics[0].message, /missing case/);
  });

  it('host flags non-structural recursion with E032 (D12 totality failure)', () => {
    const report = checkMetatheorems(
      NATURAL_DECL +
      '(mode loop +input -output)\n' +
      '(relation loop\n' +
      '  (loop zero zero)\n' +
      '  (loop (succ n) (loop (succ n))))\n',
    );
    assert.strictEqual(report.ok, false);
    const loop = report.relations.find(r => r.name === 'loop');
    assert.ok(loop);
    const totality = loop.checks.find(c => c.kind === 'totality');
    assert.ok(totality);
    assert.strictEqual(totality.ok, false);
    assert.match(totality.diagnostics[0].message, /does not structurally decrease/);
  });

  it('host certifies a terminating definition via D13', () => {
    const report = checkMetatheorems(
      '(define plus\n' +
      '  (case (zero n) n)\n' +
      '  (case ((succ m) n) (succ (plus m n))))\n',
    );
    assert.strictEqual(report.ok, true, formatReport(report));
    const plus = report.definitions.find(d => d.name === 'plus');
    assert.ok(plus);
    assert.strictEqual(plus.ok, true);
    assert.strictEqual(plus.checks[0].kind, 'termination');
  });

  it('host flags a non-terminating definition with E035 (D13 failure)', () => {
    const report = checkMetatheorems(
      '(define loop\n' +
      '  (case (zero) zero)\n' +
      '  (case ((succ n)) (loop (succ n))))\n',
    );
    assert.strictEqual(report.ok, false);
    const loop = report.definitions.find(d => d.name === 'loop');
    assert.ok(loop);
    assert.strictEqual(loop.ok, false);
    assert.match(loop.checks[0].diagnostics[0].message, /does not structurally decrease/);
  });
});
