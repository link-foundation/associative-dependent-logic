// Regression tests for issue #26 planning and filing metadata.

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const FILING_DIR = path.join(REPO_ROOT, 'experiments', 'issue-26-filing');

function read(rel) {
  return fs.readFileSync(path.join(REPO_ROOT, rel), 'utf8');
}

function readJson(rel) {
  return JSON.parse(read(rel));
}

const issues = readJson('experiments/issue-26-filing/issues.json');
const state = readJson('experiments/issue-26-filing/state.json');
const idToNumber = state.idToNumber;

describe('issue #26 filed plan metadata', () => {
  it('tracks 66 feature issues plus the J-EPIC tracking issue', () => {
    assert.equal(issues.length, 67);
    assert.equal(issues.filter((issue) => issue.id !== 'J-EPIC').length, 66);
    assert.ok(issues.some((issue) => issue.id === 'J-EPIC'));

    const ids = issues.map((issue) => issue.id);
    assert.equal(new Set(ids).size, ids.length, 'plan IDs must be unique');
    assert.deepEqual(Object.keys(idToNumber).sort(), [...ids].sort());
    assert.equal(Object.values(idToNumber).filter((number) => number === 68).length, 0);
  });

  it('keeps the committed issue numbers unique and anchored to GitHub issues', () => {
    const numbers = Object.values(idToNumber);
    assert.equal(new Set(numbers).size, numbers.length);
    assert.equal(idToNumber.A1, 28);
    assert.equal(idToNumber['J-EPIC'], 95);

    for (const issue of issues) {
      const number = idToNumber[issue.id];
      assert.equal(typeof number, 'number', `${issue.id} missing GitHub number`);
      assert.ok(number >= 28, `${issue.id} has unexpected issue number ${number}`);
    }
  });

  it('has valid dependency references and no cycles', () => {
    const byId = new Map(issues.map((issue) => [issue.id, issue]));

    for (const issue of issues) {
      for (const dep of issue.depends || []) {
        assert.ok(byId.has(dep), `${issue.id} has unknown dependency ${dep}`);
      }
      for (const block of issue.blocks || []) {
        assert.ok(byId.has(block), `${issue.id} has unknown block ${block}`);
      }
    }

    const visited = new Set();
    const stack = new Set();
    function visit(id, trace = []) {
      if (stack.has(id)) {
        assert.fail(`dependency cycle: ${[...trace, id].join(' -> ')}`);
      }
      if (visited.has(id)) return;
      stack.add(id);
      for (const dep of byId.get(id).depends || []) visit(dep, [...trace, id]);
      stack.delete(id);
      visited.add(id);
    }

    for (const issue of issues) visit(issue.id);
  });

  it('passes the standalone validator used by the filing workflow', () => {
    const output = execFileSync(process.execPath, ['validate.mjs'], {
      cwd: FILING_DIR,
      encoding: 'utf8',
    });

    assert.match(output, /Issues: 67, problems: 0/);
    assert.match(output, /Validation complete\./);
  });

  it('renders filed issue bodies with real cross-references in dry-run update mode', () => {
    const output = execFileSync(
      process.execPath,
      ['file-issues.mjs', '--update', '--dry-run', '--only=A1,J-EPIC'],
      { cwd: FILING_DIR, encoding: 'utf8' },
    );

    assert.match(output, /---- update #28 \(A1\) ----/);
    assert.match(output, /---- update #95 \(J-EPIC\) ----/);
    assert.match(output, /CONCEPTS-COMPARISON\.md/);
    assert.doesNotMatch(output, /\(planned\)/);
  });
});

describe('issue #26 documentation consistency', () => {
  it('uses the canonical comparison document names for active source links', () => {
    const readme = read('docs/case-studies/issue-26/README.md');
    assert.ok(readme.includes('CONCEPTS-COMPARISON.md'));
    assert.ok(readme.includes('FEATURE-COMPARISON.md'));
    assert.ok(
      readme.includes('compatibility stubs'),
      'README should explain the historical misspelled paths',
    );

    for (const rel of [
      'docs/case-studies/issue-26/gap-matrix.md',
      'docs/case-studies/issue-26/issue-plan.md',
      'experiments/issue-26-filing/file-issues.mjs',
    ]) {
      const text = read(rel);
      assert.ok(text.includes('COMPARISON.md'), `${rel} should reference canonical docs`);
      assert.doesNotMatch(text, /COMPARISION\.md/, `${rel} should not use misspelled docs`);
    }
  });

  it('states the corrected issue counts in the plan and completion audits', () => {
    const issuePlan = read('docs/case-studies/issue-26/issue-plan.md');
    assert.match(issuePlan, /66 planned feature issues \+ 1 tracking epic = 67 GitHub issues/);
    assert.doesNotMatch(issuePlan, /67 planned issues \+ 1 tracking epic = 68/);

    const completionAudit = read('docs/case-studies/issue-26/completion-audit.md');
    assert.match(completionAudit, /66 planned feature issues/);
    assert.match(completionAudit, /All 67 filed issues are closed/);

    const epicAudit = read('docs/case-studies/issue-95/README.md');
    assert.match(epicAudit, /66 phase issues/);
    assert.match(epicAudit, /66 \/ 66 phase issues closed/);
    assert.doesNotMatch(epicAudit, /67 \/ 67 phase issues closed/);
  });

  it('keeps the filed-issue index aligned with the generated state file', () => {
    const issuePlan = read('docs/case-studies/issue-26/issue-plan.md');

    for (const issue of issues) {
      const number = idToNumber[issue.id];
      assert.ok(
        issuePlan.includes(`| ${issue.id} | [#${number}]`),
        `filed-issue index missing ${issue.id} -> #${number}`,
      );
    }
  });

  it('documents the verification commands for unit, integration, and e2e coverage', () => {
    const audit = read('docs/case-studies/issue-26/completion-audit.md');

    for (const expected of [
      'cd js && npm test',
      'cd rust && cargo test --all-targets',
      'node scripts/check-corpus-parity.mjs',
      'cd js && npm run test:bootstrap',
      'cd js && npm run test:playground',
      'cd js && npm run lint:english',
    ]) {
      assert.ok(audit.includes(expected), `completion audit missing ${expected}`);
    }
  });
});
