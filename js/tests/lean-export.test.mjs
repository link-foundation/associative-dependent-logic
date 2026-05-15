// Lean 4 exporter tests for issue #60.
// The Rust suite mirrors these cases so the two public `rml` CLIs stay aligned.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { exportLean } from '../src/lean-export.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..', '..');
const fixturePath = path.join(repoRoot, 'examples', 'lean-export-basic.lino');
const expectedPath = path.join(repoRoot, 'examples', 'lean-export-basic.lean');

function readFixture() {
  return fs.readFileSync(fixturePath, 'utf8');
}

function readExpected() {
  return fs.readFileSync(expectedPath, 'utf8');
}

describe('exportLean()', () => {
  it('exports the supported typed subset to Lean 4 source', () => {
    const out = exportLean(readFixture(), { file: fixturePath });
    assert.deepStrictEqual(out.diagnostics, []);
    assert.strictEqual(out.source, readExpected());
  });

  it('rejects probabilistic forms with an E050 diagnostic', () => {
    const out = exportLean('(p: p is p)\n((p = p) has probability 1)\n', { file: 'prob.lino' });
    assert.strictEqual(out.source, '');
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E050');
    assert.match(out.diagnostics[0].message, /probabilistic/i);
    assert.strictEqual(out.diagnostics[0].span.file, 'prob.lino');
    assert.strictEqual(out.diagnostics[0].span.line, 2);
  });
});

describe('rml export lean CLI', () => {
  it('writes the Lean artifact to the -o path', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rml-lean-export-'));
    try {
      const outPath = path.join(dir, 'out.lean');
      const cli = path.join(repoRoot, 'js', 'src', 'rml-links.mjs');
      const result = spawnSync(process.execPath, [cli, 'export', 'lean', fixturePath, '-o', outPath], {
        encoding: 'utf8',
      });
      assert.strictEqual(result.status, 0, result.stderr);
      assert.strictEqual(result.stdout, '');
      assert.strictEqual(fs.readFileSync(outPath, 'utf8'), readExpected());
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
