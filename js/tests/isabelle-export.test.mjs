// Isabelle/HOL exporter tests for issue #62.
//
// The fixture pins the supported typed fragment and exported declaration
// order. The CLI test covers the requested API shape:
//   rml export isabelle <file.lino> -o <file.thy>

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { execFileSync } from 'node:child_process';
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { exportIsabelle } from '../src/rml-links.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..', '..');
const fixturePath = path.join(repoRoot, 'examples', 'isabelle-typed-fragment.lino');
const expectedPath = path.join(repoRoot, 'examples', 'isabelle-typed-fragment.thy');
const cliPath = path.resolve(here, '..', 'src', 'rml-links.mjs');

describe('exportIsabelle', () => {
  it('exports the typed fixture to the checked Isabelle/HOL theory fixture', () => {
    const source = readFileSync(fixturePath, 'utf8');
    const expected = readFileSync(expectedPath, 'utf8');
    const actual = exportIsabelle(source, {
      theoryName: 'Isabelle_Typed_Fragment',
    });
    assert.strictEqual(actual, expected);
  });

  it('rejects probabilistic assignments instead of approximating them', () => {
    assert.throws(
      () => exportIsabelle('((a = a) has probability 1)\n'),
      /probability assignments are outside the Isabelle exporter subset/,
    );
  });

  it('rejects dependent Pi codomains that HOL function types cannot represent', () => {
    assert.throws(
      () => exportIsabelle('(Vector: (Type 0) Vector)\n(f: (Pi (Natural n) (Vector n)))\n'),
      /dependent Pi codomain mentions "n"/,
    );
  });
});

describe('rml export isabelle CLI', () => {
  it('writes the requested .thy file', () => {
    const dir = mkdtempSync(path.join(tmpdir(), 'rml-isabelle-'));
    try {
      const input = path.join(dir, 'tiny.lino');
      const output = path.join(dir, 'Tiny.thy');
      writeFileSync(input, '(Natural: (Type 0) Natural)\n(zero: Natural zero)\n', 'utf8');
      execFileSync(process.execPath, [
        cliPath,
        'export',
        'isabelle',
        input,
        '-o',
        output,
        '--theory',
        'Tiny',
      ], { cwd: repoRoot });
      const rendered = readFileSync(output, 'utf8');
      assert.match(rendered, /^theory Tiny\n  imports Main\nbegin/);
      assert.match(rendered, /typedecl rml_natural/);
      assert.match(rendered, /rml_zero :: "rml_natural"/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
