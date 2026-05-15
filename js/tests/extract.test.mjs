// Program extraction tests (issue #66).
// A typed, non-probabilistic lambda program should extract to runnable
// JavaScript/Rust source, while probabilistic forms are rejected.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { extractProgram } from '../src/rml-links.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const jsRoot = path.resolve(here, '..');

const PROGRAM = `
(Natural: (Type 0) Natural)
(inc: lambda (Natural x) (x + 1))
(double: lambda (Natural x) (x * 2))
(combo: lambda (Natural x) (apply double (apply inc x)))
(? ((apply combo 3) = 8))
`;

function makeTmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'rml-extract-js-'));
}

describe('extractProgram()', () => {
  it('extracts a typed lambda program to runnable JavaScript with tests', () => {
    const source = extractProgram(PROGRAM, 'js');
    assert.match(source, /export function inc\(x\)/);
    assert.match(source, /export function combo\(x\)/);
    assert.match(source, /__runRmlExtractedTests/);

    const dir = makeTmp();
    try {
      const generated = path.join(dir, 'program.mjs');
      fs.writeFileSync(generated, source);
      execFileSync(process.execPath, [generated], { stdio: 'pipe' });

      const harness = path.join(dir, 'harness.mjs');
      fs.writeFileSync(
        harness,
        `import { combo } from ${JSON.stringify(pathToFileURL(generated).href)};\n` +
          `if (combo(3) !== 8) throw new Error('combo(3) did not extract correctly');\n`,
      );
      execFileSync(process.execPath, [harness], { stdio: 'pipe' });
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it('extracts the same typed lambda program to Rust with embedded tests', () => {
    const source = extractProgram(PROGRAM, 'rust');
    assert.match(source, /pub fn inc\(x: f64\) -> f64/);
    assert.match(source, /pub fn combo\(x: f64\) -> f64/);
    assert.match(source, /fn rml_query_1/);
  });

  it('rejects probabilistic forms instead of compiling them', () => {
    assert.throws(
      () => extractProgram('((a = a) has probability 1)', 'js'),
      /probability/i,
    );
  });
});

describe('rml extract CLI', () => {
  it('prints extracted JavaScript source', () => {
    const dir = makeTmp();
    try {
      const program = path.join(dir, 'program.lino');
      fs.writeFileSync(program, PROGRAM);
      const output = execFileSync(
        process.execPath,
        [path.join(jsRoot, 'src', 'rml-links.mjs'), 'extract', 'js', program],
        { cwd: jsRoot, encoding: 'utf8' },
      );
      assert.match(output, /export function combo\(x\)/);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
