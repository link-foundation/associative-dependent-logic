// Tests for `(import "...")` and `evaluateFile()` (issue #33).
// Covers linear chains, diamond imports, cycle detection (E007),
// missing-file errors, and that diagnostics from imported files surface
// with their original spans intact. The Rust suite mirrors these in
// rust/tests/imports_tests.rs to keep the two implementations in lock-step.

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  evaluate,
  evaluateFile,
  formatDiagnostic,
} from '../src/rml-links.mjs';

function makeTmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'rml-import-'));
}

describe('evaluateFile() loads a file from disk', () => {
  let dir;
  before(() => {
    dir = makeTmp();
    fs.writeFileSync(path.join(dir, 'kb.lino'), '(a: a is a)\n((a = a) has probability 1)\n(? (a = a))\n');
  });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('returns the structured evaluate() shape', () => {
    const out = evaluateFile(path.join(dir, 'kb.lino'));
    assert.ok(Array.isArray(out.results));
    assert.ok(Array.isArray(out.diagnostics));
    assert.strictEqual(out.diagnostics.length, 0);
    assert.deepStrictEqual(out.results, [1]);
  });

  it('reports a missing file as an E007 diagnostic, not a thrown error', () => {
    let threw = false;
    let out;
    try {
      out = evaluateFile(path.join(dir, 'no-such.lino'));
    } catch (_) {
      threw = true;
    }
    assert.strictEqual(threw, false, 'evaluateFile() must not throw');
    assert.ok(out.diagnostics.length >= 1);
    assert.strictEqual(out.diagnostics[0].code, 'E007');
  });
});

describe('(import "...") — linear chain', () => {
  let dir;
  before(() => {
    dir = makeTmp();
    fs.writeFileSync(path.join(dir, 'leaf.lino'), '(z: z is z)\n((z = z) has probability 1)\n');
    fs.writeFileSync(path.join(dir, 'mid.lino'), '(import "leaf.lino")\n');
    fs.writeFileSync(path.join(dir, 'top.lino'), '(import "mid.lino")\n(? (z = z))\n');
  });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('loads declarations across a multi-step chain', () => {
    const out = evaluateFile(path.join(dir, 'top.lino'));
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });
});

describe('(import "...") — diamond pattern', () => {
  let dir;
  before(() => {
    dir = makeTmp();
    // shared.lino is imported from both b.lino and c.lino; main imports both.
    // Without caching, shared.lino would be loaded twice — that's fine for
    // pure declarations but breaks operator redefinitions. The cache makes
    // this a no-op the second time.
    fs.writeFileSync(path.join(dir, 'shared.lino'), '(d: d is d)\n((d = d) has probability 1)\n');
    fs.writeFileSync(path.join(dir, 'b.lino'), '(import "shared.lino")\n');
    fs.writeFileSync(path.join(dir, 'c.lino'), '(import "shared.lino")\n');
    fs.writeFileSync(path.join(dir, 'main.lino'), '(import "b.lino")\n(import "c.lino")\n(? (d = d))\n');
  });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('loads each file at most once even when imported via multiple paths', () => {
    const out = evaluateFile(path.join(dir, 'main.lino'));
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [1]);
  });
});

describe('(import "...") — cycle detection', () => {
  let dir;
  before(() => {
    dir = makeTmp();
    fs.writeFileSync(path.join(dir, 'a.lino'), '(import "b.lino")\n');
    fs.writeFileSync(path.join(dir, 'b.lino'), '(import "a.lino")\n');
    fs.writeFileSync(path.join(dir, 'self.lino'), '(import "self.lino")\n');
    fs.writeFileSync(path.join(dir, 'tri-a.lino'), '(import "tri-b.lino")\n');
    fs.writeFileSync(path.join(dir, 'tri-b.lino'), '(import "tri-c.lino")\n');
    fs.writeFileSync(path.join(dir, 'tri-c.lino'), '(import "tri-a.lino")\n');
  });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('emits an E007 diagnostic for a two-file cycle', () => {
    const out = evaluateFile(path.join(dir, 'a.lino'));
    assert.strictEqual(out.diagnostics.length, 1);
    const d = out.diagnostics[0];
    assert.strictEqual(d.code, 'E007');
    assert.match(d.message, /cycle/i);
    // The cycle diagnostic is anchored to the span of the (import ...) form
    // that closes the loop, in b.lino.
    assert.ok(d.span.file && d.span.file.endsWith('b.lino'), d.span.file);
    assert.strictEqual(d.span.line, 1);
    assert.strictEqual(d.span.col, 1);
  });

  it('emits an E007 diagnostic for a self-import', () => {
    const out = evaluateFile(path.join(dir, 'self.lino'));
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E007');
    assert.match(out.diagnostics[0].message, /cycle/i);
  });

  it('emits an E007 diagnostic for a three-file cycle and lists the chain', () => {
    const out = evaluateFile(path.join(dir, 'tri-a.lino'));
    assert.strictEqual(out.diagnostics.length, 1);
    const d = out.diagnostics[0];
    assert.strictEqual(d.code, 'E007');
    assert.match(d.message, /tri-a\.lino/);
    assert.match(d.message, /tri-b\.lino/);
    assert.match(d.message, /tri-c\.lino/);
  });
});

describe('(import "...") — diagnostics & semantics', () => {
  let dir;
  before(() => {
    dir = makeTmp();
    fs.writeFileSync(path.join(dir, 'broken.lino'), '(=: missing identity)\n');
    fs.writeFileSync(path.join(dir, 'host.lino'), '(import "broken.lino")\n');
  });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('forwards diagnostics from the imported file with the imported span', () => {
    const out = evaluateFile(path.join(dir, 'host.lino'));
    assert.strictEqual(out.diagnostics.length, 1);
    const d = out.diagnostics[0];
    assert.strictEqual(d.code, 'E001');
    assert.ok(d.span.file && d.span.file.endsWith('broken.lino'), d.span.file);
  });

  it('missing import target reports E007 with the importing file in the span', () => {
    const tmp = makeTmp();
    try {
      fs.writeFileSync(path.join(tmp, 'main.lino'), '(import "no-such.lino")\n');
      const out = evaluateFile(path.join(tmp, 'main.lino'));
      assert.strictEqual(out.diagnostics.length, 1);
      const d = out.diagnostics[0];
      assert.strictEqual(d.code, 'E007');
      assert.match(d.message, /no-such\.lino/);
      assert.ok(d.span.file && d.span.file.endsWith('main.lino'), d.span.file);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it('formats E007 diagnostics like other codes', () => {
    const tmp = makeTmp();
    try {
      const main = path.join(tmp, 'main.lino');
      fs.writeFileSync(main, '(import "no-such.lino")\n');
      const out = evaluateFile(main);
      const text = formatDiagnostic(out.diagnostics[0], fs.readFileSync(main, 'utf8'));
      const lines = text.split('\n');
      assert.match(lines[0], /:1:1: E007:/);
      assert.strictEqual(lines[1], '(import "no-such.lino")');
      assert.strictEqual(lines[2], '^');
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it('inline (import ...) without a file resolves relative to CWD', () => {
    const tmp = makeTmp();
    const original = process.cwd();
    try {
      fs.writeFileSync(path.join(tmp, 'lib.lino'), '(a: a is a)\n((a = a) has probability 1)\n');
      process.chdir(tmp);
      const out = evaluate('(import "lib.lino")\n(? (a = a))');
      assert.deepStrictEqual(out.diagnostics, []);
      assert.deepStrictEqual(out.results, [1]);
    } finally {
      process.chdir(original);
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });
});
