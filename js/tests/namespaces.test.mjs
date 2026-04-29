// Tests for `(namespace ...)` and qualified references (issue #34).
// Covers namespace declaration, alias imports, qualified lookup, alias
// collisions (E009), and shadowing diagnostics (E008). The Rust suite
// mirrors these in rust/tests/namespaces_tests.rs to keep the two
// implementations in lock-step.

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { evaluate, evaluateFile } from '../src/rml-links.mjs';

function makeTmp() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'rml-ns-'));
}

describe('(namespace ...) — declaration', () => {
  let dir;
  before(() => { dir = makeTmp(); });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('prefixes a definition with the active namespace', () => {
    fs.writeFileSync(path.join(dir, 'lib.lino'),
      '(namespace classical)\n(and: min)\n');
    fs.writeFileSync(path.join(dir, 'main.lino'),
      '(import "lib.lino")\n(? (classical.and 1 0))\n');
    const out = evaluateFile(path.join(dir, 'main.lino'));
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0]);
  });

  it('supports multiple definitions under the same namespace', () => {
    fs.writeFileSync(path.join(dir, 'lib2.lino'),
      '(namespace classical)\n(and: min)\n(or: max)\n');
    fs.writeFileSync(path.join(dir, 'main2.lino'),
      '(import "lib2.lino")\n(? (classical.and 1 0))\n(? (classical.or 1 0))\n');
    const out = evaluateFile(path.join(dir, 'main2.lino'));
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0, 1]);
  });

  it('rejects an empty or dotted namespace name with E009', () => {
    fs.writeFileSync(path.join(dir, 'bad.lino'), '(namespace foo.bar)\n');
    const out = evaluateFile(path.join(dir, 'bad.lino'));
    assert.strictEqual(out.diagnostics.length, 1);
    assert.strictEqual(out.diagnostics[0].code, 'E009');
  });
});

describe('(import "..." as <alias>) — aliased imports', () => {
  let dir;
  before(() => { dir = makeTmp(); });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('resolves qualified references through the alias', () => {
    fs.writeFileSync(path.join(dir, 'classical.lino'),
      '(namespace classical)\n(and: min)\n(or: max)\n');
    fs.writeFileSync(path.join(dir, 'main.lino'),
      '(import "classical.lino" as cl)\n(? (cl.and 1 0))\n(? (cl.or 1 0))\n');
    const out = evaluateFile(path.join(dir, 'main.lino'));
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0, 1]);
  });

  it('emits E009 for an alias collision', () => {
    fs.writeFileSync(path.join(dir, 'a.lino'),
      '(namespace foo)\n(x: max)\n');
    fs.writeFileSync(path.join(dir, 'b.lino'),
      '(namespace bar)\n(x: min)\n');
    fs.writeFileSync(path.join(dir, 'collide.lino'),
      '(import "a.lino" as a)\n(import "b.lino" as a)\n');
    const out = evaluateFile(path.join(dir, 'collide.lino'));
    const e009s = out.diagnostics.filter(d => d.code === 'E009');
    assert.strictEqual(e009s.length, 1);
    assert.match(e009s[0].message, /alias "a"/);
  });

  it('lets two distinct aliases point at different namespaces', () => {
    fs.writeFileSync(path.join(dir, 'la.lino'),
      '(namespace foo)\n(and: max)\n');
    fs.writeFileSync(path.join(dir, 'lb.lino'),
      '(namespace bar)\n(and: min)\n');
    fs.writeFileSync(path.join(dir, 'multi.lino'),
      '(import "la.lino" as a)\n(import "lb.lino" as b)\n' +
      '(? (a.and 0.2 0.5))\n(? (b.and 0.2 0.5))\n');
    const out = evaluateFile(path.join(dir, 'multi.lino'));
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0.5, 0.2]);
  });
});

describe('(namespace ...) — shadowing (E008)', () => {
  let dir;
  before(() => { dir = makeTmp(); });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  it('warns when a top-level op definition shadows an imported op', () => {
    fs.writeFileSync(path.join(dir, 'lib.lino'), '(myop: avg)\n');
    fs.writeFileSync(path.join(dir, 'main.lino'),
      '(import "lib.lino")\n(myop: max)\n(? (myop 0.2 0.4 0.8))\n');
    const out = evaluateFile(path.join(dir, 'main.lino'));
    const e008s = out.diagnostics.filter(d => d.code === 'E008');
    assert.strictEqual(e008s.length, 1);
    assert.match(e008s[0].message, /myop/);
    assert.match(e008s[0].message, /shadows/i);
    // The redefinition still takes effect.
    assert.deepStrictEqual(out.results, [0]);
  });

  it('warns when redefining a qualified name introduced via alias', () => {
    fs.writeFileSync(path.join(dir, 'lib2.lino'),
      '(namespace classical)\n(and: min)\n');
    fs.writeFileSync(path.join(dir, 'main2.lino'),
      '(import "lib2.lino" as cl)\n(cl.and: max)\n');
    const out = evaluateFile(path.join(dir, 'main2.lino'));
    const e008s = out.diagnostics.filter(d => d.code === 'E008');
    assert.strictEqual(e008s.length, 1);
    assert.match(e008s[0].message, /cl\.and/);
  });

  it('warns once even when the same imported name is rebound twice', () => {
    fs.writeFileSync(path.join(dir, 'lib3.lino'), '(myop: avg)\n');
    fs.writeFileSync(path.join(dir, 'main3.lino'),
      '(import "lib3.lino")\n(myop: max)\n(myop: min)\n');
    const out = evaluateFile(path.join(dir, 'main3.lino'));
    const e008s = out.diagnostics.filter(d => d.code === 'E008');
    assert.strictEqual(e008s.length, 1);
  });

  it('does not warn when an importing file defines a fresh name', () => {
    fs.writeFileSync(path.join(dir, 'lib4.lino'), '(myop: avg)\n');
    fs.writeFileSync(path.join(dir, 'main4.lino'),
      '(import "lib4.lino")\n(otherop: max)\n');
    const out = evaluateFile(path.join(dir, 'main4.lino'));
    assert.strictEqual(out.diagnostics.filter(d => d.code === 'E008').length, 0);
  });
});

describe('inline (namespace ...) without import', () => {
  it('an in-source namespace prefix is honoured', () => {
    const out = evaluate(
      '(namespace foo)\n(and: min)\n(? (foo.and 1 0))\n',
    );
    assert.deepStrictEqual(out.diagnostics, []);
    assert.deepStrictEqual(out.results, [0]);
  });
});
