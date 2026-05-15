// Universal CST converter tests (issue #138).
//
// Verifies the lossless round-trip contract `print(parse(src)) === src` for
// every host-language converter, plus the CST serialisation helpers.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  list,
  token,
  trivia,
  printCst,
  cstToLino,
  linoToCst,
  cloneCst,
  DIALECTS,
  leaves,
} from '../src/cst.mjs';
import { parseRust, printRust } from '../src/cst-rust.mjs';
import { parseJs, printJs } from '../src/cst-js.mjs';
import { parseLean, printLean } from '../src/cst-lean.mjs';
import { parseRocq, printRocq } from '../src/cst-rocq.mjs';
import { parseToCst, printFromCst, roundTrip, SUPPORTED_LANGUAGES } from '../src/cst-convert.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '..', '..');

describe('cst data model', () => {
  it('printCst concatenates leaves in document order', () => {
    const tree = list('demo', [
      token('hello'),
      trivia(' '),
      token('world'),
    ]);
    assert.strictEqual(printCst(tree), 'hello world');
  });

  it('list nodes emit open/close delimiters when set', () => {
    const tree = list('demo', [token('a'), token('b')], { open: '(', close: ')' });
    assert.strictEqual(printCst(tree), '(ab)');
  });

  it('cstToLino + linoToCst is a round trip for tokens, trivia and lists', () => {
    const tree = list('lino-cst.rust.fn', [
      token('fn ', 'kw'),
      token('foo', 'ident'),
      token('(', 'punct'),
      token(')', 'punct'),
      trivia(' ', 'ws'),
      list('lino-cst.rust.block', [token('{'), token('}')]),
    ]);
    const serialised = cstToLino(tree);
    assert.match(serialised, /^\(lino-cst\.list lino-cst\.rust\.fn /);
    const parsed = linoToCst(serialised);
    assert.strictEqual(printCst(parsed), printCst(tree));
  });

  it('linoToCst preserves open/close delimiters', () => {
    const tree = list('frag', [token('x')], { open: '<', close: '>' });
    const sExpr = cstToLino(tree);
    const back = linoToCst(sExpr);
    assert.strictEqual(printCst(back), '<x>');
  });

  it('cloneCst returns a structural copy', () => {
    const tree = list('demo', [token('a'), list('inner', [token('b')])]);
    const clone = cloneCst(tree);
    assert.notStrictEqual(clone, tree);
    assert.notStrictEqual(clone.children[1], tree.children[1]);
    assert.deepStrictEqual(clone, tree);
  });

  it('leaves walks in document order', () => {
    const tree = list('demo', [token('a'), list('inner', [token('b'), token('c')]), token('d')]);
    const texts = Array.from(leaves(tree)).map(n => n.text);
    assert.deepStrictEqual(texts, ['a', 'b', 'c', 'd']);
  });

  it('exposes the four host dialects', () => {
    assert.strictEqual(DIALECTS.rust, 'lino-cst.rust');
    assert.strictEqual(DIALECTS.js, 'lino-cst.js');
    assert.strictEqual(DIALECTS.lean, 'lino-cst.lean');
    assert.strictEqual(DIALECTS.rocq, 'lino-cst.rocq');
  });
});

describe('Rust round-trip', () => {
  const samples = [
    '',
    'fn main() {}\n',
    '// hello\nfn f() -> i32 { 42 }\n',
    '/* multi\n  line */\nfn g() {}\n',
    'fn id<T>(x: T) -> T { x }\n',
    "let s = \"hello\\n\";\n",
    'let s = r#"raw"#;\n',
    "let c = 'a';\nlet lt: &'static str = \"x\";\n",
    'let n = 0xFF_FF;\nlet m = 0b1010;\nlet f = 3.14e10;\n',
    'pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n',
    'use std::collections::HashMap;\n',
    "let mut v = Vec::<i32>::new();\nv.push(1);\n",
    "macro_rules! foo { () => {}; }\n",
    'let r#match = 1;\n',
  ];
  for (const src of samples) {
    it(`round-trips: ${JSON.stringify(src.slice(0, 40))}`, () => {
      const r = roundTrip(src, 'rust');
      assert.strictEqual(r.roundTripped, src);
      assert.strictEqual(r.ok, true);
    });
  }
});

describe('JavaScript round-trip', () => {
  const samples = [
    '',
    'const x = 1;\n',
    '// hello\nconst y = 2;\n',
    '/* block */ const z = 3;\n',
    "const s = 'a\\'b';\n",
    'const t = `hello ${name}!`;\n',
    'const t2 = `nested ${`inner ${1+2}`} done`;\n',
    'const r = /foo\\/bar/g;\n',
    'const n = 0xff_ff;\nconst b = 0b1010n;\nconst f = 3.14e-2;\n',
    'function f(a, b) {\n  return a + b;\n}\n',
    'class C { method() { return 1; } }\n',
    'const obj = { a: 1, "b c": 2 };\n',
    '#!/usr/bin/env node\nconsole.log("hi");\n',
    'export default async function f() { await sleep(1); }\n',
    'let { a, b: c = 3 } = x;\n',
    'const re = /^a[/]b$/;\n',
  ];
  for (const src of samples) {
    it(`round-trips: ${JSON.stringify(src.slice(0, 40))}`, () => {
      const r = roundTrip(src, 'js');
      assert.strictEqual(r.roundTripped, src);
      assert.strictEqual(r.ok, true);
    });
  }
});

describe('Lean 4 round-trip', () => {
  const samples = [
    '',
    '-- comment\ndef f : Nat := 1\n',
    '/- block comment -/\ndef g : Nat := 2\n',
    '/-! module doc -/\n',
    '/-- decl doc -/\ndef h : Nat := 3\n',
    'def id {α : Type} (x : α) : α := x\n',
    '#check Nat.succ\n',
    'theorem t : 1 + 1 = 2 := rfl\n',
    'def s : String := "hello"\n',
    'def n : Nat := 0xff\n',
    'def m : Nat := 0b1010\n',
    'inductive List (α : Type u) where\n  | nil : List α\n  | cons : α → List α → List α\n',
    'namespace Foo\ndef x := 1\nend Foo\n',
    "def c : Char := 'a'\n",
  ];
  for (const src of samples) {
    it(`round-trips: ${JSON.stringify(src.slice(0, 40))}`, () => {
      const r = roundTrip(src, 'lean');
      assert.strictEqual(r.roundTripped, src);
      assert.strictEqual(r.ok, true);
    });
  }
});

describe('Rocq round-trip', () => {
  const samples = [
    '',
    '(* comment *)\nDefinition x := 1.\n',
    '(* nested (* inside *) *)\nDefinition y := 2.\n',
    'Definition id {A : Type} (x : A) : A := x.\n',
    'Theorem t : 1 + 1 = 2. Proof. reflexivity. Qed.\n',
    'Inductive list (A : Type) : Type :=\n  | nil\n  | cons (x : A) (xs : list A).\n',
    'Definition s := "hello, ""world""".\n',
    'Definition n := 0xff.\n',
    'Require Import Coq.Lists.List.\n',
  ];
  for (const src of samples) {
    it(`round-trips: ${JSON.stringify(src.slice(0, 40))}`, () => {
      const r = roundTrip(src, 'rocq');
      assert.strictEqual(r.roundTripped, src);
      assert.strictEqual(r.ok, true);
    });
  }
});

describe('cross-converter dispatch', () => {
  it('SUPPORTED_LANGUAGES lists the four host languages plus js alias', () => {
    assert.deepStrictEqual([...SUPPORTED_LANGUAGES].sort(), ['javascript', 'js', 'lean', 'rocq', 'rust']);
  });

  it('parseToCst / printFromCst dispatch by language name', () => {
    for (const lang of ['rust', 'js', 'lean', 'rocq']) {
      const sample = lang === 'rocq' ? 'Definition x := 1.\n' : 'x\n';
      const out = printFromCst(parseToCst(sample, lang), lang);
      assert.strictEqual(out, sample);
    }
  });

  it("'javascript' alias works the same as 'js'", () => {
    const src = 'const x = 1;\n';
    assert.strictEqual(printFromCst(parseToCst(src, 'javascript'), 'javascript'), src);
  });

  it('rejects unsupported languages', () => {
    assert.throws(() => parseToCst('x', 'python'), /unsupported language/);
    assert.throws(() => printFromCst(list('x', []), 'python'), /unsupported language/);
  });
});

describe('CST encoding preserves comments and whitespace explicitly', () => {
  it('Rust line and block comments survive as trivia leaves', () => {
    const src = '// leading\nfn f() {\n  /* mid */ 1\n}\n';
    const tree = parseRust(src);
    const leafList = Array.from(leaves(tree));
    assert.ok(leafList.some(n => n.kind === 'trivia' && n.text === '// leading'));
    assert.ok(leafList.some(n => n.kind === 'trivia' && n.text === '/* mid */'));
    assert.strictEqual(printRust(tree), src);
  });

  it('JS hashbang and comments survive', () => {
    const src = '#!/usr/bin/env node\n// hi\nlet x = 1;\n';
    const tree = parseJs(src);
    const leafList = Array.from(leaves(tree));
    assert.ok(leafList.some(n => n.kind === 'trivia' && n.text.startsWith('#!')));
    assert.ok(leafList.some(n => n.kind === 'trivia' && n.text === '// hi'));
    assert.strictEqual(printJs(tree), src);
  });

  it('Lean nested block comments survive', () => {
    const src = '/- outer /- inner -/ still outer -/\ndef x := 1\n';
    const tree = parseLean(src);
    const leafList = Array.from(leaves(tree));
    assert.ok(leafList.some(n => n.kind === 'trivia' && n.text.startsWith('/- outer')));
    assert.strictEqual(printLean(tree), src);
  });

  it('Rocq nested comments survive', () => {
    const src = '(* a (* b *) c *)\nDefinition x := 1.\n';
    const tree = parseRocq(src);
    const leafList = Array.from(leaves(tree));
    assert.ok(leafList.some(n => n.kind === 'trivia' && n.text === '(* a (* b *) c *)'));
    assert.strictEqual(printRocq(tree), src);
  });
});

describe('repository corpus round-trip', () => {
  // Verify that real, repository-tracked source files in each host language
  // round-trip byte-for-byte. This is the "bootstrap" criterion described in
  // docs/case-studies/issue-138/acceptance-tests.md.
  function readIfExists(p) {
    try { return fs.readFileSync(p, 'utf8'); } catch { return null; }
  }

  it('round-trips js/src/cst.mjs (this module is itself JS source)', () => {
    const src = readIfExists(path.join(repoRoot, 'js', 'src', 'cst.mjs'));
    if (src === null) return;
    assert.strictEqual(printJs(parseJs(src)), src);
  });

  it('round-trips rust/src/main.rs', () => {
    const src = readIfExists(path.join(repoRoot, 'rust', 'src', 'main.rs'));
    if (src === null) return;
    assert.strictEqual(printRust(parseRust(src)), src);
  });

  it('round-trips examples/lean-export-basic.lean', () => {
    const src = readIfExists(path.join(repoRoot, 'examples', 'lean-export-basic.lean'));
    if (src === null) return;
    assert.strictEqual(printLean(parseLean(src)), src);
  });

  it('round-trips examples/isabelle-typed-fragment.thy as Rocq-style block-comment source', () => {
    // Isabelle uses (* ... *) like Rocq for some comment styles; the
    // token-stream converter is permissive enough that this still round-trips.
    const src = readIfExists(path.join(repoRoot, 'examples', 'isabelle-typed-fragment.thy'));
    if (src === null) return;
    assert.strictEqual(printRocq(parseRocq(src)), src);
  });
});
