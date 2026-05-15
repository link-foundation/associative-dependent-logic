#!/usr/bin/env node
// Universal CST converter round-trip demo (issue #138).
//
// Parses one small snippet of each host language (Rust / JS / Lean / Rocq)
// into the `.lino` CST, prints it back, and asserts byte equality. Run with:
//
//   node examples/cst-roundtrip-demo.mjs
//
// On success it prints one OK line per language plus a peek at the underlying
// `.lino` S-expression. Exits non-zero on any round-trip mismatch.

import {
  parseToCst,
  printFromCst,
  roundTrip,
  SUPPORTED_LANGUAGES,
} from '../js/src/cst-convert.mjs';
import { cstToLino, leaves } from '../js/src/cst.mjs';

const samples = {
  rust: `// add two i64s
pub fn add(a: i64, b: i64) -> i64 {
    a + b
}
`,
  js: `// fetch JSON
async function load(url) {
  const r = await fetch(url);
  return r.json();
}
`,
  lean: `-- identity
def id {α : Type} (x : α) : α := x
`,
  rocq: `(* identity *)
Definition id {A : Type} (x : A) : A := x.
`,
};

let allOk = true;

console.log('Universal CST round-trip demo');
console.log('-----------------------------');
console.log('Supported languages:', [...SUPPORTED_LANGUAGES].join(', '));
console.log('');

for (const lang of ['rust', 'js', 'lean', 'rocq']) {
  const src = samples[lang];
  const { ok, roundTripped } = roundTrip(src, lang);
  const tree = parseToCst(src, lang);
  const leafCount = Array.from(leaves(tree)).length;
  const sexpPreview = cstToLino(tree).slice(0, 60) + '…';
  console.log(`${lang.padEnd(5)} ${ok ? 'OK' : 'FAIL'}  (${src.length} bytes, ${leafCount} leaves)`);
  console.log(`        sexp: ${sexpPreview}`);
  if (!ok) {
    allOk = false;
    console.error('  expected:', JSON.stringify(src));
    console.error('  got     :', JSON.stringify(roundTripped));
  }
  // Also verify trivia leaves (comments / whitespace) survive intact.
  const triviaTexts = Array
    .from(leaves(tree))
    .filter(n => n.kind === 'trivia')
    .map(n => n.text.trim())
    .filter(Boolean);
  console.log(`        trivia: ${JSON.stringify(triviaTexts)}`);
  console.log('');
}

if (!allOk) {
  process.exitCode = 1;
  console.error('One or more round-trip checks failed.');
} else {
  console.log('All round-trip checks passed.');
}

// Also sanity-check that printFromCst is a pure function (idempotent).
for (const lang of ['rust', 'js', 'lean', 'rocq']) {
  const tree = parseToCst(samples[lang], lang);
  const a = printFromCst(tree, lang);
  const b = printFromCst(tree, lang);
  if (a !== b) {
    console.error(`printFromCst not idempotent for ${lang}`);
    process.exitCode = 1;
  }
}
