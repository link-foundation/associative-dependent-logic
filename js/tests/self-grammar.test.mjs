// Tests for `lib/self/grammar.lino` (issue #84).
//
// The grammar file is the self-bootstrap data layer: it describes LiNo as
// ordinary links. These tests keep that data parseable, importable, and tied
// to the current example corpus by comparing a minimal grammar-shaped parser
// against the host parser's AST presentation.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import {
  evaluateFile,
  parseLino,
  parseOne,
  tokenizeOne,
} from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const grammarPath = join(repoRoot, 'lib', 'self', 'grammar.lino');
const examplesDir = join(repoRoot, 'examples');

const REQUIRED_RULES = [
  'document',
  'source-for-evaluation',
  'links',
  'first-line',
  'line',
  'element',
  'any-link',
  'parenthesized-link',
  'id-link',
  'value-link',
  'indented-id-link',
  'reference',
  'simple-reference',
  'quoted-reference',
  'whitespace',
  'end-of-line',
  'host-parser-presentation',
];

function grammarForms() {
  const source = readFileSync(grammarPath, 'utf8');
  return parseLino(source).map(link => parseOne(tokenizeOne(link)));
}

function ruleNames(forms) {
  const names = new Set();
  for (const form of forms) {
    if (Array.isArray(form) && form[0] === 'rule' && typeof form[1] === 'string') {
      names.add(form[1]);
    }
  }
  return names;
}

function stripRmlComments(source) {
  return String(source)
    .replace(/^[ \t]*#.*$/gm, '')
    .replace(/(\)[ \t]+)#.*$/gm, '$1')
    .replace(/\n{3,}/g, '\n\n');
}

function parseSelfPresentation(source, rules) {
  assert.ok(rules.has('host-parser-presentation'));
  assert.ok(rules.has('parenthesized-link'));
  assert.ok(rules.has('simple-reference'));

  const input = stripRmlComments(source);
  let index = 0;

  const fail = (message) => {
    throw new Error(`${message} at byte ${index}`);
  };
  const skipWhitespace = () => {
    while (index < input.length && /\s/.test(input[index])) index++;
  };
  const parseAtom = () => {
    const start = index;
    while (
      index < input.length &&
      !/\s/.test(input[index]) &&
      input[index] !== '(' &&
      input[index] !== ')'
    ) {
      index++;
    }
    if (start === index) fail('expected reference');
    return input.slice(start, index);
  };
  const parseList = () => {
    if (input[index] !== '(') fail('expected "("');
    index++;
    const children = [];
    for (;;) {
      skipWhitespace();
      if (index >= input.length) fail('expected ")"');
      if (input[index] === ')') {
        index++;
        return children;
      }
      children.push(input[index] === '(' ? parseList() : parseAtom());
    }
  };

  const forms = [];
  for (;;) {
    skipWhitespace();
    if (index >= input.length) return forms;
    forms.push(parseList());
  }
}

function hostAst(source) {
  return parseLino(source).map(link => parseOne(tokenizeOne(link)));
}

describe('self LiNo grammar', () => {
  it('is importable as a standard library file', () => {
    const out = evaluateFile(grammarPath);
    assert.deepStrictEqual(out.diagnostics, []);
  });

  it('declares the required grammar rules as links', () => {
    const rules = ruleNames(grammarForms());
    for (const name of REQUIRED_RULES) {
      assert.ok(rules.has(name), `missing rule ${name}`);
    }
  });

  for (const file of readdirSync(examplesDir).filter(f => f.endsWith('.lino')).sort()) {
    it(`parses ${file} with the same AST as the host parser`, () => {
      const source = readFileSync(join(examplesDir, file), 'utf8');
      const rules = ruleNames(grammarForms());
      assert.deepStrictEqual(parseSelfPresentation(source, rules), hostAst(source));
    });
  }
});
