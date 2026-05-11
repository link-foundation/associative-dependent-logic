// Rocq ↔ `.lino` CST converter (issue #138).
//
// Token-level lossless converter for Rocq (formerly Coq) source. Produces a
// `lino-cst.rocq.*` flat CST. Round-trip is byte-faithful:
// `printRocq(parseRocq(src)) === src`.
//
// Tokens we recognise at this layer (matches the Rocq lexer, `clexer.ml`):
//
//   - block comments `(* ... *)` (Rocq comments nest)
//   - whitespace
//   - string literals `"..."` (escaped via `""` doubling, the Rocq convention)
//   - numeric literals (decimal, hex `0x...`, oct, bin)
//   - identifiers
//   - punctuation (vernac uses `.` as a statement terminator; the lexer just
//     emits it as a punctuation token here.)
//
// Rocq does NOT use `//` for line comments — every comment is block-form.

import { list, token, trivia, DIALECTS } from './cst.mjs';

const ROCQ = DIALECTS.rocq;

/**
 * Parse Rocq source into a `lino-cst.rocq.*` CST.
 * @param {string} src
 * @returns {CstNode}
 */
export function parseRocq(src) {
  return list(`${ROCQ}.document`, tokeniseRocq(String(src)));
}

/**
 * Print a Rocq CST back to source.
 * @param {CstNode} node
 * @returns {string}
 */
export function printRocq(node) {
  const out = [];
  emit(node, out);
  return out.join('');
}

function emit(node, out) {
  if (!node) return;
  if (node.kind === 'token' || node.kind === 'trivia') {
    out.push(node.text);
    return;
  }
  if (node.kind === 'list') {
    if (node.open) out.push(node.open);
    for (const child of node.children) emit(child, out);
    if (node.close) out.push(node.close);
  }
}

const DIGIT = /[0-9]/;
const HEX = /[0-9A-Fa-f]/;

function tokeniseRocq(src) {
  const out = [];
  let i = 0;

  while (i < src.length) {
    const c = src[i];

    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      let j = i;
      while (j < src.length && (src[j] === ' ' || src[j] === '\t' || src[j] === '\r' || src[j] === '\n')) j++;
      out.push(trivia(src.substring(i, j), `${ROCQ}.whitespace`));
      i = j;
      continue;
    }

    if (c === '(' && src[i + 1] === '*') {
      const j = scanBlockComment(src, i);
      out.push(trivia(src.substring(i, j), `${ROCQ}.comment`));
      i = j;
      continue;
    }

    if (c === '"') {
      const j = scanRocqString(src, i + 1);
      out.push(token(src.substring(i, j), `${ROCQ}.string_literal`));
      i = j;
      continue;
    }

    if (DIGIT.test(c)) {
      const j = scanNumber(src, i);
      out.push(token(src.substring(i, j), `${ROCQ}.numeric_literal`));
      i = j;
      continue;
    }

    if (isIdentStart(c)) {
      let j = i + 1;
      while (j < src.length && isIdentContinue(src[j])) j++;
      out.push(token(src.substring(i, j), `${ROCQ}.ident`));
      i = j;
      continue;
    }

    const cp = src.codePointAt(i);
    const len = cp > 0xffff ? 2 : 1;
    out.push(token(src.substring(i, i + len), `${ROCQ}.punct`));
    i += len;
  }

  return out;
}

function scanBlockComment(src, i) {
  let j = i + 2;
  let depth = 1;
  while (j < src.length && depth > 0) {
    if (src[j] === '(' && src[j + 1] === '*') { depth++; j += 2; }
    else if (src[j] === '*' && src[j + 1] === ')') { depth--; j += 2; }
    else j++;
  }
  return j;
}

function scanRocqString(src, j) {
  while (j < src.length) {
    if (src[j] === '"') {
      if (src[j + 1] === '"') { j += 2; continue; } // escaped doubled quote
      return j + 1;
    }
    j++;
  }
  return j;
}

function scanNumber(src, i) {
  let j = i;
  if (src[j] === '0' && (src[j + 1] === 'x' || src[j + 1] === 'X')) {
    j += 2;
    while (j < src.length && HEX.test(src[j])) j++;
    return j;
  }
  if (src[j] === '0' && (src[j + 1] === 'o' || src[j + 1] === 'O')) {
    j += 2;
    while (j < src.length && /[0-7]/.test(src[j])) j++;
    return j;
  }
  if (src[j] === '0' && (src[j + 1] === 'b' || src[j + 1] === 'B')) {
    j += 2;
    while (j < src.length && /[01]/.test(src[j])) j++;
    return j;
  }
  while (j < src.length && DIGIT.test(src[j])) j++;
  return j;
}

function isIdentStart(c) {
  if (!c) return false;
  if (/[A-Za-z_]/.test(c)) return true;
  const cp = c.codePointAt(0);
  if (cp > 0x7f) {
    return !isRocqPunctChar(c);
  }
  return false;
}

function isIdentContinue(c) {
  if (!c) return false;
  if (/[A-Za-z0-9_']/.test(c)) return true;
  const cp = c.codePointAt(0);
  if (cp > 0x7f) {
    return !isRocqPunctChar(c);
  }
  return false;
}

function isRocqPunctChar(c) {
  return '→←⟨⟩∀∃∧∨¬'.includes(c);
}
