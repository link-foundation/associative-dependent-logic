// Lean 4 ↔ `.lino` CST converter (issue #138).
//
// Token-level lossless converter for Lean 4 source. Produces a
// `lino-cst.lean.*` flat CST. Round-trip is byte-faithful:
// `printLean(parseLean(src)) === src`.
//
// Lean 4 tokens we recognise at this layer:
//
//   - line comments (`-- ...`)
//   - block comments (`/- ... -/`, with nesting)
//   - documentation comments (`/-! ... -/`, `/-- ... -/`)
//   - whitespace
//   - string literals (`"..."`, `r"..."` raw)
//   - char literals (`'c'`)
//   - numeric literals (decimal, hex, oct, bin)
//   - identifiers — Lean 4 allows arbitrary Unicode letters (including
//     `α`, `β`, `→`, dotted hierarchical names like `Nat.succ`).
//   - punctuation
//
// The Unicode handling is generous: any non-ASCII code point that is not
// whitespace or a recognised delimiter is treated as identifier content. This
// is sufficient for token-level round-trip and matches the Lean 4 tokenizer's
// behaviour for common cases (see Lean.Parser).

import { list, token, trivia, DIALECTS } from './cst.mjs';

const LEAN = DIALECTS.lean;

/**
 * Parse Lean 4 source into a `lino-cst.lean.*` CST.
 *
 * @param {string} src Lean source.
 * @returns {CstNode}
 */
export function parseLean(src) {
  return list(`${LEAN}.module`, tokeniseLean(String(src)));
}

/**
 * Print a Lean CST back to source.
 * @param {CstNode} node
 * @returns {string}
 */
export function printLean(node) {
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
const ASCII_PUNCT = new Set(Array.from('()[]{},.;:`@#$%^&*+-=/\\<>!?|~'));

function tokeniseLean(src) {
  const out = [];
  let i = 0;

  while (i < src.length) {
    const c = src[i];

    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      let j = i;
      while (j < src.length && (src[j] === ' ' || src[j] === '\t' || src[j] === '\r' || src[j] === '\n')) j++;
      out.push(trivia(src.substring(i, j), `${LEAN}.whitespace`));
      i = j;
      continue;
    }

    if (c === '-' && src[i + 1] === '-') {
      let j = i + 2;
      while (j < src.length && src[j] !== '\n') j++;
      out.push(trivia(src.substring(i, j), `${LEAN}.comment.line`));
      i = j;
      continue;
    }

    if (c === '/' && src[i + 1] === '-') {
      const j = scanBlockComment(src, i);
      const tag = src[i + 2] === '-' ? `${LEAN}.doc.block` : `${LEAN}.comment.block`;
      out.push(trivia(src.substring(i, j), tag));
      i = j;
      continue;
    }

    if (c === '"') {
      const j = scanString(src, i + 1, '"');
      out.push(token(src.substring(i, j), `${LEAN}.string_literal`));
      i = j;
      continue;
    }

    if (c === 'r' && src[i + 1] === '"') {
      const j = scanString(src, i + 2, '"');
      out.push(token(src.substring(i, j), `${LEAN}.raw_string_literal`));
      i = j;
      continue;
    }

    if (c === "'") {
      // Char literal: `'x'` or `'\n'`, etc.
      let j = i + 1;
      if (src[j] === '\\') j += 2; else j += 1;
      if (src[j] === "'") {
        j++;
        out.push(token(src.substring(i, j), `${LEAN}.char_literal`));
        i = j;
        continue;
      }
      // Not a valid char literal; fall through to punctuation.
    }

    if (DIGIT.test(c)) {
      const j = scanNumber(src, i);
      out.push(token(src.substring(i, j), `${LEAN}.numeric_literal`));
      i = j;
      continue;
    }

    if (isIdentStart(c)) {
      let j = i + 1;
      while (j < src.length && isIdentContinue(src[j])) j++;
      // Allow dotted hierarchical name: `Nat.succ`, `List.foldr`.
      while (src[j] === '.' && isIdentStart(src[j + 1])) {
        j++;
        while (j < src.length && isIdentContinue(src[j])) j++;
      }
      out.push(token(src.substring(i, j), `${LEAN}.ident`));
      i = j;
      continue;
    }

    // Multi-byte UTF-8 codepoints other than identifier-class chars:
    // emit one full codepoint as a punctuation token to keep round-trip
    // byte-exact.
    const cp = src.codePointAt(i);
    const len = cp > 0xffff ? 2 : 1;
    out.push(token(src.substring(i, i + len), `${LEAN}.punct`));
    i += len;
  }

  return out;
}

function scanBlockComment(src, i) {
  let j = i + 2;
  let depth = 1;
  while (j < src.length && depth > 0) {
    if (src[j] === '/' && src[j + 1] === '-') { depth++; j += 2; }
    else if (src[j] === '-' && src[j + 1] === '/') { depth--; j += 2; }
    else j++;
  }
  return j;
}

function scanString(src, j, quote) {
  while (j < src.length) {
    const c = src[j];
    if (c === '\\') { j += 2; continue; }
    if (c === quote) return j + 1;
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
  if (src[j] === '.' && DIGIT.test(src[j + 1])) {
    j++;
    while (j < src.length && DIGIT.test(src[j])) j++;
    if (src[j] === 'e' || src[j] === 'E') {
      j++;
      if (src[j] === '+' || src[j] === '-') j++;
      while (j < src.length && DIGIT.test(src[j])) j++;
    }
  }
  return j;
}

function isIdentStart(c) {
  if (!c) return false;
  if (/[A-Za-z_]/.test(c)) return true;
  const cp = c.codePointAt(0);
  // Treat all non-ASCII letters and Greek-letter-class code points as identifier-start.
  if (cp > 0x7f) {
    return !isLeanPunctChar(c) && !isLeanWhitespaceChar(c);
  }
  return false;
}

function isIdentContinue(c) {
  if (!c) return false;
  if (/[A-Za-z0-9_'!?]/.test(c)) return true;
  const cp = c.codePointAt(0);
  if (cp > 0x7f) {
    return !isLeanPunctChar(c) && !isLeanWhitespaceChar(c);
  }
  return false;
}

function isLeanPunctChar(c) {
  // Reserve Lean-specific Unicode operators as punctuation: arrows, lambda, etc.
  return '→←↦⟨⟩⟦⟧«»‹›'.includes(c);
}

function isLeanWhitespaceChar(c) {
  return c === ' ' || c === ' ' || c === ' ';
}
