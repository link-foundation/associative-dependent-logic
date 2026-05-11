// Rust ↔ `.lino` CST converter (issue #138).
//
// Token-level lossless converter for Rust source. Produces a `lino-cst.rust.*`
// flat CST: every byte of input is encoded as one of the three CST node kinds.
// The round-trip is byte-faithful: `printRust(parseRust(src)) === src`.
//
// Scope: this converter operates at the token-stream level. It recognises:
//
//   - line comments (`// ...`, `/// ...`, `//! ...`)
//   - block comments (`/* ... */`, possibly nested)
//   - whitespace (spaces, tabs, CR, LF)
//   - shebang on the first line
//   - string literals (`"..."`), byte-strings (`b"..."`), raw strings (`r"..."`,
//     `r#"..."#`, etc.), char literals (`'...'`), byte-char literals.
//   - numeric literals (decimal, hex, oct, bin, with suffixes).
//   - identifiers (including raw identifiers `r#foo` and Unicode XID).
//   - punctuation (every other byte is an operator/punctuator token).
//
// The output CST is a single `lino-cst.rust.source_file` list whose children
// alternate between tokens (significant lexemes) and trivia (whitespace,
// comments). This matches the rust-analyzer rowan layout: trivia is preserved
// at the leaf level, structure is preserved at the leaf level too.
//
// A full semantic CST (one `lino-cst.rust.*` tag per `ra_ap_syntax`
// `SyntaxKind`) is built on top of this token stream — that work is
// orthogonal to round-trip fidelity and is intentionally deferred. The
// token-level converter already satisfies issue #138's requirement that "every
// variable and other name, and even whitespace if needed" survive a round
// trip.

import { list, token, trivia, DIALECTS } from './cst.mjs';

const RUST = DIALECTS.rust;

/**
 * Parse Rust source into a `lino-cst.rust.*` CST.
 *
 * The returned tree is a single `source_file` list. Every byte of `src` is
 * preserved verbatim in the resulting tree; `printRust(parseRust(src)) ===
 * src` for any well-tokenised input.
 *
 * @param {string} src Rust source.
 * @returns {CstNode} CST root.
 */
export function parseRust(src) {
  const children = tokeniseRust(String(src));
  return list(`${RUST}.source_file`, children);
}

/**
 * Print a Rust CST back to source. Inverse of `parseRust`.
 *
 * @param {CstNode} node
 * @returns {string}
 */
export function printRust(node) {
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

// ---------- Lexer ----------

const ID_START = /[A-Za-z_]/;
const ID_CONT = /[A-Za-z0-9_]/;
const DIGIT = /[0-9]/;
const HEX = /[0-9A-Fa-f]/;

function tokeniseRust(src) {
  const out = [];
  let i = 0;

  if (src.startsWith('#!') && (src.length < 3 || src[2] !== '[')) {
    // Shebang line — preserved as trivia at the start.
    let j = src.indexOf('\n');
    if (j === -1) j = src.length;
    out.push(trivia(src.substring(0, j), `${RUST}.shebang`));
    i = j;
  }

  while (i < src.length) {
    const c = src[i];

    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      let j = i;
      while (j < src.length && (src[j] === ' ' || src[j] === '\t' || src[j] === '\r' || src[j] === '\n')) j++;
      out.push(trivia(src.substring(i, j), `${RUST}.whitespace`));
      i = j;
      continue;
    }

    if (c === '/' && src[i + 1] === '/') {
      let j = i + 2;
      while (j < src.length && src[j] !== '\n') j++;
      out.push(trivia(src.substring(i, j), `${RUST}.comment.line`));
      i = j;
      continue;
    }

    if (c === '/' && src[i + 1] === '*') {
      const j = scanBlockComment(src, i);
      out.push(trivia(src.substring(i, j), `${RUST}.comment.block`));
      i = j;
      continue;
    }

    if (c === '"') {
      const j = scanString(src, i + 1, '"');
      out.push(token(src.substring(i, j), `${RUST}.string_literal`));
      i = j;
      continue;
    }

    if ((c === 'b' || c === 'r') && (src[i + 1] === '"' || (c === 'r' && src[i + 1] === '#') || (c === 'b' && src[i + 1] === 'r' && (src[i + 2] === '"' || src[i + 2] === '#')))) {
      const j = scanRawOrPrefixedString(src, i);
      if (j > i) {
        out.push(token(src.substring(i, j), `${RUST}.string_literal`));
        i = j;
        continue;
      }
    }

    if (c === "'" ) {
      // char literal or lifetime
      const lifetimeEnd = scanLifetime(src, i);
      if (lifetimeEnd > i + 1) {
        out.push(token(src.substring(i, lifetimeEnd), `${RUST}.lifetime`));
        i = lifetimeEnd;
        continue;
      }
      const j = scanString(src, i + 1, "'");
      out.push(token(src.substring(i, j), `${RUST}.char_literal`));
      i = j;
      continue;
    }

    if (c === 'b' && src[i + 1] === "'") {
      const j = scanString(src, i + 2, "'");
      out.push(token(src.substring(i, j), `${RUST}.byte_literal`));
      i = j;
      continue;
    }

    if (DIGIT.test(c)) {
      const j = scanNumber(src, i);
      out.push(token(src.substring(i, j), `${RUST}.numeric_literal`));
      i = j;
      continue;
    }

    if (c === 'r' && src[i + 1] === '#' && ID_START.test(src[i + 2] || '')) {
      let j = i + 2;
      while (j < src.length && ID_CONT.test(src[j])) j++;
      out.push(token(src.substring(i, j), `${RUST}.raw_ident`));
      i = j;
      continue;
    }

    if (ID_START.test(c) || isUnicodeIdStart(c)) {
      let j = i + 1;
      while (j < src.length && (ID_CONT.test(src[j]) || isUnicodeIdContinue(src[j]))) j++;
      out.push(token(src.substring(i, j), `${RUST}.ident`));
      i = j;
      continue;
    }

    // Punctuation: emit one byte. Operators are tokens too; the printer just
    // concatenates them, so we do not need to bundle multi-char operators.
    out.push(token(c, `${RUST}.punct`));
    i += 1;
  }

  return out;
}

function scanBlockComment(src, i) {
  let j = i + 2;
  let depth = 1;
  while (j < src.length && depth > 0) {
    if (src[j] === '/' && src[j + 1] === '*') {
      depth++;
      j += 2;
    } else if (src[j] === '*' && src[j + 1] === '/') {
      depth--;
      j += 2;
    } else {
      j++;
    }
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

function scanRawOrPrefixedString(src, i) {
  let j = i;
  if (src[j] === 'b') j++;
  if (src[j] === 'r') {
    j++;
    let hashes = 0;
    while (src[j] === '#') { hashes++; j++; }
    if (src[j] !== '"') return i;
    j++;
    const terminator = '"' + '#'.repeat(hashes);
    const end = src.indexOf(terminator, j);
    if (end === -1) return src.length;
    return end + terminator.length;
  }
  if (src[j] === '"') {
    return scanString(src, j + 1, '"');
  }
  return i;
}

function scanLifetime(src, i) {
  let j = i + 1;
  if (j < src.length && (ID_START.test(src[j]) || isUnicodeIdStart(src[j]))) {
    j++;
    while (j < src.length && (ID_CONT.test(src[j]) || isUnicodeIdContinue(src[j]))) j++;
    // If next char is a quote, this is a char literal, not a lifetime.
    if (src[j] === "'") return i;
    return j;
  }
  return i;
}

function scanNumber(src, i) {
  let j = i;
  if (src[j] === '0' && (src[j + 1] === 'x' || src[j + 1] === 'X')) {
    j += 2;
    while (j < src.length && (HEX.test(src[j]) || src[j] === '_')) j++;
  } else if (src[j] === '0' && (src[j + 1] === 'o' || src[j + 1] === 'O')) {
    j += 2;
    while (j < src.length && (/[0-7_]/.test(src[j]))) j++;
  } else if (src[j] === '0' && (src[j + 1] === 'b' || src[j + 1] === 'B')) {
    j += 2;
    while (j < src.length && (/[01_]/.test(src[j]))) j++;
  } else {
    while (j < src.length && (DIGIT.test(src[j]) || src[j] === '_')) j++;
    if (src[j] === '.' && DIGIT.test(src[j + 1])) {
      j++;
      while (j < src.length && (DIGIT.test(src[j]) || src[j] === '_')) j++;
    }
    if (src[j] === 'e' || src[j] === 'E') {
      j++;
      if (src[j] === '+' || src[j] === '-') j++;
      while (j < src.length && (DIGIT.test(src[j]) || src[j] === '_')) j++;
    }
  }
  // Type suffix: i32, u64, f64, isize, usize, etc.
  if (j < src.length && ID_START.test(src[j])) {
    while (j < src.length && ID_CONT.test(src[j])) j++;
  }
  return j;
}

function isUnicodeIdStart(c) {
  if (!c) return false;
  const code = c.codePointAt(0);
  return code > 0x7f;
}

function isUnicodeIdContinue(c) {
  return isUnicodeIdStart(c);
}
