// Universal CST converter dispatch (issue #138).
//
// Single entry point that turns host-language source into a `.lino` CST and
// back. Provides:
//
//   - `parseToCst(src, lang)` — host source → CST.
//   - `printFromCst(node, lang)` — CST → host source.
//   - `roundTrip(src, lang)` — convenience helper that asserts byte fidelity.
//
// Supported `lang` values: `'rust'`, `'js'`, `'lean'`, `'rocq'`.

import { parseRust, printRust } from './cst-rust.mjs';
import { parseJs, printJs } from './cst-js.mjs';
import { parseLean, printLean } from './cst-lean.mjs';
import { parseRocq, printRocq } from './cst-rocq.mjs';

const PARSERS = {
  rust: parseRust,
  js: parseJs,
  javascript: parseJs,
  lean: parseLean,
  rocq: parseRocq,
};

const PRINTERS = {
  rust: printRust,
  js: printJs,
  javascript: printJs,
  lean: printLean,
  rocq: printRocq,
};

/**
 * The four host languages plus their aliases.
 * @type {ReadonlyArray<'rust' | 'js' | 'javascript' | 'lean' | 'rocq'>}
 */
export const SUPPORTED_LANGUAGES = Object.freeze(['rust', 'js', 'javascript', 'lean', 'rocq']);

/**
 * Parse host-language source into a `.lino` CST.
 *
 * @param {string} src host-language source.
 * @param {'rust'|'js'|'javascript'|'lean'|'rocq'} lang
 * @returns {import('./cst.mjs').CstNode}
 */
export function parseToCst(src, lang) {
  const parse = PARSERS[lang];
  if (!parse) throw new Error(`unsupported language for parseToCst: ${lang}`);
  return parse(src);
}

/**
 * Print a CST node back to host-language source.
 *
 * @param {import('./cst.mjs').CstNode} node CST tree.
 * @param {'rust'|'js'|'javascript'|'lean'|'rocq'} lang
 * @returns {string}
 */
export function printFromCst(node, lang) {
  const print = PRINTERS[lang];
  if (!print) throw new Error(`unsupported language for printFromCst: ${lang}`);
  return print(node);
}

/**
 * Verify that `printFromCst(parseToCst(src, lang), lang) === src`.
 *
 * @param {string} src host-language source.
 * @param {'rust'|'js'|'javascript'|'lean'|'rocq'} lang
 * @returns {{ ok: boolean, source: string, roundTripped: string }}
 */
export function roundTrip(src, lang) {
  const cst = parseToCst(src, lang);
  const roundTripped = printFromCst(cst, lang);
  return { ok: roundTripped === src, source: src, roundTripped };
}
