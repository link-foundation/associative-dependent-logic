// Universal lossless CST infrastructure for issue #138.
//
// This module implements the `.lino` concrete syntax tree (CST) model
// described in `docs/case-studies/issue-138/cst-model.md`. It is the shared
// core used by every host-language converter (Rust, JavaScript, Lean 4, Rocq):
// each converter produces a tree of `CstNode` values, which can be printed
// back to bytes via `printCst()`, giving a byte-faithful round-trip.
//
// Three node kinds, exactly as the model specifies:
//
// - `list`   — a list node tagged with a dialect-specific symbol (e.g.
//              `lino-cst.rust.fn`); children are emitted in order with the
//              `open` and `close` delimiters if present.
// - `token`  — a leaf carrying a single non-trivia lexeme; its `text` field is
//              the original source bytes.
// - `trivia` — a leaf carrying whitespace or a comment; `text` is the original
//              source bytes. Trivia attaches to the *following* non-trivia
//              leaf by convention (Roslyn/libsyntax style).
//
// The tree is intentionally minimal: each converter owns the choice of tags
// and the placement of trivia. The printer is content-agnostic — it simply
// concatenates `text` in document order.

/** @typedef {Object} CstListNode
 *  @property {'list'} kind
 *  @property {string|null} tag
 *  @property {string|null} open
 *  @property {string|null} close
 *  @property {CstNode[]} children
 */
/** @typedef {Object} CstTokenNode
 *  @property {'token'} kind
 *  @property {string|null} tag
 *  @property {string} text
 */
/** @typedef {Object} CstTriviaNode
 *  @property {'trivia'} kind
 *  @property {string|null} tag
 *  @property {string} text
 */
/** @typedef {CstListNode|CstTokenNode|CstTriviaNode} CstNode */
/** @typedef {Object} CstListOptions
 *  @property {string|null} [open]
 *  @property {string|null} [close]
 */

/**
 * Construct a `list` CST node.
 *
 * @param {string|null} tag dialect-specific symbol, e.g. `lino-cst.rust.fn`.
 * @param {CstNode[]} children child nodes in document order.
 * @param {CstListOptions} [opts]
 *   optional `open`/`close` delimiter strings to emit around the children.
 * @returns {CstListNode}
 */
export function list(tag, children = [], opts = {}) {
  return {
    kind: 'list',
    tag: tag === undefined ? null : tag,
    open: opts.open ?? null,
    close: opts.close ?? null,
    children: children.slice(),
  };
}

/**
 * Construct a `token` CST leaf.
 *
 * @param {string} text original source bytes for the lexeme.
 * @param {string|null} [tag] optional dialect-specific symbol.
 * @returns {CstTokenNode}
 */
export function token(text, tag = null) {
  return { kind: 'token', tag, text: String(text) };
}

/**
 * Construct a `trivia` CST leaf (whitespace or comment).
 *
 * @param {string} text original source bytes.
 * @param {string|null} [tag] optional categorisation, e.g. `comment.line`.
 * @returns {CstTriviaNode}
 */
export function trivia(text, tag = null) {
  return { kind: 'trivia', tag, text: String(text) };
}

/**
 * Print a CST node back to its original byte-for-byte source representation.
 *
 * The walk is the five-line algorithm from the model document:
 *   - for `token` and `trivia` nodes, emit `node.text`.
 *   - for `list` nodes, emit `open`, then children in order, then `close`.
 *
 * @param {CstNode} node tree to print.
 * @returns {string} reconstructed source.
 */
export function printCst(node) {
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
    return;
  }
  throw new TypeError(`Unknown CST node kind: ${node && node.kind}`);
}

/**
 * Iterate every leaf (`token` or `trivia`) of a CST in document order.
 *
 * @param {CstNode} node tree to walk.
 * @returns {Iterable<CstTokenNode|CstTriviaNode>}
 */
export function* leaves(node) {
  if (!node) return;
  if (node.kind === 'token' || node.kind === 'trivia') {
    yield node;
    return;
  }
  if (node.kind === 'list') {
    for (const child of node.children) yield* leaves(child);
  }
}

/**
 * Serialise a CST node into a `.lino` S-expression suitable for embedding in
 * `.lino` files or for round-trip testing. The serialisation is the literal
 * shape documented in `docs/case-studies/issue-138/cst-model.md`:
 *
 *   `(lino-cst.list <tag> <child> <child> ...)`
 *   `(lino-cst.token "<text>")`
 *   `(lino-cst.trivia "<text>")`
 *
 * String contents are escaped using JSON rules so that the result is a valid
 * LiNo expression. `parseCstLino` is the exact inverse.
 *
 * @param {CstNode} node tree to serialise.
 * @returns {string}
 */
export function cstToLino(node) {
  if (!node) return '';
  if (node.kind === 'token') return `(lino-cst.token ${escapeText(node.text)})`;
  if (node.kind === 'trivia') return `(lino-cst.trivia ${escapeText(node.text)})`;
  if (node.kind === 'list') {
    const parts = ['lino-cst.list'];
    if (node.tag) parts.push(node.tag);
    if (node.open !== null && node.open !== undefined) parts.push(`(open ${escapeText(node.open)})`);
    if (node.close !== null && node.close !== undefined) parts.push(`(close ${escapeText(node.close)})`);
    for (const child of node.children) parts.push(cstToLino(child));
    return `(${parts.join(' ')})`;
  }
  throw new TypeError(`Unknown CST node kind: ${node && node.kind}`);
}

function escapeText(text) {
  return JSON.stringify(String(text));
}

function unescapeText(literal) {
  return JSON.parse(literal);
}

/**
 * Parse the `.lino` S-expression produced by `cstToLino()` back into a
 * `CstNode` tree. Inverse of `cstToLino`.
 *
 * @param {string} src `.lino` S-expression.
 * @returns {CstNode}
 */
export function linoToCst(src) {
  const tokens = tokeniseLinoCst(String(src));
  let i = 0;
  function peek() { return tokens[i]; }
  function eat() { return tokens[i++]; }
  function expect(t) {
    const got = eat();
    if (got !== t) throw new SyntaxError(`expected ${JSON.stringify(t)}, got ${JSON.stringify(got)}`);
  }
  function parseNode() {
    expect('(');
    const head = eat();
    if (head === 'lino-cst.token') {
      const lit = eat();
      expect(')');
      return token(unescapeText(lit));
    }
    if (head === 'lino-cst.trivia') {
      const lit = eat();
      expect(')');
      return trivia(unescapeText(lit));
    }
    if (head === 'lino-cst.list') {
      let tag = null;
      let open = null;
      let close = null;
      const children = [];
      while (peek() !== ')') {
        if (peek() === '(') {
          const lookahead = tokens[i + 1];
          if (lookahead === 'open') {
            eat(); eat();
            open = unescapeText(eat());
            expect(')');
            continue;
          }
          if (lookahead === 'close') {
            eat(); eat();
            close = unescapeText(eat());
            expect(')');
            continue;
          }
          children.push(parseNode());
        } else {
          if (tag !== null) throw new SyntaxError(`unexpected token ${peek()}`);
          tag = eat();
        }
      }
      expect(')');
      return list(tag, children, { open, close });
    }
    throw new SyntaxError(`unknown CST tag: ${head}`);
  }
  return parseNode();
}

function tokeniseLinoCst(src) {
  const out = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i];
    if (c === '(' || c === ')') {
      out.push(c);
      i++;
      continue;
    }
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') {
      i++;
      continue;
    }
    if (c === '"') {
      let j = i + 1;
      while (j < src.length) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src[j] === '"') break;
        j++;
      }
      if (j >= src.length) throw new SyntaxError('unterminated string literal');
      out.push(src.substring(i, j + 1));
      i = j + 1;
      continue;
    }
    let j = i;
    while (j < src.length && !' \t\n\r()'.includes(src[j])) j++;
    out.push(src.substring(i, j));
    i = j;
  }
  return out;
}

/**
 * The four host-language dialect tag prefixes plus the shared dialect.
 * Exported so that external tooling can refer to them symbolically.
 */
export const DIALECTS = Object.freeze({
  rust: 'lino-cst.rust',
  js: 'lino-cst.js',
  lean: 'lino-cst.lean',
  rocq: 'lino-cst.rocq',
  shared: 'lino-cst.shared',
});

/**
 * Compute and return a structural copy of a CST node. Used by translators that
 * need to mutate without altering the input.
 *
 * @param {CstNode} node
 * @returns {CstNode}
 */
export function cloneCst(node) {
  if (!node) return node;
  if (node.kind === 'token' || node.kind === 'trivia') {
    return { ...node };
  }
  return {
    kind: 'list',
    tag: node.tag,
    open: node.open,
    close: node.close,
    children: node.children.map(cloneCst),
  };
}
