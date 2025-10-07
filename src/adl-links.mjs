#!/usr/bin/env node
// ADL — minimal associative-dependent logic over LiNo (Links Notation)
// - Uses official LiNo parser to parse links
// - Terms are defined via (x: x is x)
// - Probabilities are assigned ONLY via: ((<expr>) has probability <p>)
// - Redefinable ops: (=: ...), (!=: not =), (and: avg|min|max|prod|ps), (or: ...), (not: ...)
// - Query: (? <expr>)

import fs from 'node:fs';
import { Parser } from '@linksplatform/protocols-lino';

// ---------- helpers: canonical keys & tokenization of a single link string ----------
function tokenizeOne(s) {
  // s is a single-link string like "( (a = a) has probability 1 )"
  // Strip inline comments (everything after #) but balance parens
  const commentIdx = s.indexOf('#');
  if (commentIdx !== -1) {
    s = s.substring(0, commentIdx);
    // Count unmatched opening parens and add closing parens to balance
    let depth = 0;
    for (let i = 0; i < s.length; i++) {
      if (s[i] === '(') depth++;
      else if (s[i] === ')') depth--;
    }
    // Add missing closing parens
    while (depth > 0) {
      s += ')';
      depth--;
    }
  }

  const out = [];
  let i = 0;
  const isWS = c => /\s/.test(c);
  while (i < s.length) {
    const c = s[i];
    if (isWS(c)) { i++; continue; }
    if (c === '(' || c === ')') { out.push(c); i++; continue; }
    let j = i;
    while (j < s.length && !isWS(s[j]) && s[j] !== '(' && s[j] !== ')') j++;
    out.push(s.slice(i, j));
    i = j;
  }
  return out;
}
function parseOne(tokens) {
  let i = 0;
  function read() {
    if (tokens[i] !== '(') throw new Error('expected "("');
    i++;
    const arr = [];
    while (i < tokens.length && tokens[i] !== ')') {
      if (tokens[i] === '(') arr.push(read());
      else { arr.push(tokens[i]); i++; }
    }
    if (tokens[i] !== ')') throw new Error('expected ")"');
    i++;
    return arr;
  }
  const ast = read();
  if (i !== tokens.length) throw new Error('extra tokens after link');
  return ast;
}
const isNum = s => /^(\d+(\.\d+)?|\.\d+)$/.test(s);
const toNum = s => Math.max(0, Math.min(1, parseFloat(s)));
const clamp01 = x => Math.max(0, Math.min(1, x));
function keyOf(node) {
  if (Array.isArray(node)) return '(' + node.map(keyOf).join(' ') + ')';
  return String(node);
}
function isStructurallySame(a,b){
  if (Array.isArray(a) && Array.isArray(b)){
    if (a.length !== b.length) return false;
    for (let i=0;i<a.length;i++) if (!isStructurallySame(a[i],b[i])) return false;
    return true;
  }
  return String(a) === String(b);
}

// ---------- Environment ----------
class Env {
  constructor(){
    this.terms = new Set();                     // declared terms (via (x: x is x))
    this.assign = new Map();                    // key(expr) -> probability
    this.symbolProb = new Map();                // optional symbol priors if you want (x: 0.7)

    // ops (redefinable)
    this.ops = new Map(Object.entries({
      'not': (x)=>1-x,
      'and': (...xs)=> xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : 0, // avg
      'or' : (...xs)=> xs.length ? Math.max(...xs) : 0,
      '='  : (L,R,ctx)=> {
        // If assigned explicitly, use that
        const k = keyOf(['=',L,R]);
        if (this.assign.has(k)) return this.assign.get(k);
        // Default: syntactic equality of terms/trees
        return isStructurallySame(L,R) ? 1 : 0;
      },
    }));
    // sugar: "!=" as not of "=" (can be redefined)
    this.defineOp('!=', (...args)=> this.getOp('not')( this.getOp('=')(...args) ));
  }
  getOp(name){
    if (!this.ops.has(name)) throw new Error(`Unknown op: ${name}`);
    return this.ops.get(name);
  }
  defineOp(name, fn){ this.ops.set(name, fn); }

  setExprProb(exprNode, p){
    this.assign.set(keyOf(exprNode), clamp01(p));
  }
  setSymbolProb(sym, p){ this.symbolProb.set(sym, clamp01(p)); }
  getSymbolProb(sym){ return this.symbolProb.has(sym) ? this.symbolProb.get(sym) : 0.5; }
}

// ---------- Eval ----------
function evalNode(node, env){
  if (typeof node === 'string') {
    if (isNum(node)) return toNum(node);
    // bare symbol → optional prior probability if set; otherwise irrelevant in calc
    return env.getSymbolProb(node);
  }

  // Definitions & operator redefs:  (head: ...)
  if (typeof node[0] === 'string' && node[0].endsWith(':')) {
    const head = node[0].slice(0,-1);
    return defineForm(head, node.slice(1), env);
  }
  if (node[1] === ':') {
    const head = node[0];
    return defineForm(head, node.slice(2), env);
  }

  // Assignment: ((expr) has probability p)
  if (node.length === 4 && node[1] === 'has' && node[2] === 'probability' && isNum(node[3])) {
    env.setExprProb(node[0], toNum(node[3]));
    return toNum(node[3]);
  }

  // Query: (? expr)
  if (node[0] === '?') {
    const v = evalNode(node[1], env);
    return { query:true, value: clamp01(v) };
  }

  // Infix AND/OR: ((A) and (B))  /  ((A) or (B))
  if (node.length === 3 && typeof node[1] === 'string' && (node[1]==='and' || node[1]==='or')) {
    const op = env.getOp(node[1]);
    const L = evalNode(node[0], env);
    const R = evalNode(node[2], env);
    return clamp01(op(L,R));
  }

  // Infix equality/inequality: (L = R), (L != R)
  if (node.length === 3 && typeof node[1] === 'string' && (node[1]==='=' || node[1]==='!=')) {
    const op = env.getOp(node[1]);
    // Equality gets raw subtrees (so syntactic/assigned equality works)
    return clamp01(op(node[0], node[2], keyOf));
  }

  // Prefix: (not X), (and X Y ...), (or X Y ...)
  const [head, ...args] = node;
  const op = env.getOp(head);
  const vals = args.map(a => evalNode(a, env));
  return clamp01(op(...vals));
}

function defineForm(head, rhs, env){
  // Term definition: (a: a is a)  → declare 'a' as a term (no probability assignment)
  if (rhs.length === 3 && typeof rhs[0]==='string' && rhs[1]==='is' && typeof rhs[2]==='string' && rhs[0]===head && rhs[2]===head) {
    env.terms.add(head);
    return 1;
  }

  // Optional symbol prior: (a: 0.7) — not required for your use-case, but allowed
  if (rhs.length === 1 && isNum(rhs[0])) {
    env.setSymbolProb(head, toNum(rhs[0]));
    return toNum(rhs[0]);
  }

  // Operator redefinitions
  if (['=','!=','and','or','not','is','?:'].includes(head) || /[=!]/.test(head)) {

    // Composition like: (!=: not =)   or  (=: =) (no-op)
    if (rhs.length === 2 && typeof rhs[0]==='string' && typeof rhs[1]==='string') {
      const outer = env.getOp(rhs[0]);
      const inner = env.getOp(rhs[1]);
      env.defineOp(head, (...xs) => clamp01( outer( inner(...xs) ) ));
      return 1;
    }

    // Aggregator selection: (and: avg|min|max|prod|ps)
    if ((head==='and' || head==='or') && rhs.length===1 && typeof rhs[0]==='string') {
      const sel = rhs[0];
      const agg =
        sel==='avg' ? xs=>xs.reduce((a,b)=>a+b,0)/xs.length :
        sel==='min' ? xs=>xs.length? Math.min(...xs) : 0 :
        sel==='max' ? xs=>xs.length? Math.max(...xs) : 0 :
        sel==='prod'? xs=>xs.reduce((a,b)=>a*b,1) :
        sel==='ps'  ? xs=> 1 - xs.reduce((a,b)=>a*(1-b),1) : null;
      if (!agg) throw new Error(`Unknown aggregator "${sel}"`);
      env.defineOp(head, (...xs)=> xs.length? agg(xs) : 0);
      return 1;
    }

    throw new Error(`Unsupported operator definition for "${head}"`);
  }

  // Generic symbol alias like (x: y) just copies y's prior probability if any
  if (rhs.length===1 && typeof rhs[0]==='string') {
    env.setSymbolProb(head, env.getSymbolProb(rhs[0]));
    return env.getSymbolProb(head);
  }

  // Else: ignore (keeps PoC minimal)
  return 0;
}

// ---------- Runner ----------
function run(text){
  const parser = new Parser();
  const links = parser.parse(text);

  // Convert each top-level LiNo link to an AST by re-tokenizing that link only.
  // Filter out comment-only links (starting with #)
  const forms = links
    .filter(linkStr => {
      const s = String(linkStr).trim();
      // Skip if it's just a comment link like "(# ...)"
      return !s.match(/^\(#\s/);
    })
    .map(linkStr => {
      const s = String(linkStr);                // link's own LiNo string
      const toks = tokenizeOne(s);
      return parseOne(toks);
    });

  const env = new Env();
  const outs = [];
  for (let form of forms) {
    // Unwrap single-element arrays (LiNo wraps everything in outer parens)
    while (Array.isArray(form) && form.length === 1 && Array.isArray(form[0])) {
      form = form[0];
    }
    const res = evalNode(form, env);
    if (res && res.query) outs.push(res.value);
  }
  return outs;
}

// CLI
if (import.meta.url === `file://${process.argv[1]}`) {
  const file = process.argv[2];
  if (!file) {
    console.error('Usage: node src/adl-links.mjs <kb.lino>');
    process.exit(1);
  }
  const text = fs.readFileSync(file, 'utf8');
  const outs = run(text);
  for (const v of outs) console.log(String(+v.toFixed(6)).replace(/\.0+$/,''));
}

export { run, tokenizeOne, parseOne, Env, evalNode };
