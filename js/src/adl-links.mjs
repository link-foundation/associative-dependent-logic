#!/usr/bin/env node
// ADL — minimal associative-dependent logic over LiNo (Links Notation)
// Supports many-valued logics from unary (1-valued) through continuous probabilistic (∞-valued).
// See: https://en.wikipedia.org/wiki/Many-valued_logic
//
// - Uses official links-notation parser to parse links
// - Terms are defined via (x: x is x)
// - Probabilities are assigned ONLY via: ((<expr>) has probability <p>)
// - Redefinable ops: (=: ...), (!=: not =), (and: avg|min|max|prod|ps), (or: ...), (not: ...)
// - Range: (range: 0 1) for [0,1] or (range: -1 1) for [-1,1] (balanced/symmetric)
// - Valence: (valence: N) to restrict truth values to N discrete levels (N=2 → Boolean, N=3 → ternary, etc.)
// - Query: (? <expr>)

import fs from 'node:fs';
import { Parser } from 'links-notation';

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
const isNum = s => /^-?(\d+(\.\d+)?|\.\d+)$/.test(s);
const clamp01 = x => Math.max(0, Math.min(1, x));

// ---------- Decimal-precision arithmetic ----------
// Round to at most `digits` significant decimal places to eliminate
// IEEE-754 floating-point artefacts (e.g. 0.1+0.2 → 0.3, not 0.30000000000000004).
const DECIMAL_PRECISION = 12;
function decRound(x) {
  if (!Number.isFinite(x)) return x;
  return +(Math.round(x + 'e' + DECIMAL_PRECISION) + 'e-' + DECIMAL_PRECISION);
}
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

// ---------- Quantization for N-valued logics ----------
// Given N discrete levels and a range [lo, hi], quantize a value to the nearest level.
// For N=2 (Boolean): levels are {lo, hi} (e.g. {0, 1} or {-1, 1})
// For N=3 (ternary): levels are {lo, mid, hi} (e.g. {0, 0.5, 1} or {-1, 0, 1})
// For N=0 or Infinity (continuous): no quantization
// See: https://en.wikipedia.org/wiki/Many-valued_logic
function quantize(x, valence, lo, hi) {
  if (valence < 2) return x; // unary or continuous — no quantization
  const step = (hi - lo) / (valence - 1);
  const level = Math.round((x - lo) / step);
  return lo + Math.max(0, Math.min(valence - 1, level)) * step;
}

// ---------- Environment ----------
class Env {
  constructor(options){
    const opts = options || {};
    this.terms = new Set();                     // declared terms (via (x: x is x))
    this.assign = new Map();                    // key(expr) -> truth value
    this.symbolProb = new Map();                // optional symbol priors if you want (x: 0.7)
    this.types = new Map();                     // key(expr) -> type expression (as string)
    this.lambdas = new Map();                   // name -> { param, paramType, body } for named lambdas

    // Range: [lo, hi] — default [0, 1] (standard probabilistic)
    // Use [-1, 1] for balanced/symmetric range
    // See: https://en.wikipedia.org/wiki/Balanced_ternary
    this.lo = opts.lo !== undefined ? opts.lo : 0;
    this.hi = opts.hi !== undefined ? opts.hi : 1;

    // Valence: number of discrete truth values (0 or Infinity = continuous)
    // N=1: unary logic (trivial, only one truth value)
    // N=2: binary/Boolean logic — https://en.wikipedia.org/wiki/Boolean_algebra
    // N=3: ternary logic — https://en.wikipedia.org/wiki/Three-valued_logic
    // N=4+: N-valued logic — https://en.wikipedia.org/wiki/Many-valued_logic
    // N=0/Infinity: continuous probabilistic / fuzzy logic — https://en.wikipedia.org/wiki/Fuzzy_logic
    this.valence = opts.valence !== undefined ? opts.valence : 0;

    // ops (redefinable)
    this.ops = new Map(Object.entries({
      'not': (x)=> this.hi - (x - this.lo),  // negation: mirrors around midpoint
      'and': (...xs)=> xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : this.lo, // avg
      'or' : (...xs)=> xs.length ? Math.max(...xs) : this.lo,
      '='  : (L,R,ctx)=> {
        // If assigned explicitly, use that (check both prefix and infix key forms)
        const kPrefix = keyOf(['=',L,R]);
        if (this.assign.has(kPrefix)) return this.assign.get(kPrefix);
        const kInfix = keyOf([L,'=',R]);
        if (this.assign.has(kInfix)) return this.assign.get(kInfix);
        // Default: syntactic equality of terms/trees
        return isStructurallySame(L,R) ? this.hi : this.lo;
      },
    }));
    // sugar: "!=" as not of "=" (can be redefined)
    this.defineOp('!=', (...args)=> this.getOp('not')( this.getOp('=')(...args) ));

    // Arithmetic operators (decimal-precision by default)
    this.defineOp('+', (a,b)=> decRound(a + b));
    this.defineOp('-', (a,b)=> decRound(a - b));
    this.defineOp('*', (a,b)=> decRound(a * b));
    this.defineOp('/', (a,b)=> b === 0 ? 0 : decRound(a / b));

    // Initialize truth constants: true, false, unknown, undefined
    // These are predefined symbol probabilities based on the current range.
    // By default: (false: min(range)), (true: max(range)),
    //             (unknown: mid(range)), (undefined: mid(range))
    // They can be redefined by the user via (true: <value>), (false: <value>), etc.
    this._initTruthConstants();
  }

  // Clamp and optionally quantize a value to the valid range
  clamp(x) {
    const clamped = Math.max(this.lo, Math.min(this.hi, x));
    if (this.valence >= 2) return quantize(clamped, this.valence, this.lo, this.hi);
    return clamped;
  }

  // Parse a numeric string respecting current range
  toNum(s) {
    return this.clamp(parseFloat(s));
  }

  // Midpoint of the range (useful for paradox resolution, default symbol prob, etc.)
  get mid() { return (this.lo + this.hi) / 2; }

  // Initialize truth constants based on current range.
  // (false: min(range)), (true: max(range)),
  // (unknown: mid(range)), (undefined: mid(range))
  _initTruthConstants() {
    this.symbolProb.set('true', this.hi);
    this.symbolProb.set('false', this.lo);
    this.symbolProb.set('unknown', this.mid);
    this.symbolProb.set('undefined', this.mid);
  }

  getOp(name){
    if (!this.ops.has(name)) throw new Error(`Unknown op: ${name}`);
    return this.ops.get(name);
  }
  defineOp(name, fn){ this.ops.set(name, fn); }

  setExprProb(exprNode, p){
    this.assign.set(keyOf(exprNode), this.clamp(p));
  }
  setType(exprNode, typeExpr){
    const key = typeof exprNode === 'string' ? exprNode : keyOf(exprNode);
    this.types.set(key, typeof typeExpr === 'string' ? typeExpr : keyOf(typeExpr));
  }
  getType(exprNode){
    const key = typeof exprNode === 'string' ? exprNode : keyOf(exprNode);
    return this.types.get(key) || null;
  }
  setLambda(name, param, paramType, body){
    this.lambdas.set(name, { param, paramType, body });
  }
  getLambda(name){
    return this.lambdas.get(name) || null;
  }
  setSymbolProb(sym, p){ this.symbolProb.set(sym, this.clamp(p)); }
  getSymbolProb(sym){ return this.symbolProb.has(sym) ? this.symbolProb.get(sym) : this.mid; }
}

// ---------- Binding parser ----------
// Parse a binding form in two supported syntaxes:
// 1. Colon form: (x: A) as ['x:', A] — standard LiNo link definition syntax
// 2. Prefix type form: (A x) as ['A', 'x'] — type-first notation for lambda/Pi bindings
//    e.g. (Natural x), used in (lambda (Natural x) body)
// Returns { paramName, paramType } or null if not a valid binding.
function parseBinding(binding) {
  if (!Array.isArray(binding)) return null;
  // ['x:', A] — two elements where first ends with colon (standard LiNo link definition)
  if (binding.length === 2 && typeof binding[0] === 'string' && binding[0].endsWith(':')) {
    return { paramName: binding[0].slice(0, -1), paramType: binding[1] };
  }
  // ['A', 'x'] — prefix type form: type name first, then variable name
  // Type names must start with uppercase (convention from Lean/Rocq)
  if (binding.length === 2 && typeof binding[0] === 'string' && typeof binding[1] === 'string'
      && /^[A-Z]/.test(binding[0]) && !binding[1].endsWith(':')) {
    return { paramName: binding[1], paramType: binding[0] };
  }
  return null;
}

// ---------- Multi-binding parser ----------
// Parse comma-separated bindings: (Natural x, Natural y) → [{paramName:'x', paramType:'Natural'}, ...]
// Tokens arrive as ['Natural', 'x,', 'Natural', 'y'] or ['Natural', 'x'] (single binding)
function parseBindings(binding) {
  if (!Array.isArray(binding)) return null;
  // Try single binding first
  const single = parseBinding(binding);
  if (single) return [single];
  // Try comma-separated: flatten tokens, split by commas
  const tokens = [];
  for (const tok of binding) {
    if (typeof tok === 'string') {
      // Split tokens that end with comma: 'x,' → 'x', separator
      if (tok.endsWith(',')) {
        tokens.push(tok.slice(0, -1));
        tokens.push(',');
      } else {
        tokens.push(tok);
      }
    } else {
      tokens.push(tok);
    }
  }
  // Group into pairs separated by commas
  const bindings = [];
  let i = 0;
  while (i < tokens.length) {
    if (tokens[i] === ',') { i++; continue; }
    if (i + 1 < tokens.length && typeof tokens[i] === 'string' && typeof tokens[i+1] === 'string' && tokens[i+1] !== ',') {
      const pair = parseBinding([tokens[i], tokens[i+1]]);
      if (pair) {
        bindings.push(pair);
        i += 2;
        continue;
      }
    }
    return null; // invalid binding format
  }
  return bindings.length > 0 ? bindings : null;
}

// ---------- Substitution (for beta-reduction) ----------
// Substitute all occurrences of variable `name` with `replacement` in `expr`.
// Both expr and replacement can be strings or arrays (AST nodes).
function substitute(expr, name, replacement) {
  if (typeof expr === 'string') {
    return expr === name ? replacement : expr;
  }
  if (Array.isArray(expr)) {
    // Don't substitute inside a binding that shadows the variable
    if (expr.length === 3 && (expr[0] === 'lambda' || expr[0] === 'Pi') && Array.isArray(expr[1])) {
      const parsed = parseBinding(expr[1]);
      if (parsed && parsed.paramName === name) {
        return expr; // shadowed
      }
    }
    return expr.map(child => substitute(child, name, replacement));
  }
  return expr;
}

// ---------- Eval ----------
// Evaluate a node in arithmetic context — numeric literals are NOT clamped to the logic range.
function evalArith(node, env){
  if (typeof node === 'string' && isNum(node)) return parseFloat(node);
  return evalNode(node, env);
}

function evalNode(node, env){
  if (typeof node === 'string') {
    if (isNum(node)) return env.toNum(node);
    // bare symbol → optional prior probability if set; otherwise irrelevant in calc
    return env.getSymbolProb(node);
  }

  // Definitions & operator redefs:  (head: ...)
  if (typeof node[0] === 'string' && node[0].endsWith(':')) {
    const head = node[0].slice(0,-1);
    return defineForm(head, node.slice(1), env);
  }
  // Note: (x : A) with spaces as a standalone colon separator is NOT supported.
  // Use (x: A) instead — the colon must be part of the link name.

  // Assignment: ((expr) has probability p)
  if (node.length === 4 && node[1] === 'has' && node[2] === 'probability' && isNum(node[3])) {
    env.setExprProb(node[0], parseFloat(node[3]));
    return env.toNum(node[3]);
  }

  // Range configuration: (range: lo hi) — sets the truth value range
  // (range: 0 1) for standard [0,1] or (range: -1 1) for balanced [-1,1]
  // See: https://en.wikipedia.org/wiki/Balanced_ternary
  // Must be checked in evalNode for (range lo hi) prefix form
  if (node.length === 3 && node[0] === 'range' && isNum(node[1]) && isNum(node[2])) {
    env.lo = parseFloat(node[1]);
    env.hi = parseFloat(node[2]);
    // Re-initialize ops for new range
    _reinitOps(env);
    return 1;
  }

  // Valence configuration: (valence N) prefix form
  if (node.length === 2 && node[0] === 'valence' && isNum(node[1])) {
    env.valence = parseInt(node[1], 10);
    return 1;
  }

  // Query: (? expr)
  if (node[0] === '?') {
    const v = evalNode(node[1], env);
    // If inner result is already a query (e.g. from (type of x)), pass it through
    if (v && typeof v === 'object' && v.query) return v;
    return { query:true, value: env.clamp(v) };
  }

  // Infix arithmetic: (A + B), (A - B), (A * B), (A / B)
  // Arithmetic uses raw numeric values (not clamped to the logic range)
  if (node.length === 3 && typeof node[1] === 'string' && ['+','-','*','/'].includes(node[1])) {
    const op = env.getOp(node[1]);
    const L = evalArith(node[0], env);
    const R = evalArith(node[2], env);
    return op(L,R);
  }

  // Infix AND/OR: ((A) and (B))  /  ((A) or (B))
  if (node.length === 3 && typeof node[1] === 'string' && (node[1]==='and' || node[1]==='or')) {
    const op = env.getOp(node[1]);
    const L = evalNode(node[0], env);
    const R = evalNode(node[2], env);
    return env.clamp(op(L,R));
  }

  // Infix equality/inequality: (L = R), (L != R)
  if (node.length === 3 && typeof node[1] === 'string' && (node[1]==='=' || node[1]==='!=')) {
    const op = env.getOp(node[1]);
    // Equality checks assigned probability first, then structural equality,
    // then falls back to numeric comparison of evaluated values (decimal-precision)
    const raw = op(node[0], node[2], keyOf);
    // If structural/assigned equality already gave a definitive answer, use it
    if (raw === env.hi || raw === env.lo) {
      // Check if there's an explicit assignment — if so, trust it
      const kPrefix = keyOf(['=',node[0],node[2]]);
      const kInfix = keyOf([node[0],'=',node[2]]);
      if (env.assign.has(kPrefix) || env.assign.has(kInfix) || isStructurallySame(node[0], node[2])) {
        return env.clamp(raw);
      }
      // No explicit assignment and not structurally same — try numeric comparison
      const L = evalNode(node[0], env);
      const R = evalNode(node[2], env);
      const numEq = decRound(L) === decRound(R) ? env.hi : env.lo;
      if (node[1] === '!=') return env.clamp(env.getOp('not')(numEq));
      return env.clamp(numEq);
    }
    return env.clamp(raw);
  }

  // ---------- Type System: "everything is a link" ----------

  // Type universe: (Type N) — the sort at universe level N
  if (node.length === 2 && node[0] === 'Type') {
    const level = isNum(node[1]) ? parseInt(node[1], 10) : 0;
    // (Type N) has type (Type N+1)
    env.setType(node, ['Type', String(level + 1)]);
    return 1; // valid expression
  }

  // Prop: (Prop) is sugar for (Type 0) in the propositions-as-types interpretation
  if (node.length === 1 && node[0] === 'Prop') {
    env.setType(['Prop'], ['Type', '1']);
    return 1;
  }

  // Dependent product (Pi-type): (Pi (A x) B) or (Pi (x: A) B)
  if (node.length === 3 && node[0] === 'Pi') {
    const binding = node[1];
    const parsed = parseBinding(binding);
    if (parsed) {
      const { paramName, paramType } = parsed;
      env.terms.add(paramName);
      env.setType(paramName, paramType);
      env.setType(node, ['Type', '0']);
    }
    return 1;
  }

  // Lambda abstraction: (lambda (A x) body) or (lambda (x: A) body)
  // Also supports multi-param: (lambda (A x, B y) body)
  if (node.length === 3 && node[0] === 'lambda') {
    const binding = node[1];
    const bindings = parseBindings(binding);
    if (bindings && bindings.length > 0) {
      // For single binding — standard case
      const { paramName, paramType } = bindings[0];
      const body = node[2];
      env.terms.add(paramName);
      env.setType(paramName, paramType);
      // Register additional bindings
      for (let i = 1; i < bindings.length; i++) {
        env.terms.add(bindings[i].paramName);
        env.setType(bindings[i].paramName, bindings[i].paramType);
      }
      const bodyType = env.getType(body);
      const paramTypeKey = typeof paramType === 'string' ? paramType : keyOf(paramType);
      const bodyTypeKey = bodyType || 'unknown';
      env.setType(node, '(Pi (' + paramTypeKey + ' ' + paramName + ') ' + bodyTypeKey + ')');
    }
    return 1;
  }

  // Application: (apply f x) — explicit application with beta-reduction
  if (node.length === 3 && node[0] === 'apply') {
    const fn = node[1];
    const arg = node[2];

    // Check if fn is a lambda: (lambda (A x) body)
    if (Array.isArray(fn) && fn.length === 3 && fn[0] === 'lambda') {
      const parsed = parseBinding(fn[1]);
      if (parsed) {
        const body = fn[2];
        const result = substitute(body, parsed.paramName, arg);
        return evalNode(result, env);
      }
    }

    // Check if fn is a named lambda
    if (typeof fn === 'string') {
      const lambda = env.getLambda(fn);
      if (lambda) {
        const result = substitute(lambda.body, lambda.param, arg);
        return evalNode(result, env);
      }
    }

    // Otherwise evaluate fn and arg normally
    const fVal = evalNode(fn, env);
    const aVal = evalNode(arg, env);
    return typeof fVal === 'number' ? fVal : (fVal && fVal.value !== undefined ? fVal.value : 0);
  }

  // Type query: (type of expr) — returns the type of an expression
  // e.g. (? (type of x)) → returns the type string
  if (node.length === 3 && node[0] === 'type' && node[1] === 'of') {
    const expr = node[2];
    const typeStr = env.getType(expr);
    if (typeStr) {
      return { query: true, value: typeStr, typeQuery: true };
    }
    return { query: true, value: 'unknown', typeQuery: true };
  }

  // Type check query: (expr of Type) — checks if expr has the given type
  // e.g. (? (x of Natural)) → returns 1 or 0
  if (node.length === 3 && node[1] === 'of') {
    const expr = node[0];
    const expectedType = node[2];
    const actualType = env.getType(expr);
    if (actualType) {
      const expectedKey = typeof expectedType === 'string' ? expectedType : keyOf(expectedType);
      return actualType === expectedKey ? env.hi : env.lo;
    }
    return env.lo;
  }

  // Prefix: (not X), (and X Y ...), (or X Y ...)
  const [head, ...args] = node;
  if (typeof head === 'string' && env.ops.has(head)) {
    const op = env.getOp(head);
    const vals = args.map(a => evalNode(a, env));
    return env.clamp(op(...vals));
  }

  // Fall through: prefix application (f x y ...) for named lambdas
  if (typeof head === 'string' && args.length >= 1) {
    const lambda = env.getLambda(head);
    if (lambda) {
      // Apply first argument, then recursively apply rest
      let result = substitute(lambda.body, lambda.param, args[0]);
      if (args.length === 1) {
        return evalNode(result, env);
      }
      // Curry: apply remaining args
      let current = evalNode(result, env);
      for (let i = 1; i < args.length; i++) {
        // If current result is a lambda, apply next arg
        current = evalNode(args[i], env);
      }
      return current;
    }
  }

  return 0;
}

// Re-initialize default ops when range changes
function _reinitOps(env) {
  env.ops.set('not', (x) => env.hi - (x - env.lo));
  env.ops.set('and', (...xs) => xs.length ? xs.reduce((a,b)=>a+b,0)/xs.length : env.lo);
  env.ops.set('or', (...xs) => xs.length ? Math.max(...xs) : env.lo);
  env.ops.set('=', (L,R,ctx) => {
    const kPrefix = keyOf(['=',L,R]);
    if (env.assign.has(kPrefix)) return env.assign.get(kPrefix);
    const kInfix = keyOf([L,'=',R]);
    if (env.assign.has(kInfix)) return env.assign.get(kInfix);
    return isStructurallySame(L,R) ? env.hi : env.lo;
  });
  env.ops.set('!=', (...args) => env.getOp('not')( env.getOp('=')(...args) ));
  env.ops.set('+', (a,b) => decRound(a + b));
  env.ops.set('-', (a,b) => decRound(a - b));
  env.ops.set('*', (a,b) => decRound(a * b));
  env.ops.set('/', (a,b) => b === 0 ? 0 : decRound(a / b));
  // Re-initialize truth constants for new range
  env._initTruthConstants();
}

function defineForm(head, rhs, env){
  // Term definition: (a: a is a)  → declare 'a' as a term (no probability assignment)
  if (rhs.length === 3 && typeof rhs[0]==='string' && rhs[1]==='is' && typeof rhs[2]==='string' && rhs[0]===head && rhs[2]===head) {
    env.terms.add(head);
    return 1;
  }

  // Prefix type notation: (name: TypeName name) → typed self-referential declaration
  // e.g. (zero: Natural zero), (boolean: Type boolean), (true: Boolean true)
  if (rhs.length === 2 && typeof rhs[0] === 'string' && typeof rhs[1] === 'string' && rhs[1] === head) {
    const typeName = rhs[0];
    // Only if typeName starts with uppercase (type convention) and is not an operator
    if (/^[A-Z]/.test(typeName)) {
      env.terms.add(head);
      env.setType(head, typeName);
      return 1;
    }
  }

  // Prefix type notation with complex type: (name: (Type 0) name) → typed self-referential declaration
  if (rhs.length === 2 && Array.isArray(rhs[0]) && typeof rhs[1] === 'string' && rhs[1] === head) {
    const typeExpr = rhs[0];
    env.terms.add(head);
    env.setType(head, typeExpr);
    evalNode(typeExpr, env);
    return 1;
  }

  // Typed declaration with complex type expression: (Natural: (Type 0)), (succ: (Pi (Natural n) Natural))
  // Only complex expressions (arrays) are accepted as type annotations in single-element form.
  // Simple name type annotations like (x: Natural) are NOT supported — use (x: Natural x) prefix form instead.
  if (rhs.length === 1 && Array.isArray(rhs[0])) {
    const isOp = ['=','!=','and','or','not','is','?:'].includes(head) || /[=!]/.test(head);
    if (!isOp) {
      const typeExpr = rhs[0];
      env.terms.add(head);
      env.setType(head, typeExpr);
      evalNode(typeExpr, env);
      return 1;
    }
  }

  // Range configuration: (range: lo hi) — sets the truth value range
  if (head === 'range' && rhs.length === 2 && isNum(rhs[0]) && isNum(rhs[1])) {
    env.lo = parseFloat(rhs[0]);
    env.hi = parseFloat(rhs[1]);
    _reinitOps(env);
    return 1;
  }

  // Valence configuration: (valence: N) — sets the number of truth values
  // N=1: unary (trivial), N=2: binary (Boolean), N=3: ternary, N=0: continuous
  if (head === 'valence' && rhs.length === 1 && isNum(rhs[0])) {
    env.valence = parseInt(rhs[0], 10);
    return 1;
  }

  // Optional symbol prior: (a: 0.7) — not required for your use-case, but allowed
  if (rhs.length === 1 && isNum(rhs[0])) {
    env.setSymbolProb(head, parseFloat(rhs[0]));
    return env.toNum(rhs[0]);
  }

  // Operator redefinitions
  if (['=','!=','and','or','not','is','?:'].includes(head) || /[=!]/.test(head)) {

    // Composition like: (!=: not =)   or  (=: =) (no-op)
    if (rhs.length === 2 && typeof rhs[0]==='string' && typeof rhs[1]==='string') {
      const outer = env.getOp(rhs[0]);
      const inner = env.getOp(rhs[1]);
      env.defineOp(head, (...xs) => env.clamp( outer( inner(...xs) ) ));
      return 1;
    }

    // Aggregator selection: (and: avg|min|max|prod|ps)
    if ((head==='and' || head==='or') && rhs.length===1 && typeof rhs[0]==='string') {
      const sel = rhs[0];
      const lo = env.lo;
      const agg =
        sel==='avg' ? xs=>xs.reduce((a,b)=>a+b,0)/xs.length :
        sel==='min' ? xs=>xs.length? Math.min(...xs) : lo :
        sel==='max' ? xs=>xs.length? Math.max(...xs) : lo :
        sel==='prod'? xs=>xs.reduce((a,b)=>a*b,1) :
        sel==='ps'  ? xs=> 1 - xs.reduce((a,b)=>a*(1-b),1) : null;
      if (!agg) throw new Error(`Unknown aggregator "${sel}"`);
      env.defineOp(head, (...xs)=> xs.length? agg(xs) : lo);
      return 1;
    }

    throw new Error(`Unsupported operator definition for "${head}"`);
  }

  // Lambda definition: (name: lambda (A x) body)
  if (rhs.length >= 2 && rhs[0] === 'lambda') {
    if (rhs.length === 3 && Array.isArray(rhs[1])) {
      const parsed = parseBinding(rhs[1]);
      if (parsed) {
        const { paramName, paramType } = parsed;
        const body = rhs[2];
        env.terms.add(head);
        env.setLambda(head, paramName, paramType, body);
        const paramTypeKey = typeof paramType === 'string' ? paramType : keyOf(paramType);
        const bodyTypeKey = env.getType(body) || (typeof body === 'string' ? body : keyOf(body));
        env.setType(head, '(Pi (' + paramTypeKey + ' ' + paramName + ') ' + bodyTypeKey + ')');
        return 1;
      }
    }
  }

  // Typed definition: (name : Type) — just a type annotation (no body)
  // Already handled by the (x: A) form in evalNode

  // Generic symbol alias like (x: y) just copies y's prior probability if any
  if (rhs.length===1 && typeof rhs[0]==='string') {
    env.setSymbolProb(head, env.getSymbolProb(rhs[0]));
    return env.getSymbolProb(head);
  }

  // Else: ignore (keeps PoC minimal)
  return 0;
}

// ---------- Runner ----------
function run(text, options){
  const parser = new Parser();
  // Strip line comments (# ...) before parsing — LiNo parser doesn't handle them
  const stripped = text.replace(/^[ \t]*#.*$/gm, '').replace(/\n{3,}/g, '\n\n');
  const links = parser.parse(stripped);

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

  const env = new Env(options);
  const outs = [];
  for (let form of forms) {
    // Unwrap single-element arrays (LiNo wraps everything in outer parens)
    while (Array.isArray(form) && form.length === 1 && Array.isArray(form[0])) {
      form = form[0];
    }
    const res = evalNode(form, env);
    if (res && res.query) {
      if (res.typeQuery) {
        // Type queries return string results
        outs.push(res.value);
      } else {
        outs.push(res.value);
      }
    }
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
  for (const v of outs) {
    if (typeof v === 'string') {
      console.log(v);
    } else {
      console.log(String(+v.toFixed(6)).replace(/\.0+$/,''));
    }
  }
}

export { run, tokenizeOne, parseOne, Env, evalNode, quantize, decRound, substitute, parseBindings };
