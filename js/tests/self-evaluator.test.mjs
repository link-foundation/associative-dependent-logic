// Tests for `lib/self/evaluator.lino` (issue #85).
//
// The evaluator file is data: it records the host evaluator as `(rule ...)`
// links. These tests keep that data parseable, require the built-in operator
// rule surface, and run a small rule-backed evaluator against the shared
// test corpus and examples so the encoded rules stay tied to host behavior.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import {
  checkProofObject,
  decRound,
  evaluateFile,
  isNum,
  isStructurallySame,
  keyOf,
  parseBinding,
  parseLino,
  parseOne,
  parseProofAssumptionForm,
  parseProofObjectForm,
  parseRuleForm,
  quantize,
  run,
  subst,
  tokenizeOne,
} from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const evaluatorPath = join(repoRoot, 'lib', 'self', 'evaluator.lino');
const examplesDir = join(repoRoot, 'examples');
const selfCorpusDir = join(repoRoot, 'test-corpus');

const REQUIRED_EVAL_RULES = [
  '(eval numeric-literal)',
  '(eval symbol)',
  '(eval (range low high))',
  '(eval (range: low high))',
  '(eval (valence levels))',
  '(eval (valence: levels))',
  '(eval (name: name is name))',
  '(eval (name: type-name name))',
  '(eval (name: type-expression name))',
  '(eval (name: type-expression))',
  '(eval (operator: aggregator))',
  '(eval (operator: outer inner))',
  '(eval (name: lambda binding body))',
  '(eval ((expression) has probability number))',
  '(eval (? expression))',
  '(eval (left + right))',
  '(eval (left - right))',
  '(eval (left * right))',
  '(eval (left / right))',
  '(eval (left < right))',
  '(eval (left <= right))',
  '(eval (not value))',
  '(eval (and a b))',
  '(eval (or a b))',
  '(eval (both a b))',
  '(eval (neither a b))',
  '(eval (a and b))',
  '(eval (a or b))',
  '(eval (both a and b))',
  '(eval (neither a nor b))',
  '(eval (= left right))',
  '(eval (!= left right))',
  '(eval (left = right))',
  '(eval (left != right))',
  '(eval (Type level))',
  '(eval (Prop))',
  '(eval (Pi binding body))',
  '(eval (lambda binding body))',
  '(eval (apply function argument))',
  '(eval (subst term variable replacement))',
  '(eval (fresh variable in body))',
  '(eval (whnf expression))',
  '(eval (nf expression))',
  '(eval (normal-form expression))',
  '(eval (type of expression))',
  '(eval (expression of type))',
  '(eval (domain name request))',
  '(eval (root-construct name details))',
  '(eval (foundation name details))',
  '(eval (with-foundation name body))',
  '(eval (foundation-report))',
  '(eval (strict-foundation pure-links))',
  '(eval (allow-host-primitive names))',
  '(eval (assumption name (judgement judgement)))',
  '(eval (axiom name (judgement judgement)))',
  '(eval (proof-object name clauses))',
  '(eval (check-proof name))',
  '(eval (encodeAnum node))',
  '(eval (decodeAnum payload))',
];

const REQUIRED_SURFACE_RULES = [
  '(foundation-clause (description text))',
  '(foundation-clause (uses name))',
  '(foundation-clause (defines operator implementation))',
  '(foundation-clause (extends name))',
  '(foundation-clause (numeric-domain name))',
  '(foundation-clause (truth-domain name))',
  '(foundation-clause (carrier values))',
  '(foundation-clause strict-carrier)',
  '(foundation-clause (truth-table operator rows))',
  '(foundation-clause experimental)',
  '(foundation-clause (root symbol))',
  '(foundation-clause (abit symbol bits))',
  '(proof-object-clause (premise judgement))',
  '(proof-object-clause (premise-by name))',
  '(proof-object-clause (uses names))',
  '(equality-provenance left right)',
];

const REQUIRED_OPERATORS = [
  'not', 'and', 'or', 'both', 'neither', '=', '!=', '+', '-', '*', '/', '<', '<=',
];

function parseForms(source) {
  return parseLino(source).map(link => parseOne(tokenizeOne(link)));
}

function evaluatorForms() {
  return parseForms(readFileSync(evaluatorPath, 'utf8'));
}

function rulePatterns(forms) {
  const patterns = new Set();
  for (const form of forms) {
    if (Array.isArray(form) && form[0] === 'rule') {
      patterns.add(keyOf(form[1]));
    }
  }
  return patterns;
}

function evalRulePatterns(forms) {
  const patterns = new Set();
  for (const form of forms) {
    if (Array.isArray(form) && form[0] === 'rule' && Array.isArray(form[1]) && form[1][0] === 'eval') {
      patterns.add(keyOf(form[1]));
    }
  }
  return patterns;
}

function builtInOperators(forms) {
  const operators = new Set();
  for (const form of forms) {
    if (Array.isArray(form) && form[0] === 'built-in-operator' && typeof form[1] === 'string') {
      operators.add(form[1]);
    }
  }
  return operators;
}

function desugarHoas(node) {
  if (!Array.isArray(node)) return node;
  const mapped = node.map(desugarHoas);
  if (mapped.length === 3 && mapped[0] === 'forall' && Array.isArray(mapped[1])) {
    return ['Pi', mapped[1], mapped[2]];
  }
  return mapped;
}

function isProofRuleShape(node) {
  return Array.isArray(node) &&
    node[0] === 'rule' &&
    typeof node[1] === 'string' &&
    node[1] &&
    node.length >= 3 &&
    node.slice(2).every(c => Array.isArray(c) && (c[0] === 'premise' || c[0] === 'conclusion')) &&
    node.slice(2).some(c => c[0] === 'conclusion');
}

class EncodedEvaluator {
  constructor(rulePatterns) {
    this.rulePatterns = rulePatterns;
    this.reset();
  }

  reset() {
    this.lo = 0;
    this.hi = 1;
    this.valence = 0;
    this.assign = new Map();
    this.symbolProb = new Map();
    this.terms = new Set();
    this.types = new Map();
    this.lambdas = new Map();
    this.ops = new Map();
    this.foundations = new Map();
    this.foundationStack = [];
    this.activeFoundation = 'default-rml';
    this.proofRules = new Map();
    this.proofAssumptions = new Map();
    this.proofObjects = new Map();
    this.reinitOps();
    this.registerBuiltinFoundations();
  }

  getProofRule(name) {
    return this.proofRules.get(name) || null;
  }

  getProofAssumption(name) {
    return this.proofAssumptions.get(name) || null;
  }

  getProofObject(name) {
    return this.proofObjects.get(name) || null;
  }

  requireRule(pattern) {
    assert.ok(this.rulePatterns.has(pattern), `missing encoded rule ${pattern}`);
  }

  reinitOps() {
    this.ops.set('not', x => this.hi - (x - this.lo));
    this.ops.set('and', (...xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : this.lo);
    this.ops.set('or', (...xs) => xs.length ? Math.max(...xs) : this.lo);
    this.ops.set('both', (...xs) => xs.length ? decRound(xs.reduce((a, b) => a + b, 0) / xs.length) : this.lo);
    this.ops.set('neither', (...xs) => xs.length ? decRound(xs.reduce((a, b) => a * b, 1)) : this.lo);
    this.ops.set('=', (left, right) => isStructurallySame(left, right) ? this.hi : this.lo);
    this.ops.set('!=', (left, right) => this.ops.get('not')(this.ops.get('=')(left, right)));
    this.ops.set('+', (a, b) => decRound(a + b));
    this.ops.set('-', (a, b) => decRound(a - b));
    this.ops.set('*', (a, b) => decRound(a * b));
    this.ops.set('/', (a, b) => b === 0 ? 0 : decRound(a / b));
    this.ops.set('<', (a, b) => a < b ? this.hi : this.lo);
    this.ops.set('<=', (a, b) => a <= b ? this.hi : this.lo);
    this.initTruthConstants();
  }

  registerBuiltinFoundations() {
    this.foundations.set('boolean-links', {
      name: 'boolean-links',
      defines: new Map(),
      truthTables: new Map([
        ['and', [
          { inputs: ['1', '1'], output: '1' },
          { inputs: ['1', '0'], output: '0' },
          { inputs: ['0', '1'], output: '0' },
          { inputs: ['0', '0'], output: '0' },
        ]],
        ['or', [
          { inputs: ['1', '1'], output: '1' },
          { inputs: ['1', '0'], output: '1' },
          { inputs: ['0', '1'], output: '1' },
          { inputs: ['0', '0'], output: '0' },
        ]],
        ['not', [
          { inputs: ['1'], output: '0' },
          { inputs: ['0'], output: '1' },
        ]],
      ]),
    });
  }

  initTruthConstants() {
    this.symbolProb.set('true', this.hi);
    this.symbolProb.set('false', this.lo);
    this.symbolProb.set('unknown', this.mid);
    this.symbolProb.set('undefined', this.mid);
  }

  get mid() {
    return (this.lo + this.hi) / 2;
  }

  clamp(value) {
    const clamped = Math.max(this.lo, Math.min(this.hi, value));
    return this.valence >= 2 ? quantize(clamped, this.valence, this.lo, this.hi) : clamped;
  }

  toNum(token) {
    return this.clamp(parseFloat(token));
  }

  setSymbolProb(name, value) {
    this.symbolProb.set(name, this.clamp(value));
  }

  getSymbolProb(name) {
    return this.symbolProb.has(name) ? this.symbolProb.get(name) : this.mid;
  }

  setType(expr, typeExpr) {
    this.types.set(typeof expr === 'string' ? expr : keyOf(expr), typeof typeExpr === 'string' ? typeExpr : keyOf(typeExpr));
  }

  getType(expr) {
    return this.types.get(typeof expr === 'string' ? expr : keyOf(expr)) ?? null;
  }

  setExprProb(expr, value) {
    this.assign.set(keyOf(expr), this.clamp(value));
  }

  run(source) {
    this.reset();
    const results = [];
    for (const form of parseForms(source)) {
      this.evalTopForm(form, results);
    }
    return results;
  }

  evalTopForm(form, results) {
    if (Array.isArray(form) && form[0] === 'with-foundation') {
      this.requireRule('(eval (with-foundation name body))');
      const name = form[1];
      const entered = this.enterFoundation(name);
      try {
        for (let i = 2; i < form.length; i++) {
          let body = form[i];
          while (Array.isArray(body) && body.length === 1 && Array.isArray(body[0])) {
            body = body[0];
          }
          this.evalTopForm(body, results);
        }
      } finally {
        if (entered) this.exitFoundation();
      }
      return;
    }
    if (Array.isArray(form) && form[0] === 'foundation') {
      this.requireRule('(eval (foundation name details))');
      this.registerFoundation(form);
      return;
    }
    if (Array.isArray(form) && form[0] === 'root-construct') {
      this.requireRule('(eval (root-construct name details))');
      return;
    }
    if (Array.isArray(form) && form[0] === 'foundation-report') {
      this.requireRule('(eval (foundation-report))');
      return;
    }
    if (Array.isArray(form) && form[0] === 'strict-foundation') {
      this.requireRule('(eval (strict-foundation pure-links))');
      return;
    }
    if (Array.isArray(form) && form[0] === 'allow-host-primitive') {
      this.requireRule('(eval (allow-host-primitive names))');
      return;
    }
    if (Array.isArray(form) && form[0] === 'rule' && isProofRuleShape(form)) {
      try {
        const rule = parseRuleForm(form);
        this.proofRules.set(rule.name, {
          name: rule.name,
          premises: rule.premises ? rule.premises.slice() : [],
          conclusion: rule.conclusion,
        });
      } catch {
        // Ignore malformed declarations; host evaluator emits diagnostics
        // separately and we only care about parity for `(check-proof ...)`.
      }
      return;
    }
    if (Array.isArray(form) && (form[0] === 'assumption' || form[0] === 'axiom')) {
      this.requireRule(`(eval (${form[0]} name (judgement judgement)))`);
      try {
        const assumption = parseProofAssumptionForm(form);
        this.proofAssumptions.set(assumption.name, {
          name: assumption.name,
          kind: assumption.kind || 'assumption',
          judgement: assumption.judgement,
        });
      } catch {
        // Same rationale as `(rule ...)` above.
      }
      return;
    }
    if (Array.isArray(form) && form[0] === 'proof-object') {
      this.requireRule('(eval (proof-object name clauses))');
      try {
        const po = parseProofObjectForm(form);
        this.proofObjects.set(po.name, {
          name: po.name,
          rule: po.rule,
          premises: po.premises ? po.premises.slice() : [],
          premiseRefs: po.premiseRefs ? po.premiseRefs.slice() : [],
          conclusion: po.conclusion,
        });
      } catch {
        // Same rationale as `(rule ...)` above.
      }
      return;
    }
    if (Array.isArray(form) && form[0] === 'check-proof') {
      this.requireRule('(eval (check-proof name))');
      if (form.length !== 2 || typeof form[1] !== 'string') return;
      const verdict = checkProofObject(this, form[1]);
      results.push(verdict.ok ? 1 : 0);
      return;
    }
    if (Array.isArray(form) && (form[0] === 'encodeAnum' || form[0] === 'decodeAnum')) {
      this.requireRule(`(eval (${form[0]} ${form[0] === 'encodeAnum' ? 'node' : 'payload'}))`);
      return;
    }
    const value = this.evalNode(form);
    if (value && typeof value === 'object' && value.query) {
      results.push(value.value);
    }
  }

  registerFoundation(form) {
    if (!Array.isArray(form) || form.length < 2 || typeof form[1] !== 'string') return;
    const name = form[1];
    const entry = { name, defines: new Map(), truthTables: new Map() };
    for (let i = 2; i < form.length; i++) {
      const part = form[i];
      if (!Array.isArray(part) || part.length === 0) continue;
      if (part[0] === 'defines' && typeof part[1] === 'string' && typeof part[2] === 'string') {
        entry.defines.set(part[1], part[2]);
      } else if (part[0] === 'truth-table' && typeof part[1] === 'string') {
        const rows = [];
        for (const row of part.slice(2)) {
          if (!Array.isArray(row)) continue;
          const arrow = row.indexOf('->');
          if (arrow <= 0 || arrow !== row.length - 2) continue;
          rows.push({ inputs: row.slice(0, arrow), output: row[row.length - 1] });
        }
        if (rows.length > 0) entry.truthTables.set(part[1], rows);
      } else if (part[0] === 'description' && typeof part[1] === 'string') {
        entry.description = part[1];
      } else if (part[0] === 'numeric-domain' && typeof part[1] === 'string') {
        entry.numericDomain = part[1];
      }
    }
    this.foundations.set(name, entry);
  }

  enterFoundation(name) {
    const foundation = this.foundations.get(name);
    if (!foundation) return false;
    const snapshot = [];
    const remember = (opName) => {
      if (snapshot.some(([name]) => name === opName)) return;
      const prev = this.ops.has(opName) ? this.ops.get(opName) : null;
      snapshot.push([opName, prev]);
    };
    for (const [opName, implName] of foundation.defines) {
      remember(opName);
      let impl;
      try {
        impl = this.aggregator(implName);
      } catch {
        if (this.ops.has(implName)) {
          impl = this.ops.get(implName);
        } else {
          continue;
        }
      }
      this.ops.set(opName, impl);
    }
    if (foundation.truthTables instanceof Map) {
      for (const [opName, rows] of foundation.truthTables) {
        remember(opName);
        const previous = this.ops.has(opName) ? this.ops.get(opName) : null;
        this.ops.set(opName, this.truthTableOp(rows, previous));
      }
    }
    this.foundationStack.push({ previous: this.activeFoundation, snapshot });
    this.activeFoundation = name;
    return true;
  }

  truthTokenValue(token) {
    if (typeof token !== 'string') return null;
    if (isNum(token)) return this.clamp(parseFloat(token));
    if (this.symbolProb.has(token)) return this.getSymbolProb(token);
    return null;
  }

  truthTableOp(rows, previous) {
    return (...args) => {
      for (const row of rows) {
        if (row.inputs.length !== args.length) continue;
        let ok = true;
        for (let i = 0; i < args.length; i++) {
          if (row.inputs[i] === '_') continue;
          const expected = this.truthTokenValue(row.inputs[i]);
          if (expected === null || Math.abs(args[i] - expected) >= 1e-9) {
            ok = false;
            break;
          }
        }
        if (ok) {
          const out = this.truthTokenValue(row.output);
          return out === null ? this.lo : out;
        }
      }
      return previous ? previous(...args) : this.lo;
    };
  }

  exitFoundation() {
    const frame = this.foundationStack.pop();
    if (!frame) return;
    for (let i = frame.snapshot.length - 1; i >= 0; i--) {
      const [opName, prev] = frame.snapshot[i];
      if (prev === null) this.ops.delete(opName);
      else this.ops.set(opName, prev);
    }
    this.activeFoundation = frame.previous;
  }

  defineForm(head, rhs) {
    if (head === 'range' && rhs.length === 2 && isNum(rhs[0]) && isNum(rhs[1])) {
      this.requireRule('(eval (range: low high))');
      this.lo = parseFloat(rhs[0]);
      this.hi = parseFloat(rhs[1]);
      this.reinitOps();
      return 1;
    }
    if (head === 'valence' && rhs.length === 1 && isNum(rhs[0])) {
      this.requireRule('(eval (valence: levels))');
      this.valence = parseInt(rhs[0], 10);
      return 1;
    }
    if (rhs.length === 3 && rhs[0] === head && rhs[1] === 'is' && rhs[2] === head) {
      this.requireRule('(eval (name: name is name))');
      this.terms.add(head);
      return 1;
    }
    if (rhs.length === 2 && typeof rhs[0] === 'string' && rhs[1] === head && /^[A-Z]/.test(rhs[0])) {
      this.requireRule('(eval (name: type-name name))');
      this.terms.add(head);
      this.setType(head, rhs[0]);
      return 1;
    }
    if (rhs.length === 2 && Array.isArray(rhs[0]) && rhs[1] === head) {
      this.requireRule('(eval (name: type-expression name))');
      this.terms.add(head);
      this.setType(head, rhs[0]);
      this.evalNode(rhs[0]);
      return 1;
    }
    if (rhs.length === 1 && Array.isArray(rhs[0]) && !this.isOperatorHead(head)) {
      this.requireRule('(eval (name: type-expression))');
      this.terms.add(head);
      this.setType(head, rhs[0]);
      this.evalNode(rhs[0]);
      return 1;
    }
    if (rhs.length === 1 && isNum(rhs[0])) {
      this.requireRule('(eval (name: number))');
      this.setSymbolProb(head, parseFloat(rhs[0]));
      return this.toNum(rhs[0]);
    }
    if (this.isOperatorHead(head)) {
      if (rhs.length === 1 && typeof rhs[0] === 'string' && this.ops.has(rhs[0])) {
        this.requireRule('(eval (operator: aggregator))');
        const target = this.ops.get(rhs[0]);
        this.ops.set(head, (...xs) => target(...xs));
        return 1;
      }
      if (rhs.length === 2 && typeof rhs[0] === 'string' && typeof rhs[1] === 'string') {
        this.requireRule('(eval (operator: outer inner))');
        const outer = this.ops.get(rhs[0]);
        const inner = this.ops.get(rhs[1]);
        this.ops.set(head, (...xs) => this.clamp(outer(inner(...xs))));
        return 1;
      }
      if (['and', 'or', 'both', 'neither'].includes(head) && rhs.length === 1 && typeof rhs[0] === 'string') {
        this.requireRule('(eval (operator: aggregator))');
        this.ops.set(head, this.aggregator(rhs[0]));
        return 1;
      }
    }
    if (rhs.length === 3 && rhs[0] === 'lambda' && Array.isArray(rhs[1])) {
      this.requireRule('(eval (name: lambda binding body))');
      const parsed = parseBinding(rhs[1]);
      if (parsed) {
        const previous = this.getType(parsed.paramName);
        this.setType(parsed.paramName, parsed.paramType);
        this.lambdas.set(head, { param: parsed.paramName, paramType: parsed.paramType, body: rhs[2] });
        const bodyType = this.getType(rhs[2]) || (typeof rhs[2] === 'string' ? rhs[2] : keyOf(rhs[2]));
        if (previous === null) this.types.delete(parsed.paramName);
        else this.setType(parsed.paramName, previous);
        this.terms.add(head);
        this.setType(head, `(Pi (${keyOf(parsed.paramType)} ${parsed.paramName}) ${bodyType})`);
        return 1;
      }
    }
    if (rhs.length === 1 && typeof rhs[0] === 'string') {
      this.setSymbolProb(head, this.getSymbolProb(rhs[0]));
      return this.getSymbolProb(head);
    }
    return 0;
  }

  isOperatorHead(head) {
    return ['=', '!=', 'and', 'or', 'not', 'is', '?:', 'both', 'neither'].includes(head) || /[=!]/.test(head);
  }

  aggregator(name) {
    const lo = this.lo;
    if (name === 'avg') return (...xs) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : lo;
    if (name === 'min') return (...xs) => xs.length ? Math.min(...xs) : lo;
    if (name === 'max') return (...xs) => xs.length ? Math.max(...xs) : lo;
    if (name === 'product' || name === 'prod') return (...xs) => xs.reduce((a, b) => a * b, 1);
    if (name === 'probabilistic_sum' || name === 'ps') {
      return (...xs) => 1 - xs.reduce((a, b) => a * (1 - b), 1);
    }
    throw new Error(`unknown aggregator ${name}`);
  }

  evalNode(rawNode) {
    if (typeof rawNode === 'string') {
      if (isNum(rawNode)) {
        this.requireRule('(eval numeric-literal)');
        return this.toNum(rawNode);
      }
      this.requireRule('(eval symbol)');
      return this.getSymbolProb(rawNode);
    }

    let node = desugarHoas(rawNode);
    if (!Array.isArray(node) || node.length === 0) return 0;

    if (typeof node[0] === 'string' && node[0].endsWith(':')) {
      return this.defineForm(node[0].slice(0, -1), node.slice(1));
    }

    if (node[0] === 'namespace') {
      this.requireRule('(eval (namespace name))');
      return 1;
    }

    if (node[0] === 'import') {
      this.requireRule(node.length === 4 ? '(eval (import path as alias))' : '(eval (import path))');
      return 1;
    }

    if (node[0] === 'domain') {
      this.requireRule('(eval (domain name request))');
      if (node[1] === 'automatic-sequences') {
        for (const request of node.slice(2)) {
          if (Array.isArray(request) && request[0] === 'theorem' && typeof request[1] === 'string') {
            this.setSymbolProb(request[1], this.hi);
          }
        }
      }
      return 1;
    }

    if (['inductive', 'coinductive', 'constructor', 'define', 'relation', 'mode', 'world'].includes(node[0])) {
      return 1;
    }

    if (node.length === 4 && node[1] === 'has' && node[2] === 'probability' && isNum(node[3])) {
      this.requireRule('(eval ((expression) has probability number))');
      this.setExprProb(node[0], parseFloat(node[3]));
      return this.toNum(node[3]);
    }

    if (node.length === 3 && node[0] === 'range' && isNum(node[1]) && isNum(node[2])) {
      this.requireRule('(eval (range low high))');
      this.lo = parseFloat(node[1]);
      this.hi = parseFloat(node[2]);
      this.reinitOps();
      return 1;
    }

    if (node.length === 2 && node[0] === 'valence' && isNum(node[1])) {
      this.requireRule('(eval (valence levels))');
      this.valence = parseInt(node[1], 10);
      return 1;
    }

    if (node[0] === '?') {
      this.requireRule('(eval (? expression))');
      const target = node.length === 2 ? node[1] : node.slice(1);
      const value = this.evalNode(target);
      if (value && typeof value === 'object' && value.query) return value;
      if (value && typeof value === 'object' && Object.hasOwn(value, 'term')) {
        return { query: true, value: keyOf(value.term), typeQuery: true };
      }
      return { query: true, value: this.clamp(value) };
    }

    if (node.length === 4 && node[0] === 'subst' && typeof node[2] === 'string') {
      this.requireRule('(eval (subst term variable replacement))');
      return { term: this.evalTermNode(node) };
    }

    if (node.length === 2 && node[0] === 'whnf') {
      this.requireRule('(eval (whnf expression))');
      return { term: this.whnfTerm(node[1]) };
    }

    if (node.length === 2 && (node[0] === 'nf' || node[0] === 'normal-form')) {
      this.requireRule(node[0] === 'nf' ? '(eval (nf expression))' : '(eval (normal-form expression))');
      return { term: this.flattenNeutralApplies(this.normalizeTerm(node[1])) };
    }

    if (node.length === 4 && node[0] === 'fresh' && node[2] === 'in' && typeof node[1] === 'string') {
      this.requireRule('(eval (fresh variable in body))');
      this.terms.add(node[1]);
      return this.evalNode(node[3]);
    }

    if (node.length === 3 && typeof node[1] === 'string' && ['+', '-', '*', '/'].includes(node[1])) {
      this.requireRule(`(eval (left ${node[1]} right))`);
      return this.ops.get(node[1])(this.evalArith(node[0]), this.evalArith(node[2]));
    }

    if (node.length === 3 && typeof node[1] === 'string' && ['<', '<='].includes(node[1])) {
      this.requireRule(`(eval (left ${node[1]} right))`);
      return this.clamp(this.ops.get(node[1])(this.evalArith(node[0]), this.evalArith(node[2])));
    }

    if (node.length === 3 && typeof node[1] === 'string' && ['and', 'or', 'both', 'neither'].includes(node[1])) {
      this.requireRule(`(eval (a ${node[1]} b))`);
      return this.clamp(this.ops.get(node[1])(this.evalNode(node[0]), this.evalNode(node[2])));
    }

    if (node.length >= 4 && (node[0] === 'both' || node[0] === 'neither')) {
      const separator = node[0] === 'both' ? 'and' : 'nor';
      let valid = node.length % 2 === 0;
      for (let i = 2; valid && i < node.length; i += 2) valid = node[i] === separator;
      if (valid) {
        this.requireRule(node[0] === 'both' ? '(eval (both a and b))' : '(eval (neither a nor b))');
        const values = [];
        for (let i = 1; i < node.length; i += 2) values.push(this.evalNode(node[i]));
        return this.clamp(this.ops.get(node[0])(...values));
      }
    }

    if (node.length === 3 && typeof node[1] === 'string' && ['=', '!='].includes(node[1])) {
      this.requireRule(`(eval (left ${node[1]} right))`);
      return this.evalEqualityNode(node[0], node[1], node[2]);
    }

    if (node.length === 2 && node[0] === 'Type') {
      this.requireRule('(eval (Type level))');
      const level = this.universeLevel(node[1]);
      if (level !== null) this.setType(node, ['Type', String(level + 1)]);
      return level === null ? 0 : 1;
    }

    if (node.length === 1 && node[0] === 'Prop') {
      this.requireRule('(eval (Prop))');
      this.setType(['Prop'], ['Type', '1']);
      return 1;
    }

    if (node.length === 3 && node[0] === 'Pi') {
      this.requireRule('(eval (Pi binding body))');
      const parsed = parseBinding(node[1]);
      if (parsed) {
        this.terms.add(parsed.paramName);
        this.setType(parsed.paramName, parsed.paramType);
        this.setType(node, ['Type', '0']);
      }
      return 1;
    }

    if (node.length === 3 && node[0] === 'lambda') {
      this.requireRule('(eval (lambda binding body))');
      const parsed = parseBinding(node[1]);
      if (parsed) {
        this.terms.add(parsed.paramName);
        this.setType(parsed.paramName, parsed.paramType);
      }
      return 1;
    }

    if (node.length === 3 && node[0] === 'apply') {
      this.requireRule('(eval (apply function argument))');
      return this.evalApply(node[1], node[2]);
    }

    if (node.length === 3 && node[0] === 'type' && node[1] === 'of') {
      this.requireRule('(eval (type of expression))');
      return { query: true, value: this.inferType(node[2]) ?? 'unknown', typeQuery: true };
    }

    if (node.length === 3 && node[1] === 'of') {
      this.requireRule('(eval (expression of type))');
      const actual = this.inferType(node[0]);
      return actual === (typeof node[2] === 'string' ? node[2] : keyOf(node[2])) ? this.hi : this.lo;
    }

    const [head, ...args] = node;
    if (typeof head === 'string' && ['=', '!='].includes(head) && args.length === 2) {
      this.requireRule(`(eval (${head} left right))`);
      return this.evalEqualityNode(args[0], head, args[1]);
    }

    if (typeof head === 'string' && this.ops.has(head)) {
      if (head === 'not') this.requireRule('(eval (not value))');
      if (['and', 'or', 'both', 'neither'].includes(head)) this.requireRule(`(eval (${head} a b))`);
      const values = args.map(arg => this.evalNode(arg));
      return this.clamp(this.ops.get(head)(...values));
    }

    if (typeof head === 'string' && args.length >= 1 && this.lambdas.has(head)) {
      this.requireRule('(eval (function argument))');
      let result = subst(this.lambdas.get(head).body, this.lambdas.get(head).param, args[0]);
      if (args.length > 1) result = [result, ...args.slice(1)];
      return this.evalReducedTerm(result);
    }

    if (Array.isArray(head) && head.length === 3 && head[0] === 'lambda' && args.length >= 1) {
      this.requireRule('(eval (function argument))');
      const parsed = parseBinding(head[1]);
      if (parsed) {
        let result = subst(head[2], parsed.paramName, args[0]);
        if (args.length > 1) result = [result, ...args.slice(1)];
        return this.evalReducedTerm(result);
      }
    }

    return 0;
  }

  evalArith(node) {
    if (typeof node === 'string' && isNum(node)) return parseFloat(node);
    const value = this.evalNode(node);
    if (value && typeof value === 'object' && Object.hasOwn(value, 'term')) {
      return this.evalArith(value.term);
    }
    return value;
  }

  evalApply(fn, arg) {
    if (Array.isArray(fn) && fn.length === 3 && fn[0] === 'lambda') {
      const parsed = parseBinding(fn[1]);
      if (parsed) return this.evalReducedTerm(subst(fn[2], parsed.paramName, arg));
    }
    if (typeof fn === 'string' && this.lambdas.has(fn)) {
      const lambda = this.lambdas.get(fn);
      return this.evalReducedTerm(subst(lambda.body, lambda.param, arg));
    }
    const fVal = this.evalNode(fn);
    this.evalNode(arg);
    return typeof fVal === 'number' ? fVal : 0;
  }

  evalTermNode(node) {
    if (!Array.isArray(node)) return node;
    if (node.length === 4 && node[0] === 'subst' && typeof node[2] === 'string') {
      return this.evalTermNode(subst(this.evalTermNode(node[1]), node[2], this.evalTermNode(node[3])));
    }
    if (node.length === 3 && node[0] === 'apply') {
      const fn = node[1];
      const arg = this.evalTermNode(node[2]);
      if (Array.isArray(fn) && fn.length === 3 && fn[0] === 'lambda') {
        const parsed = parseBinding(fn[1]);
        if (parsed) return this.evalTermNode(subst(fn[2], parsed.paramName, arg));
      }
      if (typeof fn === 'string' && this.lambdas.has(fn)) {
        const lambda = this.lambdas.get(fn);
        return this.evalTermNode(subst(lambda.body, lambda.param, arg));
      }
    }
    return node;
  }

  evalReducedTerm(reduced) {
    const term = this.normalizeTerm(reduced);
    if (this.hasUnresolvedFreeVariables(term)) return { term };
    return this.evalNode(term);
  }

  normalizeTerm(rawNode) {
    const node = desugarHoas(rawNode);
    if (!Array.isArray(node) || node.length === 0) return node;

    if (node.length === 4 && node[0] === 'subst' && typeof node[2] === 'string') {
      return this.normalizeTerm(subst(this.normalizeTerm(node[1]), node[2], this.normalizeTerm(node[3])));
    }

    if (node.length === 3 && node[0] === 'apply') {
      const fn = this.normalizeTerm(node[1]);
      const arg = this.normalizeTerm(node[2]);
      if (Array.isArray(fn) && fn.length === 3 && fn[0] === 'lambda') {
        const parsed = parseBinding(fn[1]);
        if (parsed) return this.normalizeTerm(subst(fn[2], parsed.paramName, arg));
      }
      if (typeof fn === 'string' && this.lambdas.has(fn)) {
        const lambda = this.lambdas.get(fn);
        return this.normalizeTerm(subst(lambda.body, lambda.param, arg));
      }
      return ['apply', fn, arg];
    }

    if (node.length === 3 && node[0] === 'lambda') {
      return ['lambda', this.normalizeTerm(node[1]), this.normalizeTerm(node[2])];
    }

    const [head, ...args] = node;
    if (Array.isArray(head) && head.length === 3 && head[0] === 'lambda' && args.length >= 1) {
      const parsed = parseBinding(head[1]);
      if (parsed) {
        const first = this.normalizeTerm(args[0]);
        const reduced = subst(head[2], parsed.paramName, first);
        return this.normalizeTerm(args.length === 1 ? reduced : [reduced, ...args.slice(1)]);
      }
    }
    if (typeof head === 'string' && args.length >= 1 && this.lambdas.has(head)) {
      const lambda = this.lambdas.get(head);
      const first = this.normalizeTerm(args[0]);
      const reduced = subst(lambda.body, lambda.param, first);
      return this.normalizeTerm(args.length === 1 ? reduced : [reduced, ...args.slice(1)]);
    }
    return node.map(child => this.normalizeTerm(child));
  }

  whnfTerm(rawNode) {
    const node = desugarHoas(rawNode);
    if (!Array.isArray(node) || node.length === 0) return node;

    const spineArgs = [];
    let head = node;
    while (Array.isArray(head) && head.length === 3 && head[0] === 'apply') {
      spineArgs.unshift(head[2]);
      head = head[1];
    }

    while (spineArgs.length > 0) {
      if (Array.isArray(head) && head.length === 3 && head[0] === 'lambda') {
        const parsed = parseBinding(head[1]);
        if (!parsed) break;
        head = subst(head[2], parsed.paramName, spineArgs.shift());
        continue;
      }
      if (typeof head === 'string' && this.lambdas.has(head)) {
        const lambda = this.lambdas.get(head);
        head = subst(lambda.body, lambda.param, spineArgs.shift());
        continue;
      }
      break;
    }

    let out = head;
    for (const arg of spineArgs) out = ['apply', out, arg];
    return out;
  }

  flattenNeutralApplies(node) {
    if (!Array.isArray(node)) return node;
    const flattened = node.map(child => this.flattenNeutralApplies(child));
    if (flattened.length === 3 && flattened[0] === 'apply' && typeof flattened[1] === 'string' && !this.lambdas.has(flattened[1])) {
      return [flattened[1], flattened[2]];
    }
    return flattened;
  }

  evalEqualityNode(left, op, right) {
    const direct = this.lookupAssignedInfix(op, left, right);
    if (direct !== null) return this.clamp(direct);
    const leftTerm = this.normalizeTerm(left);
    const rightTerm = this.normalizeTerm(right);
    if (!isStructurallySame(left, leftTerm) || !isStructurallySame(right, rightTerm)) {
      const normalized = this.lookupAssignedInfix(op, leftTerm, rightTerm);
      if (normalized !== null) return this.clamp(normalized);
    }
    const eq = this.equalityTruthValue(left, right, leftTerm, rightTerm);
    return op === '=' ? this.clamp(eq) : this.clamp(this.ops.get('not')(eq));
  }

  equalityTruthValue(left, right, leftTerm, rightTerm) {
    const assigned = this.lookupAssignedInfix('=', left, right);
    if (assigned !== null) return this.clamp(assigned);
    if (!isStructurallySame(left, leftTerm) || !isStructurallySame(right, rightTerm)) {
      const normalized = this.lookupAssignedInfix('=', leftTerm, rightTerm);
      if (normalized !== null) return this.clamp(normalized);
    }
    if (isStructurallySame(leftTerm, rightTerm)) return this.hi;
    const leftNum = this.tryEvalNumeric(leftTerm);
    const rightNum = this.tryEvalNumeric(rightTerm);
    if (leftNum !== null && rightNum !== null) {
      return decRound(leftNum) === decRound(rightNum) ? this.hi : this.lo;
    }
    return this.lo;
  }

  lookupAssignedInfix(op, left, right) {
    for (const expr of [[op, left, right], [left, op, right]]) {
      const k = keyOf(expr);
      if (this.assign.has(k)) return this.assign.get(k);
    }
    return null;
  }

  tryEvalNumeric(node) {
    const term = this.normalizeTerm(node);
    if (typeof term === 'string') {
      if (isNum(term)) return parseFloat(term);
      return this.symbolProb.has(term) ? this.symbolProb.get(term) : null;
    }
    if (!Array.isArray(term) || term.length === 0) return null;
    if (term.length === 3 && typeof term[1] === 'string' && ['+', '-', '*', '/'].includes(term[1])) {
      const left = this.tryEvalNumeric(term[0]);
      const right = this.tryEvalNumeric(term[2]);
      return left === null || right === null ? null : this.ops.get(term[1])(left, right);
    }
    if (term.length === 3 && typeof term[1] === 'string' && ['and', 'or', 'both', 'neither'].includes(term[1])) {
      const left = this.tryEvalNumeric(term[0]);
      const right = this.tryEvalNumeric(term[2]);
      return left === null || right === null ? null : this.clamp(this.ops.get(term[1])(left, right));
    }
    const [head, ...args] = term;
    if (typeof head === 'string' && this.ops.has(head) && head !== '=' && head !== '!=') {
      const vals = args.map(arg => this.tryEvalNumeric(arg));
      return vals.some(v => v === null) ? null : this.clamp(this.ops.get(head)(...vals));
    }
    return null;
  }

  inferType(node) {
    const recorded = this.getType(desugarHoas(node));
    if (recorded) return recorded;
    if (Array.isArray(node) && node.length === 2 && node[0] === 'Type') {
      const level = this.universeLevel(node[1]);
      return level === null ? null : `(Type ${level + 1})`;
    }
    return null;
  }

  universeLevel(token) {
    if (typeof token !== 'string' || !/^(0|[1-9]\d*)$/.test(token)) return null;
    const level = Number(token);
    return Number.isSafeInteger(level) ? level : null;
  }

  freeVariables(expr, bound = new Set()) {
    if (typeof expr === 'string') {
      if (!isNum(expr) && !['lambda', 'Pi', 'apply', 'type', 'of', 'and', 'or', 'not', '=', '!='].includes(expr) && !bound.has(expr)) {
        return new Set([expr]);
      }
      return new Set();
    }
    if (!Array.isArray(expr)) return new Set();
    if (expr.length === 3 && (expr[0] === 'lambda' || expr[0] === 'Pi')) {
      const parsed = parseBinding(expr[1]);
      const next = new Set(bound);
      if (parsed) next.add(parsed.paramName);
      return this.freeVariables(expr[2], next);
    }
    const out = new Set();
    for (const child of expr) {
      for (const name of this.freeVariables(child, bound)) out.add(name);
    }
    return out;
  }

  hasUnresolvedFreeVariables(expr) {
    for (const name of this.freeVariables(expr)) {
      if (!this.symbolProb.has(name) && !this.terms.has(name) && !this.types.has(name) && !this.lambdas.has(name) && !this.ops.has(name)) {
        return true;
      }
    }
    return false;
  }
}

function assertResultsEqual(actual, expected, label) {
  assert.strictEqual(actual.length, expected.length, `${label}: result count mismatch`);
  for (let i = 0; i < expected.length; i++) {
    if (typeof expected[i] === 'number') {
      assert.strictEqual(typeof actual[i], 'number', `${label}[${i}] expected number`);
      assert.ok(Math.abs(actual[i] - expected[i]) < 1e-9, `${label}[${i}] expected ${expected[i]}, got ${actual[i]}`);
    } else {
      assert.strictEqual(actual[i], expected[i], `${label}[${i}]`);
    }
  }
}

describe('self evaluator', () => {
  it('is importable as a standard library file', () => {
    const out = evaluateFile(evaluatorPath);
    assert.deepStrictEqual(out.diagnostics, []);
  });

  it('declares the required built-in operator rules as links', () => {
    const forms = evaluatorForms();
    const rules = evalRulePatterns(forms);
    for (const pattern of REQUIRED_EVAL_RULES) {
      assert.ok(rules.has(pattern), `missing rule ${pattern}`);
    }

    const operators = builtInOperators(forms);
    for (const operator of REQUIRED_OPERATORS) {
      assert.ok(operators.has(operator), `missing built-in operator ${operator}`);
    }
  });

  it('declares the Phase 2-9 foundation and proof clauses as links', () => {
    const forms = evaluatorForms();
    const rules = rulePatterns(forms);
    for (const pattern of REQUIRED_SURFACE_RULES) {
      assert.ok(rules.has(pattern), `missing surface rule ${pattern}`);
    }
  });

  for (const file of readdirSync(selfCorpusDir).filter(f => f.endsWith('.lino') && f !== 'expected.lino').sort()) {
    it(`replays self corpus ${file} like the host evaluator`, () => {
      const source = readFileSync(join(selfCorpusDir, file), 'utf8');
      const encoded = new EncodedEvaluator(evalRulePatterns(evaluatorForms()));
      assertResultsEqual(encoded.run(source), run(source), file);
    });
  }

  for (const file of readdirSync(examplesDir).filter(f => f.endsWith('.lino') && f !== 'expected.lino').sort()) {
    it(`replays example ${file} like the host evaluator`, () => {
      const source = readFileSync(join(examplesDir, file), 'utf8');
      const encoded = new EncodedEvaluator(evalRulePatterns(evaluatorForms()));
      assertResultsEqual(encoded.run(source), run(source), file);
    });
  }
});
