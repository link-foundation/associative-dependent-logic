// Tests for `lib/self/types.lino` (issue #86).
//
// The type file is data: it records the host bidirectional checker as
// `(rule ...)` links. These tests keep that data parseable, require the
// `synth`/`check` rule surface, and run a small rule-backed checker against
// representative acceptance and rejection cases from the host checker.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import {
  Env,
  evalNode,
  evaluateFile,
  isConvertible,
  isNum,
  isStructurallySame,
  keyOf,
  parseBinding,
  parseLino,
  parseOne,
  subst,
  synth,
  check,
  tokenizeOne,
} from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const typesPath = join(repoRoot, 'lib', 'self', 'types.lino');

const REQUIRED_SYNTH_RULES = [
  '(synth symbol)',
  '(synth numeric-literal)',
  '(synth (Type level))',
  '(synth (Prop))',
  '(synth (Pi binding body))',
  '(synth (forall type-variable body))',
  '(synth (lambda binding body))',
  '(synth (apply function argument))',
  '(synth (subst term variable replacement))',
  '(synth (type of expression))',
  '(synth (expression of expected-type))',
  '(synth recorded-expression)',
];

const REQUIRED_CHECK_RULES = [
  '(check expression expected-type)',
  '(check numeric-literal expected-type)',
  '(check (lambda (T x) body) (Pi (T x) U))',
  '(check (lambda (domain variable) body) (Pi (expected-domain expected-variable) codomain))',
  '(check (lambda binding body) non-pi-type)',
  '(check expression expected-type by-synthesis)',
];

const REQUIRED_DIAGNOSTICS = ['E020', 'E021', 'E022', 'E023', 'E024'];

function parseForms(source) {
  return parseLino(source).map(link => parseOne(tokenizeOne(link)));
}

function typeForms() {
  return parseForms(readFileSync(typesPath, 'utf8'));
}

function rulePatterns(forms, kind) {
  const patterns = new Set();
  for (const form of forms) {
    if (
      Array.isArray(form) &&
      form[0] === 'rule' &&
      Array.isArray(form[1]) &&
      form[1][0] === kind
    ) {
      patterns.add(keyOf(form[1]));
    }
  }
  return patterns;
}

function diagnosticCodes(forms) {
  const codes = new Set();
  for (const form of forms) {
    if (
      Array.isArray(form) &&
      form[0] === 'rule' &&
      Array.isArray(form[1]) &&
      form[1][0] === 'diagnostic'
    ) {
      for (const child of form.slice(2)) {
        if (Array.isArray(child) && child[0] === 'emits' && typeof child[1] === 'string') {
          codes.add(child[1]);
        }
      }
    }
  }
  return codes;
}

function parseTermInput(term) {
  if (Array.isArray(term)) return term;
  if (typeof term === 'string' && term.trim().startsWith('(')) {
    return parseOne(tokenizeOne(term.trim()));
  }
  return term;
}

function parseUniverseLevelToken(token) {
  if (typeof token !== 'string' || !/^(0|[1-9]\d*)$/.test(token)) return null;
  const level = Number(token);
  return Number.isSafeInteger(level) ? level : null;
}

function universeTypeKey(node) {
  if (!Array.isArray(node) || node.length !== 2 || node[0] !== 'Type') return null;
  const level = parseUniverseLevelToken(node[1]);
  return level === null ? null : `(Type ${level + 1})`;
}

function inferTypeKey(node, env) {
  const recorded = env.getType(node);
  if (recorded) return recorded;
  const universeType = universeTypeKey(node);
  if (universeType) {
    env.setType(node, universeType);
    return universeType;
  }
  return null;
}

function typeKeyOf(typeNode) {
  if (typeNode === null || typeNode === undefined) return null;
  return typeof typeNode === 'string' ? typeNode : keyOf(typeNode);
}

function parseTypeKeyToNode(typeKey) {
  if (typeof typeKey !== 'string') return typeKey;
  const trimmed = typeKey.trim();
  if (trimmed.startsWith('(')) return parseOne(tokenizeOne(trimmed));
  return typeKey;
}

function isForallNode(node) {
  return Array.isArray(node) &&
    node.length === 3 &&
    node[0] === 'forall' &&
    typeof node[1] === 'string';
}

function expandForall(node) {
  return isForallNode(node) ? ['Pi', ['Type', node[1]], node[2]] : node;
}

function snapshotTypeBinding(env, name) {
  return {
    name,
    hadTerm: env.terms.has(name),
    hadType: env.types.has(name),
    previousType: env.types.get(name),
  };
}

function extendTypeBinding(env, name, typeKey) {
  env.terms.add(name);
  env.types.set(name, typeKey);
}

function restoreTypeBinding(env, snap) {
  if (!snap.hadTerm) env.terms.delete(snap.name);
  if (snap.hadType) env.types.set(snap.name, snap.previousType);
  else env.types.delete(snap.name);
}

function setupNaturalEnv() {
  const env = new Env();
  evalNode(['Type:', 'Type', 'Type'], env);
  evalNode(['Natural:', ['Type', '0'], 'Natural'], env);
  evalNode(['Boolean:', 'Type', 'Boolean'], env);
  evalNode(['zero:', 'Natural', 'zero'], env);
  evalNode(['identity:', 'lambda', ['Natural', 'x'], 'x'], env);
  evalNode(['succ:', ['Pi', ['Natural', 'n'], 'Natural']], env);
  return env;
}

class EncodedTypeChecker {
  constructor(forms) {
    this.synthRules = rulePatterns(forms, 'synth');
    this.checkRules = rulePatterns(forms, 'check');
    this.diagnosticRules = diagnosticCodes(forms);
  }

  requireSynth(pattern) {
    assert.ok(this.synthRules.has(pattern), `missing encoded rule ${pattern}`);
  }

  requireCheck(pattern) {
    assert.ok(this.checkRules.has(pattern), `missing encoded rule ${pattern}`);
  }

  diagnostic(code) {
    assert.ok(this.diagnosticRules.has(code), `missing encoded diagnostic ${code}`);
    return { code };
  }

  typesAgree(a, b, env) {
    if (a === null || b === null) return false;
    const left = expandForall(a);
    const right = expandForall(b);
    if (isStructurallySame(left, right)) return true;
    try {
      return isConvertible(left, right, env);
    } catch (_) {
      return false;
    }
  }

  synthLeaf(term, env) {
    this.requireSynth(isNum(term) ? '(synth numeric-literal)' : '(synth symbol)');
    if (isNum(term)) return null;
    const recorded = inferTypeKey(term, env);
    if (recorded) return parseTypeKeyToNode(recorded);
    const resolved = env._resolveQualified(term);
    if (resolved !== term) {
      const fromAlias = env.types.get(resolved);
      if (fromAlias) return parseTypeKeyToNode(fromAlias);
    }
    return null;
  }

  synthApply(node, env, diagnostics) {
    this.requireSynth('(synth (apply function argument))');
    const fnSynth = this.synth(node[1], env);
    diagnostics.push(...fnSynth.diagnostics);
    if (!fnSynth.type) {
      diagnostics.push(this.diagnostic('E020'));
      return null;
    }

    const fnType = expandForall(fnSynth.type);
    if (!Array.isArray(fnType) || fnType.length !== 3 || fnType[0] !== 'Pi') {
      diagnostics.push(this.diagnostic('E022'));
      return null;
    }

    const parsed = parseBinding(fnType[1]);
    if (!parsed) {
      diagnostics.push(this.diagnostic('E022'));
      return null;
    }

    const argCheck = this.check(node[2], parsed.paramType, env);
    diagnostics.push(...argCheck.diagnostics);
    if (!argCheck.ok) return null;
    return subst(fnType[2], parsed.paramName, node[2]);
  }

  synthLambda(node, env, diagnostics) {
    this.requireSynth('(synth (lambda binding body))');
    const parsed = parseBinding(node[1]);
    if (!parsed) {
      diagnostics.push(this.diagnostic('E024'));
      return null;
    }

    const snap = snapshotTypeBinding(env, parsed.paramName);
    extendTypeBinding(env, parsed.paramName, typeKeyOf(parsed.paramType));
    let bodySynth;
    try {
      bodySynth = this.synth(node[2], env);
    } finally {
      restoreTypeBinding(env, snap);
    }
    diagnostics.push(...bodySynth.diagnostics);
    if (!bodySynth.type) return null;
    return ['Pi', [parsed.paramType, parsed.paramName], bodySynth.type];
  }

  synth(term, env) {
    const diagnostics = [];
    const node = parseTermInput(term);

    if (typeof node === 'string') {
      const type = this.synthLeaf(node, env);
      if (!type && !isNum(node)) diagnostics.push(this.diagnostic('E020'));
      return { type, diagnostics };
    }

    if (!Array.isArray(node)) {
      diagnostics.push(this.diagnostic('E020'));
      return { type: null, diagnostics };
    }

    if (node.length === 2 && node[0] === 'Type') {
      this.requireSynth('(synth (Type level))');
      const universeType = universeTypeKey(node);
      if (universeType) return { type: parseTypeKeyToNode(universeType), diagnostics };
      diagnostics.push(this.diagnostic('E020'));
      return { type: null, diagnostics };
    }

    if (node.length === 1 && node[0] === 'Prop') {
      this.requireSynth('(synth (Prop))');
      return { type: ['Type', '1'], diagnostics };
    }

    if (node.length === 3 && node[0] === 'Pi') {
      this.requireSynth('(synth (Pi binding body))');
      if (!parseBinding(node[1])) {
        diagnostics.push(this.diagnostic('E024'));
        return { type: null, diagnostics };
      }
      return { type: ['Type', '0'], diagnostics };
    }

    if (isForallNode(node)) {
      this.requireSynth('(synth (forall type-variable body))');
      return this.synth(expandForall(node), env);
    }

    if (node.length === 3 && node[0] === 'lambda') {
      const type = this.synthLambda(node, env, diagnostics);
      return { type, diagnostics };
    }

    if (node.length === 3 && node[0] === 'apply') {
      const type = this.synthApply(node, env, diagnostics);
      return { type, diagnostics };
    }

    if (node.length === 4 && node[0] === 'subst' && typeof node[2] === 'string') {
      this.requireSynth('(synth (subst term variable replacement))');
      return this.synth(subst(parseTermInput(node[1]), node[2], parseTermInput(node[3])), env);
    }

    if (node.length === 3 && node[0] === 'type' && node[1] === 'of') {
      this.requireSynth('(synth (type of expression))');
      const inner = this.synth(node[2], env);
      diagnostics.push(...inner.diagnostics);
      if (inner.type) return { type: ['Type', '0'], diagnostics };
      diagnostics.push(this.diagnostic('E020'));
      return { type: null, diagnostics };
    }

    if (node.length === 3 && node[1] === 'of') {
      this.requireSynth('(synth (expression of expected-type))');
      const result = this.check(node[0], node[2], env);
      diagnostics.push(...result.diagnostics);
      return { type: result.ok ? ['Type', '0'] : null, diagnostics };
    }

    this.requireSynth('(synth recorded-expression)');
    const recorded = inferTypeKey(node, env);
    if (recorded) return { type: parseTypeKeyToNode(recorded), diagnostics };

    diagnostics.push(this.diagnostic('E020'));
    return { type: null, diagnostics };
  }

  check(term, expectedType, env) {
    this.requireCheck('(check expression expected-type)');
    const diagnostics = [];
    const node = parseTermInput(term);
    let expectedNode = parseTermInput(expectedType);

    if (isForallNode(expectedNode)) expectedNode = expandForall(expectedNode);

    if (
      Array.isArray(node) && node.length === 3 && node[0] === 'lambda' &&
      Array.isArray(expectedNode) && expectedNode.length === 3 && expectedNode[0] === 'Pi'
    ) {
      this.requireCheck('(check (lambda (domain variable) body) (Pi (expected-domain expected-variable) codomain))');
      const lambdaParsed = parseBinding(node[1]);
      const piParsed = parseBinding(expectedNode[1]);
      if (lambdaParsed && piParsed) {
        if (!this.typesAgree(parseTermInput(lambdaParsed.paramType), parseTermInput(piParsed.paramType), env)) {
          diagnostics.push(this.diagnostic('E021'));
          return { ok: false, diagnostics };
        }

        const codomain = subst(expectedNode[2], piParsed.paramName, lambdaParsed.paramName);
        const snap = snapshotTypeBinding(env, lambdaParsed.paramName);
        extendTypeBinding(env, lambdaParsed.paramName, typeKeyOf(lambdaParsed.paramType));
        try {
          const bodyResult = this.check(node[2], codomain, env);
          diagnostics.push(...bodyResult.diagnostics);
          return { ok: bodyResult.ok, diagnostics };
        } finally {
          restoreTypeBinding(env, snap);
        }
      }
    }

    if (
      Array.isArray(node) && node.length === 3 && node[0] === 'lambda' &&
      !(Array.isArray(expectedNode) && expectedNode[0] === 'Pi')
    ) {
      this.requireCheck('(check (lambda binding body) non-pi-type)');
      diagnostics.push(this.diagnostic('E023'));
      return { ok: false, diagnostics };
    }

    if (typeof node === 'string' && isNum(node)) {
      this.requireCheck('(check numeric-literal expected-type)');
      return { ok: true, diagnostics };
    }

    this.requireCheck('(check expression expected-type by-synthesis)');
    const synthResult = this.synth(node, env);
    diagnostics.push(...synthResult.diagnostics);
    if (!synthResult.type) return { ok: false, diagnostics };

    const ok = this.typesAgree(synthResult.type, expectedNode, env);
    if (!ok) diagnostics.push(this.diagnostic('E021'));
    return { ok, diagnostics };
  }
}

function comparableSynth(result) {
  return {
    type: result.type ? keyOf(result.type) : null,
    codes: result.diagnostics.map(d => d.code),
  };
}

function comparableCheck(result) {
  return {
    ok: result.ok,
    codes: result.diagnostics.map(d => d.code),
  };
}

const SYNTH_CASES = [
  { name: 'known term', term: 'zero' },
  { name: 'universe successor', term: ['Type', '0'] },
  { name: 'invalid universe', term: ['Type', 'bad'] },
  { name: 'Prop universe', term: ['Prop'] },
  { name: 'Pi formation', term: ['Pi', ['Natural', 'n'], 'Natural'] },
  { name: 'prenex forall formation', term: ['forall', 'A', ['Pi', ['A', 'x'], 'A']] },
  { name: 'lambda synthesis', term: ['lambda', ['Natural', 'x'], 'x'] },
  { name: 'application synthesis', term: ['apply', 'identity', 'zero'] },
  { name: 'substitution synthesis', term: ['subst', 'x', 'x', 'zero'] },
  { name: 'type-of query', term: ['type', 'of', 'zero'] },
  { name: 'of-membership query', term: ['zero', 'of', 'Natural'] },
  { name: 'unknown symbol rejection', term: 'mystery' },
  { name: 'non-Pi application rejection', term: ['apply', 'zero', 'zero'] },
  { name: 'malformed lambda binder rejection', term: ['lambda', ['x'], 'x'] },
];

const CHECK_CASES = [
  { name: 'recorded term', term: 'zero', expected: 'Natural' },
  { name: 'lambda against Pi', term: ['lambda', ['Natural', 'x'], 'x'], expected: ['Pi', ['Natural', 'x'], 'Natural'] },
  { name: 'numeric literal', term: '0.7', expected: 'Natural' },
  {
    name: 'lambda body in extended context',
    term: ['lambda', ['Natural', 'x'], ['apply', 'succ', 'x']],
    expected: ['Pi', ['Natural', 'n'], 'Natural'],
  },
  {
    name: 'prenex forall expected type',
    term: ['lambda', ['Type', 'A'], ['lambda', ['A', 'x'], 'x']],
    expected: ['forall', 'A', ['Pi', ['A', 'x'], 'A']],
  },
  { name: 'type mismatch rejection', term: 'zero', expected: 'Boolean' },
  { name: 'lambda against non-Pi rejection', term: ['lambda', ['Natural', 'x'], 'x'], expected: 'Natural' },
  { name: 'lambda parameter mismatch rejection', term: ['lambda', ['Boolean', 'x'], 'x'], expected: ['Pi', ['Natural', 'x'], 'Natural'] },
  { name: 'unknown term rejection', term: 'mystery', expected: 'Natural' },
];

describe('self type layer', () => {
  it('is importable as a standard library file', () => {
    const out = evaluateFile(typesPath);
    assert.deepStrictEqual(out.diagnostics, []);
  });

  it('declares the required synth and check rules as links', () => {
    const forms = typeForms();
    const synthRules = rulePatterns(forms, 'synth');
    for (const pattern of REQUIRED_SYNTH_RULES) {
      assert.ok(synthRules.has(pattern), `missing synth rule ${pattern}`);
    }

    const checkRules = rulePatterns(forms, 'check');
    for (const pattern of REQUIRED_CHECK_RULES) {
      assert.ok(checkRules.has(pattern), `missing check rule ${pattern}`);
    }

    const diagnostics = diagnosticCodes(forms);
    for (const code of REQUIRED_DIAGNOSTICS) {
      assert.ok(diagnostics.has(code), `missing diagnostic ${code}`);
    }
  });

  for (const tc of SYNTH_CASES) {
    it(`synth ${tc.name} like the host checker`, () => {
      const encoded = new EncodedTypeChecker(typeForms());
      assert.deepStrictEqual(
        comparableSynth(encoded.synth(tc.term, setupNaturalEnv())),
        comparableSynth(synth(tc.term, setupNaturalEnv())),
      );
    });
  }

  for (const tc of CHECK_CASES) {
    it(`check ${tc.name} like the host checker`, () => {
      const encoded = new EncodedTypeChecker(typeForms());
      assert.deepStrictEqual(
        comparableCheck(encoded.check(tc.term, tc.expected, setupNaturalEnv())),
        comparableCheck(check(tc.term, tc.expected, setupNaturalEnv())),
      );
    });
  }
});
