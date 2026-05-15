// Bidirectional type checker tests for issue #42.
//
// Covers the documented `synth(term, env)` and `check(term, type, env)` API:
// universes, Pi formation, lambda formation/checking, application, of-membership,
// type-of queries, and stable diagnostic codes E020..E024 on failure.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  Env,
  evalNode,
  evaluate,
  synth,
  check,
  keyOf,
} from '../src/rml-links.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const dependentTypesPath = path.resolve(here, '..', '..', 'examples', 'dependent-types.lino');

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

describe('bidirectional checker — synth', () => {
  it('synthesises bare term types from `(name: T name)` declarations', () => {
    const env = setupNaturalEnv();
    const result = synth('zero', env);
    assert.deepStrictEqual(result.diagnostics, []);
    assert.strictEqual(keyOf(result.type), 'Natural');
  });

  it('synthesises (Type N) at universe (Type N+1)', () => {
    const env = new Env();
    const r0 = synth(['Type', '0'], env);
    assert.deepStrictEqual(r0.diagnostics, []);
    assert.strictEqual(keyOf(r0.type), '(Type 1)');
    const r2 = synth(['Type', '2'], env);
    assert.strictEqual(keyOf(r2.type), '(Type 3)');
  });

  it('synthesises (Pi (A x) B) at (Type 0)', () => {
    const env = setupNaturalEnv();
    const result = synth(['Pi', ['Natural', 'n'], 'Natural'], env);
    assert.deepStrictEqual(result.diagnostics, []);
    assert.strictEqual(keyOf(result.type), '(Type 0)');
  });

  it('synthesises a lambda by extending the context with its bound parameter', () => {
    const env = setupNaturalEnv();
    const result = synth(['lambda', ['Natural', 'x'], 'x'], env);
    assert.deepStrictEqual(result.diagnostics, []);
    assert.strictEqual(keyOf(result.type), '(Pi (Natural x) Natural)');
  });

  it('synthesises an application by substituting the argument into the codomain', () => {
    const env = setupNaturalEnv();
    const result = synth(['apply', 'identity', 'zero'], env);
    assert.deepStrictEqual(result.diagnostics, []);
    assert.strictEqual(keyOf(result.type), 'Natural');
  });

  it('synthesises (subst term x repl) by reducing first', () => {
    const env = setupNaturalEnv();
    const result = synth(['subst', 'x', 'x', 'zero'], env);
    assert.deepStrictEqual(result.diagnostics, []);
    assert.strictEqual(keyOf(result.type), 'Natural');
  });

  it('reports E020 when a bare symbol has no recorded type', () => {
    const env = setupNaturalEnv();
    const result = synth('mystery', env);
    assert.strictEqual(result.type, null);
    assert.strictEqual(result.diagnostics.length, 1);
    assert.strictEqual(result.diagnostics[0].code, 'E020');
    assert.match(result.diagnostics[0].message, /symbol `mystery`/);
  });

  it('reports E022 when an application head is not a Pi-type', () => {
    const env = setupNaturalEnv();
    const result = synth(['apply', 'zero', 'zero'], env);
    assert.strictEqual(result.type, null);
    const codes = result.diagnostics.map(d => d.code);
    assert.ok(codes.includes('E022'), `expected E022 in ${codes.join(',')}`);
  });

  it('reports E024 for malformed lambda binders', () => {
    const env = setupNaturalEnv();
    const result = synth(['lambda', ['x'], 'x'], env);
    assert.strictEqual(result.type, null);
    assert.strictEqual(result.diagnostics[0].code, 'E024');
    assert.match(result.diagnostics[0].message, /Lambda has malformed binder/);
  });
});

describe('bidirectional checker — check', () => {
  it('accepts a term whose synthesised type definitionally matches', () => {
    const env = setupNaturalEnv();
    const result = check('zero', 'Natural', env);
    assert.strictEqual(result.ok, true);
    assert.deepStrictEqual(result.diagnostics, []);
  });

  it('checks a lambda directly against its Pi-type without round-tripping', () => {
    const env = setupNaturalEnv();
    const result = check(
      ['lambda', ['Natural', 'x'], 'x'],
      ['Pi', ['Natural', 'x'], 'Natural'],
      env,
    );
    assert.strictEqual(result.ok, true);
    assert.deepStrictEqual(result.diagnostics, []);
  });

  it('reports E021 on definitional type mismatch', () => {
    const env = setupNaturalEnv();
    const result = check('zero', 'Boolean', env);
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.diagnostics.length, 1);
    assert.strictEqual(result.diagnostics[0].code, 'E021');
    assert.match(result.diagnostics[0].message, /Type mismatch/);
  });

  it('reports E023 when a lambda is checked against a non-Pi type', () => {
    const env = setupNaturalEnv();
    const result = check(
      ['lambda', ['Natural', 'x'], 'x'],
      'Natural',
      env,
    );
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.diagnostics[0].code, 'E023');
    assert.match(result.diagnostics[0].message, /Lambda .* cannot check against non-Pi type/);
  });

  it('reports E021 when a lambda parameter type does not match its Pi domain', () => {
    const env = setupNaturalEnv();
    const result = check(
      ['lambda', ['Boolean', 'x'], 'x'],
      ['Pi', ['Natural', 'x'], 'Natural'],
      env,
    );
    assert.strictEqual(result.ok, false);
    assert.strictEqual(result.diagnostics[0].code, 'E021');
    assert.match(result.diagnostics[0].message, /does not match Pi domain/);
  });

  it('accepts numeric literals against any annotation', () => {
    const env = setupNaturalEnv();
    const result = check('0.7', 'Natural', env);
    assert.strictEqual(result.ok, true);
  });

  it('respects an extended context: parameter type drives body checking', () => {
    const env = setupNaturalEnv();
    // lambda x:Natural. (apply succ x) :: Pi (Natural x) Natural
    const result = check(
      ['lambda', ['Natural', 'x'], ['apply', 'succ', 'x']],
      ['Pi', ['Natural', 'n'], 'Natural'],
      env,
    );
    assert.strictEqual(result.ok, true);
    assert.deepStrictEqual(result.diagnostics, []);
  });
});

describe('bidirectional checker — examples/dependent-types.lino', () => {
  it('checks each declaration in the canonical dependent-types example', () => {
    const text = fs.readFileSync(dependentTypesPath, 'utf8');
    const out = evaluate(text);
    // Sanity: the canonical example evaluates without diagnostics in full.
    assert.deepStrictEqual(out.diagnostics, []);

    // Reproduce its bindings in a fresh env, then run targeted checks.
    const env = new Env();
    evalNode(['Type:', 'Type', 'Type'], env);
    evalNode(['Natural:', 'Type', 'Natural'], env);
    evalNode(['Boolean:', 'Type', 'Boolean'], env);
    evalNode(['zero:', 'Natural', 'zero'], env);
    evalNode(['true-val:', 'Boolean', 'true-val'], env);
    evalNode(['false-val:', 'Boolean', 'false-val'], env);
    evalNode(['succ:', ['Pi', ['Natural', 'n'], 'Natural']], env);
    evalNode(['identity:', 'lambda', ['Natural', 'x'], 'x'], env);

    // (zero of Natural)
    assert.strictEqual(check('zero', 'Natural', env).ok, true);
    // (Natural of Type)
    assert.strictEqual(check('Natural', 'Type', env).ok, true);
    // (Boolean of Type)
    assert.strictEqual(check('Boolean', 'Type', env).ok, true);
    // (true-val of Boolean)
    assert.strictEqual(check('true-val', 'Boolean', env).ok, true);
    // (false-val of Boolean)
    assert.strictEqual(check('false-val', 'Boolean', env).ok, true);
    // (succ of (Pi (Natural n) Natural))
    assert.strictEqual(
      check('succ', ['Pi', ['Natural', 'n'], 'Natural'], env).ok,
      true,
    );
    // (identity of (Pi (Natural x) Natural))
    assert.strictEqual(
      check('identity', ['Pi', ['Natural', 'x'], 'Natural'], env).ok,
      true,
    );
    // type of zero == Natural
    const zeroType = synth('zero', env);
    assert.strictEqual(keyOf(zeroType.type), 'Natural');
    // (Type 0) of (Type 1)
    const type0 = synth(['Type', '0'], env);
    assert.strictEqual(keyOf(type0.type), '(Type 1)');
  });
});
