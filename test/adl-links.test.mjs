import { describe, it } from 'node:test';
import assert from 'node:assert';
import { run, tokenizeOne, parseOne, Env, evalNode } from '../src/adl-links.mjs';

describe('tokenizeOne', () => {
  it('should tokenize simple link', () => {
    const tokens = tokenizeOne('(a: a is a)');
    assert.deepStrictEqual(tokens, ['(', 'a:', 'a', 'is', 'a', ')']);
  });

  it('should tokenize nested link', () => {
    const tokens = tokenizeOne('((a = a) has probability 1)');
    assert.deepStrictEqual(tokens, ['(', '(', 'a', '=', 'a', ')', 'has', 'probability', '1', ')']);
  });

  it('should strip inline comments', () => {
    const tokens = tokenizeOne('(and: avg) # this is a comment');
    assert.deepStrictEqual(tokens, ['(', 'and:', 'avg', ')']);
  });

  it('should balance parens after stripping comments', () => {
    const tokens = tokenizeOne('((and: avg) # comment)');
    assert.deepStrictEqual(tokens, ['(', '(', 'and:', 'avg', ')', ')']);
  });
});

describe('parseOne', () => {
  it('should parse simple link', () => {
    const tokens = ['(', 'a:', 'a', 'is', 'a', ')'];
    const ast = parseOne(tokens);
    assert.deepStrictEqual(ast, ['a:', 'a', 'is', 'a']);
  });

  it('should parse nested link', () => {
    const tokens = ['(', '(', 'a', '=', 'a', ')', 'has', 'probability', '1', ')'];
    const ast = parseOne(tokens);
    assert.deepStrictEqual(ast, [['a', '=', 'a'], 'has', 'probability', '1']);
  });

  it('should parse deeply nested link', () => {
    const tokens = ['(', '?', '(', '(', 'a', '=', 'a', ')', 'and', '(', 'a', '!=', 'a', ')', ')', ')'];
    const ast = parseOne(tokens);
    assert.deepStrictEqual(ast, ['?', [['a', '=', 'a'], 'and', ['a', '!=', 'a']]]);
  });
});

describe('Env', () => {
  it('should initialize with default operators', () => {
    const env = new Env();
    assert.ok(env.ops.has('not'));
    assert.ok(env.ops.has('and'));
    assert.ok(env.ops.has('or'));
    assert.ok(env.ops.has('='));
    assert.ok(env.ops.has('!='));
  });

  it('should allow defining new operators', () => {
    const env = new Env();
    env.defineOp('test', (x) => x * 2);
    assert.ok(env.ops.has('test'));
    assert.strictEqual(env.getOp('test')(0.5), 1);
  });

  it('should store expression probabilities', () => {
    const env = new Env();
    env.setExprProb(['a', '=', 'a'], 1);
    assert.strictEqual(env.assign.get('(a = a)'), 1);
  });
});

describe('evalNode', () => {
  it('should evaluate numeric literals', () => {
    const env = new Env();
    assert.strictEqual(evalNode('1', env), 1);
    assert.strictEqual(evalNode('0.5', env), 0.5);
    assert.strictEqual(evalNode('0', env), 0);
  });

  it('should evaluate term definitions', () => {
    const env = new Env();
    evalNode(['a:', 'a', 'is', 'a'], env);
    assert.ok(env.terms.has('a'));
  });

  it('should evaluate operator redefinitions', () => {
    const env = new Env();
    evalNode(['!=:', 'not', '='], env);
    assert.ok(env.ops.has('!='));
  });

  it('should evaluate aggregator selection', () => {
    const env = new Env();
    evalNode(['and:', 'min'], env);
    const andOp = env.getOp('and');
    assert.strictEqual(andOp(0.3, 0.7), 0.3);
  });

  it('should evaluate probability assignments', () => {
    const env = new Env();
    const result = evalNode([['a', '=', 'a'], 'has', 'probability', '1'], env);
    assert.strictEqual(result, 1);
    assert.strictEqual(env.assign.get('(a = a)'), 1);
  });

  it('should evaluate equality operator', () => {
    const env = new Env();
    // Syntactic equality
    const result = evalNode(['a', '=', 'a'], env);
    assert.strictEqual(result, 1);
  });

  it('should evaluate inequality operator', () => {
    const env = new Env();
    const result = evalNode(['a', '!=', 'a'], env);
    assert.strictEqual(result, 0);
  });

  it('should evaluate not operator', () => {
    const env = new Env();
    const result = evalNode(['not', '1'], env);
    assert.strictEqual(result, 0);
  });

  it('should evaluate and operator (avg)', () => {
    const env = new Env();
    const result = evalNode(['1', 'and', '0'], env);
    assert.strictEqual(result, 0.5);
  });

  it('should evaluate or operator (max)', () => {
    const env = new Env();
    const result = evalNode(['1', 'or', '0'], env);
    assert.strictEqual(result, 1);
  });

  it('should evaluate queries', () => {
    const env = new Env();
    const result = evalNode(['?', '1'], env);
    assert.ok(result.query);
    assert.strictEqual(result.value, 1);
  });
});

describe('run', () => {
  it('should run demo.lino example', () => {
    const text = `
(a: a is a)
(!=: not =)
(and: avg)
(or: max)
((a = a) has probability 1)
((a != a) has probability 0)
(? ((a = a) and (a != a)))
(? ((a = a) or  (a != a)))
`;
    const results = run(text);
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0], 0.5);
    assert.strictEqual(results[1], 1);
  });

  it('should run flipped-axioms.lino example', () => {
    const text = `
(a: a is a)
(!=: not =)
(and: avg)
(or: max)
((a = a) has probability 0)
((a != a) has probability 1)
(? ((a = a) and (a != a)))
(? ((a = a) or  (a != a)))
`;
    const results = run(text);
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0], 0.5);
    assert.strictEqual(results[1], 1);
  });

  it('should handle different aggregators for and', () => {
    const text = `
(a: a is a)
(and: min)
((a = a) has probability 1)
((a != a) has probability 0)
(? ((a = a) and (a != a)))
`;
    const results = run(text);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 0);
  });

  it('should handle product aggregator', () => {
    const text = `
(and: prod)
(? (0.5 and 0.5))
`;
    const results = run(text);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 0.25);
  });

  it('should handle probabilistic sum aggregator', () => {
    const text = `
(or: ps)
(? (0.5 or 0.5))
`;
    const results = run(text);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 0.75);
  });

  it('should ignore comment-only links', () => {
    const text = `
# This is a comment
(# This is also a comment)
(a: a is a)
(? (a = a))
`;
    const results = run(text);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 1);
  });

  it('should handle inline comments', () => {
    const text = `
(a: a is a) # define term a
((a = a) has probability 1) # axiom
(? (a = a)) # query
`;
    const results = run(text);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 1);
  });
});
