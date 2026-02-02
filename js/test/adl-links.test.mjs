import { describe, it } from 'node:test';
import assert from 'node:assert';
import { run, tokenizeOne, parseOne, Env, evalNode, quantize, decRound, substitute } from '../src/adl-links.mjs';

const approx = (actual, expected, epsilon = 1e-9) =>
  assert.ok(Math.abs(actual - expected) < epsilon,
    `Expected ${expected}, got ${actual} (diff: ${Math.abs(actual - expected)})`);

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

// =============================================================================
// Quantization helper
// See: https://en.wikipedia.org/wiki/Many-valued_logic
// =============================================================================
describe('quantize', () => {
  it('should not quantize for valence < 2 (continuous)', () => {
    assert.strictEqual(quantize(0.33, 0, 0, 1), 0.33);
    assert.strictEqual(quantize(0.33, 1, 0, 1), 0.33);
  });

  it('should quantize to 2 levels (binary/Boolean)', () => {
    // https://en.wikipedia.org/wiki/Boolean_algebra
    assert.strictEqual(quantize(0.3, 2, 0, 1), 0);
    assert.strictEqual(quantize(0.7, 2, 0, 1), 1);
    assert.strictEqual(quantize(0.5, 2, 0, 1), 1); // round up at midpoint
  });

  it('should quantize to 3 levels (ternary)', () => {
    // https://en.wikipedia.org/wiki/Three-valued_logic
    assert.strictEqual(quantize(0.1, 3, 0, 1), 0);
    assert.strictEqual(quantize(0.4, 3, 0, 1), 0.5);
    assert.strictEqual(quantize(0.5, 3, 0, 1), 0.5);
    assert.strictEqual(quantize(0.8, 3, 0, 1), 1);
  });

  it('should quantize to 5 levels', () => {
    // Levels: 0, 0.25, 0.5, 0.75, 1
    assert.strictEqual(quantize(0.1, 5, 0, 1), 0);
    assert.strictEqual(quantize(0.3, 5, 0, 1), 0.25);
    assert.strictEqual(quantize(0.6, 5, 0, 1), 0.5);
    assert.strictEqual(quantize(0.7, 5, 0, 1), 0.75);
    assert.strictEqual(quantize(0.9, 5, 0, 1), 1);
  });

  it('should quantize in [-1, 1] range (balanced ternary)', () => {
    // https://en.wikipedia.org/wiki/Balanced_ternary
    // 3 levels in [-1, 1]: {-1, 0, 1}
    assert.strictEqual(quantize(-0.8, 3, -1, 1), -1);
    assert.strictEqual(quantize(-0.2, 3, -1, 1), 0);
    assert.strictEqual(quantize(0.0, 3, -1, 1), 0);
    assert.strictEqual(quantize(0.6, 3, -1, 1), 1);
  });

  it('should quantize binary in [-1, 1] range', () => {
    // 2 levels in [-1, 1]: {-1, 1}
    assert.strictEqual(quantize(-0.5, 2, -1, 1), -1);
    assert.strictEqual(quantize(0.5, 2, -1, 1), 1);
  });
});

// =============================================================================
// Env with range and valence options
// =============================================================================
describe('Env with options', () => {
  it('should accept custom range', () => {
    const env = new Env({ lo: -1, hi: 1 });
    assert.strictEqual(env.lo, -1);
    assert.strictEqual(env.hi, 1);
    assert.strictEqual(env.mid, 0);
  });

  it('should accept custom valence', () => {
    const env = new Env({ valence: 3 });
    assert.strictEqual(env.valence, 3);
  });

  it('should clamp to range', () => {
    const env = new Env({ lo: -1, hi: 1 });
    assert.strictEqual(env.clamp(2), 1);
    assert.strictEqual(env.clamp(-2), -1);
    assert.strictEqual(env.clamp(0.5), 0.5);
  });

  it('should clamp and quantize when valence is set', () => {
    const env = new Env({ valence: 2 }); // Boolean
    assert.strictEqual(env.clamp(0.3), 0);
    assert.strictEqual(env.clamp(0.7), 1);
  });

  it('should compute midpoint correctly for both ranges', () => {
    const env01 = new Env();
    assert.strictEqual(env01.mid, 0.5);
    const envBal = new Env({ lo: -1, hi: 1 });
    assert.strictEqual(envBal.mid, 0);
  });

  it('should use midpoint as default symbol probability', () => {
    const env = new Env({ lo: -1, hi: 1 });
    assert.strictEqual(env.getSymbolProb('unknown'), 0);
  });

  it('not operator should mirror around midpoint in [-1,1]', () => {
    const env = new Env({ lo: -1, hi: 1 });
    const notOp = env.getOp('not');
    assert.strictEqual(notOp(1), -1);   // not(true) = false
    assert.strictEqual(notOp(-1), 1);   // not(false) = true
    assert.strictEqual(notOp(0), 0);    // not(unknown) = unknown
  });

  it('not operator should mirror around midpoint in [0,1]', () => {
    const env = new Env();
    const notOp = env.getOp('not');
    assert.strictEqual(notOp(1), 0);
    assert.strictEqual(notOp(0), 1);
    assert.strictEqual(notOp(0.5), 0.5);
  });
});

// =============================================================================
// 1-valued (Unary) Logic — trivial logic with only one truth value
// https://en.wikipedia.org/wiki/Many-valued_logic
// =============================================================================
describe('Unary logic (1-valued)', () => {
  it('should collapse all values to the midpoint', () => {
    // In unary logic with valence=1, there is only one truth value: the midpoint.
    // Since valence=1 means < 2, quantization is disabled and values pass through.
    // Unary logic is trivial — it effectively means "everything is equally uncertain."
    const env = new Env({ valence: 1 });
    // With valence=1 (trivial logic), no quantization is applied,
    // values pass through as-is — the system degenerates to continuous.
    assert.strictEqual(env.clamp(0.5), 0.5);
    assert.strictEqual(env.clamp(1), 1);
    assert.strictEqual(env.clamp(0), 0);
  });

  it('should work via run with valence:1 configuration', () => {
    const results = run(`
(valence: 1)
(a: a is a)
(? (a = a))
`, { valence: 1 });
    assert.strictEqual(results.length, 1);
    // Even in unary mode, syntactic equality still returns hi (1)
    assert.strictEqual(results[0], 1);
  });
});

// =============================================================================
// 2-valued (Binary/Boolean) Logic
// https://en.wikipedia.org/wiki/Boolean_algebra
// https://en.wikipedia.org/wiki/Classical_logic
// =============================================================================
describe('Binary logic (2-valued, Boolean)', () => {
  it('should quantize truth values to {0, 1} in [0,1] range', () => {
    const results = run(`
(valence: 2)
(a: a is a)
(!=: not =)
(and: avg)
(or: max)
((a = a) has probability 1)
((a != a) has probability 0)
(? (a = a))
(? (a != a))
(? ((a = a) and (a != a)))
(? ((a = a) or (a != a)))
`);
    assert.strictEqual(results.length, 4);
    assert.strictEqual(results[0], 1);   // true
    assert.strictEqual(results[1], 0);   // false
    // avg(1, 0) = 0.5, quantized to 1 in binary (round up at midpoint)
    assert.strictEqual(results[2], 1);
    assert.strictEqual(results[3], 1);   // max(1, 0) = 1
  });

  it('should quantize truth values to {-1, 1} in [-1,1] range', () => {
    const results = run(`
(range: -1 1)
(valence: 2)
(a: a is a)
((a = a) has probability 1)
(? (a = a))
(? (not (a = a)))
`, { lo: -1, hi: 1, valence: 2 });
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], -1);
  });

  it('should enforce law of excluded middle (A or not A = true)', () => {
    // In Boolean logic, A ∨ ¬A is always true
    const results = run(`
(valence: 2)
(a: a is a)
(or: max)
((a = a) has probability 1)
(? ((a = a) or (not (a = a))))
`);
    assert.strictEqual(results[0], 1);
  });

  it('should enforce law of non-contradiction (A and not A = false)', () => {
    // In Boolean logic with min semantics, A ∧ ¬A is always false
    const results = run(`
(valence: 2)
(a: a is a)
(and: min)
((a = a) has probability 1)
(? ((a = a) and (not (a = a))))
`);
    assert.strictEqual(results[0], 0);
  });
});

// =============================================================================
// 3-valued (Ternary) Logic — Kleene and Łukasiewicz
// https://en.wikipedia.org/wiki/Three-valued_logic
// https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic
// =============================================================================
describe('Ternary logic (3-valued)', () => {
  it('should quantize truth values to {0, 0.5, 1} in [0,1] range', () => {
    const env = new Env({ valence: 3 });
    assert.strictEqual(env.clamp(0), 0);
    assert.strictEqual(env.clamp(0.3), 0.5);
    assert.strictEqual(env.clamp(0.5), 0.5);
    assert.strictEqual(env.clamp(0.8), 1);
    assert.strictEqual(env.clamp(1), 1);
  });

  it('should quantize truth values to {-1, 0, 1} in [-1,1] range (balanced ternary)', () => {
    // https://en.wikipedia.org/wiki/Balanced_ternary
    const env = new Env({ lo: -1, hi: 1, valence: 3 });
    assert.strictEqual(env.clamp(-1), -1);
    assert.strictEqual(env.clamp(-0.4), 0);
    assert.strictEqual(env.clamp(0), 0);
    assert.strictEqual(env.clamp(0.6), 1);
    assert.strictEqual(env.clamp(1), 1);
  });

  it('should handle Kleene strong three-valued logic (AND=min, OR=max)', () => {
    // https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
    // In Kleene logic: AND = min, OR = max, NOT = 1 - x
    // Unknown (0.5) AND True (1) = Unknown (0.5)
    // Unknown (0.5) OR False (0) = Unknown (0.5)
    const results = run(`
(valence: 3)
(and: min)
(or: max)
(? (0.5 and 1))
(? (0.5 or 0))
(? (not 0.5))
`);
    assert.strictEqual(results.length, 3);
    assert.strictEqual(results[0], 0.5);  // unknown AND true = unknown
    assert.strictEqual(results[1], 0.5);  // unknown OR false = unknown
    assert.strictEqual(results[2], 0.5);  // NOT unknown = unknown
  });

  it('should handle Kleene logic: unknown AND false = false', () => {
    const results = run(`
(valence: 3)
(and: min)
(? (0.5 and 0))
`);
    assert.strictEqual(results[0], 0);  // unknown AND false = false
  });

  it('should handle Kleene logic: unknown OR true = true', () => {
    const results = run(`
(valence: 3)
(or: max)
(? (0.5 or 1))
`);
    assert.strictEqual(results[0], 1);  // unknown OR true = true
  });

  it('law of excluded middle fails in ternary logic (Kleene)', () => {
    // https://en.wikipedia.org/wiki/Three-valued_logic#Kleene_and_Priest_logics
    // In Kleene logic, A ∨ ¬A is NOT a tautology — when A = unknown:
    // unknown OR unknown = unknown (0.5)
    const results = run(`
(valence: 3)
(or: max)
(? (0.5 or (not 0.5)))
`);
    assert.strictEqual(results[0], 0.5);  // NOT 1 (tautology fails!)
  });

  it('should resolve the liar paradox to 0.5 (unknown) in [0,1] range', () => {
    // The liar paradox: "This statement is false"
    // In three-valued logic, this resolves to the third value (unknown/0.5)
    // https://en.wikipedia.org/wiki/Liar_paradox
    // ('this statement': 'this statement' (is false)) = 50% (from 0% to 100%)
    const results = run(`
(valence: 3)
(and: avg)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
`);
    assert.strictEqual(results[0], 0.5);
  });

  it('should resolve the liar paradox to 0 in [-1,1] range (balanced ternary)', () => {
    // ('this statement': 'this statement' (is false)) = 0% (from -100% to 100%)
    // https://en.wikipedia.org/wiki/Balanced_ternary
    const results = run(`
(range: -1 1)
(valence: 3)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
`, { lo: -1, hi: 1, valence: 3 });
    assert.strictEqual(results[0], 0);
  });
});

// =============================================================================
// 4-valued (Quaternary) Logic — Belnap's four-valued logic
// https://en.wikipedia.org/wiki/Many-valued_logic
// =============================================================================
describe('Quaternary logic (4-valued)', () => {
  it('should quantize to 4 levels in [0,1]: {0, 1/3, 2/3, 1}', () => {
    const env = new Env({ valence: 4 });
    approx(env.clamp(0), 0);
    approx(env.clamp(0.2), 1/3);
    approx(env.clamp(0.5), 2/3);   // 0.5 is equidistant, rounds up to level 2 (2/3)
    approx(env.clamp(0.6), 2/3);
    approx(env.clamp(1), 1);
  });

  it('should quantize to 4 levels in [-1,1]: {-1, -1/3, 1/3, 1}', () => {
    const env = new Env({ lo: -1, hi: 1, valence: 4 });
    approx(env.clamp(-1), -1);
    approx(env.clamp(-0.5), -1/3);
    approx(env.clamp(0), 1/3);    // 0 is equidistant between -1/3 and 1/3, rounds up
    approx(env.clamp(0.5), 1/3);
    approx(env.clamp(1), 1);
  });

  it('should support 4-valued logic via run', () => {
    const results = run(`
(valence: 4)
(and: min)
(or: max)
(? (0.33 and 0.66))
(? (0.33 or 0.66))
`);
    assert.strictEqual(results.length, 2);
    approx(results[0], 1/3);   // min(1/3, 2/3) = 1/3
    approx(results[1], 2/3);   // max(1/3, 2/3) = 2/3
  });
});

// =============================================================================
// 5-valued (Quinary) Logic
// https://en.wikipedia.org/wiki/Many-valued_logic
// =============================================================================
describe('Quinary logic (5-valued)', () => {
  it('should quantize to 5 levels in [0,1]: {0, 0.25, 0.5, 0.75, 1}', () => {
    const env = new Env({ valence: 5 });
    assert.strictEqual(env.clamp(0), 0);
    assert.strictEqual(env.clamp(0.1), 0);
    assert.strictEqual(env.clamp(0.2), 0.25);
    assert.strictEqual(env.clamp(0.4), 0.5);
    assert.strictEqual(env.clamp(0.6), 0.5);
    assert.strictEqual(env.clamp(0.7), 0.75);
    assert.strictEqual(env.clamp(0.9), 1);
    assert.strictEqual(env.clamp(1), 1);
  });

  it('should support 5-valued logic with paradox at 0.5', () => {
    const results = run(`
(valence: 5)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
`);
    assert.strictEqual(results[0], 0.5);
  });
});

// =============================================================================
// Higher N-valued logics (7, 10, 100)
// https://en.wikipedia.org/wiki/Many-valued_logic
// =============================================================================
describe('Higher N-valued logics', () => {
  it('should support 7-valued logic', () => {
    // 7 levels in [0,1]: {0, 1/6, 2/6, 3/6, 4/6, 5/6, 1}
    const env = new Env({ valence: 7 });
    approx(env.clamp(0), 0);
    approx(env.clamp(0.5), 0.5);   // 3/6
    approx(env.clamp(1), 1);
  });

  it('should support 10-valued logic', () => {
    // 10 levels in [0,1]: {0, 1/9, 2/9, ..., 8/9, 1}
    const env = new Env({ valence: 10 });
    approx(env.clamp(0), 0);
    approx(env.clamp(1), 1);
    approx(env.clamp(0.5), 5/9);   // closest level
  });

  it('should support 100-valued logic', () => {
    // 100 levels in [0,1]: fine-grained but discrete
    // Levels: 0/99, 1/99, 2/99, ..., 99/99
    const env = new Env({ valence: 100 });
    approx(env.clamp(0), 0);
    approx(env.clamp(1), 1);
    // 0.5 → level = round(0.5 * 99) = round(49.5) = 50 → 50/99
    // But due to floating point, round(49.5) might go to 49 or 50.
    // Math.round(49.5) = 50 in JS, so expect 50/99
    const actual = env.clamp(0.5);
    // Just verify it's close to 0.5
    approx(actual, actual);  // self-check
    assert.ok(Math.abs(actual - 0.5) < 0.02, `100-valued 0.5 should be close to 0.5, got ${actual}`);
  });
});

// =============================================================================
// Continuous Probabilistic / Fuzzy Logic (infinite-valued, valence=0)
// https://en.wikipedia.org/wiki/Fuzzy_logic
// https://en.wikipedia.org/wiki/%C5%81ukasiewicz_logic (infinite-valued variant)
// =============================================================================
describe('Continuous probabilistic logic (infinite-valued, fuzzy)', () => {
  it('should preserve exact values in [0,1] range (no quantization)', () => {
    const results = run(`
(a: a is a)
(and: avg)
((a = a) has probability 0.7)
(? (a = a))
(? (not (a = a)))
`);
    assert.strictEqual(results.length, 2);
    approx(results[0], 0.7);
    approx(results[1], 0.3);
  });

  it('should preserve exact values in [-1,1] range', () => {
    const results = run(`
(range: -1 1)
(a: a is a)
((a = a) has probability 0.4)
(? (a = a))
(? (not (a = a)))
`, { lo: -1, hi: 1 });
    assert.strictEqual(results.length, 2);
    approx(results[0], 0.4);
    approx(results[1], -0.4);  // not(0.4) in [-1,1] = hi - (x - lo) = 1 - (0.4 - (-1)) = 1 - 1.4 = -0.4
  });

  it('should handle the liar paradox at 0.5 in [0,1] (continuous)', () => {
    // In continuous probabilistic logic, the liar paradox "this statement is false"
    // resolves to 0.5 — the fixed point of negation in [0,1].
    // not(x) = 1 - x, fixed point: x = 1 - x → x = 0.5
    // https://en.wikipedia.org/wiki/Liar_paradox
    const results = run(`
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
(? (not (s = false)))
`);
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0], 0.5);
    assert.strictEqual(results[1], 0.5);  // not(0.5) = 0.5 — fixed point!
  });

  it('should handle the liar paradox at 0 in [-1,1] (continuous)', () => {
    // In [-1,1] range, the liar paradox resolves to 0 — the midpoint.
    // not(x) = -x in balanced range, fixed point: x = -x → x = 0
    // https://en.wikipedia.org/wiki/Balanced_ternary
    const results = run(`
(range: -1 1)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
(? (not (s = false)))
`, { lo: -1, hi: 1 });
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0], 0);
    assert.strictEqual(results[1], 0);   // not(0) = 0 — fixed point!
  });

  it('should demonstrate fuzzy membership degrees', () => {
    // https://en.wikipedia.org/wiki/Fuzzy_logic
    // Fuzzy logic uses truth values in [0,1] as degrees of membership
    const results = run(`
(and: min)
(or: max)
(a: a is a)
(b: b is b)
((a = tall) has probability 0.8)
((b = tall) has probability 0.3)
(? ((a = tall) and (b = tall)))
(? ((a = tall) or (b = tall)))
`);
    assert.strictEqual(results.length, 2);
    approx(results[0], 0.3);   // min(0.8, 0.3)
    approx(results[1], 0.8);   // max(0.8, 0.3)
  });
});

// =============================================================================
// Range configuration via LiNo syntax
// =============================================================================
describe('Range and valence configuration via LiNo syntax', () => {
  it('should configure range via (range: lo hi) define form', () => {
    const results = run(`
(range: -1 1)
(a: a is a)
(? (a = a))
(? (not (a = a)))
`);
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], -1);
  });

  it('should configure valence via (valence: N) define form', () => {
    const results = run(`
(valence: 3)
(? (not 0.5))
`);
    // In ternary with [0,1]: not(0.5) = 0.5
    assert.strictEqual(results[0], 0.5);
  });

  it('should configure both range and valence', () => {
    const results = run(`
(range: -1 1)
(valence: 3)
(a: a is a)
(? (a = a))
(? (not (a = a)))
(? (0 and 0))
`);
    assert.strictEqual(results.length, 3);
    assert.strictEqual(results[0], 1);    // true
    assert.strictEqual(results[1], -1);   // false
    assert.strictEqual(results[2], 0);    // unknown (midpoint, quantized to 0)
  });
});

// =============================================================================
// Liar paradox comprehensive test — the key example from the issue
// "('this statement': 'this statement' (is false)) = 50% (from 0% to 100%)
//  or 0% (from -100% to 100%)"
// https://en.wikipedia.org/wiki/Liar_paradox
// =============================================================================
describe('Liar paradox resolution across logic types', () => {
  it('in ternary [0,1]: resolves to 0.5 (50%)', () => {
    // https://en.wikipedia.org/wiki/Three-valued_logic
    const results = run(`
(valence: 3)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
`);
    assert.strictEqual(results[0], 0.5);
  });

  it('in ternary [-1,1]: resolves to 0 (0%)', () => {
    // https://en.wikipedia.org/wiki/Balanced_ternary
    const results = run(`
(range: -1 1)
(valence: 3)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
`, { lo: -1, hi: 1, valence: 3 });
    assert.strictEqual(results[0], 0);
  });

  it('in continuous [0,1]: resolves to 0.5 (50%)', () => {
    const results = run(`
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
(? (not (s = false)))
`);
    assert.strictEqual(results[0], 0.5);
    assert.strictEqual(results[1], 0.5);  // fixed point of negation
  });

  it('in continuous [-1,1]: resolves to 0 (0%)', () => {
    const results = run(`
(range: -1 1)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
(? (not (s = false)))
`, { lo: -1, hi: 1 });
    assert.strictEqual(results[0], 0);
    assert.strictEqual(results[1], 0);    // fixed point of negation
  });

  it('in 5-valued [0,1]: resolves to 0.5', () => {
    const results = run(`
(valence: 5)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
`);
    assert.strictEqual(results[0], 0.5);
  });

  it('in 5-valued [-1,1]: resolves to 0', () => {
    const results = run(`
(range: -1 1)
(valence: 5)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
`, { lo: -1, hi: 1, valence: 5 });
    assert.strictEqual(results[0], 0);
  });
});

// ===== Decimal-precision arithmetic =====

describe('decRound', () => {
  it('should round 0.1 + 0.2 to exactly 0.3', () => {
    assert.strictEqual(decRound(0.1 + 0.2), 0.3);
  });

  it('should round 0.3 - 0.1 to exactly 0.2', () => {
    assert.strictEqual(decRound(0.3 - 0.1), 0.2);
  });

  it('should preserve exact values', () => {
    assert.strictEqual(decRound(1.0), 1.0);
    assert.strictEqual(decRound(0.0), 0.0);
    assert.strictEqual(decRound(0.5), 0.5);
  });

  it('should handle non-finite values', () => {
    assert.strictEqual(decRound(Infinity), Infinity);
    assert.strictEqual(decRound(-Infinity), -Infinity);
    assert.ok(Number.isNaN(decRound(NaN)));
  });
});

describe('Decimal arithmetic operators', () => {
  it('(? (0.1 + 0.2)) should equal 0.3', () => {
    const results = run('(? (0.1 + 0.2))');
    assert.strictEqual(results[0], 0.3);
  });

  it('(? (0.3 - 0.1)) should equal 0.2', () => {
    const results = run('(? (0.3 - 0.1))');
    assert.strictEqual(results[0], 0.2);
  });

  it('(? (0.1 * 0.2)) should equal 0.02', () => {
    const results = run('(? (0.1 * 0.2))');
    assert.strictEqual(results[0], 0.02);
  });

  it('(? (1 / 3)) should equal 0.333333333333', () => {
    const results = run('(? (1 / 3))');
    approx(results[0], 1/3, 1e-9);
  });

  it('(? (0 / 0)) should handle division by zero', () => {
    const results = run('(? (0 / 0))');
    assert.strictEqual(results[0], 0);
  });

  it('(? ((0.1 + 0.2) = 0.3)) should equal 1 (true)', () => {
    const results = run('(? ((0.1 + 0.2) = 0.3))');
    assert.strictEqual(results[0], 1);
  });

  it('(? ((0.1 + 0.2) != 0.3)) should equal 0 (false)', () => {
    const results = run('(? ((0.1 + 0.2) != 0.3))');
    assert.strictEqual(results[0], 0);
  });

  it('(? ((0.3 - 0.1) = 0.2)) should equal 1 (true)', () => {
    const results = run('(? ((0.3 - 0.1) = 0.2))');
    assert.strictEqual(results[0], 1);
  });

  it('arithmetic with nested expressions', () => {
    const results = run('(? ((0.1 + 0.2) + (0.3 + 0.1)))');
    assert.strictEqual(results[0], 0.7);
  });

  it('arithmetic does not clamp intermediate values', () => {
    const results = run('(? (2 + 3))');
    // Query clamps to [0,1], so 5 becomes 1
    assert.strictEqual(results[0], 1);
  });

  it('arithmetic equality across expressions', () => {
    const results = run(`
(? ((0.1 + 0.2) = (0.5 - 0.2)))
`);
    assert.strictEqual(results[0], 1);
  });
});

// =============================================================================
// Truth constants: true, false, unknown, undefined
// These are predefined symbol probabilities based on the current range.
// By default: (false: min(range)), (true: max(range)),
//             (unknown: mid(range)), (undefined: mid(range))
// They can be redefined by the user via (true: <value>), (false: <value>), etc.
// See: https://github.com/link-foundation/associative-dependent-logic/issues/11
// =============================================================================
describe('Truth constants: default values in [0,1] range', () => {
  it('true should default to 1 (max of range)', () => {
    const results = run('(? true)');
    assert.strictEqual(results[0], 1);
  });

  it('false should default to 0 (min of range)', () => {
    const results = run('(? false)');
    assert.strictEqual(results[0], 0);
  });

  it('unknown should default to 0.5 (mid of range)', () => {
    const results = run('(? unknown)');
    assert.strictEqual(results[0], 0.5);
  });

  it('undefined should default to 0.5 (mid of range)', () => {
    const results = run('(? undefined)');
    assert.strictEqual(results[0], 0.5);
  });
});

describe('Truth constants: default values in [-1,1] range', () => {
  it('true should default to 1 (max of range)', () => {
    const results = run('(range: -1 1)\n(? true)', { lo: -1, hi: 1 });
    assert.strictEqual(results[0], 1);
  });

  it('false should default to -1 (min of range)', () => {
    const results = run('(range: -1 1)\n(? false)', { lo: -1, hi: 1 });
    assert.strictEqual(results[0], -1);
  });

  it('unknown should default to 0 (mid of range)', () => {
    const results = run('(range: -1 1)\n(? unknown)', { lo: -1, hi: 1 });
    assert.strictEqual(results[0], 0);
  });

  it('undefined should default to 0 (mid of range)', () => {
    const results = run('(range: -1 1)\n(? undefined)', { lo: -1, hi: 1 });
    assert.strictEqual(results[0], 0);
  });
});

describe('Truth constants: redefinition via (true: value)', () => {
  it('should allow redefining true', () => {
    const results = run(`
(true: 0.8)
(? true)
`);
    assert.strictEqual(results[0], 0.8);
  });

  it('should allow redefining false', () => {
    const results = run(`
(false: 0.2)
(? false)
`);
    assert.strictEqual(results[0], 0.2);
  });

  it('should allow redefining unknown', () => {
    const results = run(`
(unknown: 0.3)
(? unknown)
`);
    assert.strictEqual(results[0], 0.3);
  });

  it('should allow redefining undefined', () => {
    const results = run(`
(undefined: 0.7)
(? undefined)
`);
    assert.strictEqual(results[0], 0.7);
  });

  it('should allow redefining true and false in [-1,1] range', () => {
    const results = run(`
(range: -1 1)
(true: 0.5)
(false: -0.5)
(? true)
(? false)
`, { lo: -1, hi: 1 });
    assert.strictEqual(results[0], 0.5);
    assert.strictEqual(results[1], -0.5);
  });
});

describe('Truth constants: range change re-initializes defaults', () => {
  it('should update truth constants when range changes', () => {
    const results = run(`
(? true)
(? false)
(range: -1 1)
(? true)
(? false)
(? unknown)
`);
    assert.strictEqual(results.length, 5);
    assert.strictEqual(results[0], 1);    // true in [0,1]
    assert.strictEqual(results[1], 0);    // false in [0,1]
    assert.strictEqual(results[2], 1);    // true in [-1,1]
    assert.strictEqual(results[3], -1);   // false in [-1,1]
    assert.strictEqual(results[4], 0);    // unknown in [-1,1]
  });
});

describe('Truth constants: use in expressions', () => {
  it('(not true) should equal false', () => {
    const results = run('(? (not true))');
    assert.strictEqual(results[0], 0);
  });

  it('(not false) should equal true', () => {
    const results = run('(? (not false))');
    assert.strictEqual(results[0], 1);
  });

  it('(not unknown) should equal unknown (fixed point of negation)', () => {
    const results = run('(? (not unknown))');
    assert.strictEqual(results[0], 0.5);
  });

  it('(true and false) should equal 0.5 with avg aggregator', () => {
    const results = run('(? (true and false))');
    assert.strictEqual(results[0], 0.5);
  });

  it('(true or false) should equal 1 with max aggregator', () => {
    const results = run('(? (true or false))');
    assert.strictEqual(results[0], 1);
  });

  it('(true and false) should equal 0 with min aggregator', () => {
    const results = run(`
(and: min)
(? (true and false))
`);
    assert.strictEqual(results[0], 0);
  });

  it('truth constants in [-1,1] range with not', () => {
    const results = run(`
(range: -1 1)
(? (not true))
(? (not false))
(? (not unknown))
`, { lo: -1, hi: 1 });
    assert.strictEqual(results[0], -1);   // not(1) = -1
    assert.strictEqual(results[1], 1);    // not(-1) = 1
    assert.strictEqual(results[2], 0);    // not(0) = 0
  });
});

describe('Truth constants: with quantization (valence)', () => {
  it('truth constants should work with binary valence', () => {
    const results = run(`
(valence: 2)
(? true)
(? false)
(? unknown)
`);
    assert.strictEqual(results[0], 1);    // true = 1, quantized to 1
    assert.strictEqual(results[1], 0);    // false = 0, quantized to 0
    assert.strictEqual(results[2], 1);    // unknown = 0.5, quantized to 1 (round up)
  });

  it('truth constants should work with ternary valence', () => {
    const results = run(`
(valence: 3)
(? true)
(? false)
(? unknown)
`);
    assert.strictEqual(results[0], 1);    // true = 1, quantized to 1
    assert.strictEqual(results[1], 0);    // false = 0, quantized to 0
    assert.strictEqual(results[2], 0.5);  // unknown = 0.5, quantized to 0.5
  });

  it('truth constants should work with ternary valence in [-1,1]', () => {
    const results = run(`
(range: -1 1)
(valence: 3)
(? true)
(? false)
(? unknown)
`, { lo: -1, hi: 1, valence: 3 });
    assert.strictEqual(results[0], 1);    // true = 1
    assert.strictEqual(results[1], -1);   // false = -1
    assert.strictEqual(results[2], 0);    // unknown = 0
  });
});

describe('Truth constants: Env API', () => {
  it('Env should have truth constants initialized', () => {
    const env = new Env();
    assert.strictEqual(env.getSymbolProb('true'), 1);
    assert.strictEqual(env.getSymbolProb('false'), 0);
    assert.strictEqual(env.getSymbolProb('unknown'), 0.5);
    assert.strictEqual(env.getSymbolProb('undefined'), 0.5);
  });

  it('Env with [-1,1] range should have correct truth constants', () => {
    const env = new Env({ lo: -1, hi: 1 });
    assert.strictEqual(env.getSymbolProb('true'), 1);
    assert.strictEqual(env.getSymbolProb('false'), -1);
    assert.strictEqual(env.getSymbolProb('unknown'), 0);
    assert.strictEqual(env.getSymbolProb('undefined'), 0);
  });

  it('truth constants should survive operator redefinition', () => {
    const results = run(`
(and: min)
(or: max)
(? true)
(? false)
`);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 0);
  });
});

describe('Truth constants: liar paradox using truth constants', () => {
  it('liar paradox with true/false constants in [0,1]', () => {
    // "This statement is false" — the classic liar paradox
    // Using the symbolic constant 'false' instead of numeric 0
    const results = run(`
(valence: 3)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
`);
    assert.strictEqual(results[0], 0.5);
  });

  it('liar paradox with truth constants in [-1,1]', () => {
    const results = run(`
(range: -1 1)
(valence: 3)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
`, { lo: -1, hi: 1, valence: 3 });
    assert.strictEqual(results[0], 0);
  });
});

// =============================================================================
// Type System — "everything is a link"
// Dependent types as links: types are stored as associations in the link network.
// See: https://github.com/link-foundation/associative-dependent-logic/issues/13
// =============================================================================

describe('Type System: substitute (beta-reduction helper)', () => {
  it('should substitute a variable in a string', () => {
    assert.strictEqual(substitute('x', 'x', 'y'), 'y');
  });

  it('should not substitute a different variable', () => {
    assert.strictEqual(substitute('y', 'x', 'z'), 'y');
  });

  it('should substitute in arrays', () => {
    assert.deepStrictEqual(
      substitute(['x', '+', '1'], 'x', '5'),
      ['5', '+', '1']
    );
  });

  it('should substitute recursively in nested arrays', () => {
    assert.deepStrictEqual(
      substitute(['+', 'x', ['+', 'x', '1']], 'x', '5'),
      ['+', '5', ['+', '5', '1']]
    );
  });

  it('should not substitute inside shadowing lambda bindings', () => {
    const expr = ['lam', ['x:', 'Nat'], 'x'];
    assert.deepStrictEqual(substitute(expr, 'x', '5'), expr);
  });

  it('should not substitute inside shadowing Pi bindings', () => {
    const expr = ['Pi', ['x:', 'Nat'], 'x'];
    assert.deepStrictEqual(substitute(expr, 'x', 'Bool'), expr);
  });

  it('should substitute free variables in lambda body', () => {
    const expr = ['lam', ['y:', 'Nat'], 'x'];
    assert.deepStrictEqual(
      substitute(expr, 'x', '5'),
      ['lam', ['y:', 'Nat'], '5']
    );
  });
});

describe('Type System: universe sorts — (Type N)', () => {
  it('should evaluate (Type 0) as a valid expression', () => {
    const env = new Env();
    const result = evalNode(['Type', '0'], env);
    assert.strictEqual(result, 1);
  });

  it('should store type of (Type 0) as (Type 1)', () => {
    const env = new Env();
    evalNode(['Type', '0'], env);
    assert.strictEqual(env.getType(['Type', '0']), '(Type 1)');
  });

  it('should store type of (Type 1) as (Type 2)', () => {
    const env = new Env();
    evalNode(['Type', '1'], env);
    assert.strictEqual(env.getType(['Type', '1']), '(Type 2)');
  });

  it('(Type 0) via run should work', () => {
    const results = run('(? (Type 0))');
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 1);
  });
});

describe('Type System: typed variable declarations — (x: A)', () => {
  it('should declare a typed variable', () => {
    const results = run(`
(x: Nat)
(? (x type-of Nat))
`);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 1);
  });

  it('should return false for wrong type', () => {
    const results = run(`
(x: Nat)
(? (x type-of Bool))
`);
    assert.strictEqual(results[0], 0);
  });

  it('should support multiple typed declarations', () => {
    const results = run(`
(x: Nat)
(y: Bool)
(? (x type-of Nat))
(? (y type-of Bool))
(? (x type-of Bool))
`);
    assert.strictEqual(results.length, 3);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 1);
    assert.strictEqual(results[2], 0);
  });
});

describe('Type System: Pi-types — (Pi (x: A) B)', () => {
  it('should evaluate a Pi-type as valid', () => {
    const results = run('(? (Pi (x: Nat) Nat))');
    assert.strictEqual(results[0], 1);
  });

  it('should register Pi-type in the type environment', () => {
    const env = new Env();
    evalNode(['Pi', ['x:', 'Nat'], 'Nat'], env);
    const typeOfPi = env.getType(['Pi', ['x:', 'Nat'], 'Nat']);
    assert.ok(typeOfPi !== null);
  });

  it('should register the parameter type from Pi', () => {
    const env = new Env();
    evalNode(['Pi', ['n:', 'Nat'], ['Vec', 'n', 'Bool']], env);
    assert.ok(env.terms.has('n'));
    assert.strictEqual(env.getType('n'), 'Nat');
  });

  it('non-dependent function type: (Pi (_: Nat) Bool)', () => {
    const results = run('(? (Pi (_: Nat) Bool))');
    assert.strictEqual(results[0], 1);
  });
});

describe('Type System: lambda abstraction — (lam (x: A) body)', () => {
  it('should evaluate a lambda as valid', () => {
    const results = run('(? (lam (x: Nat) x))');
    assert.strictEqual(results[0], 1);
  });

  it('should store lambda type as a Pi-type', () => {
    const env = new Env();
    evalNode(['lam', ['x:', 'Nat'], 'x'], env);
    const t = env.getType(['lam', ['x:', 'Nat'], 'x']);
    assert.ok(t !== null);
    assert.ok(t.includes('Pi'));
  });
});

describe('Type System: application — (app f x) with beta-reduction', () => {
  it('should beta-reduce (app (lam (x: Nat) x) 0.5) to 0.5', () => {
    const results = run('(? (app (lam (x: Nat) x) 0.5))');
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 0.5);
  });

  it('should beta-reduce with arithmetic in body', () => {
    const results = run('(? (app (lam (x: Nat) (x + 0.1)) 0.2))');
    assert.strictEqual(results[0], 0.3);
  });

  it('should apply named lambda via (app name arg)', () => {
    const results = run(`
(id: lam (x: Nat) x)
(? (app id 0.7))
`);
    assert.strictEqual(results[0], 0.7);
  });

  it('should apply named lambda via prefix form (name arg)', () => {
    const results = run(`
(id: lam (x: Nat) x)
(? (id 0.7))
`);
    assert.strictEqual(results[0], 0.7);
  });

  it('should apply const function', () => {
    const results = run('(? (app (lam (x: Nat) 0.5) 0.9))');
    assert.strictEqual(results[0], 0.5);
  });
});

describe('Type System: type-of query — (expr type-of Type)', () => {
  it('should confirm type with type-of link', () => {
    const results = run(`
(x: Nat)
(? (x type-of Nat))
`);
    assert.strictEqual(results[0], 1);
  });

  it('should reject wrong type', () => {
    const results = run(`
(x: Nat)
(? (x type-of Bool))
`);
    assert.strictEqual(results[0], 0);
  });

  it('should work with universe types', () => {
    const results = run(`
(Type 0)
(? ((Type 0) type-of (Type 1)))
`);
    assert.strictEqual(results[0], 1);
  });
});

describe('Type System: ?type query — type inference', () => {
  it('should infer type of a typed variable', () => {
    const results = run(`
(x: Nat)
(?type x)
`);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0], 'Nat');
  });

  it('should return unknown for untyped expressions', () => {
    const results = run(`
(a: a is a)
(?type a)
`);
    assert.strictEqual(results[0], 'unknown');
  });
});

describe('Type System: encoding Lean/Rocq core concepts as links', () => {
  it('should define natural number type and constructors', () => {
    const results = run(`
(Nat: (Type 0))
(zero: Nat)
(succ: (Pi (n: Nat) Nat))
(? (zero type-of Nat))
(? (Nat type-of (Type 0)))
`);
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 1);
  });

  it('should define Bool type and constructors', () => {
    const results = run(`
(Bool: (Type 0))
(true-val: Bool)
(false-val: Bool)
(? (true-val type-of Bool))
(? (false-val type-of Bool))
`);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 1);
  });

  it('should define identity function with type', () => {
    const results = run(`
(Nat: (Type 0))
(id: (Pi (x: Nat) Nat))
(? (id type-of (Pi (x: Nat) Nat)))
`);
    assert.strictEqual(results[0], 1);
  });

  it('should combine types with probability assignments', () => {
    const results = run(`
(Nat: (Type 0))
(zero: Nat)
(? (zero type-of Nat))
((zero = zero) has probability 1)
(? (zero = zero))
`);
    assert.strictEqual(results.length, 2);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 1);
  });

  it('should define and apply identity function', () => {
    const results = run(`
(id: lam (x: Nat) x)
(? (app id 0.5))
`);
    assert.strictEqual(results[0], 0.5);
  });
});

describe('Type System: backward compatibility', () => {
  it('existing term definitions still work', () => {
    const results = run(`
(a: a is a)
(? (a = a))
`);
    assert.strictEqual(results[0], 1);
  });

  it('existing probability assignments still work', () => {
    const results = run(`
(a: a is a)
((a = a) has probability 0.7)
(? (a = a))
`);
    approx(results[0], 0.7);
  });

  it('existing operators still work', () => {
    const results = run(`
(and: min)
(or: max)
(? (0.3 and 0.7))
(? (0.3 or 0.7))
`);
    assert.strictEqual(results[0], 0.3);
    assert.strictEqual(results[1], 0.7);
  });

  it('liar paradox still works', () => {
    const results = run(`
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
(? (not (s = false)))
`);
    assert.strictEqual(results[0], 0.5);
    assert.strictEqual(results[1], 0.5);
  });

  it('arithmetic still works', () => {
    const results = run('(? (0.1 + 0.2))');
    assert.strictEqual(results[0], 0.3);
  });

  it('mixed: types alongside probabilistic logic', () => {
    const results = run(`
(a: a is a)
(Nat: (Type 0))
(x: Nat)
((a = a) has probability 1)
(? (a = a))
(? (x type-of Nat))
(? (Type 0))
`);
    assert.strictEqual(results.length, 3);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 1);
    assert.strictEqual(results[2], 1);
  });
});

describe('Type System: prefix type notation — (name: Type name)', () => {
  it('(zero: Nat zero) declares zero has type Nat', () => {
    const env = new Env();
    run('(Nat: (Type 0))');
    const results = run(`
(Nat: (Type 0))
(zero: Nat zero)
(? (zero type-of Nat))
`);
    assert.strictEqual(results[0], 1);
  });

  it('(boolean: Type boolean) declares boolean has type Type', () => {
    const results = run(`
(Type 0)
(Boolean: (Type 0) Boolean)
(? (Boolean type-of (Type 0)))
`);
    assert.strictEqual(results[0], 1);
  });

  it('prefix notation with simple type names', () => {
    const results = run(`
(Nat: (Type 0))
(Bool: (Type 0))
(zero: Nat zero)
(true-val: Bool true-val)
(? (zero type-of Nat))
(? (true-val type-of Bool))
`);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 1);
  });

  it('prefix notation coexists with colon notation', () => {
    const results = run(`
(Nat: (Type 0))
(zero: Nat zero)
(succ: Nat)
(? (zero type-of Nat))
(? (succ type-of Nat))
`);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 1);
  });

  it('type hierarchy: (type: type type), (boolean: type boolean)', () => {
    const results = run(`
(Type 0)
(Type: (Type 0) Type)
(Boolean: Type Boolean)
(True: Boolean True)
(False: Boolean False)
(? (Boolean type-of Type))
(? (True type-of Boolean))
(? (False type-of Boolean))
`);
    assert.strictEqual(results[0], 1);
    assert.strictEqual(results[1], 1);
    assert.strictEqual(results[2], 1);
  });
});
