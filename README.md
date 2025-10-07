# associative-dependent-logic

A prototype for logic framework that can reason about anything relative to given probability of input statements.

## Overview

ADL (Associative-Dependent Logic) is a minimal probabilistic logic system built on top of [LiNo (Links Notation)](https://github.com/linksplatform/protocols-lino). It allows you to:

- Define terms
- Assign probabilities to logical expressions
- Redefine logical operators with different semantics
- Query the probability of complex expressions

## Installation

```bash
npm install
```

## Usage

### Running a knowledge base

```bash
node src/adl-links.mjs <file.lino>
```

Or use the npm script:

```bash
npm run demo
```

### Example

Create a file `example.lino`:

```lino
# Define a term
(a: a is a)

# Set up operators
(!=: not =)
(and: avg)
(or: max)

# Assign probabilities to axioms
((a = a) has probability 1)
((a != a) has probability 0)

# Query probabilities
(? ((a = a) and (a != a)))   # -> 0.5
(? ((a = a) or  (a != a)))   # -> 1
```

Run it:

```bash
node src/adl-links.mjs example.lino
```

Output:
```
0.5
1
```

## Syntax

### Term Definitions

```lino
(term_name: term_name is term_name)
```

Example: `(a: a is a)` declares `a` as a term.

### Probability Assignments

```lino
((<expression>) has probability <value>)
```

Example: `((a = a) has probability 1)` assigns probability 1 to the expression `a = a`.

### Operator Redefinitions

#### Binary operator composition

```lino
(operator: unary_op binary_op)
```

Example: `(!=: not =)` defines `!=` as the negation of `=`.

#### Aggregator selection

For `and` and `or` operators, you can choose different aggregators:

```lino
(and: avg)   # Average (default)
(and: min)   # Minimum
(and: max)   # Maximum
(and: prod)  # Product
(and: ps)    # Probabilistic sum: 1 - (1-p1)*(1-p2)*...

(or: max)    # Maximum (default)
(or: avg)    # Average
(or: min)    # Minimum
(or: prod)   # Product
(or: ps)     # Probabilistic sum
```

### Queries

```lino
(? <expression>)
```

Queries are evaluated and their probability is printed to stdout.

### Comments

```lino
# Line comments start with #
(a: a is a)  # Inline comments are also supported
```

## Built-in Operators

- `=`: Equality (syntactic by default, can be overridden with probability assignments)
- `!=`: Inequality (defined as `not =` by default)
- `not`: Logical negation (1 - p)
- `and`: Conjunction (average by default, configurable)
- `or`: Disjunction (maximum by default, configurable)

## Examples

### Standard Logic (with avg semantics)

See `demo.lino`:

```lino
(a: a is a)
(!=: not =)
(and: avg)
(or: max)

((a = a) has probability 1)
((a != a) has probability 0)

(? ((a = a) and (a != a)))   # -> 0.5
(? ((a = a) or  (a != a)))   # -> 1
```

### Flipped Axioms

See `flipped-axioms.lino` - demonstrates that the system can handle arbitrary probability assignments:

```lino
(a: a is a)
(!=: not =)
(and: avg)
(or: max)

((a = a) has probability 0)
((a != a) has probability 1)

(? ((a = a) and (a != a)))   # -> 0.5
(? ((a = a) or  (a != a)))   # -> 1
```

## Testing

Run the test suite:

```bash
npm test
```

The test suite includes:
- Unit tests for tokenization and parsing
- Unit tests for evaluation logic
- Integration tests with example knowledge bases
- Tests for different operator aggregators

## API

The module exports the following functions for programmatic use:

```javascript
import { run, tokenizeOne, parseOne, Env, evalNode } from './src/adl-links.mjs';

// Run a complete LiNo knowledge base
const results = run(linoText);

// Parse and evaluate individual expressions
const env = new Env();
const ast = parseOne(tokenizeOne('(a = a)'));
const probability = evalNode(ast, env);
```

## License

See [LICENSE](LICENSE) file.
