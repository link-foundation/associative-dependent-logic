# associative-dependent-logic — JavaScript

JavaScript implementation of the Associative-Dependent Logic (ADL) framework.

## Prerequisites

- [Node.js](https://nodejs.org/) >= 18.0.0

## Installation

```bash
cd js
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

## API

```javascript
import { run, tokenizeOne, parseOne, Env, evalNode, quantize } from './src/adl-links.mjs';

// Run a complete LiNo knowledge base
const results = run(linoText);

// Run with custom range and valence
const results2 = run(linoText, { lo: -1, hi: 1, valence: 3 });

// Parse and evaluate individual expressions
const env = new Env({ lo: 0, hi: 1, valence: 3 });
const ast = parseOne(tokenizeOne('(a = a)'));
const truthValue = evalNode(ast, env);

// Quantize a value to N discrete levels
const q = quantize(0.4, 3, 0, 1); // -> 0.5 (nearest ternary level)
```

## Testing

```bash
npm test
```

The test suite includes 78 tests covering:
- Tokenization, parsing, and quantization
- Evaluation logic and operator aggregators
- Many-valued logics: unary, binary (Boolean), ternary (Kleene), quaternary, quinary, higher N-valued, and continuous (fuzzy)
- Both `[0, 1]` and `[-1, 1]` ranges
- Liar paradox resolution across logic types

## Dependencies

- [`@linksplatform/protocols-lino`](https://github.com/linksplatform/protocols-lino) — official LiNo parser

## License

See [LICENSE](../LICENSE) file.
