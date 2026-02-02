# associative-dependent-logic — Rust

Rust implementation of the Associative-Dependent Logic (ADL) framework.

## Prerequisites

- [Rust](https://rustup.rs/) (edition 2021)

## Building

```bash
cd rust
cargo build
```

## Usage

### Running a knowledge base

```bash
cargo run -- <file.lino>
```

Or after building:

```bash
./target/release/adl <file.lino>
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

```rust
use adl::{run, tokenize_one, parse_one, Env, EnvOptions, eval_node, quantize, dec_round};

// Run a complete LiNo knowledge base
let results = run(lino_text, None);

// Run with custom range and valence
let results2 = run(lino_text, Some(EnvOptions { lo: -1.0, hi: 1.0, valence: 3 }));

// Parse and evaluate individual expressions
let mut env = Env::new(Some(EnvOptions { lo: 0.0, hi: 1.0, valence: 3 }));
let tokens = tokenize_one("(a = a)");
let ast = parse_one(&tokens).unwrap();
let truth_value = eval_node(&ast, &mut env);

// Quantize a value to N discrete levels
let q = quantize(0.4, 3, 0.0, 1.0); // -> 0.5 (nearest ternary level)
```

## Testing

```bash
cargo test
```

The test suite includes 170 tests covering:
- Tokenization, parsing, and quantization
- Evaluation logic and operator aggregators
- Many-valued logics: unary, binary (Boolean), ternary (Kleene), quaternary, quinary, higher N-valued, and continuous (fuzzy)
- Both `[0, 1]` and `[-1, 1]` ranges
- Liar paradox resolution across logic types
- Decimal-precision arithmetic and numeric equality
- Dependent type system: universes, Pi-types, lambdas, application, type queries

## Implementation Notes

The Rust implementation uses the official [`links-notation`](https://crates.io/crates/links-notation) crate for LiNo parsing. The implementation is a direct port of the JavaScript version and produces identical results for all test cases.

## License

See [LICENSE](../LICENSE) file.
