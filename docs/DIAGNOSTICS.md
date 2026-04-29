# Diagnostics

RML reports every parser, evaluator, and type-checker error as a structured
**Diagnostic** value rather than throwing or panicking. This makes failures
easy to consume programmatically (an editor, a test runner, a tool that
reports errors in a UI) and gives the CLI everything it needs to print a
familiar `file:line:col` message with a caret under the offending token.

The two reference implementations (`js/src/rml-links.mjs` and
`rust/src/lib.rs`) share the same diagnostic shape, error-code table, and
formatting rules so output stays consistent across runtimes.

## Diagnostic shape

```
Diagnostic {
  code:    "Exxx",            // see the table below
  message: "human-readable summary",
  span: {
    file:   "kb.lino" | null, // input file, when known
    line:   1,                // 1-based line of the offending form
    col:    1,                // 1-based column
    length: 1,                // length used to render carets ("^^^")
  }
}
```

### JavaScript

```js
import { evaluate, formatDiagnostic } from 'relative-meta-logic';

const { results, diagnostics } = evaluate(source, { file: 'kb.lino' });

for (const d of diagnostics) {
  console.error(formatDiagnostic(d, source));
}
```

`evaluate(code, options?)` never throws. Successful queries appear in
`results`; everything else is a `Diagnostic` in `diagnostics`.

### Rust

```rust
use rml::{evaluate, format_diagnostic};

let evaluation = evaluate(&source, Some("kb.lino"), None);
for diag in &evaluation.diagnostics {
    eprintln!("{}", format_diagnostic(diag, Some(&source)));
}
```

`evaluate` returns an `EvaluateResult { results, diagnostics }`. Internal
panics raised by the evaluator are caught and converted into diagnostics —
the panic hook is silenced for the duration of the call so a stack trace
never leaks to stderr.

## CLI output

Both CLIs (`node js/src/rml-links.mjs <file>` and `rml <file>`) print
diagnostics in this format:

```
kb.lino:3:1: E001: Unknown op: foo
(=: foo bar)
^
```

The exit code is `1` whenever any diagnostic is emitted, `0` otherwise.

## Error codes

| Code | When it fires |
|------|----------------------------------------------------------------|
| `E000` | Generic / unclassified error fallback. |
| `E001` | Reference to an undefined operator (`Unknown op: <name>`). Triggered, for example, by composing one operator from an unknown one: `(=: foo bar)`. |
| `E002` | Token-level parse error inside a single link (missing `)`, extra tokens, etc.). |
| `E003` | An operator definition has the right head but the wrong shape, e.g. `(=: a b c)` — the operator is real but the body is unsupported. |
| `E004` | Unknown aggregator selector, e.g. `(and: bogus_agg)`. Valid selectors are `avg`, `min`, `max`, `product`/`prod`, `probabilistic_sum`/`ps`. |
| `E005` | Empty meta-expression passed to a formalization helper. |
| `E006` | LiNo top-level parse failure, e.g. unclosed paren in the whole file. |

Codes are stable identifiers — they do not change between releases unless we
explicitly note a breaking change in the changelog. The accompanying
`message` field is free-form and may be improved at any time.

## Adding a new code

1. Add a new row to the table above with a brief trigger description.
2. In both implementations, throw/panic with the new code so the existing
   diagnostic dispatch picks it up:
   - JavaScript: `throw new RmlError('Eyyy', 'message');`
   - Rust: `panic!("recognisable prefix: …")` and extend
     `decode_panic_payload` in `rust/src/lib.rs` to map the prefix to
     `Eyyy`.
3. Add a test in `js/tests/diagnostics.test.mjs` and a mirrored test in
   `rust/tests/diagnostics_tests.rs` so drift between the two
   implementations fails CI.
