# `.lino` CST data model — Issue #138

This is the data-model artefact for the case study; it specifies how `.lino` is extended into a trivia-aware concrete syntax tree, and how each host language's CST is encoded inside it. The high-level rationale is in [`README.md`](./README.md); the per-requirement plans are in [`solution-plans.md`](./solution-plans.md).

## 1. Why a CST at all

A standard `.lino` AST today drops two pieces of source information:

1. **Comments** (`# foo`) — discarded during tokenisation, see [`ARCHITECTURE.md` § Stage 2](../../../ARCHITECTURE.md#stage-2-tokenization-and-ast-construction).
2. **Whitespace** (indentation, blank lines, CRLF vs LF) — never represented.

Both are required by issue #138 ("encode comments, every variable and other name, and even whitespace if needed"). The same problem has been solved twice in the ecosystem we sit next to:

- [Rust-analyzer / rowan](https://github.com/rust-analyzer/rowan) attaches whitespace and comments as **trivia tokens** to a lossless CST.
- [Roslyn / Swift libsyntax](https://github.com/rust-lang/rust-analyzer/issues/6584) does the same with a "trivia attached to the following non-trivia token" convention.

We adopt the rowan/Roslyn approach.

## 2. Three CST node kinds

| Node kind | Encodes | Children allowed | Round-trip |
|-----------|---------|------------------|------------|
| `lino-cst.list` | A parenthesised LiNo list. The existing AST list node. | List, token, trivia | Emit `(`, then children in order, then `)`. |
| `lino-cst.token` | A single non-trivia lexeme (identifier, number, operator, etc.). | None (leaf). | Emit the original lexeme. |
| `lino-cst.trivia` | A run of whitespace, or a `#`-comment, or any other byte sequence the parser considers non-significant. | None (leaf). | Emit the original bytes. |

A complete `.lino` file is a `lino-cst.list` whose children are top-level lists, with trivia leaves interspersed. By convention trivia attaches to the **following** non-trivia leaf.

Both modes share the same data type. The AST view simply hides `lino-cst.trivia` nodes and treats `lino-cst.token` leaves as bare strings, matching today's parser output.

## 3. CLI / API surface

### LiNo parser flag (R24)

```text
parseLinks(src)                       # AST view (default, today's behaviour)
parseLinks(src, { mode: 'cst' })      # CST view (new)
```

The Rust parser API mirrors this with `parse(&str)` and `parse_cst(&str)`.

### Tree-printer API

```text
printCst(tree) → string               # byte-faithful round-trip
printCst(tree, { canonicalise })      # opt-in canonicalisations
```

## 4. Host-language dialects

A *dialect* of the `.lino` CST is a set of conventional tags on `lino-cst.list` nodes. We propose four host dialects and one shared semantic dialect.

| Dialect | Top-level tag prefix | Authoritative source for the tag set |
|---------|----------------------|---------------------------------------|
| Rust | `lino-cst.rust.*` | rust-analyzer's [`ungrammar`](https://rust-analyzer.github.io//blog/2020/10/24/introducing-ungrammar.html) spec. |
| JavaScript | `lino-cst.js.*` | `swc_ecma_ast` enum constructors. |
| Lean 4 | `lino-cst.lean.*` | `Lean.SyntaxNodeKind` plus the parser's reserved `name` table. |
| Rocq | `lino-cst.rocq.*` | `coq-lsp`'s S-expression schema for `Vernacexpr` and `constr_expr`. |
| Shared (typed RML kernel) | `lino-cst.shared.*` | The existing typed `.lino` kernel documented in [`docs/case-studies/issue-13/`](../issue-13/). |

The dialect tag is the **first child** of every host-language `lino-cst.list` node. For example, a Rust function declaration becomes:

```lino
(lino-cst.list lino-cst.rust.fn
  (lino-cst.token "fn ")
  (lino-cst.list lino-cst.rust.name (lino-cst.token "foo"))
  (lino-cst.token "(")
  (lino-cst.list lino-cst.rust.param_list)
  (lino-cst.token ")")
  (lino-cst.trivia " ")
  (lino-cst.list lino-cst.rust.block (lino-cst.token "{") (lino-cst.token "}")))
```

This is verbose by design: the original Rust source is fully recoverable by concatenating leaves in order, and any tool that knows the `lino-cst.rust.*` vocabulary can semantically traverse the tree.

## 5. Generated dialect grammars

Each dialect ships a self-describing grammar file under `lib/lino-cst/<host>.lino`. The grammar is itself a `.lino` document that enumerates every node kind and its expected children. The shape is exactly what [`lib/programming-language/core.lino`](../../../lib/programming-language/core.lino) does for the typed RML fragment today.

The benefit of generating the grammar is that:

- We can run the existing RML evaluator on the grammar file and check it is well-formed.
- We can mechanically diff our grammar against the upstream source of truth (`ungrammar`, `swc_ecma_ast`, etc.) and gate CI on the diff.

## 6. Examples

The following examples are illustrative; the full set lives in `examples/round-trip-*.lino` after Phase L–O lands.

### Rust `fn id<T>(x: T) -> T { x }`

```lino
(lino-cst.list lino-cst.rust.fn
  (lino-cst.token "fn ")
  (lino-cst.list lino-cst.rust.name (lino-cst.token "id"))
  (lino-cst.list lino-cst.rust.generic_param_list
    (lino-cst.token "<")
    (lino-cst.list lino-cst.rust.type_param (lino-cst.token "T"))
    (lino-cst.token ">"))
  (lino-cst.token "(")
  (lino-cst.list lino-cst.rust.param
    (lino-cst.list lino-cst.rust.ident_pat (lino-cst.token "x"))
    (lino-cst.token ": ")
    (lino-cst.list lino-cst.rust.path_type (lino-cst.token "T")))
  (lino-cst.token ")")
  (lino-cst.token " -> ")
  (lino-cst.list lino-cst.rust.path_type (lino-cst.token "T"))
  (lino-cst.trivia " ")
  (lino-cst.list lino-cst.rust.block
    (lino-cst.token "{")
    (lino-cst.trivia " ")
    (lino-cst.list lino-cst.rust.path_expr (lino-cst.token "x"))
    (lino-cst.trivia " ")
    (lino-cst.token "}")))
```

### JavaScript `const id = (x) => x;`

```lino
(lino-cst.list lino-cst.js.variable_declaration
  (lino-cst.token "const ")
  (lino-cst.list lino-cst.js.variable_declarator
    (lino-cst.list lino-cst.js.identifier (lino-cst.token "id"))
    (lino-cst.token " = ")
    (lino-cst.list lino-cst.js.arrow_function_expression
      (lino-cst.token "(")
      (lino-cst.list lino-cst.js.identifier (lino-cst.token "x"))
      (lino-cst.token ") => ")
      (lino-cst.list lino-cst.js.identifier (lino-cst.token "x"))))
  (lino-cst.token ";"))
```

### Lean 4 `def id {α : Type} (x : α) : α := x`

```lino
(lino-cst.list lino-cst.lean.command_definition
  (lino-cst.token "def ")
  (lino-cst.list lino-cst.lean.decl_id (lino-cst.token "id"))
  (lino-cst.trivia " ")
  (lino-cst.list lino-cst.lean.implicit_binder
    (lino-cst.token "{") (lino-cst.token "α") (lino-cst.token " : ")
    (lino-cst.list lino-cst.lean.sort (lino-cst.token "Type"))
    (lino-cst.token "}"))
  (lino-cst.trivia " ")
  (lino-cst.list lino-cst.lean.binder
    (lino-cst.token "(") (lino-cst.token "x") (lino-cst.token " : ")
    (lino-cst.list lino-cst.lean.var (lino-cst.token "α"))
    (lino-cst.token ")"))
  (lino-cst.token " : ")
  (lino-cst.list lino-cst.lean.var (lino-cst.token "α"))
  (lino-cst.token " := ")
  (lino-cst.list lino-cst.lean.var (lino-cst.token "x")))
```

### Rocq `Definition id {A : Type} (x : A) : A := x.`

```lino
(lino-cst.list lino-cst.rocq.vernac_definition
  (lino-cst.token "Definition ")
  (lino-cst.list lino-cst.rocq.ident (lino-cst.token "id"))
  (lino-cst.trivia " ")
  (lino-cst.list lino-cst.rocq.implicit_binder
    (lino-cst.token "{") (lino-cst.token "A") (lino-cst.token " : ")
    (lino-cst.list lino-cst.rocq.sort (lino-cst.token "Type"))
    (lino-cst.token "}"))
  (lino-cst.trivia " ")
  (lino-cst.list lino-cst.rocq.binder
    (lino-cst.token "(") (lino-cst.token "x") (lino-cst.token " : ")
    (lino-cst.list lino-cst.rocq.var (lino-cst.token "A"))
    (lino-cst.token ")"))
  (lino-cst.token " : ")
  (lino-cst.list lino-cst.rocq.var (lino-cst.token "A"))
  (lino-cst.token " := ")
  (lino-cst.list lino-cst.rocq.var (lino-cst.token "x"))
  (lino-cst.token "."))
```

## 7. Round-trip walk

The round-trip printer is the same five lines for every dialect:

```text
def print(node):
    if node is token or trivia:
        emit(node.text)
    elif node is list:
        for child in node.children:
            print(child)
```

If `parse(src).children` preserves source order, that is the entire correctness proof.

## 8. Canonicalisations (documented)

In a small set of cases the printer is allowed to canonicalise. Each one is gated behind an explicit option so the default remains byte-faithful.

| Canonicalisation | Default | Notes |
|------------------|---------|-------|
| Trailing newline at end of file | Preserved | We never add or remove one unless asked. |
| CRLF → LF | Preserved | We may add a `--normalize-line-endings` flag in a later phase. |
| Tabs vs spaces | Preserved | Same. |
| Lean macro expansion | Off | `--expand-macros` triggers Lean to elaborate; the result is no longer round-trip-equal. |
| Rocq notation expansion | Off | Same idea via `coq-lsp`. |

## 9. Open questions

See [`risks-and-open-questions.md`](./risks-and-open-questions.md).
