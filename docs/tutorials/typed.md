# Typed LiNo Tutorial

The previous tutorials treated links as propositions with numeric truth values.
Typed LiNo adds another layer: links can also record that a term belongs to a
type, that a function has a dependent product type, or that applying a lambda
reduces to a value.

Run the full example:

```bash
node js/src/rml-links.mjs examples/dependent-types.lino
```

The source is [`../../examples/dependent-types.lino`](../../examples/dependent-types.lino).
The reference rules are in [`../../docs/KERNEL.md`](../../docs/KERNEL.md).

## 1. Declare Types As Links

The example starts with the self-referential root type:

```lino
(Type: Type Type)
```

RML allows this because it is a dynamic, many-valued system. Paradoxical or
circular facts do not crash the evaluator; they are handled by the configured
truth semantics.

User types use the same declaration pattern:

```lino
(Natural: Type Natural)
(Boolean: Type Boolean)
```

Read `(Natural: Type Natural)` as "install `Natural` as a term of type `Type`."

## 2. Declare Typed Terms

Constructors and values use prefix type notation:

```lino
(zero: Natural zero)
(true-val: Boolean true-val)
(false-val: Boolean false-val)
```

After these declarations, type-membership queries can ask whether a term has a
type:

```lino
(? (zero of Natural))
(? (Natural of Type))
(? (Type of Type))
```

The output is a truth value. In the default range, successful type membership
prints `1`.

## 3. Use Dependent Product Types

`Pi` forms a dependent product. The example declares a successor function:

```lino
(succ: (Pi (Natural n) Natural))
```

Read this as a function that accepts a `Natural` named `n` and returns a
`Natural`. When the result type does not depend on the binder, this is the
ordinary function type pattern.

## 4. Define and Apply Lambdas

The example defines identity as a lambda:

```lino
(identity: lambda (Natural x) x)
```

Then it applies the lambda:

```lino
(? (apply identity 0.7))
```

`apply` runs the kernel's capture-avoiding substitution. In this case the body
is just `x`, so the query reduces to the argument and prints `0.7`.

The same mechanism supports inline lambdas and higher-order examples. The
HOAS-focused example in
[`../../examples/lambda-calculus.lino`](../../examples/lambda-calculus.lino)
shows how object-language binders can be represented with RML's own binders.

## 5. Ask For Types

Type queries use `(type of ...)`:

```lino
(? (type of zero))
```

This prints `Natural` after the earlier `(zero: Natural zero)` declaration.

The same typed layer coexists with truth assignments:

```lino
((zero = zero) has probability 1)
(? (zero = zero))
```

RML does not switch languages when it moves from logic to types. Both are link
facts in the same environment.

## 6. Use Universe Levels When Needed

The example also shows stratified universes:

```lino
(Type 0)
(Type 1)
(? ((Type 0) of (Type 1)))
```

The typed kernel treats `(Type N)` as belonging to `(Type N+1)`. This form is
useful for fragments that need Lean/Rocq-style universe structure, while
`(Type: Type Type)` remains available for self-referential RML examples.

## 7. Read the Kernel Reference

After the example, read [`../../docs/KERNEL.md`](../../docs/KERNEL.md) for the
formal surface:

- Type declarations and typed term declarations.
- Universe hierarchy rules.
- `Pi` formation.
- `lambda` formation.
- `apply` reduction.
- Convertibility and normalization.
- The bidirectional checker and its diagnostics.

## What To Remember

Typed LiNo adds structure without leaving the link model:

1. Types are declared as links.
2. Typed terms use `(name: TypeName name)`.
3. Functions use `Pi`, `lambda`, and `apply`.
4. Type questions are ordinary queries, either `(term of Type)` or
   `(type of term)`.

The next tutorial uses typed declarations and relation clauses as input to
metatheorem checks.
