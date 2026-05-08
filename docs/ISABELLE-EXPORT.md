# Isabelle Export

RML can export a deliberately small typed LiNo fragment to Isabelle/HOL:

```bash
node js/src/rml-links.mjs export isabelle examples/isabelle-typed-fragment.lino -o Isabelle_Typed_Fragment.thy
```

The npm package exposes the same command through the `rml` binary:

```bash
rml export isabelle input.lino -o Output.thy
```

Use `--theory <Name>` when the generated theory name should differ from the
output file stem.

## Supported Subset

The exporter accepts source files made from these top-level LiNo forms:

| RML form | Isabelle/HOL output |
|----------|---------------------|
| `(Name: (Type 0) Name)` or `(Name: Type Name)` | `typedecl rml_name` |
| `(value: TypeName value)` | `consts rml_value :: "rml_type_name"` |
| `(f: (Pi (A x) B))` | `consts rml_f :: "rml_a => rml_b"` |
| `(f: lambda (A x) body)` | `definition rml_f :: "rml_a => ..."` |
| `(inductive T (constructor c) ...)` | `datatype rml_t = rml_c ...` |
| `(? ...)` and top-level `(Type n)` universe markers | ignored |

Names are sanitized into stable Isabelle identifiers with an `rml_` prefix.
For example, `true-val` becomes `rml_true_val`, and `Natural-rec` would become
`rml_natural_rec`.

## Boundaries

The target is Isabelle/HOL, so the exported fragment is simply typed. A
`Pi` type is accepted only when its codomain does not mention the bound
variable. For example, `(Pi (Natural n) Natural)` exports to
`rml_natural => rml_natural`, while `(Pi (Natural n) (Vector n))` is rejected
because HOL function types cannot represent that dependency directly.

The exporter rejects probabilistic and runtime-evaluator forms:

- `((expr) has probability p)`
- `(range: lo hi)` and `(valence: n)`
- operator redefinitions such as `(and: avg)` or `(!=: not =)`
- numeric symbol priors such as `(a: 0.7)`
- recursive relation, tactic, template, import, namespace, and coinductive
  declarations

These forms remain valid RML programs; they are just outside the certificate
fragment exported to Isabelle/HOL.

## Test Fixture

The fixture pair
[`examples/isabelle-typed-fragment.lino`](../examples/isabelle-typed-fragment.lino)
and
[`examples/isabelle-typed-fragment.thy`](../examples/isabelle-typed-fragment.thy)
is used by `js/tests/isabelle-export.test.mjs` as a round-trip snapshot for
the supported subset.
