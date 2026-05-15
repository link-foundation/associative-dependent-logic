import { evaluate } from '../src/rml-links.mjs';

// Lambda calculus example using HOAS pattern.
// Object-language `\x.x` is encoded as host-language `lambda` with a typed parameter.
// Object-language `forall x:A. B(x)` is encoded as `forall (A x) B(x)` -> desugared to Pi.
const src = `
(Term: (Type 0) Term)
(Var: (Type 0) Var)

# Identity function and its dependent type
(identity: lambda (Term x) x)
(? (type of identity))
(identity-type: (forall (Term x) Term))
(? (type of identity-type))

# Apply identity — round-trip beta reduction
(? (apply identity 0.42))

# Polymorphic-style Pi using forall surface
(succ: (forall (Term n) Term))
(? (succ of (Pi (Term n) Term)))
`;
const out = evaluate(src);
console.log(JSON.stringify(out, null, 2));
