# Syntax Comparison: Lean 4 / Rocq / LiNo

This document shows how core dependent type theory constructs look in Lean 4, Rocq, and our LiNo encoding. Items marked ✅ are implemented in v0.7.0; items marked 📋 are proposed for future phases.

## Core Constructs

### 1. Universe Sorts ✅

| Concept | Lean 4 | Rocq | LiNo |
|---------|--------|------|------|
| Type of small types | `Type` or `Type 0` | `Set` or `Type` | `(Type 0)` |
| Type of types | `Type 1` | `Type` | `(Type 1)` |
| Propositions | `Prop` | `Prop` | `(Prop)` or `(Type 0)` |

### 2. Variable Declarations ✅

```lean
-- Lean 4
variable (x : Nat)
```

```coq
(* Rocq *)
Variable x : nat.
```

```lino
# LiNo (Implemented)
(x: Nat)
```

### 3. Function Types (Non-Dependent) ✅

```lean
-- Lean 4
Nat → Bool
```

```coq
(* Rocq *)
nat -> bool
```

```lino
# LiNo (Implemented)
(Pi (_: Nat) Bool)
```

### 4. Dependent Product (Π-Type / Forall) ✅

```lean
-- Lean 4
(n : Nat) → Vec n Bool
-- or
∀ (n : Nat), Vec n Bool
```

```coq
(* Rocq *)
forall (n : nat), Vec n bool
```

```lino
# LiNo (Implemented)
(Pi (n: Nat) (Vec n Bool))
```

### 5. Lambda Abstraction ✅

```lean
-- Lean 4
fun (x : Nat) => x + 1
```

```coq
(* Rocq *)
fun (x : nat) => x + 1
```

```lino
# LiNo (Implemented)
(lam (x: Nat) (+ x 1))
```

### 6. Function Application ✅

```lean
-- Lean 4
f x
f x y z
```

```coq
(* Rocq *)
f x
f x y z
```

```lino
# LiNo (Implemented)
(app f x)
```

### 7. Let Binding 📋

```lean
-- Lean 4
let x : Nat := 5; x + 1
```

```coq
(* Rocq *)
let x : nat := 5 in x + 1
```

```lino
# LiNo (Proposed)
(let (x: Nat) 5 (+ x 1))
```

### 8. Inductive Type Definitions 📋

```lean
-- Lean 4
inductive Nat where
  | zero : Nat
  | succ : Nat → Nat
```

```coq
(* Rocq *)
Inductive nat : Set :=
  | zero : nat
  | succ : nat -> nat.
```

```lino
# LiNo (Proposed)
(inductive Nat (Type 0)
  (zero: Nat)
  (succ: (Pi (_: Nat) Nat)))
```

### 9. Pattern Matching / Recursion 📋

```lean
-- Lean 4
def add : Nat → Nat → Nat
  | .zero, b => b
  | .succ a, b => .succ (add a b)
```

```coq
(* Rocq *)
Fixpoint add (a b : nat) : nat :=
  match a with
  | zero => b
  | succ n => succ (add n b)
  end.
```

```lino
# LiNo (Proposed)
(def add: (Pi (a: Nat) (Pi (b: Nat) Nat))
  (lam (a: Nat) (lam (b: Nat)
    (match a
      (zero => b)
      ((succ n) => (succ (add n b)))))))
```

### 10. Propositions and Proofs 📋

```lean
-- Lean 4
theorem plus_zero (n : Nat) : n + 0 = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Nat.add_succ, ih]
```

```coq
(* Rocq *)
Theorem plus_zero : forall n : nat, n + 0 = n.
Proof.
  intro n. induction n.
  - reflexivity.
  - simpl. rewrite IHn. reflexivity.
Qed.
```

```lino
# LiNo (Proposed) - explicit proof term
(theorem plus_zero
  : (Pi (n: Nat) (= (+ n zero) n))
  (lam (n: Nat)
    (match n
      (zero => (refl zero))
      ((succ m) => (cong succ (plus_zero m))))))
```

### 11. Dependent Sum (Σ-Type / Exists) 📋

```lean
-- Lean 4
(n : Nat) × (n > 0)
-- or
∃ n : Nat, n > 0
```

```coq
(* Rocq *)
{ n : nat | n > 0 }
-- or
exists n : nat, n > 0
```

```lino
# LiNo (Proposed)
(Sigma (n: Nat) (> n 0))
# or:
(exists (n: Nat) (> n 0))
```

## Integration with ADL Probabilistic Logic

### Probabilistic Propositions

```lino
# Classical proposition (probability 1 or 0)
((= (+ zero (succ zero)) (succ zero)) has probability 1)

# Probabilistic proposition
((raining today) has probability 0.7)

# Type-checked probabilistic proposition
(raining: (PProp 0.7))

# Query type
(?type (lam (x: Nat) (+ x 1)))
# Output: (Pi (x: Nat) Nat)

# Query probability
(? (raining today))
# Output: 0.7
```

### Mixed Mode: Types + Probabilities

```lino
# Define a type
(inductive Weather (Type 0)
  (sunny: Weather)
  (rainy: Weather)
  (cloudy: Weather))

# Assign probabilities to inhabitants
((sunny has probability 0.3) in Weather)
((rainy has probability 0.5) in Weather)
((cloudy has probability 0.2) in Weather)

# Dependent type with probabilistic index
(forecast: (Pi (w: Weather) (PProp (weather_prob w))))
```

## Encoding in Pure Associative Links

For completeness, here is how the above would look using only the associative link model (Option D from the main document):

```lino
# Define meta-concepts as links
(type-of: type-of is type-of)
(reduces-to: reduces-to is reduces-to)
(has-constructor: has-constructor is has-constructor)

# Universe hierarchy
(Type-0: Type-0 is Type-0)
(Type-1: Type-1 is Type-1)
((Type-0 type-of Type-1) has probability 1)

# Nat type
(Nat: Nat is Nat)
((Nat type-of Type-0) has probability 1)

# Constructors
(zero: zero is zero)
((zero type-of Nat) has probability 1)
((Nat has-constructor zero) has probability 1)

(succ: succ is succ)
((succ type-of (Pi Nat Nat)) has probability 1)
((Nat has-constructor succ) has probability 1)

# A function: add
(add: add is add)
((add type-of (Pi Nat (Pi Nat Nat))) has probability 1)

# Reduction rules as links
(((add zero b) reduces-to b) has probability 1)
(((add (succ n) b) reduces-to (succ (add n b))) has probability 1)

# Type checking is querying the network
(? (zero type-of Nat))          # -> 1 (true)
(? (succ type-of (Pi Nat Nat))) # -> 1 (true)
(? ((succ zero) type-of Nat))   # -> 1 (true, by reduction)
```

This encoding treats the entire type system as data within the associative network, which aligns with the Links Theory principle that everything is a link.
