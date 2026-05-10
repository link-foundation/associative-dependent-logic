# Probabilistic Reasoning Tutorial

Probabilistic RML uses the same numeric truth channel as fuzzy logic, but the
numbers are read as event probabilities. The key change is the choice of
aggregators: independent conjunction uses multiplication, and disjunction uses
probabilistic sum.

Run the full example:

```bash
node js/src/rml-links.mjs examples/bayesian-network.lino
```

The source is [`../../examples/bayesian-network.lino`](../../examples/bayesian-network.lino).

## 1. Configure Probability Operators

The example starts with four event names:

```lino
(cloudy: cloudy is cloudy)
(sprinkler: sprinkler is sprinkler)
(rain: rain is rain)
(wet-grass: wet-grass is wet-grass)
```

Then it configures the operators:

```lino
(and: product)
(or: probabilistic_sum)
```

For independent events:

- `product` reads `A and B` as `P(A) * P(B)`.
- `probabilistic_sum` reads `A or B` as `1 - (1 - P(A)) * (1 - P(B))`.

These are aggregator choices, not special syntax. The evaluator already knows
how to ask queries over assignments.

## 2. Assign Marginal Probabilities

The example assigns event probabilities directly:

```lino
((cloudy = true) has probability 0.5)
((sprinkler = true) has probability 0.3)
((rain = true) has probability 0.5)
```

Direct queries return those numbers:

```lino
(? (cloudy = true))
(? (sprinkler = true))
(? (rain = true))
```

## 3. Compute Joint and Union Probabilities

Conjunction becomes a joint probability for independent events:

```lino
(? ((cloudy = true) and (rain = true)))
```

The result is `0.25`, because `0.5 * 0.5 = 0.25`.

Disjunction becomes probabilistic union:

```lino
(? ((sprinkler = true) or (rain = true)))
```

The result is `0.65`, because `1 - (1 - 0.3) * (1 - 0.5) = 0.65`.

The prefix form works for more than two arguments:

```lino
(? (and (cloudy = true) (sprinkler = true) (rain = true)))
```

This multiplies all three probabilities.

## 4. Encode a Network Shape

The example comments describe the classic sprinkler network:

```text
cloudy -> sprinkler -> wet-grass
cloudy -> rain      -> wet-grass
```

Plain links are enough to name the events, and probability assignments are
enough to answer the simple marginal and independent-event queries. Conditional
probability tables can also be recorded as links when a program needs them.

The example closes with an arithmetic chain-rule calculation:

```lino
(? (((0.99 * 0.15) + (0.9 * 0.15)) + ((0.9 * 0.35) + (0.01 * 0.35))))
```

The evaluator treats arithmetic as numeric computation and prints `0.602`.

## 5. Reuse the Bayesian Library

The Bayesian helper library lives in
[`../../lib/probabilistic/bayesian.lino`](../../lib/probabilistic/bayesian.lino).
It packages network descriptions, priors, conditionals, joint/union templates,
and Bayes calculations under the `bayesian` namespace:

```lino
(import "lib/probabilistic/bayesian.lino" as bn)

(bn.network sprinkler-network
  (nodes cloudy sprinkler rain wet-grass)
  (edges (bn.edge cloudy sprinkler)
         (bn.edge cloudy rain)
         (bn.edge sprinkler wet-grass)
         (bn.edge rain wet-grass)))

(bn.prior rain 0.5)
(bn.prior sprinkler 0.3)

(? (bn.joint (rain = true) (sprinkler = true)))
(? (bn.union (rain = true) (sprinkler = true)))
(? (bn.bayes 0.95 0.01 0.059))
```

The library does not hide the logic. It gives names to common link patterns so
larger probabilistic files stay readable.

## What To Remember

Probabilistic RML is fuzzy RML with probability-oriented aggregators:

1. Use `product` for independent conjunction.
2. Use `probabilistic_sum` for independent union.
3. Use assignments for priors and arithmetic for explicit calculations.
4. Move repeated network shapes into templates from the Bayesian library.

The next tutorial introduces typed terms so links can carry richer structure
than truth values alone.
