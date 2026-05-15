const coreDemo = `# Core RML demo
(a: a is a)

(!=: not =)
(and: avg)
(or: max)

((a = a) has probability 1)
((a != a) has probability 0)

(? ((a = a) and (a != a)))
(? ((a = a) or (a != a)))
`;

const classicalLogic = `# Classical Boolean logic
(valence: 2)
(and: min)
(or: max)

(p: p is p)
(q: q is q)

((p = true) has probability 1)
((q = true) has probability 0)

(? (p = true))
(? (q = true))
(? (not (p = true)))
(? ((p = true) and (q = true)))
(? ((p = true) or (not (p = true))))
`;

const probability = `# Probabilistic connectives
(and: product)
(or: probabilistic_sum)

(rain: rain is rain)
(umbrella: umbrella is umbrella)
(wet: wet is wet)

((rain = true) has probability 0.3)
((umbrella = true) has probability 0.6)
((wet = true) has probability 0.4)

(? (rain = true))
(? (umbrella = true))
(? ((rain = true) and (umbrella = true)))
(? ((rain = true) or (umbrella = true)))
(? (not (rain = true)))
(? (and (rain = true) (umbrella = true) (wet = true)))
`;

const typedLambda = `# Typed lambda fragment
(Term: (Type 0) Term)

(identity: lambda (Term x) x)

(? (identity of (Pi (Term x) Term)))
(? (apply identity 0.42))
(? ((apply identity 0.42) = 0.42))
`;

export const PLAYGROUND_EXAMPLES = [
  {
    id: 'core-demo',
    title: 'Core demo',
    source: coreDemo,
  },
  {
    id: 'classical-logic',
    title: 'Classical logic',
    source: classicalLogic,
  },
  {
    id: 'probability',
    title: 'Probability',
    source: probability,
  },
  {
    id: 'typed-lambda',
    title: 'Typed lambda',
    source: typedLambda,
  },
];

export const defaultExampleId = PLAYGROUND_EXAMPLES[0].id;

export function findExample(id) {
  return PLAYGROUND_EXAMPLES.find((example) => example.id === id) || PLAYGROUND_EXAMPLES[0];
}
