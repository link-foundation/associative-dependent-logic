// Quick smoke test for the proof-producing evaluator (issue #35).
// Exercises: structural equality, assigned probability, arithmetic,
// and-or composition, type witnesses, and the round-trip property.

import {
  evaluate,
  buildProof,
  keyOf,
  parseOne,
  tokenizeOne,
  isStructurallySame,
} from '../js/src/rml-links.mjs';

function show(label, src, opts) {
  const out = evaluate(src, opts);
  console.log(`\n=== ${label} ===`);
  console.log(`source: ${src.replace(/\n/g, ' | ')}`);
  console.log(`results:    ${JSON.stringify(out.results)}`);
  console.log(`proofs:     ${JSON.stringify(out.proofs)}`);
  if (Array.isArray(out.proofs)) {
    out.proofs.forEach((p, i) => {
      if (!p) {
        console.log(`  proof[${i}]: <none>`);
        return;
      }
      const printed = keyOf(p);
      const reparsed = parseOne(tokenizeOne(printed));
      const ok = isStructurallySame(p, reparsed) ? 'OK' : 'MISMATCH';
      console.log(`  proof[${i}] (${ok}): ${printed}`);
    });
  }
  if (out.diagnostics && out.diagnostics.length) {
    console.log(`diagnostics: ${out.diagnostics.map(d => `${d.code} ${d.message}`).join('; ')}`);
  }
  return out;
}

// 1. Structural equality from the issue example.
show('issue example: (? (a = a) with proof)', '(a: a is a)\n(? (a = a) with proof)');

// 2. Same query under the global flag, no inline keyword.
show('global withProofs', '(a: a is a)\n(? (a = a))', { withProofs: true });

// 3. Assigned-probability lookup.
show('assigned equality', '((a = a) has probability 0.7)\n(? (a = a))', { withProofs: true });

// 4. Arithmetic.
show('arithmetic +', '(? (1 + 2))', { withProofs: true });

// 5. And/or composition.
show('and/or composition',
  ['(a: a is a)', '(b: b is b)',
   '((a = a) has probability 1)', '((b = b) has probability 0)',
   '(? ((a = a) and (b = b)))'].join('\n'),
  { withProofs: true });

// 6. Composite both/neither.
show('composite both', '(? (both 1 and 0 and 1))', { withProofs: true });

// 7. Mixed: bare query + with-proof query in the same file.
show('mixed bare + with proof',
  '(? (1 + 1))\n(? (2 * 2) with proof)');

// 8. Negation (prefix operator).
show('prefix not', '(? (not 1))', { withProofs: true });

// 9. No proofs requested: must keep original {results, diagnostics} shape.
const noProofs = evaluate('(? 1)', {});
console.log(`\n=== no proofs ===\nshape: ${Object.keys(noProofs).join(',')}`);
