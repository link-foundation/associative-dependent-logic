// Tests for the probabilistic and Belnap standard libraries (issue #77).
// Mirrors rust/tests/lib_probabilistic_tests.rs so the LiNo library surface
// stays identical across both implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluate } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const virtualRootFile = join(repoRoot, 'inline-probabilistic-test.lino');

function evaluateFromRoot(source) {
  return evaluate(source, { file: virtualRootFile });
}

function assertClean(out) {
  assert.deepStrictEqual(out.diagnostics, []);
}

describe('lib/probabilistic/*.lino', () => {
  it('exports Bayesian-network helpers through the issue import surface', () => {
    const out = evaluateFromRoot(`
(import "lib/probabilistic/bayesian.lino" as bn)
(bn.prior rain 0.3)
(bn.prior sprinkler 0.6)
(? (rain = true))
(? (bn.joint (rain = true) (sprinkler = true)))
(? (bn.union (rain = true) (sprinkler = true)))
(? (bn.complement (rain = true)))
(? (bn.bayes 0.95 0.01 0.059))
(? ((bn.edge cloudy rain) =
     (bayesian.directed-edge cloudy rain)))
(? ((bn.network sprinkler-network
       (nodes cloudy rain sprinkler)
       (edges (bn.edge cloudy rain) (bn.edge cloudy sprinkler))) =
     (bayesian.network-description sprinkler-network
       (nodes cloudy rain sprinkler)
       (edges
         (bayesian.directed-edge cloudy rain)
         (bayesian.directed-edge cloudy sprinkler)))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [
      0.3,
      0.18,
      0.72,
      0.7,
      0.161016949153,
      1,
      1,
    ]);
  });

  it('exports fuzzy membership and fuzzy-control helpers', () => {
    const out = evaluateFromRoot(`
(import "lib/probabilistic/fuzzy.lino" as fz)
(fz.membership temperature hot 0.8)
(fz.membership humidity wet 0.6)
(? (fz.degree temperature hot))
(? (fz.all (fz.degree temperature hot) (fz.degree humidity wet)))
(? (fz.any 0.2 (fz.degree humidity wet)))
(? (fz.complement (fz.degree temperature hot)))
(? (fz.weighted-output 0.6 0.8))
(? (fz.centroid2 0.6 0.8 0.4 0.3))
(? ((fz.control-action fan-fast (fz.degree temperature hot) 0.8) =
     (fuzzy.control-action-description fan-fast (temperature = hot) 0.8)))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [
      0.8,
      0.6,
      0.6,
      0.19999999999999996,
      0.48,
      0.6,
      1,
    ]);
  });

  it('exports Belnap bilattice helpers for truth and knowledge orders', () => {
    const out = evaluateFromRoot(`
(import "lib/probabilistic/belnap.lino" as bl)
(? (bl.truth-meet true false))
(? (bl.truth-join true false))
(? (bl.contradiction true false))
(? (bl.gap true false))
(? (bl.knowledge-join true false))
(? (bl.knowledge-meet true false))
(? ((bl.bilattice-value contradiction
       (truth-evidence true)
       (false-evidence true)) =
     (belnap.value contradiction
       (truth-evidence true)
       (false-evidence true))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [0, 1, 0.5, 0, 0.5, 0, 1]);
  });

  it('exports a paradox catalogue with midpoint fixed-point helpers', () => {
    const out = evaluateFromRoot(`
(import "lib/probabilistic/paradoxes.lino" as px)
(s: s is s)
(px.midpoint (px.liar s))
(? (px.liar s))
(? (px.fixed-point (px.liar s)))
(? ((px.russell member-of R) =
     (= (member-of R R)
        (paradoxes.not (member-of R R)))))
(? ((px.barber shaves barber alice) =
     (= (shaves barber alice)
        (paradoxes.not (shaves alice alice)))))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [0.5, 0.5, 1, 1]);
  });

  it('exports balanced-range paradox midpoint helpers', () => {
    const out = evaluateFromRoot(`
(range: -1 1)
(import "lib/probabilistic/paradoxes.lino" as px)
(s: s is s)
(px.balanced-midpoint (px.liar s))
(? (px.liar s))
(? (px.fixed-point (px.liar s)))
`);

    assertClean(out);
    assert.deepStrictEqual(out.results, [0, 0]);
  });
});
