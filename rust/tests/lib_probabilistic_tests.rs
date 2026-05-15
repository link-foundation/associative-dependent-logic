// Tests for the probabilistic and Belnap standard libraries (issue #77).
// Mirrors js/tests/lib-probabilistic.test.mjs so the LiNo library surface
// stays identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-probabilistic-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_bayesian_network_helpers_through_the_issue_import_surface() {
    let out = evaluate_from_root(
        r#"
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
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(0.3),
            RunResult::Num(0.18),
            RunResult::Num(0.72),
            RunResult::Num(0.7),
            RunResult::Num(0.161016949153),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_fuzzy_membership_and_fuzzy_control_helpers() {
    let out = evaluate_from_root(
        r#"
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
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(0.8),
            RunResult::Num(0.6),
            RunResult::Num(0.6),
            RunResult::Num(0.19999999999999996),
            RunResult::Num(0.48),
            RunResult::Num(0.6),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_belnap_bilattice_helpers_for_truth_and_knowledge_orders() {
    let out = evaluate_from_root(
        r#"
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
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(0.0),
            RunResult::Num(1.0),
            RunResult::Num(0.5),
            RunResult::Num(0.0),
            RunResult::Num(0.5),
            RunResult::Num(0.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_a_paradox_catalogue_with_midpoint_fixed_point_helpers() {
    let out = evaluate_from_root(
        r#"
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
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(0.5),
            RunResult::Num(0.5),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_balanced_range_paradox_midpoint_helpers() {
    let out = evaluate_from_root(
        r#"
(range: -1 1)
(import "lib/probabilistic/paradoxes.lino" as px)
(s: s is s)
(px.balanced-midpoint (px.liar s))
(? (px.liar s))
(? (px.fixed-point (px.liar s)))
"#,
    );

    assert_clean(&out);
    assert_eq!(out.results, vec![RunResult::Num(0.0), RunResult::Num(0.0)]);
}
