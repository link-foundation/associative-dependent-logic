// Tests for the provability standard library (issue #72).
// Mirrors js/tests/lib-provability.test.mjs so the LiNo library surface stays
// identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-provability-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_provability_operators_through_import_alias() {
    let out = evaluate_from_root(
        r#"
(import "lib/provability/core.lino" as pr)
(? ((pr.provability-of p) = (provability-of p)))
(? ((pr.consistency-of p) =
     (provability.not
       (provability.provability-of (provability.not p)))))
(? (pr.implies true false))
(? (pr.implies false false))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(0.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_gl_axiom_schemas_as_reusable_templates() {
    let out = evaluate_from_root(
        r#"
(import "lib/provability/core.lino" as pr)
(? ((pr.axiom-k p q) =
     (pr.implies
       (pr.provability-of (pr.implies p q))
       (pr.implies
         (pr.provability-of p)
         (pr.provability-of q)))))
(? ((pr.axiom-lob p) =
     (pr.implies
       (pr.provability-of
         (pr.implies (pr.provability-of p) p))
       (pr.provability-of p))))
(? ((pr.axiom-gl p) = (pr.axiom-lob p)))
(? ((pr.necessitation-rule p) =
     (pr.implies p (pr.provability-of p))))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_interpretability_fragment_schemas() {
    let out = evaluate_from_root(
        r#"
(import "lib/provability/core.lino" as pr)
(? ((pr.interprets source target) = (interprets source target)))
(? ((pr.axiom-j1 source target) =
     (pr.implies
       (pr.provability-of (pr.implies source target))
       (pr.interprets source target))))
(? ((pr.axiom-j2 source middle target) =
     (pr.implies
       (provability.and
         (pr.interprets source middle)
         (pr.interprets middle target))
       (pr.interprets source target))))
(? ((pr.axiom-j3 left right target) =
     (pr.implies
       (provability.and
         (pr.interprets left target)
         (pr.interprets right target))
       (pr.interprets (provability.or left right) target))))
(? ((pr.axiom-j4 source target) =
     (pr.implies
       (pr.interprets source target)
       (pr.implies
         (pr.consistency-of source)
         (pr.consistency-of target)))))
(? ((pr.axiom-j5 source) =
     (pr.interprets (pr.consistency-of source) source)))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}
