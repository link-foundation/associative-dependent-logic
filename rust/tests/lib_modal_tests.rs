// Tests for the modal standard library (issue #71).
// Mirrors js/tests/lib-modal.test.mjs so the LiNo library surface stays
// identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-modal-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_modal_operators_through_import_alias() {
    let out = evaluate_from_root(
        r#"
(import "lib/modal/core.lino" as ml)
(? ((ml.necessarily p) = (necessarily p)))
(? ((ml.possibly p) = (possibly p)))
(? (ml.implies true false))
(? (ml.implies false false))
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
fn exports_k_t_s4_and_s5_axiom_schemas_as_reusable_templates() {
    let out = evaluate_from_root(
        r#"
(import "lib/modal/core.lino" as ml)
(? ((ml.axiom-k p q) =
     (ml.implies
       (ml.necessarily (ml.implies p q))
       (ml.implies (ml.necessarily p) (ml.necessarily q)))))
(? ((ml.axiom-t p) =
     (ml.implies (ml.necessarily p) p)))
(? ((ml.axiom-s4 p) =
     (ml.implies (ml.necessarily p)
       (ml.necessarily (ml.necessarily p)))))
(? ((ml.axiom-s5 p) =
     (ml.implies (ml.possibly p)
       (ml.necessarily (ml.possibly p)))))
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
fn exports_kripke_frame_interpretation_templates() {
    let out = evaluate_from_root(
        r#"
(import "lib/modal/core.lino" as ml)
(? ((ml.holds current p) = (holds current p)))
(? ((ml.accessible current next) = (accessible current next)))
(? ((ml.necessarily-at current p) =
     (forall (World accessible-world)
       (ml.implies
         (ml.accessible current accessible-world)
         (ml.holds accessible-world p)))))
(? ((ml.possibly-at current p) =
     (exists (World accessible-world)
       (modal.and
         (ml.accessible current accessible-world)
         (ml.holds accessible-world p)))))
(? ((ml.valid p) =
     (forall (World possible-world) (ml.holds possible-world p))))
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
        ]
    );
}

#[test]
fn exports_kripke_frame_conditions_for_modal_systems() {
    let out = evaluate_from_root(
        r#"
(import "lib/modal/core.lino" as ml)
(? ((ml.frame-k modal.accessible) = true))
(? ((ml.frame-t modal.accessible) = (ml.reflexive-frame modal.accessible)))
(? ((ml.frame-s4 modal.accessible) =
     (modal.and
       (ml.reflexive-frame modal.accessible)
       (ml.transitive-frame modal.accessible))))
(? ((ml.frame-s5 modal.accessible) =
     (modal.and
       (ml.reflexive-frame modal.accessible)
       (modal.and
         (ml.symmetric-frame modal.accessible)
         (ml.transitive-frame modal.accessible)))))
(? ((ml.euclidean-frame modal.accessible) =
     (forall (World source)
       (forall (World left)
         (forall (World right)
           (ml.implies
             (modal.and
               (ml.accessible source left)
               (ml.accessible source right))
             (ml.accessible left right)))))))
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
        ]
    );
}
