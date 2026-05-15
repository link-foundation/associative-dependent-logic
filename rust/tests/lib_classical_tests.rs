// Tests for the classical standard library (issue #67).
// Mirrors js/tests/lib-classical.test.mjs so the LiNo library surface stays
// identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-classical-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_boolean_operators_through_import_alias() {
    let out = evaluate_from_root(
        r#"
(import "lib/classical/core.lino" as cl)
(? (cl.and true false))
(? (cl.or true false))
(? (cl.not true))
(? (cl.not false))
(? (cl.or p (cl.not p)))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(0.0),
            RunResult::Num(1.0),
            RunResult::Num(0.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_classical_laws_as_reusable_templates() {
    let out = evaluate_from_root(
        r#"
(import "lib/classical/core.lino" as cl)
(? (cl.excluded-middle true))
(? (cl.excluded-middle false))
(? (cl.double-negation true))
(? (cl.double-negation false))
(? (cl.de-morgan-not-and true false))
(? (cl.de-morgan-not-or true false))
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

#[test]
fn exports_natural_deduction_rule_schemas_as_tautology_templates() {
    let out = evaluate_from_root(
        r#"
(import "lib/classical/core.lino" as cl)
(? (cl.implies true false))
(? (cl.implies false false))
(? (cl.and-introduction true false))
(? (cl.and-elimination-left true false))
(? (cl.and-elimination-right true false))
(? (cl.or-introduction-left false true))
(? (cl.or-introduction-right true false))
(? (cl.modus-ponens true false))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(0.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}
