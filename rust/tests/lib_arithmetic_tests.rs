// Tests for the arithmetic standard library (issue #74).
// Mirrors js/tests/lib-arithmetic.test.mjs so the LiNo library surface stays
// identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-arithmetic-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_peano_naturals_and_issue_surface_arithmetic_through_import_alias() {
    let out = evaluate_from_root(
        r#"
(import "lib/arithmetic/core.lino" as ar)
(? (zero of Natural))
(? (type of ar.succ))
(? ((plus zero zero) = zero))
(? (less-than zero (succ zero)))
(? (less-than-or-equal zero zero))
(? (ar.less-than zero (succ zero)))
(? (ar.less-than-or-equal zero zero))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(1.0),
            RunResult::Type("(Pi (Natural n) Natural)".to_string()),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_peano_axiom_schemas_as_reusable_templates() {
    let out = evaluate_from_root(
        r#"
(import "lib/arithmetic/core.lino" as ar)
(? (ar.peano-zero-is-natural zero))
(? (ar.peano-successor-is-natural zero))
(? (ar.plus-zero-left zero))
(? (ar.plus-zero-right zero))
(? (ar.plus-successor-left zero zero))
(? (ar.less-than-successor zero))
(? (ar.less-than-or-equal-reflexive zero))
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
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_decimal_precision_lemmas_for_built_in_arithmetic() {
    let out = evaluate_from_root(
        r#"
(import "lib/arithmetic/core.lino" as ar)
(? (ar.decimal-sum-equals 0.1 0.2 0.3))
(? (ar.decimal-difference-equals 0.3 0.1 0.2))
(? (ar.decimal-product-equals 0.1 0.2 0.02))
(? (ar.decimal-quotient-equals 1 3 0.333333333333))
(? (ar.less-than 0.1 0.2))
(? (ar.less-than-or-equal 0.3 (0.1 + 0.2)))
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
