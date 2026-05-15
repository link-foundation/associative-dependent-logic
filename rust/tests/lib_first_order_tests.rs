// Tests for the first-order standard library (issue #69).
// Mirrors js/tests/lib-first-order.test.mjs so the LiNo library surface stays
// identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-first-order-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_forall_as_alias_qualified_template_for_pi() {
    let out = evaluate_from_root(
        r#"
(import "lib/first-order/core.lino" as fo)
(Term: (Type 0) Term)
(? ((fo.forall (Term x) (predicate x)) = (Pi (Term x) (predicate x))))
(? (fo.forall (Term x) Term))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![RunResult::Num(1.0), RunResult::Num(1.0)]
    );
}

#[test]
fn exports_exists_as_alias_qualified_first_order_link_shape() {
    let out = evaluate_from_root(
        r#"
(import "lib/first-order/core.lino" as fo)
(Term: (Type 0) Term)
(? ((fo.exists (Term x) (predicate x)) = (exists (Term x) (predicate x))))
"#,
    );

    assert_clean(&out);
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
}

#[test]
fn expands_nested_quantified_formulas_before_evaluation() {
    let out = evaluate_from_root(
        r#"
(import "lib/first-order/core.lino" as fo)
(Term: (Type 0) Term)
(? ((fo.forall (Term x)
       (fo.exists (Term y) ((pair x y) = (pair x y))))
     =
     (Pi (Term x)
       (exists (Term y) ((pair x y) = (pair x y))))))
"#,
    );

    assert_clean(&out);
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
}
