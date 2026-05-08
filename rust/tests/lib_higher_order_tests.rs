// Tests for the higher-order standard library (issue #70).
// Mirrors js/tests/lib-higher-order.test.mjs so the LiNo library surface stays
// identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-higher-order-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_forall_as_alias_qualified_template_for_predicate_binders() {
    let out = evaluate_from_root(
        r#"
(import "lib/higher-order/core.lino" as ho)
(Natural: (Type 0) Natural)
(zero: Natural)
(succ: (Pi (Natural n) Natural))
(? ((ho.forall ((Pi (Natural n) Boolean) P)
       ((P zero) implies (ho.forall (Natural n) (P (succ n)))))
     =
     (Pi ((Pi (Natural n) Boolean) P)
       ((P zero) implies (Pi (Natural n) (P (succ n)))))))
"#,
    );

    assert_clean(&out);
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
}

#[test]
fn exports_exists_as_alias_qualified_template_for_predicate_binders() {
    let out = evaluate_from_root(
        r#"
(import "lib/higher-order/core.lino" as ho)
(Natural: (Type 0) Natural)
(zero: Natural)
(? ((ho.exists ((Pi (Natural n) Boolean) P) (P zero))
    =
    (exists ((Pi (Natural n) Boolean) P) (P zero))))
"#,
    );

    assert_clean(&out);
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
}
