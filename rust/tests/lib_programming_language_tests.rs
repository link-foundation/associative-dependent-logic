// Tests for the programming-language theory standard library (issue #76).
// Mirrors js/tests/lib-programming-language.test.mjs so the LiNo library
// surface stays identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-programming-language-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_untyped_lambda_calculus_syntax_and_beta_step_schemas() {
    let out = evaluate_from_root(
        r#"
(import "lib/programming-language/core.lino" as pl)
(? ((pl.variable x) = (programming-language.object-variable x)))
(? ((pl.abstraction x (pl.variable x)) =
     (programming-language.object-lambda x
       (programming-language.object-variable x))))
(? ((pl.application (pl.abstraction x (pl.variable x)) y) =
     (programming-language.object-apply
       (programming-language.object-lambda x
         (programming-language.object-variable x))
       y)))
(? ((pl.beta-reduction x (pl.variable x) y) =
     (programming-language.small-step
       (programming-language.object-apply
         (programming-language.object-lambda x
           (programming-language.object-variable x))
         y)
       (programming-language.object-substitution
         (programming-language.object-variable x) x y))))
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
fn exports_stlc_typing_rules_as_reusable_schemas() {
    let out = evaluate_from_root(
        r#"
(import "lib/programming-language/core.lino" as pl)
(? ((pl.function-type A B) =
     (programming-language.simple-function-type A B)))
(? ((pl.typing-abstraction gamma x A body B) =
     (programming-language.implies
       (pl.has-type (pl.extend-context gamma x A) body B)
       (pl.has-type gamma
         (pl.abstraction x body)
         (pl.function-type A B)))))
(? ((pl.typing-application gamma fn arg A B) =
     (programming-language.implies
       (programming-language.and
         (pl.has-type gamma fn (pl.function-type A B))
         (pl.has-type gamma arg A))
       (pl.has-type gamma (pl.application fn arg) B))))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![RunResult::Num(1.0), RunResult::Num(1.0), RunResult::Num(1.0),]
    );
}

#[test]
fn exports_theorem_progress_and_preservation_through_the_issue_surface() {
    let out = evaluate_from_root(
        r#"
(import "lib/programming-language/core.lino" as pl)
(? (pl.theorem progress (pl.progress term T)))
(? (pl.theorem preservation (pl.preservation term next T)))
(? ((pl.progress term T) =
     (pl.progress-in programming-language.empty-context term T)))
(? ((pl.preservation term next T) =
     (pl.preservation-in programming-language.empty-context term next T)))
(? ((pl.type-safety term next T) =
     (programming-language.and
       (pl.progress term T)
       (pl.preservation term next T))))
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
