// Tests for the algebra standard library (issue #75).
// Mirrors js/tests/lib-algebra.test.mjs so the LiNo library surface stays
// identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-algebra-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_magma_monoid_and_issue_surface_group_through_import_alias() {
    let out = evaluate_from_root(
        r#"
(import "lib/algebra/core.lino" as al)
(? (al.and true true))
(? (al.and true false))
(? (al.group (carrier G) (op times) (identity e) (inverse inv)))
(? ((al.magma (carrier G) (op times)) =
     (al.closed-under (carrier G) (op times))))
(? ((al.monoid (carrier G) (op times) (identity e)) =
     (algebra.and
       (al.semigroup (carrier G) (op times))
       (al.identity-element (carrier G) (op times) (identity e)))))
(? ((al.group (carrier G) (op times) (identity e) (inverse inv)) =
     (algebra.and
       (al.monoid (carrier G) (op times) (identity e))
       (al.inverse-operation (carrier G) (op times) (identity e) (inverse inv)))))
"#,
    );

    assert_clean(&out);
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(0.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn exports_reusable_operation_law_schemas() {
    let out = evaluate_from_root(
        r#"
(import "lib/algebra/core.lino" as al)
(? ((al.closed-under G times) =
     (forall (G left)
       (forall (G right)
         ((times left right) of G)))))
(? ((al.associative G times) =
     (forall (G left)
       (forall (G middle)
         (forall (G right)
           (= (times (times left middle) right)
              (times left (times middle right))))))))
(? ((al.identity-element G times e) =
     (algebra.and
       (al.left-identity-law G times e)
       (al.right-identity-law G times e))))
(? ((al.inverse-operation G times e inv) =
     (algebra.and
       (al.left-inverse-law G times e inv)
       (al.right-inverse-law G times e inv))))
(? ((al.commutative G times) =
     (forall (G left)
       (forall (G right)
         (= (times left right) (times right left))))))
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
fn exports_ring_schemas_from_additive_group_and_multiplicative_monoid_pieces() {
    let out = evaluate_from_root(
        r#"
(import "lib/algebra/core.lino" as al)
(? ((al.distributive R plus times) =
     (algebra.and
       (al.left-distributive R plus times)
       (al.right-distributive R plus times))))
(? ((al.ring R plus zero neg times one) =
     (algebra.and
       (al.abelian-group R plus zero neg)
       (al.monoid R times one)
       (al.distributive R plus times))))
"#,
    );

    assert_clean(&out);
    assert_eq!(out.results, vec![RunResult::Num(1.0), RunResult::Num(1.0),]);
}
