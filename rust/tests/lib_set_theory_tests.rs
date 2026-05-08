// Tests for the set-theory standard library (issue #73).
// Mirrors js/tests/lib-set-theory.test.mjs so the LiNo library surface stays
// identical across both implementations.

use rml::{evaluate, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-set-theory-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

#[test]
fn exports_membership_and_core_set_constructors_through_import_alias() {
    let out = evaluate_from_root(
        r#"
(import "lib/set-theory/core.lino" as st)
(? ((st.member-of x (set-of-naturals)) = (member-of x (set-of-naturals))))
(? ((st.singleton x) = (singleton x)))
(? ((st.unordered-pair x y) = (unordered-pair x y)))
(? ((st.union family) = (union family)))
(? ((st.separation source predicate) = (separation source predicate)))
(? ((st.replacement source mapping) = (replacement source mapping)))
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
fn exports_extensionality_and_pairing_schemas_as_reusable_templates() {
    let out = evaluate_from_root(
        r#"
(import "lib/set-theory/core.lino" as st)
(? ((st.subset-of left right) =
     (forall (Set element)
       (st.implies
         (st.member-of element left)
         (st.member-of element right)))))
(? ((st.same-set left right) =
     (forall (Set element)
       (st.iff
         (st.member-of element left)
         (st.member-of element right)))))
(? ((st.axiom-extensionality left right) =
     (st.implies
       (st.same-set left right)
       (= left right))))
(? ((st.axiom-pairing left right pair-set) =
     (forall (Set element)
       (st.iff
         (st.member-of element pair-set)
         (st.pair-membership element left right)))))
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
fn exports_union_separation_replacement_and_infinity_schemas() {
    let out = evaluate_from_root(
        r#"
(import "lib/set-theory/core.lino" as st)
(? ((st.axiom-union collection union-set) =
     (forall (Set element)
       (st.iff
         (st.member-of element union-set)
         (exists (Set member-set)
           (set-theory.and
             (st.member-of element member-set)
             (st.member-of member-set collection)))))))
(? ((st.axiom-separation source predicate subset) =
     (forall (Set element)
       (st.iff
         (st.member-of element subset)
         (set-theory.and
           (st.member-of element source)
           (predicate element))))))
(? ((st.axiom-replacement source mapping image) =
     (forall (Set output)
       (st.iff
         (st.member-of output image)
         (exists (Set input)
           (set-theory.and
             (st.member-of input source)
             (mapping input output)))))))
(? ((st.successor x) =
     (st.union (st.unordered-pair x (st.singleton x)))))
(? ((st.inductive-set naturals) =
     (set-theory.and
       (st.member-of empty-set naturals)
       (forall (Set element)
         (st.implies
           (st.member-of element naturals)
           (st.member-of (st.successor element) naturals))))))
(? ((st.axiom-infinity (set-of-naturals)) =
     (st.inductive-set (set-of-naturals))))
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
