// Phase 12 — links-defined Peano naturals (issue #97).
//
// Parallel to `js/tests/nat-links.test.mjs`. The five proof-substrate
// rules (nat-zero-formation, nat-succ-formation, nat-add-zero,
// nat-add-succ, nat-induction) are expressed inside the proof
// substrate by `examples/nat-links.lino`. The tests below pin each
// rule down on its own, then run the bundled example end-to-end, and
// finally verify that the runtime registers the `nat-links`
// foundation with the expected `uses` list.

use rml::{evaluate, evaluate_file, EvaluateOptions, EvaluateResult, RunResult};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn nums(results: &[RunResult]) -> Vec<f64> {
    results
        .iter()
        .filter_map(|r| match r {
            RunResult::Num(v) => Some(*v),
            _ => None,
        })
        .collect()
}

fn run(src: &str) -> EvaluateResult {
    evaluate(src, None, None)
}

#[test]
fn nat_links_foundation_is_pre_registered() {
    let out = evaluate("(foundation-report)", None, None);
    let foundation = out.results.iter().find_map(|r| match r {
        RunResult::Foundation(report) => report
            .foundations
            .iter()
            .find(|f| f.name == "nat-links"),
        _ => None,
    });
    let foundation = foundation.expect("nat-links foundation must be registered");
    let mut uses = foundation.uses.clone();
    uses.sort();
    assert_eq!(
        uses,
        vec![
            "nat-add-succ".to_string(),
            "nat-add-zero".to_string(),
            "nat-induction".to_string(),
            "nat-succ-formation".to_string(),
            "nat-zero-formation".to_string(),
        ]
    );
    assert_eq!(foundation.extends.as_deref(), Some("default-rml"));
}

#[test]
fn lib_self_foundations_documents_nat_links() {
    let path = repo_root().join("lib").join("self").join("foundations.lino");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
    assert!(
        source.contains("(foundation nat-links"),
        "foundations.lino must declare nat-links"
    );
    for rule in [
        "nat-zero-formation",
        "nat-succ-formation",
        "nat-add-zero",
        "nat-add-succ",
        "nat-induction",
    ] {
        assert!(
            source.contains(&format!("(uses {})", rule)),
            "nat-links must (uses {})",
            rule
        );
        let marker = format!("(root-construct {}", rule);
        let idx = source
            .find(&marker)
            .unwrap_or_else(|| panic!("missing root-construct entry for {}", rule));
        let window = &source[idx..(idx + 400).min(source.len())];
        assert!(
            window.contains("links-defined"),
            "{} root-construct must record links-defined status",
            rule
        );
    }
}

#[test]
fn nat_zero_formation_derives_zero_has_type_nat() {
    let src = r#"
(rule nat-zero-formation
  (conclusion (zero has-type Nat)))

(proof-object zero-is-nat
  (applies nat-zero-formation)
  (conclusion (zero has-type Nat)))

(check-proof zero-is-nat)
"#;
    let out = run(src);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn nat_succ_formation_lifts_a_nat_to_its_successor() {
    let src = r#"
(rule nat-zero-formation
  (conclusion (zero has-type Nat)))

(rule nat-succ-formation
  (premise (?n has-type Nat))
  (conclusion ((succ ?n) has-type Nat)))

(proof-object zero-is-nat
  (applies nat-zero-formation)
  (conclusion (zero has-type Nat)))

(proof-object one-is-nat
  (applies nat-succ-formation)
  (premise-by zero-is-nat)
  (conclusion ((succ zero) has-type Nat)))

(check-proof one-is-nat)
"#;
    let out = run(src);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn nat_add_zero_discharges_the_base_case_of_addition() {
    let src = r#"
(rule nat-add-zero
  (premise (?n has-type Nat))
  (conclusion ((add zero ?n) equals ?n)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object zero-plus-zero
  (applies nat-add-zero)
  (premise-by zero-is-nat)
  (conclusion ((add zero zero) equals zero)))

(check-proof zero-plus-zero)
"#;
    let out = run(src);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn nat_add_succ_steps_addition_through_the_successor() {
    let src = r#"
(rule nat-add-succ
  (premise ((add ?m ?n) equals ?k))
  (conclusion ((add (succ ?m) ?n) equals (succ ?k))))

(axiom zero-plus-zero
  (judgement ((add zero zero) equals zero)))

(proof-object one-plus-zero
  (applies nat-add-succ)
  (premise-by zero-plus-zero)
  (conclusion ((add (succ zero) zero) equals (succ zero))))

(check-proof one-plus-zero)
"#;
    let out = run(src);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn nat_induction_folds_a_base_case_and_a_step_into_a_universal_claim() {
    let src = r#"
(rule nat-induction
  (premise (?P at zero))
  (premise (forall ?n (implies (?P at ?n) (?P at (succ ?n)))))
  (conclusion (forall ?n (?P at ?n))))

(axiom is-nat-at-zero
  (judgement (is-nat at zero)))

(axiom is-nat-step
  (judgement (forall n (implies (is-nat at n) (is-nat at (succ n))))))

(proof-object every-nat-is-nat
  (applies nat-induction)
  (premise-by is-nat-at-zero)
  (premise-by is-nat-step)
  (conclusion (forall n (is-nat at n))))

(check-proof every-nat-is-nat)
"#;
    let out = run(src);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn rejects_a_derivation_that_contradicts_nat_succ_formation() {
    let src = r#"
(rule nat-succ-formation
  (premise (?n has-type Nat))
  (conclusion ((succ ?n) has-type Nat)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object mistyped-succ
  (applies nat-succ-formation)
  (premise-by zero-is-nat)
  (conclusion ((succ zero) has-type Bool)))

(check-proof mistyped-succ)
"#;
    let out = run(src);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert!(out.diagnostics.iter().any(|d| d.code == "E064"));
}

#[test]
fn rejects_an_add_succ_derivation_that_contradicts_its_arithmetic_premise() {
    let src = r#"
(rule nat-add-succ
  (premise ((add ?m ?n) equals ?k))
  (conclusion ((add (succ ?m) ?n) equals (succ ?k))))

(axiom zero-plus-zero
  (judgement ((add zero zero) equals zero)))

(proof-object wrong-add
  (applies nat-add-succ)
  (premise-by zero-plus-zero)
  (conclusion ((add (succ zero) zero) equals zero)))

(check-proof wrong-add)
"#;
    let out = run(src);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert!(out.diagnostics.iter().any(|d| d.code == "E064"));
}

#[test]
fn runs_the_full_phase_12_example_end_to_end_with_no_diagnostics() {
    let path = repo_root().join("examples").join("nat-links.lino");
    let out = evaluate_file(path.to_str().unwrap(), EvaluateOptions::default());
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}
