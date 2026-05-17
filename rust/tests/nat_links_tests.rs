// Phase 12 — links-defined Peano naturals (issue #97).
//
// Parallel to `js/tests/nat-links.test.mjs`. The seven proof-substrate
// rules (nat-zero-formation, nat-succ-formation, nat-add-zero,
// nat-add-succ, nat-induction, nat-refl, nat-cong-succ) are expressed
// inside the proof substrate by `examples/nat-links.lino`, together
// with the dedicated equality layer `nat-equality` that the two
// equality rules inhabit. The tests below pin each rule down on its
// own, then run the bundled example end-to-end, and finally verify
// that the runtime registers the `nat-links` foundation with the
// expected `uses` list.
//
// PR 178 added the explicit `nat-equality` layer plus the rules
// `nat-refl` and `nat-cong-succ`, switched the example from the bare
// literal `equals` to `nat-equals`, and kept the host's
// `=`/`numeric-equality` layer untouched for backward compatibility.

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
            "forall".to_string(),
            "implication".to_string(),
            "mul".to_string(),
            "nat-add-succ".to_string(),
            "nat-add-zero".to_string(),
            "nat-cong-succ".to_string(),
            "nat-eliminator".to_string(),
            "nat-equality".to_string(),
            "nat-induction".to_string(),
            "nat-mul-succ".to_string(),
            "nat-mul-zero".to_string(),
            "nat-rec-succ".to_string(),
            "nat-rec-zero".to_string(),
            "nat-recursion".to_string(),
            "nat-refl".to_string(),
            "nat-succ-formation".to_string(),
            "nat-zero-formation".to_string(),
            "predicate-application".to_string(),
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
        "nat-refl",
        "nat-cong-succ",
        "nat-recursion",
        "nat-eliminator",
        "nat-rec-zero",
        "nat-rec-succ",
        "mul",
        "nat-mul-zero",
        "nat-mul-succ",
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
    // PR 178 introduced `nat-equality` as a dedicated equality layer
    // that `nat-refl` and `nat-cong-succ` inhabit, listed by the
    // `nat-links` foundation alongside the original five rules.
    assert!(
        source.contains("(uses nat-equality)"),
        "nat-links must (uses nat-equality)"
    );
    let eq_idx = source
        .find("(root-construct nat-equality")
        .expect("missing root-construct entry for nat-equality");
    let eq_window = &source[eq_idx..(eq_idx + 400).min(source.len())];
    assert!(
        eq_window.contains("equality-layer"),
        "nat-equality must be declared as an equality-layer kind"
    );
    assert!(
        eq_window.contains("links-defined"),
        "nat-equality must record links-defined status"
    );
    // Phase 13 promotes the logical glue (forall, implication,
    // predicate-application) to first-class root constructs so the
    // trust audit can list them by name.
    for glue in ["forall", "implication", "predicate-application"] {
        assert!(
            source.contains(&format!("(uses {})", glue)),
            "nat-links must (uses {})",
            glue
        );
        let marker = format!("(root-construct {}", glue);
        let idx = source
            .find(&marker)
            .unwrap_or_else(|| panic!("missing root-construct entry for {}", glue));
        let window = &source[idx..(idx + 400).min(source.len())];
        assert!(
            window.contains("links-defined"),
            "{} root-construct must record links-defined status",
            glue
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
  (conclusion ((add zero ?n) nat-equals ?n)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object zero-plus-zero
  (applies nat-add-zero)
  (premise-by zero-is-nat)
  (conclusion ((add zero zero) nat-equals zero)))

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
  (premise ((add ?m ?n) nat-equals ?k))
  (conclusion ((add (succ ?m) ?n) nat-equals (succ ?k))))

(axiom zero-plus-zero
  (judgement ((add zero zero) nat-equals zero)))

(proof-object one-plus-zero
  (applies nat-add-succ)
  (premise-by zero-plus-zero)
  (conclusion ((add (succ zero) zero) nat-equals (succ zero))))

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
fn nat_refl_derives_n_nat_equals_n_for_a_well_typed_nat() {
    let src = r#"
(rule nat-refl
  (premise (?n has-type Nat))
  (conclusion (?n nat-equals ?n)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object zero-nat-equals-zero
  (applies nat-refl)
  (premise-by zero-is-nat)
  (conclusion (zero nat-equals zero)))

(check-proof zero-nat-equals-zero)
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
fn nat_cong_succ_lifts_a_nat_equality_through_succ() {
    let src = r#"
(rule nat-cong-succ
  (premise (?m nat-equals ?n))
  (conclusion ((succ ?m) nat-equals (succ ?n))))

(axiom zero-nat-equals-zero
  (judgement (zero nat-equals zero)))

(proof-object succ-zero-nat-equals-succ-zero
  (applies nat-cong-succ)
  (premise-by zero-nat-equals-zero)
  (conclusion ((succ zero) nat-equals (succ zero))))

(check-proof succ-zero-nat-equals-succ-zero)
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
fn rejects_a_nat_cong_succ_derivation_that_drops_one_of_the_succ_wrappers() {
    let src = r#"
(rule nat-cong-succ
  (premise (?m nat-equals ?n))
  (conclusion ((succ ?m) nat-equals (succ ?n))))

(axiom zero-nat-equals-zero
  (judgement (zero nat-equals zero)))

(proof-object bad-cong
  (applies nat-cong-succ)
  (premise-by zero-nat-equals-zero)
  (conclusion ((succ zero) nat-equals zero)))

(check-proof bad-cong)
"#;
    let out = run(src);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert!(out.diagnostics.iter().any(|d| d.code == "E064"));
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
  (premise ((add ?m ?n) nat-equals ?k))
  (conclusion ((add (succ ?m) ?n) nat-equals (succ ?k))))

(axiom zero-plus-zero
  (judgement ((add zero zero) nat-equals zero)))

(proof-object wrong-add
  (applies nat-add-succ)
  (premise-by zero-plus-zero)
  (conclusion ((add (succ zero) zero) nat-equals zero)))

(check-proof wrong-add)
"#;
    let out = run(src);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert!(out.diagnostics.iter().any(|d| d.code == "E064"));
}

#[test]
fn leaves_the_host_equality_layer_unchanged_when_nat_links_is_not_selected() {
    // PR 178 introduces `nat-equality` as an additional links-defined
    // layer; programs that never opt into the nat-links foundation
    // must keep the host's decimal-12 `=` semantics.
    let src = r#"
(? (= 1 1))
(? (= 1 2))
"#;
    let out = run(src);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0, 0.0]);
}

#[test]
fn nat_mul_zero_discharges_base_case_of_multiplication() {
    let src = r#"
(rule nat-mul-zero
  (premise (?n has-type Nat))
  (conclusion ((mul zero ?n) nat-equals zero)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object zero-mul-zero
  (applies nat-mul-zero)
  (premise-by zero-is-nat)
  (conclusion ((mul zero zero) nat-equals zero)))

(check-proof zero-mul-zero)
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
fn nat_mul_succ_steps_multiplication_through_the_successor() {
    let src = r#"
(rule nat-mul-succ
  (premise ((mul ?m ?n) nat-equals ?k))
  (premise ((add ?n ?k) nat-equals ?s))
  (conclusion ((mul (succ ?m) ?n) nat-equals ?s)))

(axiom zero-mul-one
  (judgement ((mul zero (succ zero)) nat-equals zero)))

(axiom one-plus-zero-fact
  (judgement ((add (succ zero) zero) nat-equals (succ zero))))

(proof-object one-mul-one
  (applies nat-mul-succ)
  (premise-by zero-mul-one)
  (premise-by one-plus-zero-fact)
  (conclusion ((mul (succ zero) (succ zero)) nat-equals (succ zero))))

(check-proof one-mul-one)
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
fn rejects_a_nat_mul_succ_derivation_with_inconsistent_helper_addition() {
    let src = r#"
(rule nat-mul-succ
  (premise ((mul ?m ?n) nat-equals ?k))
  (premise ((add ?n ?k) nat-equals ?s))
  (conclusion ((mul (succ ?m) ?n) nat-equals ?s)))

(axiom zero-mul-zero
  (judgement ((mul zero zero) nat-equals zero)))

(axiom zero-plus-zero-fact
  (judgement ((add zero zero) nat-equals zero)))

(proof-object wrong-one-mul-zero
  (applies nat-mul-succ)
  (premise-by zero-mul-zero)
  (premise-by zero-plus-zero-fact)
  (conclusion ((mul (succ zero) zero) nat-equals (succ zero))))

(check-proof wrong-one-mul-zero)
"#;
    let out = run(src);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert!(out.diagnostics.iter().any(|d| d.code == "E064"));
}

#[test]
fn nat_rec_zero_discharges_the_recursor_at_the_base_case() {
    let src = r#"
(rule nat-rec-zero
  (premise (?base has-type Nat))
  (conclusion (((rec ?f ?base ?step) at zero) nat-equals ?base)))

(axiom zero-is-nat
  (judgement (zero has-type Nat)))

(proof-object rec-id-at-zero
  (applies nat-rec-zero)
  (premise-by zero-is-nat)
  (conclusion (((rec id zero step) at zero) nat-equals zero)))

(check-proof rec-id-at-zero)
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
fn nat_rec_succ_steps_the_recursor_through_the_successor() {
    let src = r#"
(rule nat-rec-succ
  (premise (((rec ?f ?base ?step) at ?n) nat-equals ?prev))
  (premise (((?step ?n) at ?prev) nat-equals ?next))
  (conclusion (((rec ?f ?base ?step) at (succ ?n)) nat-equals ?next)))

(axiom rec-id-at-zero
  (judgement (((rec id zero step) at zero) nat-equals zero)))

(axiom step-zero-applied
  (judgement (((step zero) at zero) nat-equals (succ zero))))

(proof-object rec-id-at-one
  (applies nat-rec-succ)
  (premise-by rec-id-at-zero)
  (premise-by step-zero-applied)
  (conclusion (((rec id zero step) at (succ zero)) nat-equals (succ zero))))

(check-proof rec-id-at-one)
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
fn rejects_a_nat_rec_succ_derivation_that_drops_the_succ_wrapper_on_the_scrutinee() {
    let src = r#"
(rule nat-rec-succ
  (premise (((rec ?f ?base ?step) at ?n) nat-equals ?prev))
  (premise (((?step ?n) at ?prev) nat-equals ?next))
  (conclusion (((rec ?f ?base ?step) at (succ ?n)) nat-equals ?next)))

(axiom rec-id-at-zero
  (judgement (((rec id zero step) at zero) nat-equals zero)))

(axiom step-zero-applied
  (judgement (((step zero) at zero) nat-equals (succ zero))))

(proof-object bad-rec
  (applies nat-rec-succ)
  (premise-by rec-id-at-zero)
  (premise-by step-zero-applied)
  (conclusion (((rec id zero step) at zero) nat-equals (succ zero))))

(check-proof bad-rec)
"#;
    let out = run(src);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert!(out.diagnostics.iter().any(|d| d.code == "E064"));
}

#[test]
fn proof_report_returns_per_proof_dependency_and_trust_summary() {
    // Phase 13 (issue #97): `(proof-report ...)` walks the proof-object
    // tree and reports the dependencies, rules, and root constructs the
    // proof touches — together with their semantic and trust statuses,
    // so the trust audit can be done per-proof instead of only globally
    // through `(foundation-report)`. We register `Nat`, `zero`, `succ`
    // as links-defined root constructs first so the report can cite them.
    let src = r#"
(root-construct Nat
  (kind inductive-type)
  (status links-defined)
  (semantic-status links-checked))

(root-construct zero
  (kind constructor)
  (status links-defined)
  (semantic-status links-checked)
  (depends-on Nat))

(root-construct succ
  (kind constructor)
  (status links-defined)
  (semantic-status links-checked)
  (depends-on Nat))

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

(proof-report one-is-nat)
"#;
    let out = run(src);
    assert!(out.diagnostics.is_empty(), "diagnostics: {:?}", out.diagnostics);
    assert_eq!(out.results.len(), 1);
    let report = match &out.results[0] {
        RunResult::Proof(r) => r.clone(),
        other => panic!("expected RunResult::Proof, got {:?}", other),
    };
    assert_eq!(report.name, "one-is-nat");
    assert_eq!(report.rule.as_deref(), Some("nat-succ-formation"));
    assert!(report.verdict.ok);
    assert_eq!(report.premise_refs, vec!["zero-is-nat".to_string()]);
    let mut rules = report.rules.clone();
    rules.sort();
    assert_eq!(
        rules,
        vec![
            "nat-succ-formation".to_string(),
            "nat-zero-formation".to_string(),
        ]
    );
    let dep_names: Vec<String> = report.dependencies.iter().map(|d| d.name.clone()).collect();
    assert!(
        dep_names.iter().any(|n| n == "zero-is-nat"),
        "dependencies should include zero-is-nat, got {:?}",
        dep_names
    );
    assert!(report.root_constructs_used.contains(&"Nat".to_string()));
    assert!(report.root_constructs_used.contains(&"succ".to_string()));
    assert!(report.root_constructs_used.contains(&"zero".to_string()));
    let semantic = report
        .by_semantic_status
        .iter()
        .find(|(s, _)| s == "links-checked")
        .map(|(_, names)| {
            let mut v = names.clone();
            v.sort();
            v
        })
        .expect("links-checked bucket must exist");
    assert_eq!(
        semantic,
        vec!["Nat".to_string(), "succ".to_string(), "zero".to_string()]
    );
    let trust = report
        .by_trust_status
        .iter()
        .find(|(s, _)| s == "links-defined")
        .map(|(_, names)| {
            let mut v = names.clone();
            v.sort();
            v
        })
        .expect("links-defined bucket must exist");
    assert_eq!(
        trust,
        vec!["Nat".to_string(), "succ".to_string(), "zero".to_string()]
    );
}

#[test]
fn proof_report_of_unknown_proof_returns_failing_verdict() {
    let out = run("(proof-report no-such-proof)");
    assert_eq!(out.results.len(), 1);
    let report = match &out.results[0] {
        RunResult::Proof(r) => r.clone(),
        other => panic!("expected RunResult::Proof, got {:?}", other),
    };
    assert_eq!(report.name, "no-such-proof");
    assert!(!report.verdict.ok);
    let err = report
        .verdict
        .error
        .as_deref()
        .expect("unknown proof should record an error");
    assert!(
        err.contains("unknown proof-object"),
        "expected error to mention unknown proof-object, got {:?}",
        err
    );
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
    assert_eq!(
        nums(&out.results),
        vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0,
        ]
    );
}
