// Phase 5 — links-defined typed-kernel fragment (issue #97).
//
// Parallel to `js/tests/typed-kernel-links.test.mjs`. The four
// typed-kernel rules (pi-formation, lambda-introduction,
// application-elimination, beta-conversion) are expressed inside the
// proof substrate by `examples/typed-kernel-links.lino`. The tests
// below pin each rule down on its own, then run the bundled example
// end-to-end, and finally verify that the runtime registers the
// `typed-kernel-links` foundation with the expected `uses` list.

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
fn typed_kernel_links_foundation_is_pre_registered() {
    let out = evaluate("(foundation-report)", None, None);
    let foundation = out.results.iter().find_map(|r| match r {
        RunResult::Foundation(report) => report
            .foundations
            .iter()
            .find(|f| f.name == "typed-kernel-links"),
        _ => None,
    });
    let foundation = foundation.expect("typed-kernel-links foundation must be registered");
    let mut uses = foundation.uses.clone();
    uses.sort();
    assert_eq!(
        uses,
        vec![
            "application-elimination".to_string(),
            "beta-conversion".to_string(),
            "lambda-introduction".to_string(),
            "pi-formation".to_string(),
        ]
    );
}

#[test]
fn lib_self_foundations_documents_typed_kernel_links() {
    let path = repo_root().join("lib").join("self").join("foundations.lino");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
    assert!(
        source.contains("(foundation typed-kernel-links"),
        "foundations.lino must declare typed-kernel-links"
    );
    for rule in [
        "pi-formation",
        "lambda-introduction",
        "application-elimination",
        "beta-conversion",
    ] {
        assert!(
            source.contains(&format!("(uses {})", rule)),
            "typed-kernel-links must (uses {})",
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
fn pi_formation_derives_a_well_formed_pi_type() {
    let src = r#"
(rule pi-formation
  (premise (?Gamma turnstile (?A has-type Type0)))
  (premise ((?Gamma comma (?x has-type ?A)) turnstile (?B has-type Type0)))
  (conclusion (?Gamma turnstile ((Pi (?x has-type ?A) ?B) has-type Type0))))

(axiom nat-is-type
  (judgement (empty turnstile (Nat has-type Type0))))

(axiom nat-is-type-under-x
  (judgement ((empty comma (x has-type Nat)) turnstile (Nat has-type Type0))))

(proof-object pi-nat-nat
  (applies pi-formation)
  (premise-by nat-is-type)
  (premise-by nat-is-type-under-x)
  (conclusion (empty turnstile ((Pi (x has-type Nat) Nat) has-type Type0))))

(check-proof pi-nat-nat)
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
fn lambda_introduction_types_the_identity_function_on_nat() {
    let src = r#"
(rule lambda-introduction
  (premise ((?Gamma comma (?x has-type ?A)) turnstile (?body has-type ?B)))
  (conclusion (?Gamma turnstile ((lambda (?x has-type ?A) ?body) has-type (Pi (?x has-type ?A) ?B)))))

(axiom x-is-nat
  (judgement ((empty comma (x has-type Nat)) turnstile (x has-type Nat))))

(proof-object id-nat-typed
  (applies lambda-introduction)
  (premise-by x-is-nat)
  (conclusion (empty turnstile ((lambda (x has-type Nat) x) has-type (Pi (x has-type Nat) Nat)))))

(check-proof id-nat-typed)
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
fn application_elimination_substitutes_the_codomain() {
    let src = r#"
(rule application-elimination
  (premise (?Gamma turnstile (?f has-type (Pi (?x has-type ?A) ?B))))
  (premise (?Gamma turnstile (?arg has-type ?A)))
  (conclusion (?Gamma turnstile ((apply ?f ?arg) has-type (subst ?B ?x ?arg)))))

(axiom id-typed
  (judgement (empty turnstile ((lambda (x has-type Nat) x) has-type (Pi (x has-type Nat) Nat)))))

(axiom zero-is-nat
  (judgement (empty turnstile (zero has-type Nat))))

(proof-object app-id-zero
  (applies application-elimination)
  (premise-by id-typed)
  (premise-by zero-is-nat)
  (conclusion (empty turnstile ((apply (lambda (x has-type Nat) x) zero) has-type (subst Nat x zero)))))

(check-proof app-id-zero)
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
fn beta_conversion_preserves_typing_through_a_redex_reduction() {
    let src = r#"
(rule beta-conversion
  (premise (?Gamma turnstile (?redex has-type ?B)))
  (premise (?redex beta-reduces-to ?reduct))
  (conclusion (?Gamma turnstile (?reduct has-type ?B))))

(axiom app-id-zero-typed
  (judgement (empty turnstile ((apply (lambda (x has-type Nat) x) zero) has-type (subst Nat x zero)))))

(axiom id-zero-beta
  (judgement ((apply (lambda (x has-type Nat) x) zero) beta-reduces-to zero)))

(proof-object zero-after-beta
  (applies beta-conversion)
  (premise-by app-id-zero-typed)
  (premise-by id-zero-beta)
  (conclusion (empty turnstile (zero has-type (subst Nat x zero)))))

(check-proof zero-after-beta)
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
fn rejects_a_derivation_that_contradicts_the_typing_rule() {
    let src = r#"
(rule lambda-introduction
  (premise ((?Gamma comma (?x has-type ?A)) turnstile (?body has-type ?B)))
  (conclusion (?Gamma turnstile ((lambda (?x has-type ?A) ?body) has-type (Pi (?x has-type ?A) ?B)))))

(axiom x-is-nat
  (judgement ((empty comma (x has-type Nat)) turnstile (x has-type Nat))))

(proof-object id-mistyped
  (applies lambda-introduction)
  (premise-by x-is-nat)
  (conclusion (empty turnstile ((lambda (x has-type Nat) x) has-type (Pi (x has-type Bool) Nat)))))

(check-proof id-mistyped)
"#;
    let out = run(src);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert!(out.diagnostics.iter().any(|d| d.code == "E064"));
}

#[test]
fn runs_the_full_phase_5_example_end_to_end_with_no_diagnostics() {
    let path = repo_root().join("examples").join("typed-kernel-links.lino");
    let out = evaluate_file(path.to_str().unwrap(), EvaluateOptions::default());
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0, 1.0, 1.0, 1.0]);
}

// Issue #97, netkeep80's audit deliverable: "Type, Prop, Pi, lambda,
// apply have links-level rule representations; substitution, freshness,
// alpha-renaming, beta-reduction, normalization, and conversion are
// either links-level or explicitly host-trusted." This test pins the
// complete typed-kernel boundary so any future change that quietly
// removes a construct, downgrades its trust bucket, or splits a
// dependency edge fails loudly. Parallel to the JS test of the same
// name in `js/tests/typed-kernel-links.test.mjs`.
#[test]
fn reports_the_complete_typed_kernel_boundary_in_foundation_report() {
    let out = evaluate("(foundation-report)", None, None);
    let report = out
        .results
        .iter()
        .find_map(|r| match r {
            RunResult::Foundation(report) => Some(report),
            _ => None,
        })
        .expect("foundation-report must return a Foundation result");

    let lookup = |name: &str| {
        report
            .root_constructs
            .iter()
            .find(|rc| rc.name == name)
            .unwrap_or_else(|| panic!("{} should be registered as a root construct", name))
    };

    let host_trusted_kernel = [
        "Type",
        "Prop",
        "Pi",
        "lambda",
        "apply",
        "beta-reduction",
        "substitution",
        "freshness",
        "alpha-renaming",
        "normalization",
        "whnf",
        "definitional-equality",
        "conversion",
    ];
    let host_trusted_bucket = report
        .by_semantic_status
        .iter()
        .find(|(status, _)| status == "host-trusted")
        .map(|(_, constructs)| constructs.clone())
        .unwrap_or_default();
    for name in host_trusted_kernel {
        let rc = lookup(name);
        assert_eq!(
            rc.status.as_deref(),
            Some("host-primitive"),
            "{} should remain host-primitive at the kernel boundary",
            name
        );
        assert_eq!(
            rc.semantic_status.as_deref(),
            Some("host-trusted"),
            "{} should report semantic-status host-trusted",
            name
        );
        assert!(
            host_trusted_bucket.iter().any(|c| c == name),
            "{} should be bucketed under host-trusted in the report",
            name
        );
    }

    let links_checked_kernel = [
        "pi-formation",
        "lambda-introduction",
        "application-elimination",
        "beta-conversion",
    ];
    for name in links_checked_kernel {
        let rc = lookup(name);
        assert_eq!(
            rc.status.as_deref(),
            Some("links-defined"),
            "{} should be links-defined",
            name
        );
        assert_eq!(
            rc.semantic_status.as_deref(),
            Some("links-checked"),
            "{} should report semantic-status links-checked",
            name
        );
    }

    // Dependency edges that document the boundary in the audit graph.
    assert_eq!(
        lookup("beta-reduction").depends_on,
        vec![
            "substitution".to_string(),
            "freshness".to_string(),
            "alpha-renaming".to_string()
        ]
    );
    assert_eq!(
        lookup("normalization").depends_on,
        vec!["beta-reduction".to_string()]
    );
    assert_eq!(
        lookup("whnf").depends_on,
        vec!["beta-reduction".to_string()]
    );
    assert_eq!(
        lookup("conversion").depends_on,
        vec![
            "beta-reduction".to_string(),
            "normalization".to_string(),
            "structural-equality".to_string(),
        ]
    );
    assert_eq!(
        lookup("definitional-equality").depends_on,
        vec![
            "beta-reduction".to_string(),
            "structural-equality".to_string()
        ]
    );
    assert_eq!(
        lookup("apply").depends_on,
        vec!["lambda".to_string(), "beta-reduction".to_string()]
    );
    assert_eq!(
        lookup("Pi").depends_on,
        vec![
            "Type".to_string(),
            "substitution".to_string(),
            "freshness".to_string()
        ]
    );
}

#[test]
fn lib_self_foundations_declares_conversion_at_the_host_boundary() {
    let path = repo_root().join("lib").join("self").join("foundations.lino");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
    let marker = "(root-construct conversion";
    let idx = source
        .find(marker)
        .expect("conversion root-construct must be documented");
    let window = &source[idx..(idx + 400).min(source.len())];
    assert!(
        window.contains("(status host-primitive)"),
        "conversion must declare status host-primitive"
    );
    assert!(
        window.contains("(semantic-status host-trusted)"),
        "conversion must declare semantic-status host-trusted"
    );
    assert!(
        window.contains("(depends-on beta-reduction normalization structural-equality)"),
        "conversion must depend on beta-reduction, normalization, structural-equality"
    );
}

// Replays the full typed-kernel-links example and then asks the
// foundation report which constructs the typed-kernel-links foundation
// exposes. The Phase 5 deliverable wants a nontrivial typed derivation
// with all dependencies visible — this composes that derivation with the
// dependency closure so the test pins both halves.
#[test]
fn runs_a_nontrivial_typed_derivation_with_every_kernel_dependency_visible() {
    let path = repo_root().join("examples").join("typed-kernel-links.lino");
    let out = evaluate_file(path.to_str().unwrap(), EvaluateOptions::default());
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0, 1.0, 1.0, 1.0]);

    let report_out = evaluate("(foundation-report)", None, None);
    let report = report_out
        .results
        .iter()
        .find_map(|r| match r {
            RunResult::Foundation(report) => Some(report),
            _ => None,
        })
        .expect("foundation-report must return a Foundation result");
    let typed_kernel = report
        .foundations
        .iter()
        .find(|f| f.name == "typed-kernel-links")
        .expect("typed-kernel-links foundation must be reported");
    for rule in &typed_kernel.uses {
        let rc = report
            .root_constructs
            .iter()
            .find(|rc| &rc.name == rule)
            .unwrap_or_else(|| panic!("{} should be a registered root construct", rule));
        assert_eq!(
            rc.semantic_status.as_deref(),
            Some("links-checked"),
            "{} should report semantic-status links-checked",
            rule
        );
    }
}
