// Proof-object substrate tests (issue #97, Phase 3).
//
// Parallel to `js/tests/proof-substrate.test.mjs`. The substrate lets
// `.lino` programs declare rules of inference and concrete derivations as
// data, then verify the derivations by structural pattern matching against
// the declared rule. The three surface forms are:
//
//   (rule <name> (premise <pat>)... (conclusion <pat>))
//   (assumption <name> (judgement <judgement>)) / (axiom <name> ...)
//   (proof-object <name> (applies <rule>) (premise-by <dependency>)... (conclusion <judgement>))
//   (check-proof <name>)
//
// `?meta` leaves inside patterns are metavariables; repeated metavariables
// must structurally match. Failures emit an `E064` diagnostic and push `0.0`
// into the result stream; successes push `1.0`. Rules and proof objects are
// surfaced on `foundation_report()` so the trust audit can inspect them.
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97

use rml::{
    check_proof_object, evaluate, evaluate_with_env, format_foundation_report, match_proof_pattern,
    parse_proof_assumption_form, parse_proof_object_form, parse_rule_form, CheckProofVerdict, Env,
    Node, RunResult,
};
use std::collections::HashMap;

fn nums(results: &[RunResult]) -> Vec<f64> {
    results
        .iter()
        .filter_map(|r| match r {
            RunResult::Num(v) => Some(*v),
            _ => None,
        })
        .collect()
}

fn leaf(s: &str) -> Node {
    Node::Leaf(s.to_string())
}

fn list(items: Vec<Node>) -> Node {
    Node::List(items)
}

#[test]
fn parses_a_rule_form_into_a_structured_descriptor() {
    let node = list(vec![
        leaf("rule"),
        leaf("modus-ponens"),
        list(vec![
            leaf("premise"),
            list(vec![leaf("?a"), leaf("implies"), leaf("?b")]),
        ]),
        list(vec![leaf("premise"), leaf("?a")]),
        list(vec![leaf("conclusion"), leaf("?b")]),
    ]);
    let rule = parse_rule_form(&node).expect("parse");
    assert_eq!(rule.name, "modus-ponens");
    assert_eq!(rule.premises.len(), 2);
    assert!(matches!(rule.conclusion, Node::Leaf(ref s) if s == "?b"));
}

#[test]
fn parses_a_proof_object_form_into_a_structured_descriptor() {
    let node = list(vec![
        leaf("proof-object"),
        leaf("mp-1"),
        list(vec![leaf("applies"), leaf("modus-ponens")]),
        list(vec![
            leaf("premise"),
            list(vec![leaf("raining"), leaf("implies"), leaf("wet")]),
        ]),
        list(vec![leaf("premise-by"), leaf("rain-implies-wet")]),
        list(vec![leaf("premise"), leaf("raining")]),
        list(vec![leaf("uses"), leaf("rain")]),
        list(vec![leaf("conclusion"), leaf("wet")]),
    ]);
    let po = parse_proof_object_form(&node).expect("parse");
    assert_eq!(po.name, "mp-1");
    assert_eq!(po.rule, "modus-ponens");
    assert_eq!(po.premises.len(), 2);
    assert_eq!(
        po.premise_refs,
        vec!["rain-implies-wet".to_string(), "rain".to_string()]
    );
    assert!(matches!(po.conclusion, Node::Leaf(ref s) if s == "wet"));
}

#[test]
fn parses_assumptions_and_axioms_as_explicit_proof_leaves() {
    let node = list(vec![
        leaf("assumption"),
        leaf("rain"),
        list(vec![leaf("judgement"), leaf("raining")]),
    ]);
    let assumption = parse_proof_assumption_form(&node).expect("parse");
    assert_eq!(assumption.name, "rain");
    assert_eq!(assumption.kind, "assumption");
    assert!(matches!(assumption.judgement, Node::Leaf(ref s) if s == "raining"));
}

#[test]
fn matcher_binds_metavariables_into_the_substitution_map() {
    let pattern = list(vec![leaf("?a"), leaf("implies"), leaf("?b")]);
    let candidate = list(vec![leaf("raining"), leaf("implies"), leaf("wet")]);
    let mut subs: HashMap<String, Node> = HashMap::new();
    assert!(match_proof_pattern(&pattern, &candidate, &mut subs));
    assert!(matches!(subs.get("?a"), Some(Node::Leaf(s)) if s == "raining"));
    assert!(matches!(subs.get("?b"), Some(Node::Leaf(s)) if s == "wet"));
}

#[test]
fn matcher_rejects_inconsistent_bindings_for_the_same_metavariable() {
    let pattern = list(vec![
        list(vec![leaf("?a")]),
        leaf("implies"),
        list(vec![leaf("?a")]),
    ]);
    let candidate = list(vec![
        list(vec![leaf("raining")]),
        leaf("implies"),
        list(vec![leaf("snowing")]),
    ]);
    let mut subs: HashMap<String, Node> = HashMap::new();
    assert!(!match_proof_pattern(&pattern, &candidate, &mut subs));
}

#[test]
fn verifies_a_valid_modus_ponens_derivation_end_to_end() {
    let src = r#"
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(assumption rain (judgement raining))
(proof-object mp-rain
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (premise-by rain)
  (conclusion wet))
(check-proof mp-rain)
"#;
    let out = evaluate(src, None, None);
    assert_eq!(
        out.diagnostics.len(),
        0,
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn fails_with_e064_when_a_premise_does_not_match_the_rule_pattern() {
    let src = r#"
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(assumption sunny-now (judgement sunny))
(proof-object mp-bad
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (premise-by sunny-now)
  (conclusion wet))
(check-proof mp-bad)
"#;
    let out = evaluate(src, None, None);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("premise 2 does not match rule modus-ponens"));
}

#[test]
fn fails_with_e064_when_premise_count_differs_from_rule_arity() {
    let src = r#"
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(proof-object mp-short
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (conclusion wet))
(check-proof mp-short)
"#;
    let out = evaluate(src, None, None);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("expected 2 premise(s) for rule modus-ponens, got 1"));
}

#[test]
fn fails_with_e064_when_proof_object_references_unknown_rule() {
    let src = r#"
(proof-object orphan
  (applies nonexistent-rule)
  (premise foo)
  (conclusion bar))
(check-proof orphan)
"#;
    let out = evaluate(src, None, None);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("references unknown rule nonexistent-rule"));
}

#[test]
fn fails_with_e064_when_check_proof_targets_unknown_proof_object() {
    let out = evaluate("(check-proof never-declared)", None, None);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("unknown proof-object never-declared"));
}

#[test]
fn enforces_metavariable_consistency_across_premises_and_conclusion() {
    // The rule's conclusion uses `?b`, which is bound by the first premise's
    // implication. Swapping the conclusion to a different leaf must fail.
    let src = r#"
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(assumption rain (judgement raining))
(proof-object mp-skew
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (premise-by rain)
  (conclusion snowing))
(check-proof mp-skew)
"#;
    let out = evaluate(src, None, None);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("conclusion does not match rule modus-ponens"));
}

#[test]
fn rejects_malformed_rule_forms_with_e064() {
    // The routing guard requires at least one `conclusion` clause to
    // distinguish proof-substrate rules from data-only `(rule ...)` forms
    // used by self-bootstrap files. Once routed in, malformed structure
    // (here: a premise clause with the wrong arity) is caught by
    // `parse_rule_form` and surfaced as E064.
    let src = r#"
(rule bad-arity
  (premise ?a ?extra)
  (conclusion ?b))
"#;
    let out = evaluate(src, None, None);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("(premise <pat>) requires exactly one pattern"));
}

#[test]
fn rejects_malformed_proof_object_forms_with_e064() {
    // Missing (applies <rule>) clause.
    let src = r#"
(proof-object missing-applies
  (premise foo)
  (conclusion bar))
"#;
    let out = evaluate(src, None, None);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("(applies <rule>) clause is required"));
}

#[test]
fn rejects_check_proof_without_a_name_argument() {
    let out = evaluate("(check-proof)", None, None);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("requires a proof-object name"));
}

#[test]
fn surfaces_rules_and_proof_objects_on_foundation_report() {
    let mut env = Env::new(None);
    let src = r#"
(rule reflexivity
  (conclusion (?a = ?a)))
(proof-object refl-a
  (applies reflexivity)
  (conclusion (apple = apple)))
"#;
    evaluate_with_env(src, None, &mut env);
    let report = env.foundation_report();
    assert_eq!(report.proof_rules.len(), 1);
    assert_eq!(report.proof_rules[0].name, "reflexivity");
    assert_eq!(report.proof_rules[0].premises.len(), 0);
    assert_eq!(report.proof_rules[0].conclusion, "(?a = ?a)");

    assert_eq!(report.proof_objects.len(), 1);
    assert_eq!(report.proof_objects[0].name, "refl-a");
    assert_eq!(report.proof_objects[0].rule, "reflexivity");
    assert!(report.proof_objects[0].premise_refs.is_empty());
    assert_eq!(report.proof_objects[0].conclusion, "(apple = apple)");

    let printed = format_foundation_report(&report);
    assert!(printed.contains("proof rules:"));
    assert!(printed.contains("reflexivity (0 premises → (?a = ?a))"));
    assert!(printed.contains("proof objects:"));
    assert!(printed.contains("refl-a : applies reflexivity (0 premises → (apple = apple))"));
}

#[test]
fn surfaces_proof_assumptions_and_dependency_refs_on_foundation_report() {
    let mut env = Env::new(None);
    let src = r#"
(rule identity
  (premise ?a)
  (conclusion ?a))
(assumption rain (judgement raining))
(proof-object id-rain
  (applies identity)
  (premise-by rain)
  (conclusion raining))
"#;
    evaluate_with_env(src, None, &mut env);
    let report = env.foundation_report();
    assert_eq!(report.proof_assumptions.len(), 1);
    assert_eq!(report.proof_assumptions[0].name, "rain");
    assert_eq!(report.proof_assumptions[0].kind, "assumption");
    assert_eq!(report.proof_assumptions[0].judgement, "raining");
    assert_eq!(
        report.proof_objects[0].premise_refs,
        vec!["rain".to_string()]
    );
    let printed = format_foundation_report(&report);
    assert!(printed.contains("proof assumptions:"));
    assert!(printed.contains("rain [assumption] : raining"));
    assert!(printed.contains("id-rain : applies identity (0 premises using rain → raining)"));
}

#[test]
fn check_proof_object_returns_substitution_witness_on_success() {
    let mut env = Env::new(None);
    let src = r#"
(rule modus-ponens
  (premise (?a implies ?b))
  (premise ?a)
  (conclusion ?b))
(assumption rain-implies-wet (judgement (raining implies wet)))
(assumption rain (judgement raining))
(proof-object mp-rain
  (applies modus-ponens)
  (premise-by rain-implies-wet)
  (premise-by rain)
  (conclusion wet))
"#;
    evaluate_with_env(src, None, &mut env);
    match check_proof_object(&env, "mp-rain") {
        CheckProofVerdict::Ok(subs) => {
            assert!(matches!(subs.get("?a"), Some(Node::Leaf(s)) if s == "raining"));
            assert!(matches!(subs.get("?b"), Some(Node::Leaf(s)) if s == "wet"));
        }
        CheckProofVerdict::Err(msg) => panic!("expected ok, got err: {msg}"),
    }
}

#[test]
fn rejects_raw_proof_object_premises_that_cite_no_dependency() {
    let src = r#"
(rule identity
  (premise ?a)
  (conclusion ?a))
(proof-object raw
  (applies identity)
  (premise raining)
  (conclusion raining))
(check-proof raw)
"#;
    let out = evaluate(src, None, None);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("premise 1 is unjustified"));
}

#[test]
fn accepts_proof_object_dependencies_produced_by_other_proof_objects() {
    let src = r#"
(rule identity
  (premise ?a)
  (conclusion ?a))
(rule reflexivity
  (conclusion (?a = ?a)))
(proof-object refl-rain
  (applies reflexivity)
  (conclusion (raining = raining)))
(proof-object id-rain
  (applies identity)
  (premise-by refl-rain)
  (conclusion (raining = raining)))
(check-proof id-rain)
"#;
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn detects_cyclic_proof_object_dependencies() {
    let src = r#"
(rule identity
  (premise ?a)
  (conclusion ?a))
(proof-object a
  (applies identity)
  (premise-by b)
  (conclusion rain))
(proof-object b
  (applies identity)
  (premise-by a)
  (conclusion rain))
(check-proof a)
"#;
    let out = evaluate(src, None, None);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert_eq!(out.diagnostics[0].code, "E064");
    assert!(out.diagnostics[0]
        .message
        .contains("cyclic proof dependency: a -> b -> a"));
}

#[test]
fn does_not_hijack_existing_rule_data_forms_from_self_bootstrap_files() {
    // The self-grammar bootstrap uses `(rule <name> (sequence ...) ...)`.
    // The proof substrate must let these forms pass through to the legacy
    // data path with no E064 diagnostic and no result emission.
    let src = r#"
(rule source-for-evaluation
  (sequence parse normalize evaluate)
  (normalizes-to document))
"#;
    let out = evaluate(src, None, None);
    let proof_diags: Vec<&rml::Diagnostic> = out
        .diagnostics
        .iter()
        .filter(|d| d.code == "E064")
        .collect();
    assert!(
        proof_diags.is_empty(),
        "did not expect E064 diagnostics: {:?}",
        proof_diags
    );
    assert!(nums(&out.results).is_empty());
}

#[test]
fn keeps_proof_rules_separate_from_evaluator_behaviour() {
    // Declaring a rule must not change baseline query semantics.
    let with_rule = evaluate(
        r#"
(rule unused
  (premise ?a)
  (conclusion ?a))
(? (1 + 2))
"#,
        None,
        None,
    );
    let without_rule = evaluate("(? (1 + 2))", None, None);
    assert!(with_rule.diagnostics.is_empty());
    assert_eq!(nums(&with_rule.results), nums(&without_rule.results));
}
