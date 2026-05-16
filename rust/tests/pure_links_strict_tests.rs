// Pure-links strict mode tests (issue #97, Phase 6).
//
// Parallel to `js/tests/pure-links-strict.test.mjs`. The form
// `(strict-foundation pure-links)` flips the strict audit on for every
// subsequent query; any operator inside the queried form whose registered
// root-construct status is `host-primitive` or `host-derived` triggers an
// E065 diagnostic. `(allow-host-primitive <name>...)` lets a program opt
// in to specific constructs while keeping everything else strict. The
// mode is surfaced on `foundation_report()` so the trust audit can prove
// the engine is running in pure-links territory.
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97

use rml::{
    evaluate, evaluate_with_env, format_foundation_report, parse_allow_host_primitive_form,
    parse_strict_foundation_form, scan_pure_links_offenders, Env, Node, RunResult,
};

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
fn strict_mode_is_off_by_default() {
    let out = evaluate("(? (1 + 2))", None, None);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.results.len(), 1);
}

#[test]
fn emits_e065_when_query_depends_on_host_primitive_arithmetic_op() {
    let src = r#"
(strict-foundation pure-links)
(? (1 + 2))
"#;
    let out = evaluate(src, None, None);
    assert_eq!(out.results.len(), 1);
    assert_eq!(
        out.diagnostics.len(),
        1,
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.diagnostics[0].code, "E065");
    assert!(out.diagnostics[0]
        .message
        .contains("pure-links strict mode"));
    assert!(out.diagnostics[0].message.contains('+'));
}

#[test]
fn lists_every_offending_construct_in_a_single_e065_diagnostic() {
    let src = r#"
(strict-foundation pure-links)
(? ((1 + 2) - (3 * 4)))
"#;
    let out = evaluate(src, None, None);
    assert_eq!(
        out.diagnostics.len(),
        1,
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.diagnostics[0].code, "E065");
    let msg = &out.diagnostics[0].message;
    assert!(msg.contains('*'));
    assert!(msg.contains('+'));
    assert!(msg.contains('-'));
}

#[test]
fn flags_host_derived_constructs_too() {
    let src = r#"
(strict-foundation pure-links)
(? (a != b))
"#;
    let out = evaluate(src, None, None);
    assert_eq!(
        out.diagnostics.len(),
        1,
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.diagnostics[0].code, "E065");
    assert!(out.diagnostics[0].message.contains("!="));
}

#[test]
fn flags_user_configurable_truth_operators_whose_active_implementation_is_host_backed() {
    let src = r#"
(strict-foundation pure-links)
(? (1 and 0))
(? (not 1))
"#;
    let out = evaluate(src, None, None);
    assert_eq!(
        out.diagnostics.len(),
        2,
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.diagnostics[0].code, "E065");
    assert_eq!(out.diagnostics[1].code, "E065");
    assert!(out.diagnostics[0]
        .message
        .contains("and -> avg -> host-primitive"));
    assert!(out.diagnostics[1]
        .message
        .contains("not -> decimal-12-arithmetic -> host-primitive"));
    assert_eq!(out.results.len(), 2);
}

#[test]
fn accepts_truth_operators_when_active_foundation_provides_links_defined_truth_tables() {
    let src = r#"
(strict-foundation pure-links)
(with-foundation boolean-links
  (? (1 and 0))
  (? (not 1)))
"#;
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![0.0, 0.0]);
}

#[test]
fn rejects_partial_truth_tables_because_unmatched_rows_retain_a_host_fallback() {
    let src = r#"
(foundation partial-boolean
  (carrier 0 1)
  (strict-carrier)
  (truth-table and (1 1 -> 1)))
(strict-foundation pure-links)
(with-foundation partial-boolean
  (? (1 and 1)))
"#;
    let out = evaluate(src, None, None);
    assert_eq!(out.results.len(), 1);
    assert_eq!(
        out.diagnostics.len(),
        1,
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.diagnostics[0].code, "E065");
    assert!(out.diagnostics[0]
        .message
        .contains("and -> avg -> host-primitive"));
    assert!(out.diagnostics[0]
        .message
        .contains("truth-table-fallback -> host-derived"));
}

#[test]
fn honours_allow_host_primitive_for_specific_constructs() {
    let src = r#"
(strict-foundation pure-links)
(allow-host-primitive + -)
(? (1 + 2))
(? (5 - 2))
"#;
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.results.len(), 2);
}

#[test]
fn still_flags_constructs_not_in_allow_list() {
    let src = r#"
(strict-foundation pure-links)
(allow-host-primitive +)
(? (1 + 2))
(? (3 * 4))
"#;
    let out = evaluate(src, None, None);
    assert_eq!(out.results.len(), 2);
    assert_eq!(
        out.diagnostics.len(),
        1,
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.diagnostics[0].code, "E065");
    assert!(out.diagnostics[0].message.contains('*'));
}

#[test]
fn does_not_affect_query_results_strict_mode_is_observation_only() {
    let plain = evaluate("(? (1 + 2))", None, None);
    let strict = evaluate(
        r#"
(strict-foundation pure-links)
(? (1 + 2))
"#,
        None,
        None,
    );
    assert_eq!(nums(&plain.results), nums(&strict.results));
}

#[test]
fn rejects_unknown_strict_foundation_profile_with_e065() {
    let out = evaluate("(strict-foundation handwritten)", None, None);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E065");
    assert!(out.diagnostics[0]
        .message
        .contains("unknown strict-foundation profile"));
}

#[test]
fn rejects_malformed_strict_foundation_forms_with_e065() {
    let out = evaluate("(strict-foundation)", None, None);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E065");
    assert!(out.diagnostics[0]
        .message
        .contains("requires a single profile name"));
}

#[test]
fn rejects_malformed_allow_host_primitive_forms_with_e065() {
    let out = evaluate("(allow-host-primitive)", None, None);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E065");
    assert!(out.diagnostics[0]
        .message
        .contains("requires at least one construct name"));
}

#[test]
fn surfaces_strict_pure_links_state_on_foundation_report() {
    let mut env = Env::new(None);
    let src = r#"
(strict-foundation pure-links)
(allow-host-primitive + -)
"#;
    evaluate_with_env(src, None, &mut env);
    let report = env.foundation_report();
    assert!(report.strict_pure_links);
    assert_eq!(
        report.allowed_host_primitives,
        vec!["+".to_string(), "-".to_string()]
    );
    let printed = format_foundation_report(&report);
    assert!(printed.contains("pure-links strict mode: on"));
    assert!(printed.contains("allowed host primitives: +, -"));
}

#[test]
fn parse_strict_foundation_form_parses_profile_name() {
    let node = list(vec![leaf("strict-foundation"), leaf("pure-links")]);
    let decl = parse_strict_foundation_form(&node).expect("parse");
    assert_eq!(decl.profile, "pure-links");
}

#[test]
fn parse_allow_host_primitive_form_accepts_multiple_construct_names() {
    let node = list(vec![
        leaf("allow-host-primitive"),
        leaf("+"),
        leaf("-"),
        leaf("*"),
    ]);
    let decl = parse_allow_host_primitive_form(&node).expect("parse");
    assert_eq!(
        decl.names,
        vec!["+".to_string(), "-".to_string(), "*".to_string()]
    );
}

#[test]
fn scan_pure_links_offenders_returns_empty_when_strict_mode_is_off() {
    let env = Env::new(None);
    let node = list(vec![leaf("1"), leaf("+"), leaf("2")]);
    assert!(scan_pure_links_offenders(&node, &env).is_empty());
}

#[test]
fn scan_pure_links_offenders_surfaces_every_host_primitive_operator() {
    let mut env = Env::new(None);
    env.strict_pure_links = true;
    let node = list(vec![
        list(vec![leaf("1"), leaf("+"), leaf("2")]),
        leaf("-"),
        list(vec![leaf("3"), leaf("*"), leaf("4")]),
    ]);
    let offenders = scan_pure_links_offenders(&node, &env);
    assert_eq!(
        offenders,
        vec![
            "* -> decimal-12-arithmetic -> host-primitive".to_string(),
            "+ -> decimal-12-arithmetic -> host-primitive".to_string(),
            "- -> decimal-12-arithmetic -> host-primitive".to_string()
        ]
    );
}

#[test]
fn lets_links_encoded_self_bootstrap_forms_keep_working_under_strict_mode() {
    let src = r#"
(strict-foundation pure-links)
(rule source-for-evaluation
  (sequence parse normalize evaluate)
  (normalizes-to document))
"#;
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
}

#[test]
fn lets_proof_substrate_stay_clean_under_strict_mode() {
    let src = r#"
(strict-foundation pure-links)
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
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn allows_queries_that_only_reference_user_constants_under_strict_mode() {
    let src = r#"
(strict-foundation pure-links)
(? 1)
(? 0)
"#;
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0, 0.0]);
}
