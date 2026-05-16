// Foundation / root-construct registry tests (issue #97).
//
// Parallel to `js/tests/foundations.test.mjs`. The foundation surface is
// the user-facing mechanism for replacing kernel-level interpretations
// (`and`, `or`, ...) without touching the evaluator. These tests cover
// the same three layers as the JS suite:
//   1. The data registry (`(root-construct ...)`, `(foundation ...)`)
//      round-trips through `evaluate_with_env()` without losing fields.
//   2. `(with-foundation <name> ...)` swaps operator semantics inside
//      its body and restores them on exit, leaving outer scopes intact.
//   3. `(foundation-report)` returns a structured snapshot whose printed
//      form matches the canonical layout (kept byte-identical with JS).
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97

use rml::{
    evaluate, evaluate_with_env, evaluate_with_options, format_foundation_report, Aggregator, Env,
    EvaluateOptions, Op, RunResult,
};

fn agg_of(op: Option<&Op>) -> Option<Aggregator> {
    match op {
        Some(Op::Agg(agg)) => Some(*agg),
        _ => None,
    }
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

fn run_clean(src: &str) -> Vec<f64> {
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    nums(&out.results)
}

#[test]
fn preregisters_default_rml_so_legacy_programs_need_no_migration() {
    let env = Env::new(None);
    assert_eq!(env.active_foundation, "default-rml");
    let default = env
        .get_foundation("default-rml")
        .expect("default-rml descriptor is missing");
    let description = default
        .description
        .as_ref()
        .expect("default-rml needs a description");
    assert!(!description.is_empty());
}

#[test]
fn baseline_semantics_unchanged_when_no_foundation_is_declared() {
    let results = run_clean(
        r#"
(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)
(? ((a = true) and (b = true)))
(? ((a = true) or (b = true)))
"#,
    );
    assert_eq!(results, vec![0.5, 0.6]);
}

#[test]
fn switches_and_or_semantics_inside_with_foundation() {
    let results = run_clean(
        r#"
(foundation classical-min (defines and min) (defines or max))
(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)
(? ((a = true) and (b = true)))
(with-foundation classical-min
  (? ((a = true) and (b = true)))
  (? ((a = true) or (b = true))))
(? ((a = true) and (b = true)))
"#,
    );
    assert_eq!(results, vec![0.5, 0.4, 0.6, 0.5]);
}

#[test]
fn nests_with_foundation_scopes_correctly() {
    let results = run_clean(
        r#"
(foundation use-min (defines and min))
(foundation use-prod (defines and product))
(a: a is a)
(b: b is b)
((a = true) has probability 0.5)
((b = true) has probability 0.4)
(with-foundation use-min
  (? ((a = true) and (b = true)))
  (with-foundation use-prod
    (? ((a = true) and (b = true))))
  (? ((a = true) and (b = true))))
"#,
    );
    assert!((results[0] - 0.4).abs() < 1e-9, "first = {}", results[0]);
    assert!((results[1] - 0.2).abs() < 1e-9, "second = {}", results[1]);
    assert!((results[2] - 0.4).abs() < 1e-9, "third = {}", results[2]);
}

#[test]
fn reports_unknown_foundation_as_e062_without_aborting() {
    let out = evaluate(
        r#"
(a: a is a)
((a = true) has probability 0.5)
(with-foundation does-not-exist
  (? ((a = true) and (a = true))))
(? ((a = true) and (a = true)))
"#,
        None,
        None,
    );
    assert_eq!(nums(&out.results), vec![0.5]);
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E062");
}

#[test]
fn records_root_constructs_and_foundations_via_the_data_registry() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        r#"
(root-construct my-and
  (kind truth-operator)
  (status links-defined)
  (depends-on truth-range))
(foundation my-foundation
  (description my-toy-foundation)
  (defines and min)
  (numeric-domain unit-interval))
"#,
        None,
        &mut env,
    );
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    let my = env
        .get_root_construct("my-and")
        .expect("my-and not registered");
    assert_eq!(my.kind.as_deref(), Some("truth-operator"));
    assert_eq!(my.status.as_deref(), Some("links-defined"));
    let foundation = env
        .get_foundation("my-foundation")
        .expect("my-foundation not registered");
    assert_eq!(foundation.description.as_deref(), Some("my-toy-foundation"));
    assert_eq!(foundation.numeric_domain.as_deref(), Some("unit-interval"));
    assert_eq!(
        foundation.defines,
        vec![("and".to_string(), "min".to_string())]
    );
}

#[test]
fn builds_a_structured_foundation_report_snapshot() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        r#"
(foundation tiny
  (description toy-foundation)
  (defines and min))
"#,
        None,
        &mut env,
    );
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    env.enter_foundation("tiny").expect("enter tiny");
    let report = env.foundation_report();
    assert_eq!(report.active_foundation, "tiny");
    assert_eq!(report.description.as_deref(), Some("toy-foundation"));
    assert!(
        !report.root_constructs.is_empty(),
        "root constructs should be seeded by default"
    );
    let text = format_foundation_report(&report);
    assert!(
        text.contains("active foundation: tiny"),
        "report text missing active line: {}",
        text
    );
    assert!(
        text.contains("description: toy-foundation"),
        "report text missing description line: {}",
        text
    );
    env.exit_foundation();
    assert_eq!(env.active_foundation, "default-rml");
}

#[test]
fn enter_foundation_snapshots_ops_so_exit_restores_them() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        r#"
(foundation only-min (defines and min))
"#,
        None,
        &mut env,
    );
    assert!(out.diagnostics.is_empty());
    let before_and = agg_of(env.get_op("and"));
    assert_eq!(before_and, Some(Aggregator::Avg));
    env.enter_foundation("only-min").expect("enter only-min");
    let inside_and = agg_of(env.get_op("and"));
    assert_eq!(inside_and, Some(Aggregator::Min));
    assert_ne!(
        before_and, inside_and,
        "`and` should be re-bound inside the foundation"
    );
    env.exit_foundation();
    let after_and = agg_of(env.get_op("and"));
    assert_eq!(before_and, after_and, "original `and` op was not restored");
    // End-to-end: avg outside, min inside, avg restored after exit.
    let restored = run_clean(
        r#"
(foundation only-min (defines and min))
(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)
(? ((a = true) and (b = true)))
(with-foundation only-min
  (? ((a = true) and (b = true))))
(? ((a = true) and (b = true)))
"#,
    );
    assert_eq!(restored, vec![0.5, 0.4, 0.5]);
}

// ---------------------------------------------------------------------------
// Equality-layer provenance (issue #97). Parallel to the JS suite — every
// JS provenance test has a Rust twin so drift between the engines fails
// both suites simultaneously.
// ---------------------------------------------------------------------------

fn prov(out_provenance: &[Option<String>]) -> Vec<Option<String>> {
    out_provenance.to_vec()
}

#[test]
fn provenance_is_empty_when_no_equality_query_is_present() {
    let out = evaluate(
        r#"
(a: a is a)
((a = true) has probability 0.5)
(? ((a = true) and (a = true)))
"#,
        None,
        None,
    );
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    assert!(
        out.provenance.is_empty(),
        "provenance should be empty when no top-level equality query fires: {:?}",
        out.provenance
    );
}

#[test]
fn provenance_reports_structural_equality_for_self_equality() {
    let out = evaluate("(a: a is a)\n(? (a = a))", None, None);
    assert!(out.diagnostics.is_empty());
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
    assert_eq!(
        prov(&out.provenance),
        vec![Some("structural-equality".to_string())]
    );
}

#[test]
fn provenance_reports_assigned_equality_when_rule_exists() {
    let out = evaluate(
        "((a = a) has probability 0.7)\n(? (a = a))",
        None,
        None,
    );
    assert!(out.diagnostics.is_empty());
    assert_eq!(out.results, vec![RunResult::Num(0.7)]);
    assert_eq!(
        prov(&out.provenance),
        vec![Some("assigned-equality".to_string())]
    );
}

#[test]
fn provenance_reports_numeric_equality_for_constants() {
    let out = evaluate("(? ((0.1 + 0.2) = 0.3))", None, None);
    assert!(out.diagnostics.is_empty());
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
    assert_eq!(
        prov(&out.provenance),
        vec![Some("numeric-equality".to_string())]
    );
}

#[test]
fn provenance_reports_definitional_equality_for_beta_reducible_terms() {
    // (apply (lambda (Natural x) x) y) ≡ y via one beta step.
    let out = evaluate("(? ((apply (lambda (Natural x) x) y) = y))", None, None);
    assert!(out.diagnostics.is_empty());
    assert_eq!(
        prov(&out.provenance),
        vec![Some("definitional-equality".to_string())]
    );
}

#[test]
fn provenance_reports_assigned_inequality() {
    let out = evaluate(
        "((a = a) has probability 0.7)\n(? (a != a))",
        None,
        None,
    );
    assert!(out.diagnostics.is_empty());
    assert_eq!(
        prov(&out.provenance),
        vec![Some("assigned-inequality".to_string())]
    );
}

#[test]
fn provenance_aligns_with_results_when_equality_and_non_equality_queries_mix() {
    let out = evaluate(
        r#"
(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)
(? ((a = true) and (b = true)))
(? (a = a))
(? (1 = 2))
"#,
        None,
        None,
    );
    assert!(out.diagnostics.is_empty());
    assert_eq!(
        out.results,
        vec![
            RunResult::Num(0.5),
            RunResult::Num(1.0),
            RunResult::Num(0.0),
        ]
    );
    assert_eq!(
        prov(&out.provenance),
        vec![
            None,
            Some("structural-equality".to_string()),
            Some("numeric-equality".to_string()),
        ]
    );
}

#[test]
fn provenance_propagates_inside_with_foundation_bodies() {
    let out = evaluate(
        r#"
(foundation classical-min (defines and min) (defines or max))
(a: a is a)
(with-foundation classical-min
  (? (a = a)))
(? (a = a))
"#,
        None,
        None,
    );
    assert!(out.diagnostics.is_empty());
    assert_eq!(
        prov(&out.provenance),
        vec![
            Some("structural-equality".to_string()),
            Some("structural-equality".to_string())
        ]
    );
}

#[test]
fn provenance_emits_equality_layer_trace_event_per_classified_query() {
    let options = EvaluateOptions {
        trace: true,
        ..EvaluateOptions::default()
    };
    let out = evaluate_with_options("(a: a is a)\n(? (a = a))", None, options);
    assert!(out.diagnostics.is_empty());
    let equality_events: Vec<_> = out
        .trace
        .iter()
        .filter(|e| e.kind == "equality-layer")
        .collect();
    assert_eq!(equality_events.len(), 1);
    assert_eq!(equality_events[0].detail, "structural-equality");
}

// ---------------------------------------------------------------------------
// Carrier enforcement (issue #97, Section 2 of netkeep80's punch-list).
// Mirrors `js/tests/foundations.test.mjs > 'foundation carrier enforcement'`.
//
// A foundation may now declare `(carrier <v1> <v2> ...)` to list its legal
// values and `(strict-carrier)` to opt into runtime enforcement. The check
// is active only inside a `(with-foundation ...)` whose descriptor carries
// both clauses; legacy programs and foundations that omit either clause stay
// backward-compatible. Violations emit `E063` diagnostics; they never
// silently coerce values.
// ---------------------------------------------------------------------------

#[test]
fn carrier_parses_carrier_and_strict_carrier_onto_descriptor() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        r#"
(foundation two-valued
  (carrier 0 1)
  (strict-carrier)
  (defines and min)
  (defines or max))
"#,
        None,
        &mut env,
    );
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    let descriptor = env
        .get_foundation("two-valued")
        .expect("foundation should be registered");
    assert_eq!(descriptor.carrier, vec!["0".to_string(), "1".to_string()]);
    assert!(descriptor.strict_carrier);
}

#[test]
fn carrier_stays_informational_without_strict_carrier() {
    let out = evaluate(
        r#"
(foundation lax-two-valued (carrier 0 1) (defines and min) (defines or max))
(a: a is a)
((a = true) has probability 0.4)
(with-foundation lax-two-valued
  (? ((a = true) and (a = true))))
"#,
        None,
        None,
    );
    // No (strict-carrier) -> backward compatible, no E063.
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![0.4]);
}

#[test]
fn carrier_flags_out_of_carrier_query_result_with_e063() {
    let ok = evaluate(
        r#"
(foundation two-valued (carrier 0 1) (strict-carrier)
  (defines and min) (defines or max))
(a: a is a)
(b: b is b)
((a = true) has probability 1)
((b = true) has probability 1)
(with-foundation two-valued
  (? ((a = true) and (b = true)))
  (? ((a = true) or (b = false))))
"#,
        None,
        None,
    );
    // min(1,1)=1 and max(1,0)=1 -> both legal -> no diagnostics.
    assert!(
        ok.diagnostics.is_empty(),
        "unexpected diagnostics on in-carrier values: {:?}",
        ok.diagnostics
    );
    assert_eq!(nums(&ok.results), vec![1.0, 1.0]);

    let bad = evaluate(
        r#"
(foundation two-valued (carrier 0 1) (strict-carrier))
(a: a is a)
((a = true) has probability 0.5)
(with-foundation two-valued
  (? (a = true)))
"#,
        None,
        None,
    );
    // The probability assignment runs OUTSIDE the with-foundation body so it
    // is allowed; the query inside returns 0.5 -> E063.
    let codes: Vec<&str> = bad.diagnostics.iter().map(|d| d.code.as_str()).collect();
    assert!(
        codes.contains(&"E063"),
        "expected an E063 diagnostic, got {:?}",
        bad.diagnostics
    );
    assert_eq!(nums(&bad.results), vec![0.5]);
}

#[test]
fn carrier_flags_out_of_carrier_probability_assignment_with_e063() {
    let out = evaluate(
        r#"
(foundation two-valued (carrier 0 1) (strict-carrier))
(a: a is a)
(with-foundation two-valued
  ((a = true) has probability 0.5)
  (? (a = true)))
"#,
        None,
        None,
    );
    // The probability assignment inside the strict foundation violates the
    // carrier; the diagnostic is E063 and the assignment is rejected so the
    // query falls back to the default symbol probability.
    let codes: Vec<&str> = out.diagnostics.iter().map(|d| d.code.as_str()).collect();
    assert!(
        codes.contains(&"E063"),
        "expected an E063 diagnostic, got {:?}",
        out.diagnostics
    );
}

#[test]
fn carrier_is_restored_on_exit_foundation_for_nested_scopes() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        r#"
(foundation outer (carrier 0 1) (strict-carrier))
(foundation inner (carrier 0 0.5 1) (strict-carrier))
"#,
        None,
        &mut env,
    );
    assert!(out.diagnostics.is_empty());
    env.enter_foundation("outer").expect("enter outer");
    assert!(env.strict_carrier);
    let mut outer_carrier = env.carrier.clone().expect("carrier should be set");
    outer_carrier.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(outer_carrier, vec![0.0, 1.0]);

    env.enter_foundation("inner").expect("enter inner");
    let mut inner_carrier = env.carrier.clone().expect("carrier should be set");
    inner_carrier.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(inner_carrier, vec![0.0, 0.5, 1.0]);

    env.exit_foundation();
    let mut back_outer = env.carrier.clone().expect("carrier should be set");
    back_outer.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(back_outer, vec![0.0, 1.0]);

    env.exit_foundation();
    assert!(!env.strict_carrier);
    assert!(env.carrier.is_none());
}

#[test]
fn carrier_is_exposed_on_foundation_report() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        r#"
(foundation two-valued (carrier 0 1) (strict-carrier))
"#,
        None,
        &mut env,
    );
    assert!(out.diagnostics.is_empty());
    let report = env.foundation_report();
    let tv = report
        .foundations
        .iter()
        .find(|f| f.name == "two-valued")
        .expect("two-valued should be reported");
    assert_eq!(tv.carrier, vec!["0".to_string(), "1".to_string()]);
    assert!(tv.strict_carrier);
}

#[test]
fn carrier_is_not_enforced_at_the_top_level() {
    let out = evaluate(
        r#"
(foundation two-valued (carrier 0 1) (strict-carrier))
(a: a is a)
((a = true) has probability 0.5)
(? (a = true))
"#,
        None,
        None,
    );
    // Carrier strictness lives inside the foundation; declaring the
    // foundation alone must not break ordinary programs.
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics at top level: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![0.5]);
}

// ---------------------------------------------------------------------------
// Links-defined finite truth tables (issue #97, punch-list #3). Parallel to
// the JS suite: a foundation can declare a finite truth table for any
// operator using `(truth-table <op> (in1 in2 ... -> out) ...)`. When the
// foundation is active, calls to that operator match the row first and fall
// back to the previously installed op for unmatched inputs.
// ---------------------------------------------------------------------------

#[test]
fn truth_table_parses_clauses_onto_the_foundation_descriptor() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        r#"
(foundation boolean-classical
  (truth-table and (1 1 -> 1) (1 0 -> 0) (0 1 -> 0) (0 0 -> 0))
  (truth-table or  (0 0 -> 0) (0 1 -> 1) (1 0 -> 1) (1 1 -> 1)))
"#,
        None,
        &mut env,
    );
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    let f = env
        .get_foundation("boolean-classical")
        .expect("boolean-classical should be registered");
    assert_eq!(f.truth_tables.len(), 2);
    let and_rows = f
        .truth_tables
        .iter()
        .find(|(op, _)| op == "and")
        .map(|(_, rows)| rows)
        .expect("and truth table missing");
    assert_eq!(and_rows.len(), 4);
    assert_eq!(and_rows[0].inputs, vec!["1".to_string(), "1".to_string()]);
    assert_eq!(and_rows[0].output, "1".to_string());
    assert_eq!(and_rows[3].inputs, vec!["0".to_string(), "0".to_string()]);
    assert_eq!(and_rows[3].output, "0".to_string());
}

#[test]
fn truth_table_rejects_malformed_rows_with_e061() {
    let out = evaluate("(foundation bad (truth-table and (1 1 1)))", None, None);
    let code = out.diagnostics.iter().find(|d| d.code == "E061");
    assert!(code.is_some(), "malformed row should raise E061");
}

#[test]
fn truth_table_overrides_operator_semantics_inside_with_foundation() {
    let results = run_clean(
        r#"
(foundation boolean-and-table
  (truth-table and (1 1 -> 1) (1 0 -> 0) (0 1 -> 0) (0 0 -> 0)))
(a: a is a)
(b: b is b)
((a = true) has probability 1)
((b = true) has probability 0)
(with-foundation boolean-and-table
  (? ((a = true) and (b = true)))
  (? ((b = true) and (b = true))))
"#,
    );
    // Default `and` would avg 1 and 0 → 0.5, but the table pins both
    // (1, 0) and (0, 0) rows to 0.
    assert_eq!(results, vec![0.0, 0.0]);
}

#[test]
fn truth_table_restores_operator_bindings_on_exit() {
    let results = run_clean(
        r#"
(foundation boolean-and-table
  (truth-table and (1 1 -> 1) (1 0 -> 0) (0 1 -> 0) (0 0 -> 0)))
(a: a is a)
(b: b is b)
((a = true) has probability 1)
((b = true) has probability 0)
(? ((a = true) and (b = true)))
(with-foundation boolean-and-table
  (? ((a = true) and (b = true))))
(? ((a = true) and (b = true)))
"#,
    );
    // Outer uses default avg (0.5), foundation pins to 0, outer recovers.
    assert_eq!(results, vec![0.5, 0.0, 0.5]);
}

#[test]
fn truth_table_falls_through_to_host_default_for_unpinned_rows() {
    let results = run_clean(
        r#"
(foundation partial-and (truth-table and (1 1 -> 1)))
(a: a is a)
(b: b is b)
((a = true) has probability 0.6)
((b = true) has probability 0.4)
(with-foundation partial-and
  (? ((a = true) and (b = true))))
"#,
    );
    // 0.6 and 0.4 don't match the pinned (1,1) row, so the host default
    // (avg → 0.5) takes over.
    assert_eq!(results, vec![0.5]);
}

#[test]
fn truth_table_honours_symbolic_truth_constants() {
    let results = run_clean(
        r#"
(true: 1)
(false: 0)
(foundation symbolic-and
  (truth-table and (true true -> true) (true false -> false) (false true -> false) (false false -> false)))
(a: a is a)
(b: b is b)
((a = true) has probability 1)
((b = true) has probability 0)
(with-foundation symbolic-and
  (? ((a = true) and (b = true))))
"#,
    );
    assert_eq!(results, vec![0.0]);
}

#[test]
fn truth_table_models_kleene_three_valued_conjunction() {
    let results = run_clean(
        r#"
(unknown: 0.5)
(foundation kleene-and
  (truth-table and
    (1   1   -> 1)
    (1   0.5 -> 0.5)
    (0.5 1   -> 0.5)
    (1   0   -> 0)
    (0   1   -> 0)
    (0.5 0.5 -> 0.5)
    (0.5 0   -> 0)
    (0   0.5 -> 0)
    (0   0   -> 0)))
(a: a is a)
(b: b is b)
((a = true) has probability 0.5)
((b = true) has probability 1)
(with-foundation kleene-and
  (? ((a = true) and (b = true))))
"#,
    );
    assert_eq!(results, vec![0.5]);
}

#[test]
fn truth_table_exposes_truth_tables_on_foundation_report() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        r#"
(foundation boolean-classical
  (truth-table and (1 1 -> 1) (1 0 -> 0) (0 1 -> 0) (0 0 -> 0))
  (truth-table or  (0 0 -> 0) (0 1 -> 1) (1 0 -> 1) (1 1 -> 1)))
"#,
        None,
        &mut env,
    );
    assert!(out.diagnostics.is_empty());
    let report = env.foundation_report();
    let bc = report
        .foundations
        .iter()
        .find(|f| f.name == "boolean-classical")
        .expect("boolean-classical should appear in report");
    assert_eq!(bc.truth_tables.len(), 2);
    let and_table = bc
        .truth_tables
        .iter()
        .find(|(op, _)| op == "and")
        .expect("and truth table missing from report");
    assert_eq!(and_table.1.len(), 4);
    let printed = format_foundation_report(&report);
    assert!(
        printed.contains("truth tables: and(4 rows), or(4 rows)"),
        "printed report missing truth tables line:\n{}",
        printed
    );
}
