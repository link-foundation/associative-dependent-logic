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

use rml::{evaluate, evaluate_with_env, format_foundation_report, Aggregator, Env, Op, RunResult};

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
