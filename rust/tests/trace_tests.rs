// Tests for evaluation trace mode (issue #30).
// Mirrors js/tests/trace.test.mjs so any drift between the two
// implementations fails both test suites.

use rml::{
    evaluate, evaluate_with_options, format_trace_event, EvaluateOptions, TraceEvent,
};

const DEMO: &str = "(a: a is a)\n\
(!=: not =)\n\
(and: avg)\n\
((a = a) has probability 1)\n\
((a != a) has probability 0)\n\
(? ((a = a) and (a != a)))";

fn trace_demo() -> Vec<TraceEvent> {
    let out = evaluate_with_options(
        DEMO,
        Some("demo.lino"),
        EvaluateOptions {
            env: None,
            trace: true,
            ..EvaluateOptions::default()
        },
    );
    out.trace
}

#[test]
fn evaluate_without_trace_returns_empty_trace() {
    let out = evaluate("(? 1)", Some("q.lino"), None);
    assert!(out.trace.is_empty());
}

#[test]
fn evaluate_with_trace_returns_non_empty_trace() {
    let trace = trace_demo();
    assert!(!trace.is_empty());
    for ev in &trace {
        assert!(matches!(
            ev.kind.as_str(),
            "resolve" | "assign" | "lookup" | "eval"
        ));
        assert_eq!(ev.span.file.as_deref(), Some("demo.lino"));
        assert!(ev.span.line >= 1);
        assert!(ev.span.col >= 1);
    }
}

#[test]
fn trace_is_deterministic_across_runs() {
    let a = trace_demo();
    let b = trace_demo();
    let fmt = |events: &[TraceEvent]| -> String {
        events
            .iter()
            .map(format_trace_event)
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(fmt(&a), fmt(&b));
}

#[test]
fn trace_does_not_affect_results_or_diagnostics() {
    let plain = evaluate(DEMO, Some("demo.lino"), None);
    let traced = evaluate_with_options(
        DEMO,
        Some("demo.lino"),
        EvaluateOptions {
            env: None,
            trace: true,
            ..EvaluateOptions::default()
        },
    );
    assert_eq!(plain.results, traced.results);
    assert_eq!(plain.diagnostics, traced.diagnostics);
}

#[test]
fn resolve_event_for_aggregator_redef() {
    let out = evaluate_with_options(
        "(and: avg)",
        Some("op.lino"),
        EvaluateOptions {
            env: None,
            trace: true,
            ..EvaluateOptions::default()
        },
    );
    let resolves: Vec<_> = out
        .trace
        .iter()
        .filter(|e| e.kind == "resolve")
        .collect();
    assert_eq!(resolves.len(), 1);
    assert_eq!(resolves[0].detail, "(and: avg)");
    assert_eq!(resolves[0].span.file.as_deref(), Some("op.lino"));
    assert_eq!(resolves[0].span.line, 1);
    assert_eq!(resolves[0].span.col, 1);
}

#[test]
fn assign_event_for_probability_form() {
    let out = evaluate_with_options(
        "((a = a) has probability 1)",
        Some("p.lino"),
        EvaluateOptions {
            env: None,
            trace: true,
            ..EvaluateOptions::default()
        },
    );
    let assigns: Vec<_> = out
        .trace
        .iter()
        .filter(|e| e.kind == "assign")
        .collect();
    assert_eq!(assigns.len(), 1);
    assert_eq!(assigns[0].detail, "(a = a) ← 1");
    assert_eq!(assigns[0].span.line, 1);
}

#[test]
fn lookup_event_when_assigned_equality_fires() {
    let src = "((a = a) has probability 0.7)\n(? (a = a))";
    let out = evaluate_with_options(
        src,
        Some("lk.lino"),
        EvaluateOptions {
            env: None,
            trace: true,
            ..EvaluateOptions::default()
        },
    );
    let lookups: Vec<_> = out
        .trace
        .iter()
        .filter(|e| e.kind == "lookup")
        .collect();
    assert!(!lookups.is_empty());
    assert!(
        lookups[0].detail.contains("(a = a) → 0.7"),
        "detail was: {}",
        lookups[0].detail
    );
    assert_eq!(lookups[0].span.line, 2);
}

#[test]
fn eval_event_per_top_level_form() {
    let trace = trace_demo();
    let evals: Vec<_> = trace.iter().filter(|e| e.kind == "eval").collect();
    // One eval event per top-level form (6 forms in DEMO).
    assert_eq!(evals.len(), 6);
    let last = evals.last().unwrap();
    assert!(
        last.detail.ends_with("→ query 0.5"),
        "detail was: {}",
        last.detail
    );
    assert_eq!(last.span.line, 6);
}

#[test]
fn format_trace_event_renders_span_kind_detail() {
    let out = evaluate_with_options(
        "(and: avg)",
        Some("fmt.lino"),
        EvaluateOptions {
            env: None,
            trace: true,
            ..EvaluateOptions::default()
        },
    );
    let line = format_trace_event(&out.trace[0]);
    assert_eq!(line, "[span fmt.lino:1:1] resolve (and: avg)");
}

#[test]
fn format_trace_event_falls_back_to_input_when_no_file() {
    let out = evaluate_with_options(
        "(and: avg)",
        None,
        EvaluateOptions {
            env: None,
            trace: true,
            ..EvaluateOptions::default()
        },
    );
    let line = format_trace_event(&out.trace[0]);
    assert_eq!(line, "[span <input>:1:1] resolve (and: avg)");
}
