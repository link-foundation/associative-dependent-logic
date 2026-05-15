// Tests for `lib/self/operators.lino` (issue #87).
//
// Mirrors js/tests/self-operators.test.mjs so both runtimes keep the encoded
// operator library importable and tied to host arithmetic/aggregator outputs.

use rml::{
    dec_round, evaluate, evaluate_file, evaluate_with_env, key_of, Env, EvaluateOptions,
    EvaluateResult, RunResult,
};
use std::fs;
use std::path::PathBuf;

const REQUIRED_RELATIONS: &[(&str, &str)] = &[
    ("avg", "(avg a b ((a + b) / 2))"),
    (
        "min",
        "(min a b (self.not (self.or (self.not a) (self.not b))))",
    ),
    ("max", "(max a b (self.or a b))"),
    ("product", "(product a b (a * b))"),
    (
        "probabilistic_sum",
        "(probabilistic_sum a b (1 - ((1 - a) * (1 - b))))",
    ),
    ("decimal-sum", "(decimal-sum left right (left + right))"),
    (
        "decimal-difference",
        "(decimal-difference left right (left - right))",
    ),
    (
        "decimal-product",
        "(decimal-product left right (left * right))",
    ),
    (
        "decimal-quotient",
        "(decimal-quotient left right (left / right))",
    ),
];

const OUTPUT_CASES: &[(&str, &str, &str)] = &[
    (
        "avg",
        "(? (ops.avg 0.1 0.2))",
        "(and: avg)\n(? (0.1 and 0.2))",
    ),
    (
        "min",
        "(? (ops.min 0.42 0.9))",
        "(and: min)\n(? (0.42 and 0.9))",
    ),
    (
        "max",
        "(? (ops.max 0.42 0.9))",
        "(or: max)\n(? (0.42 or 0.9))",
    ),
    (
        "product",
        "(? (ops.product 0.2 0.3))",
        "(and: product)\n(? (0.2 and 0.3))",
    ),
    (
        "probabilistic_sum",
        "(? (ops.probabilistic_sum 0.2 0.3))",
        "(or: probabilistic_sum)\n(? (0.2 or 0.3))",
    ),
    (
        "decimal-sum",
        "(? (ops.decimal-sum 0.1 0.2))",
        "(? (0.1 + 0.2))",
    ),
    (
        "decimal-difference",
        "(? (ops.decimal-difference 0.3 0.1))",
        "(? (0.3 - 0.1))",
    ),
    (
        "decimal-product",
        "(? (ops.decimal-product 0.1 0.2))",
        "(? (0.1 * 0.2))",
    ),
    (
        "decimal-quotient",
        "(? (ops.decimal-quotient 1 3))",
        "(? (1 / 3))",
    ),
    (
        "decimal-quotient-zero",
        "(? (ops.decimal-quotient 1 0))",
        "(? (1 / 0))",
    ),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn operators_path() -> PathBuf {
    repo_root().join("lib").join("self").join("operators.lino")
}

fn evaluate_from_root(source: &str) -> EvaluateResult {
    let virtual_root_file = repo_root().join("inline-self-operators-test.lino");
    evaluate(source, Some(virtual_root_file.to_str().unwrap()), None)
}

fn assert_clean(out: &EvaluateResult) {
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
}

fn single_number(out: EvaluateResult, label: &str) -> f64 {
    assert_clean(&out);
    assert_eq!(out.results.len(), 1, "{}: expected one result", label);
    match out.results[0] {
        RunResult::Num(value) => value,
        RunResult::Type(ref value) => panic!("{}: expected number, got {}", label, value),
        RunResult::Foundation(ref report) => panic!(
            "{}: expected number, got foundation report for {}",
            label, report.active_foundation
        ),
    }
}

#[test]
fn self_operators_is_importable() {
    let path = operators_path();
    let out = evaluate_file(path.to_str().unwrap(), EvaluateOptions::default());
    assert_clean(&out);
}

#[test]
fn self_operators_declares_issue_surface_relations() {
    let path = operators_path();
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("could not read {}: {}", path.display(), err));
    let mut env = Env::new(None);
    let out = evaluate_with_env(&source, Some(path.to_str().unwrap()), &mut env);
    assert_clean(&out);

    for (name, expected_clause) in REQUIRED_RELATIONS {
        let clauses = env
            .relations
            .get(*name)
            .unwrap_or_else(|| panic!("missing relation {}", name));
        let actual: Vec<String> = clauses.iter().map(key_of).collect();
        assert_eq!(actual, vec![expected_clause.to_string()]);
    }
}

#[test]
fn self_operator_outputs_match_host_to_12_decimal_places() {
    for (name, encoded, host) in OUTPUT_CASES {
        let encoded_source = format!(
            "\n(import \"lib/self/operators.lino\" as ops)\n{}\n",
            encoded
        );
        let encoded_value = single_number(evaluate_from_root(&encoded_source), name);
        let host_value = single_number(evaluate_from_root(host), &format!("{} host", name));

        assert_eq!(dec_round(encoded_value), dec_round(host_value), "{}", name);
    }
}
