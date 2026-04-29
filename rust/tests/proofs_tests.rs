// Tests for the proof-producing evaluator (issue #35).
// Mirrors js/tests/proofs.test.mjs so any drift between the two
// implementations fails both test suites. Covers the global
// `with_proofs` option, the inline `(? expr with proof)` keyword
// pair, derivation shape per built-in operator, the
// `parse(print(proof)) == proof` round-trip, and that proofs do
// not affect query results or diagnostics.

use rml::{
    build_proof, evaluate, evaluate_with_options, is_structurally_same, key_of, parse_one,
    tokenize_one, Env, EvaluateOptions, Node, RunResult,
};

// Build an `EvaluateOptions` with proofs enabled.
fn opts_with_proofs() -> EvaluateOptions {
    EvaluateOptions {
        with_proofs: true,
        ..EvaluateOptions::default()
    }
}

// Round-trip a derivation through print -> tokenize -> parse and assert
// the resulting AST is structurally identical to the original. This is
// the acceptance criterion from the issue: "round-trip parse(print(D)) == D".
fn assert_round_trip(proof: &Node) {
    let printed = key_of(proof);
    let reparsed = parse_one(&tokenize_one(&printed)).expect("reparse failed");
    assert!(
        is_structurally_same(proof, &reparsed),
        "proof did not round-trip: {}",
        printed
    );
}

fn rule_name(proof: &Node) -> &str {
    match proof {
        Node::List(children) => match &children[1] {
            Node::Leaf(s) => s.as_str(),
            _ => panic!("rule name slot was not a leaf"),
        },
        _ => panic!("proof was not a list"),
    }
}

// ===== evaluate() returns proofs only when requested =====

#[test]
fn omits_proofs_when_neither_flag_nor_inline_keyword_used() {
    let out = evaluate("(? 1)", None, None);
    assert!(out.proofs.is_empty());
}

#[test]
fn returns_proofs_array_when_with_proofs_is_true() {
    let out = evaluate_with_options("(? 1)", None, opts_with_proofs());
    assert_eq!(out.proofs.len(), 1);
    assert!(out.proofs[0].is_some());
}

#[test]
fn returns_proofs_array_when_inline_with_proof_is_present() {
    let out = evaluate("(? 1)\n(? 2 with proof)", None, None);
    assert_eq!(out.proofs.len(), 2);
    // First query did not opt in -> None; second one did.
    assert!(out.proofs[0].is_none());
    assert!(out.proofs[1].is_some());
}

// ===== issue example reproduction =====

#[test]
fn produces_canonical_structural_equality_witness() {
    let out = evaluate("(a: a is a)\n(? (a = a) with proof)", None, None);
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
    assert_eq!(out.proofs.len(), 1);
    let proof = out.proofs[0].as_ref().expect("proof missing");
    assert_eq!(key_of(proof), "(by structural-equality (a a))");
    assert_round_trip(proof);
}

#[test]
fn produces_same_witness_under_global_flag() {
    let out =
        evaluate_with_options("(a: a is a)\n(? (a = a))", None, opts_with_proofs());
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
    let proof = out.proofs[0].as_ref().expect("proof missing");
    assert_eq!(key_of(proof), "(by structural-equality (a a))");
}

// ===== per-rule witness shapes =====

#[test]
fn records_assigned_equality_when_assignment_exists() {
    let src = "((a = a) has probability 0.7)\n(? (a = a))";
    let out = evaluate_with_options(src, None, opts_with_proofs());
    assert_eq!(out.results, vec![RunResult::Num(0.7)]);
    let proof = out.proofs[0].as_ref().expect("proof missing");
    assert_eq!(key_of(proof), "(by assigned-equality (a a))");
    assert_round_trip(proof);
}

#[test]
fn records_assigned_inequality_for_inequality_query() {
    let src = "((a = a) has probability 0.7)\n(? (a != a))";
    let out = evaluate_with_options(src, None, opts_with_proofs());
    let proof = out.proofs[0].as_ref().expect("proof missing");
    assert_eq!(key_of(proof), "(by assigned-inequality (a a))");
    assert_round_trip(proof);
}

#[test]
fn records_numeric_equality_when_no_assignment_and_operands_differ() {
    // 1 = 2 is false; rule fires regardless of clamped truth value.
    let out = evaluate_with_options("(? (1 = 2))", None, opts_with_proofs());
    let proof = out.proofs[0].as_ref().expect("proof missing");
    assert_eq!(key_of(proof), "(by numeric-equality (1 2))");
    assert_round_trip(proof);
}

#[test]
fn records_arithmetic_rules() {
    let src = "(? (1 + 2))\n(? (5 - 2))\n(? (3 * 4))\n(? (8 / 2))";
    let out = evaluate_with_options(src, None, opts_with_proofs());
    assert_eq!(out.results.len(), 4);
    let rules: Vec<&str> = out
        .proofs
        .iter()
        .map(|p| rule_name(p.as_ref().expect("proof missing")))
        .collect();
    assert_eq!(rules, vec!["sum", "difference", "product", "quotient"]);
    for p in &out.proofs {
        assert_round_trip(p.as_ref().unwrap());
    }
}

#[test]
fn records_and_or_for_binary_infix_logic() {
    let src = "(? (1 and 0))\n(? (1 or 0))";
    let out = evaluate_with_options(src, None, opts_with_proofs());
    assert_eq!(rule_name(out.proofs[0].as_ref().unwrap()), "and");
    assert_eq!(rule_name(out.proofs[1].as_ref().unwrap()), "or");
    for p in &out.proofs {
        assert_round_trip(p.as_ref().unwrap());
    }
}

#[test]
fn records_both_neither_for_composite_chains() {
    let src = "(? (both 1 and 0 and 1))\n(? (neither 0 nor 0))";
    let out = evaluate_with_options(src, None, opts_with_proofs());
    let p0 = out.proofs[0].as_ref().unwrap();
    assert_eq!(rule_name(p0), "both");
    if let Node::List(children) = p0 {
        // (by both s1 s2 s3) -> length 5
        assert_eq!(children.len(), 5);
    } else {
        panic!("not a list");
    }
    assert_eq!(rule_name(out.proofs[1].as_ref().unwrap()), "neither");
    for p in &out.proofs {
        assert_round_trip(p.as_ref().unwrap());
    }
}

#[test]
fn records_prefix_operator_names() {
    let src = "(? (not 1))\n(? (and 1 1))\n(? (or 0 1))";
    let out = evaluate_with_options(src, None, opts_with_proofs());
    let rules: Vec<&str> = out
        .proofs
        .iter()
        .map(|p| rule_name(p.as_ref().unwrap()))
        .collect();
    assert_eq!(rules, vec!["not", "and", "or"]);
    for p in &out.proofs {
        assert_round_trip(p.as_ref().unwrap());
    }
}

#[test]
fn records_assigned_probability_for_top_level_form() {
    let env = Env::new(None);
    // ((a = a) has probability 0.5)
    let node = Node::List(vec![
        Node::List(vec![
            Node::Leaf("a".to_string()),
            Node::Leaf("=".to_string()),
            Node::Leaf("a".to_string()),
        ]),
        Node::Leaf("has".to_string()),
        Node::Leaf("probability".to_string()),
        Node::Leaf("0.5".to_string()),
    ]);
    let proof = build_proof(&node, &env);
    assert_eq!(key_of(&proof), "(by assigned-probability (a = a) 0.5)");
    assert_round_trip(&proof);
}

#[test]
fn records_type_universe_prop_pi_lambda() {
    let env = Env::new(None);

    // Type 0
    let u = build_proof(
        &Node::List(vec![Node::Leaf("Type".to_string()), Node::Leaf("0".to_string())]),
        &env,
    );
    assert_eq!(key_of(&u), "(by type-universe 0)");
    assert_round_trip(&u);

    // Prop
    let p = build_proof(&Node::List(vec![Node::Leaf("Prop".to_string())]), &env);
    assert_eq!(key_of(&p), "(by prop)");
    assert_round_trip(&p);

    // Pi (x: A) B
    let pi = build_proof(
        &Node::List(vec![
            Node::Leaf("Pi".to_string()),
            Node::List(vec![Node::Leaf("x:".to_string()), Node::Leaf("A".to_string())]),
            Node::Leaf("B".to_string()),
        ]),
        &env,
    );
    assert_eq!(key_of(&pi), "(by pi-formation (x: A) B)");
    assert_round_trip(&pi);

    // lambda (x: A) x
    let lam = build_proof(
        &Node::List(vec![
            Node::Leaf("lambda".to_string()),
            Node::List(vec![Node::Leaf("x:".to_string()), Node::Leaf("A".to_string())]),
            Node::Leaf("x".to_string()),
        ]),
        &env,
    );
    assert_eq!(key_of(&lam), "(by lambda-formation (x: A) x)");
    assert_round_trip(&lam);
}

// ===== proofs are index-aligned with results =====

#[test]
fn emits_none_for_bare_queries_in_inline_mode() {
    let src = "(? 1)\n(? 0 with proof)\n(? 1)";
    let out = evaluate(src, None, None);
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.proofs.len(), 3);
    assert!(out.proofs[0].is_none());
    assert!(out.proofs[1].is_some());
    assert!(out.proofs[2].is_none());
}

#[test]
fn produces_proof_for_every_query_when_with_proofs_is_true() {
    let src = "(a: a is a)\n(? 1)\n(? (1 + 0))\n(? (a = a))";
    let out = evaluate_with_options(src, None, opts_with_proofs());
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.proofs.len(), 3);
    for p in &out.proofs {
        let proof = p.as_ref().expect("proof missing");
        assert_round_trip(proof);
    }
}

// ===== proofs do not affect query results or diagnostics =====

#[test]
fn produces_identical_results_and_diagnostics_with_and_without_proofs() {
    let src = [
        "(a: a is a)",
        "(b: b is b)",
        "((a = a) has probability 1)",
        "((b = b) has probability 0)",
        "(? ((a = a) and (b = b)))",
    ]
    .join("\n");
    let plain = evaluate(&src, None, None);
    let proven = evaluate_with_options(&src, None, opts_with_proofs());
    assert_eq!(plain.results, proven.results);
    assert_eq!(plain.diagnostics, proven.diagnostics);
}

// ===== round-trip property holds for every produced proof =====

#[test]
fn round_trip_holds_for_representative_operator_bundle() {
    let src = [
        "(a: a is a)",
        "((a = a) has probability 0.7)",
        "(? (a = a))",
        "(? (1 + 2))",
        "(? (5 - 2))",
        "(? (3 * 4))",
        "(? (8 / 2))",
        "(? (not 1))",
        "(? (1 and 0))",
        "(? (0 or 1))",
        "(? (both 1 and 1 and 0))",
        "(? (neither 0 nor 0))",
        "(? (1 = 2))",
        "(? (1 != 2))",
    ]
    .join("\n");
    let out = evaluate_with_options(&src, None, opts_with_proofs());
    assert_eq!(out.proofs.len(), out.results.len());
    for p in &out.proofs {
        let proof = p.as_ref().expect("proof missing");
        assert_round_trip(proof);
    }
}
