// Tests for the link-based tactic engine (issue #55).
// Mirrors js/tests/tactics.test.mjs so drift between runtimes fails CI.

use rml::{
    key_of, parse_one, parse_tactic_links, rewrite_with_options, run_tactics,
    run_tactics_with_options, simplify, simplify_with_options, tokenize_one, Node, ProofGoal,
    ProofState, RewriteDirection, RewriteOccurrence, RewriteOptions, SimplifyOptions,
    TacticOptions,
};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn link(src: &str) -> Node {
    parse_one(&tokenize_one(src)).expect("parse failed")
}

fn state(goals: &[&str]) -> ProofState {
    ProofState {
        goals: goals
            .iter()
            .map(|goal| ProofGoal {
                goal: link(goal),
                context: Vec::new(),
            })
            .collect(),
        proof: Vec::new(),
    }
}

fn goal_keys(proof_state: &ProofState) -> Vec<String> {
    proof_state
        .goals
        .iter()
        .map(|goal| key_of(&goal.goal))
        .collect()
}

fn with_temp_dir<T>(f: impl FnOnce(&PathBuf) -> T) -> T {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("rml-smt-rust-{}-{nonce}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir should be creatable");
    let result = f(&dir);
    let _ = fs::remove_dir_all(&dir);
    result
}

#[test]
fn closes_equality_goal_with_by_reflexivity() {
    let out = run_tactics(state(&["(a = a)"]), &[link("(by reflexivity)")]);

    assert!(out.diagnostics.is_empty());
    assert!(out.state.goals.is_empty());
    assert_eq!(
        out.state.proof.iter().map(key_of).collect::<Vec<_>>(),
        vec!["(by reflexivity)"]
    );
}

#[test]
fn parses_tactic_text_into_links() {
    let tactics = parse_tactic_links("(reflexivity)");
    let out = run_tactics(state(&["(a = a)"]), &tactics);

    assert!(out.diagnostics.is_empty());
    assert!(out.state.goals.is_empty());
    assert_eq!(
        out.state.proof.iter().map(key_of).collect::<Vec<_>>(),
        vec!["(reflexivity)"]
    );
}

#[test]
fn transforms_goals_with_symmetry_and_transitivity() {
    let out = run_tactics(
        state(&["(a = c)"]),
        &[link("(symmetry)"), link("(transitivity b)")],
    );

    assert!(out.diagnostics.is_empty());
    assert_eq!(goal_keys(&out.state), vec!["(c = b)", "(b = a)"]);
    assert_eq!(
        out.state.proof.iter().map(key_of).collect::<Vec<_>>(),
        vec!["(symmetry)", "(transitivity b)"]
    );
}

#[test]
fn introduces_pi_binders_into_current_context() {
    let introduced = run_tactics(
        state(&["(Pi (Natural n) (n = n))"]),
        &[link("(introduce k)")],
    );

    assert!(introduced.diagnostics.is_empty());
    assert_eq!(goal_keys(&introduced.state), vec!["(k = k)"]);
    assert_eq!(
        introduced.state.goals[0]
            .context
            .iter()
            .map(key_of)
            .collect::<Vec<_>>(),
        vec!["(k of Natural)"]
    );

    let closed = run_tactics(introduced.state, &[link("(by reflexivity)")]);
    assert!(closed.diagnostics.is_empty());
    assert!(closed.state.goals.is_empty());
}

#[test]
fn adds_assumptions_with_suppose_and_closes_them_with_exact() {
    let supposed = run_tactics(state(&["(p = q)"]), &[link("(suppose (p = q))")]);

    assert!(supposed.diagnostics.is_empty());
    assert_eq!(goal_keys(&supposed.state), vec!["(p = q)"]);
    assert_eq!(
        supposed.state.goals[0]
            .context
            .iter()
            .map(key_of)
            .collect::<Vec<_>>(),
        vec!["(p = q)"]
    );

    let closed = run_tactics(supposed.state, &[link("(exact (p = q))")]);
    assert!(closed.diagnostics.is_empty());
    assert!(closed.state.goals.is_empty());
}

#[test]
fn rewrites_current_goal_with_equality_link() {
    let out = run_tactics(
        state(&["((f a) = (f a))"]),
        &[link("(rewrite (a = b) in goal)"), link("(by reflexivity)")],
    );

    assert!(out.diagnostics.is_empty());
    assert!(out.state.goals.is_empty());
    assert_eq!(
        out.state.proof.iter().map(key_of).collect::<Vec<_>>(),
        vec!["(rewrite (a = b) in goal)", "(by reflexivity)"]
    );
}

#[test]
fn rewrites_in_requested_direction() {
    let rewritten = rewrite_with_options(
        &link("(b = b)"),
        &link("(a = b)"),
        RewriteOptions {
            direction: RewriteDirection::Backward,
            occurrence: RewriteOccurrence::All,
        },
    )
    .expect("rewrite should succeed");
    assert_eq!(key_of(&rewritten.node), "(a = a)");

    let out = run_tactics(
        state(&["(b = b)"]),
        &[link("(rewrite <- (a = b) in goal)"), link("(by reflexivity)")],
    );

    assert!(out.diagnostics.is_empty());
    assert!(out.state.goals.is_empty());
}

#[test]
fn rewrites_only_selected_occurrence() {
    let out = run_tactics(
        state(&["((pair a a) = (pair b a))"]),
        &[link("(rewrite (a = b) in goal at 2)")],
    );

    assert!(out.diagnostics.is_empty());
    assert_eq!(goal_keys(&out.state), vec!["((pair a b) = (pair b a))"]);
}

#[test]
fn simplifies_current_goal_with_configured_rewrite_rules() {
    let simplified = simplify(&link("((f a) = (f a))"), &[link("(a = b)")])
        .expect("simplify should succeed");
    assert_eq!(key_of(&simplified), "((f b) = (f b))");

    let out = run_tactics_with_options(
        state(&["((f a) = (f a))"]),
        &[link("(simplify in goal)"), link("(by reflexivity)")],
        TacticOptions {
            rewrite_rules: vec![link("(a = b)")],
            simplify_max_steps: 10,
            ..TacticOptions::default()
        },
    );

    assert!(out.diagnostics.is_empty());
    assert!(out.state.goals.is_empty());
}

#[test]
fn stops_simplification_when_termination_guard_is_reached() {
    let err = simplify_with_options(
        &link("(a = a)"),
        &[link("(a = b)"), link("(b = a)")],
        SimplifyOptions { max_steps: 3 },
    )
    .expect_err("cyclic rules should hit the guard");

    assert_eq!(err.code, "E039");
    assert!(err.message.contains("termination guard"));
}

#[test]
fn runs_per_case_tactic_links_during_induction() {
    let out = run_tactics(
        state(&["(n = n)"]),
        &[link(
            "(induction n (case zero (by reflexivity)) (case (succ m) (by reflexivity)))",
        )],
    );

    assert!(out.diagnostics.is_empty());
    assert!(out.state.goals.is_empty());
    assert_eq!(
        key_of(&out.state.proof[0]),
        "(induction n (case zero (by reflexivity)) (case (succ m) (by reflexivity)))"
    );
}

#[test]
fn failed_tactic_reports_current_goal() {
    let out = run_tactics(state(&["(a = b)"]), &[link("(by reflexivity)")]);

    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E039");
    assert!(out.diagnostics[0].message.contains("current goal: (a = b)"));
    assert_eq!(goal_keys(&out.state), vec!["(a = b)"]);
}

#[test]
fn closes_goal_with_smt_when_solver_returns_unsat() {
    with_temp_dir(|dir| {
        let capture = dir.join("input.smt2");
        let out = run_tactics_with_options(
            state(&["(a = a)"]),
            &[link("(by smt)")],
            TacticOptions {
                smt_solver: Some("sh".to_string()),
                smt_solver_args: vec![
                    "-c".to_string(),
                    "cat > \"$1\"; echo unsat".to_string(),
                    "sh".to_string(),
                    capture.display().to_string(),
                ],
                smt_timeout_ms: 1000,
                ..TacticOptions::default()
            },
        );

        assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
        assert!(out.state.goals.is_empty());
        assert_eq!(
            out.state.proof.iter().map(key_of).collect::<Vec<_>>(),
            vec!["(by smt-trusted sh)"]
        );

        let smt_lib = fs::read_to_string(capture).expect("SMT input should be captured");
        assert!(smt_lib.contains("(declare-const |a| Real)"));
        assert!(smt_lib.contains("(assert (not (= |a| |a|)))"));
        assert!(smt_lib.contains("(check-sat)"));
    });
}

#[test]
fn reports_unknown_from_smt_solver_without_closing_goal() {
    let out = run_tactics_with_options(
        state(&["(a = a)"]),
        &[link("(by smt)")],
        TacticOptions {
            smt_solver: Some("sh".to_string()),
            smt_solver_args: vec!["-c".to_string(), "echo unknown".to_string()],
            smt_timeout_ms: 1000,
            ..TacticOptions::default()
        },
    );

    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E039");
    assert!(out.diagnostics[0].message.contains("returned unknown"));
    assert_eq!(goal_keys(&out.state), vec!["(a = a)"]);
}

#[test]
fn reports_smt_solver_timeout_without_closing_goal() {
    let out = run_tactics_with_options(
        state(&["(a = a)"]),
        &[link("(by smt)")],
        TacticOptions {
            smt_solver: Some("sh".to_string()),
            smt_solver_args: vec!["-c".to_string(), "sleep 2".to_string()],
            smt_timeout_ms: 20,
            ..TacticOptions::default()
        },
    );

    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E039");
    assert!(out.diagnostics[0].message.contains("timed out"));
    assert_eq!(goal_keys(&out.state), vec!["(a = a)"]);
}
