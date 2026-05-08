// Tests for the link-based tactic engine (issue #55).
// Mirrors js/tests/tactics.test.mjs so drift between runtimes fails CI.

use rml::{
    goal_to_tptp, key_of, parse_atp_status, parse_one, parse_tactic_links, rewrite_with_options,
    run_tactics, run_tactics_with_options, simplify, simplify_with_options, tokenize_one,
    AtpOptions, AtpStatus, AtpStatusKind, Node, ProofGoal, ProofState, RewriteDirection,
    RewriteOccurrence, RewriteOptions, SimplifyOptions, TacticOptions,
};

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

fn mock_atp_options(script: &str, timeout_ms: u64) -> AtpOptions {
    AtpOptions {
        path: Some("/bin/sh".to_string()),
        args: vec!["-c".to_string(), script.to_string()],
        name: Some("mock-atp".to_string()),
        timeout_ms,
    }
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
            atp: AtpOptions::default(),
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
fn exports_first_order_goals_and_context_to_tptp_fof_problems() {
    let goal = ProofGoal {
        goal: link("(P a)"),
        context: vec![link("(forall (Thing x) (P x))"), link("(a of Thing)")],
    };
    let tptp = goal_to_tptp(&goal).expect("TPTP export should succeed");

    assert!(tptp.contains("fof(rml_context_1, axiom, (![X] : (p(X))))."));
    assert!(tptp.contains("fof(rml_context_2, axiom, (thing(a)))."));
    assert!(tptp.contains("fof(rml_goal, conjecture, (p(a)))."));
}

#[test]
fn parses_szs_statuses_from_atp_output() {
    assert_eq!(
        parse_atp_status("% SZS status Theorem for rml_goal\n"),
        Some(AtpStatus {
            status: "Theorem".to_string(),
            kind: AtpStatusKind::Proved,
        })
    );
    assert_eq!(
        parse_atp_status("% SZS status Unknown for rml_goal\n"),
        Some(AtpStatus {
            status: "Unknown".to_string(),
            kind: AtpStatusKind::Unknown,
        })
    );
}

#[test]
fn closes_current_goal_with_configured_atp_proving_status() {
    let out = run_tactics_with_options(
        state(&["(P a)"]),
        &[link("(by atp)")],
        TacticOptions {
            atp: mock_atp_options(
                "cat >/dev/null; printf '%s\n' '% SZS status Theorem for rml_goal'",
                1000,
            ),
            ..TacticOptions::default()
        },
    );

    assert!(out.diagnostics.is_empty());
    assert!(out.state.goals.is_empty());
    assert_eq!(
        out.state.proof.iter().map(key_of).collect::<Vec<_>>(),
        vec!["(by atp-trusted mock-atp)"]
    );
}

#[test]
fn reports_atp_failure_modes_without_closing_the_goal() {
    let unconfigured = run_tactics(state(&["(P a)"]), &[link("(by atp)")]);
    assert_eq!(unconfigured.diagnostics.len(), 1);
    assert!(unconfigured.diagnostics[0]
        .message
        .contains("ATP path is not configured"));
    assert_eq!(goal_keys(&unconfigured.state), vec!["(P a)"]);

    let unknown = run_tactics_with_options(
        state(&["(P a)"]),
        &[link("(by atp)")],
        TacticOptions {
            atp: mock_atp_options(
                "cat >/dev/null; printf '%s\n' '% SZS status Unknown for rml_goal'",
                1000,
            ),
            ..TacticOptions::default()
        },
    );
    assert_eq!(unknown.diagnostics.len(), 1);
    assert!(unknown.diagnostics[0]
        .message
        .contains("ATP returned Unknown"));
    assert_eq!(goal_keys(&unknown.state), vec!["(P a)"]);

    let timed_out = run_tactics_with_options(
        state(&["(P a)"]),
        &[link("(by atp)")],
        TacticOptions {
            atp: mock_atp_options("sleep 1", 1),
            ..TacticOptions::default()
        },
    );
    assert_eq!(timed_out.diagnostics.len(), 1);
    assert!(timed_out.diagnostics[0].message.contains("ATP timed out"));
    assert_eq!(goal_keys(&timed_out.state), vec!["(P a)"]);
}
