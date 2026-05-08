// Tests for the Pecan-style automatic-sequence domain plugin (issue #63).
// Mirrors js/tests/automatic-sequences.test.mjs so drift between runtimes
// fails CI.

use rml::{evaluate, evaluate_with_env, key_of, Env, RunResult};

#[test]
fn automatic_sequences_decides_thue_morse_cube_free() {
    let mut env = Env::new(None);
    let out = evaluate_with_env(
        "(domain automatic-sequences\n  (theorem thue-morse-cube-free))\n\
         (? thue-morse-cube-free)",
        None,
        &mut env,
    );

    assert!(
        out.diagnostics.is_empty(),
        "unexpected: {:?}",
        out.diagnostics
    );
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
    let decision = env
        .automatic_sequence_decisions
        .get("thue-morse-cube-free")
        .expect("decision should be recorded");
    assert!(decision.value);
    assert_eq!(
        key_of(&decision.certificate),
        "(buchi-emptiness thue-morse cube-free)"
    );
}

#[test]
fn automatic_sequences_domain_form_can_be_queried_directly() {
    let out = evaluate(
        "(? (domain automatic-sequences (theorem thue-morse-cube-free)))",
        None,
        None,
    );

    assert!(
        out.diagnostics.is_empty(),
        "unexpected: {:?}",
        out.diagnostics
    );
    assert_eq!(out.results, vec![RunResult::Num(1.0)]);
}

#[test]
fn automatic_sequences_rejects_unknown_theorem_with_e041() {
    let out = evaluate(
        "(domain automatic-sequences (theorem thue-morse-square-free))",
        None,
        None,
    );

    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E041");
    assert!(
        out.diagnostics[0]
            .message
            .contains("unknown automatic-sequences theorem"),
        "message was: {}",
        out.diagnostics[0].message
    );
}

#[test]
fn domain_rejects_unknown_plugin_with_e041() {
    let out = evaluate(
        "(domain imaginary (theorem thue-morse-cube-free))",
        None,
        None,
    );

    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E041");
    assert!(
        out.diagnostics[0].message.contains("Unknown domain plugin"),
        "message was: {}",
        out.diagnostics[0].message
    );
}
