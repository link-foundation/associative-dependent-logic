// Bidirectional type checker tests for issue #42.
//
// Mirrors `js/tests/bidirectional.test.mjs` so both runtimes lock in the
// same `synth(term, env)` / `check(term, type, env)` semantics.

use rml::{check, eval_node, evaluate, key_of, synth, Env, Node};
use std::fs;
use std::path::PathBuf;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("examples")
}

fn leaf(s: &str) -> Node {
    Node::Leaf(s.to_string())
}

fn list(children: Vec<Node>) -> Node {
    Node::List(children)
}

fn natural_env() -> Env {
    let mut env = Env::new(None);
    eval_node(
        &list(vec![leaf("Type:"), leaf("Type"), leaf("Type")]),
        &mut env,
    );
    eval_node(
        &list(vec![
            leaf("Natural:"),
            list(vec![leaf("Type"), leaf("0")]),
            leaf("Natural"),
        ]),
        &mut env,
    );
    eval_node(
        &list(vec![leaf("Boolean:"), leaf("Type"), leaf("Boolean")]),
        &mut env,
    );
    eval_node(
        &list(vec![leaf("zero:"), leaf("Natural"), leaf("zero")]),
        &mut env,
    );
    eval_node(
        &list(vec![
            leaf("identity:"),
            leaf("lambda"),
            list(vec![leaf("Natural"), leaf("x")]),
            leaf("x"),
        ]),
        &mut env,
    );
    eval_node(
        &list(vec![
            leaf("succ:"),
            list(vec![
                leaf("Pi"),
                list(vec![leaf("Natural"), leaf("n")]),
                leaf("Natural"),
            ]),
        ]),
        &mut env,
    );
    env
}

#[test]
fn synth_known_term_returns_recorded_type() {
    let mut env = natural_env();
    let result = synth(&leaf("zero"), &mut env);
    assert!(result.diagnostics.is_empty());
    assert_eq!(key_of(&result.typ.unwrap()), "Natural");
}

#[test]
fn synth_universe_has_successor_universe_type() {
    let mut env = Env::new(None);
    let r0 = synth(&list(vec![leaf("Type"), leaf("0")]), &mut env);
    assert!(r0.diagnostics.is_empty());
    assert_eq!(key_of(&r0.typ.unwrap()), "(Type 1)");
    let r2 = synth(&list(vec![leaf("Type"), leaf("2")]), &mut env);
    assert_eq!(key_of(&r2.typ.unwrap()), "(Type 3)");
}

#[test]
fn synth_pi_returns_type_zero() {
    let mut env = natural_env();
    let result = synth(
        &list(vec![
            leaf("Pi"),
            list(vec![leaf("Natural"), leaf("n")]),
            leaf("Natural"),
        ]),
        &mut env,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(key_of(&result.typ.unwrap()), "(Type 0)");
}

#[test]
fn synth_lambda_extends_context_and_returns_pi_type() {
    let mut env = natural_env();
    let result = synth(
        &list(vec![
            leaf("lambda"),
            list(vec![leaf("Natural"), leaf("x")]),
            leaf("x"),
        ]),
        &mut env,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(key_of(&result.typ.unwrap()), "(Pi (Natural x) Natural)");
}

#[test]
fn synth_apply_substitutes_argument_into_codomain() {
    let mut env = natural_env();
    let result = synth(
        &list(vec![leaf("apply"), leaf("identity"), leaf("zero")]),
        &mut env,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(key_of(&result.typ.unwrap()), "Natural");
}

#[test]
fn synth_subst_reduces_then_synthesises() {
    let mut env = natural_env();
    let result = synth(
        &list(vec![
            leaf("subst"),
            leaf("x"),
            leaf("x"),
            leaf("zero"),
        ]),
        &mut env,
    );
    assert!(result.diagnostics.is_empty());
    assert_eq!(key_of(&result.typ.unwrap()), "Natural");
}

#[test]
fn synth_unknown_symbol_emits_e020() {
    let mut env = natural_env();
    let result = synth(&leaf("mystery"), &mut env);
    assert!(result.typ.is_none());
    assert_eq!(result.diagnostics.len(), 1);
    assert_eq!(result.diagnostics[0].code, "E020");
    assert!(result.diagnostics[0].message.contains("symbol `mystery`"));
}

#[test]
fn synth_apply_against_non_pi_head_emits_e022() {
    let mut env = natural_env();
    let result = synth(
        &list(vec![leaf("apply"), leaf("zero"), leaf("zero")]),
        &mut env,
    );
    assert!(result.typ.is_none());
    let codes: Vec<String> = result
        .diagnostics
        .iter()
        .map(|d| d.code.clone())
        .collect();
    assert!(
        codes.iter().any(|c| c == "E022"),
        "expected E022 in {:?}",
        codes
    );
}

#[test]
fn synth_lambda_with_malformed_binder_emits_e024() {
    let mut env = natural_env();
    let result = synth(
        &list(vec![
            leaf("lambda"),
            list(vec![leaf("x")]),
            leaf("x"),
        ]),
        &mut env,
    );
    assert!(result.typ.is_none());
    assert_eq!(result.diagnostics[0].code, "E024");
}

#[test]
fn check_term_against_recorded_type_succeeds() {
    let mut env = natural_env();
    let result = check(&leaf("zero"), &leaf("Natural"), &mut env);
    assert!(result.ok);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn check_lambda_against_pi_directly_without_round_tripping() {
    let mut env = natural_env();
    let result = check(
        &list(vec![
            leaf("lambda"),
            list(vec![leaf("Natural"), leaf("x")]),
            leaf("x"),
        ]),
        &list(vec![
            leaf("Pi"),
            list(vec![leaf("Natural"), leaf("x")]),
            leaf("Natural"),
        ]),
        &mut env,
    );
    assert!(result.ok);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn check_type_mismatch_emits_e021() {
    let mut env = natural_env();
    let result = check(&leaf("zero"), &leaf("Boolean"), &mut env);
    assert!(!result.ok);
    assert_eq!(result.diagnostics.len(), 1);
    assert_eq!(result.diagnostics[0].code, "E021");
    assert!(result.diagnostics[0].message.contains("Type mismatch"));
}

#[test]
fn check_lambda_against_non_pi_emits_e023() {
    let mut env = natural_env();
    let result = check(
        &list(vec![
            leaf("lambda"),
            list(vec![leaf("Natural"), leaf("x")]),
            leaf("x"),
        ]),
        &leaf("Natural"),
        &mut env,
    );
    assert!(!result.ok);
    assert_eq!(result.diagnostics[0].code, "E023");
    assert!(result.diagnostics[0]
        .message
        .contains("cannot check against non-Pi type"));
}

#[test]
fn check_lambda_param_type_mismatch_emits_e021() {
    let mut env = natural_env();
    let result = check(
        &list(vec![
            leaf("lambda"),
            list(vec![leaf("Boolean"), leaf("x")]),
            leaf("x"),
        ]),
        &list(vec![
            leaf("Pi"),
            list(vec![leaf("Natural"), leaf("x")]),
            leaf("Natural"),
        ]),
        &mut env,
    );
    assert!(!result.ok);
    assert_eq!(result.diagnostics[0].code, "E021");
    assert!(result.diagnostics[0]
        .message
        .contains("does not match Pi domain"));
}

#[test]
fn check_numeric_literals_against_any_annotation() {
    let mut env = natural_env();
    let result = check(&leaf("0.7"), &leaf("Natural"), &mut env);
    assert!(result.ok);
}

#[test]
fn check_lambda_body_in_extended_context() {
    let mut env = natural_env();
    let result = check(
        &list(vec![
            leaf("lambda"),
            list(vec![leaf("Natural"), leaf("x")]),
            list(vec![leaf("apply"), leaf("succ"), leaf("x")]),
        ]),
        &list(vec![
            leaf("Pi"),
            list(vec![leaf("Natural"), leaf("n")]),
            leaf("Natural"),
        ]),
        &mut env,
    );
    assert!(result.ok);
    assert!(result.diagnostics.is_empty());
}

#[test]
fn checks_each_declaration_in_dependent_types_example() {
    let path = examples_dir().join("dependent-types.lino");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
    let out = evaluate(&text, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics from canonical example: {:?}",
        out.diagnostics,
    );

    let mut env = Env::new(None);
    eval_node(
        &list(vec![leaf("Type:"), leaf("Type"), leaf("Type")]),
        &mut env,
    );
    eval_node(
        &list(vec![leaf("Natural:"), leaf("Type"), leaf("Natural")]),
        &mut env,
    );
    eval_node(
        &list(vec![leaf("Boolean:"), leaf("Type"), leaf("Boolean")]),
        &mut env,
    );
    eval_node(
        &list(vec![leaf("zero:"), leaf("Natural"), leaf("zero")]),
        &mut env,
    );
    eval_node(
        &list(vec![leaf("true-val:"), leaf("Boolean"), leaf("true-val")]),
        &mut env,
    );
    eval_node(
        &list(vec![
            leaf("false-val:"),
            leaf("Boolean"),
            leaf("false-val"),
        ]),
        &mut env,
    );
    eval_node(
        &list(vec![
            leaf("succ:"),
            list(vec![
                leaf("Pi"),
                list(vec![leaf("Natural"), leaf("n")]),
                leaf("Natural"),
            ]),
        ]),
        &mut env,
    );
    eval_node(
        &list(vec![
            leaf("identity:"),
            leaf("lambda"),
            list(vec![leaf("Natural"), leaf("x")]),
            leaf("x"),
        ]),
        &mut env,
    );

    assert!(check(&leaf("zero"), &leaf("Natural"), &mut env).ok);
    assert!(check(&leaf("Natural"), &leaf("Type"), &mut env).ok);
    assert!(check(&leaf("Boolean"), &leaf("Type"), &mut env).ok);
    assert!(check(&leaf("true-val"), &leaf("Boolean"), &mut env).ok);
    assert!(check(&leaf("false-val"), &leaf("Boolean"), &mut env).ok);
    assert!(check(
        &leaf("succ"),
        &list(vec![
            leaf("Pi"),
            list(vec![leaf("Natural"), leaf("n")]),
            leaf("Natural")
        ]),
        &mut env,
    )
    .ok);
    assert!(check(
        &leaf("identity"),
        &list(vec![
            leaf("Pi"),
            list(vec![leaf("Natural"), leaf("x")]),
            leaf("Natural"),
        ]),
        &mut env,
    )
    .ok);
    let zero_type = synth(&leaf("zero"), &mut env);
    assert_eq!(key_of(&zero_type.typ.unwrap()), "Natural");
    let type0 = synth(&list(vec![leaf("Type"), leaf("0")]), &mut env);
    assert_eq!(key_of(&type0.typ.unwrap()), "(Type 1)");
}
