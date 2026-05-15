// HOAS desugaring (issue #51).
//
// Mirrors `js/tests/hoas.test.mjs`. `forall` is surface sugar for `Pi`. The
// rewrite happens at the AST level so every downstream pass (typing, beta,
// definitional equality, proofs) only ever sees `Pi`.

use rml::{desugar_hoas, evaluate, key_of, synth, Env, Node, RunResult};

fn evaluate_clean(src: &str) -> Vec<RunResult> {
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    out.results
}

fn leaf(s: &str) -> Node {
    Node::Leaf(s.to_string())
}

fn list(children: Vec<Node>) -> Node {
    Node::List(children)
}

#[test]
fn classifies_lambda_under_both_pi_and_forall() {
    let results = evaluate_clean(
        r#"
(Term: (Type 0) Term)
(identity: lambda (Term x) x)
(? (identity of (Pi     (Term x) Term)))
(? (identity of (forall (Term x) Term)))
"#,
    );
    assert_eq!(
        results,
        vec![RunResult::Num(1.0), RunResult::Num(1.0)],
    );
}

#[test]
fn forall_typed_declaration_records_as_pi() {
    let results = evaluate_clean(
        r#"
(Natural: (Type 0) Natural)
(succ: (forall (Natural n) Natural))
(? (type of succ))
"#,
    );
    assert_eq!(
        results,
        vec![RunResult::Type("(Pi (Natural n) Natural)".to_string())],
    );
}

#[test]
fn synth_treats_forall_as_pi() {
    let mut env = Env::new(None);
    let _ = evaluate("(Term: (Type 0) Term)", None, None);
    rml::eval_node(
        &list(vec![
            leaf("Term:"),
            list(vec![leaf("Type"), leaf("0")]),
            leaf("Term"),
        ]),
        &mut env,
    );

    let pi_term = list(vec![
        leaf("Pi"),
        list(vec![leaf("Term"), leaf("x")]),
        leaf("Term"),
    ]);
    let forall_term = list(vec![
        leaf("forall"),
        list(vec![leaf("Term"), leaf("x")]),
        leaf("Term"),
    ]);

    let pi_result = synth(&pi_term, &mut env);
    let forall_result = synth(&desugar_hoas(forall_term), &mut env);

    assert!(pi_result.diagnostics.is_empty());
    assert!(forall_result.diagnostics.is_empty());
    assert_eq!(
        key_of(&pi_result.typ.unwrap()),
        key_of(&forall_result.typ.unwrap()),
    );
}

#[test]
fn desugar_hoas_rewrites_forall_head() {
    let input = list(vec![
        leaf("forall"),
        list(vec![leaf("Term"), leaf("x")]),
        leaf("Term"),
    ]);
    let expected = list(vec![
        leaf("Pi"),
        list(vec![leaf("Term"), leaf("x")]),
        leaf("Term"),
    ]);
    assert_eq!(desugar_hoas(input), expected);
}

#[test]
fn desugar_hoas_recurses_into_subterms() {
    // forall nested under a lambda must also be rewritten.
    let input = list(vec![
        leaf("lambda"),
        list(vec![leaf("Term"), leaf("x")]),
        list(vec![
            leaf("forall"),
            list(vec![leaf("Term"), leaf("y")]),
            leaf("Term"),
        ]),
    ]);
    let expected = list(vec![
        leaf("lambda"),
        list(vec![leaf("Term"), leaf("x")]),
        list(vec![
            leaf("Pi"),
            list(vec![leaf("Term"), leaf("y")]),
            leaf("Term"),
        ]),
    ]);
    assert_eq!(desugar_hoas(input), expected);
}

#[test]
fn desugar_hoas_does_not_rewrite_non_head_forall() {
    // 2-element list: not the (forall <binder> <body>) shape, so untouched.
    let input = list(vec![leaf("forall"), leaf("x")]);
    let expected = list(vec![leaf("forall"), leaf("x")]);
    assert_eq!(desugar_hoas(input), expected);
}
