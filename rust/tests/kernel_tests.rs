// Typed kernel rules for issues #37, #38, and #41.
//
// These tests keep the documented D1 surface honest: Pi formation, lambda
// formation, application by beta-reduction, capture-avoiding substitution,
// freshness, and type membership/query links.

use rml::{
    eval_node, evaluate, is_convertible, is_convertible_with_options, ConvertOptions, Env, Node,
    RunResult,
};

fn evaluate_clean(src: &str) -> Vec<RunResult> {
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        out.diagnostics
    );
    out.results
}

#[test]
fn forms_pi_type_and_records_type_zero_membership() {
    let results = evaluate_clean(
        r#"
(Natural: (Type 0) Natural)
(succ: (Pi (Natural n) Natural))
(? (Pi (Natural n) Natural))
(? ((Pi (Natural n) Natural) of (Type 0)))
(? (succ of (Pi (Natural n) Natural)))
(? (type of succ))
"#,
    );
    assert_eq!(
        results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Type("(Pi (Natural n) Natural)".to_string()),
        ]
    );
}

#[test]
fn types_named_lambda_under_bound_parameter_context() {
    let results = evaluate_clean(
        r#"
(Natural: (Type 0) Natural)
(identity: lambda (Natural x) x)
(? (identity of (Pi (Natural x) Natural)))
(? (type of identity))
"#,
    );
    assert_eq!(
        results,
        vec![
            RunResult::Num(1.0),
            RunResult::Type("(Pi (Natural x) Natural)".to_string()),
        ]
    );
}

#[test]
fn keeps_named_lambda_parameter_scoped_to_lambda_body() {
    let results = evaluate_clean(
        r#"
(Natural: (Type 0) Natural)
(identity: lambda (Natural x) x)
(? (x of Natural))
"#,
    );
    assert_eq!(results, vec![RunResult::Num(0.0)]);
}

#[test]
fn applies_lambdas_by_beta_reducing_argument_into_body() {
    let results = evaluate_clean(
        r#"
(Natural: (Type 0) Natural)
(zero: Natural zero)
(identity: lambda (Natural x) x)
(? ((apply identity zero) = zero))
(? (apply (lambda (Natural x) (x + 1)) 0))
(? (apply (lambda (Natural x) (x + 0.1)) 0.2))
"#,
    );
    assert_eq!(
        results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(0.3)
        ]
    );
}

#[test]
fn decides_definitional_equality_by_beta_normalizing_terms() {
    let results = evaluate_clean(
        r#"
(? ((apply (lambda (Natural x) x) 0) = 0))
(? ((pair (apply (lambda (Natural x) x) y)) = (pair y)))
(? ((pair x) = (pair y)))
"#,
    );
    assert_eq!(
        results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(0.0)
        ]
    );
}

#[test]
fn uses_explicit_equality_assignments_before_conversion() {
    let results = evaluate_clean(
        r#"
(Natural: (Type 0) Natural)
(zero: Natural zero)
(identity: lambda (Natural x) x)
(((apply identity zero) = zero) has probability 0.5)
(? ((apply identity zero) = zero))
"#,
    );
    assert_eq!(results, vec![RunResult::Num(0.5)]);
}

#[test]
fn exposes_is_convertible_for_beta_assignment_lookup_and_opt_in_eta() {
    let mut env = Env::new(None);
    eval_node(
        &Node::List(vec![
            Node::Leaf("zero:".into()),
            Node::Leaf("Natural".into()),
            Node::Leaf("zero".into()),
        ]),
        &mut env,
    );
    eval_node(
        &Node::List(vec![
            Node::Leaf("identity:".into()),
            Node::Leaf("lambda".into()),
            Node::List(vec![Node::Leaf("Natural".into()), Node::Leaf("x".into())]),
            Node::Leaf("x".into()),
        ]),
        &mut env,
    );
    eval_node(
        &Node::List(vec![
            Node::List(vec![
                Node::Leaf("zero".into()),
                Node::Leaf("=".into()),
                Node::Leaf("alias".into()),
            ]),
            Node::Leaf("has".into()),
            Node::Leaf("probability".into()),
            Node::Leaf("1".into()),
        ]),
        &mut env,
    );

    assert!(is_convertible(
        &Node::List(vec![
            Node::Leaf("apply".into()),
            Node::Leaf("identity".into()),
            Node::Leaf("zero".into()),
        ]),
        &Node::Leaf("zero".into()),
        &mut env,
    ));
    assert!(is_convertible(
        &Node::Leaf("zero".into()),
        &Node::Leaf("alias".into()),
        &mut env,
    ));

    let eta_lhs = Node::List(vec![
        Node::Leaf("lambda".into()),
        Node::List(vec![Node::Leaf("Natural".into()), Node::Leaf("x".into())]),
        Node::List(vec![
            Node::Leaf("apply".into()),
            Node::Leaf("f".into()),
            Node::Leaf("x".into()),
        ]),
    ]);
    let eta_rhs = Node::Leaf("f".into());
    assert!(!is_convertible(&eta_lhs, &eta_rhs, &mut env));
    assert!(is_convertible_with_options(
        &eta_lhs,
        &eta_rhs,
        &mut env,
        ConvertOptions { eta: true },
    ));
}

#[test]
fn beta_reduces_open_terms_without_evaluating_free_variables_as_probabilities() {
    let results = evaluate_clean(
        r#"
(? (apply (lambda (Natural x) (x + y)) z))
"#,
    );
    assert_eq!(results, vec![RunResult::Type("(z + y)".to_string())]);
}

#[test]
fn beta_reduction_is_capture_avoiding_for_open_replacements() {
    let results = evaluate_clean(
        r#"
(? (apply (lambda (Natural x) (lambda (Natural y) (x + y))) y))
"#,
    );
    assert_eq!(
        results,
        vec![RunResult::Type(
            "(lambda (Natural y_1) (y + y_1))".to_string()
        )]
    );
}

#[test]
fn exposes_substitution_as_capture_avoiding_kernel_primitive() {
    let results = evaluate_clean(
        r#"
(? (subst (lambda (Natural y) (x + y)) x y))
(? ((subst (lambda (Natural y) (x + y)) x y) = (lambda (Natural y_1) (y + y_1))))
(? ((subst (x + 0.1) x 0.2) = 0.3))
"#,
    );
    assert_eq!(
        results,
        vec![
            RunResult::Type("(lambda (Natural y_1) (y + y_1))".to_string()),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn scopes_fresh_variables_and_rejects_names_already_in_context() {
    let ok = evaluate(
        r#"
(? (fresh y in ((lambda (Natural x) (x + y)) y)))
(? (y of Natural))
"#,
        None,
        None,
    );
    assert!(
        ok.diagnostics.is_empty(),
        "unexpected diagnostics: {:?}",
        ok.diagnostics
    );
    assert_eq!(ok.results, vec![RunResult::Num(1.0), RunResult::Num(0.0)]);

    let bad = evaluate(
        r#"
(Natural: (Type 0) Natural)
(y: Natural y)
(? (fresh y in y))
"#,
        None,
        None,
    );
    assert!(bad.results.is_empty());
    assert_eq!(bad.diagnostics.len(), 1);
    assert_eq!(bad.diagnostics[0].code, "E010");
    assert!(
        bad.diagnostics[0].message.contains("fresh variable \"y\""),
        "message: {}",
        bad.diagnostics[0].message
    );
}

#[test]
fn checks_type_membership_and_returns_stored_types_through_of_links() {
    let results = evaluate_clean(
        r#"
(Type: Type Type)
(Natural: Type Natural)
(zero: Natural zero)
(Type 0)
(Type 1)
(? (zero of Natural))
(? (Natural of Type))
(? (type of zero))
(? ((Type 0) of (Type 1)))
"#,
    );
    assert_eq!(
        results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Type("Natural".to_string()),
            RunResult::Num(1.0),
        ]
    );
}

#[test]
fn checks_universe_hierarchy_directly_across_adjacent_levels() {
    let results = evaluate_clean(
        r#"
(? ((Type 0) of (Type 1)))
(? ((Type 1) of (Type 2)))
(? ((Type 2) of (Type 3)))
(? ((Type 1) of (Type 0)))
(? ((Type 2) of (Type 1)))
(? ((Type 0) of (Type 2)))
(? (type of (Type 0)))
(? (type of (Type 2)))
"#,
    );
    assert_eq!(
        results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(0.0),
            RunResult::Num(0.0),
            RunResult::Num(0.0),
            RunResult::Type("(Type 1)".to_string()),
            RunResult::Type("(Type 3)".to_string()),
        ]
    );
}

#[test]
fn keeps_self_referential_type_separate_from_stratified_universes() {
    let results = evaluate_clean(
        r#"
(Type: Type Type)
(Natural: (Type 0) Natural)
(Boolean: Type Boolean)
(? (Type of Type))
(? (Natural of (Type 0)))
(? (Boolean of Type))
(? ((Type 0) of (Type 1)))
(? ((Type 1) of (Type 0)))
"#,
    );
    assert_eq!(
        results,
        vec![
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(1.0),
            RunResult::Num(0.0),
        ]
    );
}
