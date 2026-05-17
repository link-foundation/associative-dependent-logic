// Dependency-graph traversal tests (issue #97, Phase 7).
//
// Parallel to `js/tests/dependency-graph.test.mjs`. The root-construct
// registry records each construct's direct dependencies via `depends_on`.
// The dependency-graph helpers expose the transitive closure
// deterministically — for the global graph (sorted construct list, each
// with its sorted transitive deps) and for a single construct
// (`Env::dependency_closure(name)`).
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97

use rml::{build_dependency_graph, evaluate_with_env, format_foundation_report, Env};

fn graph_lookup<'a>(
    graph: &'a [(String, Vec<String>)],
    name: &str,
) -> Option<&'a Vec<String>> {
    graph
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, deps)| deps)
}

#[test]
fn exposes_dependency_graph_field_on_foundation_report() {
    let env = Env::new(None);
    let report = env.foundation_report();
    assert!(
        !report.dependency_graph.is_empty(),
        "expected dependency_graph to be populated"
    );
}

#[test]
fn includes_every_seeded_root_construct_as_a_key() {
    let env = Env::new(None);
    let graph = env.foundation_report().dependency_graph;
    for name in &["+", "-", "=", "!=", "and", "lambda"] {
        assert!(
            graph_lookup(&graph, name).is_some(),
            "{} missing from dependency graph",
            name
        );
    }
}

#[test]
fn returns_sorted_deduplicated_transitive_deps_for_each_construct() {
    let env = Env::new(None);
    let graph = env.foundation_report().dependency_graph;

    let plus = graph_lookup(&graph, "+").expect("+ present");
    assert_eq!(plus, &vec!["decimal-12-arithmetic".to_string()]);

    let ineq = graph_lookup(&graph, "!=").expect("!= present");
    assert!(ineq.contains(&"=".to_string()));
    assert!(ineq.contains(&"not".to_string()));
    assert!(ineq.contains(&"decimal-12-arithmetic".to_string()));
    assert!(ineq.contains(&"structural-equality".to_string()));
    assert!(ineq.contains(&"truth-range".to_string()));

    let mut sorted = ineq.clone();
    sorted.sort();
    assert_eq!(&sorted, ineq);
}

#[test]
fn returns_empty_for_leaf_constructs_with_no_dependencies() {
    let env = Env::new(None);
    let graph = env.foundation_report().dependency_graph;
    assert_eq!(
        graph_lookup(&graph, "decimal-12-arithmetic"),
        Some(&Vec::<String>::new())
    );
    assert_eq!(
        graph_lookup(&graph, "structural-equality"),
        Some(&Vec::<String>::new())
    );
}

#[test]
fn dependency_closure_returns_same_closure_as_graph() {
    let env = Env::new(None);
    let graph = env.foundation_report().dependency_graph;
    for (name, deps) in &graph {
        let closure = env
            .dependency_closure(name)
            .expect("registered construct has a closure");
        assert_eq!(&closure, deps, "mismatch for {}", name);
    }
}

#[test]
fn dependency_closure_returns_none_for_unknown_construct() {
    let env = Env::new(None);
    assert!(env.dependency_closure("no-such-construct").is_none());
}

#[test]
fn dependency_closure_does_not_include_the_construct_itself() {
    let env = Env::new(None);
    let closure = env.dependency_closure("!=").expect("registered");
    assert!(
        !closure.contains(&"!=".to_string()),
        "closure should not include the root"
    );
}

#[test]
fn tolerates_dangling_deps_keeps_unknown_names_visible() {
    let mut env = Env::new(None);
    let src = r#"
(root-construct my-op
  (status host-primitive)
  (kind arithmetic-operator)
  (depends-on ghost-construct decimal-12-arithmetic))
"#;
    evaluate_with_env(src, None, &mut env);
    let closure = env.dependency_closure("my-op").expect("registered");
    // The closure preserves the dangling name `ghost-construct` so
    // downstream tools can detect it has no entry in the graph itself.
    assert_eq!(
        closure,
        vec![
            "decimal-12-arithmetic".to_string(),
            "ghost-construct".to_string()
        ]
    );
}

#[test]
fn format_foundation_report_renders_the_dependency_graph_section() {
    let env = Env::new(None);
    let printed = format_foundation_report(&env.foundation_report());
    assert!(printed.contains("dependency graph (transitive):"));
    assert!(printed.contains("+ → decimal-12-arithmetic"));
}

#[test]
fn only_renders_entries_with_at_least_one_dep() {
    let env = Env::new(None);
    let printed = format_foundation_report(&env.foundation_report());
    // `decimal-12-arithmetic` is a leaf; it should not appear with an
    // arrow on its own line in the dependency-graph section.
    let dep_section_start = printed
        .find("dependency graph (transitive):")
        .expect("section present");
    let dep_section = &printed[dep_section_start..];
    assert!(
        !dep_section.contains("decimal-12-arithmetic →"),
        "leaf construct should not render with an arrow"
    );
}

#[test]
fn build_dependency_graph_helper_is_exported_and_idempotent() {
    let env = Env::new(None);
    let a = build_dependency_graph(&env);
    let b = build_dependency_graph(&env);
    assert_eq!(a, b);
}

#[test]
fn user_registered_constructs_surface_in_the_graph_immediately() {
    let mut env = Env::new(None);
    let src = r#"
(root-construct fancy-op
  (status user-overridden)
  (kind custom-operator)
  (depends-on + -))
"#;
    evaluate_with_env(src, None, &mut env);
    let graph = env.foundation_report().dependency_graph;
    let closure = graph_lookup(&graph, "fancy-op").expect("fancy-op registered");
    assert!(closure.contains(&"+".to_string()));
    assert!(closure.contains(&"-".to_string()));
    assert!(closure.contains(&"decimal-12-arithmetic".to_string()));
}

#[test]
fn handles_cycles_without_infinite_looping() {
    let mut env = Env::new(None);
    let src = r#"
(root-construct cyc-a
  (status host-primitive)
  (depends-on cyc-b))
(root-construct cyc-b
  (status host-primitive)
  (depends-on cyc-a))
"#;
    evaluate_with_env(src, None, &mut env);
    let closure_a = env.dependency_closure("cyc-a").expect("registered");
    let closure_b = env.dependency_closure("cyc-b").expect("registered");
    assert_eq!(closure_a, vec!["cyc-b".to_string()]);
    assert_eq!(closure_b, vec!["cyc-a".to_string()]);
}
