// MTC/anum experimental foundation profile tests (issue #97, Phase 9).
//
// Parallel to `js/tests/mtc-anum.test.mjs`. The `mtc-anum` foundation is
// pre-seeded but opt-in. Activating it does not rewire host arithmetic —
// it is metadata plus a four-abit serialization alphabet (`[`, `]`, `0`,
// `1`). The trust audit surfaces the profile with its `[experimental]`
// tag, root symbol, and abit list. `encode_anum` / `decode_anum`
// round-trip arbitrary Node values through strings written only in that
// alphabet.
//
// See: https://github.com/link-foundation/relative-meta-logic/issues/97
use rml::{
    decode_anum, encode_anum, evaluate, evaluate_file, evaluate_with_env,
    format_foundation_report, parse_one, tokenize_one, Env, EvaluateOptions, Node, RunResult,
};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn nums(results: &[RunResult]) -> Vec<f64> {
    results
        .iter()
        .filter_map(|r| match r {
            RunResult::Num(v) => Some(*v),
            _ => None,
        })
        .collect()
}

#[test]
fn mtc_anum_is_preseeded_but_never_activated_implicitly() {
    let env = Env::new(None);
    assert!(
        env.foundations.contains_key("mtc-anum"),
        "mtc-anum should be pre-seeded"
    );
    let report = env.foundation_report();
    assert_eq!(report.active_foundation, "default-rml");
}

#[test]
fn does_not_perturb_baseline_semantics_when_not_selected() {
    let out = evaluate("(? (1 + 2))", None, None);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.results.len(), 1);
    let env = Env::new(None);
    let report = env.foundation_report();
    assert_eq!(report.active_foundation, "default-rml");
}

#[test]
fn surfaces_experimental_flag_root_and_abits_on_foundation_report() {
    let env = Env::new(None);
    let report = env.foundation_report();
    let mtc = report
        .foundations
        .iter()
        .find(|f| f.name == "mtc-anum")
        .expect("mtc-anum should appear in foundations");
    assert!(mtc.experimental);
    assert_eq!(mtc.root.as_deref(), Some("∞"));
    let mut symbols: Vec<String> = mtc.abits.iter().map(|(s, _)| s.clone()).collect();
    symbols.sort();
    assert_eq!(
        symbols,
        vec!["0".to_string(), "1".to_string(), "[".to_string(), "]".to_string()]
    );
}

#[test]
fn renders_experimental_tag_and_abits_in_printed_report() {
    let env = Env::new(None);
    let printed = format_foundation_report(&env.foundation_report());
    assert!(
        printed.contains("mtc-anum [experimental]"),
        "missing experimental tag in:\n{}",
        printed
    );
    assert!(
        printed.contains("root: ∞"),
        "missing root symbol in:\n{}",
        printed
    );
    assert!(
        printed.contains("abits:"),
        "missing abits section in:\n{}",
        printed
    );
    assert!(
        printed.contains("[=start-of-meaning"),
        "missing start-of-meaning abit in:\n{}",
        printed
    );
}

#[test]
fn parser_accepts_experimental_root_and_abit_clauses() {
    let mut env = Env::new(None);
    let src = r#"
(foundation toy-mtc
  (description "a toy mtc-style profile")
  (experimental)
  (root ★)
  (abit ▲ up)
  (abit ▼ down))
"#;
    let out = evaluate_with_env(src, None, &mut env);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    let f = env
        .get_foundation("toy-mtc")
        .expect("toy-mtc should be registered");
    assert!(f.experimental);
    assert_eq!(f.root.as_deref(), Some("★"));
    let syms: Vec<String> = f.abits.iter().map(|(s, _)| s.clone()).collect();
    assert_eq!(syms, vec!["▲".to_string(), "▼".to_string()]);
}

#[test]
fn round_trips_leaf_strings_through_anum() {
    for c in ["x", "hello", "+", "∞", ""] {
        let leaf = Node::Leaf(c.to_string());
        let enc = encode_anum(&leaf);
        assert!(
            enc.chars().all(|ch| ch == '[' || ch == ']' || ch == '0' || ch == '1'),
            "encoding of {:?} not in alphabet: {}",
            c,
            enc
        );
        let dec = decode_anum(&enc).expect("decode");
        assert_eq!(dec, leaf);
    }
}

#[test]
fn round_trips_lists_through_anum() {
    let cases: Vec<Node> = vec![
        Node::List(vec![]),
        Node::List(vec![Node::Leaf("a".into()), Node::Leaf("b".into())]),
        Node::List(vec![
            Node::Leaf("+".into()),
            Node::Leaf("1".into()),
            Node::Leaf("2".into()),
        ]),
        Node::List(vec![
            Node::Leaf("lambda".into()),
            Node::List(vec![Node::Leaf("x".into())]),
            Node::List(vec![
                Node::Leaf("+".into()),
                Node::Leaf("x".into()),
                Node::Leaf("1".into()),
            ]),
        ]),
    ];
    for c in cases {
        let enc = encode_anum(&c);
        assert!(enc.chars().all(|ch| ch == '[' || ch == ']' || ch == '0' || ch == '1'));
        let dec = decode_anum(&enc).expect("decode");
        assert_eq!(dec, c);
    }
}

#[test]
fn round_trips_a_real_parsed_link_form() {
    let tokens = tokenize_one("(? (1 + 2))");
    let node = parse_one(&tokens).expect("parse one form");
    let enc = encode_anum(&node);
    assert!(enc.chars().all(|ch| ch == '[' || ch == ']' || ch == '0' || ch == '1'));
    let dec = decode_anum(&enc).expect("decode");
    assert_eq!(dec, node);
}

#[test]
fn decode_anum_rejects_characters_outside_the_alphabet() {
    assert!(decode_anum("[0AB]").is_err());
    assert!(decode_anum("[2]").is_err());
    assert!(decode_anum("xyz").is_err());
}

#[test]
fn decode_anum_rejects_unbalanced_frames() {
    assert!(decode_anum("[0").is_err());
    assert!(decode_anum("[1[0]").is_err());
    assert!(decode_anum("[0]extra").is_err());
}

#[test]
fn decode_anum_rejects_misaligned_leaf_payloads() {
    // 7 bits — not byte-aligned, must error.
    let unaligned = format!("[0{}]", "0101010");
    assert!(decode_anum(&unaligned).is_err());
}

// --- Serialization invariants (issue #97, Phase 9 strengthening) ---
//
// The acceptance criteria from PR review "Blocking issue 7" ask for
// stated invariants and explicit tests for the canonicality and
// injectivity of `encode_anum`/`decode_anum`. The three properties
// below — canonicality, injectivity, totality — together establish
// that the four-abit alphabet is a faithful serialization domain.

fn theory_samples() -> Vec<Node> {
    vec![
        Node::Leaf(String::new()),
        Node::Leaf("x".into()),
        Node::Leaf("∞".into()),
        Node::List(vec![]),
        Node::List(vec![Node::Leaf("a".into()), Node::Leaf("b".into())]),
        Node::List(vec![
            Node::Leaf("lambda".into()),
            Node::List(vec![Node::Leaf("x".into())]),
            Node::List(vec![
                Node::Leaf("+".into()),
                Node::Leaf("x".into()),
                Node::Leaf("1".into()),
            ]),
        ]),
        Node::List(vec![
            Node::Leaf("frame".into()),
            Node::List(vec![
                Node::Leaf("pair".into()),
                Node::Leaf("∞".into()),
                Node::List(vec![Node::Leaf("frame".into()), Node::Leaf("∞".into())]),
            ]),
        ]),
    ]
}

#[test]
fn encode_anum_is_canonical_repeated_encoding_yields_same_string() {
    for x in theory_samples() {
        let a = encode_anum(&x);
        let b = encode_anum(&x);
        assert_eq!(a, b, "canonicality failed for {:?}", x);
    }
}

#[test]
fn encode_anum_is_injective_distinct_inputs_encode_to_distinct_strings() {
    use std::collections::HashMap;
    let samples: Vec<Node> = vec![
        Node::Leaf(String::new()),
        Node::Leaf("a".into()),
        Node::Leaf("b".into()),
        Node::Leaf("ab".into()),
        Node::Leaf("ba".into()),
        Node::Leaf("∞".into()),
        Node::Leaf("[".into()),
        Node::Leaf("]".into()),
        Node::List(vec![]),
        Node::List(vec![Node::Leaf("a".into())]),
        Node::List(vec![Node::Leaf("a".into()), Node::Leaf("b".into())]),
        Node::List(vec![Node::Leaf("b".into()), Node::Leaf("a".into())]),
        Node::List(vec![Node::List(vec![Node::Leaf("a".into())])]),
        Node::List(vec![Node::Leaf("frame".into()), Node::Leaf("∞".into())]),
        Node::List(vec![
            Node::Leaf("pair".into()),
            Node::Leaf("∞".into()),
            Node::Leaf("∞".into()),
        ]),
        Node::List(vec![
            Node::Leaf("pair".into()),
            Node::Leaf("∞".into()),
            Node::List(vec![Node::Leaf("frame".into()), Node::Leaf("∞".into())]),
        ]),
    ];
    let mut seen: HashMap<String, Node> = HashMap::new();
    for x in &samples {
        let enc = encode_anum(x);
        if let Some(other) = seen.get(&enc) {
            panic!(
                "injectivity violated: {:?} and {:?} both encode to {}",
                x, other, enc
            );
        }
        seen.insert(enc, x.clone());
    }
    // Totality: every encoding must decode back to the original input.
    for x in &samples {
        let dec = decode_anum(&encode_anum(x)).expect("decode");
        assert_eq!(&dec, x);
    }
}

// --- Theory replay (issue #97, Phase 9 strengthening) ---
//
// The acceptance criteria also require at least one non-trivial MTC
// theorem/rule replay through the proof substrate. The example
// `examples/mtc-anum-theory.lino` declares three theory rules
// (root-is-link, frame-makes-link, pair-makes-link) and three
// proof-objects that build a composite link from them.

#[test]
fn runs_the_mtc_theory_example_end_to_end_with_no_diagnostics() {
    let path = repo_root().join("examples").join("mtc-anum-theory.lino");
    let out = evaluate_file(path.to_str().unwrap(), EvaluateOptions::default());
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0, 1.0, 1.0]);
}

#[test]
fn frame_makes_link_types_a_single_frame_over_the_root_link() {
    let src = r#"
(axiom root-is-link
  (judgement (∞ is-a link)))

(rule frame-makes-link
  (premise (?x is-a link))
  (conclusion ((frame ?x) is-a link)))

(proof-object framed-root-is-link
  (applies frame-makes-link)
  (premise-by root-is-link)
  (conclusion ((frame ∞) is-a link)))

(check-proof framed-root-is-link)
"#;
    let out = evaluate(src, None, None);
    assert!(
        out.diagnostics.is_empty(),
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(nums(&out.results), vec![1.0]);
}

#[test]
fn rejects_an_mtc_derivation_whose_conclusion_swaps_the_rule_shape() {
    let src = r#"
(axiom root-is-link
  (judgement (∞ is-a link)))

(rule frame-makes-link
  (premise (?x is-a link))
  (conclusion ((frame ?x) is-a link)))

(proof-object misframed
  (applies frame-makes-link)
  (premise-by root-is-link)
  (conclusion ((unframe ∞) is-a link)))

(check-proof misframed)
"#;
    let out = evaluate(src, None, None);
    assert_eq!(nums(&out.results), vec![0.0]);
    assert!(out.diagnostics.iter().any(|d| d.code == "E064"));
}

#[test]
fn makes_the_theory_serialization_boundary_explicit() {
    // The example file pins the theory/serialization distinction down
    // in prose and as a links-defined rule; the four-abit alphabet of
    // the pre-seeded foundation lives in the serialization domain and
    // does not appear in any theory rule.
    let path = repo_root().join("examples").join("mtc-anum-theory.lino");
    let source =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    assert!(
        source.contains("THEORY domain"),
        "example must explicitly mention the theory domain"
    );
    assert!(
        source.contains("SERIALIZATION domain"),
        "example must explicitly mention the serialization domain"
    );
    assert!(
        source.contains("(rule frame-makes-link"),
        "example must declare a theory rule"
    );
    let env = Env::new(None);
    let mtc = env
        .foundations
        .get("mtc-anum")
        .expect("mtc-anum must be pre-seeded");
    let mut symbols: Vec<String> = mtc.abits.iter().map(|(s, _)| s.clone()).collect();
    symbols.sort();
    assert_eq!(
        symbols,
        vec![
            "0".to_string(),
            "1".to_string(),
            "[".to_string(),
            "]".to_string(),
        ]
    );
}
