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
    decode_anum, encode_anum, evaluate, evaluate_with_env, format_foundation_report, parse_one,
    tokenize_one, Env, Node,
};

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
