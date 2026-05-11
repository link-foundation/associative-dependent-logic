//! Universal CST converter integration tests (issue #138).
//!
//! Mirrors `js/tests/cst.test.mjs`: verifies the lossless round-trip contract
//! `print(parse(src)) == src` for every host-language converter plus the CST
//! serialisation helpers and the dispatch entry point.

use std::path::PathBuf;

use rml::cst::{
    clone_cst, cst_to_lino, dialects, lino_to_cst, print_cst, CstNode,
};
use rml::cst_convert::{round_trip, SUPPORTED_LANGUAGES};
use rml::cst_js::{parse_js, print_js};
use rml::cst_lean::{parse_lean, print_lean};
use rml::cst_rocq::{parse_rocq, print_rocq};
use rml::cst_rust::{parse_rust, print_rust};

fn repo_root() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent().unwrap().to_path_buf()
}

// -------------------- CST data model --------------------

#[test]
fn print_cst_concatenates_leaves_in_document_order() {
    let tree = CstNode::list(
        "demo",
        vec![
            CstNode::token("hello", None),
            CstNode::trivia(" ", None),
            CstNode::token("world", None),
        ],
    );
    assert_eq!(print_cst(&tree), "hello world");
}

#[test]
fn list_nodes_emit_open_close_delimiters_when_set() {
    let tree = CstNode::list_with_delims(
        Some("demo".into()),
        Some("(".into()),
        Some(")".into()),
        vec![CstNode::token("a", None), CstNode::token("b", None)],
    );
    assert_eq!(print_cst(&tree), "(ab)");
}

#[test]
fn cst_to_lino_and_back_round_trips() {
    let tree = CstNode::list(
        "lino-cst.rust.fn",
        vec![
            CstNode::token("fn ", Some("kw")),
            CstNode::token("foo", Some("ident")),
            CstNode::token("(", Some("punct")),
            CstNode::token(")", Some("punct")),
            CstNode::trivia(" ", Some("ws")),
            CstNode::list(
                "lino-cst.rust.block",
                vec![CstNode::token("{", None), CstNode::token("}", None)],
            ),
        ],
    );
    let serialised = cst_to_lino(&tree);
    assert!(
        serialised.starts_with("(lino-cst.list lino-cst.rust.fn"),
        "unexpected serialisation: {}",
        serialised
    );
    let parsed = lino_to_cst(&serialised).unwrap();
    assert_eq!(print_cst(&parsed), print_cst(&tree));
}

#[test]
fn lino_round_trip_preserves_open_close_delimiters() {
    let tree = CstNode::list_with_delims(
        Some("frag".into()),
        Some("<".into()),
        Some(">".into()),
        vec![CstNode::token("x", None)],
    );
    let sexp = cst_to_lino(&tree);
    let back = lino_to_cst(&sexp).unwrap();
    assert_eq!(print_cst(&back), "<x>");
}

#[test]
fn clone_cst_returns_structural_copy() {
    let tree = CstNode::list(
        "demo",
        vec![
            CstNode::token("a", None),
            CstNode::list("inner", vec![CstNode::token("b", None)]),
        ],
    );
    let clone = clone_cst(&tree);
    assert_eq!(clone, tree);
}

#[test]
fn leaves_walks_in_document_order() {
    let tree = CstNode::list(
        "demo",
        vec![
            CstNode::token("a", None),
            CstNode::list(
                "inner",
                vec![CstNode::token("b", None), CstNode::token("c", None)],
            ),
            CstNode::token("d", None),
        ],
    );
    let texts: Vec<&str> = tree.leaves().iter().map(|n| n.text()).collect();
    assert_eq!(texts, vec!["a", "b", "c", "d"]);
}

#[test]
fn exposes_four_host_dialects() {
    assert_eq!(dialects::RUST, "lino-cst.rust");
    assert_eq!(dialects::JS, "lino-cst.js");
    assert_eq!(dialects::LEAN, "lino-cst.lean");
    assert_eq!(dialects::ROCQ, "lino-cst.rocq");
}

// -------------------- Rust round-trip --------------------

#[test]
fn rust_round_trip_samples() {
    let samples = [
        "",
        "fn main() {}\n",
        "// hello\nfn f() -> i32 { 42 }\n",
        "/* multi\n  line */\nfn g() {}\n",
        "fn id<T>(x: T) -> T { x }\n",
        "let s = \"hello\\n\";\n",
        "let s = r#\"raw\"#;\n",
        "let c = 'a';\nlet lt: &'static str = \"x\";\n",
        "let n = 0xFF_FF;\nlet m = 0b1010;\nlet f = 3.14e10;\n",
        "pub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n",
        "use std::collections::HashMap;\n",
        "let mut v = Vec::<i32>::new();\nv.push(1);\n",
        "macro_rules! foo { () => {}; }\n",
        "let r#match = 1;\n",
    ];
    for src in samples {
        let r = round_trip(src, "rust").unwrap();
        assert!(r.ok, "rust round-trip failed for {:?}", src);
        assert_eq!(r.round_tripped, src);
    }
}

// -------------------- JavaScript round-trip --------------------

#[test]
fn js_round_trip_samples() {
    let samples = [
        "",
        "const x = 1;\n",
        "// hello\nconst y = 2;\n",
        "/* block */ const z = 3;\n",
        "const s = 'a\\'b';\n",
        "const t = `hello ${name}!`;\n",
        "const t2 = `nested ${`inner ${1+2}`} done`;\n",
        "const r = /foo\\/bar/g;\n",
        "const n = 0xff_ff;\nconst b = 0b1010n;\nconst f = 3.14e-2;\n",
        "function f(a, b) {\n  return a + b;\n}\n",
        "class C { method() { return 1; } }\n",
        "const obj = { a: 1, \"b c\": 2 };\n",
        "#!/usr/bin/env node\nconsole.log(\"hi\");\n",
        "export default async function f() { await sleep(1); }\n",
        "let { a, b: c = 3 } = x;\n",
        "const re = /^a[/]b$/;\n",
    ];
    for src in samples {
        let r = round_trip(src, "js").unwrap();
        assert!(r.ok, "js round-trip failed for {:?}", src);
        assert_eq!(r.round_tripped, src);
    }
}

// -------------------- Lean 4 round-trip --------------------

#[test]
fn lean_round_trip_samples() {
    let samples = [
        "",
        "-- comment\ndef f : Nat := 1\n",
        "/- block comment -/\ndef g : Nat := 2\n",
        "/-! module doc -/\n",
        "/-- decl doc -/\ndef h : Nat := 3\n",
        "def id {α : Type} (x : α) : α := x\n",
        "#check Nat.succ\n",
        "theorem t : 1 + 1 = 2 := rfl\n",
        "def s : String := \"hello\"\n",
        "def n : Nat := 0xff\n",
        "def m : Nat := 0b1010\n",
        "inductive List (α : Type u) where\n  | nil : List α\n  | cons : α → List α → List α\n",
        "namespace Foo\ndef x := 1\nend Foo\n",
        "def c : Char := 'a'\n",
    ];
    for src in samples {
        let r = round_trip(src, "lean").unwrap();
        assert!(r.ok, "lean round-trip failed for {:?}", src);
        assert_eq!(r.round_tripped, src);
    }
}

// -------------------- Rocq round-trip --------------------

#[test]
fn rocq_round_trip_samples() {
    let samples = [
        "",
        "(* comment *)\nDefinition x := 1.\n",
        "(* nested (* inside *) *)\nDefinition y := 2.\n",
        "Definition id {A : Type} (x : A) : A := x.\n",
        "Theorem t : 1 + 1 = 2. Proof. reflexivity. Qed.\n",
        "Inductive list (A : Type) : Type :=\n  | nil\n  | cons (x : A) (xs : list A).\n",
        "Definition s := \"hello, \"\"world\"\"\".\n",
        "Definition n := 0xff.\n",
        "Require Import Coq.Lists.List.\n",
    ];
    for src in samples {
        let r = round_trip(src, "rocq").unwrap();
        assert!(r.ok, "rocq round-trip failed for {:?}", src);
        assert_eq!(r.round_tripped, src);
    }
}

// -------------------- Cross-converter dispatch --------------------

#[test]
fn supported_languages_lists_four_hosts_plus_alias() {
    let mut langs: Vec<&str> = SUPPORTED_LANGUAGES.to_vec();
    langs.sort();
    assert_eq!(langs, vec!["javascript", "js", "lean", "rocq", "rust"]);
}

// -------------------- Comment / whitespace preservation --------------------

#[test]
fn rust_comments_survive_as_trivia() {
    let src = "// leading\nfn f() {\n  /* mid */ 1\n}\n";
    let tree = parse_rust(src);
    let leaves = tree.leaves();
    assert!(leaves.iter().any(|n| matches!(n, CstNode::Trivia { text, .. } if text == "// leading")));
    assert!(leaves.iter().any(|n| matches!(n, CstNode::Trivia { text, .. } if text == "/* mid */")));
    assert_eq!(print_rust(&tree), src);
}

#[test]
fn js_hashbang_and_comments_survive() {
    let src = "#!/usr/bin/env node\n// hi\nlet x = 1;\n";
    let tree = parse_js(src);
    let leaves = tree.leaves();
    assert!(leaves
        .iter()
        .any(|n| matches!(n, CstNode::Trivia { text, .. } if text.starts_with("#!"))));
    assert!(leaves
        .iter()
        .any(|n| matches!(n, CstNode::Trivia { text, .. } if text == "// hi")));
    assert_eq!(print_js(&tree), src);
}

#[test]
fn lean_nested_block_comments_survive() {
    let src = "/- outer /- inner -/ still outer -/\ndef x := 1\n";
    let tree = parse_lean(src);
    let leaves = tree.leaves();
    assert!(leaves
        .iter()
        .any(|n| matches!(n, CstNode::Trivia { text, .. } if text.starts_with("/- outer"))));
    assert_eq!(print_lean(&tree), src);
}

#[test]
fn rocq_nested_comments_survive() {
    let src = "(* a (* b *) c *)\nDefinition x := 1.\n";
    let tree = parse_rocq(src);
    let leaves = tree.leaves();
    assert!(leaves
        .iter()
        .any(|n| matches!(n, CstNode::Trivia { text, .. } if text == "(* a (* b *) c *)")));
    assert_eq!(print_rocq(&tree), src);
}

// -------------------- Repository corpus round-trip --------------------

fn read_if_exists(path: &PathBuf) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

#[test]
fn round_trips_js_src_cst_mjs() {
    let p = repo_root().join("js").join("src").join("cst.mjs");
    if let Some(src) = read_if_exists(&p) {
        assert_eq!(print_js(&parse_js(&src)), src);
    }
}

#[test]
fn round_trips_rust_src_main_rs() {
    let p = repo_root().join("rust").join("src").join("main.rs");
    if let Some(src) = read_if_exists(&p) {
        assert_eq!(print_rust(&parse_rust(&src)), src);
    }
}

#[test]
fn round_trips_lean_export_basic() {
    let p = repo_root()
        .join("examples")
        .join("lean-export-basic.lean");
    if let Some(src) = read_if_exists(&p) {
        assert_eq!(print_lean(&parse_lean(&src)), src);
    }
}

#[test]
fn round_trips_isabelle_typed_fragment_as_rocq_style() {
    let p = repo_root()
        .join("examples")
        .join("isabelle-typed-fragment.thy");
    if let Some(src) = read_if_exists(&p) {
        assert_eq!(print_rocq(&parse_rocq(&src)), src);
    }
}
