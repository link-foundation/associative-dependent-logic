//! Universal CST converter round-trip demo (issue #138).
//!
//! Rust counterpart of `examples/cst-roundtrip-demo.mjs`. Run with:
//!
//!     cargo run --example cst_roundtrip_demo --manifest-path rust/Cargo.toml
//!
//! On success it prints one OK line per language plus a peek at the underlying
//! `.lino` S-expression. Exits non-zero on any round-trip mismatch.

use rml::cst::{cst_to_lino, CstNode};
use rml::cst_convert::{parse_to_cst, print_from_cst, round_trip, SUPPORTED_LANGUAGES};

fn main() {
    let samples: &[(&str, &str)] = &[
        (
            "rust",
            "// add two i64s\npub fn add(a: i64, b: i64) -> i64 {\n    a + b\n}\n",
        ),
        (
            "js",
            "// fetch JSON\nasync function load(url) {\n  const r = await fetch(url);\n  return r.json();\n}\n",
        ),
        ("lean", "-- identity\ndef id {α : Type} (x : α) : α := x\n"),
        (
            "rocq",
            "(* identity *)\nDefinition id {A : Type} (x : A) : A := x.\n",
        ),
    ];

    println!("Universal CST round-trip demo");
    println!("-----------------------------");
    println!("Supported languages: {}", SUPPORTED_LANGUAGES.join(", "));
    println!();

    let mut all_ok = true;
    for (lang, src) in samples {
        let r = round_trip(src, lang).expect("known language");
        let tree = parse_to_cst(src, lang).expect("known language");
        let leaf_count = tree.leaves().len();
        let sexp = cst_to_lino(&tree);
        let preview: String = sexp.chars().take(60).collect();
        println!(
            "{:<5} {}  ({} bytes, {} leaves)",
            lang,
            if r.ok { "OK" } else { "FAIL" },
            src.len(),
            leaf_count,
        );
        println!("        sexp: {}…", preview);
        if !r.ok {
            all_ok = false;
            eprintln!("  expected: {:?}", r.source);
            eprintln!("  got     : {:?}", r.round_tripped);
        }
        let trivia: Vec<String> = tree
            .leaves()
            .into_iter()
            .filter_map(|n| match n {
                CstNode::Trivia { text, .. } => {
                    let t = text.trim();
                    if t.is_empty() {
                        None
                    } else {
                        Some(t.to_string())
                    }
                }
                _ => None,
            })
            .collect();
        println!("        trivia: {:?}", trivia);
        println!();
    }

    // Idempotency sanity check.
    for (lang, src) in samples {
        let tree = parse_to_cst(src, lang).unwrap();
        let a = print_from_cst(&tree, lang).unwrap();
        let b = print_from_cst(&tree, lang).unwrap();
        if a != b {
            eprintln!("print_from_cst not idempotent for {}", lang);
            all_ok = false;
        }
    }

    if all_ok {
        println!("All round-trip checks passed.");
    } else {
        eprintln!("One or more round-trip checks failed.");
        std::process::exit(1);
    }
}
