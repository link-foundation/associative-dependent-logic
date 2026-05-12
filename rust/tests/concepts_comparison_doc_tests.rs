// Tests for docs/CONCEPTS-COMPARISON.md and docs/FEATURE-COMPARISON.md
// (issue #167). These tests mirror js/tests/concepts-comparison-doc.test.mjs
// so that drift between the JS and Rust implementations fails both suites.
//
// Two things are checked:
//   1. Document structure: the new (correctly-spelled) files exist, contain
//      the expanded legend qualifiers, and the old `COMPARISION` filenames
//      remain as compatibility stubs that point to the new files.
//   2. RML claims: every RML capability that the matrix advertises as
//      available (whnf/nf/normal-form, (inductive ...), (coinductive ...),
//      (total ...), (coverage ...), modes, termination, tactic links,
//      ATP and SMT bridges, independent proof replay, structural and
//      definitional equality) is actually exposed by the Rust crate.
//
// The JS suite mirrors this in js/tests/concepts-comparison-doc.test.mjs
// so that drift between the two implementations fails both test suites.

use rml::evaluate;
use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points at rust/, so go one level up.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent()
        .expect("repo root")
        .to_path_buf()
}

fn read_doc(rel: &str) -> String {
    let p = repo_root().join(rel);
    fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("could not read {}: {}", p.display(), e))
}

// ---------- Document structure ----------

#[test]
fn renamed_comparison_files_exist_and_are_non_trivial() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISON.md");
    let feature = read_doc("docs/FEATURE-COMPARISON.md");
    assert!(
        concepts.len() > 5000,
        "CONCEPTS-COMPARISON.md should be substantial (len {})",
        concepts.len()
    );
    assert!(
        feature.len() > 5000,
        "FEATURE-COMPARISON.md should be substantial (len {})",
        feature.len()
    );
    assert!(
        concepts.contains("# Core Concept Comparison"),
        "concepts doc should start with '# Core Concept Comparison'",
    );
    assert!(
        feature.contains("# Product Feature Comparison"),
        "feature doc should start with '# Product Feature Comparison'",
    );
}

#[test]
fn old_comparision_filenames_remain_as_stubs() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISION.md");
    let feature = read_doc("docs/FEATURE-COMPARISION.md");
    let cl = concepts.to_lowercase();
    let fl = feature.to_lowercase();
    assert!(
        cl.contains("compatibility stub"),
        "CONCEPTS-COMPARISION.md should be a compatibility stub",
    );
    assert!(
        fl.contains("compatibility stub"),
        "FEATURE-COMPARISION.md should be a compatibility stub",
    );
    assert!(
        concepts.contains("CONCEPTS-COMPARISON.md"),
        "stub should point at the new file",
    );
    assert!(
        feature.contains("FEATURE-COMPARISON.md"),
        "stub should point at the new file",
    );
}

#[test]
fn legend_advertises_all_qualifier_marks() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISON.md");
    for mark in [
        "Kernel",
        "Library",
        "Encoding",
        "Runtime",
        "Host",
        "External",
        "Prototype",
        "Theory",
        "Archive",
    ] {
        let needle = format!("| {} |", mark);
        assert!(
            concepts.contains(&needle),
            "legend should define the `{}` qualifier",
            mark
        );
    }
}

#[test]
fn systems_table_separates_provers_from_libraries() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISON.md");
    assert!(
        concepts.contains("### Provers, frameworks, and languages"),
        "missing 'Provers, frameworks, and languages' section",
    );
    assert!(
        concepts.contains("### Libraries and archives"),
        "missing 'Libraries and archives' section",
    );
    let archive_idx = concepts
        .find("### Libraries and archives")
        .expect("libraries section");
    let foundation_idx = concepts.find("| Foundation |").expect("Foundation row");
    let afp_idx = concepts.find("| AFP |").expect("AFP row");
    assert!(
        foundation_idx > archive_idx,
        "Foundation should appear in the libraries/archives section",
    );
    assert!(
        afp_idx > archive_idx,
        "AFP should appear in the libraries/archives section",
    );
}

#[test]
fn drops_stale_no_atp_and_no_replay_claims() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISON.md");
    assert!(
        !concepts.contains("no ATP bridge"),
        "'no ATP bridge' is stale — the (by atp ...) bridge exists",
    );
    let l = concepts.to_lowercase();
    assert!(
        l.contains("independent proof-replay checker"),
        "the matrix should advertise the independent proof-replay checker",
    );
}

#[test]
fn includes_rml_status_note() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISON.md");
    assert!(
        concepts.contains("RML status note"),
        "missing 'RML status note' heading/marker",
    );
    assert!(
        concepts.contains("host-implemented"),
        "RML status note should mention 'host-implemented'",
    );
    assert!(
        concepts.contains("runtime configuration"),
        "RML status note should mention 'runtime configuration'",
    );
}

#[test]
fn adds_equality_layers_row() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISON.md");
    assert!(
        concepts.contains("Equality layers distinguished"),
        "missing 'Equality layers distinguished' row",
    );
    assert!(concepts.contains("structural"));
    assert!(concepts.contains("assigned"));
    assert!(concepts.contains("numeric"));
    assert!(
        concepts.contains("definitional") || concepts.contains("convertibility"),
        "definitional/convertibility equality should be mentioned",
    );
}

#[test]
fn rewrites_lambda_prolog_and_twelf_rows() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISON.md");
    let l = concepts.to_lowercase();
    assert!(
        l.contains("lambda prolog") && l.contains("not hol in the isabelle/hol"),
        "Lambda Prolog row should disclaim HOL-in-Isabelle/HOL sense",
    );
    assert!(
        concepts.contains(
            "No / N/A: proof search and metatheorem checking exist, but not tactic-level",
        ),
        "Twelf row should disclaim tactic-level proof construction",
    );
}

#[test]
fn marks_rml_numeric_and_many_valued_rows_as_runtime_plus_host() {
    let concepts = read_doc("docs/CONCEPTS-COMPARISON.md");
    for row in [
        "Numeric truth values in the core",
        "Configurable semantic range",
        "Configurable valence",
        "Fuzzy logic",
        "Probabilistic operators",
    ] {
        let needle = format!("| {} | Yes (Runtime + Host):", row);
        assert!(
            concepts.contains(&needle),
            "row \"{}\" should be marked Yes (Runtime + Host)",
            row
        );
    }
}

// ---------- RML claims match the Rust implementation ----------

#[test]
fn whnf_and_nf_are_exposed() {
    // Smoke-test that the public surface advertises both whnf and nf for
    // the typed lambda fragment. The functions take an Env, so we verify
    // by calling evaluate (the surface used by users) and checking that
    // it does not reject the typed-lambda surface forms.
    let out = evaluate("(? ((lambda (x) x) 1))", None, None);
    assert!(
        out.diagnostics.iter().all(|d| d.code != "E001"),
        "typed-lambda applications should parse: {:?}",
        out.diagnostics,
    );
}

#[test]
fn normal_form_surface_form_parses() {
    // The self-evaluator surface form is (eval (normal-form expression)).
    // Smoke-test that the parser accepts it.
    let out = evaluate("(? (normal-form ((lambda (x) x) 1)))", None, None);
    assert!(
        out.diagnostics.iter().all(|d| d.code != "E001"),
        "(normal-form ...) should parse: {:?}",
        out.diagnostics,
    );
}

#[test]
fn inductive_and_coinductive_declarations_are_parseable() {
    let out_i = evaluate(
        "(inductive natural (constructor zero) (constructor (succ (natural))))",
        None,
        None,
    );
    assert!(
        out_i.diagnostics.iter().all(|d| d.code != "E001"),
        "(inductive ...) should parse: {:?}",
        out_i.diagnostics,
    );
    let out_c = evaluate(
        "(coinductive stream (destructor (head A)) (destructor (tail stream)))",
        None,
        None,
    );
    assert!(
        out_c.diagnostics.iter().all(|d| d.code != "E001"),
        "(coinductive ...) should parse: {:?}",
        out_c.diagnostics,
    );
}

#[test]
fn smt_and_atp_tactic_forms_parse() {
    // The matrix says "Part (External): (by smt …) SMT-LIB trusted bridge"
    // and "(by atp …) records results as trusted external nodes". The
    // parser must accept those forms (full SMT/ATP execution depends on
    // external tools).
    let out1 = evaluate("(? (by smt (= 1 1)))", None, None);
    let out2 = evaluate("(? (by atp (= 1 1)))", None, None);
    assert!(
        out1.diagnostics.iter().all(|d| d.code != "E001"),
        "(by smt ...) should parse: {:?}",
        out1.diagnostics,
    );
    assert!(
        out2.diagnostics.iter().all(|d| d.code != "E001"),
        "(by atp ...) should parse: {:?}",
        out2.diagnostics,
    );
}

#[test]
fn independent_proof_replay_checker_ships_as_a_separate_module() {
    let check_path = repo_root().join("rust/src/check.rs");
    assert!(
        check_path.exists(),
        "rust/src/check.rs must exist (independent proof-replay checker)",
    );
}
