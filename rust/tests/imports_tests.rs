// Tests for `(import "...")` and `evaluate_file()` (issue #33).
// Mirrors js/tests/imports.test.mjs so any drift between the two
// implementations fails both test suites.

use rml::{evaluate, evaluate_file, format_diagnostic, EvaluateOptions, RunResult};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// Each test gets its own scratch directory under the system temp directory.
// We avoid pulling in `tempfile` to keep the dependency surface unchanged.
static COUNTER: AtomicU64 = AtomicU64::new(0);

fn make_tmp(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let mut p = std::env::temp_dir();
    p.push(format!("rml-rust-import-{}-{}-{}", tag, pid, n));
    fs::create_dir_all(&p).expect("create tmp");
    p
}

fn write(dir: &PathBuf, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    fs::write(&p, body).expect("write tmp file");
    p
}

fn cleanup(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

fn expect_num(r: &RunResult) -> f64 {
    match r {
        RunResult::Num(n) => *n,
        other => panic!("expected numeric result, got {:?}", other),
    }
}

#[test]
fn evaluate_file_returns_structured_result() {
    let dir = make_tmp("kb");
    let main = write(
        &dir,
        "kb.lino",
        "(a: a is a)\n((a = a) has probability 1)\n(? (a = a))\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    assert_eq!(out.diagnostics.len(), 0);
    assert_eq!(out.results.len(), 1);
    assert!((expect_num(&out.results[0]) - 1.0).abs() < 1e-9);
    cleanup(&dir);
}

#[test]
fn evaluate_file_reports_missing_file_as_e007() {
    let dir = make_tmp("missing");
    let missing = dir.join("no-such.lino");
    let out = evaluate_file(missing.to_str().unwrap(), EvaluateOptions::default());
    assert!(!out.diagnostics.is_empty());
    assert_eq!(out.diagnostics[0].code, "E007");
    cleanup(&dir);
}

#[test]
fn linear_chain_loads_declarations_across_three_files() {
    let dir = make_tmp("chain");
    write(
        &dir,
        "leaf.lino",
        "(z: z is z)\n((z = z) has probability 1)\n",
    );
    write(&dir, "mid.lino", "(import \"leaf.lino\")\n");
    let top = write(
        &dir,
        "top.lino",
        "(import \"mid.lino\")\n(? (z = z))\n",
    );
    let out = evaluate_file(top.to_str().unwrap(), EvaluateOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.results.len(), 1);
    assert!((expect_num(&out.results[0]) - 1.0).abs() < 1e-9);
    cleanup(&dir);
}

#[test]
fn diamond_pattern_loads_each_file_at_most_once() {
    let dir = make_tmp("diamond");
    write(
        &dir,
        "shared.lino",
        "(d: d is d)\n((d = d) has probability 1)\n",
    );
    write(&dir, "b.lino", "(import \"shared.lino\")\n");
    write(&dir, "c.lino", "(import \"shared.lino\")\n");
    let main = write(
        &dir,
        "main.lino",
        "(import \"b.lino\")\n(import \"c.lino\")\n(? (d = d))\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.results.len(), 1);
    assert!((expect_num(&out.results[0]) - 1.0).abs() < 1e-9);
    cleanup(&dir);
}

#[test]
fn two_file_cycle_emits_e007_with_span_of_closing_import() {
    let dir = make_tmp("cycle2");
    let a = write(&dir, "a.lino", "(import \"b.lino\")\n");
    let b = write(&dir, "b.lino", "(import \"a.lino\")\n");
    let out = evaluate_file(a.to_str().unwrap(), EvaluateOptions::default());
    assert_eq!(out.diagnostics.len(), 1, "diagnostics: {:?}", out.diagnostics);
    let d = &out.diagnostics[0];
    assert_eq!(d.code, "E007");
    assert!(d.message.to_lowercase().contains("cycle"), "{}", d.message);
    let span_file = d.span.file.as_deref().unwrap_or("");
    assert!(
        span_file.ends_with("b.lino"),
        "span file was {}",
        span_file
    );
    assert_eq!(d.span.line, 1);
    assert_eq!(d.span.col, 1);
    let _ = b;
    cleanup(&dir);
}

#[test]
fn self_import_emits_e007() {
    let dir = make_tmp("self");
    let f = write(&dir, "self.lino", "(import \"self.lino\")\n");
    let out = evaluate_file(f.to_str().unwrap(), EvaluateOptions::default());
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E007");
    assert!(out.diagnostics[0].message.to_lowercase().contains("cycle"));
    cleanup(&dir);
}

#[test]
fn three_file_cycle_lists_chain_in_message() {
    let dir = make_tmp("cycle3");
    let a = write(&dir, "tri-a.lino", "(import \"tri-b.lino\")\n");
    write(&dir, "tri-b.lino", "(import \"tri-c.lino\")\n");
    write(&dir, "tri-c.lino", "(import \"tri-a.lino\")\n");
    let out = evaluate_file(a.to_str().unwrap(), EvaluateOptions::default());
    assert_eq!(out.diagnostics.len(), 1);
    let msg = &out.diagnostics[0].message;
    assert_eq!(out.diagnostics[0].code, "E007");
    assert!(msg.contains("tri-a.lino"), "{}", msg);
    assert!(msg.contains("tri-b.lino"), "{}", msg);
    assert!(msg.contains("tri-c.lino"), "{}", msg);
    cleanup(&dir);
}

#[test]
fn diagnostics_from_imported_file_keep_their_span() {
    let dir = make_tmp("forward");
    write(&dir, "broken.lino", "(=: missing identity)\n");
    let host = write(&dir, "host.lino", "(import \"broken.lino\")\n");
    let out = evaluate_file(host.to_str().unwrap(), EvaluateOptions::default());
    assert_eq!(out.diagnostics.len(), 1);
    let d = &out.diagnostics[0];
    assert_eq!(d.code, "E001");
    let span_file = d.span.file.as_deref().unwrap_or("");
    assert!(span_file.ends_with("broken.lino"), "{}", span_file);
    cleanup(&dir);
}

#[test]
fn missing_import_target_reports_e007_with_importing_span() {
    let dir = make_tmp("missing-import");
    let main = write(&dir, "main.lino", "(import \"no-such.lino\")\n");
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    assert_eq!(out.diagnostics.len(), 1);
    let d = &out.diagnostics[0];
    assert_eq!(d.code, "E007");
    assert!(d.message.contains("no-such.lino"), "{}", d.message);
    let span_file = d.span.file.as_deref().unwrap_or("");
    assert!(span_file.ends_with("main.lino"), "{}", span_file);
    cleanup(&dir);
}

#[test]
fn e007_diagnostic_formats_like_other_codes() {
    let dir = make_tmp("format");
    let main = write(&dir, "main.lino", "(import \"no-such.lino\")\n");
    let src = fs::read_to_string(&main).unwrap();
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    assert_eq!(out.diagnostics.len(), 1);
    let text = format_diagnostic(&out.diagnostics[0], Some(&src));
    let lines: Vec<&str> = text.split('\n').collect();
    assert!(lines[0].contains(":1:1: E007:"), "line[0]: {}", lines[0]);
    assert_eq!(lines[1], "(import \"no-such.lino\")");
    assert_eq!(lines[2], "^");
    cleanup(&dir);
}

#[test]
fn inline_import_without_file_resolves_against_cwd() {
    // Drop into a temp dir, evaluate an in-memory program that imports a
    // sibling file. We don't compete with other tests for CWD here because
    // each scratch directory is unique.
    let dir = make_tmp("inline");
    write(
        &dir,
        "lib.lino",
        "(a: a is a)\n((a = a) has probability 1)\n",
    );
    let original = std::env::current_dir().unwrap();
    std::env::set_current_dir(&dir).unwrap();
    let result = std::panic::catch_unwind(|| {
        evaluate(
            "(import \"lib.lino\")\n(? (a = a))",
            None,
            None,
        )
    });
    std::env::set_current_dir(&original).unwrap();
    cleanup(&dir);
    let out = result.expect("evaluate panicked");
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.results.len(), 1);
    assert!((expect_num(&out.results[0]) - 1.0).abs() < 1e-9);
}
