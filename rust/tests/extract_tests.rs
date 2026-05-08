// Program extraction tests (issue #66).
// Mirrors js/tests/extract.test.mjs so the JavaScript and Rust
// implementations expose the same extraction surface.

use rml::{extract_program, ExtractTarget};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

const PROGRAM: &str = r#"
(Natural: (Type 0) Natural)
(inc: lambda (Natural x) (x + 1))
(double: lambda (Natural x) (x * 2))
(combo: lambda (Natural x) (apply double (apply inc x)))
(? ((apply combo 3) = 8))
"#;

fn make_tmp(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let mut p = std::env::temp_dir();
    p.push(format!("rml-rust-extract-{}-{}-{}", tag, pid, n));
    fs::create_dir_all(&p).expect("create temp directory");
    p
}

fn cleanup(dir: &PathBuf) {
    let _ = fs::remove_dir_all(dir);
}

#[test]
fn extracts_typed_lambda_program_to_runnable_rust_with_tests() {
    let source = extract_program(PROGRAM, ExtractTarget::Rust).expect("extract rust");
    assert!(source.contains("pub fn inc(x: f64) -> f64"), "{}", source);
    assert!(source.contains("pub fn combo(x: f64) -> f64"), "{}", source);
    assert!(source.contains("fn rml_query_1"), "{}", source);

    let dir = make_tmp("compile");
    let generated = dir.join("program.rs");
    let bin = dir.join("program-tests");
    fs::write(&generated, source).expect("write generated Rust");

    let compile = Command::new("rustc")
        .arg("--test")
        .arg(&generated)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("run rustc");
    assert!(
        compile.status.success(),
        "rustc failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&bin).output().expect("run generated tests");
    assert!(
        run.status.success(),
        "generated tests failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    cleanup(&dir);
}

#[test]
fn extracts_same_program_to_javascript_source() {
    let source = extract_program(PROGRAM, ExtractTarget::JavaScript).expect("extract js");
    assert!(source.contains("export function inc(x)"), "{}", source);
    assert!(source.contains("export function combo(x)"), "{}", source);
    assert!(source.contains("__runRmlExtractedTests"), "{}", source);
}

#[test]
fn rejects_probabilistic_forms() {
    let err = extract_program("((a = a) has probability 1)", ExtractTarget::Rust)
        .expect_err("probability assignment must be rejected");
    assert!(err.to_lowercase().contains("probability"), "{}", err);
}

#[test]
fn rml_extract_cli_prints_rust_source() {
    let dir = make_tmp("cli");
    let program = dir.join("program.lino");
    fs::write(&program, PROGRAM).expect("write program");

    let output = Command::new(env!("CARGO_BIN_EXE_rml"))
        .arg("extract")
        .arg("rust")
        .arg(&program)
        .output()
        .expect("run rml extract");
    assert!(
        output.status.success(),
        "rml extract failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("pub fn combo(x: f64) -> f64"), "{}", stdout);
    cleanup(&dir);
}
