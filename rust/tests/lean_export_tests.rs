// Lean 4 exporter tests for issue #60.
// Mirrors js/tests/lean-export.test.mjs so the JS and Rust CLIs stay aligned.

use rml::export_lean;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn fixture_path() -> PathBuf {
    repo_root().join("examples").join("lean-export-basic.lino")
}

fn expected_path() -> PathBuf {
    repo_root().join("examples").join("lean-export-basic.lean")
}

fn read_fixture() -> String {
    fs::read_to_string(fixture_path()).expect("fixture")
}

fn read_expected() -> String {
    fs::read_to_string(expected_path()).expect("expected")
}

#[test]
fn exports_supported_typed_subset_to_lean_source() {
    let fixture = read_fixture();
    let out = export_lean(&fixture, fixture_path().to_str());
    assert_eq!(
        out.diagnostics.len(),
        0,
        "diagnostics: {:?}",
        out.diagnostics
    );
    assert_eq!(out.source, read_expected());
}

#[test]
fn rejects_probabilistic_forms_with_e050() {
    let out = export_lean(
        "(p: p is p)\n((p = p) has probability 1)\n",
        Some("prob.lino"),
    );
    assert_eq!(out.source, "");
    assert_eq!(out.diagnostics.len(), 1);
    assert_eq!(out.diagnostics[0].code, "E050");
    assert!(out.diagnostics[0].message.contains("probabilistic"));
    assert_eq!(out.diagnostics[0].span.file.as_deref(), Some("prob.lino"));
    assert_eq!(out.diagnostics[0].span.line, 2);
}

#[test]
fn cli_writes_lean_artifact_to_output_path() {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("rml-lean-export-{}-{}", std::process::id(), stamp));
    fs::create_dir_all(&dir).expect("tmp dir");
    let out_path = dir.join("out.lean");
    let output = Command::new(env!("CARGO_BIN_EXE_rml"))
        .args([
            "export",
            "lean",
            fixture_path().to_str().expect("fixture path"),
            "-o",
            out_path.to_str().expect("out path"),
        ])
        .output()
        .expect("run rml");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&output.stdout), "");
    assert_eq!(
        fs::read_to_string(&out_path).expect("out file"),
        read_expected()
    );
    let _ = fs::remove_dir_all(&dir);
}
