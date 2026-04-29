// Tests for `(namespace ...)` and qualified references (issue #34).
// Mirrors js/tests/namespaces.test.mjs so any drift between the two
// implementations fails both test suites.

use rml::{evaluate, evaluate_file, EvaluateOptions, RunResult};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn make_tmp(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let mut p = std::env::temp_dir();
    p.push(format!("rml-rust-ns-{}-{}-{}", tag, pid, n));
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

// ---- (namespace ...) — declaration ----

#[test]
fn namespace_prefixes_a_definition_with_the_active_namespace() {
    let dir = make_tmp("decl1");
    write(&dir, "lib.lino", "(namespace classical)\n(and: min)\n");
    let main = write(
        &dir,
        "main.lino",
        "(import \"lib.lino\")\n(? (classical.and 1 0))\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.results.len(), 1);
    assert!((expect_num(&out.results[0]) - 0.0).abs() < 1e-9);
    cleanup(&dir);
}

#[test]
fn namespace_supports_multiple_definitions() {
    let dir = make_tmp("decl2");
    write(
        &dir,
        "lib2.lino",
        "(namespace classical)\n(and: min)\n(or: max)\n",
    );
    let main = write(
        &dir,
        "main2.lino",
        "(import \"lib2.lino\")\n(? (classical.and 1 0))\n(? (classical.or 1 0))\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.results.len(), 2);
    assert!((expect_num(&out.results[0]) - 0.0).abs() < 1e-9);
    assert!((expect_num(&out.results[1]) - 1.0).abs() < 1e-9);
    cleanup(&dir);
}

#[test]
fn namespace_rejects_dotted_name_with_e009() {
    let dir = make_tmp("decl3");
    let bad = write(&dir, "bad.lino", "(namespace foo.bar)\n");
    let out = evaluate_file(bad.to_str().unwrap(), EvaluateOptions::default());
    assert_eq!(out.diagnostics.len(), 1, "{:?}", out.diagnostics);
    assert_eq!(out.diagnostics[0].code, "E009");
    cleanup(&dir);
}

// ---- (import "..." as <alias>) — aliased imports ----

#[test]
fn aliased_import_resolves_qualified_references() {
    let dir = make_tmp("alias1");
    write(
        &dir,
        "classical.lino",
        "(namespace classical)\n(and: min)\n(or: max)\n",
    );
    let main = write(
        &dir,
        "main.lino",
        "(import \"classical.lino\" as cl)\n(? (cl.and 1 0))\n(? (cl.or 1 0))\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.results.len(), 2);
    assert!((expect_num(&out.results[0]) - 0.0).abs() < 1e-9);
    assert!((expect_num(&out.results[1]) - 1.0).abs() < 1e-9);
    cleanup(&dir);
}

#[test]
fn aliased_import_emits_e009_on_alias_collision() {
    let dir = make_tmp("alias2");
    write(&dir, "a.lino", "(namespace foo)\n(x: max)\n");
    write(&dir, "b.lino", "(namespace bar)\n(x: min)\n");
    let collide = write(
        &dir,
        "collide.lino",
        "(import \"a.lino\" as a)\n(import \"b.lino\" as a)\n",
    );
    let out = evaluate_file(collide.to_str().unwrap(), EvaluateOptions::default());
    let e009s: Vec<_> = out.diagnostics.iter().filter(|d| d.code == "E009").collect();
    assert_eq!(e009s.len(), 1, "{:?}", out.diagnostics);
    assert!(
        e009s[0].message.contains("alias \"a\""),
        "{}",
        e009s[0].message
    );
    cleanup(&dir);
}

#[test]
fn distinct_aliases_point_at_different_namespaces() {
    let dir = make_tmp("alias3");
    write(&dir, "la.lino", "(namespace foo)\n(and: max)\n");
    write(&dir, "lb.lino", "(namespace bar)\n(and: min)\n");
    let multi = write(
        &dir,
        "multi.lino",
        "(import \"la.lino\" as a)\n(import \"lb.lino\" as b)\n\
         (? (a.and 0.2 0.5))\n(? (b.and 0.2 0.5))\n",
    );
    let out = evaluate_file(multi.to_str().unwrap(), EvaluateOptions::default());
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.results.len(), 2);
    assert!((expect_num(&out.results[0]) - 0.5).abs() < 1e-9);
    assert!((expect_num(&out.results[1]) - 0.2).abs() < 1e-9);
    cleanup(&dir);
}

// ---- (namespace ...) — shadowing (E008) ----

#[test]
fn shadowing_emits_e008_for_op_redefinition() {
    let dir = make_tmp("shadow1");
    write(&dir, "lib.lino", "(myop: avg)\n");
    let main = write(
        &dir,
        "main.lino",
        "(import \"lib.lino\")\n(myop: max)\n(? (myop 0.2 0.4 0.8))\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    let e008s: Vec<_> = out.diagnostics.iter().filter(|d| d.code == "E008").collect();
    assert_eq!(e008s.len(), 1, "{:?}", out.diagnostics);
    assert!(e008s[0].message.contains("myop"), "{}", e008s[0].message);
    assert!(
        e008s[0].message.to_lowercase().contains("shadows"),
        "{}",
        e008s[0].message
    );
    // The redefinition still takes effect (mirrors the JS suite assertion).
    assert_eq!(out.results.len(), 1);
    assert!((expect_num(&out.results[0]) - 0.0).abs() < 1e-9);
    cleanup(&dir);
}

#[test]
fn shadowing_emits_e008_for_qualified_rebind() {
    let dir = make_tmp("shadow2");
    write(
        &dir,
        "lib2.lino",
        "(namespace classical)\n(and: min)\n",
    );
    let main = write(
        &dir,
        "main2.lino",
        "(import \"lib2.lino\" as cl)\n(cl.and: max)\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    let e008s: Vec<_> = out.diagnostics.iter().filter(|d| d.code == "E008").collect();
    assert_eq!(e008s.len(), 1, "{:?}", out.diagnostics);
    assert!(
        e008s[0].message.contains("cl.and"),
        "{}",
        e008s[0].message
    );
    cleanup(&dir);
}

#[test]
fn shadowing_warns_only_once_for_double_rebind() {
    let dir = make_tmp("shadow3");
    write(&dir, "lib3.lino", "(myop: avg)\n");
    let main = write(
        &dir,
        "main3.lino",
        "(import \"lib3.lino\")\n(myop: max)\n(myop: min)\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    let e008s: Vec<_> = out.diagnostics.iter().filter(|d| d.code == "E008").collect();
    assert_eq!(e008s.len(), 1, "{:?}", out.diagnostics);
    cleanup(&dir);
}

#[test]
fn shadowing_does_not_warn_for_fresh_name() {
    let dir = make_tmp("shadow4");
    write(&dir, "lib4.lino", "(myop: avg)\n");
    let main = write(
        &dir,
        "main4.lino",
        "(import \"lib4.lino\")\n(otherop: max)\n",
    );
    let out = evaluate_file(main.to_str().unwrap(), EvaluateOptions::default());
    let e008s: Vec<_> = out.diagnostics.iter().filter(|d| d.code == "E008").collect();
    assert_eq!(e008s.len(), 0, "{:?}", out.diagnostics);
    cleanup(&dir);
}

// ---- inline (namespace ...) without import ----

#[test]
fn inline_namespace_prefix_is_honoured() {
    let out = evaluate(
        "(namespace foo)\n(and: min)\n(? (foo.and 1 0))\n",
        None,
        None,
    );
    assert!(out.diagnostics.is_empty(), "{:?}", out.diagnostics);
    assert_eq!(out.results.len(), 1);
    assert!((expect_num(&out.results[0]) - 0.0).abs() < 1e-9);
}
