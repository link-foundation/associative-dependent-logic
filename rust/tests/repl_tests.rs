// Tests for the interactive REPL (issue #29).
// Mirrors js/tests/repl.test.mjs so any drift between the two
// implementations fails both test suites.

use rml::repl::{format_env, run_repl, Repl, ReplStep};
use rml::EnvOptions;
use std::env;
use std::fs;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_tmpdir(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut d = env::temp_dir();
    d.push(format!("rml-repl-{}-{}", tag, nanos));
    fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn case_study_session_declare_assign_query() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    let s1 = repl.feed("(a: a is a)");
    assert!(s1.error.is_empty(), "errors: {}", s1.error);
    let s2 = repl.feed("((a = a) has probability 1)");
    assert!(s2.error.is_empty(), "errors: {}", s2.error);
    let s3 = repl.feed("(? (a = a))");
    assert!(s3.error.is_empty(), "errors: {}", s3.error);
    assert_eq!(s3.output, "1");
    assert!(!s3.exit);
}

#[test]
fn blank_line_is_a_noop_and_preserves_state() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    repl.feed("(a: a is a)");
    let blank = repl.feed("   ");
    assert_eq!(blank.output, "");
    assert_eq!(blank.error, "");
    assert!(!blank.exit);
    assert!(repl.env.terms.contains("a"));
}

#[test]
fn later_error_does_not_lose_earlier_results() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    let step = repl.feed("(valence: 2)\n(p has probability 1)\n(? p)\n(=: missing identity)");
    assert_eq!(step.output, "1");
    assert!(step.error.contains("E001"), "error: {}", step.error);
}

#[test]
fn help_returns_help_text() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    let step = repl.feed(":help");
    assert!(step.output.contains(":load"), "output: {}", step.output);
    assert!(step.output.contains(":reset"), "output: {}", step.output);
}

#[test]
fn question_alias_for_help() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    let step = repl.feed(":?");
    assert!(step.output.contains(":load"), "output: {}", step.output);
}

#[test]
fn quit_and_exit_request_termination() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    assert!(repl.feed(":quit").exit);
    assert!(repl.feed(":exit").exit);
}

#[test]
fn reset_clears_terms_and_transcript() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    repl.feed("(a: a is a)");
    assert!(repl.env.terms.contains("a"));
    assert!(!repl.transcript.is_empty());
    let step = repl.feed(":reset");
    assert_eq!(step.output, "Env reset.");
    assert!(!repl.env.terms.contains("a"));
    assert!(repl.transcript.is_empty());
}

#[test]
fn env_meta_command_prints_summary() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    repl.feed("(a: a is a)");
    repl.feed("((a = a) has probability 1)");
    let step = repl.feed(":env");
    assert!(step.output.contains("range:"), "output: {}", step.output);
    assert!(step.output.contains("valence:"), "output: {}", step.output);
    assert!(step.output.contains("terms:"), "output: {}", step.output);
    assert!(step.output.contains('a'), "output: {}", step.output);
    assert!(
        step.output.contains("assignments:"),
        "output: {}",
        step.output
    );
}

#[test]
fn load_reads_a_file_into_running_env() {
    let dir = unique_tmpdir("load");
    let file = dir.join("kb.lino");
    fs::write(&file, "(a: a is a)\n((a = a) has probability 1)\n").unwrap();
    let mut repl = Repl::new(EnvOptions::default(), Some(dir.clone()));
    let load_step = repl.feed(&format!(":load {}", file.display()));
    assert!(load_step.error.is_empty(), "errors: {}", load_step.error);
    let q = repl.feed("(? (a = a))");
    assert_eq!(q.output, "1");
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_reports_missing_files() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    let step = repl.feed(":load /no/such/path.lino");
    assert!(step.error.contains(":load failed"), "error: {}", step.error);
    assert_eq!(step.output, "");
}

#[test]
fn load_with_no_argument_errors() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    let step = repl.feed(":load");
    assert!(
        step.error.contains(":load requires"),
        "error: {}",
        step.error
    );
}

#[test]
fn save_writes_transcript_to_disk() {
    let dir = unique_tmpdir("save");
    let file = dir.join("session.lino");
    let mut repl = Repl::new(EnvOptions::default(), Some(dir.clone()));
    repl.feed("(a: a is a)");
    repl.feed("((a = a) has probability 1)");
    let saved = repl.feed(&format!(":save {}", file.display()));
    assert!(saved.error.is_empty(), "errors: {}", saved.error);
    assert!(saved.output.contains("Saved"), "output: {}", saved.output);
    let text = fs::read_to_string(&file).unwrap();
    assert!(text.contains("(a: a is a)"), "saved: {}", text);
    assert!(
        text.contains("((a = a) has probability 1)"),
        "saved: {}",
        text
    );
    fs::remove_dir_all(&dir).ok();
}

#[test]
fn save_with_no_argument_errors() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    let step = repl.feed(":save");
    assert!(
        step.error.contains(":save requires"),
        "error: {}",
        step.error
    );
}

#[test]
fn unknown_meta_command_surfaces_friendly_error() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    let step = repl.feed(":bogus");
    assert!(
        step.error.contains("Unknown meta-command"),
        "error: {}",
        step.error
    );
    assert!(step.error.contains(":bogus"), "error: {}", step.error);
}

#[test]
fn completion_offers_meta_commands_for_colon_prefix() {
    let repl = Repl::new(EnvOptions::default(), None);
    let hits = repl.completion_candidates(":lo");
    assert!(hits.iter().any(|s| s == ":load"), "hits: {:?}", hits);
}

#[test]
fn completion_offers_identifiers_from_env() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    repl.feed("(apple: apple is apple)");
    let hits = repl.completion_candidates("app");
    assert!(hits.iter().any(|s| s == "apple"), "hits: {:?}", hits);
}

#[test]
fn completion_returns_builtin_keywords_for_empty_env() {
    let repl = Repl::new(EnvOptions::default(), None);
    let hits = repl.completion_candidates("");
    assert!(
        hits.iter().any(|s| s == "probability"),
        "hits: {:?}",
        hits
    );
    assert!(hits.iter().any(|s| s == "lambda"), "hits: {:?}", hits);
}

#[test]
fn format_env_shows_continuous_valence_by_default() {
    let repl = Repl::new(EnvOptions::default(), None);
    let text = format_env(&repl.env);
    assert!(
        text.contains("valence:  continuous"),
        "text: {}",
        text
    );
}

#[test]
fn format_env_shows_numeric_valence_when_set() {
    let mut repl = Repl::new(EnvOptions::default(), None);
    repl.feed("(valence: 2)");
    let text = format_env(&repl.env);
    assert!(text.contains("valence:  2"), "text: {}", text);
}

#[test]
fn run_repl_drives_io_streams_to_completion() {
    let stdin = b"(a: a is a)\n((a = a) has probability 1)\n(? (a = a))\n:quit\n";
    let mut input = Cursor::new(stdin.to_vec());
    let mut output: Vec<u8> = Vec::new();
    let mut err: Vec<u8> = Vec::new();
    run_repl(EnvOptions::default(), false, &mut input, &mut output, &mut err).unwrap();
    let out = String::from_utf8(output).unwrap();
    assert_eq!(out.trim(), "1");
    assert!(err.is_empty(), "stderr: {}", String::from_utf8_lossy(&err));
}

#[test]
fn replstep_default_is_empty_no_exit() {
    let s = ReplStep::default();
    assert_eq!(s.output, "");
    assert_eq!(s.error, "");
    assert!(!s.exit);
}
