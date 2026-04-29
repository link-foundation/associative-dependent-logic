// RML CLI — run a LiNo knowledge base and print query results
use rml::{
    evaluate_with_options, format_diagnostic, format_trace_event, EvaluateOptions, RunResult,
};
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut trace = false;
    let mut positionals: Vec<String> = Vec::new();
    for arg in args.iter().skip(1) {
        if arg == "--trace" {
            trace = true;
        } else {
            positionals.push(arg.clone());
        }
    }
    if positionals.is_empty() {
        eprintln!("Usage: rml [--trace] <kb.lino>");
        return ExitCode::from(1);
    }
    let file = &positionals[0];
    let text = match fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", file, e);
            return ExitCode::from(1);
        }
    };
    let evaluation = evaluate_with_options(
        &text,
        Some(file),
        EvaluateOptions {
            env: None,
            trace,
        },
    );
    if trace {
        for event in &evaluation.trace {
            eprintln!("{}", format_trace_event(event));
        }
    }
    for v in evaluation.results {
        match v {
            RunResult::Num(n) => {
                let formatted = format!("{:.6}", n);
                let formatted = formatted.trim_end_matches('0').trim_end_matches('.');
                println!("{}", formatted);
            }
            RunResult::Type(s) => println!("{}", s),
        }
    }
    let has_diagnostics = !evaluation.diagnostics.is_empty();
    for diag in &evaluation.diagnostics {
        eprintln!("{}", format_diagnostic(diag, Some(&text)));
    }
    if has_diagnostics {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
