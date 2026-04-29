// RML CLI — run a LiNo knowledge base and print query results
use rml::{evaluate, format_diagnostic, RunResult};
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rml <kb.lino>");
        return ExitCode::from(1);
    }
    let file = &args[1];
    let text = match fs::read_to_string(file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", file, e);
            return ExitCode::from(1);
        }
    };
    let evaluation = evaluate(&text, Some(file), None);
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
