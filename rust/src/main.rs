// RML CLI — run a LiNo knowledge base or launch the interactive REPL
use rml::repl::run_repl;
use rml::{
    evaluate_with_options, extract_program, format_diagnostic, format_trace_event, EnvOptions,
    EvaluateOptions, ExtractTarget, RunResult,
};
use std::env;
use std::fs;
use std::io::{self, BufReader, IsTerminal};
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
        eprintln!(
            "Usage: rml [--trace] <kb.lino>   |   rml repl   |   rml extract <js|rust> <kb.lino>"
        );
        return ExitCode::from(1);
    }
    let arg = &positionals[0];
    if arg == "extract" {
        if positionals.len() != 3 {
            eprintln!("Usage: rml extract <js|rust> <kb.lino>");
            return ExitCode::from(1);
        }
        let target = match ExtractTarget::from_name(&positionals[1]) {
            Some(target) => target,
            None => {
                eprintln!("Unknown extraction target: {}", positionals[1]);
                return ExitCode::from(1);
            }
        };
        let file = &positionals[2];
        let text = match fs::read_to_string(file) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error reading {}: {}", file, e);
                return ExitCode::from(1);
            }
        };
        match extract_program(&text, target) {
            Ok(source) => {
                println!("{}", source);
                return ExitCode::SUCCESS;
            }
            Err(message) => {
                eprintln!("{}", message);
                return ExitCode::from(1);
            }
        }
    }
    if arg == "repl" {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let stderr = io::stderr();
        let show_prompt = stdin.is_terminal();
        let mut input = BufReader::new(stdin.lock());
        let mut out = stdout.lock();
        let mut err = stderr.lock();
        if let Err(e) = run_repl(
            EnvOptions::default(),
            show_prompt,
            &mut input,
            &mut out,
            &mut err,
        ) {
            eprintln!("REPL error: {}", e);
            return ExitCode::from(1);
        }
        return ExitCode::SUCCESS;
    }
    let file = arg;
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
            ..EvaluateOptions::default()
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
