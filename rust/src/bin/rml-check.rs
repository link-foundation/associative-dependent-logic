// rml-check — independent proof-replay checker (issue #36).
//
// CLI front-end for the `rml::check` module. Reads a program file and a
// proof file, validates that every derivation in the proof file
// corresponds to a query in the program, and prints `OK: N derivations
// replayed.` on success.

use rml::check::check_program;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: rml-check <program.lino> <proofs.lino>");
        return ExitCode::from(2);
    }
    let program_path = &args[1];
    let proofs_path = &args[2];
    let program = match fs::read_to_string(program_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", program_path, e);
            return ExitCode::from(1);
        }
    };
    let proofs = match fs::read_to_string(proofs_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error reading {}: {}", proofs_path, e);
            return ExitCode::from(1);
        }
    };
    let result = check_program(&program, &proofs);
    if result.is_ok() {
        println!("OK: {} derivations replayed.", result.ok.len());
        ExitCode::SUCCESS
    } else {
        for err in &result.errors {
            let path = if err.path.is_empty() {
                String::from("<root>")
            } else {
                err.path.join(" / ")
            };
            eprintln!("FAIL [{}]: {}", path, err.message);
        }
        ExitCode::from(1)
    }
}
