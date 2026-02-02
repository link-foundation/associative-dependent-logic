// ADL CLI — run a LiNo knowledge base and print query results
use adl::run;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: adl <kb.lino>");
        std::process::exit(1);
    }
    let file = &args[1];
    let text = fs::read_to_string(file).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", file, e);
        std::process::exit(1);
    });
    let outs = run(&text, None);
    for v in outs {
        let formatted = format!("{:.6}", v);
        // Remove trailing zeros after decimal point
        let formatted = formatted.trim_end_matches('0').trim_end_matches('.');
        println!("{}", formatted);
    }
}
