//! Universal CST converter dispatch (issue #138).
//!
//! Single entry point that turns host-language source into a `.lino` CST and
//! back. Mirrors `js/src/cst-convert.mjs`. Provides:
//!
//! - [`parse_to_cst`] — host source → CST.
//! - [`print_from_cst`] — CST → host source.
//! - [`round_trip`] — convenience helper that asserts byte fidelity.
//!
//! Supported `lang` values: `"rust"`, `"js"` (or `"javascript"`), `"lean"`, `"rocq"`.

use crate::cst::CstNode;
use crate::cst_js::{parse_js, print_js};
use crate::cst_lean::{parse_lean, print_lean};
use crate::cst_rocq::{parse_rocq, print_rocq};
use crate::cst_rust::{parse_rust, print_rust};

/// The four host languages plus the `javascript` alias.
pub const SUPPORTED_LANGUAGES: &[&str] = &["rust", "js", "javascript", "lean", "rocq"];

/// Parse host-language source into a `.lino` CST.
pub fn parse_to_cst(src: &str, lang: &str) -> Result<CstNode, String> {
    match lang {
        "rust" => Ok(parse_rust(src)),
        "js" | "javascript" => Ok(parse_js(src)),
        "lean" => Ok(parse_lean(src)),
        "rocq" => Ok(parse_rocq(src)),
        other => Err(format!("unsupported language for parse_to_cst: {}", other)),
    }
}

/// Print a CST node back to host-language source.
pub fn print_from_cst(node: &CstNode, lang: &str) -> Result<String, String> {
    match lang {
        "rust" => Ok(print_rust(node)),
        "js" | "javascript" => Ok(print_js(node)),
        "lean" => Ok(print_lean(node)),
        "rocq" => Ok(print_rocq(node)),
        other => Err(format!("unsupported language for print_from_cst: {}", other)),
    }
}

/// Result of a round-trip check.
pub struct RoundTripResult {
    pub ok: bool,
    pub source: String,
    pub round_tripped: String,
}

/// Verify that `print_from_cst(parse_to_cst(src, lang), lang) == src`.
pub fn round_trip(src: &str, lang: &str) -> Result<RoundTripResult, String> {
    let cst = parse_to_cst(src, lang)?;
    let round_tripped = print_from_cst(&cst, lang)?;
    Ok(RoundTripResult {
        ok: round_tripped == src,
        source: src.to_string(),
        round_tripped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatches_by_language_name() {
        for lang in &["rust", "js", "lean", "rocq"] {
            let sample = if *lang == "rocq" {
                "Definition x := 1.\n"
            } else {
                "x\n"
            };
            let out = print_from_cst(&parse_to_cst(sample, lang).unwrap(), lang).unwrap();
            assert_eq!(out, sample);
        }
    }

    #[test]
    fn javascript_alias_works_like_js() {
        let src = "const x = 1;\n";
        let out =
            print_from_cst(&parse_to_cst(src, "javascript").unwrap(), "javascript").unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn rejects_unsupported_language() {
        assert!(parse_to_cst("x", "python").is_err());
        let n = CstNode::list("x", vec![]);
        assert!(print_from_cst(&n, "python").is_err());
    }

    #[test]
    fn round_trip_helper_reports_ok() {
        let r = round_trip("fn main() {}\n", "rust").unwrap();
        assert!(r.ok);
        assert_eq!(r.round_tripped, "fn main() {}\n");
    }
}
