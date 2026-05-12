//! Lean 4 ↔ `.lino` CST converter (issue #138).
//!
//! Token-level lossless converter for Lean 4 source. Produces a
//! `lino-cst.lean.*` flat CST whose round-trip is byte-faithful:
//! `print_lean(&parse_lean(src)) == src`. Mirrors `js/src/cst-lean.mjs`
//! line for line.

use crate::cst::{dialects::LEAN, print_cst, CstNode};

/// Parse Lean 4 source into a `lino-cst.lean.*` CST.
pub fn parse_lean(src: &str) -> CstNode {
    let children = tokenise(src);
    CstNode::list(format!("{}.module", LEAN), children)
}

/// Print a Lean CST back to source.
pub fn print_lean(node: &CstNode) -> String {
    print_cst(node)
}

fn tokenise(src: &str) -> Vec<CstNode> {
    let chars: Vec<char> = src.chars().collect();
    let mut out: Vec<CstNode> = Vec::new();
    let mut i = 0usize;

    while i < chars.len() {
        let c = chars[i];

        if c == ' ' || c == '\t' || c == '\r' || c == '\n' {
            let mut j = i;
            while j < chars.len()
                && (chars[j] == ' ' || chars[j] == '\t' || chars[j] == '\r' || chars[j] == '\n')
            {
                j += 1;
            }
            out.push(CstNode::trivia(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.whitespace", LEAN)),
            ));
            i = j;
            continue;
        }

        if c == '-' && chars.get(i + 1) == Some(&'-') {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '\n' {
                j += 1;
            }
            out.push(CstNode::trivia(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.comment.line", LEAN)),
            ));
            i = j;
            continue;
        }

        if c == '/' && chars.get(i + 1) == Some(&'-') {
            let j = scan_block_comment(&chars, i);
            let tag = if chars.get(i + 2) == Some(&'-') {
                format!("{}.doc.block", LEAN)
            } else {
                format!("{}.comment.block", LEAN)
            };
            out.push(CstNode::trivia(
                chars[i..j].iter().collect::<String>(),
                Some(&tag),
            ));
            i = j;
            continue;
        }

        if c == '"' {
            let j = scan_string(&chars, i + 1, '"');
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.string_literal", LEAN)),
            ));
            i = j;
            continue;
        }

        if c == 'r' && chars.get(i + 1) == Some(&'"') {
            let j = scan_string(&chars, i + 2, '"');
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.raw_string_literal", LEAN)),
            ));
            i = j;
            continue;
        }

        if c == '\'' {
            let mut j = i + 1;
            if chars.get(j) == Some(&'\\') {
                j += 2;
            } else {
                j += 1;
            }
            if chars.get(j) == Some(&'\'') {
                j += 1;
                out.push(CstNode::token(
                    chars[i..j].iter().collect::<String>(),
                    Some(&format!("{}.char_literal", LEAN)),
                ));
                i = j;
                continue;
            }
            // Not a valid char literal; fall through to punctuation.
        }

        if c.is_ascii_digit() {
            let j = scan_number(&chars, i);
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.numeric_literal", LEAN)),
            ));
            i = j;
            continue;
        }

        if is_ident_start(c) {
            let mut j = i + 1;
            while j < chars.len() && is_ident_continue(chars[j]) {
                j += 1;
            }
            // Dotted hierarchical name: `Nat.succ`, `List.foldr`.
            while chars.get(j) == Some(&'.')
                && chars
                    .get(j + 1)
                    .map(|c| is_ident_start(*c))
                    .unwrap_or(false)
            {
                j += 1;
                while j < chars.len() && is_ident_continue(chars[j]) {
                    j += 1;
                }
            }
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.ident", LEAN)),
            ));
            i = j;
            continue;
        }

        // Multi-byte / other punctuation: emit one codepoint.
        out.push(CstNode::token(
            c.to_string(),
            Some(&format!("{}.punct", LEAN)),
        ));
        i += 1;
    }

    out
}

fn scan_block_comment(chars: &[char], i: usize) -> usize {
    let mut j = i + 2;
    let mut depth = 1;
    while j < chars.len() && depth > 0 {
        if chars[j] == '/' && chars.get(j + 1) == Some(&'-') {
            depth += 1;
            j += 2;
        } else if chars[j] == '-' && chars.get(j + 1) == Some(&'/') {
            depth -= 1;
            j += 2;
        } else {
            j += 1;
        }
    }
    j
}

fn scan_string(chars: &[char], mut j: usize, quote: char) -> usize {
    while j < chars.len() {
        let c = chars[j];
        if c == '\\' {
            j += 2;
            continue;
        }
        if c == quote {
            return j + 1;
        }
        j += 1;
    }
    j
}

fn scan_number(chars: &[char], i: usize) -> usize {
    let mut j = i;
    if chars.get(j) == Some(&'0') && matches!(chars.get(j + 1), Some('x') | Some('X')) {
        j += 2;
        while j < chars.len() && chars[j].is_ascii_hexdigit() {
            j += 1;
        }
        return j;
    }
    if chars.get(j) == Some(&'0') && matches!(chars.get(j + 1), Some('o') | Some('O')) {
        j += 2;
        while j < chars.len() && matches!(chars[j], '0'..='7') {
            j += 1;
        }
        return j;
    }
    if chars.get(j) == Some(&'0') && matches!(chars.get(j + 1), Some('b') | Some('B')) {
        j += 2;
        while j < chars.len() && matches!(chars[j], '0' | '1') {
            j += 1;
        }
        return j;
    }
    while j < chars.len() && chars[j].is_ascii_digit() {
        j += 1;
    }
    if chars.get(j) == Some(&'.')
        && chars.get(j + 1).map(|c| c.is_ascii_digit()).unwrap_or(false)
    {
        j += 1;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
        if matches!(chars.get(j), Some('e') | Some('E')) {
            j += 1;
            if matches!(chars.get(j), Some('+') | Some('-')) {
                j += 1;
            }
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
        }
    }
    j
}

fn is_ident_start(c: char) -> bool {
    if c == '_' || c.is_ascii_alphabetic() {
        return true;
    }
    if (c as u32) > 0x7F {
        return !is_lean_punct_char(c);
    }
    false
}

fn is_ident_continue(c: char) -> bool {
    if c == '_' || c == '\'' || c == '!' || c == '?' || c.is_ascii_alphanumeric() {
        return true;
    }
    if (c as u32) > 0x7F {
        return !is_lean_punct_char(c);
    }
    false
}

fn is_lean_punct_char(c: char) -> bool {
    matches!(
        c,
        '→' | '←' | '↦' | '⟨' | '⟩' | '⟦' | '⟧' | '«' | '»' | '‹' | '›'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(src: &str) {
        let node = parse_lean(src);
        let back = print_lean(&node);
        assert_eq!(back, src, "round-trip mismatch");
    }

    #[test]
    fn empty_string() {
        rt("");
    }

    #[test]
    fn simple_def() {
        rt("def f : Nat := 1\n");
    }

    #[test]
    fn line_comment() {
        rt("-- comment\ndef f : Nat := 1\n");
    }

    #[test]
    fn block_and_doc_comments() {
        rt("/- block -/\n/-- doc -/\n/-! module -/\n");
    }

    #[test]
    fn nested_block_comment() {
        rt("/- outer /- inner -/ still outer -/\ndef x := 1\n");
    }

    #[test]
    fn unicode_ident_and_arrow() {
        rt("def id {α : Type} (x : α) : α := x\n");
    }

    #[test]
    fn dotted_ident() {
        rt("#check Nat.succ\n");
    }

    #[test]
    fn char_literal() {
        rt("def c : Char := 'a'\n");
    }
}
