//! Rust ↔ `.lino` CST converter (issue #138).
//!
//! Token-level lossless converter for Rust source. Produces a
//! `lino-cst.rust.*` flat CST whose round-trip is byte-faithful:
//! `print_rust(&parse_rust(src)) == src`. Mirrors `js/src/cst-rust.mjs`
//! line for line.

use crate::cst::{dialects::RUST, print_cst, CstNode};

/// Parse Rust source into a `lino-cst.rust.*` CST.
pub fn parse_rust(src: &str) -> CstNode {
    let children = tokenise(src);
    CstNode::list(format!("{}.source_file", RUST), children)
}

/// Print a Rust CST back to source.
pub fn print_rust(node: &CstNode) -> String {
    print_cst(node)
}

fn tokenise(src: &str) -> Vec<CstNode> {
    let chars: Vec<char> = src.chars().collect();
    let mut out: Vec<CstNode> = Vec::new();
    let mut i = 0usize;

    if chars.len() >= 2 && chars[0] == '#' && chars[1] == '!' && chars.get(2) != Some(&'[') {
        let mut j = i;
        while j < chars.len() && chars[j] != '\n' {
            j += 1;
        }
        out.push(CstNode::trivia(
            chars[i..j].iter().collect::<String>(),
            Some(&format!("{}.shebang", RUST)),
        ));
        i = j;
    }

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
                Some(&format!("{}.whitespace", RUST)),
            ));
            i = j;
            continue;
        }

        if c == '/' && chars.get(i + 1) == Some(&'/') {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '\n' {
                j += 1;
            }
            out.push(CstNode::trivia(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.comment.line", RUST)),
            ));
            i = j;
            continue;
        }

        if c == '/' && chars.get(i + 1) == Some(&'*') {
            let j = scan_block_comment(&chars, i);
            out.push(CstNode::trivia(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.comment.block", RUST)),
            ));
            i = j;
            continue;
        }

        if c == '"' {
            let j = scan_string(&chars, i + 1, '"');
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.string_literal", RUST)),
            ));
            i = j;
            continue;
        }

        // Raw/byte strings: b"...", r"...", r#"..."#, br"...".
        if (c == 'b' || c == 'r')
            && (chars.get(i + 1) == Some(&'"')
                || (c == 'r' && chars.get(i + 1) == Some(&'#'))
                || (c == 'b'
                    && chars.get(i + 1) == Some(&'r')
                    && (chars.get(i + 2) == Some(&'"') || chars.get(i + 2) == Some(&'#'))))
        {
            if let Some(j) = scan_raw_or_prefixed_string(&chars, i) {
                out.push(CstNode::token(
                    chars[i..j].iter().collect::<String>(),
                    Some(&format!("{}.string_literal", RUST)),
                ));
                i = j;
                continue;
            }
        }

        if c == '\'' {
            let lifetime_end = scan_lifetime(&chars, i);
            if lifetime_end > i + 1 {
                out.push(CstNode::token(
                    chars[i..lifetime_end].iter().collect::<String>(),
                    Some(&format!("{}.lifetime", RUST)),
                ));
                i = lifetime_end;
                continue;
            }
            let j = scan_string(&chars, i + 1, '\'');
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.char_literal", RUST)),
            ));
            i = j;
            continue;
        }

        if c == 'b' && chars.get(i + 1) == Some(&'\'') {
            let j = scan_string(&chars, i + 2, '\'');
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.byte_literal", RUST)),
            ));
            i = j;
            continue;
        }

        if c.is_ascii_digit() {
            let j = scan_number(&chars, i);
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.numeric_literal", RUST)),
            ));
            i = j;
            continue;
        }

        if c == 'r'
            && chars.get(i + 1) == Some(&'#')
            && chars
                .get(i + 2)
                .map(|c| is_ident_start(*c))
                .unwrap_or(false)
        {
            let mut j = i + 2;
            while j < chars.len() && is_ident_continue(chars[j]) {
                j += 1;
            }
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.raw_ident", RUST)),
            ));
            i = j;
            continue;
        }

        if is_ident_start(c) {
            let mut j = i + 1;
            while j < chars.len() && is_ident_continue(chars[j]) {
                j += 1;
            }
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.ident", RUST)),
            ));
            i = j;
            continue;
        }

        out.push(CstNode::token(
            c.to_string(),
            Some(&format!("{}.punct", RUST)),
        ));
        i += 1;
    }

    out
}

fn scan_block_comment(chars: &[char], i: usize) -> usize {
    let mut j = i + 2;
    let mut depth = 1;
    while j < chars.len() && depth > 0 {
        if chars[j] == '/' && chars.get(j + 1) == Some(&'*') {
            depth += 1;
            j += 2;
        } else if chars[j] == '*' && chars.get(j + 1) == Some(&'/') {
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

fn scan_raw_or_prefixed_string(chars: &[char], i: usize) -> Option<usize> {
    let mut j = i;
    if chars.get(j) == Some(&'b') {
        j += 1;
    }
    if chars.get(j) == Some(&'r') {
        j += 1;
        let mut hashes = 0;
        while chars.get(j) == Some(&'#') {
            hashes += 1;
            j += 1;
        }
        if chars.get(j) != Some(&'"') {
            return None;
        }
        j += 1;
        let terminator: String =
            std::iter::once('"').chain(std::iter::repeat('#').take(hashes)).collect();
        // search for terminator
        let rest: String = chars[j..].iter().collect();
        match rest.find(&terminator) {
            Some(rel) => Some(j + rel + terminator.chars().count()),
            None => Some(chars.len()),
        }
    } else if chars.get(j) == Some(&'"') {
        Some(scan_string(chars, j + 1, '"'))
    } else {
        None
    }
}

fn scan_lifetime(chars: &[char], i: usize) -> usize {
    let mut j = i + 1;
    if j < chars.len() && is_ident_start(chars[j]) {
        j += 1;
        while j < chars.len() && is_ident_continue(chars[j]) {
            j += 1;
        }
        if chars.get(j) == Some(&'\'') {
            return i;
        }
        return j;
    }
    i
}

fn scan_number(chars: &[char], i: usize) -> usize {
    let mut j = i;
    if chars.get(j) == Some(&'0') && matches!(chars.get(j + 1), Some('x') | Some('X')) {
        j += 2;
        while j < chars.len() && (chars[j].is_ascii_hexdigit() || chars[j] == '_') {
            j += 1;
        }
    } else if chars.get(j) == Some(&'0') && matches!(chars.get(j + 1), Some('o') | Some('O')) {
        j += 2;
        while j < chars.len() && matches!(chars[j], '0'..='7' | '_') {
            j += 1;
        }
    } else if chars.get(j) == Some(&'0') && matches!(chars.get(j + 1), Some('b') | Some('B')) {
        j += 2;
        while j < chars.len() && matches!(chars[j], '0' | '1' | '_') {
            j += 1;
        }
    } else {
        while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '_') {
            j += 1;
        }
        if chars.get(j) == Some(&'.')
            && chars.get(j + 1).map(|c| c.is_ascii_digit()).unwrap_or(false)
        {
            j += 1;
            while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '_') {
                j += 1;
            }
        }
        if matches!(chars.get(j), Some('e') | Some('E')) {
            j += 1;
            if matches!(chars.get(j), Some('+') | Some('-')) {
                j += 1;
            }
            while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == '_') {
                j += 1;
            }
        }
    }
    if j < chars.len() && is_ident_start(chars[j]) {
        while j < chars.len() && is_ident_continue(chars[j]) {
            j += 1;
        }
    }
    j
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c.is_ascii_alphabetic() || (c as u32) > 0x7F
}

fn is_ident_continue(c: char) -> bool {
    c == '_' || c.is_ascii_alphanumeric() || (c as u32) > 0x7F
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(src: &str) {
        let node = parse_rust(src);
        let back = print_rust(&node);
        assert_eq!(back, src, "round-trip mismatch");
    }

    #[test]
    fn empty_string() {
        rt("");
    }

    #[test]
    fn simple_fn() {
        rt("fn main() {}\n");
    }

    #[test]
    fn line_and_block_comments() {
        rt("// hi\nfn f() {\n    /* mid */ 1\n}\n");
    }

    #[test]
    fn raw_and_byte_strings() {
        rt("let s = r#\"raw\"#;\nlet b = b\"abc\";\n");
    }

    #[test]
    fn lifetime_vs_char() {
        rt("let c = 'a';\nlet lt: &'static str = \"x\";\n");
    }

    #[test]
    fn numeric_with_suffix() {
        rt("let n = 0xFF_FFu32;\nlet f = 3.14e10_f64;\n");
    }

    #[test]
    fn raw_ident() {
        rt("let r#match = 1;\n");
    }
}
