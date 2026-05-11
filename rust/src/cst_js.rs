//! JavaScript ↔ `.lino` CST converter (issue #138).
//!
//! Token-level lossless converter for JavaScript source. Produces a
//! `lino-cst.js.*` flat CST whose round-trip is byte-faithful:
//! `print_js(&parse_js(src)) == src`. Mirrors `js/src/cst-js.mjs`
//! line for line.

use crate::cst::{dialects::JS, print_cst, CstNode};

/// Parse JavaScript source into a `lino-cst.js.*` CST.
pub fn parse_js(src: &str) -> CstNode {
    let children = tokenise(src);
    CstNode::list(format!("{}.program", JS), children)
}

/// Print a JS CST back to source.
pub fn print_js(node: &CstNode) -> String {
    print_cst(node)
}

#[derive(Clone, Copy)]
enum LastKind {
    None,
    Trivia,
    Ident,
    Punct,
    Other,
}

fn tokenise(src: &str) -> Vec<CstNode> {
    let chars: Vec<char> = src.chars().collect();
    let mut out: Vec<CstNode> = Vec::new();
    let mut i = 0usize;
    let mut last_kind = LastKind::None;
    let mut last_text: String = String::new();

    if chars.len() >= 2 && chars[0] == '#' && chars[1] == '!' {
        let mut j = 0usize;
        while j < chars.len() && chars[j] != '\n' {
            j += 1;
        }
        out.push(CstNode::trivia(
            chars[..j].iter().collect::<String>(),
            Some(&format!("{}.hashbang", JS)),
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
                Some(&format!("{}.whitespace", JS)),
            ));
            i = j;
            last_kind = LastKind::Trivia;
            continue;
        }

        if c == '/' && chars.get(i + 1) == Some(&'/') {
            let mut j = i + 2;
            while j < chars.len() && chars[j] != '\n' {
                j += 1;
            }
            out.push(CstNode::trivia(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.comment.line", JS)),
            ));
            i = j;
            last_kind = LastKind::Trivia;
            continue;
        }

        if c == '/' && chars.get(i + 1) == Some(&'*') {
            let j = scan_block_comment(&chars, i);
            out.push(CstNode::trivia(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.comment.block", JS)),
            ));
            i = j;
            last_kind = LastKind::Trivia;
            continue;
        }

        if c == '"' || c == '\'' {
            let j = scan_string(&chars, i + 1, c);
            let text: String = chars[i..j].iter().collect();
            out.push(CstNode::token(
                text.clone(),
                Some(&format!("{}.string_literal", JS)),
            ));
            i = j;
            last_kind = LastKind::Other;
            last_text = text;
            continue;
        }

        if c == '`' {
            let j = scan_template(&chars, i);
            let text: String = chars[i..j].iter().collect();
            out.push(CstNode::token(
                text.clone(),
                Some(&format!("{}.template_literal", JS)),
            ));
            i = j;
            last_kind = LastKind::Other;
            last_text = text;
            continue;
        }

        if c == '/' && can_be_regex(last_kind, &last_text) {
            let j = scan_regex(&chars, i);
            if j > i + 1 {
                let text: String = chars[i..j].iter().collect();
                out.push(CstNode::token(
                    text.clone(),
                    Some(&format!("{}.regexp_literal", JS)),
                ));
                i = j;
                last_kind = LastKind::Other;
                last_text = text;
                continue;
            }
        }

        if c.is_ascii_digit()
            || (c == '.' && chars.get(i + 1).map(|x| x.is_ascii_digit()).unwrap_or(false))
        {
            let j = scan_number(&chars, i);
            let text: String = chars[i..j].iter().collect();
            out.push(CstNode::token(
                text.clone(),
                Some(&format!("{}.numeric_literal", JS)),
            ));
            i = j;
            last_kind = LastKind::Other;
            last_text = text;
            continue;
        }

        if is_ident_start(c) {
            let mut j = i + 1;
            while j < chars.len() && is_ident_continue(chars[j]) {
                j += 1;
            }
            let text: String = chars[i..j].iter().collect();
            out.push(CstNode::token(
                text.clone(),
                Some(&format!("{}.ident", JS)),
            ));
            i = j;
            last_kind = LastKind::Ident;
            last_text = text;
            continue;
        }

        let text = c.to_string();
        out.push(CstNode::token(
            text.clone(),
            Some(&format!("{}.punct", JS)),
        ));
        i += 1;
        last_kind = LastKind::Punct;
        last_text = text;
    }

    out
}

fn scan_block_comment(chars: &[char], i: usize) -> usize {
    let mut j = i + 2;
    while j < chars.len() {
        if chars[j] == '*' && chars.get(j + 1) == Some(&'/') {
            return j + 2;
        }
        j += 1;
    }
    chars.len()
}

fn scan_string(chars: &[char], mut j: usize, quote: char) -> usize {
    while j < chars.len() {
        let c = chars[j];
        if c == '\\' {
            j += 2;
            continue;
        }
        if c == '\n' && (quote == '"' || quote == '\'') {
            return j;
        }
        if c == quote {
            return j + 1;
        }
        j += 1;
    }
    j
}

fn scan_template(chars: &[char], i: usize) -> usize {
    let mut j = i + 1;
    while j < chars.len() {
        let c = chars[j];
        if c == '\\' {
            j += 2;
            continue;
        }
        if c == '`' {
            return j + 1;
        }
        if c == '$' && chars.get(j + 1) == Some(&'{') {
            j += 2;
            let mut depth = 1;
            while j < chars.len() && depth > 0 {
                let k = chars[j];
                if k == '{' {
                    depth += 1;
                } else if k == '}' {
                    depth -= 1;
                } else if k == '"' || k == '\'' {
                    j = scan_string(chars, j + 1, k).saturating_sub(1);
                } else if k == '`' {
                    j = scan_template(chars, j).saturating_sub(1);
                } else if k == '/' && chars.get(j + 1) == Some(&'/') {
                    while j < chars.len() && chars[j] != '\n' {
                        j += 1;
                    }
                    continue;
                } else if k == '/' && chars.get(j + 1) == Some(&'*') {
                    j = scan_block_comment(chars, j);
                    continue;
                }
                j += 1;
            }
            continue;
        }
        j += 1;
    }
    j
}

fn scan_regex(chars: &[char], i: usize) -> usize {
    let mut j = i + 1;
    let mut in_class = false;
    while j < chars.len() {
        let c = chars[j];
        if c == '\\' {
            j += 2;
            continue;
        }
        if c == '[' {
            in_class = true;
        } else if c == ']' {
            in_class = false;
        } else if c == '/' && !in_class {
            j += 1;
            while j < chars.len() && chars[j].is_ascii_alphabetic() {
                j += 1;
            }
            return j;
        } else if c == '\n' {
            return i + 1;
        }
        j += 1;
    }
    j
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
        if chars.get(j) == Some(&'.') {
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
    if chars.get(j) == Some(&'n') {
        j += 1;
    }
    j
}

const REGEX_PRECEDING_KEYWORDS: &[&str] = &[
    "return",
    "typeof",
    "instanceof",
    "in",
    "of",
    "do",
    "else",
    "throw",
    "new",
    "delete",
    "void",
    "await",
    "yield",
    "case",
];

fn can_be_regex(last_kind: LastKind, last_text: &str) -> bool {
    match last_kind {
        LastKind::None | LastKind::Trivia => true,
        LastKind::Ident => REGEX_PRECEDING_KEYWORDS.iter().any(|k| *k == last_text),
        LastKind::Punct => !(last_text == ")" || last_text == "]"),
        LastKind::Other => false,
    }
}

fn is_ident_start(c: char) -> bool {
    c == '_' || c == '$' || c.is_ascii_alphabetic() || (c as u32) > 0x7F
}

fn is_ident_continue(c: char) -> bool {
    c == '_' || c == '$' || c.is_ascii_alphanumeric() || (c as u32) > 0x7F
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(src: &str) {
        let node = parse_js(src);
        let back = print_js(&node);
        assert_eq!(back, src, "round-trip mismatch");
    }

    #[test]
    fn empty_string() {
        rt("");
    }

    #[test]
    fn simple_const() {
        rt("const x = 1;\n");
    }

    #[test]
    fn comments_and_strings() {
        rt("// hi\n/* block */ const s = 'a\\'b';\n");
    }

    #[test]
    fn template_with_nested_expr() {
        rt("const t = `nested ${`inner ${1+2}`} done`;\n");
    }

    #[test]
    fn regex_after_assignment() {
        rt("const r = /foo\\/bar/g;\n");
    }

    #[test]
    fn divide_after_paren_is_not_regex() {
        rt("const x = (1)/2;\n");
    }

    #[test]
    fn bigint_and_hex() {
        rt("const n = 0xff_ff;\nconst b = 0b1010n;\nconst f = 3.14e-2;\n");
    }

    #[test]
    fn hashbang_preserved() {
        rt("#!/usr/bin/env node\nconsole.log(\"hi\");\n");
    }
}
