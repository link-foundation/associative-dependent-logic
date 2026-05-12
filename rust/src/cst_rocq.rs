//! Rocq ↔ `.lino` CST converter (issue #138).
//!
//! Token-level lossless converter for Rocq (formerly Coq) source. Produces a
//! `lino-cst.rocq.*` flat CST whose round-trip is byte-faithful:
//! `print_rocq(&parse_rocq(src)) == src`. Mirrors `js/src/cst-rocq.mjs`
//! line for line.

use crate::cst::{dialects::ROCQ, print_cst, CstNode};

/// Parse Rocq source into a `lino-cst.rocq.*` CST.
pub fn parse_rocq(src: &str) -> CstNode {
    let children = tokenise(src);
    CstNode::list(format!("{}.document", ROCQ), children)
}

/// Print a Rocq CST back to source.
pub fn print_rocq(node: &CstNode) -> String {
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
                Some(&format!("{}.whitespace", ROCQ)),
            ));
            i = j;
            continue;
        }

        if c == '(' && chars.get(i + 1) == Some(&'*') {
            let j = scan_block_comment(&chars, i);
            out.push(CstNode::trivia(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.comment", ROCQ)),
            ));
            i = j;
            continue;
        }

        if c == '"' {
            let j = scan_rocq_string(&chars, i + 1);
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.string_literal", ROCQ)),
            ));
            i = j;
            continue;
        }

        if c.is_ascii_digit() {
            let j = scan_number(&chars, i);
            out.push(CstNode::token(
                chars[i..j].iter().collect::<String>(),
                Some(&format!("{}.numeric_literal", ROCQ)),
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
                Some(&format!("{}.ident", ROCQ)),
            ));
            i = j;
            continue;
        }

        out.push(CstNode::token(
            c.to_string(),
            Some(&format!("{}.punct", ROCQ)),
        ));
        i += 1;
    }

    out
}

fn scan_block_comment(chars: &[char], i: usize) -> usize {
    let mut j = i + 2;
    let mut depth = 1;
    while j < chars.len() && depth > 0 {
        if chars[j] == '(' && chars.get(j + 1) == Some(&'*') {
            depth += 1;
            j += 2;
        } else if chars[j] == '*' && chars.get(j + 1) == Some(&')') {
            depth -= 1;
            j += 2;
        } else {
            j += 1;
        }
    }
    j
}

fn scan_rocq_string(chars: &[char], mut j: usize) -> usize {
    while j < chars.len() {
        if chars[j] == '"' {
            if chars.get(j + 1) == Some(&'"') {
                j += 2;
                continue;
            }
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
    j
}

fn is_ident_start(c: char) -> bool {
    if c == '_' || c.is_ascii_alphabetic() {
        return true;
    }
    if (c as u32) > 0x7F {
        return !is_rocq_punct_char(c);
    }
    false
}

fn is_ident_continue(c: char) -> bool {
    if c == '_' || c == '\'' || c.is_ascii_alphanumeric() {
        return true;
    }
    if (c as u32) > 0x7F {
        return !is_rocq_punct_char(c);
    }
    false
}

fn is_rocq_punct_char(c: char) -> bool {
    matches!(c, '→' | '←' | '⟨' | '⟩' | '∀' | '∃' | '∧' | '∨' | '¬')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt(src: &str) {
        let node = parse_rocq(src);
        let back = print_rocq(&node);
        assert_eq!(back, src, "round-trip mismatch");
    }

    #[test]
    fn empty_string() {
        rt("");
    }

    #[test]
    fn simple_def() {
        rt("Definition x := 1.\n");
    }

    #[test]
    fn nested_block_comment() {
        rt("(* a (* b *) c *)\nDefinition x := 1.\n");
    }

    #[test]
    fn doubled_quote_string() {
        rt("Definition s := \"hello, \"\"world\"\"\".\n");
    }

    #[test]
    fn proof_block() {
        rt("Theorem t : 1 + 1 = 2. Proof. reflexivity. Qed.\n");
    }

    #[test]
    fn hex_number() {
        rt("Definition n := 0xff.\n");
    }

    #[test]
    fn require_import() {
        rt("Require Import Coq.Lists.List.\n");
    }
}
