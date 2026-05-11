//! Universal lossless CST infrastructure for issue #138.
//!
//! This module mirrors `js/src/cst.mjs`: three node kinds (`list`, `token`,
//! `trivia`), a content-agnostic round-trip printer, and `.lino`
//! serialisation/deserialisation helpers. Used by `cst_rust`, `cst_js`,
//! `cst_lean` and `cst_rocq` to express their token streams.

use std::fmt::Write;

/// The four host-language dialect tag prefixes plus the shared dialect.
pub mod dialects {
    pub const RUST: &str = "lino-cst.rust";
    pub const JS: &str = "lino-cst.js";
    pub const LEAN: &str = "lino-cst.lean";
    pub const ROCQ: &str = "lino-cst.rocq";
    pub const SHARED: &str = "lino-cst.shared";
}

/// A CST node. Three kinds: `List` (with optional `open`/`close` delimiters),
/// `Token` (significant lexeme), `Trivia` (whitespace or comment).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CstNode {
    /// A list node with an optional dialect tag and optional open/close
    /// delimiter strings (e.g. `(`, `)`).
    List {
        tag: Option<String>,
        open: Option<String>,
        close: Option<String>,
        children: Vec<CstNode>,
    },
    /// A non-trivia lexeme. `text` holds the original source bytes.
    Token { tag: Option<String>, text: String },
    /// Whitespace or comment trivia. `text` holds the original source bytes.
    Trivia { tag: Option<String>, text: String },
}

impl CstNode {
    /// Construct a `List` node.
    pub fn list(tag: impl Into<String>, children: Vec<CstNode>) -> Self {
        CstNode::List {
            tag: Some(tag.into()),
            open: None,
            close: None,
            children,
        }
    }

    /// Construct an untagged `List` node with custom open/close delimiters.
    pub fn list_with_delims(
        tag: Option<String>,
        open: Option<String>,
        close: Option<String>,
        children: Vec<CstNode>,
    ) -> Self {
        CstNode::List {
            tag,
            open,
            close,
            children,
        }
    }

    /// Construct a `Token` leaf.
    pub fn token(text: impl Into<String>, tag: Option<&str>) -> Self {
        CstNode::Token {
            tag: tag.map(String::from),
            text: text.into(),
        }
    }

    /// Construct a `Trivia` leaf.
    pub fn trivia(text: impl Into<String>, tag: Option<&str>) -> Self {
        CstNode::Trivia {
            tag: tag.map(String::from),
            text: text.into(),
        }
    }

    /// True if this node is a list.
    pub fn is_list(&self) -> bool {
        matches!(self, CstNode::List { .. })
    }

    /// The dialect tag, if any.
    pub fn tag(&self) -> Option<&str> {
        match self {
            CstNode::List { tag, .. } => tag.as_deref(),
            CstNode::Token { tag, .. } => tag.as_deref(),
            CstNode::Trivia { tag, .. } => tag.as_deref(),
        }
    }

    /// The textual content of a leaf node. Returns `""` for lists.
    pub fn text(&self) -> &str {
        match self {
            CstNode::Token { text, .. } => text,
            CstNode::Trivia { text, .. } => text,
            CstNode::List { .. } => "",
        }
    }

    /// Iterate every leaf (token/trivia) of this subtree in document order.
    pub fn leaves(&self) -> Vec<&CstNode> {
        let mut out = Vec::new();
        collect_leaves(self, &mut out);
        out
    }
}

fn collect_leaves<'a>(node: &'a CstNode, out: &mut Vec<&'a CstNode>) {
    match node {
        CstNode::List { children, .. } => {
            for c in children {
                collect_leaves(c, out);
            }
        }
        _ => out.push(node),
    }
}

/// Print a CST node back to its byte-faithful source form.
pub fn print_cst(node: &CstNode) -> String {
    let mut out = String::new();
    emit(node, &mut out);
    out
}

fn emit(node: &CstNode, out: &mut String) {
    match node {
        CstNode::Token { text, .. } | CstNode::Trivia { text, .. } => {
            out.push_str(text);
        }
        CstNode::List {
            children,
            open,
            close,
            ..
        } => {
            if let Some(o) = open {
                out.push_str(o);
            }
            for c in children {
                emit(c, out);
            }
            if let Some(c) = close {
                out.push_str(c);
            }
        }
    }
}

/// Serialise a CST node into a `.lino` S-expression matching the format
/// produced by `js/src/cst.mjs`'s `cstToLino`.
pub fn cst_to_lino(node: &CstNode) -> String {
    let mut out = String::new();
    write_lino(node, &mut out);
    out
}

fn write_lino(node: &CstNode, out: &mut String) {
    match node {
        CstNode::Token { text, .. } => {
            let _ = write!(out, "(lino-cst.token {})", escape_text(text));
        }
        CstNode::Trivia { text, .. } => {
            let _ = write!(out, "(lino-cst.trivia {})", escape_text(text));
        }
        CstNode::List {
            tag,
            open,
            close,
            children,
        } => {
            out.push('(');
            out.push_str("lino-cst.list");
            if let Some(t) = tag {
                out.push(' ');
                out.push_str(t);
            }
            if let Some(o) = open {
                let _ = write!(out, " (open {})", escape_text(o));
            }
            if let Some(c) = close {
                let _ = write!(out, " (close {})", escape_text(c));
            }
            for c in children {
                out.push(' ');
                write_lino(c, out);
            }
            out.push(')');
        }
    }
}

fn escape_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Parse the `.lino` S-expression produced by `cst_to_lino` back into a
/// `CstNode`. Inverse of `cst_to_lino`.
pub fn lino_to_cst(src: &str) -> Result<CstNode, String> {
    let tokens = tokenise_lino_cst(src)?;
    let mut idx = 0usize;
    let node = parse_node(&tokens, &mut idx)?;
    Ok(node)
}

fn parse_node(tokens: &[String], idx: &mut usize) -> Result<CstNode, String> {
    expect(tokens, idx, "(")?;
    let head = eat(tokens, idx)?;
    match head.as_str() {
        "lino-cst.token" => {
            let lit = eat(tokens, idx)?;
            expect(tokens, idx, ")")?;
            Ok(CstNode::token(unescape_text(&lit)?, None))
        }
        "lino-cst.trivia" => {
            let lit = eat(tokens, idx)?;
            expect(tokens, idx, ")")?;
            Ok(CstNode::trivia(unescape_text(&lit)?, None))
        }
        "lino-cst.list" => {
            let mut tag: Option<String> = None;
            let mut open: Option<String> = None;
            let mut close: Option<String> = None;
            let mut children: Vec<CstNode> = Vec::new();
            while peek(tokens, *idx)? != ")" {
                if peek(tokens, *idx)? == "(" {
                    let lookahead = peek(tokens, *idx + 1)?;
                    if lookahead == "open" {
                        *idx += 2;
                        let lit = eat(tokens, idx)?;
                        expect(tokens, idx, ")")?;
                        open = Some(unescape_text(&lit)?);
                        continue;
                    }
                    if lookahead == "close" {
                        *idx += 2;
                        let lit = eat(tokens, idx)?;
                        expect(tokens, idx, ")")?;
                        close = Some(unescape_text(&lit)?);
                        continue;
                    }
                    children.push(parse_node(tokens, idx)?);
                } else {
                    if tag.is_some() {
                        return Err(format!("unexpected token {:?}", peek(tokens, *idx)?));
                    }
                    tag = Some(eat(tokens, idx)?);
                }
            }
            expect(tokens, idx, ")")?;
            Ok(CstNode::list_with_delims(tag, open, close, children))
        }
        other => Err(format!("unknown CST tag: {}", other)),
    }
}

fn peek(tokens: &[String], idx: usize) -> Result<&str, String> {
    tokens
        .get(idx)
        .map(String::as_str)
        .ok_or_else(|| "unexpected end of input".to_string())
}

fn eat(tokens: &[String], idx: &mut usize) -> Result<String, String> {
    let s = tokens
        .get(*idx)
        .cloned()
        .ok_or_else(|| "unexpected end of input".to_string())?;
    *idx += 1;
    Ok(s)
}

fn expect(tokens: &[String], idx: &mut usize, t: &str) -> Result<(), String> {
    let got = eat(tokens, idx)?;
    if got != t {
        return Err(format!("expected {:?}, got {:?}", t, got));
    }
    Ok(())
}

fn tokenise_lino_cst(src: &str) -> Result<Vec<String>, String> {
    let bytes: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if c == '(' || c == ')' {
            out.push(c.to_string());
            i += 1;
            continue;
        }
        if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
            i += 1;
            continue;
        }
        if c == '"' {
            let mut j = i + 1;
            while j < bytes.len() {
                if bytes[j] == '\\' {
                    j += 2;
                    continue;
                }
                if bytes[j] == '"' {
                    break;
                }
                j += 1;
            }
            if j >= bytes.len() {
                return Err("unterminated string literal".to_string());
            }
            out.push(bytes[i..=j].iter().collect());
            i = j + 1;
            continue;
        }
        let mut j = i;
        while j < bytes.len() && !" \t\n\r()".contains(bytes[j]) {
            j += 1;
        }
        out.push(bytes[i..j].iter().collect());
        i = j;
    }
    Ok(out)
}

fn unescape_text(literal: &str) -> Result<String, String> {
    let chars: Vec<char> = literal.chars().collect();
    if chars.first() != Some(&'"') || chars.last() != Some(&'"') {
        return Err(format!("not a string literal: {}", literal));
    }
    let mut out = String::new();
    let mut i = 1;
    while i < chars.len() - 1 {
        let c = chars[i];
        if c == '\\' {
            i += 1;
            if i >= chars.len() - 1 {
                return Err("trailing backslash".to_string());
            }
            let esc = chars[i];
            match esc {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\x08'),
                'f' => out.push('\x0C'),
                'u' => {
                    if i + 4 >= chars.len() {
                        return Err("bad \\u escape".to_string());
                    }
                    let hex: String = chars[i + 1..=i + 4].iter().collect();
                    let cp = u32::from_str_radix(&hex, 16)
                        .map_err(|e| format!("bad \\u escape: {}", e))?;
                    if let Some(ch) = char::from_u32(cp) {
                        out.push(ch);
                    }
                    i += 4;
                }
                other => return Err(format!("unknown escape: \\{}", other)),
            }
            i += 1;
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}

/// Convenience: return a clone of `node`. Provided for parity with the JS
/// module which exports `cloneCst` explicitly.
pub fn clone_cst(node: &CstNode) -> CstNode {
    node.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prints_list_with_open_close() {
        let n = CstNode::list_with_delims(
            Some("demo".into()),
            Some("(".into()),
            Some(")".into()),
            vec![CstNode::token("a", None), CstNode::token("b", None)],
        );
        assert_eq!(print_cst(&n), "(ab)");
    }

    #[test]
    fn lino_round_trip_for_token_trivia_list() {
        let n = CstNode::list(
            "lino-cst.rust.fn",
            vec![
                CstNode::token("fn ", Some("kw")),
                CstNode::token("foo", Some("ident")),
                CstNode::trivia(" ", Some("ws")),
            ],
        );
        let s = cst_to_lino(&n);
        let back = lino_to_cst(&s).unwrap();
        assert_eq!(print_cst(&n), print_cst(&back));
    }
}
