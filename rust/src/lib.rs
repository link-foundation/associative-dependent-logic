// RML — minimal relative meta-logic over LiNo (Links Notation)
// Supports many-valued logics from unary (1-valued) through continuous probabilistic (∞-valued).
// See: https://en.wikipedia.org/wiki/Many-valued_logic
//
// - Uses official links-notation crate to parse LiNo text into links
// - Terms are defined via (x: x is x)
// - Probabilities are assigned ONLY via: ((<expr>) has probability <p>)
// - Redefinable ops: (=: ...), (!=: not =), (and: avg|min|max|product|probabilistic_sum), (or: ...), (not: ...), (both: ...), (neither: ...)
// - Range: (range: 0 1) for [0,1] or (range: -1 1) for [-1,1] (balanced/symmetric)
// - Valence: (valence: N) to restrict truth values to N discrete levels (N=2 → Boolean, N=3 → ternary, etc.)
// - Query: (? <expr>)

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

// ========== Structured Diagnostics ==========
// Every parser/evaluator error is reported as a `Diagnostic` with an error
// code, human-readable message, and source span (file/line/col, 1-based).
// See `docs/DIAGNOSTICS.md` for the full code list.

/// A source span: 1-based `line`/`col`, optional file path, and a `length`
/// of the offending region (used to render carets in the CLI).
#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub file: Option<String>,
    pub line: usize,
    pub col: usize,
    pub length: usize,
}

impl Span {
    pub fn new(file: Option<String>, line: usize, col: usize, length: usize) -> Self {
        Self {
            file,
            line,
            col,
            length,
        }
    }

    pub fn unknown() -> Self {
        Self {
            file: None,
            line: 1,
            col: 1,
            length: 0,
        }
    }
}

/// A single diagnostic emitted by parser, evaluator, or type checker.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub span: Span,
}

impl Diagnostic {
    pub fn new(code: &str, message: impl Into<String>, span: Span) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
            span,
        }
    }
}

/// Result of `evaluate(src)`: a list of query results (numeric or type) plus
/// any diagnostics emitted while parsing/evaluating.
#[derive(Debug, Clone, Default)]
pub struct EvaluateResult {
    pub results: Vec<RunResult>,
    pub diagnostics: Vec<Diagnostic>,
}

/// Format a diagnostic for human-readable CLI output:
///     `<file>:<line>:<col>: <CODE>: <message>`
///         `<source line>`
///         `^`
pub fn format_diagnostic(diag: &Diagnostic, source: Option<&str>) -> String {
    let file = diag.span.file.as_deref().unwrap_or("<input>");
    let mut out = format!(
        "{}:{}:{}: {}: {}",
        file, diag.span.line, diag.span.col, diag.code, diag.message
    );
    if let Some(src) = source {
        let lines: Vec<&str> = src.split('\n').collect();
        if diag.span.line >= 1 && diag.span.line <= lines.len() {
            let line_text = lines[diag.span.line - 1];
            out.push('\n');
            out.push_str(line_text);
            out.push('\n');
            let pad = diag.span.col.saturating_sub(1);
            let caret_count = diag.span.length.max(1);
            out.push_str(&" ".repeat(pad));
            out.push_str(&"^".repeat(caret_count));
        }
    }
    out
}

/// Compute (line, col) source positions for every top-level link in `text`.
/// Mirrors `compute_form_spans` in the JavaScript implementation.
pub fn compute_form_spans(text: &str, file: Option<&str>) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut depth: i32 = 0;
    let mut line: usize = 1;
    let mut col: usize = 1;
    let mut pending_start: Option<(usize, usize)> = None;
    let mut in_line_comment = false;
    let mut line_start_idx: usize = 0;
    let bytes = text.as_bytes();
    for (off, &b) in bytes.iter().enumerate() {
        let ch = b as char;
        if ch == '\n' {
            in_line_comment = false;
            line += 1;
            col = 1;
            line_start_idx = off + 1;
            continue;
        }
        if in_line_comment {
            col += 1;
            continue;
        }
        if ch == '#' && depth == 0 {
            // Determine if the line so far contains only whitespace.
            let line_so_far = &text[line_start_idx..off];
            if line_so_far.chars().all(|c| c == ' ' || c == '\t') {
                in_line_comment = true;
                col += 1;
                continue;
            }
        }
        if ch == '(' {
            if depth == 0 {
                pending_start = Some((line, col));
            }
            depth += 1;
        } else if ch == ')' {
            depth -= 1;
            if depth == 0 {
                if let Some((sl, sc)) = pending_start.take() {
                    spans.push(Span::new(file.map(|s| s.to_string()), sl, sc, 1));
                }
            }
        }
        col += 1;
    }
    spans
}

// ========== LiNo Parser ==========
// Uses the official links-notation crate for parsing LiNo text.
// See: https://github.com/link-foundation/links-notation

// Find the index of an inline comment marker `#` that follows a `)` plus
// whitespace, mirroring the JS regex `(\)[ \t]+)#.*$`.
fn inline_comment_index(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let mut last_close: Option<usize> = None;
    for (i, b) in bytes.iter().enumerate() {
        match *b {
            b')' => last_close = Some(i),
            b'#' => {
                if let Some(close_idx) = last_close {
                    let between = &line[close_idx + 1..i];
                    if !between.is_empty() && between.chars().all(|c| c == ' ' || c == '\t') {
                        return Some(i);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse LiNo text into a vector of link strings (each a top-level parenthesized expression).
pub fn parse_lino(text: &str) -> Vec<String> {
    // Strip both full-line and inline comments (# ...) before parsing —
    // the LiNo parser doesn't handle them and an inline comment containing a
    // colon would otherwise be misread as a binding.
    let stripped: String = text
        .lines()
        .map(|line| {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                String::new()
            } else if let Some(idx) = inline_comment_index(line) {
                line[..idx].trim_end().to_string()
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<String>>()
        .join("\n");

    // The links-notation crate treats blank lines as group separators,
    // so we split the input by blank lines and parse each segment separately.
    let mut all_links = Vec::new();
    for segment in stripped.split("\n\n") {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            continue;
        }
        match links_notation::parse_lino_to_links(trimmed) {
            Ok(links) => {
                for link in links {
                    all_links.push(link.to_string());
                }
            }
            Err(_) => {}
        }
    }
    all_links
}

// ========== AST ==========

/// AST node: either a leaf string or a list of child nodes.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Leaf(String),
    List(Vec<Node>),
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Leaf(s) => write!(f, "{}", s),
            Node::List(children) => {
                write!(f, "(")?;
                for (i, child) in children.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{}", child)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ========== Helpers ==========

/// Tokenize a single link string into tokens (parens and words).
pub fn tokenize_one(s: &str) -> Vec<String> {
    let mut s = s.to_string();

    // Strip inline comments (everything after #) but balance parens
    if let Some(comment_idx) = s.find('#') {
        s = s[..comment_idx].to_string();
        // Count unmatched opening parens and add closing parens to balance
        let mut depth: i32 = 0;
        for c in s.chars() {
            if c == '(' {
                depth += 1;
            } else if c == ')' {
                depth -= 1;
            }
        }
        while depth > 0 {
            s.push(')');
            depth -= 1;
        }
    }

    let mut out = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '(' || c == ')' {
            out.push(c.to_string());
            i += 1;
            continue;
        }
        let j_start = i;
        while i < chars.len() && !chars[i].is_whitespace() && chars[i] != '(' && chars[i] != ')' {
            i += 1;
        }
        out.push(chars[j_start..i].iter().collect());
    }
    out
}

/// Parse tokens into an AST node.
pub fn parse_one(tokens: &[String]) -> Result<Node, String> {
    let mut i = 0;

    fn read(tokens: &[String], i: &mut usize) -> Result<Node, String> {
        if *i >= tokens.len() || tokens[*i] != "(" {
            return Err("expected \"(\"".to_string());
        }
        *i += 1;
        let mut arr = Vec::new();
        while *i < tokens.len() && tokens[*i] != ")" {
            if tokens[*i] == "(" {
                arr.push(read(tokens, i)?);
            } else {
                arr.push(Node::Leaf(tokens[*i].clone()));
                *i += 1;
            }
        }
        if *i >= tokens.len() || tokens[*i] != ")" {
            return Err("expected \")\"".to_string());
        }
        *i += 1;
        Ok(Node::List(arr))
    }

    let ast = read(tokens, &mut i)?;
    if i != tokens.len() {
        return Err("extra tokens after link".to_string());
    }
    Ok(ast)
}

/// Check if a string is numeric (including negative).
pub fn is_num(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let s = if let Some(stripped) = s.strip_prefix('-') {
        stripped
    } else {
        s
    };
    if s.is_empty() {
        return false;
    }
    if let Some(rest) = s.strip_prefix('.') {
        // .digits
        !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
    } else {
        // digits or digits.digits
        let parts: Vec<&str> = s.splitn(2, '.').collect();
        if parts.is_empty() || !parts[0].chars().all(|c| c.is_ascii_digit()) || parts[0].is_empty()
        {
            return false;
        }
        if parts.len() == 2 {
            parts[1].chars().all(|c| c.is_ascii_digit())
        } else {
            true
        }
    }
}

/// Create a canonical key representation of a node.
pub fn key_of(node: &Node) -> String {
    match node {
        Node::Leaf(s) => s.clone(),
        Node::List(children) => {
            let inner: Vec<String> = children.iter().map(key_of).collect();
            format!("({})", inner.join(" "))
        }
    }
}

/// Check structural equality of two nodes.
pub fn is_structurally_same(a: &Node, b: &Node) -> bool {
    match (a, b) {
        (Node::Leaf(sa), Node::Leaf(sb)) => sa == sb,
        (Node::List(la), Node::List(lb)) => {
            la.len() == lb.len()
                && la
                    .iter()
                    .zip(lb.iter())
                    .all(|(x, y)| is_structurally_same(x, y))
        }
        _ => false,
    }
}

// ========== Decimal-precision arithmetic ==========
// Round to at most DECIMAL_PRECISION significant decimal places to eliminate
// IEEE-754 floating-point artefacts (e.g. 0.1+0.2 → 0.3, not 0.30000000000000004).
const DECIMAL_PRECISION: i32 = 12;

pub fn dec_round(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let factor = 10f64.powi(DECIMAL_PRECISION);
    (x * factor).round() / factor
}

// ========== Quantization ==========

/// Quantize a value to N discrete levels in range [lo, hi].
/// For N=2 (Boolean): levels are {lo, hi}
/// For N=3 (ternary): levels are {lo, mid, hi}
/// For N<2 (continuous/unary): no quantization
/// See: https://en.wikipedia.org/wiki/Many-valued_logic
pub fn quantize(x: f64, valence: u32, lo: f64, hi: f64) -> f64 {
    if valence < 2 {
        return x; // unary or continuous — no quantization
    }
    let step = (hi - lo) / (valence as f64 - 1.0);
    let level = ((x - lo) / step).round();
    let level = level.max(0.0).min(valence as f64 - 1.0);
    lo + level * step
}

// ========== Aggregator Types ==========

/// Supported aggregator types for AND/OR operators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Aggregator {
    Avg,
    Min,
    Max,
    Prod,
    Ps, // Probabilistic sum: 1 - ∏(1-xi)
}

impl Aggregator {
    pub fn apply(&self, xs: &[f64], lo: f64) -> f64 {
        if xs.is_empty() {
            return lo;
        }
        match self {
            Aggregator::Avg => xs.iter().sum::<f64>() / xs.len() as f64,
            Aggregator::Min => xs.iter().copied().fold(f64::INFINITY, f64::min),
            Aggregator::Max => xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            Aggregator::Prod => xs.iter().copied().fold(1.0, |a, b| a * b),
            Aggregator::Ps => 1.0 - xs.iter().copied().fold(1.0, |a, b| a * (1.0 - b)),
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "avg" => Some(Aggregator::Avg),
            "min" => Some(Aggregator::Min),
            "max" => Some(Aggregator::Max),
            "product" | "prod" => Some(Aggregator::Prod),
            "probabilistic_sum" | "ps" => Some(Aggregator::Ps),
            _ => None,
        }
    }
}

// ========== Operator ==========

/// Operator types supported by the environment.
#[derive(Debug, Clone)]
pub enum Op {
    /// Negation: mirrors around midpoint. not(x) = hi - (x - lo)
    Not,
    /// Aggregator-based operator (for and/or).
    Agg(Aggregator),
    /// Equality operator: checks assigned probability or structural equality.
    Eq,
    /// Inequality: not(eq(...))
    Neq,
    /// Composition: outer(inner(...))
    Compose {
        outer: String,
        inner: String,
    },
    /// Arithmetic: +, -, *, / (decimal-precision)
    Add,
    Sub,
    Mul,
    Div,
}

// ========== Environment ==========

/// Options for creating an Env.
#[derive(Debug, Clone)]
pub struct EnvOptions {
    pub lo: f64,
    pub hi: f64,
    pub valence: u32,
}

impl Default for EnvOptions {
    fn default() -> Self {
        Self {
            lo: 0.0,
            hi: 1.0,
            valence: 0,
        }
    }
}

/// A stored lambda definition (param name, param type, body).
#[derive(Debug, Clone)]
pub struct Lambda {
    pub param: String,
    pub param_type: String,
    pub body: Node,
}

/// The evaluation environment: holds terms, assignments, operators, and range/valence config.
pub struct Env {
    pub terms: HashSet<String>,
    pub assign: HashMap<String, f64>,
    pub symbol_prob: HashMap<String, f64>,
    pub lo: f64,
    pub hi: f64,
    pub valence: u32,
    pub ops: HashMap<String, Op>,
    pub types: HashMap<String, String>,
    pub lambdas: HashMap<String, Lambda>,
}

impl Env {
    pub fn new(options: Option<EnvOptions>) -> Self {
        let opts = options.unwrap_or_default();
        let mut ops = HashMap::new();
        ops.insert("not".to_string(), Op::Not);
        ops.insert("and".to_string(), Op::Agg(Aggregator::Avg));
        ops.insert("or".to_string(), Op::Agg(Aggregator::Max));
        // Belnap operators: AND-altering operators for four-valued logic
        // "both" (gullibility): avg — contradiction resolves to midpoint
        // "neither" (consensus): product — gap resolves to zero (no info propagates)
        // See: https://en.wikipedia.org/wiki/Four-valued_logic#Belnap
        ops.insert("both".to_string(), Op::Agg(Aggregator::Avg));
        ops.insert("neither".to_string(), Op::Agg(Aggregator::Prod));
        ops.insert("=".to_string(), Op::Eq);
        ops.insert("!=".to_string(), Op::Neq);
        ops.insert("+".to_string(), Op::Add);
        ops.insert("-".to_string(), Op::Sub);
        ops.insert("*".to_string(), Op::Mul);
        ops.insert("/".to_string(), Op::Div);

        let mut env = Self {
            terms: HashSet::new(),
            assign: HashMap::new(),
            symbol_prob: HashMap::new(),
            lo: opts.lo,
            hi: opts.hi,
            valence: opts.valence,
            ops,
            types: HashMap::new(),
            lambdas: HashMap::new(),
        };

        // Initialize truth constants: true, false, unknown, undefined
        // These are predefined symbol probabilities based on the current range.
        // By default: (false: min(range)), (true: max(range)),
        //             (unknown: mid(range)), (undefined: mid(range))
        // They can be redefined by the user via (true: <value>), (false: <value>), etc.
        env.init_truth_constants();
        env
    }

    /// Midpoint of the range.
    pub fn mid(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    /// Initialize truth constants based on current range.
    /// (false: min(range)), (true: max(range)),
    /// (unknown: mid(range)), (undefined: mid(range))
    pub fn init_truth_constants(&mut self) {
        self.symbol_prob.insert("true".to_string(), self.hi);
        self.symbol_prob.insert("false".to_string(), self.lo);
        let mid = self.mid();
        self.symbol_prob.insert("unknown".to_string(), mid);
        self.symbol_prob.insert("undefined".to_string(), mid);
        // Note: "both" and "neither" are operators (not constants) — see Env::new()
        // See: https://en.wikipedia.org/wiki/Four-valued_logic#Belnap
    }

    /// Clamp and optionally quantize a value to the valid range.
    pub fn clamp(&self, x: f64) -> f64 {
        let clamped = x.max(self.lo).min(self.hi);
        if self.valence >= 2 {
            quantize(clamped, self.valence, self.lo, self.hi)
        } else {
            clamped
        }
    }

    /// Parse a numeric string respecting current range.
    pub fn to_num(&self, s: &str) -> f64 {
        self.clamp(s.parse::<f64>().unwrap_or(0.0))
    }

    pub fn define_op(&mut self, name: &str, op: Op) {
        self.ops.insert(name.to_string(), op);
    }

    pub fn get_op(&self, name: &str) -> Option<&Op> {
        self.ops.get(name)
    }

    pub fn set_expr_prob(&mut self, expr_node: &Node, p: f64) {
        self.assign.insert(key_of(expr_node), self.clamp(p));
    }

    pub fn set_symbol_prob(&mut self, sym: &str, p: f64) {
        self.symbol_prob.insert(sym.to_string(), self.clamp(p));
    }

    pub fn get_symbol_prob(&self, sym: &str) -> f64 {
        self.symbol_prob
            .get(sym)
            .copied()
            .unwrap_or_else(|| self.mid())
    }

    pub fn set_type(&mut self, expr: &str, type_expr: &str) {
        self.types.insert(expr.to_string(), type_expr.to_string());
    }

    pub fn get_type(&self, expr: &str) -> Option<&String> {
        self.types.get(expr)
    }

    pub fn set_lambda(&mut self, name: &str, lambda: Lambda) {
        self.lambdas.insert(name.to_string(), lambda);
    }

    pub fn get_lambda(&self, name: &str) -> Option<&Lambda> {
        self.lambdas.get(name)
    }

    /// Apply an operator by name to the given values.
    pub fn apply_op(&self, name: &str, vals: &[f64]) -> f64 {
        let op = match self.ops.get(name) {
            Some(op) => op.clone(),
            None => panic!("Unknown op: {}", name),
        };
        match op {
            Op::Not => {
                if vals.is_empty() {
                    self.lo
                } else {
                    self.hi - (vals[0] - self.lo)
                }
            }
            Op::Agg(agg) => dec_round(agg.apply(vals, self.lo)),
            Op::Eq | Op::Neq => self.lo,
            Op::Compose {
                ref outer,
                ref inner,
            } => {
                let inner_result = self.apply_op(inner, vals);
                self.apply_op(outer, &[inner_result])
            }
            Op::Add => {
                if vals.len() >= 2 {
                    dec_round(vals[0] + vals[1])
                } else {
                    0.0
                }
            }
            Op::Sub => {
                if vals.len() >= 2 {
                    dec_round(vals[0] - vals[1])
                } else {
                    0.0
                }
            }
            Op::Mul => {
                if vals.len() >= 2 {
                    dec_round(vals[0] * vals[1])
                } else {
                    0.0
                }
            }
            Op::Div => {
                if vals.len() >= 2 && vals[1] != 0.0 {
                    dec_round(vals[0] / vals[1])
                } else {
                    0.0
                }
            }
        }
    }

    /// Apply equality operator, checking assigned probabilities first.
    pub fn apply_eq(&self, left: &Node, right: &Node) -> f64 {
        // Check prefix form: (= L R)
        let k_prefix = key_of(&Node::List(vec![
            Node::Leaf("=".to_string()),
            left.clone(),
            right.clone(),
        ]));
        if let Some(&v) = self.assign.get(&k_prefix) {
            return v;
        }
        // Check infix form: (L = R)
        let k_infix = key_of(&Node::List(vec![
            left.clone(),
            Node::Leaf("=".to_string()),
            right.clone(),
        ]));
        if let Some(&v) = self.assign.get(&k_infix) {
            return v;
        }
        // Default: syntactic equality
        if is_structurally_same(left, right) {
            self.hi
        } else {
            self.lo
        }
    }

    /// Apply inequality operator: not(eq(L, R))
    pub fn apply_neq(&self, left: &Node, right: &Node) -> f64 {
        let eq_val = self.apply_eq(left, right);
        self.apply_op("not", &[eq_val])
    }

    /// Reinitialize ops when range changes (resets to defaults for current range).
    pub fn reinit_ops(&mut self) {
        self.ops.insert("not".to_string(), Op::Not);
        self.ops.insert("and".to_string(), Op::Agg(Aggregator::Avg));
        self.ops.insert("or".to_string(), Op::Agg(Aggregator::Max));
        self.ops
            .insert("both".to_string(), Op::Agg(Aggregator::Avg));
        self.ops
            .insert("neither".to_string(), Op::Agg(Aggregator::Prod));
        self.ops.insert("=".to_string(), Op::Eq);
        self.ops.insert("!=".to_string(), Op::Neq);
        self.ops.insert("+".to_string(), Op::Add);
        self.ops.insert("-".to_string(), Op::Sub);
        self.ops.insert("*".to_string(), Op::Mul);
        self.ops.insert("/".to_string(), Op::Div);
        // Re-initialize truth constants for new range
        self.init_truth_constants();
    }
}

// ========== Query Result ==========

/// Result of evaluating an expression: either a plain value or a query result.
#[derive(Debug, Clone)]
pub enum EvalResult {
    Value(f64),
    Query(f64),
    TypeQuery(String),
}

impl EvalResult {
    pub fn as_f64(&self) -> f64 {
        match self {
            EvalResult::Value(v) | EvalResult::Query(v) => *v,
            EvalResult::TypeQuery(_) => 0.0,
        }
    }

    pub fn is_query(&self) -> bool {
        matches!(self, EvalResult::Query(_) | EvalResult::TypeQuery(_))
    }

    pub fn is_type_query(&self) -> bool {
        matches!(self, EvalResult::TypeQuery(_))
    }

    pub fn type_string(&self) -> Option<&str> {
        match self {
            EvalResult::TypeQuery(s) => Some(s),
            _ => None,
        }
    }
}

// ========== Binding Parser ==========

/// Parse a binding form in two supported syntaxes:
/// 1. Colon form: (x: A) as ["x:", A] — standard LiNo link definition syntax
/// 2. Prefix type form: (A x) as ["A", "x"] — type-first notation for lambda/Pi bindings
///    e.g. (Natural x), used in (lambda (Natural x) body)
/// Returns (param_name, param_type_key) or None.
pub fn parse_binding(binding: &Node) -> Option<(String, String)> {
    if let Node::List(children) = binding {
        if children.len() == 2 {
            // Colon form: ["x:", A]
            if let Node::Leaf(ref s) = children[0] {
                if s.ends_with(':') {
                    let param_name = s[..s.len() - 1].to_string();
                    let param_type = match &children[1] {
                        Node::Leaf(s) => s.clone(),
                        other => key_of(other),
                    };
                    return Some((param_name, param_type));
                }
            }
            // Prefix type form: ["A", "x"] — type name first (must start with uppercase)
            if let (Node::Leaf(ref type_name), Node::Leaf(ref var_name)) =
                (&children[0], &children[1])
            {
                if type_name.starts_with(|c: char| c.is_uppercase()) && !var_name.ends_with(':') {
                    return Some((var_name.clone(), type_name.clone()));
                }
            }
        }
    }
    None
}

/// Parse comma-separated bindings: (Natural x, Natural y) → vec of (name, type) pairs.
/// Tokens arrive as ["Natural", "x,", "Natural", "y"] or ["Natural", "x"] (single binding).
pub fn parse_bindings(binding: &Node) -> Option<Vec<(String, String)>> {
    // Try single binding first
    if let Some(single) = parse_binding(binding) {
        return Some(vec![single]);
    }
    // Try comma-separated
    if let Node::List(children) = binding {
        let mut tokens: Vec<String> = Vec::new();
        for child in children {
            if let Node::Leaf(ref s) = child {
                if s.ends_with(',') {
                    tokens.push(s[..s.len() - 1].to_string());
                    tokens.push(",".to_string());
                } else {
                    tokens.push(s.clone());
                }
            } else {
                return None;
            }
        }
        let mut bindings = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if tokens[i] == "," {
                i += 1;
                continue;
            }
            if i + 1 < tokens.len() && tokens[i + 1] != "," {
                let type_name = &tokens[i];
                let var_name = &tokens[i + 1];
                if type_name.starts_with(|c: char| c.is_uppercase()) {
                    bindings.push((var_name.clone(), type_name.clone()));
                    i += 2;
                    continue;
                }
            }
            return None;
        }
        if !bindings.is_empty() {
            return Some(bindings);
        }
    }
    None
}

// ========== Substitution ==========

/// Substitute all occurrences of variable `name` with `replacement` in `expr`.
pub fn substitute(expr: &Node, name: &str, replacement: &Node) -> Node {
    match expr {
        Node::Leaf(s) => {
            if s == name {
                replacement.clone()
            } else {
                expr.clone()
            }
        }
        Node::List(children) => {
            // Don't substitute inside a binding that shadows the variable
            if children.len() == 3 {
                if let Node::Leaf(ref head) = children[0] {
                    if head == "lambda" || head == "Pi" {
                        if let Some((param, _)) = parse_binding(&children[1]) {
                            if param == name {
                                return expr.clone(); // shadowed
                            }
                        }
                    }
                }
            }
            Node::List(
                children
                    .iter()
                    .map(|child| substitute(child, name, replacement))
                    .collect(),
            )
        }
    }
}

// ========== Evaluator ==========

/// Evaluate a node in arithmetic context — numeric literals are NOT clamped to the logic range.
fn eval_arith(node: &Node, env: &mut Env) -> f64 {
    if let Node::Leaf(ref s) = node {
        if is_num(s) {
            return s.parse::<f64>().unwrap_or(0.0);
        }
    }
    eval_node(node, env).as_f64()
}

/// Evaluate an AST node in the given environment.
pub fn eval_node(node: &Node, env: &mut Env) -> EvalResult {
    match node {
        Node::Leaf(s) => {
            if is_num(s) {
                EvalResult::Value(env.to_num(s))
            } else {
                EvalResult::Value(env.get_symbol_prob(s))
            }
        }
        Node::List(children) => {
            if children.is_empty() {
                return EvalResult::Value(0.0);
            }

            // Definitions & operator redefs: (head: ...) form
            if let Node::Leaf(ref s) = children[0] {
                if s.ends_with(':') {
                    let head = &s[..s.len() - 1];
                    return define_form(head, &children[1..], env);
                }
            }

            // Note: (x : A) with spaces as a standalone colon separator is NOT supported.
            // Use (x: A) instead — the colon must be part of the link name.

            // Assignment: ((expr) has probability p)
            if children.len() == 4 {
                if let (Node::Leaf(ref w1), Node::Leaf(ref w2), Node::Leaf(ref w3)) =
                    (&children[1], &children[2], &children[3])
                {
                    if w1 == "has" && w2 == "probability" && is_num(w3) {
                        let p: f64 = w3.parse().unwrap_or(0.0);
                        env.set_expr_prob(&children[0], p);
                        return EvalResult::Value(env.to_num(w3));
                    }
                }
            }

            // Range configuration: (range lo hi) prefix form
            if children.len() == 3 {
                if let Node::Leaf(ref first) = children[0] {
                    if first == "range" {
                        if let (Node::Leaf(ref lo_s), Node::Leaf(ref hi_s)) =
                            (&children[1], &children[2])
                        {
                            if is_num(lo_s) && is_num(hi_s) {
                                env.lo = lo_s.parse().unwrap_or(0.0);
                                env.hi = hi_s.parse().unwrap_or(1.0);
                                env.reinit_ops();
                                return EvalResult::Value(1.0);
                            }
                        }
                    }
                }
            }

            // Valence configuration: (valence N) prefix form
            if children.len() == 2 {
                if let Node::Leaf(ref first) = children[0] {
                    if first == "valence" {
                        if let Node::Leaf(ref val_s) = children[1] {
                            if is_num(val_s) {
                                env.valence = val_s.parse::<f64>().unwrap_or(0.0) as u32;
                                return EvalResult::Value(1.0);
                            }
                        }
                    }
                }
            }

            // Query: (? expr)
            if let Node::Leaf(ref first) = children[0] {
                if first == "?" {
                    let result = eval_node(&children[1], env);
                    // If inner result is already a type query, pass it through
                    if result.is_type_query() {
                        return result;
                    }
                    let v = result.as_f64();
                    return EvalResult::Query(env.clamp(v));
                }
            }

            // Infix arithmetic: (A + B), (A - B), (A * B), (A / B)
            // Arithmetic uses raw numeric values (not clamped to the logic range)
            if children.len() == 3 {
                if let Node::Leaf(ref op_name) = children[1] {
                    if op_name == "+" || op_name == "-" || op_name == "*" || op_name == "/" {
                        let l = eval_arith(&children[0], env);
                        let r = eval_arith(&children[2], env);
                        return EvalResult::Value(env.apply_op(op_name, &[l, r]));
                    }
                }
            }

            // Infix AND/OR/BOTH/NEITHER: ((A) and (B)) / ((A) or (B)) / ((A) both (B)) / ((A) neither (B))
            if children.len() == 3 {
                if let Node::Leaf(ref op_name) = children[1] {
                    if op_name == "and"
                        || op_name == "or"
                        || op_name == "both"
                        || op_name == "neither"
                    {
                        let l = eval_node(&children[0], env).as_f64();
                        let r = eval_node(&children[2], env).as_f64();
                        return EvalResult::Value(env.clamp(env.apply_op(op_name, &[l, r])));
                    }
                }
            }

            // Composite natural language operators: (both A and B [and C ...]), (neither A nor B [nor C ...])
            if children.len() >= 4 && children.len() % 2 == 0 {
                if let Node::Leaf(ref head) = children[0] {
                    if head == "both" || head == "neither" {
                        let sep = if head == "both" { "and" } else { "nor" };
                        let mut valid = true;
                        for i in (2..children.len()).step_by(2) {
                            if let Node::Leaf(ref s) = children[i] {
                                if s != sep {
                                    valid = false;
                                    break;
                                }
                            } else {
                                valid = false;
                                break;
                            }
                        }
                        if valid {
                            let head_str = head.clone();
                            let vals: Vec<f64> = (1..children.len())
                                .step_by(2)
                                .map(|i| eval_node(&children[i], env).as_f64())
                                .collect();
                            return EvalResult::Value(env.clamp(env.apply_op(&head_str, &vals)));
                        }
                    }
                }
            }

            // Infix equality/inequality: (L = R), (L != R)
            if children.len() == 3 {
                if let Node::Leaf(ref op_name) = children[1] {
                    if op_name == "=" {
                        match env.ops.get("=").cloned() {
                            Some(Op::Compose { outer, inner }) => {
                                let inner_val = apply_named_op_on_nodes(
                                    env,
                                    &inner,
                                    &children[0],
                                    &children[2],
                                );
                                let outer_val = env.apply_op(&outer, &[inner_val]);
                                return EvalResult::Value(env.clamp(outer_val));
                            }
                            _ => {
                                let raw = env.apply_eq(&children[0], &children[2]);
                                // If there's an explicit assignment or structural match, trust it
                                let k_prefix = key_of(&Node::List(vec![
                                    Node::Leaf("=".to_string()),
                                    children[0].clone(),
                                    children[2].clone(),
                                ]));
                                let k_infix = key_of(&Node::List(vec![
                                    children[0].clone(),
                                    Node::Leaf("=".to_string()),
                                    children[2].clone(),
                                ]));
                                if env.assign.contains_key(&k_prefix)
                                    || env.assign.contains_key(&k_infix)
                                    || is_structurally_same(&children[0], &children[2])
                                {
                                    return EvalResult::Value(env.clamp(raw));
                                }
                                // No explicit assignment — try numeric comparison (decimal-precision)
                                let l = eval_arith(&children[0], env);
                                let r = eval_arith(&children[2], env);
                                let num_eq = if dec_round(l) == dec_round(r) {
                                    env.hi
                                } else {
                                    env.lo
                                };
                                return EvalResult::Value(env.clamp(num_eq));
                            }
                        }
                    }
                    if op_name == "!=" {
                        match env.ops.get("!=").cloned() {
                            Some(Op::Compose { outer, inner }) => {
                                let inner_val = apply_named_op_on_nodes(
                                    env,
                                    &inner,
                                    &children[0],
                                    &children[2],
                                );
                                let outer_val = env.apply_op(&outer, &[inner_val]);
                                return EvalResult::Value(env.clamp(outer_val));
                            }
                            _ => {
                                // Check explicit assignment or structural match first
                                let k_prefix = key_of(&Node::List(vec![
                                    Node::Leaf("=".to_string()),
                                    children[0].clone(),
                                    children[2].clone(),
                                ]));
                                let k_infix = key_of(&Node::List(vec![
                                    children[0].clone(),
                                    Node::Leaf("=".to_string()),
                                    children[2].clone(),
                                ]));
                                if env.assign.contains_key(&k_prefix)
                                    || env.assign.contains_key(&k_infix)
                                    || is_structurally_same(&children[0], &children[2])
                                {
                                    return EvalResult::Value(
                                        env.clamp(env.apply_neq(&children[0], &children[2])),
                                    );
                                }
                                // No explicit assignment — try numeric comparison
                                let l = eval_arith(&children[0], env);
                                let r = eval_arith(&children[2], env);
                                let num_eq = if dec_round(l) == dec_round(r) {
                                    env.hi
                                } else {
                                    env.lo
                                };
                                let neq = env.apply_op("not", &[num_eq]);
                                return EvalResult::Value(env.clamp(neq));
                            }
                        }
                    }
                }
            }

            // ---------- Type System: "everything is a link" ----------

            // Type universe: (Type N)
            if children.len() == 2 {
                if let Node::Leaf(ref first) = children[0] {
                    if first == "Type" {
                        if let Node::Leaf(ref level_s) = children[1] {
                            let level: i64 = level_s.parse().unwrap_or(0);
                            let key = key_of(&Node::List(children.clone()));
                            env.set_type(&key, &format!("(Type {})", level + 1));
                            return EvalResult::Value(1.0);
                        }
                    }
                }
            }

            // Prop: (Prop) sugar for (Type 0)
            if children.len() == 1 {
                if let Node::Leaf(ref first) = children[0] {
                    if first == "Prop" {
                        env.set_type("(Prop)", "(Type 1)");
                        return EvalResult::Value(1.0);
                    }
                }
            }

            // Dependent product (Pi-type): (Pi (x: A) B)
            if children.len() == 3 {
                if let Node::Leaf(ref first) = children[0] {
                    if first == "Pi" {
                        if let Some((param_name, param_type)) = parse_binding(&children[1]) {
                            env.terms.insert(param_name.clone());
                            env.set_type(&param_name, &param_type);
                            let key = key_of(&Node::List(children.clone()));
                            env.set_type(&key, "(Type 0)");
                        }
                        return EvalResult::Value(1.0);
                    }
                }
            }

            // Lambda abstraction: (lambda (A x) body) or (lambda (x: A) body)
            // Also supports multi-param: (lambda (A x, B y) body)
            if children.len() == 3 {
                if let Node::Leaf(ref first) = children[0] {
                    if first == "lambda" {
                        if let Some(bindings) = parse_bindings(&children[1]) {
                            if !bindings.is_empty() {
                                let (ref param_name, ref param_type) = bindings[0];
                                env.terms.insert(param_name.clone());
                                env.set_type(param_name, param_type);
                                // Register additional bindings
                                for binding in &bindings[1..] {
                                    env.terms.insert(binding.0.clone());
                                    env.set_type(&binding.0, &binding.1);
                                }
                                let body_key = key_of(&children[2]);
                                let body_type = env
                                    .get_type(&body_key)
                                    .cloned()
                                    .unwrap_or_else(|| "unknown".to_string());
                                let key = key_of(&Node::List(children.clone()));
                                env.set_type(
                                    &key,
                                    &format!("(Pi ({} {}) {})", param_type, param_name, body_type),
                                );
                            }
                        }
                        return EvalResult::Value(1.0);
                    }
                }
            }

            // Application: (apply f x) — explicit application with beta-reduction
            if children.len() == 3 {
                if let Node::Leaf(ref first) = children[0] {
                    if first == "apply" {
                        let fn_node = &children[1];
                        let arg = &children[2];

                        // Check if fn is a lambda: (lambda (A x) body)
                        if let Node::List(ref fn_children) = fn_node {
                            if fn_children.len() == 3 {
                                if let Node::Leaf(ref fn_head) = fn_children[0] {
                                    if fn_head == "lambda" {
                                        if let Some((param_name, _)) =
                                            parse_binding(&fn_children[1])
                                        {
                                            let body = &fn_children[2];
                                            let result = substitute(body, &param_name, arg);
                                            return eval_node(&result, env);
                                        }
                                    }
                                }
                            }
                        }

                        // Check if fn is a named lambda
                        if let Node::Leaf(ref fn_name) = fn_node {
                            if let Some(lambda) = env.get_lambda(fn_name).cloned() {
                                let result = substitute(&lambda.body, &lambda.param, arg);
                                return eval_node(&result, env);
                            }
                        }

                        // Otherwise evaluate both
                        let f_val = eval_node(fn_node, env).as_f64();
                        return EvalResult::Value(f_val);
                    }
                }
            }

            // Type query: (type of expr) — returns the type of an expression
            // e.g. (? (type of x)) → returns the type string
            if children.len() == 3 {
                if let (Node::Leaf(ref first), Node::Leaf(ref mid)) = (&children[0], &children[1]) {
                    if first == "type" && mid == "of" {
                        let expr_key = match &children[2] {
                            Node::Leaf(s) => s.clone(),
                            other => key_of(other),
                        };
                        let type_str = env
                            .get_type(&expr_key)
                            .cloned()
                            .unwrap_or_else(|| "unknown".to_string());
                        return EvalResult::TypeQuery(type_str);
                    }
                }
            }

            // Type check query: (expr of Type) — checks if expr has the given type
            // e.g. (? (x of Natural)) → returns 1 or 0
            if children.len() == 3 {
                if let Node::Leaf(ref mid) = children[1] {
                    if mid == "of" {
                        let expr_key = match &children[0] {
                            Node::Leaf(s) => s.clone(),
                            other => key_of(other),
                        };
                        let expected_key = match &children[2] {
                            Node::Leaf(s) => s.clone(),
                            other => key_of(other),
                        };
                        if let Some(actual) = env.get_type(&expr_key) {
                            return EvalResult::Value(if *actual == expected_key {
                                env.hi
                            } else {
                                env.lo
                            });
                        }
                        return EvalResult::Value(env.lo);
                    }
                }
            }

            // Prefix: (not X), (and X Y ...), (or X Y ...)
            if let Node::Leaf(ref head) = children[0] {
                let head_str = head.clone();
                if env.ops.contains_key(&head_str) {
                    let vals: Vec<f64> = children[1..]
                        .iter()
                        .map(|a| eval_node(a, env).as_f64())
                        .collect();
                    return EvalResult::Value(env.clamp(env.apply_op(&head_str, &vals)));
                }

                // Named lambda application: (name arg ...)
                if children.len() >= 2 {
                    if let Some(lambda) = env.get_lambda(&head_str).cloned() {
                        let result = substitute(&lambda.body, &lambda.param, &children[1]);
                        if children.len() == 2 {
                            return eval_node(&result, env);
                        }
                        // For now, just apply first argument
                        return eval_node(&result, env);
                    }
                }
            }

            EvalResult::Value(0.0)
        }
    }
}

/// Helper for applying a named op when dealing with node-based equality.
fn apply_named_op_on_nodes(env: &Env, op_name: &str, left: &Node, right: &Node) -> f64 {
    match env.ops.get(op_name) {
        Some(Op::Eq) => env.apply_eq(left, right),
        Some(Op::Neq) => env.apply_neq(left, right),
        _ => env.lo,
    }
}

/// Process definition forms: (head: rhs...)
fn define_form(head: &str, rhs: &[Node], env: &mut Env) -> EvalResult {
    // Term definition: (a: a is a) → declare 'a' as a term
    if rhs.len() == 3 {
        if let (Node::Leaf(ref r0), Node::Leaf(ref r1), Node::Leaf(ref r2)) =
            (&rhs[0], &rhs[1], &rhs[2])
        {
            if r1 == "is" && r0 == head && r2 == head {
                env.terms.insert(head.to_string());
                return EvalResult::Value(1.0);
            }
        }
    }

    // Prefix type notation: (name: TypeName name) → typed self-referential declaration
    // e.g. (zero: Natural zero), (boolean: Type boolean), (true: Boolean true)
    if rhs.len() == 2 {
        if let Node::Leaf(ref last) = rhs[1] {
            if last == head {
                match &rhs[0] {
                    Node::Leaf(ref type_name)
                        if type_name.starts_with(|c: char| c.is_uppercase()) =>
                    {
                        env.terms.insert(head.to_string());
                        env.types.insert(head.to_string(), type_name.clone());
                        return EvalResult::Value(1.0);
                    }
                    Node::List(_) => {
                        env.terms.insert(head.to_string());
                        let type_key = key_of(&rhs[0]);
                        env.types.insert(head.to_string(), type_key);
                        eval_node(&rhs[0], env);
                        return EvalResult::Value(1.0);
                    }
                    _ => {}
                }
            }
        }
    }

    // Range configuration: (range: lo hi)
    if head == "range" && rhs.len() == 2 {
        if let (Node::Leaf(ref lo_s), Node::Leaf(ref hi_s)) = (&rhs[0], &rhs[1]) {
            if is_num(lo_s) && is_num(hi_s) {
                env.lo = lo_s.parse().unwrap_or(0.0);
                env.hi = hi_s.parse().unwrap_or(1.0);
                env.reinit_ops();
                return EvalResult::Value(1.0);
            }
        }
    }

    // Valence configuration: (valence: N)
    if head == "valence" && rhs.len() == 1 {
        if let Node::Leaf(ref val_s) = rhs[0] {
            if is_num(val_s) {
                env.valence = val_s.parse::<f64>().unwrap_or(0.0) as u32;
                return EvalResult::Value(1.0);
            }
        }
    }

    // Optional symbol prior: (a: 0.7)
    if rhs.len() == 1 {
        if let Node::Leaf(ref val_s) = rhs[0] {
            if is_num(val_s) {
                let p: f64 = val_s.parse().unwrap_or(0.0);
                env.set_symbol_prob(head, p);
                return EvalResult::Value(env.to_num(val_s));
            }
        }
    }

    // Operator redefinitions
    let is_op_name = head == "="
        || head == "!="
        || head == "and"
        || head == "or"
        || head == "both"
        || head == "neither"
        || head == "not"
        || head == "is"
        || head == "?:"
        || head.contains('=')
        || head.contains('!');

    if is_op_name {
        // Composition like: (!=: not =) or (=: =) (no-op)
        if rhs.len() == 2 {
            if let (Node::Leaf(ref outer), Node::Leaf(ref inner)) = (&rhs[0], &rhs[1]) {
                if env.ops.contains_key(outer.as_str()) && env.ops.contains_key(inner.as_str()) {
                    env.define_op(
                        head,
                        Op::Compose {
                            outer: outer.clone(),
                            inner: inner.clone(),
                        },
                    );
                    return EvalResult::Value(1.0);
                }
                // Mirror JS behavior: surface a diagnostic for the missing op.
                if !env.ops.contains_key(outer.as_str()) {
                    panic!("Unknown op: {}", outer);
                }
                if !env.ops.contains_key(inner.as_str()) {
                    panic!("Unknown op: {}", inner);
                }
            }
        }

        // Aggregator selection: (and: avg|min|max|product|probabilistic_sum)
        if (head == "and" || head == "or" || head == "both" || head == "neither") && rhs.len() == 1
        {
            if let Node::Leaf(ref sel) = rhs[0] {
                if let Some(agg) = Aggregator::from_name(sel) {
                    env.define_op(head, Op::Agg(agg));
                    return EvalResult::Value(1.0);
                } else {
                    panic!("Unknown aggregator \"{}\"", sel);
                }
            }
        }
    }

    // Lambda definition: (name: lambda (A x) body)
    if rhs.len() >= 2 {
        if let Node::Leaf(ref first) = rhs[0] {
            if first == "lambda" && rhs.len() == 3 {
                if let Some((param_name, param_type)) = parse_binding(&rhs[1]) {
                    let body = rhs[2].clone();
                    env.terms.insert(head.to_string());
                    let body_key = key_of(&body);
                    let body_type =
                        env.get_type(&body_key)
                            .cloned()
                            .unwrap_or_else(|| match &body {
                                Node::Leaf(s) => s.clone(),
                                other => key_of(other),
                            });
                    env.set_type(
                        head,
                        &format!("(Pi ({} {}) {})", param_type, param_name, body_type),
                    );
                    env.set_lambda(
                        head,
                        Lambda {
                            param: param_name,
                            param_type,
                            body,
                        },
                    );
                    return EvalResult::Value(1.0);
                }
            }
        }
    }

    // Typed declaration with complex type expression: (succ: (Pi (Natural n) Natural))
    // Only complex expressions (arrays/lists) are accepted as type annotations in single-element form.
    // Simple name type annotations like (x: Natural) are NOT supported — use (x: Natural x) prefix form instead.
    if rhs.len() == 1 {
        let is_op = head == "="
            || head == "!="
            || head == "and"
            || head == "or"
            || head == "both"
            || head == "neither"
            || head == "not"
            || head == "is"
            || head == "?:"
            || head.contains('=')
            || head.contains('!');

        if !is_op {
            if let Node::List(_) = &rhs[0] {
                env.terms.insert(head.to_string());
                let type_key = key_of(&rhs[0]);
                env.set_type(head, &type_key);
                eval_node(&rhs[0], env);
                return EvalResult::Value(1.0);
            }
        }
    }

    // Generic symbol alias like (x: y) just copies y's prior probability if any
    if rhs.len() == 1 {
        if let Node::Leaf(ref sym) = rhs[0] {
            let prob = env.get_symbol_prob(sym);
            env.set_symbol_prob(head, prob);
            return EvalResult::Value(env.get_symbol_prob(head));
        }
    }

    // Else: ignore (keeps PoC minimal)
    EvalResult::Value(0.0)
}

// ========== Meta-expression Adapter ==========

/// Selected interpretation supplied by a consumer such as meta-expression.
#[derive(Debug, Clone, PartialEq)]
pub struct Interpretation {
    pub kind: String,
    pub expression: Option<String>,
    pub summary: Option<String>,
    pub lino: Option<String>,
}

impl Interpretation {
    pub fn arithmetic_equality(expression: &str) -> Self {
        Self {
            kind: "arithmetic-equality".to_string(),
            expression: Some(expression.to_string()),
            summary: None,
            lino: None,
        }
    }

    pub fn arithmetic_question(expression: &str) -> Self {
        Self {
            kind: "arithmetic-question".to_string(),
            expression: Some(expression.to_string()),
            summary: None,
            lino: None,
        }
    }

    pub fn real_world_claim(summary: &str) -> Self {
        Self {
            kind: "real-world-claim".to_string(),
            expression: None,
            summary: Some(summary.to_string()),
            lino: None,
        }
    }

    pub fn lino(expression: &str) -> Self {
        Self {
            kind: "lino".to_string(),
            expression: None,
            summary: None,
            lino: Some(expression.to_string()),
        }
    }
}

/// Explicit dependency record used to keep unsupported claims partial.
#[derive(Debug, Clone, PartialEq)]
pub struct Dependency {
    pub id: String,
    pub status: String,
    pub description: String,
}

impl Dependency {
    pub fn missing(id: &str, description: &str) -> Self {
        Self {
            id: id.to_string(),
            status: "missing".to_string(),
            description: description.to_string(),
        }
    }
}

/// Request object for `formalize_selected_interpretation`.
#[derive(Debug, Clone, PartialEq)]
pub struct FormalizationRequest {
    pub text: String,
    pub interpretation: Interpretation,
    pub formal_system: String,
    pub dependencies: Vec<Dependency>,
}

/// A dependency-aware RML formalization.
#[derive(Debug, Clone, PartialEq)]
pub struct Formalization {
    pub source_text: String,
    pub interpretation: Interpretation,
    pub formal_system: String,
    pub dependencies: Vec<Dependency>,
    pub computable: bool,
    pub formalization_level: u8,
    pub unknowns: Vec<String>,
    pub value_kind: String,
    pub ast: Option<Node>,
    pub lino: Option<String>,
}

/// Result value from evaluating a formalization.
#[derive(Debug, Clone, PartialEq)]
pub enum FormalizationResultValue {
    Number(f64),
    TruthValue(f64),
    Type(String),
    Partial(String),
}

/// Evaluation result for the meta-expression adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct FormalizationEvaluation {
    pub computable: bool,
    pub formalization_level: u8,
    pub unknowns: Vec<String>,
    pub result: FormalizationResultValue,
}

fn normalize_question_expression(text: &str) -> String {
    let mut out = text.trim().trim_end_matches('?').trim().to_string();
    let lower = out.to_lowercase();
    if lower.starts_with("what is ") {
        out = out[8..].trim().to_string();
    }
    out
}

fn split_top_level_equals(expression: &str) -> Option<(String, String)> {
    let mut depth: i32 = 0;
    let chars: Vec<char> = expression.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            '=' if depth == 0 => {
                if i > 0 && chars[i - 1] == '!' {
                    continue;
                }
                if i + 1 < chars.len() && chars[i + 1] == '=' {
                    continue;
                }
                let left: String = chars[..i].iter().collect();
                let right: String = chars[i + 1..].iter().collect();
                return Some((left.trim().to_string(), right.trim().to_string()));
            }
            _ => {}
        }
    }
    None
}

fn parse_expression_shape(expression: &str, unwrap_single: bool) -> Result<Node, String> {
    let trimmed = expression.trim();
    if trimmed.is_empty() {
        return Err("empty expression".to_string());
    }
    let source = if trimmed.starts_with('(') && trimmed.ends_with(')') {
        trimmed.to_string()
    } else {
        format!("({})", trimmed)
    };
    let mut ast = parse_one(&tokenize_one(&source))?;
    loop {
        match ast {
            Node::List(ref children) if children.len() == 1 => {
                if unwrap_single || matches!(&children[0], Node::List(_)) {
                    ast = children[0].clone();
                    continue;
                }
                return Ok(ast);
            }
            _ => return Ok(ast),
        }
    }
}

fn unique_unknowns(unknowns: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    for unknown in unknowns {
        if !out.contains(&unknown) {
            out.push(unknown);
        }
    }
    out
}

fn partial_formalization(
    request: FormalizationRequest,
    unknowns: Vec<String>,
    formalization_level: u8,
) -> Formalization {
    Formalization {
        source_text: request.text,
        interpretation: request.interpretation,
        formal_system: request.formal_system,
        dependencies: request.dependencies,
        computable: false,
        formalization_level,
        unknowns: unique_unknowns(unknowns),
        value_kind: "partial".to_string(),
        ast: None,
        lino: None,
    }
}

fn build_arithmetic_formalization(
    expression: &str,
    value_kind: &str,
) -> Result<(Node, String), String> {
    let ast = if value_kind == "truth-value" {
        if let Some((left, right)) = split_top_level_equals(expression) {
            Node::List(vec![
                parse_expression_shape(&left, true)?,
                Node::Leaf("=".to_string()),
                parse_expression_shape(&right, true)?,
            ])
        } else {
            parse_expression_shape(expression, true)?
        }
    } else {
        parse_expression_shape(expression, true)?
    };
    let lino = key_of(&ast);
    Ok((ast, lino))
}

/// Convert an explicitly selected interpretation into an executable or partial RML formalization.
pub fn formalize_selected_interpretation(request: FormalizationRequest) -> Formalization {
    let kind = request.interpretation.kind.to_lowercase();
    let raw_expression = request
        .interpretation
        .expression
        .clone()
        .or_else(|| request.interpretation.lino.clone())
        .unwrap_or_else(|| normalize_question_expression(&request.text));
    let can_use_arithmetic = request.formal_system == "rml-arithmetic"
        || request.formal_system == "arithmetic"
        || kind.starts_with("arithmetic");

    if can_use_arithmetic && !raw_expression.is_empty() {
        let value_kind =
            if kind.contains("equal") || split_top_level_equals(&raw_expression).is_some() {
                "truth-value"
            } else {
                "number"
            };
        match build_arithmetic_formalization(&raw_expression, value_kind) {
            Ok((ast, lino)) => Formalization {
                source_text: request.text,
                interpretation: request.interpretation,
                formal_system: request.formal_system,
                dependencies: request.dependencies,
                computable: true,
                formalization_level: 3,
                unknowns: vec![],
                value_kind: value_kind.to_string(),
                ast: Some(ast),
                lino: Some(lino),
            },
            Err(error) => partial_formalization(
                request,
                vec!["unsupported-arithmetic-shape".to_string(), error],
                1,
            ),
        }
    } else if request.interpretation.lino.is_some() && !raw_expression.is_empty() {
        match parse_expression_shape(&raw_expression, false) {
            Ok(ast) => {
                let lino = key_of(&ast);
                Formalization {
                    source_text: request.text,
                    interpretation: request.interpretation,
                    formal_system: request.formal_system,
                    dependencies: request.dependencies,
                    computable: true,
                    formalization_level: 3,
                    unknowns: vec![],
                    value_kind: if matches!(&ast, Node::List(children) if matches!(children.first(), Some(Node::Leaf(head)) if head == "?"))
                    {
                        "query".to_string()
                    } else {
                        "truth-value".to_string()
                    },
                    ast: Some(ast),
                    lino: Some(lino),
                }
            }
            Err(error) => partial_formalization(
                request,
                vec!["unsupported-lino-shape".to_string(), error],
                1,
            ),
        }
    } else {
        let mut unknowns = vec![
            "selected-subject".to_string(),
            "selected-relation".to_string(),
            "evidence-source".to_string(),
            "formal-shape".to_string(),
        ];
        for dependency in &request.dependencies {
            if dependency.status == "missing"
                || dependency.status == "unknown"
                || dependency.status == "partial"
            {
                unknowns.push(format!("dependency:{}", dependency.id));
            }
        }
        partial_formalization(request, unknowns, 2)
    }
}

/// Evaluate a formalization when it has an executable RML AST.
pub fn evaluate_formalization(formalization: &Formalization) -> FormalizationEvaluation {
    let Some(ast) = formalization.ast.as_ref() else {
        return FormalizationEvaluation {
            computable: false,
            formalization_level: formalization.formalization_level,
            unknowns: formalization.unknowns.clone(),
            result: FormalizationResultValue::Partial("unknown".to_string()),
        };
    };

    if !formalization.computable {
        return FormalizationEvaluation {
            computable: false,
            formalization_level: formalization.formalization_level,
            unknowns: formalization.unknowns.clone(),
            result: FormalizationResultValue::Partial("unknown".to_string()),
        };
    }

    let mut env = Env::new(None);
    let evaluated = eval_node(ast, &mut env);
    let result = match formalization.value_kind.as_str() {
        "truth-value" => FormalizationResultValue::TruthValue(evaluated.as_f64()),
        "query" => match evaluated {
            EvalResult::TypeQuery(s) => FormalizationResultValue::Type(s),
            other => FormalizationResultValue::Number(other.as_f64()),
        },
        _ => FormalizationResultValue::Number(evaluated.as_f64()),
    };

    FormalizationEvaluation {
        computable: true,
        formalization_level: formalization.formalization_level,
        unknowns: vec![],
        result,
    }
}

// ========== Runner ==========

/// A result from running a query: either a numeric value or a type string.
#[derive(Debug, Clone, PartialEq)]
pub enum RunResult {
    Num(f64),
    Type(String),
}

/// Evaluate a complete LiNo knowledge base and return both results and any
/// diagnostics emitted by the parser, evaluator, or type checker.
///
/// Each diagnostic carries a code (`E001`, `E002`, ...), a message, and a
/// source span (1-based line/col).  See `docs/DIAGNOSTICS.md` for the
/// full code list.  Errors do not abort evaluation: independent forms
/// continue to be processed after a failing one.
pub fn evaluate(text: &str, file: Option<&str>, options: Option<EnvOptions>) -> EvaluateResult {
    let mut env = Env::new(options);
    evaluate_with_env(text, file, &mut env)
}

/// Variant of [`evaluate`] that runs against a caller-owned `Env` instead of
/// allocating a fresh one.  Used by the REPL to preserve state across inputs.
pub fn evaluate_with_env(text: &str, file: Option<&str>, env: &mut Env) -> EvaluateResult {
    let mut diagnostics: Vec<Diagnostic> = Vec::new();
    let spans = compute_form_spans(text, file);

    let links = parse_lino(text);
    let forms: Vec<Node> = links
        .iter()
        .filter(|link_str| {
            let s = link_str.trim();
            !(s.starts_with("(#") && s.chars().nth(2).map_or(false, |c| c.is_whitespace()))
        })
        .filter_map(|link_str| {
            let toks = tokenize_one(link_str);
            match parse_one(&toks) {
                Ok(node) => Some(node),
                Err(msg) => {
                    diagnostics.push(Diagnostic::new(
                        "E002",
                        msg,
                        Span::new(file.map(|s| s.to_string()), 1, 1, 0),
                    ));
                    None
                }
            }
        })
        .collect();

    let mut results: Vec<RunResult> = Vec::new();

    // Silence the default panic hook while we deliberately catch evaluator
    // panics — otherwise they'd leak to stderr alongside the diagnostics.
    let prev_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    for (idx, form) in forms.into_iter().enumerate() {
        let mut form = form;
        loop {
            match form {
                Node::List(ref children) if children.len() == 1 => {
                    if let Node::List(_) = &children[0] {
                        form = children[0].clone();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        let span = spans
            .get(idx)
            .cloned()
            .unwrap_or_else(|| Span::new(file.map(|s| s.to_string()), 1, 1, 0));
        let result = catch_unwind(AssertUnwindSafe(|| eval_node(&form, env)));
        match result {
            Ok(EvalResult::Query(v)) => results.push(RunResult::Num(v)),
            Ok(EvalResult::TypeQuery(s)) => results.push(RunResult::Type(s)),
            Ok(_) => {}
            Err(payload) => {
                let (code, message) = decode_panic_payload(&payload);
                diagnostics.push(Diagnostic::new(&code, message, span));
            }
        }
    }

    std::panic::set_hook(prev_hook);

    EvaluateResult {
        results,
        diagnostics,
    }
}

/// Map a panic payload to a diagnostic `(code, message)` pair.  Known panic
/// messages emitted by the evaluator are mapped to the canonical `E001`/etc.
/// codes; anything else falls back to `E000`.
fn decode_panic_payload(payload: &Box<dyn std::any::Any + Send>) -> (String, String) {
    let raw_msg: String = if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "evaluation panicked".to_string()
    };
    if raw_msg.starts_with("Unknown op:") {
        ("E001".to_string(), raw_msg)
    } else if raw_msg.starts_with("Unknown aggregator") {
        ("E004".to_string(), raw_msg)
    } else {
        ("E000".to_string(), raw_msg)
    }
}

/// Run a complete LiNo knowledge base and return query results (including type queries).
pub fn run_typed(text: &str, options: Option<EnvOptions>) -> Vec<RunResult> {
    let links = parse_lino(text);
    let forms: Vec<Node> = links
        .iter()
        .filter(|link_str| {
            let s = link_str.trim();
            !(s.starts_with("(#") && s.chars().nth(2).map_or(false, |c| c.is_whitespace()))
        })
        .filter_map(|link_str| {
            let toks = tokenize_one(link_str);
            parse_one(&toks).ok()
        })
        .collect();

    let mut env = Env::new(options);
    let mut outs = Vec::new();

    for form in forms {
        let mut form = form;
        loop {
            match form {
                Node::List(ref children) if children.len() == 1 => {
                    if let Node::List(_) = &children[0] {
                        form = children[0].clone();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        let res = eval_node(&form, &mut env);
        match res {
            EvalResult::Query(v) => outs.push(RunResult::Num(v)),
            EvalResult::TypeQuery(s) => outs.push(RunResult::Type(s)),
            _ => {}
        }
    }
    outs
}

/// Run a complete LiNo knowledge base and return query results.
pub fn run(text: &str, options: Option<EnvOptions>) -> Vec<f64> {
    let links = parse_lino(text);

    // Filter out comment-only links and parse each link
    let forms: Vec<Node> = links
        .iter()
        .filter(|link_str| {
            let s = link_str.trim();
            // Skip if it's just a comment link like "(# ...)"
            !(s.starts_with("(#") && s.chars().nth(2).map_or(false, |c| c.is_whitespace()))
        })
        .filter_map(|link_str| {
            let toks = tokenize_one(link_str);
            parse_one(&toks).ok()
        })
        .collect();

    let mut env = Env::new(options);
    let mut outs = Vec::new();

    for form in forms {
        // Unwrap single-element arrays (LiNo wraps everything in outer parens)
        let mut form = form;
        loop {
            match form {
                Node::List(ref children) if children.len() == 1 => {
                    if let Node::List(_) = &children[0] {
                        form = children[0].clone();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        let res = eval_node(&form, &mut env);
        if let EvalResult::Query(v) = res {
            outs.push(v);
        }
    }
    outs
}

// Tests are in the tests/ directory (integration tests).
// To run: cargo test

pub mod repl;
