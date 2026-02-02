// ADL — minimal associative-dependent logic over LiNo (Links Notation)
// Supports many-valued logics from unary (1-valued) through continuous probabilistic (∞-valued).
// See: https://en.wikipedia.org/wiki/Many-valued_logic
//
// - Uses official links-notation crate to parse LiNo text into links
// - Terms are defined via (x: x is x)
// - Probabilities are assigned ONLY via: ((<expr>) has probability <p>)
// - Redefinable ops: (=: ...), (!=: not =), (and: avg|min|max|prod|ps), (or: ...), (not: ...)
// - Range: (range: 0 1) for [0,1] or (range: -1 1) for [-1,1] (balanced/symmetric)
// - Valence: (valence: N) to restrict truth values to N discrete levels (N=2 → Boolean, N=3 → ternary, etc.)
// - Query: (? <expr>)

use std::collections::{HashMap, HashSet};
use std::fmt;

// ========== LiNo Parser ==========
// Uses the official links-notation crate for parsing LiNo text.
// See: https://github.com/link-foundation/links-notation

/// Parse LiNo text into a vector of link strings (each a top-level parenthesized expression).
pub fn parse_lino(text: &str) -> Vec<String> {
    // The links-notation crate treats blank lines as group separators,
    // so we split the input by blank lines and parse each segment separately.
    let mut all_links = Vec::new();
    for segment in text.split("\n\n") {
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
    fn apply(&self, xs: &[f64], lo: f64) -> f64 {
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

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "avg" => Some(Aggregator::Avg),
            "min" => Some(Aggregator::Min),
            "max" => Some(Aggregator::Max),
            "prod" => Some(Aggregator::Prod),
            "ps" => Some(Aggregator::Ps),
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
    Compose { outer: String, inner: String },
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

/// The evaluation environment: holds terms, assignments, operators, and range/valence config.
pub struct Env {
    pub terms: HashSet<String>,
    pub assign: HashMap<String, f64>,
    pub symbol_prob: HashMap<String, f64>,
    pub lo: f64,
    pub hi: f64,
    pub valence: u32,
    pub ops: HashMap<String, Op>,
}

impl Env {
    pub fn new(options: Option<EnvOptions>) -> Self {
        let opts = options.unwrap_or_default();
        let mut ops = HashMap::new();
        ops.insert("not".to_string(), Op::Not);
        ops.insert("and".to_string(), Op::Agg(Aggregator::Avg));
        ops.insert("or".to_string(), Op::Agg(Aggregator::Max));
        ops.insert("=".to_string(), Op::Eq);
        ops.insert("!=".to_string(), Op::Neq);
        ops.insert("+".to_string(), Op::Add);
        ops.insert("-".to_string(), Op::Sub);
        ops.insert("*".to_string(), Op::Mul);
        ops.insert("/".to_string(), Op::Div);

        Self {
            terms: HashSet::new(),
            assign: HashMap::new(),
            symbol_prob: HashMap::new(),
            lo: opts.lo,
            hi: opts.hi,
            valence: opts.valence,
            ops,
        }
    }

    /// Midpoint of the range.
    pub fn mid(&self) -> f64 {
        (self.lo + self.hi) / 2.0
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
            Op::Agg(agg) => agg.apply(vals, self.lo),
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
        self.ops.insert("=".to_string(), Op::Eq);
        self.ops.insert("!=".to_string(), Op::Neq);
        self.ops.insert("+".to_string(), Op::Add);
        self.ops.insert("-".to_string(), Op::Sub);
        self.ops.insert("*".to_string(), Op::Mul);
        self.ops.insert("/".to_string(), Op::Div);
    }
}

// ========== Query Result ==========

/// Result of evaluating an expression: either a plain value or a query result.
#[derive(Debug, Clone)]
pub enum EvalResult {
    Value(f64),
    Query(f64),
}

impl EvalResult {
    pub fn as_f64(&self) -> f64 {
        match self {
            EvalResult::Value(v) | EvalResult::Query(v) => *v,
        }
    }

    pub fn is_query(&self) -> bool {
        matches!(self, EvalResult::Query(_))
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

            // (head : ...) form
            if children.len() >= 3 {
                if let Node::Leaf(ref colon) = children[1] {
                    if colon == ":" {
                        let head = match &children[0] {
                            Node::Leaf(s) => s.clone(),
                            other => key_of(other),
                        };
                        return define_form(&head, &children[2..], env);
                    }
                }
            }

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
                    let v = eval_node(&children[1], env).as_f64();
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

            // Infix AND/OR: ((A) and (B)) / ((A) or (B))
            if children.len() == 3 {
                if let Node::Leaf(ref op_name) = children[1] {
                    if op_name == "and" || op_name == "or" {
                        let l = eval_node(&children[0], env).as_f64();
                        let r = eval_node(&children[2], env).as_f64();
                        return EvalResult::Value(env.clamp(env.apply_op(op_name, &[l, r])));
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

            // Prefix: (not X), (and X Y ...), (or X Y ...)
            if let Node::Leaf(ref head) = children[0] {
                let head = head.clone();
                let vals: Vec<f64> = children[1..]
                    .iter()
                    .map(|a| eval_node(a, env).as_f64())
                    .collect();
                return EvalResult::Value(env.clamp(env.apply_op(&head, &vals)));
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
            }
        }

        // Aggregator selection: (and: avg|min|max|prod|ps)
        if (head == "and" || head == "or") && rhs.len() == 1 {
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

// ========== Runner ==========

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

// ========== Tests ==========

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: f64, expected: f64) {
        let epsilon = 1e-9;
        assert!(
            (actual - expected).abs() < epsilon,
            "Expected {}, got {} (diff: {})",
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    // ===== tokenize_one =====

    #[test]
    fn tokenize_simple_link() {
        let tokens = tokenize_one("(a: a is a)");
        assert_eq!(tokens, vec!["(", "a:", "a", "is", "a", ")"]);
    }

    #[test]
    fn tokenize_nested_link() {
        let tokens = tokenize_one("((a = a) has probability 1)");
        assert_eq!(
            tokens,
            vec!["(", "(", "a", "=", "a", ")", "has", "probability", "1", ")"]
        );
    }

    #[test]
    fn tokenize_strip_inline_comments() {
        let tokens = tokenize_one("(and: avg) # this is a comment");
        assert_eq!(tokens, vec!["(", "and:", "avg", ")"]);
    }

    #[test]
    fn tokenize_balance_parens_after_stripping_comments() {
        let tokens = tokenize_one("((and: avg) # comment)");
        assert_eq!(tokens, vec!["(", "(", "and:", "avg", ")", ")"]);
    }

    // ===== parse_one =====

    #[test]
    fn parse_simple_link() {
        let tokens: Vec<String> = vec!["(", "a:", "a", "is", "a", ")"]
            .into_iter()
            .map(String::from)
            .collect();
        let ast = parse_one(&tokens).unwrap();
        assert_eq!(
            ast,
            Node::List(vec![
                Node::Leaf("a:".into()),
                Node::Leaf("a".into()),
                Node::Leaf("is".into()),
                Node::Leaf("a".into()),
            ])
        );
    }

    #[test]
    fn parse_nested_link() {
        let tokens: Vec<String> =
            vec!["(", "(", "a", "=", "a", ")", "has", "probability", "1", ")"]
                .into_iter()
                .map(String::from)
                .collect();
        let ast = parse_one(&tokens).unwrap();
        assert_eq!(
            ast,
            Node::List(vec![
                Node::List(vec![
                    Node::Leaf("a".into()),
                    Node::Leaf("=".into()),
                    Node::Leaf("a".into()),
                ]),
                Node::Leaf("has".into()),
                Node::Leaf("probability".into()),
                Node::Leaf("1".into()),
            ])
        );
    }

    #[test]
    fn parse_deeply_nested_link() {
        let tokens: Vec<String> = vec![
            "(", "?", "(", "(", "a", "=", "a", ")", "and", "(", "a", "!=", "a", ")", ")", ")",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let ast = parse_one(&tokens).unwrap();
        assert_eq!(
            ast,
            Node::List(vec![
                Node::Leaf("?".into()),
                Node::List(vec![
                    Node::List(vec![
                        Node::Leaf("a".into()),
                        Node::Leaf("=".into()),
                        Node::Leaf("a".into()),
                    ]),
                    Node::Leaf("and".into()),
                    Node::List(vec![
                        Node::Leaf("a".into()),
                        Node::Leaf("!=".into()),
                        Node::Leaf("a".into()),
                    ]),
                ]),
            ])
        );
    }

    // ===== Env =====

    #[test]
    fn env_default_operators() {
        let env = Env::new(None);
        assert!(env.ops.contains_key("not"));
        assert!(env.ops.contains_key("and"));
        assert!(env.ops.contains_key("or"));
        assert!(env.ops.contains_key("="));
        assert!(env.ops.contains_key("!="));
    }

    #[test]
    fn env_define_new_operators() {
        let mut env = Env::new(None);
        env.define_op("test", Op::Agg(Aggregator::Min));
        assert!(env.ops.contains_key("test"));
        assert_eq!(env.apply_op("test", &[0.5, 1.0]), 0.5);
    }

    #[test]
    fn env_store_expression_probabilities() {
        let mut env = Env::new(None);
        let expr = Node::List(vec![
            Node::Leaf("a".into()),
            Node::Leaf("=".into()),
            Node::Leaf("a".into()),
        ]);
        env.set_expr_prob(&expr, 1.0);
        assert_eq!(env.assign.get("(a = a)"), Some(&1.0));
    }

    // ===== eval_node =====

    #[test]
    fn eval_numeric_literals() {
        let mut env = Env::new(None);
        assert_eq!(eval_node(&Node::Leaf("1".into()), &mut env).as_f64(), 1.0);
        assert_eq!(
            eval_node(&Node::Leaf("0.5".into()), &mut env).as_f64(),
            0.5
        );
        assert_eq!(eval_node(&Node::Leaf("0".into()), &mut env).as_f64(), 0.0);
    }

    #[test]
    fn eval_term_definitions() {
        let mut env = Env::new(None);
        eval_node(
            &Node::List(vec![
                Node::Leaf("a:".into()),
                Node::Leaf("a".into()),
                Node::Leaf("is".into()),
                Node::Leaf("a".into()),
            ]),
            &mut env,
        );
        assert!(env.terms.contains("a"));
    }

    #[test]
    fn eval_operator_redefinitions() {
        let mut env = Env::new(None);
        eval_node(
            &Node::List(vec![
                Node::Leaf("!=:".into()),
                Node::Leaf("not".into()),
                Node::Leaf("=".into()),
            ]),
            &mut env,
        );
        assert!(env.ops.contains_key("!="));
    }

    #[test]
    fn eval_aggregator_selection() {
        let mut env = Env::new(None);
        eval_node(
            &Node::List(vec![
                Node::Leaf("and:".into()),
                Node::Leaf("min".into()),
            ]),
            &mut env,
        );
        assert_eq!(env.apply_op("and", &[0.3, 0.7]), 0.3);
    }

    #[test]
    fn eval_probability_assignments() {
        let mut env = Env::new(None);
        let result = eval_node(
            &Node::List(vec![
                Node::List(vec![
                    Node::Leaf("a".into()),
                    Node::Leaf("=".into()),
                    Node::Leaf("a".into()),
                ]),
                Node::Leaf("has".into()),
                Node::Leaf("probability".into()),
                Node::Leaf("1".into()),
            ]),
            &mut env,
        );
        assert_eq!(result.as_f64(), 1.0);
        assert_eq!(env.assign.get("(a = a)"), Some(&1.0));
    }

    #[test]
    fn eval_equality_operator() {
        let mut env = Env::new(None);
        let result = eval_node(
            &Node::List(vec![
                Node::Leaf("a".into()),
                Node::Leaf("=".into()),
                Node::Leaf("a".into()),
            ]),
            &mut env,
        );
        assert_eq!(result.as_f64(), 1.0);
    }

    #[test]
    fn eval_inequality_operator() {
        let mut env = Env::new(None);
        let result = eval_node(
            &Node::List(vec![
                Node::Leaf("a".into()),
                Node::Leaf("!=".into()),
                Node::Leaf("a".into()),
            ]),
            &mut env,
        );
        assert_eq!(result.as_f64(), 0.0);
    }

    #[test]
    fn eval_not_operator() {
        let mut env = Env::new(None);
        let result = eval_node(
            &Node::List(vec![Node::Leaf("not".into()), Node::Leaf("1".into())]),
            &mut env,
        );
        assert_eq!(result.as_f64(), 0.0);
    }

    #[test]
    fn eval_and_operator_avg() {
        let mut env = Env::new(None);
        let result = eval_node(
            &Node::List(vec![
                Node::Leaf("1".into()),
                Node::Leaf("and".into()),
                Node::Leaf("0".into()),
            ]),
            &mut env,
        );
        assert_eq!(result.as_f64(), 0.5);
    }

    #[test]
    fn eval_or_operator_max() {
        let mut env = Env::new(None);
        let result = eval_node(
            &Node::List(vec![
                Node::Leaf("1".into()),
                Node::Leaf("or".into()),
                Node::Leaf("0".into()),
            ]),
            &mut env,
        );
        assert_eq!(result.as_f64(), 1.0);
    }

    #[test]
    fn eval_queries() {
        let mut env = Env::new(None);
        let result = eval_node(
            &Node::List(vec![Node::Leaf("?".into()), Node::Leaf("1".into())]),
            &mut env,
        );
        assert!(result.is_query());
        assert_eq!(result.as_f64(), 1.0);
    }

    // ===== run =====

    #[test]
    fn run_demo_example() {
        let text = r#"
(a: a is a)
(!=: not =)
(and: avg)
(or: max)
((a = a) has probability 1)
((a != a) has probability 0)
(? ((a = a) and (a != a)))
(? ((a = a) or  (a != a)))
"#;
        let results = run(text, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.5);
        assert_eq!(results[1], 1.0);
    }

    #[test]
    fn run_flipped_axioms_example() {
        let text = r#"
(a: a is a)
(!=: not =)
(and: avg)
(or: max)
((a = a) has probability 0)
((a != a) has probability 1)
(? ((a = a) and (a != a)))
(? ((a = a) or  (a != a)))
"#;
        let results = run(text, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.5);
        assert_eq!(results[1], 1.0);
    }

    #[test]
    fn run_different_aggregators_for_and() {
        let text = r#"
(a: a is a)
(and: min)
((a = a) has probability 1)
((a != a) has probability 0)
(? ((a = a) and (a != a)))
"#;
        let results = run(text, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0.0);
    }

    #[test]
    fn run_product_aggregator() {
        let text = r#"
(and: prod)
(? (0.5 and 0.5))
"#;
        let results = run(text, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0.25);
    }

    #[test]
    fn run_probabilistic_sum_aggregator() {
        let text = r#"
(or: ps)
(? (0.5 or 0.5))
"#;
        let results = run(text, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 0.75);
    }

    #[test]
    fn run_ignore_comment_only_links() {
        let text = r#"
# This is a comment
(# This is also a comment)
(a: a is a)
(? (a = a))
"#;
        let results = run(text, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1.0);
    }

    #[test]
    fn run_handle_inline_comments() {
        let text = r#"
(a: a is a) # define term a
((a = a) has probability 1) # axiom
(? (a = a)) # query
"#;
        let results = run(text, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1.0);
    }

    // ===== quantize =====

    #[test]
    fn quantize_continuous() {
        assert_eq!(quantize(0.33, 0, 0.0, 1.0), 0.33);
        assert_eq!(quantize(0.33, 1, 0.0, 1.0), 0.33);
    }

    #[test]
    fn quantize_binary_boolean() {
        assert_eq!(quantize(0.3, 2, 0.0, 1.0), 0.0);
        assert_eq!(quantize(0.7, 2, 0.0, 1.0), 1.0);
        assert_eq!(quantize(0.5, 2, 0.0, 1.0), 1.0); // round up at midpoint
    }

    #[test]
    fn quantize_ternary() {
        assert_eq!(quantize(0.1, 3, 0.0, 1.0), 0.0);
        assert_eq!(quantize(0.4, 3, 0.0, 1.0), 0.5);
        assert_eq!(quantize(0.5, 3, 0.0, 1.0), 0.5);
        assert_eq!(quantize(0.8, 3, 0.0, 1.0), 1.0);
    }

    #[test]
    fn quantize_5_levels() {
        assert_eq!(quantize(0.1, 5, 0.0, 1.0), 0.0);
        assert_eq!(quantize(0.3, 5, 0.0, 1.0), 0.25);
        assert_eq!(quantize(0.6, 5, 0.0, 1.0), 0.5);
        assert_eq!(quantize(0.7, 5, 0.0, 1.0), 0.75);
        assert_eq!(quantize(0.9, 5, 0.0, 1.0), 1.0);
    }

    #[test]
    fn quantize_balanced_ternary() {
        assert_eq!(quantize(-0.8, 3, -1.0, 1.0), -1.0);
        assert_eq!(quantize(-0.2, 3, -1.0, 1.0), 0.0);
        assert_eq!(quantize(0.0, 3, -1.0, 1.0), 0.0);
        assert_eq!(quantize(0.6, 3, -1.0, 1.0), 1.0);
    }

    #[test]
    fn quantize_binary_balanced() {
        assert_eq!(quantize(-0.5, 2, -1.0, 1.0), -1.0);
        assert_eq!(quantize(0.5, 2, -1.0, 1.0), 1.0);
    }

    // ===== Env with options =====

    #[test]
    fn env_custom_range() {
        let env = Env::new(Some(EnvOptions {
            lo: -1.0,
            hi: 1.0,
            valence: 0,
        }));
        assert_eq!(env.lo, -1.0);
        assert_eq!(env.hi, 1.0);
        assert_eq!(env.mid(), 0.0);
    }

    #[test]
    fn env_custom_valence() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 3,
        }));
        assert_eq!(env.valence, 3);
    }

    #[test]
    fn env_clamp_to_range() {
        let env = Env::new(Some(EnvOptions {
            lo: -1.0,
            hi: 1.0,
            valence: 0,
        }));
        assert_eq!(env.clamp(2.0), 1.0);
        assert_eq!(env.clamp(-2.0), -1.0);
        assert_eq!(env.clamp(0.5), 0.5);
    }

    #[test]
    fn env_clamp_and_quantize() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 2,
        }));
        assert_eq!(env.clamp(0.3), 0.0);
        assert_eq!(env.clamp(0.7), 1.0);
    }

    #[test]
    fn env_midpoint_both_ranges() {
        let env01 = Env::new(None);
        assert_eq!(env01.mid(), 0.5);
        let env_bal = Env::new(Some(EnvOptions {
            lo: -1.0,
            hi: 1.0,
            valence: 0,
        }));
        assert_eq!(env_bal.mid(), 0.0);
    }

    #[test]
    fn env_default_symbol_probability() {
        let env = Env::new(Some(EnvOptions {
            lo: -1.0,
            hi: 1.0,
            valence: 0,
        }));
        assert_eq!(env.get_symbol_prob("unknown"), 0.0);
    }

    #[test]
    fn not_operator_mirror_balanced() {
        let env = Env::new(Some(EnvOptions {
            lo: -1.0,
            hi: 1.0,
            valence: 0,
        }));
        assert_eq!(env.apply_op("not", &[1.0]), -1.0);
        assert_eq!(env.apply_op("not", &[-1.0]), 1.0);
        assert_eq!(env.apply_op("not", &[0.0]), 0.0);
    }

    #[test]
    fn not_operator_mirror_standard() {
        let env = Env::new(None);
        assert_eq!(env.apply_op("not", &[1.0]), 0.0);
        assert_eq!(env.apply_op("not", &[0.0]), 1.0);
        assert_eq!(env.apply_op("not", &[0.5]), 0.5);
    }

    // ===== Unary logic (1-valued) =====

    #[test]
    fn unary_collapse_values() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 1,
        }));
        assert_eq!(env.clamp(0.5), 0.5);
        assert_eq!(env.clamp(1.0), 1.0);
        assert_eq!(env.clamp(0.0), 0.0);
    }

    #[test]
    fn unary_via_run() {
        let results = run(
            r#"
(valence: 1)
(a: a is a)
(? (a = a))
"#,
            Some(EnvOptions {
                lo: 0.0,
                hi: 1.0,
                valence: 1,
            }),
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1.0);
    }

    // ===== Binary logic (2-valued, Boolean) =====

    #[test]
    fn binary_quantize_01() {
        let results = run(
            r#"
(valence: 2)
(a: a is a)
(!=: not =)
(and: avg)
(or: max)
((a = a) has probability 1)
((a != a) has probability 0)
(? (a = a))
(? (a != a))
(? ((a = a) and (a != a)))
(? ((a = a) or (a != a)))
"#,
            None,
        );
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], 1.0);
        assert_eq!(results[1], 0.0);
        assert_eq!(results[2], 1.0);
        assert_eq!(results[3], 1.0);
    }

    #[test]
    fn binary_quantize_balanced() {
        let results = run(
            r#"
(range: -1 1)
(valence: 2)
(a: a is a)
((a = a) has probability 1)
(? (a = a))
(? (not (a = a)))
"#,
            Some(EnvOptions {
                lo: -1.0,
                hi: 1.0,
                valence: 2,
            }),
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 1.0);
        assert_eq!(results[1], -1.0);
    }

    #[test]
    fn binary_law_excluded_middle() {
        let results = run(
            r#"
(valence: 2)
(a: a is a)
(or: max)
((a = a) has probability 1)
(? ((a = a) or (not (a = a))))
"#,
            None,
        );
        assert_eq!(results[0], 1.0);
    }

    #[test]
    fn binary_law_non_contradiction() {
        let results = run(
            r#"
(valence: 2)
(a: a is a)
(and: min)
((a = a) has probability 1)
(? ((a = a) and (not (a = a))))
"#,
            None,
        );
        assert_eq!(results[0], 0.0);
    }

    // ===== Ternary logic (3-valued) =====

    #[test]
    fn ternary_quantize_01() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 3,
        }));
        assert_eq!(env.clamp(0.0), 0.0);
        assert_eq!(env.clamp(0.3), 0.5);
        assert_eq!(env.clamp(0.5), 0.5);
        assert_eq!(env.clamp(0.8), 1.0);
        assert_eq!(env.clamp(1.0), 1.0);
    }

    #[test]
    fn ternary_quantize_balanced() {
        let env = Env::new(Some(EnvOptions {
            lo: -1.0,
            hi: 1.0,
            valence: 3,
        }));
        assert_eq!(env.clamp(-1.0), -1.0);
        assert_eq!(env.clamp(-0.4), 0.0);
        assert_eq!(env.clamp(0.0), 0.0);
        assert_eq!(env.clamp(0.6), 1.0);
        assert_eq!(env.clamp(1.0), 1.0);
    }

    #[test]
    fn ternary_kleene_logic() {
        let results = run(
            r#"
(valence: 3)
(and: min)
(or: max)
(? (0.5 and 1))
(? (0.5 or 0))
(? (not 0.5))
"#,
            None,
        );
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], 0.5);
        assert_eq!(results[1], 0.5);
        assert_eq!(results[2], 0.5);
    }

    #[test]
    fn ternary_kleene_unknown_and_false() {
        let results = run(
            r#"
(valence: 3)
(and: min)
(? (0.5 and 0))
"#,
            None,
        );
        assert_eq!(results[0], 0.0);
    }

    #[test]
    fn ternary_kleene_unknown_or_true() {
        let results = run(
            r#"
(valence: 3)
(or: max)
(? (0.5 or 1))
"#,
            None,
        );
        assert_eq!(results[0], 1.0);
    }

    #[test]
    fn ternary_excluded_middle_fails() {
        let results = run(
            r#"
(valence: 3)
(or: max)
(? (0.5 or (not 0.5)))
"#,
            None,
        );
        assert_eq!(results[0], 0.5);
    }

    #[test]
    fn ternary_liar_paradox_01() {
        let results = run(
            r#"
(valence: 3)
(and: avg)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
"#,
            None,
        );
        assert_eq!(results[0], 0.5);
    }

    #[test]
    fn ternary_liar_paradox_balanced() {
        let results = run(
            r#"
(range: -1 1)
(valence: 3)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
"#,
            Some(EnvOptions {
                lo: -1.0,
                hi: 1.0,
                valence: 3,
            }),
        );
        assert_eq!(results[0], 0.0);
    }

    // ===== Quaternary logic (4-valued) =====

    #[test]
    fn quaternary_quantize_01() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 4,
        }));
        approx(env.clamp(0.0), 0.0);
        approx(env.clamp(0.2), 1.0 / 3.0);
        approx(env.clamp(0.5), 2.0 / 3.0);
        approx(env.clamp(0.6), 2.0 / 3.0);
        approx(env.clamp(1.0), 1.0);
    }

    #[test]
    fn quaternary_quantize_balanced() {
        let env = Env::new(Some(EnvOptions {
            lo: -1.0,
            hi: 1.0,
            valence: 4,
        }));
        approx(env.clamp(-1.0), -1.0);
        approx(env.clamp(-0.5), -1.0 / 3.0);
        approx(env.clamp(0.0), 1.0 / 3.0);
        approx(env.clamp(0.5), 1.0 / 3.0);
        approx(env.clamp(1.0), 1.0);
    }

    #[test]
    fn quaternary_via_run() {
        let results = run(
            r#"
(valence: 4)
(and: min)
(or: max)
(? (0.33 and 0.66))
(? (0.33 or 0.66))
"#,
            None,
        );
        assert_eq!(results.len(), 2);
        approx(results[0], 1.0 / 3.0);
        approx(results[1], 2.0 / 3.0);
    }

    // ===== Quinary logic (5-valued) =====

    #[test]
    fn quinary_quantize_01() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 5,
        }));
        assert_eq!(env.clamp(0.0), 0.0);
        assert_eq!(env.clamp(0.1), 0.0);
        assert_eq!(env.clamp(0.2), 0.25);
        assert_eq!(env.clamp(0.4), 0.5);
        assert_eq!(env.clamp(0.6), 0.5);
        assert_eq!(env.clamp(0.7), 0.75);
        assert_eq!(env.clamp(0.9), 1.0);
        assert_eq!(env.clamp(1.0), 1.0);
    }

    #[test]
    fn quinary_paradox_at_05() {
        let results = run(
            r#"
(valence: 5)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
"#,
            None,
        );
        assert_eq!(results[0], 0.5);
    }

    // ===== Higher N-valued logics =====

    #[test]
    fn seven_valued_logic() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 7,
        }));
        approx(env.clamp(0.0), 0.0);
        approx(env.clamp(0.5), 0.5);
        approx(env.clamp(1.0), 1.0);
    }

    #[test]
    fn ten_valued_logic() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 10,
        }));
        approx(env.clamp(0.0), 0.0);
        approx(env.clamp(1.0), 1.0);
        approx(env.clamp(0.5), 5.0 / 9.0);
    }

    #[test]
    fn hundred_valued_logic() {
        let env = Env::new(Some(EnvOptions {
            lo: 0.0,
            hi: 1.0,
            valence: 100,
        }));
        approx(env.clamp(0.0), 0.0);
        approx(env.clamp(1.0), 1.0);
        let actual = env.clamp(0.5);
        assert!(
            (actual - 0.5).abs() < 0.02,
            "100-valued 0.5 should be close to 0.5, got {}",
            actual
        );
    }

    // ===== Continuous probabilistic logic =====

    #[test]
    fn continuous_preserve_values_01() {
        let results = run(
            r#"
(a: a is a)
(and: avg)
((a = a) has probability 0.7)
(? (a = a))
(? (not (a = a)))
"#,
            None,
        );
        assert_eq!(results.len(), 2);
        approx(results[0], 0.7);
        approx(results[1], 0.3);
    }

    #[test]
    fn continuous_preserve_values_balanced() {
        let results = run(
            r#"
(range: -1 1)
(a: a is a)
((a = a) has probability 0.4)
(? (a = a))
(? (not (a = a)))
"#,
            Some(EnvOptions {
                lo: -1.0,
                hi: 1.0,
                valence: 0,
            }),
        );
        assert_eq!(results.len(), 2);
        approx(results[0], 0.4);
        approx(results[1], -0.4);
    }

    #[test]
    fn continuous_liar_paradox_01() {
        let results = run(
            r#"
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
(? (not (s = false)))
"#,
            None,
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.5);
        assert_eq!(results[1], 0.5);
    }

    #[test]
    fn continuous_liar_paradox_balanced() {
        let results = run(
            r#"
(range: -1 1)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
(? (not (s = false)))
"#,
            Some(EnvOptions {
                lo: -1.0,
                hi: 1.0,
                valence: 0,
            }),
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 0.0);
        assert_eq!(results[1], 0.0);
    }

    #[test]
    fn continuous_fuzzy_membership() {
        let results = run(
            r#"
(and: min)
(or: max)
(a: a is a)
(b: b is b)
((a = tall) has probability 0.8)
((b = tall) has probability 0.3)
(? ((a = tall) and (b = tall)))
(? ((a = tall) or (b = tall)))
"#,
            None,
        );
        assert_eq!(results.len(), 2);
        approx(results[0], 0.3);
        approx(results[1], 0.8);
    }

    // ===== Range and valence configuration via LiNo syntax =====

    #[test]
    fn config_range_via_lino() {
        let results = run(
            r#"
(range: -1 1)
(a: a is a)
(? (a = a))
(? (not (a = a)))
"#,
            None,
        );
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], 1.0);
        assert_eq!(results[1], -1.0);
    }

    #[test]
    fn config_valence_via_lino() {
        let results = run(
            r#"
(valence: 3)
(? (not 0.5))
"#,
            None,
        );
        assert_eq!(results[0], 0.5);
    }

    #[test]
    fn config_both_range_and_valence() {
        let results = run(
            r#"
(range: -1 1)
(valence: 3)
(a: a is a)
(? (a = a))
(? (not (a = a)))
(? (0 and 0))
"#,
            None,
        );
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], 1.0);
        assert_eq!(results[1], -1.0);
        assert_eq!(results[2], 0.0);
    }

    // ===== Liar paradox resolution across logic types =====

    #[test]
    fn liar_paradox_ternary_01() {
        let results = run(
            r#"
(valence: 3)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
"#,
            None,
        );
        assert_eq!(results[0], 0.5);
    }

    #[test]
    fn liar_paradox_ternary_balanced() {
        let results = run(
            r#"
(range: -1 1)
(valence: 3)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
"#,
            Some(EnvOptions {
                lo: -1.0,
                hi: 1.0,
                valence: 3,
            }),
        );
        assert_eq!(results[0], 0.0);
    }

    #[test]
    fn liar_paradox_continuous_01() {
        let results = run(
            r#"
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
(? (not (s = false)))
"#,
            None,
        );
        assert_eq!(results[0], 0.5);
        assert_eq!(results[1], 0.5);
    }

    #[test]
    fn liar_paradox_continuous_balanced() {
        let results = run(
            r#"
(range: -1 1)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
(? (not (s = false)))
"#,
            Some(EnvOptions {
                lo: -1.0,
                hi: 1.0,
                valence: 0,
            }),
        );
        assert_eq!(results[0], 0.0);
        assert_eq!(results[1], 0.0);
    }

    #[test]
    fn liar_paradox_5valued_01() {
        let results = run(
            r#"
(valence: 5)
(s: s is s)
((s = false) has probability 0.5)
(? (s = false))
"#,
            None,
        );
        assert_eq!(results[0], 0.5);
    }

    #[test]
    fn liar_paradox_5valued_balanced() {
        let results = run(
            r#"
(range: -1 1)
(valence: 5)
(s: s is s)
((s = false) has probability 0)
(? (s = false))
"#,
            Some(EnvOptions {
                lo: -1.0,
                hi: 1.0,
                valence: 5,
            }),
        );
        assert_eq!(results[0], 0.0);
    }

    // ===== Decimal-precision arithmetic =====

    #[test]
    fn dec_round_01_plus_02() {
        assert_eq!(dec_round(0.1_f64 + 0.2_f64), 0.3);
    }

    #[test]
    fn dec_round_03_minus_01() {
        assert_eq!(dec_round(0.3_f64 - 0.1_f64), 0.2);
    }

    #[test]
    fn dec_round_exact_values() {
        assert_eq!(dec_round(1.0), 1.0);
        assert_eq!(dec_round(0.0), 0.0);
        assert_eq!(dec_round(0.5), 0.5);
    }

    #[test]
    fn dec_round_non_finite() {
        assert_eq!(dec_round(f64::INFINITY), f64::INFINITY);
        assert_eq!(dec_round(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(dec_round(f64::NAN).is_nan());
    }

    #[test]
    fn arith_add() {
        let results = run("(? (0.1 + 0.2))", None);
        assert_eq!(results[0], 0.3);
    }

    #[test]
    fn arith_sub() {
        let results = run("(? (0.3 - 0.1))", None);
        assert_eq!(results[0], 0.2);
    }

    #[test]
    fn arith_mul() {
        let results = run("(? (0.1 * 0.2))", None);
        assert_eq!(results[0], 0.02);
    }

    #[test]
    fn arith_div() {
        let results = run("(? (1 / 3))", None);
        approx(results[0], 1.0 / 3.0);
    }

    #[test]
    fn arith_div_by_zero() {
        let results = run("(? (0 / 0))", None);
        assert_eq!(results[0], 0.0);
    }

    #[test]
    fn arith_add_eq_03() {
        let results = run("(? ((0.1 + 0.2) = 0.3))", None);
        assert_eq!(results[0], 1.0);
    }

    #[test]
    fn arith_add_neq_03() {
        let results = run("(? ((0.1 + 0.2) != 0.3))", None);
        assert_eq!(results[0], 0.0);
    }

    #[test]
    fn arith_sub_eq_02() {
        let results = run("(? ((0.3 - 0.1) = 0.2))", None);
        assert_eq!(results[0], 1.0);
    }

    #[test]
    fn arith_nested() {
        let results = run("(? ((0.1 + 0.2) + (0.3 + 0.1)))", None);
        assert_eq!(results[0], 0.7);
    }

    #[test]
    fn arith_clamps_in_query() {
        // 2 + 3 = 5, but query clamps to [0,1], so result is 1
        let results = run("(? (2 + 3))", None);
        assert_eq!(results[0], 1.0);
    }

    #[test]
    fn arith_equality_across_expressions() {
        let results = run("(? ((0.1 + 0.2) = (0.5 - 0.2)))", None);
        assert_eq!(results[0], 1.0);
    }
}
