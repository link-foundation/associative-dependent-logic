// Walks the repository-root /examples folder and runs every .lino file
// through the Rust implementation. Asserts the output matches the canonical
// fixtures in /examples/expected.json.
//
// The JavaScript test suite asserts against the same fixtures file, so any
// drift between the two implementations fails both test suites.

use rml::{run_typed, RunResult};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
enum ExpectedValue {
    Num(f64),
    Type(String),
}

fn examples_dir() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join("..").join("examples")
}

fn list_lino_files(dir: &Path) -> Vec<String> {
    let mut files: Vec<String> = fs::read_dir(dir)
        .expect("examples dir exists")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".lino"))
        .collect();
    files.sort();
    files
}

// Minimal JSON parser specialised for examples/expected.json.
// The fixtures file has a fixed shape:
//   { "<filename>.lino": [ { "num": <number> } | { "type": "<string>" }, ... ], ... }
// We don't pull in serde_json just to read this — a hand-rolled parser
// keeps the crate's dependency surface unchanged.
struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            bytes: text.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, ch: u8) {
        self.skip_ws();
        assert!(
            self.pos < self.bytes.len() && self.bytes[self.pos] == ch,
            "expected '{}' at byte {}",
            ch as char,
            self.pos
        );
        self.pos += 1;
    }

    fn peek(&mut self) -> u8 {
        self.skip_ws();
        assert!(self.pos < self.bytes.len(), "unexpected end of JSON");
        self.bytes[self.pos]
    }

    fn parse_string(&mut self) -> String {
        self.expect(b'"');
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'"' {
                self.pos += 1;
                return out;
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = self.bytes[self.pos];
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b't' => out.push('\t'),
                    b'r' => out.push('\r'),
                    other => panic!("unsupported escape \\{}", other as char),
                }
                continue;
            }
            out.push(b as char);
            self.pos += 1;
        }
        panic!("unterminated string");
    }

    fn parse_number(&mut self) -> f64 {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            let is_num_char = matches!(b,
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E');
            if !is_num_char {
                break;
            }
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        s.parse::<f64>().unwrap_or_else(|_| panic!("bad number {}", s))
    }

    fn parse_value(&mut self) -> ExpectedValue {
        self.expect(b'{');
        self.skip_ws();
        let key = self.parse_string();
        self.expect(b':');
        let result = match key.as_str() {
            "num" => ExpectedValue::Num(self.parse_number()),
            "type" => ExpectedValue::Type(self.parse_string()),
            other => panic!("unexpected key {} in expected.json entry", other),
        };
        self.expect(b'}');
        result
    }

    fn parse_array(&mut self) -> Vec<ExpectedValue> {
        self.expect(b'[');
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == b']' {
            self.pos += 1;
            return out;
        }
        loop {
            out.push(self.parse_value());
            self.skip_ws();
            let b = self.peek();
            if b == b',' {
                self.pos += 1;
            } else if b == b']' {
                self.pos += 1;
                return out;
            } else {
                panic!("expected ',' or ']' at byte {}", self.pos);
            }
        }
    }

    fn parse_object(&mut self) -> Vec<(String, Vec<ExpectedValue>)> {
        self.expect(b'{');
        let mut out = Vec::new();
        self.skip_ws();
        if self.peek() == b'}' {
            self.pos += 1;
            return out;
        }
        loop {
            self.skip_ws();
            let key = self.parse_string();
            self.expect(b':');
            let arr = self.parse_array();
            out.push((key, arr));
            self.skip_ws();
            let b = self.peek();
            if b == b',' {
                self.pos += 1;
            } else if b == b'}' {
                self.pos += 1;
                return out;
            } else {
                panic!("expected ',' or '}}' at byte {}", self.pos);
            }
        }
    }
}

fn load_expected() -> Vec<(String, Vec<ExpectedValue>)> {
    let path = examples_dir().join("expected.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
    let mut parser = JsonParser::new(&text);
    parser.parse_object()
}

#[test]
fn every_example_file_is_in_expected_json() {
    let on_disk = list_lino_files(&examples_dir());
    let expected = load_expected();
    let expected_keys: Vec<String> = expected.iter().map(|(k, _)| k.clone()).collect();
    for file in &on_disk {
        assert!(
            expected_keys.contains(file),
            "{} is missing from expected.json",
            file
        );
    }
    for key in &expected_keys {
        assert!(
            on_disk.contains(key),
            "expected.json references missing file {}",
            key
        );
    }
}

#[test]
fn every_example_runs_and_matches_expected_outputs() {
    let expected = load_expected();
    let dir = examples_dir();
    let mut failures: Vec<String> = Vec::new();

    for (file, expected_results) in &expected {
        let path = dir.join(file);
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("could not read {}: {}", path.display(), e));
        let actual = run_typed(&text, None);

        if actual.len() != expected_results.len() {
            failures.push(format!(
                "{}: expected {} results, got {}",
                file,
                expected_results.len(),
                actual.len()
            ));
            continue;
        }

        for (i, (got, exp)) in actual.iter().zip(expected_results.iter()).enumerate() {
            match (got, exp) {
                (RunResult::Num(n), ExpectedValue::Num(en)) => {
                    if (n - en).abs() >= 1e-9 {
                        failures.push(format!(
                            "{}[{}]: expected {}, got {} (diff {})",
                            file,
                            i,
                            en,
                            n,
                            (n - en).abs()
                        ));
                    }
                }
                (RunResult::Type(s), ExpectedValue::Type(es)) => {
                    if s != es {
                        failures.push(format!(
                            "{}[{}]: expected type {:?}, got {:?}",
                            file, i, es, s
                        ));
                    }
                }
                (RunResult::Num(n), ExpectedValue::Type(es)) => failures.push(format!(
                    "{}[{}]: expected type {:?}, got numeric {}",
                    file, i, es, n
                )),
                (RunResult::Type(s), ExpectedValue::Num(en)) => failures.push(format!(
                    "{}[{}]: expected numeric {}, got type {:?}",
                    file, i, en, s
                )),
            }
        }
    }

    assert!(
        failures.is_empty(),
        "shared example mismatches:\n  {}",
        failures.join("\n  ")
    );
}
