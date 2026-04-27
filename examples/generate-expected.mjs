#!/usr/bin/env node
// Internal helper used to (re)generate examples/expected.lino.
// Runs every .lino file under examples/ through the JS implementation and
// records the canonical output as Links Notation. Both implementations must
// reproduce these outputs — the JS and Rust example tests assert against this
// file.
//
// Format (Links Notation):
//   (<filename>.lino: <result> <result> ...)
//
// A numeric result is a bare number. A type result is the link
// (type <Name>). The order matches the order of `(? ...)` queries
// inside the corresponding example file.
//
// Usage: node examples/generate-expected.mjs

import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { run } from '../js/src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));

const HEADER = `# Canonical expected outputs for every shared \`.lino\` knowledge base.
#
# Format (Links Notation):
#   (<filename>.lino: <result> <result> ...)
#
# A numeric result is a bare number. A type result is the link
# (type <Name>). The order matches the order of \`(? ...)\` queries
# inside the corresponding example file.
#
# This file is the contract between the JavaScript and Rust
# implementations: both run every example through their own runtime
# and assert the output matches what is recorded here.
#
# Regenerate after intentional changes:
#   node examples/generate-expected.mjs
`;

function formatResult(v) {
  if (typeof v === 'string') return `(type ${v})`;
  return String(v);
}

const files = readdirSync(here)
  .filter((f) => f.endsWith('.lino') && f !== 'expected.lino')
  .sort();

const lines = [HEADER];
for (const file of files) {
  const text = readFileSync(join(here, file), 'utf8');
  const results = run(text).map(formatResult).join(' ');
  lines.push(`(${file}: ${results})`);
}

writeFileSync(join(here, 'expected.lino'), lines.join('\n\n') + '\n');
console.log(`Wrote expected.lino (${files.length} files)`);
