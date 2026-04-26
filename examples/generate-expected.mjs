#!/usr/bin/env node
// Internal helper used to (re)generate examples/expected.json.
// Runs every .lino file under examples/ through the JS implementation and
// records the canonical output. Both implementations must reproduce these
// outputs — the JS and Rust example tests assert against this file.
//
// Usage: node examples/generate-expected.mjs

import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { run } from '../js/src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));

const files = readdirSync(here)
  .filter((f) => f.endsWith('.lino'))
  .sort();

const expected = {};
for (const file of files) {
  const text = readFileSync(join(here, file), 'utf8');
  const results = run(text);
  expected[file] = results.map((v) => (typeof v === 'string' ? { type: v } : { num: v }));
}

writeFileSync(join(here, 'expected.json'), JSON.stringify(expected, null, 2) + '\n');
console.log(`Wrote expected.json (${Object.keys(expected).length} files)`);
