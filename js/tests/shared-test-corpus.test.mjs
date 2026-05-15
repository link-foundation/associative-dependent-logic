// Walks the repository-root /test-corpus folder and runs every .lino file
// through the JavaScript implementation. Asserts the output matches the
// canonical fixtures in /test-corpus/expected.lino (Links Notation).
//
// The Rust integration tests assert against the same fixtures file, so
// regression inputs cannot drift between implementations.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { evaluateFile, parseLino, tokenizeOne, parseOne, keyOf } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const corpusDir = resolve(here, '..', '..', 'test-corpus');

// Parse expected.lino into a Map<filename, ExpectedValue[]>.
// Each link is shaped (filename.lino: <result> <result> ...) where a numeric
// result is a bare number and a structured result is wrapped as (type <value>).
function loadExpected() {
  const text = readFileSync(join(corpusDir, 'expected.lino'), 'utf8');
  const map = new Map();
  for (const linkStr of parseLino(text)) {
    const ast = parseOne(tokenizeOne(linkStr));
    if (!Array.isArray(ast) || ast.length < 1 || typeof ast[0] !== 'string' || !ast[0].endsWith(':')) {
      throw new Error(`test-corpus/expected.lino: malformed entry ${linkStr}`);
    }
    const filename = ast[0].slice(0, -1);
    const values = ast.slice(1).map((node) => {
      if (Array.isArray(node) && node.length === 2 && node[0] === 'type') {
        return { type: typeof node[1] === 'string' ? node[1] : keyOf(node[1]) };
      }
      if (typeof node === 'string' && /^-?(\d+(\.\d+)?|\.\d+)$/.test(node)) {
        return { num: parseFloat(node) };
      }
      throw new Error(`test-corpus/expected.lino: unsupported result ${JSON.stringify(node)} in ${filename}`);
    });
    map.set(filename, values);
  }
  return map;
}

const expected = loadExpected();

const corpusFiles = readdirSync(corpusDir)
  .filter((f) => f.endsWith('.lino') && f !== 'expected.lino')
  .sort();

describe('shared test corpus (root /test-corpus folder)', () => {
  it('every .lino file is covered by expected.lino', () => {
    const missing = corpusFiles.filter((f) => !expected.has(f));
    assert.deepStrictEqual(missing, [],
      `test-corpus/expected.lino is missing entries for: ${missing.join(', ')}`);
  });

  it('expected.lino has no orphan entries', () => {
    const onDisk = new Set(corpusFiles);
    const orphans = [...expected.keys()].filter((f) => !onDisk.has(f));
    assert.deepStrictEqual(orphans, [],
      `test-corpus/expected.lino references missing files: ${orphans.join(', ')}`);
  });

  for (const file of corpusFiles) {
    describe(file, () => {
      const out = evaluateFile(join(corpusDir, file));
      const expectedResults = expected.get(file);

      it('has no diagnostics', () => {
        assert.deepStrictEqual(out.diagnostics, []);
      });

      it('produces the expected number of results', () => {
        assert.strictEqual(out.results.length, expectedResults.length,
          `expected ${expectedResults.length} results, got ${out.results.length}`);
      });

      for (let i = 0; i < expectedResults.length; i++) {
        const exp = expectedResults[i];
        it(`result[${i}] matches expected`, () => {
          const actual = out.results[i];
          if ('type' in exp) {
            assert.strictEqual(typeof actual, 'string',
              `expected structured result, got numeric ${actual}`);
            assert.strictEqual(actual, exp.type);
          } else {
            assert.strictEqual(typeof actual, 'number',
              `expected numeric result, got structured ${actual}`);
            const diff = Math.abs(actual - exp.num);
            assert.ok(diff < 1e-9,
              `expected ${exp.num}, got ${actual} (diff ${diff})`);
          }
        });
      }
    });
  }
});
