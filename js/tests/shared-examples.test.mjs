// Walks the repository-root /examples folder and runs every .lino file
// through the JavaScript implementation. Asserts the output matches the
// canonical fixtures in /examples/expected.json.
//
// The Rust integration tests assert against the same fixtures file, so
// any drift between the two implementations fails both test suites.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { run } from '../src/rml-links.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const examplesDir = resolve(here, '..', '..', 'examples');

const expected = JSON.parse(
  readFileSync(join(examplesDir, 'expected.json'), 'utf8')
);

const lineFiles = readdirSync(examplesDir)
  .filter((f) => f.endsWith('.lino'))
  .sort();

describe('shared examples (root /examples folder)', () => {
  it('every .lino file is covered by expected.json', () => {
    const missing = lineFiles.filter((f) => !(f in expected));
    assert.deepStrictEqual(missing, [],
      `expected.json is missing entries for: ${missing.join(', ')}`);
  });

  it('expected.json has no orphan entries', () => {
    const onDisk = new Set(lineFiles);
    const orphans = Object.keys(expected).filter((f) => !onDisk.has(f));
    assert.deepStrictEqual(orphans, [],
      `expected.json references missing files: ${orphans.join(', ')}`);
  });

  for (const file of lineFiles) {
    describe(file, () => {
      const text = readFileSync(join(examplesDir, file), 'utf8');
      const results = run(text);
      const expectedResults = expected[file];

      it('produces the expected number of results', () => {
        assert.strictEqual(results.length, expectedResults.length,
          `expected ${expectedResults.length} results, got ${results.length}`);
      });

      for (let i = 0; i < expectedResults.length; i++) {
        const exp = expectedResults[i];
        it(`result[${i}] matches expected`, () => {
          const actual = results[i];
          if ('type' in exp) {
            assert.strictEqual(typeof actual, 'string',
              `expected type result, got numeric ${actual}`);
            assert.strictEqual(actual, exp.type);
          } else {
            assert.strictEqual(typeof actual, 'number',
              `expected numeric result, got type ${actual}`);
            const diff = Math.abs(actual - exp.num);
            assert.ok(diff < 1e-9,
              `expected ${exp.num}, got ${actual} (diff ${diff})`);
          }
        });
      }
    });
  }
});
