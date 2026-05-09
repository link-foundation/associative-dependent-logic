// Regression tests for issue #83: the GitHub Pages site must include a
// browser playground with embedded examples and shareable URL state.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');
const PLAYGROUND_DIR = path.join(REPO_ROOT, 'docs', 'playground');

function playgroundUrl(rel) {
  return pathToFileURL(path.join(PLAYGROUND_DIR, rel)).href;
}

describe('browser playground static site', () => {
  it('ships the static assets needed by GitHub Pages', () => {
    for (const rel of [
      'index.html',
      'styles.css',
      'app.mjs',
      'examples.mjs',
      'url-state.mjs',
      'rml-playground-runtime.mjs',
    ]) {
      assert.ok(fs.existsSync(path.join(PLAYGROUND_DIR, rel)), `missing docs/playground/${rel}`);
    }
  });

  it('defines runnable embedded examples', async () => {
    const { PLAYGROUND_EXAMPLES, defaultExampleId } = await import(playgroundUrl('examples.mjs'));
    assert.ok(Array.isArray(PLAYGROUND_EXAMPLES));
    assert.ok(PLAYGROUND_EXAMPLES.length >= 3, 'expected at least three examples');
    assert.ok(PLAYGROUND_EXAMPLES.some((example) => example.id === defaultExampleId));
    for (const example of PLAYGROUND_EXAMPLES) {
      assert.equal(typeof example.id, 'string');
      assert.equal(typeof example.title, 'string');
      assert.equal(typeof example.source, 'string');
      assert.ok(example.source.includes('(?'), `${example.id} should include a query`);
    }
  });

  it('round-trips source through URL hash state', async () => {
    const {
      decodePlaygroundState,
      encodePlaygroundState,
      normalizePlaygroundState,
    } = await import(playgroundUrl('url-state.mjs'));
    const original = {
      example: 'probability',
      source: '(rain: rain is rain)\n((rain = true) has probability 0.3)\n(? (rain = true))\n',
    };
    const encoded = encodePlaygroundState(original);
    assert.match(encoded, /^#state=/);
    assert.deepStrictEqual(decodePlaygroundState(encoded), normalizePlaygroundState(original));
  });

  it('evaluates every embedded example in the browser runtime', async () => {
    const { PLAYGROUND_EXAMPLES } = await import(playgroundUrl('examples.mjs'));
    const { evaluate } = await import(playgroundUrl('rml-playground-runtime.mjs'));
    for (const example of PLAYGROUND_EXAMPLES) {
      const out = evaluate(example.source, { file: `${example.id}.lino` });
      assert.deepStrictEqual(
        out.diagnostics.map((diagnostic) => diagnostic.code),
        [],
        `${example.id} should evaluate without diagnostics`,
      );
      assert.ok(out.results.length > 0, `${example.id} should produce results`);
    }
  });
});
