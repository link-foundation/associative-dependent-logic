import { describe, it } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function readJson(relativePath) {
  return JSON.parse(fs.readFileSync(path.join(extensionRoot, relativePath), 'utf8'));
}

describe('VS Code extension manifest', () => {
  it('contributes the lino language, grammar, language config, and LSP activation', () => {
    const manifest = readJson('package.json');
    const language = manifest.contributes.languages.find(entry => entry.id === 'lino');
    assert.ok(language, 'missing lino language contribution');
    assert.ok(language.extensions.includes('.lino'));
    assert.strictEqual(language.configuration, './language-configuration.json');

    const grammar = manifest.contributes.grammars.find(entry => entry.language === 'lino');
    assert.ok(grammar, 'missing lino grammar contribution');
    assert.strictEqual(grammar.scopeName, 'source.lino');
    assert.ok(fs.existsSync(path.join(extensionRoot, grammar.path)));

    assert.ok(manifest.activationEvents.includes('onLanguage:lino'));
    assert.strictEqual(manifest.main, './src/extension.js');
    assert.ok(manifest.contributes.configuration.properties['rml.server.command']);
  });

  it('ships syntax highlighting rules for LiNo comments, definitions, keywords, and strings', () => {
    const grammar = readJson('syntaxes/lino.tmLanguage.json');
    const patterns = JSON.stringify(grammar);
    assert.match(patterns, /comment\.line\.number-sign\.lino/);
    assert.match(patterns, /entity\.name\.function\.definition\.lino/);
    assert.match(patterns, /keyword\.control\.lino/);
    assert.match(patterns, /string\.quoted\.double\.lino/);
  });
});
