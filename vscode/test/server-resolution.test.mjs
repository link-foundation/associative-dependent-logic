import { describe, it } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { LspClient } from './lsp-client.mjs';

const require = createRequire(import.meta.url);
const extensionRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { resolveServerCommand } = require('../src/server.js');

describe('VS Code extension LSP integration', () => {
  it('resolves an executable command for the bundled or checkout language server', () => {
    const server = resolveServerCommand(extensionRoot, {});
    assert.strictEqual(server.command, process.execPath);
    assert.ok(server.args[0].endsWith(path.join('rml-lsp.mjs')));
    assert.ok(fs.existsSync(server.args[0]), server.args[0]);
  });

  it('starts the resolved LSP and receives diagnostics and completions for .lino text', async t => {
    const server = resolveServerCommand(extensionRoot, {});
    const client = new LspClient(server.command, server.args, server.options || {});
    t.after(() => client.close());

    const init = await client.request('initialize', {
      processId: process.pid,
      rootUri: pathToFileURL(path.resolve(extensionRoot, '..')).href,
      capabilities: {},
    });
    assert.ifError(init.error);
    assert.deepStrictEqual(init.result.capabilities.textDocumentSync, {
      openClose: true,
      change: 1,
    });
    assert.strictEqual(init.result.capabilities.hoverProvider, true);
    assert.strictEqual(init.result.capabilities.definitionProvider, true);
    assert.ok(init.result.capabilities.completionProvider);

    client.notify('initialized', {});

    const uri = 'file:///workspace/vscode-extension-demo.lino';
    client.notify('textDocument/didOpen', {
      textDocument: {
        uri,
        languageId: 'lino',
        version: 1,
        text: '(=: missing_op identity)\n',
      },
    });
    const bad = await client.waitForNotification('textDocument/publishDiagnostics', p =>
      p.uri === uri && p.diagnostics.length === 1);
    assert.strictEqual(bad.params.diagnostics[0].code, 'E001');

    client.notify('textDocument/didChange', {
      textDocument: { uri, version: 2 },
      contentChanges: [{
        text: '(a: a is a)\n((a = a) has probability 1)\n(? (a = a))\n',
      }],
    });
    await client.waitForNotification('textDocument/publishDiagnostics', p =>
      p.uri === uri && p.diagnostics.length === 0);

    const completion = await client.request('textDocument/completion', {
      textDocument: { uri },
      position: { line: 2, character: 3 },
    });
    const labels = completion.result.items.map(item => item.label);
    assert.ok(labels.includes('a'), labels);
    assert.ok(labels.includes('probability'), labels);

    const shutdown = await client.request('shutdown', null);
    assert.strictEqual(shutdown.result, null);
    client.notify('exit', null);
  });
});
