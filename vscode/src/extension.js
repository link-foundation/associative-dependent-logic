const vscode = require('vscode');
const { LanguageClient } = require('vscode-languageclient/node');
const {
  LINO_DOCUMENT_SELECTOR,
  resolveServerOptions,
} = require('./server');

let client;

function activate(context) {
  const configuration = vscode.workspace.getConfiguration('rml');
  const serverOptions = resolveServerOptions(context, configuration);
  const clientOptions = {
    documentSelector: LINO_DOCUMENT_SELECTOR,
    synchronize: {
      fileEvents: vscode.workspace.createFileSystemWatcher('**/*.lino'),
    },
  };

  client = new LanguageClient(
    'rml',
    'RML Language Server',
    serverOptions,
    clientOptions,
  );
  client.start();
}

function deactivate() {
  if (!client) return undefined;
  return client.stop();
}

module.exports = {
  activate,
  deactivate,
};
