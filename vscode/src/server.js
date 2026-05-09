const fs = require('node:fs');
const path = require('node:path');

const LINO_DOCUMENT_SELECTOR = [
  { scheme: 'file', language: 'lino' },
  { scheme: 'untitled', language: 'lino' },
];

function configurationValue(configuration, key, fallback) {
  if (configuration && typeof configuration.get === 'function') {
    return configuration.get(key, fallback);
  }
  if (!configuration || typeof configuration !== 'object') return fallback;
  if (Object.prototype.hasOwnProperty.call(configuration, key)) {
    return configuration[key];
  }

  let current = configuration;
  for (const segment of key.split('.')) {
    if (!current || typeof current !== 'object') return fallback;
    current = current[segment];
  }
  return current === undefined ? fallback : current;
}

function stringArray(value) {
  if (Array.isArray(value)) return value.map(item => String(item));
  if (typeof value === 'string' && value.length > 0) return [value];
  return [];
}

function extensionPathFrom(contextOrPath) {
  if (typeof contextOrPath === 'string') return contextOrPath;
  if (contextOrPath && typeof contextOrPath.extensionPath === 'string') {
    return contextOrPath.extensionPath;
  }
  return process.cwd();
}

function findDefaultServerScript(extensionPath) {
  const candidates = [
    path.join(extensionPath, 'server', 'rml-lsp.mjs'),
    path.resolve(extensionPath, '..', 'js', 'src', 'rml-lsp.mjs'),
  ];
  return candidates.find(candidate => fs.existsSync(candidate)) || candidates[0];
}

function resolveServerCommand(contextOrPath, configuration = {}) {
  const extensionPath = extensionPathFrom(contextOrPath);
  const configuredCommand = configurationValue(configuration, 'server.command', '');
  const configuredArgs = stringArray(configurationValue(configuration, 'server.args', []));

  if (configuredCommand) {
    return {
      command: configuredCommand,
      args: configuredArgs,
      options: { cwd: extensionPath },
    };
  }

  const serverScript = findDefaultServerScript(extensionPath);
  return {
    command: process.execPath,
    args: [serverScript, ...configuredArgs],
    options: { cwd: path.dirname(serverScript) },
  };
}

function resolveServerOptions(contextOrPath, configuration = {}) {
  const command = resolveServerCommand(contextOrPath, configuration);
  return {
    run: command,
    debug: command,
  };
}

module.exports = {
  LINO_DOCUMENT_SELECTOR,
  findDefaultServerScript,
  resolveServerCommand,
  resolveServerOptions,
};
