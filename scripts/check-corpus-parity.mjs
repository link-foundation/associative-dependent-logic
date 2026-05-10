#!/usr/bin/env node
import { existsSync, readdirSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');

function normalizeOutput(text) {
  return String(text ?? '').replace(/\r\n/g, '\n');
}

function statusOf(result) {
  if (result.error) return `spawn error: ${result.error.message}`;
  if (result.signal) return `signal ${result.signal}`;
  return String(result.status ?? 0);
}

function runCommand(command, args) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
  });

  return {
    status: result.status,
    signal: result.signal,
    stdout: normalizeOutput(result.stdout),
    stderr: normalizeOutput(result.stderr),
    error: result.error ?? null,
  };
}

export function compareCommandResults(jsResult, rustResult) {
  const failures = [];
  if (statusOf(jsResult) !== statusOf(rustResult)) {
    failures.push(`exit status differs: js=${statusOf(jsResult)} rust=${statusOf(rustResult)}`);
  }
  if (jsResult.stdout !== rustResult.stdout) {
    failures.push([
      'stdout differs:',
      '--- JavaScript stdout ---',
      jsResult.stdout || '<empty>',
      '--- Rust stdout ---',
      rustResult.stdout || '<empty>',
    ].join('\n'));
  }
  if (jsResult.stderr !== rustResult.stderr) {
    failures.push([
      'stderr differs:',
      '--- JavaScript stderr ---',
      jsResult.stderr || '<empty>',
      '--- Rust stderr ---',
      rustResult.stderr || '<empty>',
    ].join('\n'));
  }
  return failures;
}

function parseArgs(argv) {
  const options = {
    corpusDir: resolve(repoRoot, 'test-corpus'),
    jsCli: resolve(repoRoot, 'js', 'src', 'rml-links.mjs'),
    rustBin: resolve(repoRoot, 'rust', 'target', 'debug', process.platform === 'win32' ? 'rml.exe' : 'rml'),
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      return { ...options, help: true };
    }
    if (arg === '--corpus-dir' && argv[i + 1]) {
      options.corpusDir = resolve(repoRoot, argv[++i]);
      continue;
    }
    if (arg === '--js-cli' && argv[i + 1]) {
      options.jsCli = resolve(repoRoot, argv[++i]);
      continue;
    }
    if (arg === '--rust-bin' && argv[i + 1]) {
      options.rustBin = resolve(repoRoot, argv[++i]);
      continue;
    }
    throw new Error(`unknown option: ${arg}`);
  }

  return options;
}

function listCorpusFiles(corpusDir) {
  return readdirSync(corpusDir)
    .filter((name) => name.endsWith('.lino') && name !== 'expected.lino')
    .sort();
}

function printHelp() {
  console.log([
    'Usage: node scripts/check-corpus-parity.mjs [options]',
    '',
    'Runs every root test-corpus/*.lino input through the JavaScript and Rust CLIs',
    'and fails if exit status, stdout, or stderr diverge.',
    '',
    'Options:',
    '  --corpus-dir <dir>  Corpus directory, defaults to test-corpus',
    '  --js-cli <file>     JavaScript CLI, defaults to js/src/rml-links.mjs',
    '  --rust-bin <file>   Built Rust CLI, defaults to rust/target/debug/rml',
  ].join('\n'));
}

export function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  if (options.help) {
    printHelp();
    return 0;
  }

  if (!existsSync(options.jsCli)) {
    throw new Error(`JavaScript CLI not found: ${options.jsCli}`);
  }
  if (!existsSync(options.rustBin)) {
    throw new Error(`Rust CLI not found: ${options.rustBin}\nRun: cargo build --manifest-path rust/Cargo.toml --bin rml`);
  }

  const files = listCorpusFiles(options.corpusDir);
  if (files.length === 0) {
    throw new Error(`no .lino corpus files found in ${options.corpusDir}`);
  }

  const failures = [];
  for (const file of files) {
    const filePath = join(options.corpusDir, file);
    const jsResult = runCommand(process.execPath, [options.jsCli, filePath]);
    const rustResult = runCommand(options.rustBin, [filePath]);
    const fileFailures = compareCommandResults(jsResult, rustResult);
    if (fileFailures.length > 0) {
      failures.push(`${file}\n  ${fileFailures.join('\n  ')}`);
    }
  }

  if (failures.length > 0) {
    console.error(`Corpus parity failed for ${failures.length} file(s):\n\n${failures.join('\n\n')}`);
    return 1;
  }

  console.log(`Corpus parity passed for ${files.length} file(s).`);
  return 0;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    process.exitCode = main();
  } catch (err) {
    console.error(err && err.message ? err.message : String(err));
    process.exitCode = 1;
  }
}
