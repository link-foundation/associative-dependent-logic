import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const extensionRoot = path.resolve(scriptDir, '..');
const repoRoot = path.resolve(extensionRoot, '..');
const sourceDir = path.join(repoRoot, 'js', 'src');
const targetDir = path.join(extensionRoot, 'server');

if (!fs.existsSync(sourceDir)) {
  throw new Error(`Cannot find RML JavaScript sources at ${sourceDir}`);
}

fs.mkdirSync(targetDir, { recursive: true });
for (const entry of fs.readdirSync(targetDir)) {
  if (entry.endsWith('.mjs')) {
    fs.unlinkSync(path.join(targetDir, entry));
  }
}

let copied = 0;
for (const entry of fs.readdirSync(sourceDir)) {
  if (!entry.endsWith('.mjs')) continue;
  fs.copyFileSync(path.join(sourceDir, entry), path.join(targetDir, entry));
  copied += 1;
}

console.log(`Copied ${copied} RML language server source files to ${path.relative(extensionRoot, targetDir)}.`);
