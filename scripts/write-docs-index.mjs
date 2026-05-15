#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const outDir = path.resolve(process.argv[2] || '_site');
fs.mkdirSync(outDir, { recursive: true });

const playgroundSource = path.join(repoRoot, 'docs', 'playground');
if (fs.existsSync(playgroundSource)) {
  fs.cpSync(playgroundSource, path.join(outDir, 'playground'), { recursive: true });
}

const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>relative-meta-logic API Reference</title>
  <style>
    :root {
      color-scheme: light dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    body {
      margin: 0;
      padding: 3rem 1.5rem;
    }
    main {
      max-width: 48rem;
      margin: 0 auto;
    }
    h1 {
      font-size: 2rem;
      margin: 0 0 0.5rem;
    }
    p {
      margin: 0 0 1.5rem;
    }
    ul {
      padding-left: 1.25rem;
    }
    li {
      margin: 0.5rem 0;
    }
  </style>
</head>
<body>
  <main>
    <h1>relative-meta-logic</h1>
    <p>Browser tools and generated reference documentation.</p>
    <ul>
      <li><a href="./playground/">Online playground</a></li>
      <li><a href="./api/js/">JavaScript API reference</a></li>
      <li><a href="./api/rust/rml/">Rust API reference</a></li>
    </ul>
  </main>
</body>
</html>
`;

fs.writeFileSync(path.join(outDir, 'index.html'), html, 'utf8');
fs.writeFileSync(path.join(outDir, '.nojekyll'), '', 'utf8');
