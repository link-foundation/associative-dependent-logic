import { evaluate, evaluateFile } from '../js/src/rml-links.mjs';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rml-ns-'));

// Write a classical lib
fs.writeFileSync(path.join(dir, 'classical.lino'), `(namespace classical)
(and: min)
(or: max)
`);

// Test 1: aliased import
fs.writeFileSync(path.join(dir, 'main.lino'), `(import "classical.lino" as cl)
(? (cl.and 1 0))
(? (cl.or 1 0))
`);

const out1 = evaluateFile(path.join(dir, 'main.lino'));
console.log('Test 1 (aliased import):');
console.log('  diagnostics:', out1.diagnostics);
console.log('  results:', out1.results);

// Test 2: namespace, no alias — direct qualified access
fs.writeFileSync(path.join(dir, 'main2.lino'), `(import "classical.lino")
(? (classical.and 1 0))
(? (classical.or 1 0))
`);
const out2 = evaluateFile(path.join(dir, 'main2.lino'));
console.log('Test 2 (qualified direct access):');
console.log('  diagnostics:', out2.diagnostics);
console.log('  results:', out2.results);

// Test 3: shadowing
fs.writeFileSync(path.join(dir, 'lib3.lino'), `((true) has probability 0.7)
`);
fs.writeFileSync(path.join(dir, 'main3.lino'), `(import "lib3.lino")
(true: 0.3)
(? (true = true))
`);
const out3 = evaluateFile(path.join(dir, 'main3.lino'));
console.log('Test 3 (shadowing):');
console.log('  diagnostics:', out3.diagnostics);
console.log('  results:', out3.results);

fs.rmSync(dir, { recursive: true, force: true });
