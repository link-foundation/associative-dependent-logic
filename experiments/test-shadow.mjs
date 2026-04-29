import { evaluate, evaluateFile } from '../js/src/rml-links.mjs';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rml-shadow-'));

// Test A: import a lib that defines `myop` as an op alias, then redefine
fs.writeFileSync(path.join(dir, 'lib.lino'), `(myop: avg)
`);
fs.writeFileSync(path.join(dir, 'main.lino'), `(import "lib.lino")
(myop: max)
(? (myop 0.2 0.4 0.8))
`);
const outA = evaluateFile(path.join(dir, 'main.lino'));
console.log('Test A (op shadow):');
console.log('  diagnostics:', outA.diagnostics);
console.log('  results:', outA.results);

// Test B: import a lib that uses namespace, then redefine the qualified name
fs.writeFileSync(path.join(dir, 'lib2.lino'), `(namespace classical)
(and: min)
`);
fs.writeFileSync(path.join(dir, 'main2.lino'), `(import "lib2.lino" as cl)
(cl.and: max)
(? (cl.and 0.2 0.4 0.8))
`);
const outB = evaluateFile(path.join(dir, 'main2.lino'));
console.log('Test B (qualified rebind):');
console.log('  diagnostics:', outB.diagnostics);
console.log('  results:', outB.results);

// Test C: same alias used twice -> E009
fs.writeFileSync(path.join(dir, 'lib3a.lino'), `(namespace foo)
(x: max)
`);
fs.writeFileSync(path.join(dir, 'lib3b.lino'), `(namespace bar)
(x: min)
`);
fs.writeFileSync(path.join(dir, 'main3.lino'), `(import "lib3a.lino" as a)
(import "lib3b.lino" as a)
(? (a.x 0.2 0.5))
`);
const outC = evaluateFile(path.join(dir, 'main3.lino'));
console.log('Test C (alias collision):');
console.log('  diagnostics:', outC.diagnostics);
console.log('  results:', outC.results);

// Test D: term shadow
fs.writeFileSync(path.join(dir, 'lib4.lino'), `(foo: foo is foo)
`);
fs.writeFileSync(path.join(dir, 'main4.lino'), `(import "lib4.lino")
(foo: foo is foo)
(? (foo = foo))
`);
const outD = evaluateFile(path.join(dir, 'main4.lino'));
console.log('Test D (term shadow):');
console.log('  diagnostics:', outD.diagnostics);
console.log('  results:', outD.results);

fs.rmSync(dir, { recursive: true, force: true });
