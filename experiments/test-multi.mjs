import { evaluateFile } from '../js/src/rml-links.mjs';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rml-multi-'));

fs.writeFileSync(path.join(dir, 'la.lino'), `(namespace foo)
(x: max)
`);
fs.writeFileSync(path.join(dir, 'lb.lino'), `(namespace bar)
(x: min)
`);
fs.writeFileSync(path.join(dir, 'multi.lino'), `(import "la.lino" as a)
(import "lb.lino" as b)
(? (a.x 0.2 0.5))
(? (b.x 0.2 0.5))
`);
const out = evaluateFile(path.join(dir, 'multi.lino'));
console.log('diagnostics:', out.diagnostics);
console.log('results:', out.results);

// Now check just one
fs.writeFileSync(path.join(dir, 'simple.lino'), `(import "la.lino" as a)
(? (a.x 0.2 0.5))
`);
const out2 = evaluateFile(path.join(dir, 'simple.lino'));
console.log('simple diagnostics:', out2.diagnostics);
console.log('simple results:', out2.results);

// And: just call foo.x directly
fs.writeFileSync(path.join(dir, 'direct.lino'), `(import "la.lino")
(? (foo.x 0.2 0.5))
`);
const out3 = evaluateFile(path.join(dir, 'direct.lino'));
console.log('direct diagnostics:', out3.diagnostics);
console.log('direct results:', out3.results);

fs.rmSync(dir, { recursive: true, force: true });
