import { readFileSync } from 'node:fs';
import { evaluate } from '../js/src/rml-links.mjs';

const body = readFileSync('examples/nat-links.lino', 'utf8');
const src = `(strict-foundation pure-links)\n(with-foundation nat-links\n${body}\n)`;
const out = evaluate(src);

console.log('Diagnostics count:', out.diagnostics.length);
for (const d of out.diagnostics) {
  console.log(`  ${d.code}: ${d.message}`);
}
console.log('Results count:', out.results.length);
