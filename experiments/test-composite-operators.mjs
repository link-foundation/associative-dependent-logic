// Test composite natural language operators: (both X and Y), (neither X nor Y)
import { run } from '../js/src/rml-links.mjs';

// Test (both X and Y)
console.log('=== both X and Y ===');
console.log('(both true and false):', run('(? (both true and false))'));       // expect 0.5
console.log('(both true and true):', run('(? (both true and true))'));         // expect 1
console.log('(both false and false):', run('(? (both false and false))'));     // expect 0

// Test (neither X nor Y)
console.log('\n=== neither X nor Y ===');
console.log('(neither true nor false):', run('(? (neither true nor false))')); // expect 0
console.log('(neither true nor true):', run('(? (neither true nor true))'));   // expect 1
console.log('(neither false nor false):', run('(? (neither false nor false))')); // expect 0

// Test variadic: (both A and B and C)
console.log('\n=== variadic ===');
console.log('(both true and true and false):', run('(? (both true and true and false))')); // avg(1,1,0)=0.666...
console.log('(neither true nor true nor false):', run('(? (neither true nor true nor false))')); // product(1,1,0)=0

// Test backward compatibility: old infix form
console.log('\n=== backward compat (old infix) ===');
console.log('(true both false):', run('(? (true both false))'));     // expect 0.5
console.log('(true neither false):', run('(? (true neither false))')); // expect 0

// Test backward compatibility: old prefix form
console.log('\n=== backward compat (old prefix) ===');
console.log('(both true false):', run('(? (both true false))'));     // expect 0.5
console.log('(neither true false):', run('(? (neither true false))')); // expect 0

// Test redefinability
console.log('\n=== redefinable ===');
console.log('(both: min) then (both true and false):', run('(both: min)\n(? (both true and false))')); // expect 0

console.log('\nAll tests passed!');
