import { evaluate, synth } from '../src/rml-links.mjs';

// Test 1: forall as type for succ desugars to Pi
const r1 = evaluate(`
(Natural: (Type 0) Natural)
(succ: (forall (Natural n) Natural))
(? (succ of (Pi (Natural n) Natural)))
(? (type of succ))
`);
console.log('Test 1 (forall in declaration):', JSON.stringify(r1));

// Test 2: forall as a standalone form
const r2 = evaluate(`
(Natural: (Type 0) Natural)
(? (forall (Natural n) Natural))
(? ((forall (Natural n) Natural) of (Type 0)))
`);
console.log('Test 2 (forall standalone):', JSON.stringify(r2));

// Test 3: forall in synth/check
const r3 = synth(['forall', ['Natural', 'n'], 'Natural']);
console.log('Test 3 (synth):', JSON.stringify(r3));

// Test 4: nested forall in lambda body type
const r4 = evaluate(`
(Natural: (Type 0) Natural)
(Boolean: (Type 0) Boolean)
(eq: (forall (Natural a) (forall (Natural b) Boolean)))
(? (type of eq))
`);
console.log('Test 4 (nested forall):', JSON.stringify(r4));
