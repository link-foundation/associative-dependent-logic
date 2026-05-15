import { synth, check, Env, evalNode } from '../src/rml-links.mjs';

const env = new Env();
evalNode(['Natural:', ['Type', '0'], 'Natural'], env);
evalNode(['zero:', 'Natural', 'zero'], env);
evalNode(['identity:', 'lambda', ['Natural', 'x'], 'x'], env);

console.log('synth zero =>', JSON.stringify(synth('zero', env)));
console.log('synth identity =>', JSON.stringify(synth('identity', env)));
console.log('synth (apply identity zero) =>', JSON.stringify(synth(['apply','identity','zero'], env)));
console.log('check zero against Natural =>', JSON.stringify(check('zero', 'Natural', env)));
console.log('check zero against Boolean =>', JSON.stringify(check('zero', 'Boolean', env)));
console.log('check identity against Pi =>', JSON.stringify(check('identity', ['Pi',['Natural','x'],'Natural'], env)));
console.log('synth unknown =>', JSON.stringify(synth('mystery', env)));
console.log('check lambda against non-Pi =>', JSON.stringify(check(['lambda',['Natural','x'],'x'], 'Natural', env)));
