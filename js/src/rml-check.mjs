#!/usr/bin/env node
// rml-check — independent proof-replay checker (issue #36).
//
// CLI front-end for the `check.mjs` module. Reads a program file and a
// proof file, validates that every derivation in the proof file
// corresponds to a query in the program, and prints
// `OK: N derivations replayed.` on success.

import fs from 'node:fs';
import { checkProgram, isOk } from './check.mjs';

function main(argv) {
  if (argv.length !== 4) {
    process.stderr.write('Usage: rml-check <program.lino> <proofs.lino>\n');
    return 2;
  }
  const programPath = argv[2];
  const proofsPath = argv[3];

  let program;
  let proofs;
  try {
    program = fs.readFileSync(programPath, 'utf8');
  } catch (e) {
    process.stderr.write(`Error reading ${programPath}: ${e.message}\n`);
    return 1;
  }
  try {
    proofs = fs.readFileSync(proofsPath, 'utf8');
  } catch (e) {
    process.stderr.write(`Error reading ${proofsPath}: ${e.message}\n`);
    return 1;
  }

  const result = checkProgram(program, proofs);
  if (isOk(result)) {
    process.stdout.write(`OK: ${result.ok.length} derivations replayed.\n`);
    return 0;
  }
  for (const err of result.errors) {
    const path = err.path.length === 0 ? '<root>' : err.path.join(' / ');
    process.stderr.write(`FAIL [${path}]: ${err.message}\n`);
  }
  return 1;
}

process.exit(main(process.argv));
