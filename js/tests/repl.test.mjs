// Tests for the interactive REPL (issue #29).
// Exercises the in-process Repl class so we don't have to spin up a real
// readline TTY.  The Rust suite mirrors these in rust/tests/repl_tests.rs.

import { describe, it } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  Repl,
  formatEnv,
  makeCompleter,
  envCompletionCandidates,
  HELP_TEXT,
} from '../src/rml-repl.mjs';

describe('Repl preserves Env state across feeds', () => {
  it('runs the case-study session: declare, assign, query', () => {
    const repl = new Repl();
    let step = repl.feed('(a: a is a)');
    assert.strictEqual(step.error, '');
    step = repl.feed('((a = a) has probability 1)');
    assert.strictEqual(step.error, '');
    step = repl.feed('(? (a = a))');
    assert.strictEqual(step.error, '');
    assert.strictEqual(step.output, '1');
    assert.strictEqual(step.exit, false);
  });

  it('blank lines are no-ops and do not disturb state', () => {
    const repl = new Repl();
    repl.feed('(a: a is a)');
    const blank = repl.feed('   ');
    assert.deepStrictEqual(blank, { output: '', error: '', exit: false });
    assert.ok(repl.env.terms.has('a'));
  });

  it('a later error does not lose earlier results in the same feed', () => {
    const repl = new Repl();
    const step = repl.feed('(valence: 2)\n(p has probability 1)\n(? p)\n(=: missing identity)');
    assert.strictEqual(step.output, '1');
    assert.ok(step.error.includes('E001'), step.error);
  });
});

describe('Meta-commands', () => {
  it(':help prints the help text', () => {
    const repl = new Repl();
    const step = repl.feed(':help');
    assert.strictEqual(step.output, HELP_TEXT);
    assert.strictEqual(step.error, '');
  });

  it(':? is an alias for :help', () => {
    const repl = new Repl();
    const step = repl.feed(':?');
    assert.strictEqual(step.output, HELP_TEXT);
  });

  it(':quit and :exit request termination', () => {
    const repl = new Repl();
    assert.strictEqual(repl.feed(':quit').exit, true);
    assert.strictEqual(repl.feed(':exit').exit, true);
  });

  it(':reset clears terms and transcript', () => {
    const repl = new Repl();
    repl.feed('(a: a is a)');
    assert.ok(repl.env.terms.has('a'));
    assert.ok(repl.transcript.length > 0);
    const step = repl.feed(':reset');
    assert.strictEqual(step.output, 'Env reset.');
    assert.strictEqual(repl.env.terms.has('a'), false);
    assert.strictEqual(repl.transcript.length, 0);
  });

  it(':env reports range, valence, terms, and assignments', () => {
    const repl = new Repl();
    repl.feed('(a: a is a)');
    repl.feed('((a = a) has probability 1)');
    const step = repl.feed(':env');
    assert.ok(step.output.includes('range:'), step.output);
    assert.ok(step.output.includes('valence:'), step.output);
    assert.ok(step.output.includes('terms:'), step.output);
    assert.ok(step.output.includes('a'), step.output);
    assert.ok(step.output.includes('assignments:'), step.output);
  });

  it(':load reads a file into the running env', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rml-repl-load-'));
    const file = path.join(tmp, 'kb.lino');
    fs.writeFileSync(file, '(a: a is a)\n((a = a) has probability 1)\n');
    try {
      const repl = new Repl({ cwd: tmp });
      const loaded = repl.feed(`:load ${file}`);
      assert.strictEqual(loaded.error, '');
      const queried = repl.feed('(? (a = a))');
      assert.strictEqual(queried.output, '1');
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it(':load reports a clear error for missing files', () => {
    const repl = new Repl();
    const step = repl.feed(':load /no/such/path.lino');
    assert.ok(step.error.includes(':load failed'), step.error);
    assert.strictEqual(step.output, '');
  });

  it(':load with no argument reports an error', () => {
    const repl = new Repl();
    const step = repl.feed(':load');
    assert.ok(step.error.includes(':load requires'), step.error);
  });

  it(':save writes the transcript to disk', () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rml-repl-save-'));
    const file = path.join(tmp, 'session.lino');
    try {
      const repl = new Repl({ cwd: tmp });
      repl.feed('(a: a is a)');
      repl.feed('((a = a) has probability 1)');
      const saved = repl.feed(`:save ${file}`);
      assert.strictEqual(saved.error, '');
      assert.ok(saved.output.includes('Saved'), saved.output);
      const text = fs.readFileSync(file, 'utf8');
      assert.ok(text.includes('(a: a is a)'), text);
      assert.ok(text.includes('((a = a) has probability 1)'), text);
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  });

  it(':save with no argument reports an error', () => {
    const repl = new Repl();
    const step = repl.feed(':save');
    assert.ok(step.error.includes(':save requires'), step.error);
  });

  it('unknown meta-commands surface a friendly error', () => {
    const repl = new Repl();
    const step = repl.feed(':bogus');
    assert.ok(step.error.includes('Unknown meta-command'), step.error);
    assert.ok(step.error.includes(':bogus'), step.error);
  });
});

describe('Tab-completion', () => {
  it('offers meta-commands when the line starts with `:`', () => {
    const repl = new Repl();
    const completer = makeCompleter(() => repl.env);
    const [hits, prefix] = completer(':lo');
    assert.strictEqual(prefix, ':lo');
    assert.ok(hits.includes(':load'), hits);
  });

  it('offers identifiers declared in the env', () => {
    const repl = new Repl();
    repl.feed('(apple: apple is apple)');
    const candidates = envCompletionCandidates(repl.env);
    assert.ok(candidates.includes('apple'), candidates);
  });

  it('returns built-in keywords even on an empty env', () => {
    const candidates = envCompletionCandidates(new Repl().env);
    assert.ok(candidates.includes('probability'), candidates);
    assert.ok(candidates.includes('lambda'), candidates);
  });

  it('completer narrows by prefix at the cursor', () => {
    const repl = new Repl();
    repl.feed('(apple: apple is apple)');
    const completer = makeCompleter(() => repl.env);
    const [hits] = completer('(? app');
    assert.ok(hits.includes('apple'), hits);
  });
});

describe('formatEnv()', () => {
  it('shows continuous valence by default', () => {
    const repl = new Repl();
    const text = formatEnv(repl.env);
    assert.ok(text.includes('valence:  continuous'), text);
  });

  it('shows numeric valence when set', () => {
    const repl = new Repl();
    repl.feed('(valence: 2)');
    const text = formatEnv(repl.env);
    assert.ok(text.includes('valence:  2'), text);
  });
});
