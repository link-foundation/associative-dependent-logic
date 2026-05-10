import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { compareCommandResults } from './check-corpus-parity.mjs';

function result({ status = 0, stdout = '', stderr = '', signal = null, error = null } = {}) {
  return { status, stdout, stderr, signal, error };
}

describe('corpus parity comparison', () => {
  it('accepts identical command results', () => {
    assert.deepStrictEqual(
      compareCommandResults(
        result({ stdout: '1\n(type of x)\n' }),
        result({ stdout: '1\n(type of x)\n' }),
      ),
      [],
    );
  });

  it('rejects output divergence', () => {
    const failures = compareCommandResults(
      result({ stdout: '1\n' }),
      result({ stdout: '0\n' }),
    );

    assert.equal(failures.length, 1);
    assert.match(failures[0], /stdout differs/);
  });

  it('rejects status divergence', () => {
    const failures = compareCommandResults(
      result({ status: 0 }),
      result({ status: 1 }),
    );

    assert.equal(failures.length, 1);
    assert.match(failures[0], /exit status differs/);
  });
});
