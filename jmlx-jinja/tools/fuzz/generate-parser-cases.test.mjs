import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { generate } from './generate-parser-cases.mjs';

test('parser candidate protocol is deterministic and lossless', async () => {
  const records = generate({ seed: 0x5EEDC0DE, count: 100, maxSourceCodeUnits: 512 });
  const fixture = await readFile('tools/fuzz/testdata/parser-cases-5eedc0de.ndjson', 'utf8');
  assert.equal(records.map(JSON.stringify).join('\n') + '\n', fixture);
  assert.equal(new Set(records.slice(1).map(record => record.id)).size, records.length - 1);
  assert.deepEqual(new Set(records.slice(1).map(record => record.family)), new Set(['grammar', 'hostile']));
  assert.throws(() => generate({ seed: -1 }), /unsigned/);
  assert.throws(() => generate({ seed: 0x1_0000_0000 }), /unsigned/);
  assert.throws(() => generate({ seed: 1, maxSourceCodeUnits: 513 }), /max-source/);
  assert.throws(() => generate({ seed: 1, maxSourceCodeUnits: 1 }), /retry budget exhausted/);
  for (const record of records.slice(1)) {
    const source = Buffer.from(record.source, 'base64').toString('utf16le');
    assert.equal(source.length, record.sourceCodeUnits);
    assert.ok(source.length <= 512);
  }
});
