#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { tokenize, parse } from '../../upstream/vendor/dist/index.js';
import { emit } from './ast-serialize.mjs';

const lock = JSON.parse(await readFile('upstream/upstream-lock.json', 'utf8'));
if (process.version !== lock.nodeVersion) {
  throw new Error(`Node oracle version ${process.version} does not match lock ${lock.nodeVersion}`);
}

const args = process.argv.slice(2);
if (args[0] !== '--fixtures' || !args[1]) throw new Error('Usage: snapshot.mjs --fixtures <path> [--check <path>]');
const path = args[1];
const fixtures = JSON.parse(await readFile(path, 'utf8'));

let output = '';
for (const fixture of fixtures) output += `=== ${fixture.name} ${JSON.stringify(fixture.source)}\n` + emit(parse(tokenize(fixture.source, { lstrip_blocks: true, trim_blocks: true })));
if (args[2] === '--check') {
  if (output !== await readFile(args[3], 'utf8')) throw new Error(`Stale AST snapshot: ${args[3]}`);
} else process.stdout.write(output);
