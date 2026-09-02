#!/usr/bin/env node
import readline from 'node:readline';
import { readFile } from 'node:fs/promises';
import { tokenize, parse } from '../../upstream/vendor/dist/index.js';
import { emit } from '../ast-snapshot/ast-serialize.mjs';

const lock = JSON.parse(await readFile('upstream/upstream-lock.json', 'utf8'));
if (process.version !== lock.nodeVersion) throw new Error(`Node oracle version ${process.version} does not match lock ${lock.nodeVersion}`);
const UPSTREAM_MISSING_TOKEN_MESSAGE = "Cannot read properties of undefined (reading 'type')";
const reply = result => process.stdout.write(`${JSON.stringify(result)}\n`);
for await (const line of readline.createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  try {
    const candidate = JSON.parse(line);
    const source = Buffer.from(candidate.source, 'base64').toString('utf16le');
    if (source.length !== candidate.sourceCodeUnits) throw new Error('HARNESS source length differs');
    try {
      const ast = parse(tokenize(source, { trim_blocks: candidate.trimBlocks, lstrip_blocks: candidate.lstripBlocks }));
      reply({ id: candidate.id, result: 'PARSED', ast: candidate.family === 'grammar' ? Buffer.from(emit(ast)).toString('base64') : undefined });
    } catch (error) {
      // Pinned upstream's expect() throws plain Error("Parser Error: ..."), not SyntaxError.
      const message = String(error.message ?? error);
      const result = error instanceof RangeError ? 'LIMIT' : error instanceof SyntaxError
        || (error instanceof Error && (message.startsWith('Parser Error:') || message === UPSTREAM_MISSING_TOKEN_MESSAGE)) ? 'SYNTAX' : 'OTHER_ERROR';
      reply({ id: candidate.id, result, message, ...(result === 'OTHER_ERROR' ? {detail: String(error)} : {}) });
    }
  } catch (error) {
    reply({ id: null, result: 'HARNESS', detail: String(error) });
  }
}
