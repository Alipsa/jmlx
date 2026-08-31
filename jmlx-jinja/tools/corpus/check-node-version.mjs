#!/usr/bin/env node
import { readFile } from 'node:fs/promises';

if (process.argv.length !== 4 || process.argv[2] !== '--lock') {
  throw new Error('Usage: check-node-version.mjs --lock <json>');
}
const lock = JSON.parse(await readFile(process.argv[3], 'utf8'));
if (process.version !== lock.nodeVersion) {
  throw new Error(`Node oracle version ${process.version} does not match lock ${lock.nodeVersion}`);
}
console.log(`Node oracle version ${process.version}`);
