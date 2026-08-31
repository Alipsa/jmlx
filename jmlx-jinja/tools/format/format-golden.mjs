#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { Template, tokenize, parse } from '../../upstream/vendor/dist/index.js';
import { errorClassifier } from '../corpus/corpus.mjs';

const args = process.argv.slice(2);
if (args[0] !== '--vectors' || !args[1]) {
  throw new Error('Usage: format-golden.mjs --vectors <path> [--check <path>] [--report <path>] [--update <path>]');
}
const vectors = JSON.parse(await readFile(args[1], 'utf8'));
const lock = JSON.parse(await readFile('upstream/upstream-lock.json', 'utf8'));
if (process.version !== lock.nodeVersion) throw new Error(`Node oracle version ${process.version} does not match lock ${lock.nodeVersion}`);
const classifyError = await errorClassifier('tools/corpus/error-patterns-0.5.9.json', `${lock.package}@${lock.version}`);
const types = new Set();
function walk(value) {
  if (!value || typeof value !== 'object') return;
  if (value.constructor?.name && value.constructor.name !== 'Object' && value.constructor.name !== 'Array') types.add(value.constructor.name);
  if (value instanceof Map) value.forEach((item, key) => { walk(key); walk(item); });
  else if (Array.isArray(value)) value.forEach(walk);
  else Object.values(value).forEach(walk);
}
const output = vectors.map(vector => {
  if (!['preserves', 'upstream-diverges', 'reformat-fails', 'not-renderable'].includes(vector.roundTrip)) {
    throw new Error(`Unknown roundTrip expectation for ${vector.name}: ${vector.roundTrip}`);
  }
  const template = new Template(vector.source);
  const indent = vector.indent.default ? undefined : vector.indent.number ?? vector.indent.string;
  const formatted = indent === undefined ? template.format() : template.format({ indent });
  walk(parse(tokenize(vector.source, { lstrip_blocks: true, trim_blocks: true })));
  let originalRendered = null;
  let reformattedRendered = null;
  let reformattedError = null;
  let reformattedCategory = null;
  if (vector.roundTrip !== 'not-renderable') {
    originalRendered = template.render(vector.context ?? {});
    try {
      reformattedRendered = new Template(formatted).render(vector.context ?? {});
    } catch (error) {
      reformattedError = `${error.name}: ${error.message}`;
      if (vector.roundTrip === 'reformat-fails') reformattedCategory = classifyError(error.message);
    }
    if (vector.roundTrip === 'preserves' && (reformattedError !== null || originalRendered !== reformattedRendered)) {
      throw new Error(`Format round trip changed rendering for ${vector.name}`);
    }
    if (vector.roundTrip === 'upstream-diverges' && (reformattedError !== null || originalRendered === reformattedRendered)) {
      throw new Error(`Expected upstream round-trip divergence for ${vector.name}`);
    }
    if (vector.roundTrip === 'reformat-fails' && reformattedError === null) {
      throw new Error(`Expected formatted rendering to fail for ${vector.name}`);
    }
  }
  return {
    name: vector.name,
    source: vector.source,
    indent: vector.indent,
    roundTrip: vector.roundTrip,
    formatted,
    context: JSON.stringify(vector.context ?? {}),
    originalRendered,
    reformattedRendered,
    reformattedError,
    reformattedCategory,
  };
});
const text = JSON.stringify(output) + '\n';
const option = name => {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
};
const check = option('--check');
const update = option('--update');
const report = option('--report');
if (check) {
  if (text !== await readFile(check, 'utf8')) throw new Error(`Stale format golden: ${check}`);
  const allowed = new Set(Object.keys(JSON.parse(await readFile('upstream/ast-allowlist.json', 'utf8'))));
  for (const abstract of ['Statement', 'Expression', 'Literal']) allowed.delete(abstract);
  const missing = [...allowed].filter(type => !types.has(type));
  if (report) {
    await mkdir(dirname(report), { recursive: true });
    await writeFile(report, `# Template formatter AST coverage\n\nSeen: ${[...types].sort().join(', ')}\n\nMissing: ${missing.join(', ') || '(none)'}\n`);
  }
  if (missing.length) throw new Error(`Format vectors miss AST node types: ${missing.join(', ')}`);
} else if (update) {
  await writeFile(update, text);
} else process.stdout.write(text);
