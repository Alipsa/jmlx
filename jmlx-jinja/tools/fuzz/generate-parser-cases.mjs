#!/usr/bin/env node
/** Deterministic UTF-16 parser-candidate generator. PRNG: mulberry32-v1. */
import { writeFile } from 'node:fs/promises';

export const ALGORITHM = 'mulberry32-v1';
export const SMOKE_SEEDS = [0x5EEDC0DE, 0xC0FFEE42, 0x13579BDF];

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6D2B79F5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1) >>> 0;
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
  };
}

const pick = (random, values) => values[Math.floor(random() * values.length)];
const source64 = source => Buffer.from(source, 'utf16le').toString('base64');

function grammar(random, index) {
  const atoms = ['none', 'true', 'false', '0', '1', "'x'", '"y"'];
  function expression(depth = 0) {
    if (depth >= 12) return pick(random, atoms);
    const next = () => expression(depth + 1);
    return pick(random, [
      () => pick(random, atoms),
      () => `(${next()})`,
      () => `${next()} + ${next()}`,
      () => `${next()} == ${next()}`,
      () => `not (${next()})`,
      () => 'foo.bar[0:1]',
      () => 'range(range(1))',
    ])();
  }
  const value = expression();
  const templates = [
    () => `text-${index} {{ ${value}|default('x') }}`,
    () => `{%- set value = ${value} -%}{{ value }}`,
    () => `{% if ${value} is defined %}{{ ${value} if true else 'x' }}{% else %}no{% endif %}`,
    () => `{% for item in [1, 2] %}{{ item }}{% else %}empty{% endfor %}`,
    () => `{% macro hello(value='x') %}{{ value }}{% endmacro %}{{ hello('y') }}`,
    () => `{% filter default('x') %}value{% endfilter %}`,
    () => `{% macro wrapper() %}{{ caller() }}{% endmacro %}{% call wrapper() %}body{% endcall %}`,
    () => `{% if true %}{% for item in [1] %}{{ ${value} }}{% endfor %}{% endif %}`,
    () => `{# comment #}\r\n{{ ${value} }}`,
  ];
  return pick(random, templates)();
}

function hostile(random) {
  // Single-cause probes are deliberately short: random junk alone tends to fail for many reasons.
  const probes = ['{{ 1 \0 }}', '{{ 1 \uD800 }}', '{{ 1 ] }}', '{{ 1 ( }}', '{{ 1 foo[ }}', '{{ 1 " }}', '{% if true %}{% endif', '{{ 1 }}{%'];
  if (random() < 0.15) return pick(random, probes);
  const pieces = ['{{', '{%', '{#', '}}', '%}', '#}', "'", '"', '[', ']', '(', ')', '{', '}', '\0', '\r\n', '\r', '\n', '😀', '\uD800', '\uDC00', '...', '////', '!!!', 'text'];
  const count = 1 + Math.floor(random() * 32);
  let value = '';
  for (let i = 0; i < count; i++) value += pick(random, pieces);
  return value.slice(0, 512);
}

export function generate({ seed, count = 100, maxSourceCodeUnits = 512 }) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xFFFFFFFF) throw new Error('seed must be an unsigned 32-bit word');
  if (!Number.isInteger(count) || count < 1) throw new Error('count must be positive');
  if (!Number.isInteger(maxSourceCodeUnits) || maxSourceCodeUnits < 1 || maxSourceCodeUnits > 512) throw new Error('max-source-code-units must be 1..512');
  const random = rng(seed);
  const records = [{ protocol: 'hfjinja-parser-fuzz-v1', algorithm: ALGORITHM, seed: seed >>> 0 }];
  for (const family of ['grammar', 'hostile']) for (let index = 0; index < count; index++) {
    let raw = family === 'grammar' ? grammar(random, index) : hostile(random);
    // Grammar candidates promise upstream validity; retry rather than truncate a deep expression.
    let attempts = 0;
    while (family === 'grammar' && raw.length > maxSourceCodeUnits) {
      if (++attempts === 100)
        throw new Error(`Grammar retry budget exhausted: no candidate within ${maxSourceCodeUnits} code units after ${attempts} attempts`);
      raw = grammar(random, index);
    }
    const source = raw.slice(0, maxSourceCodeUnits);
    records.push({ id: `${(seed >>> 0).toString(16).padStart(8, '0')}-${family}-${index}`, family,
      source: source64(source), trimBlocks: random() < 0.5, lstripBlocks: random() < 0.5,
      sourceCodeUnits: source.length });
  }
  return records;
}

function argument(name, fallback) { const index = process.argv.indexOf(name); return index < 0 ? fallback : process.argv[index + 1]; }
if (import.meta.main) {
  const seedText = argument('--seed');
  if (seedText == null) throw new Error('Usage: generate-parser-cases.mjs --seed <u32> [--count N] [--max-source-code-units N] [--output path]');
  const seed = Number(seedText);
  const records = generate({ seed, count: Number(argument('--count', '100')), maxSourceCodeUnits: Number(argument('--max-source-code-units', '512')) });
  const output = records.map(record => JSON.stringify(record)).join('\n') + '\n';
  const target = argument('--output');
  if (target) await writeFile(target, output); else process.stdout.write(output);
}
