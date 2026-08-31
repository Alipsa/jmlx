import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

const categories = new Set([
  'SYNTAX', 'UNDEFINED_OR_ACCESS', 'TYPE', 'ARITY', 'VALUE', 'EXPLICIT_RAISE',
  'HOST_FUNCTION', 'HOST_CONVERSION', 'RESOURCE_LIMIT', 'OUTPUT',
]);
const recordKeys = new Set([
  'id', 'source', 'template', 'templateSha256', 'modelRepo', 'modelRevision', 'templatePath',
  'context', 'templateOptions', 'instant', 'zone', 'globals', 'expected',
]);
const lineNumber = Symbol('lineNumber');
const instantPattern = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.(\d{1,3}))?Z$/;
const canonicalZones = new Set(
  (await readFile(new URL('./iana-time-zones.txt', import.meta.url), 'utf8')).trim().split('\n'),
);

export async function readCorpus(path) {
  const content = await readFile(path, 'utf8');
  const records = [];
  for (const [index, line] of content.split(/\r?\n/).entries()) {
    if (!line) continue;
    try {
      const record = JSON.parse(line);
      if (!isObject(record)) throw new Error('record must be a JSON object');
      Object.defineProperty(record, lineNumber, {value: index + 1});
      records.push(record);
    } catch (error) {
      const description = error instanceof SyntaxError ? `invalid JSON: ${error.message}` : error.message;
      throw new Error(`${path}:${index + 1}: ${description}`);
    }
  }
  return records;
}

export function corpusLine(record, fallback) {
  return record[lineNumber] ?? fallback;
}

export function validateCorpus(records, label = 'corpus') {
  const ids = new Set();
  for (const [index, record] of records.entries()) {
    const recordLabel = `${label}:${corpusLine(record, index + 1)}`;
    validateRecord(record, recordLabel);
    if (ids.has(record.id)) throw new Error(`${recordLabel}: duplicate id ${record.id}`);
    ids.add(record.id);
  }
}

export function validateRecord(record, label = 'record') {
  if (!isObject(record)) throw new Error(`${label}: must be an object`);
  const unknownKeys = Object.keys(record).filter((key) => !recordKeys.has(key));
  if (unknownKeys.length) throw new Error(`${label}: unknown fields: ${unknownKeys.join(', ')}`);
  if (!nonBlank(record.id) || !nonBlank(record.source) || !isObject(record.context)) {
    throw new Error(`${label}: id, source, and object context are required`);
  }
  const text = typeof record.template === 'string';
  const hashOnly = record.templateSha256 !== undefined;
  if (text === hashOnly) throw new Error(`${label}: provide exactly one of template or templateSha256`);
  if (hashOnly && !sha256(record.templateSha256)) throw new Error(`${label}: templateSha256 must be 64 lowercase hex characters`);
  if (hashOnly && (!nonBlank(record.modelRepo)
      || !(typeof record.modelRevision === 'string' && /^[0-9a-f]{40}$/.test(record.modelRevision))
      || !nonBlank(record.templatePath))) {
    throw new Error(`${label}: hash-only records require modelRepo, 40-hex modelRevision, and templatePath`);
  }
  if (text && ['modelRepo', 'modelRevision', 'templatePath'].some((field) => record[field] !== undefined)) {
    throw new Error(`${label}: text records must not include hash-only provenance metadata`);
  }
  validateTemplateOptions(record.templateOptions, label);
  if (record.instant !== undefined && record.zone === undefined) {
    throw new Error(`${label}: instant requires an explicit zone`);
  }
  if (record.zone !== undefined && record.instant === undefined) {
    throw new Error(`${label}: zone requires an explicit instant`);
  }
  if (record.instant !== undefined && !validInstant(record.instant)) {
    throw new Error(`${label}: instant must be an ISO-8601 instant`);
  }
  if (record.zone !== undefined) validateZone(record.zone, label);
  validateGlobals(record.globals, label);
  validateExpected(record.expected, hashOnly, label);
}

export function sha256Utf8(value) {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

export async function errorClassifier(path, expectedVersion) {
  const definition = JSON.parse(await readFile(path, 'utf8'));
  if (!isObject(definition) || !nonBlank(definition.version) || !Array.isArray(definition.patterns)) {
    throw new Error(`Invalid error pattern table: ${path}`);
  }
  if (expectedVersion !== undefined && definition.version !== expectedVersion) {
    throw new Error(`Error pattern table version ${definition.version} does not match ${expectedVersion}`);
  }
  const patterns = definition.patterns.map((pattern, index) => {
    if (!isObject(pattern) || !nonBlank(pattern.regex) || !categories.has(pattern.category)) {
      throw new Error(`Invalid error pattern ${index + 1} in ${path}`);
    }
    if (pattern.constructors !== undefined
        && (!Array.isArray(pattern.constructors) || pattern.constructors.length === 0
          || pattern.constructors.some((name) => !nonBlank(name)))) {
      throw new Error(`Invalid error pattern ${index + 1} in ${path}`);
    }
    if (pattern.phases !== undefined
        && (!Array.isArray(pattern.phases) || pattern.phases.length === 0
          || pattern.phases.some((phase) => !nonBlank(phase)))) {
      throw new Error(`Invalid error pattern ${index + 1} in ${path}`);
    }
    try {
      return {
        category: pattern.category,
        regex: new RegExp(pattern.regex),
        constructors: pattern.constructors,
        phases: pattern.phases,
      };
    } catch (error) {
      throw new Error(`Invalid error pattern ${index + 1} in ${path}: ${error.message}`);
    }
  });
  return (message, constructorName, phase) => {
    for (const pattern of patterns) {
      if (pattern.regex.test(message)
          && (pattern.constructors === undefined || pattern.constructors.includes(constructorName))
          && (pattern.phases === undefined || pattern.phases.includes(phase))) {
        return pattern.category;
      }
    }
    throw new Error(`Unmatched upstream error for ${definition.version}: ${message}`);
  };
}

function validateGlobals(globals, label) {
  if (globals === undefined) return;
  throw new Error(`${label}: record globals are not supported by the pinned Template API`);
}

function validateTemplateOptions(options, label) {
  if (options === undefined) return;
  if (!isObject(options)) throw new Error(`${label}: templateOptions must be an object`);
  const allowed = new Set(['trimBlocks', 'lstripBlocks']);
  const unknown = Object.keys(options).filter((key) => !allowed.has(key));
  if (unknown.length) throw new Error(`${label}: unknown templateOptions fields: ${unknown.join(', ')}`);
  for (const [key, value] of Object.entries(options)) {
    if (typeof value !== 'boolean') throw new Error(`${label}: templateOptions.${key} must be boolean`);
  }
}

function validateZone(zone, label) {
  if (!nonBlank(zone) || (zone !== 'UTC' && !canonicalZones.has(zone))) {
    throw new Error(`${label}: zone must be an IANA time-zone identifier`);
  }
}

function validInstant(instant) {
  const match = instantPattern.exec(instant);
  if (match === null) return false;
  const parsed = new Date(instant);
  const canonical = instant.replace(
    /(?:\.(\d{1,3}))?Z$/,
    (_, fraction) => `.${(fraction ?? '').padEnd(3, '0')}Z`,
  );
  return !Number.isNaN(parsed.valueOf()) && parsed.toISOString() === canonical;
}

function validateExpected(expected, hashOnly, label) {
  if (!isObject(expected)) throw new Error(`${label}: expected is required`);
  const keys = Object.keys(expected);
  const exactError = keys.length === 2 && keys.includes('errorCategory') && keys.includes('errorMessage');
  if (keys.length !== 1 && !exactError) throw new Error(`${label}: expected must have exactly one outcome`);
  const [outcome] = keys;
  if (expected.errorCategory !== undefined) {
    if (!categories.has(expected.errorCategory)) throw new Error(`${label}: invalid error category`);
    if (exactError && typeof expected.errorMessage !== 'string')
      throw new Error(`${label}: errorMessage must be a string`);
    return;
  }
  if (hashOnly && outcome === 'sha256' && sha256(expected.sha256)) return;
  if (!hashOnly && outcome === 'text' && typeof expected.text === 'string') return;
  throw new Error(`${label}: expected outcome does not match fixture form`);
}

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function nonBlank(value) {
  return typeof value === 'string' && value.length > 0;
}

function sha256(value) {
  return typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
}
