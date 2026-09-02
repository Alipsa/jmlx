#!/usr/bin/env node
import { mkdir, readdir, readFile, realpath, stat, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import vm from 'node:vm';
import { readCorpus, validateCorpus, validateRecord } from './corpus.mjs';

const sourcePath = 'upstream/vendor/test/templates.test.js';
const interpreterSourcePath = 'upstream/vendor/test/interpreter.test.js';
const corpusPath = 'src/test/resources/corpus/v1.jsonl';
const lockPath = 'upstream/upstream-lock.json';
const sourceInventoryPath = 'upstream/corpus-source-inventory.json';
const runtimeSourcePath = 'upstream/vendor/src/runtime.ts';
const errorPatternsPath = 'tools/corpus/error-patterns-0.5.9.json';
// These upstream fixtures inject JavaScript functions through the render context. The public
// differential-corpus schema intentionally does not serialize functions: its `globals` field is
// reserved until the pinned Template API can inject non-built-in globals. Keep the exclusions
// named here, rather than silently dropping them, so the coverage report remains reviewable.
const unsupportedContextFixtures = new Map([
  ['NUMBERS', 'context supplies the add JavaScript function'],
  ['FUNCTIONS', 'context supplies JavaScript functions'],
  ['OBJ_METHODS', 'context supplies JavaScript methods'],
  ['IS_OPERATOR_4', 'context supplies the custom testFunction JavaScript function'],
]);
const upstreamErrorCategories = new Map([
  ['Missing closing curly brace', 'SYNTAX'],
  ['Unclosed string literal', 'SYNTAX'],
  ['Unexpected character', 'SYNTAX'],
  ['Invalid quote character', 'SYNTAX'],
  ['Unclosed statement', 'SYNTAX'],
  ['Unclosed expression', 'SYNTAX'],
  ['Unmatched control structure', 'SYNTAX'],
  ['Missing variable in for loop', 'SYNTAX'],
  ['Unclosed parentheses in expression', 'SYNTAX'],
  ['Invalid variable name', 'SYNTAX'],
  ['Invalid control structure usage', 'SYNTAX'],
  ['Undefined function call', 'TYPE'],
  ['Incorrect function call', 'TYPE'],
  ['Looping over non-iterable', 'TYPE'],
  ['Invalid variable assignment', 'SYNTAX'],
]);
const upstreamErrorMessages = new Map([
  ['Missing closing curly brace', 'Unexpected end of input'],
  ['Unclosed string literal', 'Unexpected end of input'],
  ['Unexpected character', 'Unexpected character: !'],
  ['Invalid quote character', 'Unexpected character: ‘'],
  // The upstream parser leaks this TypeError, but hfjinja intentionally retains descriptive
  // end-of-input diagnostics. These fixtures therefore compare category only.
  ['Unclosed statement', null],
  ['Unclosed expression', null],
  ['Unmatched control structure', 'Unknown statement type: endfor'],
  ['Missing variable in for loop', 'Unexpected token: CloseStatement'],
  ['Unclosed parentheses in expression', 'Parser Error: Expected closing parenthesis, got ${tokens[current].type} instead.. CloseExpression !== CloseParen.'],
  ['Invalid variable name', 'Parser Error: Expected closing expression token. Identifier !== CloseExpression.'],
  ['Invalid control structure usage', 'Unexpected token: CloseStatement'],
  ['Undefined function call', 'Cannot call something that is not a function: got UndefinedValue'],
  ['Incorrect function call', 'Cannot call something that is not a function: got BooleanValue'],
  ['Looping over non-iterable', 'Expected iterable or object type in for loop: got IntegerValue'],
  ['Invalid variable assignment', 'Invalid LHS inside assignment expression: {"type":"IntegerLiteral","value":42}'],
]);
const syntaxErrorGroups = new Set(['Lexing errors', 'Parsing errors']);
// runtime.ts evaluates the invalid assignment after parsing it, but Java rejects the left-hand
// side during parsing. Both public APIs expose it as SYNTAX despite that detection-phase split.
const syntaxRuntimeErrorCases = new Set(['Invalid variable assignment']);
// templates.test.js injects `true` into a bare Environment. The oracle's setupGlobals path already
// installs that built-in, so retaining the injected context would test a deliberate collision error
// rather than the upstream test's attempted call of BooleanValue.
const publicApiErrorContextOverrides = new Map([['Incorrect function call', {}]]);
const options = new Set(process.argv.slice(2));
if (![...options].every((option) => option === '--check' || option === '--write' || option.startsWith('--report='))) {
  throw new Error('Usage: convert-upstream-tests.mjs --check [--write] [--report=<path>]');
}
if (!options.has('--check') && !options.has('--write')) {
  throw new Error('Usage: convert-upstream-tests.mjs --check [--write] [--report=<path>]');
}

const source = await readFile(sourcePath, 'utf8');
const capture = extractConstants(source);
const upstreamFixtureCount = Object.keys(capture.templates).length;
const functionContextFixtures = new Set(
  Object.entries(capture.contexts)
    .filter(([, context]) => containsFunction(context))
    .map(([name]) => name),
);
const declaredExclusions = new Set(unsupportedContextFixtures.keys());
if (!sameSet(functionContextFixtures, declaredExclusions)) {
  throw new Error(
    `Function-context exclusions differ from upstream: expected ${[...functionContextFixtures].join(', ') || '(none)'}, `
      + `declared ${[...declaredExclusions].join(', ') || '(none)'}`,
  );
}
const templateStringsEnd = source.indexOf('\nconst TEST_PARSED', source.indexOf('const TEST_STRINGS = {'));
if (templateStringsEnd < 0) throw new Error(`Could not locate TEST_STRINGS boundary in ${sourcePath}`);
const generated = Object.keys(capture.templates)
  .filter((name) => !unsupportedContextFixtures.has(name))
  .map((upstreamName) => ({
    id: `templates.${fixtureId(upstreamName)}`,
    source: `${sourcePath}:${propertyLine(source, upstreamName, templateStringsEnd)}`,
    template: capture.templates[upstreamName],
    context: capture.contexts[upstreamName],
    expected: {text: capture.outputs[upstreamName]},
  }));
const interpreterSource = await readFile(interpreterSourcePath, 'utf8');
const interpreterGenerated = extractWhitespaceCases(interpreterSource).map(
  (test, index) => ({
    id: `interpreter.whitespace-control-${index + 1}`,
    source: `${interpreterSourcePath}:${whitespaceCaseLine(interpreterSource, index)}`,
    template: test.template,
    templateOptions: {
      ...(test.options.trim_blocks === undefined ? {} : {trimBlocks: test.options.trim_blocks}),
      ...(test.options.lstrip_blocks === undefined ? {} : {lstripBlocks: test.options.lstrip_blocks}),
    },
    context: test.data,
    expected: {text: test.target},
  }),
);
generated.push(...interpreterGenerated);
generated.push(...extractErrorCases(source));
for (const record of generated) validateRecord(record, record.id);
const records = await readCorpus(corpusPath);
validateCorpus(records, corpusPath);

if (options.has('--write')) {
  const retained = records.filter(
    (record) => !record.id.startsWith('templates.') && !record.id.startsWith('interpreter.'));
  await writeFile(corpusPath, [...generated, ...retained].map((record) => JSON.stringify(record)).join('\n') + '\n', 'utf8');
}

if (options.has('--check')) {
  const actual = new Map(records.map((record) => [record.id, record]));
  for (const record of generated) {
    const committed = actual.get(record.id);
    if (!committed || !sameFixture(committed, record)) {
      throw new Error(`${corpusPath}: ${record.id} differs from ${record.source}; rerun the reviewed converter`);
    }
  }
  const generatedIds = new Set(generated.map((record) => record.id));
  const staleIds = [...actual.keys()].filter(
    (id) => (id.startsWith('templates.') || id.startsWith('interpreter.')) && !generatedIds.has(id));
  if (staleIds.length) throw new Error(`${corpusPath}: stale extracted fixtures: ${staleIds.join(', ')}`);
}

const reportOption = [...options].find((option) => option.startsWith('--report='));
if (reportOption) await writeCoverage(reportOption.slice('--report='.length), generated, upstreamFixtureCount, records);

function extractConstants(source) {
  const executable = source.replace(/^(?:import[\s\S]*?;\s*)+/, '');
  const context = {describe: () => {}};
  try {
    vm.runInNewContext(`${executable}\nglobalThis.capture = {TEST_STRINGS, TEST_CONTEXT, EXPECTED_OUTPUTS};`, context, {
      filename: sourcePath,
    });
  } catch (error) {
    throw new Error(`Could not extract corpus constants from ${sourcePath}: ${error.message}`);
  }
  const capture = context.capture;
  if (!capture || !capture.TEST_STRINGS || !capture.TEST_CONTEXT || !capture.EXPECTED_OUTPUTS) {
    throw new Error(`Could not extract corpus constants from ${sourcePath}`);
  }
  return {templates: capture.TEST_STRINGS, contexts: capture.TEST_CONTEXT, outputs: capture.EXPECTED_OUTPUTS};
}

function extractWhitespaceCases(source) {
  const executable = source
    .replace(/^(?:import[\s\S]*?;\s*)+/, '')
    .replace(
      'for (const test of TESTS) {',
      'globalThis.captureWhitespace = TESTS; return; for (const test of TESTS) {',
    );
  const context = {describe: (_, callback) => callback(), it: (_, callback) => callback()};
  try {
    vm.runInNewContext(executable, context, {filename: interpreterSourcePath});
  } catch (error) {
    throw new Error(`Could not extract whitespace fixtures from ${interpreterSourcePath}: ${error.message}`);
  }
  if (!Array.isArray(context.captureWhitespace)) {
    throw new Error(`Could not extract whitespace fixtures from ${interpreterSourcePath}`);
  }
  return context.captureWhitespace;
}

function extractErrorCases(source) {
  const captured = [];
  const describeStack = [];
  let currentName;
  let currentTemplate;
  let currentEnvironment;
  class Environment {
    values = {};

    constructor() {
      currentEnvironment = this;
    }

    set(name, value) {
      this.values[name] = value;
    }
  }
  class Interpreter {
    constructor(environment) {
      currentEnvironment = environment;
    }

    run(ast) {
      currentTemplate = ast.template;
      return {value: ''};
    }
  }
  const tokenize = (template) => {
    currentTemplate = template;
    return {template};
  };
  const parse = (tokens) => tokens;
  const expect = (actual) => ({
    toMatchObject: () => {},
    toEqual: () => {},
    toThrowError: () => {
      if (!describeStack.includes('Error checking')) return;
      actual();
      captured.push({
        name: currentName,
        template: currentTemplate,
        context: currentEnvironment?.values ?? {},
        group: describeStack.at(-1),
      });
    },
  });
  const executable = source.replace(/^(?:import[\s\S]*?;\s*)+/, '');
  try {
    vm.runInNewContext(executable, {
      describe: (name, callback) => {
        describeStack.push(name);
        try {
          callback();
        } finally {
          describeStack.pop();
        }
      },
      it: (name, callback) => {
        currentName = name;
        currentTemplate = undefined;
        currentEnvironment = undefined;
        callback();
      },
      expect,
      tokenize,
      parse,
      setupGlobals: () => {},
      Environment,
      Interpreter,
      Template: class {
        render() {
          return '';
        }

        format() {
          return '';
        }
      },
      console: {error: () => {}, warn: () => {}},
    }, {filename: sourcePath});
  } catch (error) {
    throw new Error(`Could not extract upstream error cases from ${sourcePath}: ${error.message}`);
  }
  if (!sameSet(new Set(captured.map((entry) => entry.name)), new Set(upstreamErrorCategories.keys()))) {
    throw new Error(`Upstream error cases differ from reviewed category mapping in ${sourcePath}`);
  }
  if (!sameSet(new Set(captured.map((entry) => entry.name)), new Set(upstreamErrorMessages.keys()))) {
    throw new Error(`Upstream error cases differ from reviewed message mapping in ${sourcePath}`);
  }
  for (const entry of captured) {
    const category = upstreamErrorCategories.get(entry.name);
    if (syntaxErrorGroups.has(entry.group) && category !== 'SYNTAX') {
      throw new Error(`Upstream ${entry.group} case must map to SYNTAX: ${entry.name}`);
    }
    if (entry.group === 'Runtime errors'
        && (category === 'SYNTAX') !== syntaxRuntimeErrorCases.has(entry.name)) {
      throw new Error(`Upstream Runtime errors category mapping is not declared: ${entry.name}`);
    }
  }
  return captured.map((entry) => {
    if (typeof entry.template !== 'string') {
      throw new Error(`Could not capture template for upstream error case: ${entry.name}`);
    }
    const errorMessage = upstreamErrorMessages.get(entry.name);
    return {
      id: `templates.error-${fixtureId(entry.name)}`,
      source: `${sourcePath}:${errorCaseLine(source, entry.name)}`,
      template: entry.template,
      context: publicApiErrorContextOverrides.get(entry.name) ?? entry.context,
      expected: {
        errorCategory: upstreamErrorCategories.get(entry.name),
        ...(errorMessage === null ? {} : {errorMessage}),
      },
    };
  });
}

function propertyLine(source, name, end) {
  const templateStrings = source.slice(0, end);
  const match = new RegExp(`^[^\\S\\r\\n]*${name}:`, 'm').exec(templateStrings);
  if (!match) throw new Error(`Could not locate ${name} in ${sourcePath}`);
  return source.slice(0, match.index).split('\n').length;
}

function whitespaceCaseLine(source, index) {
  const testsStart = source.indexOf('const TESTS = [');
  if (testsStart < 0) throw new Error(`Could not locate TESTS in ${interpreterSourcePath}`);
  const cases = [...source.slice(testsStart).matchAll(/^\s*\{\s*$/gm)];
  const match = cases[index];
  if (!match) throw new Error(`Could not locate whitespace case ${index + 1} in ${interpreterSourcePath}`);
  return source.slice(0, testsStart + match.index).split('\n').length;
}

function errorCaseLine(source, name) {
  const quotedName = JSON.stringify(name).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(`^[^\\S\\r\\n]*it\\(${quotedName},`, 'm').exec(source);
  if (!match) throw new Error(`Could not locate upstream error case: ${name}`);
  return source.slice(0, match.index).split('\n').length;
}

function sameFixture(actual, generated) {
  return Object.keys(actual).length === Object.keys(generated).length
    && actual.source === generated.source
    && actual.template === generated.template
    && canonicalJson(actual.templateOptions) === canonicalJson(generated.templateOptions)
    && canonicalJson(actual.context) === canonicalJson(generated.context)
    && canonicalJson(actual.expected) === canonicalJson(generated.expected);
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value !== null && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function fixtureId(name) {
  return name.toLowerCase().replaceAll('_', '-').replaceAll(/[^a-z0-9]+/g, '-').replaceAll(/^-|-$/g, '');
}

function containsFunction(value) {
  if (typeof value === 'function') return true;
  if (Array.isArray(value)) return value.some(containsFunction);
  return value !== null && typeof value === 'object' && Object.values(value).some(containsFunction);
}

function sameSet(left, right) {
  return left.size === right.size && [...left].every((value) => right.has(value));
}

async function writeCoverage(path, generated, upstreamFixtureCount, committedRecords) {
  const lock = JSON.parse(await readFile(lockPath, 'utf8'));
  const inventory = JSON.parse(await readFile(sourceInventoryPath, 'utf8'));
  const committedTemplateRecords = committedRecords.filter((record) => typeof record.template === 'string').length;
  const testFiles = await testSources('upstream/vendor/test');
  const excludedTestFiles = Object.keys(lock.excludedFiles ?? {})
    .filter((file) => file.startsWith('test/') && file.endsWith('.test.js'));
  validateSourceInventory(inventory, testFiles, excludedTestFiles);
  const runtimeCoverage = runtimeSurfaceCoverage(await readFile(runtimeSourcePath, 'utf8'), committedRecords);
  const errorCoverage = errorFamilyCoverage(
    JSON.parse(await readFile(errorPatternsPath, 'utf8')),
    committedRecords,
  );
  const lines = [
    '# Differential corpus coverage', '',
    `Vendored non-model unit sources: ${testFiles.length}`,
    `Vendored template fixture definitions: ${upstreamFixtureCount}`,
    `Automatically extracted fixture definitions: ${generated.length} (${interpreterGenerated.length} interpreter whitespace vectors, ${upstreamErrorCategories.size} error vectors)`,
    `Schema-excluded fixture definitions: ${unsupportedContextFixtures.size}`,
    `Committed template-bearing corpus records: ${committedTemplateRecords} (executed by both the pinned Node oracle and Java differential runner)`,
    `Policy-excluded test sources: ${excludedTestFiles.length}${excludedTestFiles.length ? ` (${excludedTestFiles.join(', ')})` : ''}`, '',
    '## Extracted fixtures', '', '| Corpus id | Upstream source |', '| --- | --- |',
    ...generated.map((record) => `| \`${record.id}\` | \`${record.source}\` |`), '',
    '## Schema exclusions', '', '| Upstream fixture | Reason |', '| --- | --- |',
    ...[...unsupportedContextFixtures].map(([name, reason]) => `| \`${name}\` | ${reason} |`), '',
    '## Source inventory', '', '| Upstream source | Coverage | Evidence |', '| --- | --- | --- |',
    ...Object.entries(inventory.sources).map(([source, entry]) => `| \`${source}\` | ${entry.coverage} | ${entry.evidence ?? entry.reason} |`), '',
    'Every vendored unit source and every policy-excluded test source is represented above.', '',
  ];
  lines.splice(
    lines.length - 1,
    0,
    '## Runtime surface inventory', '',
    '| Surface | Name | Differential-corpus evidence |', '| --- | --- | --- |',
    ...runtimeCoverage.flatMap(([surface, entries]) => entries.map(([name, evidence]) =>
      '| ' + surface + ' | ' + name + ' | ' + evidence.join(', ') + ' |')),
    '',
    '## Error-family inventory', '', '| Category | Differential-corpus evidence |', '| --- | --- |',
    ...errorCoverage.map(([category, evidence]) => '| ' + category + ' | ' + evidence + ' |'), '',
  );
  const reportPath = resolve(path);
  await mkdir(dirname(reportPath), {recursive: true});
  await writeFile(reportPath, lines.join('\n'), 'utf8');
}

function errorFamilyCoverage(patterns, records) {
  const excluded = new Map([
    ['UNDEFINED_OR_ACCESS', 'Excluded: unknown variables and missing members render as undefined in the pinned public Template API.'],
  ]);
  const categories = [...new Set(patterns.patterns.map((pattern) => pattern.category))].sort();
  return categories.map((category) => {
    const evidence = records.filter((record) => record.expected.errorCategory === category).map((record) => record.id);
    if (evidence.length === 0 && !excluded.has(category)) {
      throw new Error('Pinned error family lacks differential-corpus coverage: ' + category);
    }
    return [category, evidence.length === 0 ? excluded.get(category) : evidence.join(', ')];
  });
}

function validateSourceInventory(inventory, testFiles, excludedTestFiles) {
  if (inventory?.version !== 1 || inventory.sources === null || typeof inventory.sources !== 'object') {
    throw new Error(`Invalid corpus source inventory: ${sourceInventoryPath}`);
  }
  const expected = new Set([...testFiles.map((file) => `test/${file}`), ...excludedTestFiles]);
  const actual = new Set(Object.keys(inventory.sources));
  if (!sameSet(expected, actual)) {
    throw new Error(
      `Corpus source inventory differs from vendored tests: expected ${[...expected].sort().join(', ')}, `
        + `actual ${[...actual].sort().join(', ')}`,
    );
  }
  for (const [source, entry] of Object.entries(inventory.sources)) {
    if (!['differential-corpus', 'format-golden', 'direct-unit', 'policy-excluded'].includes(entry?.coverage)
        || typeof entry.reason !== 'string' || entry.reason.length === 0) {
      throw new Error(`Invalid corpus source inventory entry: ${source}`);
    }
  }
}

function runtimeSurfaceCoverage(runtimeSource, records) {
  const between = (start, end) => {
    const from = runtimeSource.indexOf(start);
    const to = runtimeSource.indexOf(end, from);
    if (from < 0 || to < 0) throw new Error('Could not extract pinned runtime surface');
    return runtimeSource.slice(from, to);
  };
  const mapEntries = (source) => [...source.matchAll(/\[\s*"([^"]+)"\s*,/g)].map((match) => match[1]);
  const cases = (source) => [...source.matchAll(/case "([^"]+)"/g)].map((match) => match[1]);
  const unique = (names) => [...new Set(names)].sort();
  const tests = unique(mapEntries(between('private static readonly TESTS', 'tests: ReadonlyMap')));
  const globals = unique([...between('export function setupGlobals', '/**\n * Helper function').matchAll(/env\.set\("([^"]+)"/g)]
    .map((match) => match[1]));
  const stringMembers = unique(mapEntries(between('export class StringValue', 'export class BooleanValue')));
  const objectMembers = unique(
    mapEntries(between('export class ObjectValue', '\n\titems():')).filter((name) => name !== 'key'),
  );
  const arrayMembers = unique(mapEntries(between('export class ArrayValue', 'export class TupleValue')));
  const filtersSource = between('private applyFilter', 'private evaluateFilterExpression');
  const filters = unique([
    ...cases(filtersSource),
    ...[...filtersSource.matchAll(/filter(?:Name|\.value) === "([^"]+)"/g)].map((match) => match[1]),
    ...objectMembers,
  ]);
  const surfaces = [
    ['filter', filters], ['test', tests], ['string member', stringMembers],
    ['object member', objectMembers], ['array member', arrayMembers], ['global', globals],
  ];
  return surfaces.map(([surface, names]) => [surface, names.map((name) => {
      const usage = runtimeUsage(surface, name);
      const evidence = records.filter((record) => typeof record.template === 'string' && usage.test(record.template))
        .map((record) => record.id);
      if (evidence.length === 0) {
        throw new Error('Pinned runtime ' + surface + ' lacks differential-corpus coverage: ' + name);
      }
      return [name, evidence];
    })]);
}

function runtimeUsage(surface, name) {
  const escaped = escapeRegex(name);
  const terminator = '(?=\\s|\\(|\\}|\\||\\)|,|\\]|$)';
  switch (surface) {
    case 'filter':
      return new RegExp(`(?:\\|\\s*${escaped}${terminator}|\\bfilter\\s+${escaped}${terminator})`);
    case 'test':
      return new RegExp(`\\bis\\s+(?:not\\s+)?${escaped}${terminator}`);
    case 'string member':
    case 'object member':
    case 'array member':
      return new RegExp(`(?:\\.${escaped}(?=\\s*\\(|\\s|\\}|\\||\\)|,|\\]|$)|\\|\\s*${escaped}${terminator})`);
    case 'global':
      return new RegExp(`\\b${escaped}(?=\\s*\\(|\\s|\\}|\\||\\)|,|\\]|$)`);
    default:
      throw new Error(`Unknown runtime surface: ${surface}`);
  }
}

function escapeRegex(value) {
  return value.replace(/[.*+?^\${}()|[\]\\]/g, '\\$&');
}

async function testSources(directory, relative = '', seen = new Set()) {
  let resolved;
  try {
    resolved = await realpath(directory);
  } catch {
    return [];
  }
  if (seen.has(resolved)) return [];
  const visited = new Set(seen).add(resolved);
  const entries = await readdir(directory, {withFileTypes: true});
  const files = await Promise.all(entries.map(async (entry) => {
    const entryRelative = relative ? `${relative}/${entry.name}` : entry.name;
    const entryPath = `${directory}/${entry.name}`;
    let target;
    try {
      target = entry.isSymbolicLink() ? await stat(entryPath) : undefined;
    } catch {
      return [];
    }
    if (entry.isDirectory() || target?.isDirectory()) return testSources(entryPath, entryRelative, visited);
    return (entry.isFile() || target?.isFile()) && entry.name.endsWith('.test.js') ? [entryRelative] : [];
  }));
  return files.flat().sort();
}
