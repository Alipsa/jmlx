import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { corpusLine, errorClassifier, readCorpus, sha256Utf8, validateCorpus, validateRecord } from './corpus.mjs';

test('pins canonical IANA zones to the Node runtime', async () => {
  const zones = (await readFile(new URL('./iana-time-zones.txt', import.meta.url), 'utf8')).trim().split('\n');
  assert.deepEqual(zones, Intl.supportedValuesOf('timeZone'));
});

test('accepts text and hash-only record forms', () => {
  validateRecord({
    id: 'text', source: 'test', template: 'hello', context: {}, expected: {text: 'hello'},
  });
  validateRecord({
    id: 'hash', source: 'test', templateSha256: sha256Utf8('hello'), modelRepo: 'example/model',
    modelRevision: 'a'.repeat(40), templatePath: 'tokenizer_config.json', context: {messages: []},
    expected: {sha256: sha256Utf8('world')},
  });
  validateRecord({
    id: 'error', source: 'test', templateSha256: sha256Utf8('hello'), modelRepo: 'example/model',
    modelRevision: 'a'.repeat(40), templatePath: 'tokenizer_config.json', context: {},
    expected: {errorCategory: 'EXPLICIT_RAISE'},
  });
  validateRecord({
    id: 'exact-error', source: 'test', template: 'hello', context: {},
    expected: {errorCategory: 'EXPLICIT_RAISE', errorMessage: 'hello'},
  });
});

test('rejects duplicate ids and malformed deterministic-time fields', () => {
  const record = {id: 'duplicate', source: 'test', template: 'hello', context: {}, expected: {text: 'hello'}};
  assert.throws(() => validateCorpus([record, record]), /duplicate id/);
  assert.throws(() => validateRecord({...record, instant: '2026-08-19', zone: 'UTC'}), /ISO-8601/);
  assert.throws(() => validateRecord({...record, instant: '2026-08-19T00:00:00Z'}), /requires an explicit zone/);
  assert.throws(() => validateRecord({...record, zone: 'UTC'}), /requires an explicit instant/);
  assert.throws(() => validateRecord({...record, instant: '2026-08-19T00:00:00Z', zone: 'not a zone'}), /IANA/);
  assert.throws(() => validateRecord({...record, instant: '2000-01-02T03:04Z', zone: 'UTC'}), /ISO-8601/);
  assert.throws(() => validateRecord({...record, instant: '2000-02-30T00:00:00Z', zone: 'UTC'}), /ISO-8601/);
  assert.throws(() => validateRecord({...record, instant: '2000-01-01T24:00:00Z', zone: 'UTC'}), /ISO-8601/);
  validateRecord({...record, instant: '2000-01-02T03:04:05.12Z', zone: 'UTC'});
  validateRecord({...record, instant: '2000-01-02T03:04:05Z', zone: 'Africa/Abidjan'});
  assert.throws(() => validateRecord({...record, instant: '2000-01-02T03:04:05Z', zone: 'utc'}), /IANA/);
  assert.throws(() => validateRecord({...record, instant: '2000-01-02T03:04:05Z', zone: 'GMT+5'}), /IANA/);
  assert.throws(() => validateRecord({...record, globals: {strftime_now: {kind: 'strftime_now'}}}), /not supported/);
});

test('rejects mixed fixture forms and invalid expected results', () => {
  assert.throws(() => validateRecord({
    id: 'mixed', source: 'test', template: 'hello', templateSha256: sha256Utf8('hello'),
    context: {}, expected: {text: 'hello'},
  }), /exactly one/);
  assert.throws(() => validateRecord({
    id: 'wrong-outcome', source: 'test', template: 'hello', context: {},
    expected: {sha256: sha256Utf8('hello')},
  }), /does not match/);
  assert.throws(() => validateRecord({
    id: 'bad-exact-error', source: 'test', template: 'hello', context: {},
    expected: {errorCategory: 'EXPLICIT_RAISE', errorMessage: 1},
  }), /errorMessage must be a string/);
  assert.throws(() => validateRecord({
    id: 'bad-hash', source: 'test', template: 'hello', templateSha256: 'bad', context: {},
    expected: {text: 'hello'},
  }), /exactly one/);
  assert.throws(() => validateRecord({
    id: 'text-with-provenance', source: 'test', template: 'hello', modelRepo: 'example/model', context: {},
    expected: {text: 'hello'},
  }), /must not include hash-only provenance/);
  validateRecord({id: 'empty', source: 'test', template: '', context: {}, expected: {text: ''}});
  assert.throws(() => validateRecord({
    id: 'unknown-key', source: 'test', template: 'hello', context: {}, expected: {text: 'hello'}, typoInstant: 'x',
  }), /unknown fields/);
});

test('fails loudly for an unmatched upstream error', async () => {
  const lock = JSON.parse(await readFile('upstream/upstream-lock.json', 'utf8'));
  const classify = await errorClassifier('tools/corpus/error-patterns-0.5.9.json', `${lock.package}@${lock.version}`);
  assert.equal(classify('Unknown variable: absent'), 'UNDEFINED_OR_ACCESS');
  for (const type of ['ArrayValue', 'StringValue', 'NumericValue', 'ObjectValue', 'BooleanValue', 'FunctionValue']) {
    assert.equal(classify(`Unknown ${type} filter: frob`), 'TYPE');
  }
  assert.equal(classify('Cannot apply filter "abs" to type: FloatValue'), 'TYPE');
  assert.equal(classify('Unexpected end of input'), 'SYNTAX');
  assert.equal(classify('Unexpected character: !'), 'SYNTAX');
  assert.equal(classify('Unknown statement type: endfor'), 'SYNTAX');
  assert.equal(classify('Unexpected token: CloseStatement'), 'SYNTAX');
  assert.equal(
    classify('Parser Error: Expected closing parenthesis, got ${tokens[current].type} instead.. CloseExpression !== CloseParen.'),
    'SYNTAX',
  );
  assert.equal(classify('Invalid LHS inside assignment expression: {"type":"IntegerLiteral","value":42}'), 'SYNTAX');
  assert.equal(classify('Cannot call something that is not a function: got BooleanValue'), 'TYPE');
  assert.equal(classify('Expected iterable or object type in for loop: got IntegerValue'), 'TYPE');
  assert.equal(classify('`selectattr` can only be applied to array of objects'), 'TYPE');
  assert.equal(classify('`map` expressions without `attribute` set are not currently supported.'), 'TYPE');
  assert.equal(classify('wp7-eager-sentinel'), 'EXPLICIT_RAISE');
  assert.equal(classify('arguments of `selectattr` must be strings'), 'TYPE');
  assert.equal(classify('arguments of `rejectattr` must be strings'), 'TYPE');
  assert.equal(classify('Cannot convert to JSON: KeywordArgumentsValue'), 'TYPE');
  assert.equal(classify('sep argument must be a string or null'), 'TYPE');
  assert.equal(classify('maxsplit argument must be a number'), 'TYPE');
  assert.equal(classify('replace() arguments must be strings'), 'TYPE');
  assert.equal(classify('replace() requires at least two arguments'), 'TYPE');
  assert.equal(classify('Missing positional argument: a'), 'ARITY');
  assert.equal(classify('Object key must be a string: got KeywordArgumentsValue'), 'TYPE');
  assert.equal(classify('Positional arguments must come before keyword arguments'), 'SYNTAX');
  assert.equal(classify('', 'BreakControl'), 'SYNTAX');
  assert.equal(classify('', 'ContinueControl'), 'SYNTAX');
  assert.throws(() => classify('', 'Error'), /Unmatched upstream error/);
  assert.equal(classify("Cannot read properties of undefined (reading 'toLowerCase')"), 'TYPE');
  assert.equal(classify("Cannot read properties of undefined (reading 'toString')"), 'TYPE');
  assert.equal(classify("Cannot read properties of undefined (reading 'type')"), 'TYPE');
  assert.equal(
    classify(
      "Cannot read properties of undefined (reading 'type')",
      'TypeError',
      'parse',
    ),
    'SYNTAX',
  );
  assert.equal(classify("Cannot read properties of undefined (reading 'value')"), 'TYPE');
  assert.equal(classify("Cannot read properties of undefined (reading '__bool__')"), 'TYPE');
  assert.equal(classify("Cannot read properties of undefined (reading 'builtins')"), 'TYPE');
  assert.equal(classify("Cannot read properties of undefined (reading 'includes')"), 'TYPE');
  assert.equal(classify("Cannot read properties of undefined (reading 'at')"), 'TYPE');
  assert.equal(classify("Cannot read properties of undefined (reading 'length')"), 'TYPE');
  assert.equal(classify('undefined is not iterable'), 'TYPE');
  assert.equal(
    classify('undefined is not iterable (cannot read property Symbol(Symbol.iterator))'),
    'TYPE',
  );
  assert.equal(classify('format2.replace is not a function'), 'TYPE');
  assert.throws(() => classify('Cannot apply filter abs to type: FloatValue'), /Unmatched upstream error/);
  assert.throws(() => classify('selectattr can only be applied to array of objects'), /Unmatched upstream error/);
  assert.throws(() => classify('`map` expressions without attribute set are not currently supported.'), /Unmatched upstream error/);
  assert.throws(() => classify('wp7-eager-sentinel-extra'), /Unmatched upstream error/);
  assert.throws(() => classify('arguments of selectattr must be strings'), /Unmatched upstream error/);
  assert.throws(() => classify('sep argument must be a string or undefined'), /Unmatched upstream error/);
  assert.throws(() => classify('replace arguments must be strings'), /Unmatched upstream error/);
  assert.throws(() => classify('Object key must be a string: KeywordArgumentsValue'), /Unmatched upstream error/);
  assert.throws(() => classify('Positional arguments cannot follow keyword arguments'), /Unmatched upstream error/);
  assert.throws(() => classify('unmapped upstream error'), /Unmatched upstream error/);
  await assert.rejects(
    errorClassifier('tools/corpus/error-patterns-0.5.9.json', '@huggingface/jinja@other'), /does not match/,
  );
});

test('rejects malformed error-pattern selectors', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'jmlx-jinja-error-patterns-'));
  const path = join(directory, 'patterns.json');
  const writePatterns = (pattern) => writeFile(path, JSON.stringify({version: 'test', patterns: [pattern]}), 'utf8');
  const pattern = {regex: '^error$', category: 'TYPE'};
  try {
    await writePatterns({...pattern, constructors: []});
    await assert.rejects(errorClassifier(path, 'test'), /Invalid error pattern/);
    await writePatterns({...pattern, constructors: ['']});
    await assert.rejects(errorClassifier(path, 'test'), /Invalid error pattern/);
    await writePatterns({...pattern, phases: []});
    await assert.rejects(errorClassifier(path, 'test'), /Invalid error pattern/);
    await writePatterns({...pattern, phases: ['']});
    await assert.rejects(errorClassifier(path, 'test'), /Invalid error pattern/);
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});

test('preserves physical JSONL line numbers across blank lines', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'jmlx-jinja-corpus-lines-'));
  const path = join(directory, 'corpus.jsonl');
  try {
    await writeFile(path, '{"id":"first"}\n\n{"id":"third"}\n', 'utf8');
    const records = await readCorpus(path);
    assert.equal(corpusLine(records[0]), 1);
    assert.equal(corpusLine(records[1]), 3);
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});

test('identifies valid non-object JSON separately from invalid JSON', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'jmlx-jinja-corpus-json-'));
  const path = join(directory, 'corpus.jsonl');
  try {
    await writeFile(path, '42\n', 'utf8');
    await assert.rejects(readCorpus(path), /record must be a JSON object/);
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
});

test('reports harness mismatches and unmatched upstream errors per record', async () => {
  const result = await runOracle([
    {id: 'expected-error', source: 'test', template: 'Hello', context: {}, expected: {errorCategory: 'SYNTAX'}},
    {id: 'unmatched-error', source: 'test', template: "{{ raise_exception('unmapped upstream error') }}", context: {}, expected: {errorCategory: 'SYNTAX'}},
    {id: 'after-error', source: 'test', template: 'still runs', context: {}, expected: {text: 'still runs'}},
  ]);
  assert.equal(result.status, 1);
  assert.match(result.stderr, /expected error=SYNTAX, got output="Hello"/);
  assert.doesNotMatch(result.stderr, /FAIL .*expected-error.*output mismatch/);
  assert.match(result.stderr, /Unmatched upstream error.*unmapped upstream error/);
  assert.match(result.stdout, /PASS after-error/);
});

test('compares opted-in exact upstream error messages', async () => {
  const matching = await runOracle([{
    id: 'exact-message', source: 'test', template: "{{ raise_exception('exact') }}", context: {},
    expected: {errorCategory: 'EXPLICIT_RAISE', errorMessage: 'exact'},
  }]);
  assert.equal(matching.status, 0, matching.stderr);
  assert.match(matching.stdout, /PASS exact-message error=EXPLICIT_RAISE message=exact/);

  const mismatching = await runOracle([{
    id: 'wrong-message', source: 'test', template: "{{ raise_exception('actual') }}", context: {},
    expected: {errorCategory: 'EXPLICIT_RAISE', errorMessage: 'expected'},
  }]);
  assert.equal(mismatching.status, 1);
  assert.match(mismatching.stderr, /error message mismatch; expected="expected", actual="actual"/);

  const miscategorised = await runOracle([{
    id: 'miscategorised-exact', source: 'test', template: '{{ [1]|nosuchfilter }}', context: {},
    expected: {errorCategory: 'SYNTAX', errorMessage: 'Unknown ArrayValue filter: nosuchfilter'},
  }]);
  assert.equal(miscategorised.status, 1);
  assert.match(miscategorised.stderr, /error category mismatch; expected=SYNTAX, actual=TYPE/);
});

test('reports skipped hash-only records and rejects an all-hash-only run', async () => {
  const result = await runOracle([{
    id: 'hash-only', source: 'test', templateSha256: sha256Utf8('template'), modelRepo: 'example/model',
    modelRevision: 'a'.repeat(40), templatePath: 'tokenizer_config.json', context: {},
    expected: {sha256: sha256Utf8('output')},
  }]);
  assert.equal(result.status, 1);
  assert.match(result.stdout, /SKIP hash-only hash-only fixture/);
  assert.match(result.stdout, /SUMMARY executed=0 skipped=1/);
  assert.match(result.stderr, /no text-bearing corpus records were executed/);
});

test('uses a fixed UTC clock when a text record omits time fields', async () => {
  const result = await runOracle([{
    id: 'default-time', source: 'test', template: "{{ strftime_now('%Y-%m-%d %H:%M %B') }}",
    context: {}, expected: {text: '2000-01-02 03:04 January'},
  }]);
  assert.equal(result.status, 0);
  assert.match(result.stdout, /PASS default-time/);
});

test('uses fixed English collation when locale-sensitive sorting is rendered', async () => {
  const result = await runOracle([{
    id: 'default-collation', source: 'test', template: '{{ d | tojson(sort_keys=true) }}',
    context: {d: {z: 1, ä: 2, a: 3}}, expected: {text: '{"a": 3, "ä": 2, "z": 1}'},
  }], {LC_ALL: 'sv_SE.UTF-8', LANG: 'sv_SE.UTF-8'});
  assert.equal(result.status, 0);
  assert.match(result.stdout, /PASS default-collation/);
});

test('pins the Step3 macro-heavy resource and tool-use golden against the Node oracle', async () => {
  const template = await readFile('src/test/resources/model-templates/step3.jinja', 'utf8');
  const expected = await readFile('src/test/resources/model-templates/step3-tooluse.expected.txt', 'utf8');
  assert.equal(template.length, 2847);
  assert.equal(sha256Utf8(template), 'fc7bfeffd0dcee65d97834d2f0d60fb81c5db9f3e2567d038e3437f2bbdd54ca');
  const tool = {
    type: 'function', function: {
      name: 'get_weather', description: 'Météo', parameters: {
        type: 'object', properties: {city: {type: 'string'}}, required: ['city'],
      },
    },
  };
  const result = await runOracle([{
    id: 'step3-macro-heavy-tooluse', source: 'self-authored context; retained model resource',
    template, context: {
      bos_token: '<s>', tools: [tool], messages: [
        {role: 'system', content: 'You are helpful.'},
        {role: 'tool_description', content: 'Use tools.'},
        {role: 'user', content: [{type: 'text', text: 'What is the weather?'}, {type: 'image'}]},
        {role: 'assistant', content: 'Checking', tool_calls: [{
          type: 'function', function: {name: 'get_weather', arguments: {city: 'Paris', unit: 'C'}},
        }]},
        {role: 'tool_response', content: [{text: 'sunny'}]},
      ],
    }, expected: {text: expected},
  }]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /PASS step3-macro-heavy-tooluse/);
});

test('pins the primary Qwen3.8 MLX resource and goldens against the Node oracle', async () => {
  const template = await readFile('src/test/resources/model-templates/qwen3.8-27b-4bit.jinja', 'utf8');
  const normal = await readFile('src/test/resources/model-templates/qwen3.8-normal.expected.txt', 'utf8');
  const vision = await readFile('src/test/resources/model-templates/qwen3.8-vision.expected.txt', 'utf8');
  const tooluse = await readFile('src/test/resources/model-templates/qwen3.8-tooluse.expected.txt', 'utf8');
  assert.equal(Buffer.byteLength(template, 'utf8'), 8952);
  assert.equal(sha256Utf8(template), 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041');
  const tool = {
    type: 'function', function: {
      name: 'get_weather', description: 'Get weather', parameters: {
        type: 'object', properties: {city: {type: 'string'}}, required: ['city'],
      },
    },
  };
  const result = await runOracle([
    {
      id: 'qwen3.8-normal', source: 'self-authored context; retained model resource', template,
      context: {
        add_generation_prompt: true, enable_thinking: false, reasoning_effort: 'medium',
        messages: [{role: 'user', content: 'Hello!'}],
      }, expected: {text: normal},
    },
    {
      id: 'qwen3.8-vision', source: 'self-authored context; retained model resource', template,
      context: {
        add_generation_prompt: false, add_vision_id: true, enable_thinking: false,
        messages: [{role: 'user', content: [
          {type: 'text', text: 'Describe '}, {type: 'image'}, {type: 'text', text: ' then '}, {type: 'video'},
        ]}],
      }, expected: {text: vision},
    },
    {
      id: 'qwen3.8-tooluse', source: 'self-authored context; retained model resource', template,
      context: {
        add_generation_prompt: true, enable_thinking: true, reasoning_effort: 'low', tools: [tool], messages: [
          {role: 'user', content: 'What is the weather?'},
          {role: 'assistant', content: 'Checking', tool_calls: [{
            function: {name: 'get_weather', arguments: {city: 'Paris', units: ['metric', 'celsius']}},
          }]},
          {role: 'tool', content: 'Sunny'},
        ],
      }, expected: {text: tooluse},
    },
  ]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /PASS qwen3.8-normal/);
  assert.match(result.stdout, /PASS qwen3.8-vision/);
  assert.match(result.stdout, /PASS qwen3.8-tooluse/);
});

test('pins the remaining upstream runtime feature inventory against the Node oracle', async () => {
  const template = "{{ [3,1,3]|first }}|{{ [3,1,3]|last }}|{{ [3,1,3]|reverse }}|"
    + "{{ [3,1,3]|unique }}|{{ [3,1,2]|sort }}|{{ xs|map(attribute='n', default=0) }}|"
    + "{{ 'hello world'|title }}|{{ 'hello'|capitalize }}|{{ 'a\\nb'|indent(2) }}|"
    + "{{ 'abab'|replace('a','x',1) }}|{{ -2|abs }}|{{ true|bool }}|"
    + "{{ ' a b '.split() }}|{{ 'a-b-c'.replace('-','/',1) }}|{{ o.get('x','d') }}|"
    + "{{ o.keys() }}|{{ o.values() }}|{{ o.dictsort() }}|{{ 3 is odd }}{{ 4 is even }}"
    + "{{ 4 is integer }}{{ 'ABC' is upper }}{{ 'abc' is lower }}";
  const expected = '3|3|[3, 1, 3]|[3, 1]|[1, 2, 3]|[2, 0]|Hello World|Hello|a\n  b|xbab|2|true|["a", "b"]|a/b-c|d|["b", "a"]|[2, 1]|[["a", 1], ["b", 2]]|truetruetruetruetrue';
  const result = await runOracle([{
    id: 'remaining-runtime-inventory', source: 'self-authored upstream runtime inventory', template,
    context: {xs: [{n: 2}, {}], o: {b: 2, a: 1}}, expected: {text: expected},
  }]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /PASS remaining-runtime-inventory/);
});

async function runOracle(records, environment = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'jmlx-jinja-corpus-'));
  const corpus = join(directory, 'corpus.jsonl');
  try {
    await writeFile(corpus, `${records.map((record) => JSON.stringify(record)).join('\n')}\n`, 'utf8');
    return spawnSync(process.execPath, [
      'tools/corpus/run-node-oracle.mjs', '--corpus', corpus,
      '--patterns', 'tools/corpus/error-patterns-0.5.9.json', '--lock', 'upstream/upstream-lock.json',
    ], {encoding: 'utf8', env: {...process.env, ...environment}});
  } finally {
    await rm(directory, {recursive: true, force: true});
  }
}
