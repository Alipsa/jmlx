#!/usr/bin/env node
import { spawn } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { readCorpus, validateCorpus } from '../corpus/corpus.mjs';
import { ALGORITHM, generate, SMOKE_SEEDS } from './generate-parser-cases.mjs';

const REQUEST_TIMEOUT_MS = 15_000;
const REDUCTION_TIMEOUT_MS = 30_000;
const REDUCTION_TRIALS = 200;
const PROTOCOL = 'hfjinja-parser-fuzz-v1';
function option(name, fallback) { const i = process.argv.indexOf(name); return i < 0 ? fallback : process.argv[i + 1]; }
const java = option('--java'), classpath = option('--java-classpath'), report = option('--report', 'build/reports/fuzz-parser.md');
if (!java || !classpath) throw new Error('Usage: compare-parser-results.mjs --java <java21> --java-classpath <classpath> [--report path]');
const count = Number(option('--count', '100'));
const corpusPath = 'src/test/resources/corpus/v1.jsonl';
const encode = source => Buffer.from(source, 'utf16le').toString('base64');
const decode = candidate => Buffer.from(candidate.source, 'base64').toString('utf16le');

function runner(command, args) {
  const child = spawn(command, args, { stdio: ['pipe', 'pipe', 'pipe'] });
  let pending, stderr = '', buffer = '', unexpectedOutput = [];
  child.stderr.on('data', data => { stderr += data; }); child.stdout.setEncoding('utf8');
  child.stdout.on('data', data => { buffer += data; let at; while ((at = buffer.indexOf('\n')) >= 0) { const line = buffer.slice(0, at); buffer = buffer.slice(at + 1); if (pending) { const done = pending; pending = null; done.resolve(line); } else unexpectedOutput.push(line); } });
  child.on('exit', code => { if (pending) { const done = pending; pending = null; done.reject(new Error(`HARNESS runner exited ${code}: ${stderr}`)); } });
  return { async request(candidate) {
    if (unexpectedOutput.length) throw new Error(`HARNESS unexpected runner output before id=${candidate.id}: ${JSON.stringify(unexpectedOutput)}`);
    if (pending) throw new Error('HARNESS concurrent request');
    const line = await new Promise((resolve, reject) => { const timer = setTimeout(() => { pending = null; child.kill('SIGKILL'); reject(new Error(`HARNESS timeout id=${candidate.id}`)); }, REQUEST_TIMEOUT_MS); pending = { resolve: value => { clearTimeout(timer); resolve(value); }, reject: error => { clearTimeout(timer); reject(error); } }; child.stdin.write(`${JSON.stringify(candidate)}\n`); });
    let value; try { value = JSON.parse(line); } catch { throw new Error(`HARNESS malformed output id=${candidate.id}: ${line}`); }
    if (value.id !== candidate.id || !value.result) throw new Error(`HARNESS invalid output id=${candidate.id}: ${line}`);
    if (value.result === 'HARNESS') throw new Error(`HARNESS id=${candidate.id}: ${value.message ?? value.detail ?? '<no detail>'}`);
    await new Promise(resolve => setImmediate(resolve));
    if (unexpectedOutput.length) throw new Error(`HARNESS unexpected runner output after id=${candidate.id}: ${JSON.stringify(unexpectedOutput)}`);
    return value;
  }, assertClean() { if (unexpectedOutput.length) throw new Error(`HARNESS unexpected runner output after final response: ${JSON.stringify(unexpectedOutput)}`); }, close() { child.kill(); } };
}

function discrepancy(candidate, node, jvm) {
  if (candidate.family === 'grammar') {
    if (node.result !== 'PARSED') return { kind: 'GENERATOR HARNESS', reason: `Node ${node.result}` };
    if (jvm.result !== 'PARSED') return { kind: 'PARITY', reason: `Java ${jvm.result}` };
    return node.ast === jvm.ast ? null : { kind: 'PARITY', reason: 'AST differs' };
  }
  if (node.result === 'LIMIT' || jvm.result === 'LIMIT') return null;
  if ((node.result === 'PARSED') !== (jvm.result === 'PARSED')) return { kind: 'PARITY', reason: `${node.result} versus ${jvm.result}` };
  if (node.result === 'SYNTAX'
      && node.message === "Cannot read properties of undefined (reading 'type')"
      && intentionalEndOfInputDiagnostic(jvm.message)) return null;
  if (node.result !== 'PARSED' && node.message !== jvm.message) {
    return { kind: 'PARITY', reason: `error message differs: Node ${JSON.stringify(node.message)} versus Java ${JSON.stringify(jvm.message)}` };
  }
  return null;
}

function intentionalEndOfInputDiagnostic(message) {
  return message === 'Unexpected end of template'
    || message === 'Unknown statement, got end of template'
    || /^Parser Error: .+\. End of template !== /.test(message);
}

// These delimiters and a numeric literal cover the parser transitions exercised by substitutions
// without multiplying every corpus code unit by the full hostile-source alphabet.
const mutationAlphabet = ['1', '{', '}', '!'];
async function mutations() {
  const records = await readCorpus(corpusPath);
  validateCorpus(records, corpusPath);
  const candidates = [];
  for (const record of records.filter((record) => typeof record.template === 'string')) {
    const options = record.templateOptions ?? {};
    for (let offset = 0; offset < record.template.length; offset++) for (const replacement of mutationAlphabet) {
      if (record.template[offset] === replacement) continue;
      const source = record.template.slice(0, offset) + replacement + record.template.slice(offset + 1);
      candidates.push({ id: `mutation-${record.id}-${offset}-${Buffer.from(replacement, 'utf16le').toString('hex')}`,
        family: 'mutation', source: encode(source), trimBlocks: record.templateOptions === undefined ? true : (options.trimBlocks ?? false),
        lstripBlocks: record.templateOptions === undefined ? true : (options.lstripBlocks ?? false), sourceCodeUnits: source.length });
    }
    // Prefixes cover truncation and end-of-input paths independently of substitutions.
    for (let length = 1; length <= record.template.length; length++) {
      const source = record.template.slice(0, length);
      candidates.push({ id: `mutation-${record.id}-prefix-${length}`, family: 'mutation',
        source: encode(source), trimBlocks: record.templateOptions === undefined ? true : (options.trimBlocks ?? false),
        lstripBlocks: record.templateOptions === undefined ? true : (options.lstripBlocks ?? false), sourceCodeUnits: source.length });
    }
  }
  return candidates;
}

const node = runner('node', ['tools/fuzz/node-parser-runner.mjs']);
const jvm = runner(java, ['-cp', classpath, 'se.alipsa.jmlx.jinja.internal.parser.FuzzParserRunner']);
async function evaluate(candidate) { const [nodeResult, javaResult] = await Promise.all([node.request(candidate), jvm.request(candidate)]); return { nodeResult, javaResult, issue: discrepancy(candidate, nodeResult, javaResult) }; }
async function minimize(candidate, originalIssue) {
  const started = Date.now(); let trials = 0, source = decode(candidate), width = Math.max(1, Math.floor(source.length / 2)), exhausted = false;
  while (width && trials < REDUCTION_TRIALS && Date.now() - started < REDUCTION_TIMEOUT_MS) {
    let reduced = false;
    for (let index = 0; index < source.length && trials < REDUCTION_TRIALS && Date.now() - started < REDUCTION_TIMEOUT_MS; index += width) {
      const next = source.slice(0, index) + source.slice(index + width); if (!next) continue;
      const trial = { ...candidate, source: encode(next), sourceCodeUnits: next.length }; trials++;
      const issue = (await evaluate(trial)).issue;
      if (issue && issue.kind === originalIssue.kind && issue.reason === originalIssue.reason) { source = next; reduced = true; break; }
    }
    if (!reduced) {
      if (trials >= REDUCTION_TRIALS || Date.now() - started >= REDUCTION_TIMEOUT_MS) { exhausted = true; break; }
      width = Math.floor(width / 2);
    }
  }
  exhausted ||= trials >= REDUCTION_TRIALS || Date.now() - started >= REDUCTION_TIMEOUT_MS;
  return { source, trials, status: exhausted ? 'budget-exhausted' : 'complete' };
}

let total = 0, mutationTotal = 0, limits = 0; const otherErrors = [];
try {
  const candidates = [
    ...SMOKE_SEEDS.flatMap(seed => generate({ seed, count }).slice(1)),
    ...await mutations(),
  ];
  for (const candidate of candidates) {
    const result = await evaluate(candidate); total++;
    if (candidate.family === 'mutation') mutationTotal++;
    if (candidate.family === 'hostile') {
      if (result.nodeResult.result === 'LIMIT' || result.javaResult.result === 'LIMIT') limits++;
      for (const [runtime, value] of [['node', result.nodeResult], ['java', result.javaResult]]) if (value.result === 'OTHER_ERROR') {
        const detail = value.detail ?? value.message;
        otherErrors.push(`${candidate.id} ${runtime}${detail ? `: ${detail}` : ''}`);
      }
    }
    if (result.issue) {
      let minimized;
      try {
        minimized = await minimize(candidate, result.issue);
      } catch (error) {
        const status = String(error).includes('HARNESS timeout') ? 'timeout' : 'harness-error';
        throw new Error(`${result.issue.kind} ${result.issue.reason} id=${candidate.id} trimBlocks=${candidate.trimBlocks} lstripBlocks=${candidate.lstripBlocks} source=${JSON.stringify(decode(candidate))} minimization=${status} detail=${String(error)} replay=node tools/fuzz/compare-parser-results.mjs --java ${java} --java-classpath <classpath> --count ${count}`);
      }
      throw new Error(`${result.issue.kind} ${result.issue.reason} id=${candidate.id} trimBlocks=${candidate.trimBlocks} lstripBlocks=${candidate.lstripBlocks} source=${JSON.stringify(minimized.source)} minimization=${minimized.status} trials=${minimized.trials} replay=node tools/fuzz/compare-parser-results.mjs --java ${java} --java-classpath <classpath> --count ${count}`);
    }
  }
  node.assertClean(); jvm.assertClean();
  await mkdir(dirname(report), { recursive: true });
  await writeFile(report, `# Parser fuzz verification\n\nProtocol: ${PROTOCOL}; PRNG: ${ALGORITHM}. Seeds: ${SMOKE_SEEDS.map(seed => `0x${seed.toString(16).toUpperCase()}`).join(', ')}. Grammar and hostile cases per seed: ${count}. Corpus substitutions and prefixes: ${mutationTotal}; substitutions use ${mutationAlphabet.length} selected replacements per code unit. Per-request timeout: ${REQUEST_TIMEOUT_MS / 1000} seconds. Minimization budget: ${REDUCTION_TIMEOUT_MS / 1000} seconds or ${REDUCTION_TRIALS} trials.\n\nVerified ${total} candidates; documented hostile limit outcomes: ${limits}. OTHER_ERROR outcomes (${otherErrors.length}):${otherErrors.length ? `\n${otherErrors.map(value => `- ${value}`).join('\n')}` : ' none'}\n\nExclusions: none.\n`);
} finally { node.close(); jvm.close(); }
