# hfjinja — requirements and design plan

## Purpose

`hfjinja` is a dependency-free Java port of Hugging Face's
[`@huggingface/jinja`](https://github.com/huggingface/huggingface.js/tree/main/packages/jinja).
It renders the Jinja subset used by Hugging Face model chat templates. It is deliberately not a
general-purpose or Python-compatible Jinja2 engine.

The project lets JVM model clients such as jmlx evaluate a model's `tokenizer_config.json`
`chat_template` without a JavaScript engine. Compatibility with one pinned upstream package
revision is its principal requirement. The base package is `se.alipsa.jmlx.jinja`; the Maven artifact
is `se.alipsa:jmlx-jinja`. The intended initial upstream pin is `@huggingface/jinja` **0.5.9**;
M0 records its exact commit, tarball integrity, and Node oracle version before implementation.
That pin changes only through a reviewed `sync-upstream` PR, never incidentally.

The ordered execution work is maintained in [implementation-plan.md](implementation-plan.md).

## Scope

In scope:

- The complete language and runtime implemented by the pinned upstream package: text/comments,
  output expressions, `if`/`elif`/`else`, `for`, `set`, macros, call/filter blocks, `break`,
  `continue`, literals, access/calls, filters/tests, slices, operators, ternaries, keyword and
  spread arguments.
- Upstream semantics for undefined/null, truthiness, numeric/string/list/object operations,
  scoping, loop metadata, built-in filters/tests, `tojson`, and `raise_exception`.
- The in-scope syntax list is generated at M0 from the pinned `ast.ts` node list and asserted by
  a test; the prose list above is a readable summary, not an independent contract.
- Conversion of an explicitly supported Java value graph; immutable parsed templates; concurrent
  rendering.
- Exact-output compatibility for imported upstream fixtures and retained Llama, Qwen, Mistral,
  and tool-use model-template goldens.
- A reproducible, reviewable upstream-update workflow.

Out of scope: Python or JavaScript execution, network/Hub downloads, includes/imports, async
rendering, general HTML templating, and automatic exposure of Java methods, beans, or reflection.
The sole callable exception is an explicitly named `HostFunction` supplied in `RenderOptions`.

## Requirements

| ID | Requirement | Acceptance criterion |
| --- | --- | --- |
| R1 | Native Java | The published JAR has no JS engine, JNI, Node.js, or production dependency. |
| R2 | Upstream parity | Every imported vector yields the same output or an equivalent documented error. |
| R3 | Determinism | Results do not depend on host locale, timezone, Java map implementation, or ambient clock. Date/time globals use only a caller-supplied `Clock` and fixed `ZoneId` in `RenderOptions`; no wall clock is read implicitly. If no clock is supplied, first use of a time global fails with `TemplateRenderException`. Strftime formatting uses fixed C/POSIX names, never host locale, with a mapped test. Determinism inside a `HostFunction` is the caller's responsibility. |
| R4 | Closed value boundary | Templates cannot access Java methods, classes, reflection, or arbitrary objects, except invocation of an explicitly named, `RenderOptions`-supplied `HostFunction`. |
| R5 | Diagnostics | Syntax/render errors report kind, template location, and useful context; `raise_exception` preserves its message. |
| R6 | Thread safety | `Template` is immutable and reusable, with no lazy memoization caches; each render gets a fresh environment. |
| R7 | Resource limits | Source, token, AST-depth, render-step, loop, macro-depth, and output-size limits are configurable. Differential-corpus runs use effectively unbounded budgets; budget exhaustion is a separate safety error, excluded from parity comparison and tested in the Safety layer. |
| R8 | Provenance | Every release records upstream version, commit, tarball SHA-256, source hashes, and fixture revision. |
| R9 | Packaging | Java 21 LTS+ build, reproducible modular JAR with `module-info.java`, and normal Maven metadata. |
| R10 | Licensing | Upstream MIT license, attribution, imported-fixture notices, and retained model-template licenses/notices are kept in `NOTICE`. Llama template material is handled under its applicable Llama Community License and attribution/naming conditions; Qwen and Mistral template material retains Apache-2.0 notices. Where Llama terms make vendoring unsuitable, retain only an approved hash and test input rather than template text. |
| R11 | Numeric semantics | Match the pinned JavaScript behavior for `//`, `%` (including negatives), division result types, integer-looking float rendering, JSON number formatting, and non-finite results. At the host boundary, accept any `Number` whose finite JS-double result prints, using JS's shortest round-trip representation, as the input's canonical decimal text after trailing zeros are stripped; integral values must also be in the JS safe-integer range. Thus `0.7` is accepted and `0.1234567890123456789` is rejected. Reject `NaN` and infinities. Inside the runtime, computed values follow JS semantics without exception, including precision loss and non-finite values; `tojson` renders computed `NaN`/infinities as `null`, matching `JSON.stringify`. Dedicated mapped tests specify each case. |
| R12 | Host-value semantics | Absent map key converts to undefined; present key with Java `null` converts to template null. Truthiness, `default`, and `tojson` distinguish them exactly as the pinned package does. |
| R13 | Globals and calls | The lock inventories globals known to pinned `index.ts` (names, arities, and host-supplied flag); the flag distinguishes globals it ships from names it expects the host to supply. Mapped tests document their behavior and the pinned context-versus-global shadowing order. `RenderOptions` admits explicitly named `HostFunction` entries for any name not occupied by a built-in; duplicates and built-in collisions fail when `RenderOptions` is constructed. The admission rule and differential-oracle shim both consume the flag. No other Java callable is visible. |

### Error taxonomy

M1 defines and documents the corpus error enum: `SYNTAX`, `UNDEFINED_OR_ACCESS`, `TYPE`, `ARITY`,
`VALUE`, `EXPLICIT_RAISE`, `HOST_FUNCTION`, `HOST_CONVERSION`, `RESOURCE_LIMIT`, and `OUTPUT`.
Java exceptions retain their detailed message and an optional source location; compatibility
compares only Node-comparable categories when the pinned Node package reports an error.
`HOST_FUNCTION`, `HOST_CONVERSION`, `RESOURCE_LIMIT`, and `OUTPUT` are not parity outcomes and
are excluded from the differential corpus. `RESOURCE_LIMIT` means any configured budget exhaustion,
including output size; `OUTPUT` means an `Appendable`/I/O write failure only.
`ErrorCategory` is public v1 API: consumers must include a `default` when switching, because new
categories may be added in minor releases.

## Public API

Keep version 1 deliberately small; implementation packages remain internal.

```java
package se.alipsa.jmlx.jinja;

public final class Template {
  public static Template parse(String source);
  public static Template parse(String source, TemplateOptions options);
  public String render(Map<String, ?> context);
  public String render(Map<String, ?> context, RenderOptions options);
  public void render(Map<String, ?> context, Appendable output);
  public void render(Map<String, ?> context, Appendable output, RenderOptions options);
}

public final class TemplateOptions { /* parsing limits and syntax options */ }
public final class RenderOptions { /* render limits; explicitly named HostFunctions; Clock and ZoneId */ }
@FunctionalInterface
public interface HostFunction { Object apply(List<Object> arguments); }
public record SourceLocation(int offset, int line, int column) { }
public class HfJinjaException extends RuntimeException {
  public Optional<SourceLocation> location();
  public ErrorCategory category();
}
public enum ErrorCategory {
  SYNTAX, UNDEFINED_OR_ACCESS, TYPE, ARITY, VALUE, EXPLICIT_RAISE,
  HOST_FUNCTION, HOST_CONVERSION, RESOURCE_LIMIT, OUTPUT
}
public final class TemplateSyntaxException extends HfJinjaException { }
public final class TemplateRenderException extends HfJinjaException { }
```

**Amendment: `HfJinjaException` was renamed to `JinjaException` when this project was migrated into
the `jmlx` repository as `jmlx-jinja`.** The package move to `se.alipsa.jmlx.jinja` was already a
breaking change for every consumer, so the rename rode along in the same release rather than
costing a second break later. `TemplateSyntaxException`/`TemplateRenderException` and the rest of
this API sketch are otherwise unchanged.

`render` converts and copies the supplied value graph, retaining no caller collections. Supported
inputs are `String`, `Boolean`, `Number` as defined by R11, Java arrays, `List<?>`, and
`Map<String, ?>`. This admits ordinary Gson/Jackson/org.json number subclasses and plain
`BigDecimal` decimal values such as `0.7`. `Character`, `Instant`, enums, records, beans, and
arbitrary objects are rejected. Map keys must be strings and cycles fail clearly. These entry
conversion failures are `TemplateRenderException`s in `HOST_CONVERSION` with no source location.
Never fall back to JavaBean access or reflection.

M0 verifies the globals shipped by the pinned `index.ts`, records their inventory in the lock, and
records behavior in mapped tests and API documentation rather than assuming a Jinja default.
`RenderOptions` may add a bounded map of explicit names to `HostFunction` values.
Unknown-to-upstream names are permitted; duplicate and built-in names fail when options are
constructed. The inventory host-supplied flag distinguishes built-ins from host-expected names and
is consumed by both this admission rule and the differential-oracle shim. M0 derives and tests the
pinned context-versus-global shadowing order from `index.ts`. Host functions are invoked only as
template functions and receive converted, inert
positional argument values (scalars and recursively immutable lists/maps only, never caller
objects). A HostFunction call with keyword arguments fails as `HOST_FUNCTION`; keyword arguments
are not represented in this v1 API. Template `undefined` cannot be represented at this closed Java
boundary, so a HostFunction call containing it (including nested in a list or map) fails as
`HOST_FUNCTION` rather than silently converting it to `null`. Its return value undergoes the same
closed conversion above. `HostFunction.floatResult(double)` is the explicit return-only marker for
an integral template float such as `2.0`; `HostFunction.integerResult(double)` is the corresponding
marker for a computed integral result outside the JavaScript safe-integer range. Neither is accepted
in a render context. A
conversion violation is a `TemplateRenderException` in `HOST_FUNCTION`, never a raw
`ClassCastException`. Any `RuntimeException` thrown by the function is likewise wrapped in a
location-bearing `TemplateRenderException` in `HOST_FUNCTION`. `Appendable` write failures are
reported as `TemplateRenderException` in `OUTPUT`.
This supports caller globals such as `strftime_now` without exposing host methods. Time-dependent
functions must use the supplied `Clock` and `ZoneId`; callers seeking reproducibility provide fixed
values. A render without options has no clock: first use of a time global fails rather than reading
the system clock. Strftime-style output uses C/POSIX month/day names. The `String` overloads are
atomic from the caller's perspective; streaming `Appendable` overloads offer no rollback after
partial output. An `Environment` extension API and arbitrary filters remain out of scope for v1.

## Architecture

Port upstream concepts one-to-one while using idiomatic Java. This correspondence is a deliberate
maintenance feature.

| Upstream file | Java area | Responsibility |
| --- | --- | --- |
| `lexer.ts` | `internal.lexer` | Scanner, whitespace controls, tokens, source spans. |
| `parser.ts` | `internal.parser` | Recursive-descent grammar and precedence. |
| `ast.ts` | `internal.ast` | Sealed AST interfaces and records with spans. |
| `runtime.ts` | `internal.runtime` | Values, environment, interpreter, filters, tests, JSON formatting. |
| `utils.ts` | `internal.util` | Slicing and string/date helpers matching upstream. |
| `format.ts` | `reviewed-no-port-impact` | No v1 public or runtime path requires it; retain this explicit ledger record rather than port untestable dead code. |
| `index.ts` | `Template` facade | Public entry points only. |

The runtime uses a sealed `Value` hierarchy: undefined, null, boolean, integer, float, string,
array, object, and function. Do not use Java `null`, `equals`, collection defaults, or numeric
promotion as implicit template semantics. Use insertion-ordered object keys and implement the
specific ordering points required by upstream.

The interpreter receives an immutable AST plus a per-render environment and `RenderBudget`.
Internal result variants represent break/continue; user-visible problems use location-aware
exceptions. `tojson` is implemented locally to avoid a production JSON dependency and is tested
for escaping, indentation, key ordering, undefined values, separators, and number formatting.

The parser has no static mutable state. Parsed ASTs and filter/test tables are immutable; scopes,
budgets, and conversion state are per render. `SourceLocation` is v1 exception data only: syntax
and render exceptions carry the offending span; `Template` has no synthetic whole-template
`location()` method.

## Upstream sync design

Upstream synchronization is a product feature, not a release-time manual exercise.

### Pin and vendor source

Commit `upstream/upstream-lock.json` with the npm package name/version, Node oracle version, full
`huggingface.js` commit, tarball URL, npm `dist.integrity`, derived tarball SHA-256, fixture
revision, shipped-global inventory, and SHA-256 for every retained source file. Record every
policy-excluded tarball file in `excludedFiles` with its SHA-256 and reason. The initial values are
real verified values—not placeholders. A normal Java build never fetches network data.

Commit the selected upstream `src/`, applicable `test/`, package metadata, and LICENSE under
`upstream/vendor/`. This gives reviewers an offline source-of-truth and makes updates reproducible.

### Sync tool

Provide `tools/sync-upstream` as a reviewed Gradle task or small Java CLI:

1. `fetch --version X` downloads the npm tarball only when explicitly requested, verifies npm
   registry `dist.integrity` on trust-on-first-use, records the derived SHA-256 and the integrity
   value in the lock, and verifies every current-version `excludedFiles` digest against the
   extracted tarball before removing the excluded paths. For a version change, it instead records
   candidate exclusion digests from the extracted tarball; reviewers approve their paths, reasons,
   and new digests in the diff report before they replace the lock entries. It then extracts the
   approved files and updates the lock. The generated diff report is the review gate for accepting
   a new version.
2. `report` compares vendor content to the lock and generates `build/reports/upstream-diff.md`:
   changed files, changed line counts, changed exports/tests, and mapped Java areas.
3. `verify` runs offline and fails on a missing/changed vendor hash, malformed or missing policy
   exclusions, a present policy-excluded file, stale lock, or stale mapping.

### Mapping ledger and differential oracle

Commit `upstream/mapping.yml`, for example:

```yaml
runtime.ts:
  java:
    - src/main/java/se/alipsa/jmlx/jinja/internal/runtime/Interpreter.java
    - src/main/java/se/alipsa/jmlx/jinja/internal/runtime/Values.java
  tests:
    - src/test/java/se/alipsa/jmlx/jinja/UpstreamRuntimeTest.java
parser.ts:
  java:
    - src/main/java/se/alipsa/jmlx/jinja/internal/parser/Parser.java
```

If a sync changes a mapped upstream file, the update must modify the corresponding Java code/tests
or add an explicit `reviewed-no-port-impact` record. Every such record includes the reviewed
upstream file SHA-256; a changed hash invalidates the record and forces re-review. CI enforces that
discipline.

The tool converts upstream unit/e2e tests and real-model templates into a JSONL differential corpus
(`template`, `context`, optional fixed `instant`, `zone`, named host globals, and expected value or
error category). Successful expected output is generated by the pinned `@huggingface/jinja` Node
package, using the Node version recorded in the lock; it is compared byte-exact, including Unicode,
whitespace, and line endings. Transformers' Python Jinja2 output is not an oracle, and divergence
from it is a documented non-goal. The runners share a versioned oracle shim: for each record it
always supplies the fixed instant and zone, plus every non-built-in global—both inventory-flagged
host-supplied globals and record-declared names absent from the inventory—with identical
implementations. Built-in globals run independently in Node and Java so the corpus tests both
implementations. On an update branch the corpus runs once against that pinned Node package and once
against Java. Normalization applies only to errors, by matching a documented category; Node is an
update-time oracle only, never a runtime dependency. Corpus processes have an external wall-clock
timeout, reported as harness failure and never as a parity result; render budgets remain effectively
unbounded. Preserve every mismatch as a Java regression test before accepting a new upstream
revision.

## Project layout and build

```text
hfjinja/
  build.gradle                  # Java 21 toolchain; no production dependencies
  settings.gradle
  src/main/java/module-info.java
  src/main/java/se/alipsa/jmlx/jinja/
  src/test/java/se/alipsa/jmlx/jinja/
  src/test/resources/fixtures/
  upstream/
    upstream-lock.json
    mapping.yml
    vendor/
  tools/sync-upstream/
  req/project-description.md
  NOTICE
  LICENSE
```

Use Gradle, JUnit 5, formatting/linting, reproducible JAR configuration, and an SBOM/dependency
report. Ship `module-info.java` in M0 and export only `se.alipsa.jmlx.jinja`; do not use an
`Automatic-Module-Name` transitional configuration.

## Delivery phases

### M0 — bootstrap and baseline

- Create the Gradle project, package namespace, CI, license/NOTICE, and release metadata.
- Pin and vendor one upstream package revision; implement `sync-upstream report` and `verify`.
- Confirm that npm release `0.5.9` has the intended initial surface before accepting its lock;
  otherwise choose and document a different version through this same reviewed pin step.
- Generate the supported AST-node inventory from pinned `ast.ts` and compare it with a declared
  allowlist: each node is either implemented or has an explicit milestone exemption in the mapping
  ledger. CI fails on an unaccounted node. Verify and record every shipped global from pinned
  `index.ts`: put its name, arity, and host-supplied flag in the lock and its behavior in mapped
  tests.
- Define `HostFunction`, deterministic `Clock`/`ZoneId` handling, and host conversion rules before
  runtime implementation.
- Import smoke cases as pending fixtures—literal text, lookup, loop, filter, one Llama template,
  and one Qwen template—and enable them as passing differential cases in M2.

Exit: `./gradlew check upstreamVerify` works offline and verifies provenance.

### M1 — lexer, AST, and parser

- Port lexical states, whitespace behavior, spans, AST, grammar, and precedence.
- Convert lexer/parser tests and add malformed-input location tests. Publish the M1 error taxonomy
  and map every corpus failure to it.

Exit: all imported parse cases produce equivalent AST snapshots or equivalent syntax failures.

### M2 — runtime core

- Implement values, host conversion, scopes, expressions, assignment, conditionals, loops, and
  loop metadata.
- Implement operators, tests, and filters in upstream source order, each with converted tests.
- Add concurrency and limit tests.

Exit: baseline differential corpus and Llama/Qwen goldens pass. Mistral and tool-use templates,
which depend on macros, are M3 gates rather than M2-adjacent requirements.

### M3 — full upstream feature parity

- Implement macros, call/filter blocks, slices, keyword/spread arguments, `tojson`, runtime
  formatting helpers (not unreachable `format.ts`), and `raise_exception`.
- Add fuzz/property tests for lexer/parser termination and retain every differential regression.

Exit: the complete pinned upstream suite and retained model corpus pass, and the AST allowlist has
no remaining milestone exemptions.

### M4 — release readiness

- Document API, limits, errors, and tokenizer-client integration.
- Add a minimal local example; do not add a Hub client.
- Complete clean-room dependency and license review, then publish `0.5.0`.

Exit: a consumer adds one JAR, parses once, renders safely from several threads, and reproduces
documented model-template output.

## Test matrix

| Layer | Required coverage |
| --- | --- |
| Lexer/parser | Imported token/AST cases, whitespace edges, malformed locations. |
| Values | Truthiness, equality, coercion, absent-key undefined versus present-null, member access, slices, and explicit host conversion table. |
| Numbers | `//`, `%`, negative operands, division types, integer-looking floats, computed non-finite values and `tojson`, safe-integer boundaries, and value-based conversion of Gson/Jackson-style `Number`s; mapped in `mapping.yml` to a dedicated numeric test file. |
| Host functions/globals | Arity and keyword rejection, thrown exceptions, return-value conversion, clock/zone injection, missing-clock behavior, C/POSIX strftime names, built-in collision and duplicate-name rejection at construction, and the M0-derived context-versus-global shadowing order. |
| Runtime | Every filter/test/operator/control-flow feature, scopes, macros, JSON. |
| Compatibility | Imported upstream vectors and pinned-Node-vs-Java corpus using effectively unbounded budgets; success text byte-compares and only error categories normalize. |
| Models | Exact Unicode/line-ending Llama, Qwen, Mistral, and tool-use goldens. |
| Safety | Cycles, unknown objects, deep nesting, giant output, budget exhaustion, concurrency. |
| Updates | Broken hash, stale mapping, changed upstream files, and a `reviewed-no-port-impact` record whose reviewed hash no longer matches each fail verification. |

Never normalize whitespace in goldens: for chat templates, whitespace is product behavior.

## Key decisions and risks

| Topic | Decision | Why |
| --- | --- | --- |
| Compatibility source | Pinned `@huggingface/jinja`, not generic Jinja2 docs | HF validates this implementation for chat templates. |
| Golden source | Pinned Node package and locked Node version | Keeps R2 falsifiable; Python transformers/Jinja2 differences are non-goals. |
| Callable globals | Explicit `HostFunction` entries plus injected clock/zone | Supports real Llama caller globals without reflection or ambient time. |
| Port style | Conceptual source mapping, idiomatic Java types | Makes upstream changes discoverable and reviewable. |
| Dependencies | None at runtime | Keeps the library portable and avoids JS-engine complexity. |
| Update cadence | Deliberate, reviewable sync PRs | Avoids unreviewed behavior drift from an active upstream. |
| Security | Closed values plus budgets | A model template must not gain ambient JVM capability or unlimited work. |

The chief risk is upstream evolution. The lock, committed source, mapping ledger, and differential
corpus make each difference visible rather than silently diverging. Java collection, Unicode,
numeric, and JSON differences are the other main risk; exact imported fixtures and model goldens
are therefore release gates, not optional test coverage.

## 1.0 definition of done

Release 1.0 only after the public API is stable; all fixtures and retained model templates pass;
`upstreamVerify` proves offline provenance; concurrent rendering and resource limits are tested;
license obligations are met; and a new upstream release can be assessed through one report command
plus the differential test suite.
