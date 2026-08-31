# hfjinja implementation plan

This plan implements the contract in [project-description.md](project-description.md). Work is
ordered so that provenance and a repeatable oracle exist before behavior is ported.

## Working rules

- Pin `@huggingface/jinja` 0.5.9 only through a reviewed `sync-upstream` change. The Java build is
  offline; Node is used only by the explicit oracle/update workflow.
- Keep `upstream/mapping.yml` current in the same change as every ported upstream behavior.
- Add a differential mismatch as a focused Java regression before changing expected results.
- Do not add a production dependency to solve parsing, JSON, date formatting, or execution.
- Keep implementation packages unexported. Only `se.alipsa.jmlx.jinja` is public.
- Keep Gradle's configuration cache enabled. `upstreamVerify` must resolve project paths during
  configuration rather than retain Gradle model objects in execution closures. This single-project
  build uses two forked test JVMs for actual test parallelism; project-level parallel execution is
  intentionally not enabled. Corpus tasks must declare every resource tree they read as an input.

## Work packages

### WP0 — bootstrap, provenance, and build

1. Create `settings.gradle`, `build.gradle`, Gradle wrapper, Java 21 toolchain, JUnit 5, format
   and lint tasks, reproducible JAR configuration, and `src/main/java/module-info.java`. Configure
   a Gradle white-box test source set using `--patch-module`/qualified opens (or equivalent) so
   JUnit can test `internal.*` without exporting it from the published JAR.
2. Create the public package skeleton: `Template`, `TemplateOptions`, `RenderOptions`,
   `HostFunction`, `SourceLocation`, `ErrorCategory`, and exception classes. Keep methods as
   explicit stubs until their work package lands.
3. Add `LICENSE`, `NOTICE`, Maven publication metadata, CI for `check` and `upstreamVerify`, and
   dependency/SBOM reporting.
4. Implement the lock reader and offline `upstreamVerify` task. It verifies package/version
   against the vendored package, source hashes, fixture revision, AST/global inventories, and the
   required policy-exclusion records; it rejects a malformed exclusion record or an excluded file
   present in the vendor tree. It records commit, tarball integrity/SHA-256, and Node version as
   provenance metadata that remains unverifiable offline until the deferred `fetch` workflow lands.
5. Implement `sync-upstream report` and `verify`. `report` emits mapped-file impact and invalid
   no-impact-record hashes. `refresh-lock` updates vendored file hashes only; AST and global
   inventories require explicit review and lock updates. The networked `fetch` workflow is deferred
   until an upstream update workflow is implemented.

Deliverables: green offline build; a real 0.5.9 lock and vendor tree; `NOTICE`; a reviewable
upstream diff report.

### WP1a — upstream inventory and licensing

1. Vendor the selected upstream sources/tests and generate an `ast.ts` node inventory.
2. Add an allowlist ledger in `mapping.yml`: every node is either implemented or carries a named
   milestone exemption. Fail verification on an unaccounted node.
3. Extract the `index.ts` global inventory: name, arity, built-in versus host-supplied flag, and
   context/global shadowing order. Record the inventory in the lock and behavior in mapped tests.
4. Make the model-template licensing determination before committing model template text. Record
   notices in `NOTICE`; retain a hash-only Llama case where its terms make vendoring unsuitable.
   The current decision is recorded in `model-fixture-policy.md` and permits no model template text
   until a fixture-specific source and license record is added.

Deliverables: `upstreamVerify` detects stale ledger entries; the AST/global inventories and
milestone ledger are complete; the model fixture form is approved. WP1a blocks WP2 and WP3.

### WP1b — differential corpus and oracle harness

1. Define the JSONL corpus schema:

   ```json
   {"template":"...","context":{},"instant":"...","zone":"UTC",
    "globals":{"strftime_now":{"kind":"strftime_now"}},"expected":{"text":"..."}}
   ```

   Text-bearing records are used only for fixture revisions approved for text/output retention.
   Hash-only records carry no model expression:

   ```json
   {"templateSha256":"...","modelRepo":"...","modelRevision":"...",
    "templatePath":"...","context":{},"expected":{"sha256":"..."}}
   ```

   `template`/`expected.text` and `templateSha256`/`expected.sha256` are mutually exclusive. A
   hash-only failure instead uses `"expected":{"errorCategory":"..."}`. Hash-only records may
   retain self-authored test context but no model template or rendered output.

   The normal build never downloads a model and does not execute hash-only corpus cases. An explicit
   corpus invocation receives externally supplied fixture material, fails loudly if it is absent or
   fails `templateSha256`, and either compares the UTF-8 SHA-256 of its exact rendered output to
   `expected.sha256` or compares the error category. No newline normalization is applied; invalid
   UTF-8 fixture material is a harness failure.

   The `strftime_now` entry is illustrative only: it is a built-in and the runner always supplies
   a fixed instant/zone/locale (defaulting to `2000-01-02T03:04:05Z`, `UTC`, and `en-US`).
   `instant` and `zone` override that pair together; `globals` is reserved until the pinned Template API can inject
   non-built-in globals. Expected failures carry an `ErrorCategory`; a failure whose diagnostic
   text is itself part of the compatibility contract may also pin it with `errorMessage`.
2. Implement the versioned Node-message-to-`ErrorCategory` pattern mapping table as part of the
   oracle shim. Patterns extract interpolated message values (for example a filter name) rather
   than matching literals. Every known upstream error maps explicitly; an unmatched message fails
   the harness loudly and never defaults to a category unless the record pins the diagnostic with
   `errorMessage`. Exact-message records still cross-check a recognized category. They may bypass
   the pattern table only for deliberately unclassifiable diagnostics whose literal text is the
   contract (for example `raise_exception`'s arbitrary user message), not as a substitute for
   adding a generalizable category pattern.
3. Build the upstream-test-to-corpus converter. Automate extraction from vendored non-model unit
   sources where structurally representable. The upstream e2e source is excluded in full because it
   contains model-derived material. Retain reviewed manual transcriptions only for unsupported
   harness constructs and approved real-model templates, with source locations recorded.
4. Implement Node and Java corpus runners. The versioned shim always supplies fixed time/zone and
   every non-built-in record global; built-in globals run independently. Apply an external
   wall-clock timeout to each process and report it as a harness failure, never a parity result.
5. Import literal, lookup, loop, filter, and licensing-approved Llama/Qwen fixtures as pending
   cases; enable them when WP4 provides the required runtime behavior.

Deliverables: the converter produces a baseline corpus and coverage report over vendored test
sources, including the enumerated manual-transcription list; Node errors have an exhaustive
versioned pattern classifier; the Node runner produces byte-exact goldens; corpus records can
represent deterministic time and globals. WP1b blocks G4, not WP2 or WP3.

### WP2 — values, public boundary, and diagnostics

1. Implement immutable internal `Value` variants: undefined, null, boolean, integer, float,
   string, array, object, and function. Preserve the split integer/float model through evaluation,
   output, and JSON formatting. Use insertion-ordered objects and avoid Java collection semantics
   as a substitute for template semantics.
2. Implement recursive context conversion with cycle detection and caller-collection copying.
   Accept `String`, `Boolean`, arrays, lists, string-keyed maps, and `Number` according to R11's
   shortest-JS-round-trip rule. Reject unsupported values with `TemplateRenderException` and
   `HOST_CONVERSION`, with an empty location.
3. Implement `SourceLocation`, all public exceptions, and stable `ErrorCategory` propagation.
   Syntax and evaluated-render failures include locations; conversion-time failures do not.
4. Implement `RenderOptions` validation: reject duplicate function registrations and collisions
   with inventory-built-ins; permit unknown and host-supplied names. Define time globals to fail on
   first use when clock/zone is absent.
5. Implement HostFunction dispatch: positional-only arguments, keyword rejection as
   `HOST_FUNCTION`, defensive immutable arguments, return conversion, and wrapping of runtime
   exceptions at the call location.
6. Expose conversion tests before a full parser exists through package-private conversion tests (or
   a narrow internal test seam); do not defer the G2 boundary contract to WP3.

Tests: conversion matrix across Gson/Jackson-style `Number` subclasses; null versus undefined;
cycles and bad map keys; built-in collision/duplicate registration; missing clock; HostFunction
arity, return, and exception cases.

### WP3 — lexer, AST, and parser

1. Port lexer states, delimiters, comments, trim behavior, tokens, and source spans from the
   vendored source. Enforce `TemplateOptions` source and token limits while scanning.
2. Model `ast.ts` as sealed Java interfaces/records. Remove the matching allowlist exemption only
   when a node type and parser coverage are present.
3. Port grammar, precedence, expressions, statements, calls, slices, filters/tests, macros, and
   location-aware syntax errors incrementally. Enforce AST-depth limits while constructing nodes.
4. Treat parsed ASTs as complete immutable values: do not add lazy memoization caches or mutable
   per-template state. Rendering state remains per invocation.
5. Convert upstream lexer/parser fixtures and compare deterministic AST snapshots where meaningful.

Tests: whitespace and Unicode edges, malformed source locations, precedence tables, AST inventory
coverage, lexer/parser termination properties.

### WP4 — interpreter core

Status (2026-08-26): the pinned runtime's ordinary rendering feature *inventory* is implemented:
the formerly absent sequence/string/object filters and members, plus the remaining named tests,
have Java and Node-oracle coverage. This is not a claim of complete behavioral parity; WP7 records
the remaining semantic, diagnostic, corpus, and public-API closure work before release.

1. Build a fresh render environment per invocation: scopes, assignment, context/global lookup in
   the M0-derived order, loop state, and break/continue result variants.
2. Implement expression evaluation, member/index access, truthiness, equality, string/list/object
   operations, and undefined/null behavior in upstream source order.
3. Implement JavaScript double arithmetic under the split integer/float value model, independently
   of host-boundary validation: precision loss, `NaN`, infinity, negative modulo/floor division,
   division result types, and rendering.
4. Implement all upstream filters/tests as mapped work items. The pinned ordinary runtime
   filter/test/member names are implemented; WP7 owns complete per-operation oracle coverage and
   semantic closure. Implement deterministic JSON output locally, including JS-number formatting
   and `NaN`/infinity becoming `null`.
5. Implement all `RenderBudget` counters. String renders accumulate privately and are atomic;
   streaming renders write progressively and have no rollback after output or I/O failure.
   Do not introduce lazy memoization caches in `Template`; scopes, budgets, conversion state, and
   interpreter state are per render.

Gate: enable baseline corpus plus Llama/Qwen pending fixtures. Both runners must agree byte-for-byte
on successful output and by category on Node-comparable errors.

### WP5 — advanced language features and models

1. Implement macros, call blocks, filter blocks, keyword/spread arguments, slices, and
   `raise_exception`.
2. Implement date/format runtime helpers required by upstream, including C/POSIX names and injected
   `Clock`/`ZoneId`; do not port unreachable `format.ts` unless a mapped runtime path requires it.
3. Enable Mistral and tool-use model goldens, including macro-heavy templates. Slice 4 adds
   the pre-approved Mistral-7B-Instruct-v0.3 and Qwen2.5-32B-Instruct goldens; Slice 5 adds the
   separately reviewed, source-recorded, attributed Step3 macro-heavy fixture. The retained
   templates are resource-backed goldens, not `v1.jsonl` records: importing them as corpus cases
   remains a follow-up that first requires a schema change because text records currently reject
   model provenance fields.
4. Run fuzz/property suites with harness timeouts; preserve every oracle mismatch as a regression.
5. Remove every remaining AST allowlist exemption. Closed: `Interpreter.evaluateExpression`'s
   only remaining placeholder arms — `SliceExpression`, `KeywordArgumentExpression`, and
   `SpreadExpression` — now assert unreachability instead of throwing a template-facing "not
   yet supported" error, matching the parser guarantee that these three node types never reach
   that generic dispatch path. `evaluateStatement`'s M3 nodes (`Macro`, `FilterStatement`,
   `CallStatement`) already had no such placeholder. `upstream/ast-allowlist.json`'s per-node
   milestone tags are unaffected: `upstreamVerify` only checks that every discovered AST node
   has a key in the allowlist, and `AstInventoryTest` checks exact key-set equality — neither
   reads the milestone value, so there was nothing there to clear.

Gate: complete pinned upstream suite, all retained model corpus entries, and no unimplemented AST
nodes or stale mapping/no-impact records.

### WP6 — release hardening

1. Add concurrent-render tests over one parsed template; assert no lazy per-template caches or
   caller graph retention.
2. Exercise source/token/depth/step/loop/macro/output budgets separately from the parity corpus.
3. Confirm the WP1a model-template licensing decision and NOTICE remain accurate for release; do
   not introduce a new vendored template form at this gate.
4. Validate module metadata, Maven publication, reproducibility, API documentation, and a minimal
   local tokenizer-config integration example.
5. Run a clean checkout build, offline verification, dependency review, and release checklist.

### WP7 — pinned-upstream parity closure

1. Close every remaining normal-rendering semantic divergence against the pinned
   `@huggingface/jinja` runtime. The current known set includes eager filter-argument evaluation;
   Java arity caps and unknown-keyword rejection where upstream ignores arguments; no-argument
   sequence filters, macro/call-block `break`/`continue` propagation, the
   `tojson(sort_keys=true)` undefined-key edge, empty `first`/`last`, filter-form `replace` with
   one positional argument, `Object.get` with keyword arguments, `dictsort()` when an
   undefined-backed key must be compared with another key, and `is lower` on an undefined-backed
   string, and function-value rendering through filters such as `safe` and `default()`. The
   one-key undefined-key `dictsort()` corpus case is byte-exact; the two-key case must become
   byte-exact as well. Every difference must have a Node-oracle corpus case before
   production code changes, and no normal-rendering semantic or diagnostic difference may be
   accepted as a release exception. Java-only safety limits and host-boundary failures remain
   explicitly outside the Node runtime contract.
2. Expand `v1.jsonl` from representative cases to a complete reviewed mapping of all executable,
   non-model upstream runtime vectors. Add an explicit inventory report that fails when a supported
   upstream filter, test, member, global, or error family lacks either byte-exact coverage or an
   approved exclusion. Keep model-template provenance constraints intact.
3. Port the pinned `Template.format()` API with oracle coverage. Design its Java options surface
   explicitly for upstream's string-or-number `indent` value, retain the default-tab behavior, and
   map `index.ts` and `format.ts` as implemented only when public API, formatter, and oracle tests
   land together. A no-runtime-path note or public-API exclusion is insufficient for feature
   parity.
4. Specify the intended error contract per feature. Where exact upstream messages are practical,
   compare them; otherwise retain category-level comparison only after documenting why. In
   particular, resolve or document call-form filter diagnostics such as `safe(...)`/`items(...)`.
   The pinned upstream's `Unknown …Value filter: …` diagnostics, including `FunctionValue`, map
   explicitly to `TYPE`, so no-argument sequence-filter regressions are corpus-comparable.
5. Re-run the complete pinned upstream test suite through the reviewable converter and make the
   coverage report a release-blocking check. New upstream versions remain a separate reviewed
   sync, not an implicit upgrade.

Gate: all ordinary rendering behavior and public APIs in the pinned upstream are byte-exact; the
corpus/mapping inventory is complete; and `Template.format()` is implemented and oracle-tested.

## Delivery gates

| Gate | Required evidence |
| --- | --- |
| G0 | `./gradlew check upstreamVerify` works offline with real lock/vendor hashes. |
| G1a | AST and global inventories are complete and recorded in the lock; the milestone ledger rejects an unaccounted node; licensing determination, NOTICE, and approved fixture form are present. |
| G1 | Corpus runner has deterministic Node goldens, time/global schema, external timeout, no unmatched classifier messages across the corpus, and converter coverage over vendored tests with every manual transcription enumerated. |
| G2 | Public boundary, conversion, errors, and options validation tests pass. |
| G3 | Lexer/parser fixtures and AST ledger pass with only intentional later exemptions. |
| G4 | Core corpus plus Llama/Qwen goldens pass under effectively unbounded budgets. |
| G5 | Full pinned suite, Mistral/tool-use goldens, and complete AST ledger pass. |
| G6 | Packaging, notices, concurrency, limits, reproducibility, and consumer example pass. |
| G7 | Pinned-upstream parity closure: complete reviewed runtime/vector inventory, no normal-runtime exceptions, and an implemented `Template.format()` contract. This is the final release gate. |

## Suggested implementation sequence

Complete WP0 and WP1a before starting language code. WP1b may run in parallel with WP2 and WP3;
it blocks the WP4 corpus gate, not the lexer or value work. WP2 and WP3 may proceed in parallel once
the public API and upstream inventory are stable. WP4 follows their merged value/AST contracts. WP5
and the first three WP6 safety/licensing slices establish the runtime evidence needed for parity.
Prioritize WP7 parity closure, including `Template.format()`, ahead of the remaining WP6 packaging
and release-checklist work; release polish cannot compensate for a missing pinned-upstream feature.
