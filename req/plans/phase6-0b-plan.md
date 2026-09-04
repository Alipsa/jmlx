# Phase 6.0b implementation plan — native inventory, oracle, and fixtures

**Roadmap:** `req/full-roadmap.md` §Phase 6.0b  
**Parent plan:** `req/plans/phase6-plan.md` §6.0b

## Objective

Phase 6.0 is merged; Phase 6.0b is the next dependency-gated milestone. Establish the evidence needed
before Phase 6.1: probe selection and explicit RNG-key behavior in the pinned MLX runtime, publish a
deterministic binding inventory, block unrecorded handwritten FFI use, and run a small pin-derived
Python oracle plus Tier-A fixtures in existing CI. It changes neither the public generation contract
nor sampling APIs.

## Constraints

- `scripts/bootstrap-native.sh` is the sole source for `MLX_METAL_VERSION`, wheel provenance, and
  `MLX_C_COMMIT`; one `buildSrc` parser supplies every report/task that needs them. A native-only
  check compares its result with staged `native/install/lib/native-pin.properties` when present.
- Inventory input is the committed generated binding sources under
  `jmlx-ffi/src/main/generated/java`, not a network header. `mlx_h.java` supplies function and
  constant accessors; sibling generated sources supply layouts/accessors and upcall interfaces.
- Native probes use the existing `MLXScope`/ `NativeOps` cleanup patterns and native-availability
  gating. Required PR checks stay credential-free and never download real model artifacts.
- The new scanner considers only handwritten Java sources; generated bindings are audit input,
  never violations.

## Implementation slices

### 1. Deterministic inventory

Apply Gradle's `base` plugin in the root `build.gradle` to create the root lifecycle `check` task,
then add `buildSrc` inventory rendering and root tasks:

- `generateMlxApiInventory` renders `req/mlx-api-inventory.md`;
- `verifyMlxApiInventory` regenerates to a temporary location and fails when the checked-in file is
  stale; attach this verification task to root `check`, while retaining independent
  `buildSrc:check`.

Commit a machine-readable mapping, `req/mlx-api-inventory-overrides.json`, containing explicit
records for every non-`unplanned` symbol/group and every symbol/type reached by handwritten code.
Each record supplies generated binding identity, category, status
(`unplanned`, `planned`, `implemented`, `unsupported-by-runtime`, `derived-java-api`, or
`internal/unstable-upstream`), public facade/reason, test location, and optional probe reference.
The generator extracts all callable `mlx_h.mlx_*` methods and generated enum/constant accessors,
sorts them by native symbol, and auto-renders an `unplanned` record for an unlisted binding. It
rejects unknown, duplicate, and stale explicit records, parses both pins through the shared parser,
and renders per-group/status totals. Inventory sections also enumerate generated layout/accessor
types and upcall interfaces. The generated default is a reporting convenience only: it never
authorizes handwritten use.

Add a native-job-only header cross-check after bootstrap. It parses staged
`native/install/include/mlx/c/` declarations, compares them to generated binding identities, and
reports counts for `declared-and-bound`, `declared-but-not-bound`, and `bound-without-header-match`.
The report is evidence, not an automatic assertion that every mlx-c declaration should be public:
it makes jextract filtering/silent drops visible so later parity accounting can distinguish
unbound declarations from `unsupported-by-runtime` entries. Render it separately as committed
`req/mlx-api-header-coverage.md` through `generateMlxApiHeaderCoverage`; verify it only in the
native job with `verifyMlxApiHeaderCoverage`. `verifyMlxApiInventory` remains fully reproducible on
Ubuntu because it has no staged-header-dependent output.

Test renderer ordering, byte-identical repeat output, automatic-unplanned rendering, explicit
mapping validation, stale/unknown/duplicate mappings, missing pins, and rendered fields with small
binding/bootstrap fixtures in `buildSrc`.

### 2. Native-call guard

Add root `verifyMlxApiCallSites` in `buildSrc`, wired to root `check`. It is source-only and has no
compile/task dependency, so it remains runnable in the Ubuntu CI environment without resolving the
Java 25 toolchain. It scans handwritten source and requires an explicit mapping-file record for:

1. `mlx_h.mlx_*` downcalls;
2. generated layout/accessor type uses such as `mlx_array_`;
3. generated upcall interfaces, including generated `$fun` and `$dtor` names; and
4. generated `mlx_h` enum/constant accessors, including dtype/device constants.

Use a Java parser or token-aware scanner, not substring matching, so comments, literals, and imports
do not fail. A use is a violation when its symbol/type has no explicit record in
`req/mlx-api-inventory-overrides.json`, or when its explicit record is `unplanned`; a matching
auto-rendered inventory row is not sufficient. Exclude generated and build output. Explicitly classify jextract implementation
infrastructure (`mlx_h.C_*` layouts and `mlx_h.upcallHandle`) as named non-violations rather than
as inventory-backed MLX API use; any other new constant accessor, such as `MLX_COMPLEX64`, needs an
inventory record. Test-only exemptions live in the mapping file with an exact source path and
justification; reject broad wildcards. Add fixture tests for every detected form, false-positive
avoidance, unknown uses, an auto-unplanned symbol used from handwritten source (which must fail),
generated-source exclusion, named infrastructure exemptions, and allowed/disallowed test exceptions.
Classify all existing production/probe sites before enabling the task and document the commands in
`CLAUDE.md`.

### 3. Selection and RNG probes

Add guarded native tests in `jmlx-core` (for example `SelectionAndRandomProbeTest`). Use compact
explicit inputs, force evaluation/readback, and clean up all handles on success and failure.
`MLXOps.argmaxAxis` is the existing public path for argmax tie-breaking, laziness, and resulting
shape; its documented INT32 conversion means it cannot evidence the native result dtype. Include
`mlx_argmax_axis` in the direct-binding probe exception solely to observe the native UINT32 dtype.
Direct binding use for the remaining probes is limited to the missing-facade selection/random
declarations below and is a documented inventory probe exception.

Probe the exact pinned behavior of:

- `mlx_argmax_axis`: native UINT32 dtype (direct binding), axes, tie-breaking, lazy evaluation, and
  output shape (facade and direct result as appropriate);
- axis-aware decoder candidates `mlx_topk_axis`, `mlx_sort_axis`, `mlx_argsort_axis`,
  `mlx_partition_axis`, and `mlx_argpartition_axis`, plus their flat counterparts (`mlx_topk`,
  `mlx_sort`, `mlx_argsort`, `mlx_partition`, `mlx_argpartition`) to explicitly establish the
  flattening distinction: ordering, ties, boundary cases, shape/dtype, partition guarantees, lazy
  behavior, and ownership;
- categorical selection: accepted input representation, result shape/dtype, invalid input,
  fixed-key repeatability, and evaluation boundary; and
- explicit random key construction and both `mlx_random_split` (two-way) and
  `mlx_random_split_num` (N-way): key representation, parent mutation, split order, fresh-JVM
  reproducibility, and independence of sibling keys. Commit expected key bytes as a golden and
  compare it in every probe run, providing fresh-JVM evidence without a separate forked test JVM.
  The `split_num` result decides whether per-request/batched streams are viable.

Do not treat global `MLXRandom.seed` as per-request evidence. If bindings or results cannot support
isolated request streams, stop before Phase 6.1 API design and amend its contract. If argmax tie
breaking or laziness invalidates the merged deterministic-greedy claim, also amend
`jmlx-models/README.md` and `req/phase6-compatibility.md` before accepting Phase 6.0. Record exact
input arrays, commands, Java/macOS/device details, both pins, observations versus inferences,
cleanup findings, and 6.1/6.0 follow-up decisions in `req/plans/phase6-0b-probe-findings.md`; link
records from the inventory.

### 4. Pin-derived Python oracle

Create `tools/mlx-oracle/` with a JSON-in/canonical-JSON-out reference runner. It is used to
explicitly generate and verify named fixtures, never invoked by ordinary Java unit tests.

Add `generateMlxOracleFixtures`, `verifyMlxOracle`, and `verifyMlxOracleFixtures` Gradle tasks.
They use the shared bootstrap-pin parser. In native CI, provision Python MLX only through a lock file
with fully pinned frontend `mlx` and backend `mlx-metal` distributions and `pip --require-hashes`;
do not attempt to install the unpacked bootstrap wheel directory. The lock's `mlx-metal` hash must
equal `MLX_METAL_WHEEL_SHA256` from `scripts/bootstrap-native.sh`. Record the frontend and backend
versions separately, along with wheel SHA-256/provenance, mlx-c commit, platform, and Python
version. `verifyMlxOracle` verifies that exact environment and provenance;
`generateMlxOracleFixtures` is the only rewriting task; `verifyMlxOracleFixtures` is read-only.
Fail clearly on pin/version mismatch and report oracle infrastructure failure rather than asserting
Java incorrectness.

### 5. Tier-A fixtures and CI

Commit small documented fixture specifications/seeds, prompt and token data, expected shapes/dtypes,
and oracle JSON. Generate tiny safetensors during tests using the established
`LlamaModelTest`/`MLXIO.saveSafetensors` pattern rather than committing review-opaque binary blobs;
commit a binary only if a separately documented parser/interoperability behavior cannot be represented
by that deterministic generator. Add a minimal decoder fixture exercising the existing common
Llama/Qwen API through prefill and short decode, and pure-Java tokenizer golden tests. Assert resource
cleanup on terminal/failure paths. Do not add an architecture to fill this milestone.

Add `req/phase6-tier-b-artifacts.md` as a manifest template for artifact/revision/hash, license,
access requirement, expected metadata, cache/size policy, and trigger owner. A real-artifact workflow
is manual/scheduled only and is not a PR gate.

Amend both existing CI jobs; this is mandatory because neither currently executes a root lifecycle
task. Ubuntu explicitly invokes root inventory/call-site verification after `buildSrc:check` and runs
pure-Java tokenizer goldens. The native job runs `verifyMlxApiHeaderCoverage`, installs/verifies the
oracle after bootstrap from verified/locked artifacts, then explicitly invokes root inventory/call-site
verification, oracle verification, and Tier-A native tests. CI diagnostics distinguish stale
inventory, unrecorded native use, header/binding drift, bootstrap/provenance, oracle, and Java-fixture
failure.

## Change map

| Area | Change |
| --- | --- |
| `buildSrc` and root build | Shared pin parser, inventory renderer, native-use scanner, header cross-check, root `base` lifecycle and verification tasks. |
| `req/` | Generated inventory, separate native header-coverage report, mapping, probe findings, Tier-B manifest. |
| `jmlx-core` | Guarded selection/RNG probes and Tier-A decoder integration checks. |
| `tools/mlx-oracle` | Pin-derived setup/provenance and explicit fixture commands. |
| `jmlx-tokenizer` | Pure-Java Tier-A golden checks. |
| CI and `CLAUDE.md` | Native post-bootstrap checks and contributor instructions. |

## Verification and gate

On Apple Silicon after `./scripts/bootstrap-native.sh`:

```text
./gradlew -p buildSrc check
./gradlew verifyMlxApiInventory verifyMlxApiCallSites
./gradlew :jmlx-tokenizer:check
./gradlew :jmlx-core:check :jmlx-models:check
./gradlew verifyMlxApiHeaderCoverage
./gradlew verifyMlxOracle verifyMlxOracleFixtures
```

Exact task names may evolve, but generation must be explicit and verification read-only. Java checks,
including ordinary root `check`, remain network-free; the native CI-only oracle provisioning step is
the explicitly locked/verified exception. Accept 6.0b only when the inventory is reproducibly
generated with both pins; the separately verified native header-coverage report records
header/binding counts; the guard covers all four native-use forms; probe findings resolve argmax and
per-request `split_num` semantics (and correct the shipped 6.0 claim if needed); oracle provenance
and a deterministic fixture are committed; and credential-free Tier-A decoder/tokenizer fixtures
pass in their respective CI jobs.

## Non-goals

No public top-k/sort/categorical/key APIs, sampling policy, log probabilities, batching, cache work,
new model family claims, hidden Python execution during Java tests, or required Hugging Face download.
