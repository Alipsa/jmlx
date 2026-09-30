# Phase 6.3 performance follow-up — gather-based mixture-of-experts

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `MoeMlp`'s dense expert loop (every expert runs on every token, then `where`
masks) with a gather-based route that only computes the `top_k` selected experts per token.
Routing semantics (expert choice, ties, renormalisation) are unchanged. Float32 output matches
the dense algorithm and the Hugging Face golden to `1e-4`. Half-precision output stays within a
measured bound of an exact closed form, but is **not** bit-identical to the dense algorithm: the
reduction order over the top-k axis differs (Verified facts 8).

**Architecture:** Expert weights are stacked once at load time into `[E, out, in]` tensors held by
a new `SwitchGlu` module. `MoeMlp` keeps its current router (float32 softmax, successive `argmax`
with lowest-index ties, renormalised top-k weights) and hands the `[B, T, K]` index tensor to
`SwitchGlu`, which runs gate/up/down through `mlx_gather_mm`. Large index counts are sorted by
expert first (`sorted_indices=true`) and unsorted afterwards, matching mlx-lm's `SwitchGLU`.

**Tech Stack:** Java 25 (`jmlx-core`, `jmlx-models`), mlx-c `fba4470` / `mlx-metal==0.31.2`
(`mlx_gather_mm`, `mlx_stack_axis` — both already in the committed jextract bindings), JUnit 6.

**Spec:** PR #25 review finding #6 (dense MoE compute is `E/top_k` × the MLP FLOPs), and
`req/plans/phase6-3-plan.md` Scope decision 4, which recorded dense MoE as "a documented 6.4
optimization target, not a 6.3 defect". This plan is that optimization. Reference implementation:
mlx-lm 0.31.3 `mlx_lm/models/switch_layers.py` (`SwitchLinear`, `SwitchGLU`, `_gather_sort`) and
`mlx_lm/models/mixtral.py` (`MixtralSparseMoeBlock`).

**Prerequisite:** PR #25 (`phase6-3`) merged, or this branch based on `phase6-3` **at or after
`c7613ad`** ("Address PR #25 review findings"). `SwitchGlu` uses `GELU(MLXScope, boolean)`,
which was added in that commit. Check before starting:
`git merge-base --is-ancestor c7613ad HEAD && echo ok`.
**Branch:** `phase6-3-moe-gather` (feature branch + PR; never commit to `main`).

## Global Constraints

- Platform: macOS on Apple Silicon, Java 25 toolchain, pinned runtime mlx-c `fba4470` against
  `mlx-metal==0.31.2`. No native pin change and no binding regeneration: `mlx_gather_mm` and
  `mlx_stack_axis` are already present in `jmlx-ffi/src/main/generated/java`.
- No new module, no new runtime dependency. `jmlx-core` keeps `jmlx-ffi` as `implementation`; no
  `jmlx-ffi` type appears on a public signature.
- Every new generated-binding use gets a non-`unplanned` record in
  `req/mlx-api-inventory-overrides.json`, then `./gradlew generateMlxApiInventory`;
  `verifyMlxApiCallSites` and root `check` must pass.
- Routing semantics are frozen: float32 softmax over router logits, `top_k` chosen by successive
  `argmaxAxis` (lowest index wins exact ties), weights = selected probabilities / their sum, cast to
  the hidden-state dtype. `MoeRoutingProbeTest` stays unchanged.
- Numerics: the Mixtral Tier-A golden (`MixtralModelTest`, tolerance `1e-4`) must pass unchanged,
  with no regenerated goldens. That golden is float32 and only exercises the **unsorted** path
  (6-token prompt × top-2 = 12 slots). The sorted path and half precision are anchored by
  Task 3's closed-form tests, not by the golden.
- Memory model: every per-call view of a model-scope weight (the `swapaxes` feeding `gather_mm`)
  is allocated into `x.scope()`, never into the weight's model scope (req/phase4-plan.md §2,
  mitigation 1 — the same rule `Linear.forward` follows with `transpose(W, x.scope())`).
- Autograd must keep working through `MoeMlp` (`ModuleGrad`): routing indices go through
  `MLXOps.stopGradient` before `gather_mm` (see Verified facts, item 4).
- Google Java Style, 2-space indent, 100 columns; `spotlessCheck` and checkstyle clean. The Java
  snippets in this plan are **not** google-java-format output: every verification step runs
  `spotlessApply` (`-p buildSrc` for buildSrc) before any `spotlessCheck`/`check`, which also
  fixes import order. Native
  tests use `@EnabledIfNativeAvailable` and are skipped, not failed, without a bootstrap.

## Verified facts (MLX 0.31.2 Python frontend, same wheel as the native pin)

These were checked before writing the plan and drive specific design choices. **They are
reproducible:** `tools/mlx-oracle/probes/moe_gather_probe.py` checks every fact, printing
`PASS`/`FAIL` for correctness facts and `MEASURED` for runtime behaviour, memory and timing, and
exits non-zero on any failure. Run it with the hash-locked oracle environment:

```sh
./tools/mlx-oracle/install.sh
./tools/mlx-oracle/.venv/bin/python tools/mlx-oracle/probes/moe_gather_probe.py   # add --no-slow to skip 6 and 7
```

Re-run it whenever the native pin changes. Facts 3 and 5 describe behaviour that no Java test can
observe; the probe is the only guard against them silently changing. Task 5 also appends a
summary to `req/plans/phase6-3-probe-findings.md`. The probe's first recorded run (2026-09-30,
mlx 0.31.2) reproduced every correctness result and memory ratio below exactly. Probe timings
cover the expert path only; the Java whole-forward numbers in fact 7 come from
`MoeMlpBenchmarkTest`, not the probe. All timings are approximate and vary by machine and run
(a later probe run measured T=1 dense 1.85 ms vs gathered 0.36 ms); treat ranges and ratios,
not individual milliseconds, as the claim.

1. `gather_mm(x[B,T,1,1,H], W.swapaxes(-1,-2)[E,H,F], rhs_indices=idx[B,T,K])` returns
   `[B,T,K,1,F]` and equals the per-token loop `x[t] @ W[idx[t,k]].T` exactly. INT32 indices are
   accepted (no UINT32 conversion needed); float indices are rejected.
2. The sorted path (flatten indices, `argsort`, gather tokens by `order`, `gather_mm(...,
   sorted_indices=True)`, then take by `argsort(order)`) matches the unsorted path to ~1e-6,
   forward and backward, for `top_k=2` and `top_k == E`.
3. `sorted_indices=True` on *unsorted* indices still returned the correct result on the probe. The
   flag is therefore a performance hint only, and **no test can detect a wrong flag** — review the
   call sites instead.
4. **Autograd:** when the indices derive from the differentiated parameters (router → argmax →
   indices), `mx.grad` fails with `[GatherMM] Cannot calculate VJP with respect to indices.`
   Wrapping the indices in `stop_gradient` fixes it, and the gradients then equal the dense
   `where`-based implementation exactly for all five parameter groups. The existing dense code has
   no such failure because `where`'s condition is not differentiated.
5. Out-of-range indices are **not** bounds-checked by `gather_mm` (no error, garbage output).
   Router-produced indices are always in range; `SwitchGlu`'s public entry point documents the
   precondition.
6. Load-time memory (4 layers × 8 experts × w1/w3): stacking per layer and freeing each layer's
   source tensors right after `eval` gives peak and steady memory of **1.125×** the weights.
   Keeping the source tensors alive gives **2.0×**, and freeing them without per-layer `eval` still
   peaks at **2.0×**. Loading from safetensors is lazy (0 bytes before first use) in the Python
   frontend. Whether jmlx's `MLXIO.loadSafetensors` path is equally lazy is **unverified**, and
   the design doesn't depend on it. If loading is lazy, per-layer `eval` + close peaks at ~1.125×.
   If it is eager, every source is already resident and the per-layer stack adds only one
   layer's worth on top before the sources are released. Either way the total stays near 1×,
   not 2×.
7. Approximate timing (M-series GPU, bf16, H=1024, F=3584, E=8, K=2). **Two measurements that
   must not be conflated:**
   - *Python probe, expert path only* (no router, no scopes): decode (T=1) dense ≈1.67 ms
     vs gathered ≈0.32 ms (≈5×); prefill (T=128) dense ≈5.51 ms vs unsorted gathered
     ≈5.47 ms vs sorted gathered ≈2.85 ms. This is the kernel-level ceiling.
   - *Java `MoeMlpBenchmarkTest`, whole `MoeMlp.forward`* (router softmax, K successive
     argmax/take/where, two `concatenate`s, three per-call `swapaxes` views, scope and
     `eval` overhead): decode ≈2.1 ms dense vs ≈1.0 ms gathered. Across eight runs the
     decode speedup ranged from **1.88× to 2.29×** (the first five: 1.94×, 1.98×, 2.16×,
     2.19×, 2.29×), and back-to-back runs 0.4 s apart can differ by ~0.2×. Prefill (T=128,
     256 slots, so the **sorted** path) ranged **1.64× to 2.38×**, sometimes well below
     decode.
   - The Java prefill number is not comparable with the probe's "unsorted ≈ dense" at
     T=128. `MoeMlp` switches to the sorted path at ≥ 64 slots, so Java never runs
     unsorted prefill at this size; the closest probe counterpart is the ≈1.9× sorted
     result.
   - Dense time is close in both (≈1.7-2.1 ms), so the gap is on the gathered side: Java
     adds ≈0.65 ms per call that the probe lacks. The cause is **not yet identified**. The
     candidates are the per-call `swapaxes` views, the routing ops (fixed cost per call, so
     they dominate once the expert work shrinks), and per-op FFM/scope/`eval` overhead. The
     benchmark asserts a floor below every observed Java run (Task 5), not the probe ratio.
     Profiling the gap is deferred.
8. **Half precision** (exact closed-form oracle evaluated on the dtype-rounded input; error metric
   `|actual - ref| / (|ref| + 1)`, |ref| up to ~9): worst **bf16 0.0076**, **f16 0.0011** across
   T ∈ {2, 40} and K ∈ {2, 3}. Gathered and dense agree bit-for-bit at K=2, but differ at K=3
   (bf16 0.031, f16 0.0039 absolute) because of the reduction order: dense adds one slot at a
   time in the model dtype, while gathered does one `sum` over the top-k axis. Task 3's bf16/f16
   value tests use bounds of **0.02 (bf16)** and **0.005 (f16)**, about 2.5-4.5× the observed
   worst case.

## Review Focus

Inputs and conditions the current tests don't cover that are most likely to bite users, most
likely first. Each line has a pinning test in the task named.

1. **Training/fine-tuning through a Mixtral MoE layer** — `ModuleGrad` over `MoeMlp` must not throw
   the GatherMM VJP error, must match the dense oracle's gradients, and must give exactly zero
   gradient to an unselected expert. It must hold on **both** paths: any real training step has
   ≥ 64 slots and takes the sorted path (Task 3, `gradientsFlowOnlyToSelectedExperts`, run at 3
   and 40 tokens).
2. **Prefill vs decode give the same logits** — prompts large enough to cross the sort threshold
   (`B*T*K >= 64`) must agree with the unsorted path and with the dense oracle (Task 2
   `sortedAndUnsortedPathsAgree`, Task 3 `matchesDenseOracleOnBothSidesOfSortThreshold`).
3. **bf16/f16 checkpoints stay in the model dtype and stay accurate** — stacking and `gather_mm`
   must not promote to float32, and values must stay within Verified facts 8's bounds of an exact
   closed form on both paths (Task 2 `bf16StaysBf16`; Task 3 `bf16OutputUsesHiddenStateDtype`,
   `halfPrecisionMatchesClosedFormOnBothPaths`).
4. **Loading a real Mixtral must not double resident weight memory** — per-expert source tensors
   are closed after stacking (Task 4 `expertSourceTensorsAreClosedAfterStacking`).
5. **A malformed expert tensor names the offending checkpoint key** rather than failing inside
   `stack` with a generic native shape error (Task 4 `mismatchedExpertShapeNamesTheTensorKey`).

## File structure

- Create `tools/mlx-oracle/probes/moe_gather_probe.py` — re-runnable evidence for every Verified
  fact (committed in Task 0).
- Modify `buildSrc/src/main/java/se/alipsa/jmlx/buildsrc/RepositorySources.java` and its test —
  exclude `.claude/worktrees` from the handwritten-source scan (Task 0).
- Modify `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java` — add `gatherMatmul`.
- Modify `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXShape.java` — add `stack` and a
  target-scope `swapaxes` overload.
- Create `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXGatherOpsTest.java`.
- Create `jmlx-core/src/main/java/se/alipsa/jmlx/nn/SwitchGlu.java` — stacked experts + gathered
  gate/up/down; sorted and unsorted paths.
- Create `jmlx-core/src/test/java/se/alipsa/jmlx/nn/SwitchGluTest.java`.
- Modify `jmlx-core/src/main/java/se/alipsa/jmlx/nn/MoeMlp.java` — router unchanged, expert loop
  replaced by `SwitchGlu`.
- Create `jmlx-core/src/test/java/se/alipsa/jmlx/nn/DenseMoeReference.java` — the current dense
  algorithm, moved into test code as the oracle.
- Modify `jmlx-core/src/test/java/se/alipsa/jmlx/nn/MoeMlpTest.java`.
- Modify `jmlx-models/src/main/java/se/alipsa/jmlx/models/DecoderAssembler.java` — stack expert
  tensors per layer, validate shapes by key, close the sources.
- Modify `jmlx-models/src/test/java/se/alipsa/jmlx/models/DecoderAssemblerTest.java` and
  `MixtralModelTest.java`.
- Create `jmlx-core/src/test/java/se/alipsa/jmlx/nn/MoeMlpBenchmarkTest.java` (opt-in) and wire
  `jmlx.benchmark` in `jmlx-core/build.gradle`.
- Modify `req/mlx-api-inventory-overrides.json` (and regenerate `req/mlx-api-inventory.md`),
  `jmlx-models/README.md`, `req/phase6-compatibility.md`, `req/plans/phase6-3-plan.md` (amend
  Scope decision 4 in place), `req/plans/phase6-3-probe-findings.md` (append the gather probe
  summary), and `CLAUDE.md` (architecture list).

---

### Task 0: Branch, commit the plan and probe, unblock `verifyMlxApiCallSites`

**Why the scan fix is here:** in the checkout this plan was written in, root `check` already
fails. `verifyMlxApiCallSites` walks the whole repository, and `RepositorySources.excluded` skips
only `.git`, `.gradle`, `.venv`, `native`, `build` and the generated-bindings directory. So it
descends into the nested git worktree `.claude/worktrees/worktree-phase4-m4` (detached at
`f6cb9bc`, a PR #11-era snapshot with its own copy of the generated bindings) and reports
dozens of `add an explicit req/mlx-api-inventory-overrides.json record` failures for files
under that path. Task 1 Step 4 and the acceptance gate both depend on this task.

Excluding the directory is the durable fix: any Claude Code worktree recreates the same
problem. Removing this particular worktree is optional cleanup. It has an untracked
`req/phase5-plan.md` that differs from the main checkout's copy, so don't run `git worktree
remove --force` without the user's say-so.

**Files:**
- Modify: `buildSrc/src/main/java/se/alipsa/jmlx/buildsrc/RepositorySources.java` (`excluded`)
- Modify: `buildSrc/src/test/java/se/alipsa/jmlx/buildsrc/MlxApiInventoryTest.java`
- Modify: `build.gradle` (the `generateMlxApiInventory` test-source `fileTree` excludes)
- Add: `req/plans/phase6-3-performance.md` (this file), `tools/mlx-oracle/probes/moe_gather_probe.py`

**Interfaces:**
- Produces: `RepositorySources.discover` skips any directory whose repository-relative path starts
  with `.claude/worktrees`.

- [ ] **Step 1: Create the branch from a base that has `c7613ad`**

```bash
git -C /Users/pernyf/project/jmlx merge-base --is-ancestor c7613ad phase6-3 && echo ok
git -C /Users/pernyf/project/jmlx switch -c phase6-3-moe-gather phase6-3
```
Expected: `ok`, then the new branch is checked out.

- [ ] **Step 2: Reproduce the scan failure**

Run: `./gradlew verifyMlxApiCallSites`
Expected: FAIL, with every listed path starting `.claude/worktrees/worktree-phase4-m4/`. (If
there's no such worktree, the task still pins the exclusion so the failure can't come back.)

- [ ] **Step 3: Write the failing buildSrc test**

Add next to `callSiteGuardDoesNotParseJavaSourcesInsideToolVirtualEnvironments` in
`MlxApiInventoryTest`:

```java
  @Test
  void callSiteGuardDoesNotScanNestedClaudeWorktrees() throws Exception {
    Path root = fixture("{\"records\":[]}");
    Path nested =
        root.resolve(".claude/worktrees/old/jmlx-core/src/main/java/sample/Stale.java");
    Files.createDirectories(nested.getParent());
    Files.writeString(
        nested,
        "import se.alipsa.jmlx.ffi.mlx_h; class Stale { void use() { mlx_h.mlx_add(); } }");

    assertDoesNotThrow(() -> MlxApiCallSites.verify(root));
  }
```

Run: `./gradlew -p buildSrc test --tests "se.alipsa.jmlx.buildsrc.MlxApiInventoryTest"`
Expected: the new test FAILS (unrecorded `mlx_add` call site under `.claude/worktrees`).

- [ ] **Step 4: Exclude the worktree directory**

In `RepositorySources.excluded`, add one condition:

```java
    return relative.startsWith(GENERATED_DIRECTORY)
        || relative.startsWith(Path.of(".claude", "worktrees"))
        || directoryName.equals(".git")
```

In root `build.gradle`, keep `generateMlxApiInventory`'s declared inputs consistent with the scan:

```groovy
        exclude '**/.git/**', '**/.gradle/**', '**/.venv/**', '**/build/**', '**/native/**',
                '.claude/worktrees/**'
```

- [ ] **Step 5: Verify**

Run:
```sh
./gradlew -p buildSrc spotlessApply
./gradlew -p buildSrc check
./gradlew verifyMlxApiCallSites generateMlxApiInventory
git diff --exit-code req/mlx-api-inventory.md
```
`spotlessApply` must run first. The test snippet above isn't google-java-format output (the
formatter re-joins the `nested` declaration), and `check` includes `spotlessCheck`.
Expected: buildSrc tests and style PASS; `verifyMlxApiCallSites` PASSES; the inventory is
unchanged (the worktree contributed nothing legitimate).

- [ ] **Step 6: Run the probe once and record its output**

```sh
./tools/mlx-oracle/install.sh
./tools/mlx-oracle/.venv/bin/python tools/mlx-oracle/probes/moe_gather_probe.py
```
Expected: `all correctness checks passed`, exit 0. Fact 3 must print `SAME result` and fact 5
must print `NO error`. If either differs, stop: the plan's reasoning for those facts no longer
holds (Task 2's review items and the "Explicitly deferred" bounds-check note depend on them).
Keep the output for Task 5 Step 4.

- [ ] **Step 7: Commit**

```bash
git add buildSrc/src/main/java/se/alipsa/jmlx/buildsrc/RepositorySources.java \
  buildSrc/src/test/java/se/alipsa/jmlx/buildsrc/MlxApiInventoryTest.java build.gradle \
  req/plans/phase6-3-performance.md tools/mlx-oracle/probes/moe_gather_probe.py
git commit -m "Add gathered-MoE plan and MLX probe; skip nested worktrees in call-site scan"
```

### Task 1: `gatherMatmul`, `stack` and target-scope `swapaxes` facades

**Files:**
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java` (next to `matmul`, ~line 78)
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXShape.java` (next to `swapaxes`
  ~line 137 and `concatenate` ~line 282)
- Modify: `req/mlx-api-inventory-overrides.json`, regenerate `req/mlx-api-inventory.md`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXGatherOpsTest.java` (new)

**Interfaces:**
- Consumes: `NativeOps.scopeOf`, `NativeOps.nullableHandle`, `NativeOps.checked`,
  `NativeOps.vectorInOp`, `NativeOps.axis2Op(String, MLXArray, MLXScope, int, int, Axis2Op)`.
- Produces:
  - `public static MLXArray MLXOps.gatherMatmul(MLXArray a, MLXArray b, MLXArray lhsIndices,
    MLXArray rhsIndices, boolean sortedIndices)`: `lhsIndices`/`rhsIndices` may be null; result
    goes to the innermost scope of all non-null operands.
  - `public static MLXArray MLXShape.stack(MLXArray[] arrays, int axis)`.
  - `public static MLXArray MLXShape.swapaxes(MLXArray a, MLXScope target, int axis1, int axis2)`.

- [ ] **Step 1: Write the failing test**

Create `MLXGatherOpsTest.java`. Every expected value below was computed with MLX 0.31.2.

```java
package se.alipsa.jmlx.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Native coverage for the gathered-matmul, stack and target-scope swapaxes facades. */
@EnabledIfNativeAvailable
class MLXGatherOpsTest {

  private static final float EPS = 1e-6f;

  /** Three 2x2 "experts": identity, 2 * identity, and a row swap. */
  private static MLXArray experts(MLXScope scope) {
    return MLX.array(
        scope, new float[] {1, 0, 0, 1, 2, 0, 0, 2, 0, 1, 1, 0}, new int[] {3, 2, 2});
  }

  @Test
  void rhsIndicesSelectOneMatrixPerBatchEntry() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 1, 2});
      MLXArray idx = MLX.array(scope, new int[] {2, 0}, new int[] {2});
      MLXArray r = MLXOps.gatherMatmul(a, experts(scope), null, idx, false);
      assertArrayEquals(new int[] {2, 1, 2}, r.shape());
      assertArrayEquals(new float[] {2, 1, 3, 4}, r.toFloatArray(), EPS);
    }
  }

  @Test
  void sortedIndicesFlagGivesSameResultForSortedInput() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 1, 2});
      MLXArray idx = MLX.array(scope, new int[] {0, 2}, new int[] {2});
      MLXArray r = MLXOps.gatherMatmul(a, experts(scope), null, idx, true);
      assertArrayEquals(new float[] {1, 2, 4, 3}, r.toFloatArray(), EPS);
    }
  }

  @Test
  void lhsIndicesSelectRowsOfA() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {2, 1, 2});
      MLXArray lhs = MLX.array(scope, new int[] {1, 1}, new int[] {2});
      MLXArray rhs = MLX.array(scope, new int[] {0, 2}, new int[] {2});
      MLXArray r = MLXOps.gatherMatmul(a, experts(scope), lhs, rhs, false);
      assertArrayEquals(new float[] {3, 4, 4, 3}, r.toFloatArray(), EPS);
    }
  }

  @Test
  void broadcastsTokenAxesAgainstTopKIndices() {
    try (MLXScope scope = new MLXScope()) {
      // x: [B=1, T=2, 1, 1, H=2]; idx: [B=1, T=2, K=2] -> [1, 2, 2, 1, 2]
      MLXArray x = MLX.array(scope, new float[] {1, 2, 3, 4}, new int[] {1, 2, 1, 1, 2});
      MLXArray idx = MLX.array(scope, new int[] {0, 2, 1, 0}, new int[] {1, 2, 2});
      MLXArray r = MLXOps.gatherMatmul(x, experts(scope), null, idx, false);
      assertArrayEquals(new int[] {1, 2, 2, 1, 2}, r.shape());
      assertArrayEquals(new float[] {1, 2, 2, 1, 6, 8, 3, 4}, r.toFloatArray(), EPS);
    }
  }

  @Test
  void bf16StaysBf16() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a =
          MLX.astype(MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 2}), DType.BFLOAT16);
      MLXArray b = MLX.astype(experts(scope), DType.BFLOAT16);
      MLXArray idx = MLX.array(scope, new int[] {1}, new int[] {1});
      assertEquals(DType.BFLOAT16, MLXOps.gatherMatmul(a, b, null, idx, false).dtype());
    }
  }

  @Test
  void floatIndicesAreRejectedAsMlxException() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray a = MLX.array(scope, new float[] {1, 2}, new int[] {1, 1, 2});
      MLXArray idx = MLX.array(scope, new float[] {0}, new int[] {1});
      assertThrows(
          MLXException.class, () -> MLXOps.gatherMatmul(a, experts(scope), null, idx, false));
    }
  }

  @Test
  void resultLandsInInnermostOperandScope() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray b = experts(model);
      MLXArray a = MLX.array(step, new float[] {1, 2}, new int[] {1, 1, 2});
      MLXArray idx = MLX.array(step, new int[] {0}, new int[] {1});
      assertEquals(step, MLXOps.gatherMatmul(a, b, null, idx, false).scope());
    }
  }

  @Test
  void stackInsertsANewAxis() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      MLXArray y = MLX.array(scope, new float[] {3, 4}, new int[] {2});
      MLXArray s = MLXShape.stack(new MLXArray[] {x, y}, 1);
      assertArrayEquals(new int[] {2, 2}, s.shape());
      assertArrayEquals(new float[] {1, 3, 2, 4}, s.toFloatArray(), EPS);
      assertArrayEquals(new int[] {2, 2}, MLXShape.stack(new MLXArray[] {x, y}, 0).shape());
    }
  }

  @Test
  void stackRejectsEmptyAndNullAndMismatchedShapes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray x = MLX.array(scope, new float[] {1, 2}, new int[] {2});
      MLXArray z = MLX.array(scope, new float[] {1, 2, 3}, new int[] {3});
      assertThrows(IllegalArgumentException.class, () -> MLXShape.stack(new MLXArray[0], 0));
      assertThrows(
          IllegalArgumentException.class, () -> MLXShape.stack(new MLXArray[] {x, null}, 0));
      assertThrows(MLXException.class, () -> MLXShape.stack(new MLXArray[] {x, z}, 0));
    }
  }

  @Test
  void swapaxesWithTargetAllocatesIntoTheChildScope() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray w = experts(model);
      MLXArray t = MLXShape.swapaxes(w, step, -1, -2);
      assertEquals(step, t.scope());
      assertArrayEquals(new float[] {1, 0, 0, 1, 2, 0, 0, 2, 0, 1, 1, 0}, t.toFloatArray(), EPS);
    }
  }
}
```

(The three expert matrices are symmetric, so the swapped values equal the originals. The test
checks scope placement, not the transpose itself; `MLXArrayTest` already covers `swapaxes` values.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.core.MLXGatherOpsTest"`
Expected: compilation FAILS — `gatherMatmul`, `stack` and `swapaxes(MLXArray, MLXScope, int, int)`
are undefined.

- [ ] **Step 3: Implement the facades**

In `MLXOps.java`, after `matmul` (add `import java.lang.foreign.Arena;` if absent):

```java
  /**
   * Batched matmul that picks, per batch entry, which matrix of {@code a} and/or {@code b} to use
   * ({@code mlx_gather_mm}). {@code lhsIndices}/{@code rhsIndices} index {@code a}'s/{@code b}'s
   * leading batch axes and may be null (meaning "use the natural batch position"). The index
   * arrays must be integral, and each index must be in range: the pinned native op does NOT
   * bounds-check index values, so an out-of-range index returns unspecified values instead of
   * failing. {@code sortedIndices} is a performance hint that the indices are non-decreasing; it
   * never changes the result on the pinned runtime, so no test can catch a wrong value. Get it
   * right at the call site. The result is allocated into the innermost scope of every non-null
   * operand.
   */
  public static MLXArray gatherMatmul(
      MLXArray a, MLXArray b, MLXArray lhsIndices, MLXArray rhsIndices, boolean sortedIndices) {
    Objects.requireNonNull(a, "gatherMatmul: a must not be null");
    Objects.requireNonNull(b, "gatherMatmul: b must not be null");
    MLXScope scope = NativeOps.scopeOf("gatherMatmul", a, b, lhsIndices, rhsIndices);
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment lhs = NativeOps.nullableHandle(lhsIndices, tmp);
      MemorySegment rhs = NativeOps.nullableHandle(rhsIndices, tmp);
      MemorySegment res = mlx_h.mlx_array_new(scope);
      NativeOps.checked(
          "gatherMatmul",
          () ->
              mlx_h.mlx_gather_mm(
                  res, a.handle(), b.handle(), lhs, rhs, sortedIndices, NativeOps.DEFAULT_STREAM));
      return new MLXArray(scope, res);
    }
  }
```

(Add `import java.util.Objects;` if `MLXOps` does not already import it.)

In `MLXShape.java`, after `swapaxes(MLXArray, int, int)`:

```java
  /**
   * Swaps two axes, allocating the result into {@code target} instead of {@code a.scope()} -- the
   * {@code swapaxes} counterpart of {@link #transpose(MLXArray, MLXScope)}, for a weight-derived
   * view computed inside {@code forward()} that must land in the step scope.
   */
  public static MLXArray swapaxes(MLXArray a, MLXScope target, int axis1, int axis2) {
    return NativeOps.axis2Op("swapaxes", a, target, axis1, axis2, mlx_h::mlx_swapaxes);
  }
```

After `concatenate`:

```java
  /** Stacks equal-shaped {@code arrays} along a new axis inserted at {@code axis}. */
  public static MLXArray stack(MLXArray[] arrays, int axis) {
    if (arrays.length == 0) {
      throw new IllegalArgumentException("stack: requires at least one array");
    }
    for (int i = 0; i < arrays.length; i++) {
      if (arrays[i] == null) {
        throw new IllegalArgumentException("stack: arrays[" + i + "] must not be null");
      }
    }
    return NativeOps.vectorInOp("stack", arrays, axis, mlx_h::mlx_stack_axis);
  }
```

In `req/mlx-api-inventory-overrides.json`, add a record inside `"records"` (put it next to the
existing `MLXShape`/`MLXOps` records):

```json
    {
      "bindings": ["mlx_h.mlx_gather_mm", "mlx_h.mlx_stack_axis"],
      "category": "downcall",
      "status": "implemented",
      "facadeOrReason": "MLXOps.gatherMatmul; MLXShape.stack",
      "tests": "MLXGatherOpsTest"
    },
```

- [ ] **Step 4: Run the tests and regenerate the inventory** (requires Task 0; without it,
  `verifyMlxApiCallSites` fails on the nested worktree regardless of this change)

Run:
```sh
./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.core.MLXGatherOpsTest"
./gradlew generateMlxApiInventory
./gradlew spotlessApply
./gradlew verifyMlxApiCallSites spotlessCheck :jmlx-core:checkstyleMain :jmlx-core:checkstyleTest
```
Expected: all 10 tests PASS; `req/mlx-api-inventory.md` shows `mlx_gather_mm` and
`mlx_stack_axis` as `implemented`; verification tasks pass.

- [ ] **Step 5: Commit**

```bash
git add jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java \
  jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXShape.java \
  jmlx-core/src/test/java/se/alipsa/jmlx/core/MLXGatherOpsTest.java \
  req/mlx-api-inventory-overrides.json req/mlx-api-inventory.md
git commit -m "Add gatherMatmul, stack and target-scope swapaxes facades"
```

### Task 2: `SwitchGlu` — stacked experts evaluated by gathered matmul

**Files:**
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/SwitchGlu.java`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/SwitchGluTest.java` (new)

**Interfaces:**
- Consumes (Task 1): `MLXOps.gatherMatmul(MLXArray, MLXArray, MLXArray, MLXArray, boolean)`,
  `MLXShape.swapaxes(MLXArray, MLXScope, int, int)`, `MLXShape.stack(MLXArray[], int)` (tests only).
  Existing: `MLXOps.stopGradient`, `MLXOps.argsortAxis` (returns INT32), `MLXShape.takeAxis`,
  `MLXShape.reshape`, `MLXShape.broadcastTo`, `MLX.arange`, `SiLU`, `GELU(scope, boolean)`.
- Produces:
  - `public final class SwitchGlu extends Module`
  - `public SwitchGlu(MLXScope scope, MLXArray gateWeight, MLXArray upWeight, MLXArray downWeight,
    Activation activation)`: gate/up `[E, F, H]`, down `[E, H, F]`; registers params
    `gateWeight`, `upWeight`, `downWeight`.
  - `public int experts()`
  - `public MLXArray forward(MLXArray x, MLXArray indices)`: `x` `[B, T, H]`, `indices` INT32
    `[B, T, K]` with every value in `[0, E)`; returns `[B, T, K, H]`, the per-slot expert outputs
    (not yet weighted or summed).
  - package-private `MLXArray forward(MLXArray x, MLXArray indices, boolean sort)` and
    `static final int SORT_THRESHOLD = 64`, so tests can force either path.

- [ ] **Step 1: Write the failing test**

The oracle is a plain `GatedMlp` per expert, built from the same per-expert weights that are
stacked for `SwitchGlu`, so the test never re-derives the math by hand.

```java
package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Gathered expert evaluation checked against one dense {@link GatedMlp} per expert. */
@EnabledIfNativeAvailable
class SwitchGluTest {

  private static final float EPS = 1e-4f;
  // Package-private: MoeMlpTest and MoeMlpBenchmarkTest reuse these fixtures.
  static final int E = 4;
  static final int H = 6;
  static final int F = 5;

  /** Deterministic, non-symmetric fill so a transposed weight cannot pass by accident. */
  static float[] pattern(int size, int salt) {
    float[] out = new float[size];
    for (int i = 0; i < size; i++) {
      out[i] = (float) (0.5 * Math.sin(0.37 * i + 1.3 * salt));
    }
    return out;
  }

  /** Per-expert weights: {@code [expert][0=gate,1=up,2=down]}. */
  static MLXArray[][] expertWeights(MLXScope scope) {
    MLXArray[][] w = new MLXArray[E][3];
    for (int e = 0; e < E; e++) {
      w[e][0] = MLX.array(scope, pattern(F * H, 10 * e + 1), new int[] {F, H});
      w[e][1] = MLX.array(scope, pattern(F * H, 10 * e + 2), new int[] {F, H});
      w[e][2] = MLX.array(scope, pattern(H * F, 10 * e + 3), new int[] {H, F});
    }
    return w;
  }

  static SwitchGlu switchGlu(MLXScope scope, MLXArray[][] w, Activation activation) {
    MLXArray[][] byKind = new MLXArray[3][E];
    for (int e = 0; e < E; e++) {
      for (int kind = 0; kind < 3; kind++) {
        byKind[kind][e] = w[e][kind];
      }
    }
    return new SwitchGlu(
        scope,
        MLXShape.stack(byKind[0], 0),
        MLXShape.stack(byKind[1], 0),
        MLXShape.stack(byKind[2], 0),
        activation);
  }

  static List<GatedMlp> denseExperts(MLXScope scope, MLXArray[][] w, Activation activation) {
    List<GatedMlp> out = new ArrayList<>();
    for (MLXArray[] e : w) {
      out.add(
          new GatedMlp(
              scope,
              new Linear(scope, e[0], null),
              new Linear(scope, e[1], null),
              new Linear(scope, e[2], null),
              activation));
    }
    return out;
  }

  /** Row-major {@code [B, T, K]} indices where slot k of token t picks {@code (t + 3k) % E}. */
  static int[] indices(int b, int t, int k) {
    int[] out = new int[b * t * k];
    for (int i = 0; i < b * t; i++) {
      for (int j = 0; j < k; j++) {
        out[i * k + j] = (i + 3 * j) % E;
      }
    }
    return out;
  }

  /** Expected {@code [B, T, K, H]} assembled from each dense expert's full-batch output. */
  static float[] expected(List<GatedMlp> dense, MLXArray x, int[] idx, int k) {
    float[][] perExpert = new float[E][];
    for (int e = 0; e < E; e++) {
      perExpert[e] = dense.get(e).forward(x).toFloatArray();
    }
    int tokens = idx.length / k;
    float[] out = new float[tokens * k * H];
    for (int token = 0; token < tokens; token++) {
      for (int slot = 0; slot < k; slot++) {
        int e = idx[token * k + slot];
        System.arraycopy(perExpert[e], token * H, out, (token * k + slot) * H, H);
      }
    }
    return out;
  }

  private static void assertMatchesDense(int b, int t, int k, boolean sort, Activation act) {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray[][] w = expertWeights(model);
      SwitchGlu glu = switchGlu(model, w, act);
      MLXArray x = MLX.array(step, pattern(b * t * H, 99), new int[] {b, t, H});
      int[] idx = indices(b, t, k);
      MLXArray actual = glu.forward(x, MLX.array(step, idx, new int[] {b, t, k}), sort);
      assertArrayEquals(new int[] {b, t, k, H}, actual.shape());
      assertArrayEquals(
          expected(denseExperts(model, w, act), x, idx, k), actual.toFloatArray(), EPS);
    }
  }

  @Test
  void unsortedPathMatchesDenseExperts() {
    assertMatchesDense(2, 3, 2, false, Activation.SILU);
  }

  @Test
  void sortedPathMatchesDenseExperts() {
    assertMatchesDense(2, 3, 2, true, Activation.SILU);
  }

  @Test
  void topKEqualToExpertCountMatchesDenseExperts() {
    assertMatchesDense(1, 5, E, false, Activation.SILU);
    assertMatchesDense(1, 5, E, true, Activation.SILU);
  }

  @Test
  void geluTanhActivationMatchesDenseExperts() {
    assertMatchesDense(1, 4, 2, false, Activation.GELU_TANH);
  }

  @Test
  void sortedAndUnsortedPathsAgreeAboveThreshold() {
    // 1 * 40 * 2 = 80 >= SORT_THRESHOLD: the public overload takes the sorted path.
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      SwitchGlu glu = switchGlu(model, expertWeights(model), Activation.SILU);
      MLXArray x = MLX.array(step, pattern(40 * H, 7), new int[] {1, 40, H});
      MLXArray idx = MLX.array(step, indices(1, 40, 2), new int[] {1, 40, 2});
      assertEquals(true, idx.size() >= SwitchGlu.SORT_THRESHOLD);
      assertArrayEquals(
          glu.forward(x, idx, false).toFloatArray(), glu.forward(x, idx).toFloatArray(), EPS);
    }
  }

  @Test
  void bf16StaysBf16() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[][] w = expertWeights(scope);
      for (MLXArray[] e : w) {
        for (int kind = 0; kind < 3; kind++) {
          e[kind] = MLX.astype(e[kind], DType.BFLOAT16);
        }
      }
      SwitchGlu glu = switchGlu(scope, w, Activation.SILU);
      MLXArray x =
          MLX.astype(MLX.array(scope, pattern(2 * H, 5), new int[] {1, 2, H}), DType.BFLOAT16);
      MLXArray idx = MLX.array(scope, indices(1, 2, 2), new int[] {1, 2, 2});
      assertEquals(DType.BFLOAT16, glu.forward(x, idx, false).dtype());
      assertEquals(DType.BFLOAT16, glu.forward(x, idx, true).dtype());
    }
  }

  @Test
  void registersStackedWeightsAsParameters() {
    try (MLXScope scope = new MLXScope()) {
      SwitchGlu glu = switchGlu(scope, expertWeights(scope), Activation.SILU);
      assertEquals(List.of("gateWeight", "upWeight", "downWeight"),
          List.copyOf(glu.parameters().keySet()));
      assertEquals(E, glu.experts());
    }
  }

  @Test
  void rejectsInconsistentWeightShapes() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray gate = MLX.zeros(scope, new int[] {E, F, H}, DType.FLOAT32);
      MLXArray down = MLX.zeros(scope, new int[] {E, H, F}, DType.FLOAT32);
      MLXArray wrongUp = MLX.zeros(scope, new int[] {E, F + 1, H}, DType.FLOAT32);
      MLXArray wrongDown = MLX.zeros(scope, new int[] {E, F, H}, DType.FLOAT32);
      MLXArray rank2 = MLX.zeros(scope, new int[] {F, H}, DType.FLOAT32);
      assertThrows(IllegalArgumentException.class,
          () -> new SwitchGlu(scope, gate, wrongUp, down, Activation.SILU));
      assertThrows(IllegalArgumentException.class,
          () -> new SwitchGlu(scope, gate, gate, wrongDown, Activation.SILU));
      assertThrows(IllegalArgumentException.class,
          () -> new SwitchGlu(scope, rank2, rank2, rank2, Activation.SILU));
    }
  }

  @Test
  void rejectsMalformedInputs() {
    try (MLXScope scope = new MLXScope()) {
      SwitchGlu glu = switchGlu(scope, expertWeights(scope), Activation.SILU);
      MLXArray x = MLX.zeros(scope, new int[] {1, 2, H}, DType.FLOAT32);
      MLXArray wrongHidden = MLX.zeros(scope, new int[] {1, 2, H + 1}, DType.FLOAT32);
      MLXArray idx = MLX.array(scope, new int[] {0, 1}, new int[] {1, 2, 1});
      MLXArray floatIdx = MLX.zeros(scope, new int[] {1, 2, 1}, DType.FLOAT32);
      MLXArray wrongTokens = MLX.array(scope, new int[] {0, 1, 2}, new int[] {1, 3, 1});
      assertThrows(IllegalArgumentException.class, () -> glu.forward(wrongHidden, idx));
      assertThrows(IllegalArgumentException.class, () -> glu.forward(x, floatIdx));
      assertThrows(IllegalArgumentException.class, () -> glu.forward(x, wrongTokens));
    }
  }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.nn.SwitchGluTest"`
Expected: compilation FAILS — `SwitchGlu` is undefined.

- [ ] **Step 3: Implement `SwitchGlu`**

```java
package se.alipsa.jmlx.nn;

import java.util.Arrays;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Stacked gated experts, {@code down(act(gate(x)) * up(x))} per expert, evaluated only for the
 * experts each token was routed to, using {@link MLXOps#gatherMatmul}. Weights are the Hugging
 * Face per-expert matrices stacked on a new leading axis: gate/up {@code [E, F, H]}, down {@code
 * [E, H, F]}. See req/plans/phase6-3-performance.md.
 */
public final class SwitchGlu extends Module {

  /**
   * Index count ({@code B * T * K}) at or above which slots are sorted by expert before the
   * gathered matmuls, so each expert's weights are read contiguously (mlx-lm's {@code SwitchGLU}
   * uses the same value). Below it the sort costs more than it saves.
   */
  static final int SORT_THRESHOLD = 64;

  private final UnaryModule activationLayer;
  private final int experts;
  private final int hidden;

  /** Registers the stacked projections as {@code gateWeight}, {@code upWeight}, {@code downWeight}. */
  public SwitchGlu(
      MLXScope scope,
      MLXArray gateWeight,
      MLXArray upWeight,
      MLXArray downWeight,
      Activation activation) {
    super(scope);
    int[] gate = rank3(gateWeight, "gateWeight");
    int[] up = rank3(upWeight, "upWeight");
    int[] down = rank3(downWeight, "downWeight");
    if (!Arrays.equals(gate, up)) {
      throw new IllegalArgumentException(
          "upWeight must have gateWeight's shape " + Arrays.toString(gate)
              + ", got " + Arrays.toString(up));
    }
    if (down[0] != gate[0] || down[1] != gate[2] || down[2] != gate[1]) {
      throw new IllegalArgumentException(
          "downWeight must have shape [E, H, F] = [" + gate[0] + ", " + gate[2] + ", " + gate[1]
              + "], got " + Arrays.toString(down));
    }
    param("gateWeight", gateWeight);
    param("upWeight", upWeight);
    param("downWeight", downWeight);
    experts = gate[0];
    hidden = gate[2];
    activationLayer =
        switch (Objects.requireNonNull(activation, "activation")) {
          case SILU -> new SiLU(scope);
          case GELU -> new GELU(scope);
          case GELU_TANH -> new GELU(scope, true);
        };
  }

  /** Number of stacked experts, {@code E}. */
  public int experts() {
    return experts;
  }

  /**
   * Returns the {@code [B, T, K, H]} outputs of the expert chosen for each of a token's {@code K}
   * slots, before routing weights are applied. {@code indices} must be INT32 {@code [B, T, K]}
   * with every value in {@code [0, experts())}: the native gather does not bounds-check, so an
   * out-of-range index returns unspecified values instead of failing.
   */
  public MLXArray forward(MLXArray x, MLXArray indices) {
    Objects.requireNonNull(indices, "indices");
    return forward(x, indices, indices.size() >= SORT_THRESHOLD);
  }

  MLXArray forward(MLXArray x, MLXArray indices, boolean sort) {
    Objects.requireNonNull(x, "x");
    Objects.requireNonNull(indices, "indices");
    int[] xs = x.shape();
    int[] is = indices.shape();
    if (xs.length != 3 || xs[2] != hidden) {
      throw new IllegalArgumentException(
          "x must have shape [batch, tokens, " + hidden + "], got " + Arrays.toString(xs));
    }
    if (is.length != 3 || is[0] != xs[0] || is[1] != xs[1] || indices.dtype() != DType.INT32) {
      throw new IllegalArgumentException(
          "indices must be INT32 [batch, tokens, topK] matching x, got "
              + indices.dtype() + " " + Arrays.toString(is));
    }
    int b = xs[0];
    int t = xs[1];
    int k = is[2];
    MLXScope s = x.scope();
    // gather_mm has no VJP with respect to its indices; routing is not differentiable anyway.
    MLXArray idx = MLXOps.stopGradient(indices);
    // Weight views go into the caller's scope, never the model scope (req/phase4-plan.md §2).
    MLXArray gateT = MLXShape.swapaxes(param("gateWeight"), s, -1, -2);
    MLXArray upT = MLXShape.swapaxes(param("upWeight"), s, -1, -2);
    MLXArray downT = MLXShape.swapaxes(param("downWeight"), s, -1, -2);
    if (!sort) {
      MLXArray rows = MLXShape.reshape(x, new int[] {b, t, 1, 1, hidden});
      MLXArray h = glu(rows, gateT, upT, idx, false);
      MLXArray y = MLXOps.gatherMatmul(h, downT, null, idx, false);
      return MLXShape.reshape(y, new int[] {b, t, k, hidden});
    }
    int n = b * t;
    int m = n * k;
    MLXArray flat = MLXShape.reshape(idx, new int[] {m});
    MLXArray order = MLXOps.argsortAxis(flat, 0);
    MLXArray inverse = MLXOps.argsortAxis(order, 0);
    MLXArray tokenOfSlot =
        MLXShape.reshape(
            MLXShape.broadcastTo(
                MLXShape.reshape(MLX.arange(s, 0, n, 1, DType.INT32), new int[] {n, 1}),
                new int[] {n, k}),
            new int[] {m});
    MLXArray rows =
        MLXShape.takeAxis(
            MLXShape.reshape(x, new int[] {n, 1, hidden}),
            MLXShape.takeAxis(tokenOfSlot, order, 0),
            0);
    MLXArray sortedIdx = MLXShape.takeAxis(flat, order, 0);
    MLXArray h = glu(rows, gateT, upT, sortedIdx, true);
    MLXArray y = MLXOps.gatherMatmul(h, downT, null, sortedIdx, true);
    return MLXShape.reshape(MLXShape.takeAxis(y, inverse, 0), new int[] {b, t, k, hidden});
  }

  private MLXArray glu(
      MLXArray rows, MLXArray gateT, MLXArray upT, MLXArray idx, boolean sortedIndices) {
    MLXArray gate = MLXOps.gatherMatmul(rows, gateT, null, idx, sortedIndices);
    MLXArray up = MLXOps.gatherMatmul(rows, upT, null, idx, sortedIndices);
    return MLXOps.multiply(activationLayer.forward(gate), up);
  }

  private static int[] rank3(MLXArray weight, String name) {
    Objects.requireNonNull(weight, name);
    if (weight.ndim() != 3) {
      throw new IllegalArgumentException(
          name + " must be rank 3, got shape " + Arrays.toString(weight.shape()));
    }
    return weight.shape();
  }
}
```

**Review items no test can catch** (per Verified facts 3 and the memory rule). Reviewers must
check both by reading the code:
- `sortedIndices` is `true` only on the path that actually sorted (`sortedIdx`), never on `idx`.
- All three `swapaxes` calls pass `s` (`x.scope()`), not the weights' own scope.

- [ ] **Step 4: Run the tests**

Run: `./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.nn.SwitchGluTest" && ./gradlew spotlessApply :jmlx-core:checkstyleMain :jmlx-core:checkstyleTest`
Expected: all 9 tests PASS; formatting and checkstyle clean (`spotlessApply` rewraps the
test's long `assertThrows` lines).

- [ ] **Step 5: Commit**

```bash
git add jmlx-core/src/main/java/se/alipsa/jmlx/nn/SwitchGlu.java \
  jmlx-core/src/test/java/se/alipsa/jmlx/nn/SwitchGluTest.java
git commit -m "Add SwitchGlu gathered-expert module"
```

### Task 3: `MoeMlp` routes through `SwitchGlu`; the dense algorithm becomes the test oracle

`MoeMlp` is new on the `phase6-3` branch (not on `main`, never released), so changing its
constructor is not a published API break. Parameter paths change from `expert{i}.gateProj.weight`
to `experts.gateWeight` (stacked). No production code reads these paths; checkpoint loading is by
Hugging Face key in `DecoderAssembler` (Task 4).

**Files:**
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/MoeMlp.java`
- Create: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/DenseMoeReference.java`
- Modify: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/MoeMlpTest.java`

**Interfaces:**
- Consumes (Task 2): `SwitchGlu(MLXScope, MLXArray, MLXArray, MLXArray, Activation)`,
  `SwitchGlu.experts()`, `SwitchGlu.forward(MLXArray x, MLXArray indices)`, and the test helpers
  `SwitchGluTest.pattern`, `expertWeights`, `switchGlu`, `denseExperts`, `E`, `H` (package-private
  static, same package).
- Produces:
  - `public MoeMlp(MLXScope scope, UnaryLayer router, SwitchGlu experts, int topK)`: children
    `router` and `experts`.
  - `MoeMlp.forward(MLXArray x)`: unchanged contract, `[B, T, H]` → `[B, T, H]` in `x.dtype()`.
  - Test-only `final class DenseMoeReference extends UnaryLayer`, with
    `DenseMoeReference(MLXScope, UnaryLayer router, List<GatedMlp> experts, int topK)`:
    today's `MoeMlp` code moved verbatim, used as the oracle.

- [ ] **Step 1: Move the current implementation into test code as the oracle**

Copy today's `MoeMlp.java` to `jmlx-core/src/test/java/se/alipsa/jmlx/nn/DenseMoeReference.java`.
Rename the class and constructor to `DenseMoeReference`, make the class package-private
(`final class DenseMoeReference extends UnaryLayer`), and replace the class javadoc with:

```java
/**
 * Test oracle: the Phase 6.3 dense mixture of experts, which runs every expert on every token and
 * masks unselected outputs with {@code where}. Kept verbatim from the pre-gather {@code MoeMlp}
 * so the gathered implementation is checked against an independent, simpler algorithm (routing,
 * tie-breaking and renormalisation must agree exactly).
 */
```

Change nothing else, so the oracle stays the reviewed Phase 6.3 algorithm.

- [ ] **Step 2: Rewrite `MoeMlpTest` against the new constructor (failing)**

Replace `MoeMlpTest.java` with the following. Expert biases are gone: Mixtral experts have
none (`ArchitectureMappings` rejects `mlp_bias` for Mixtral), and `SwitchGlu` has no bias.
Expected values come from `DenseMoeReference` built from the same arrays, instead of from
"constant" bias-driven experts.

```java
package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Gathered MoE routing checked against the dense Phase 6.3 oracle. */
@EnabledIfNativeAvailable
class MoeMlpTest {

  private static final float EPS = 1e-4f;
  private static final int E = SwitchGluTest.E;
  private static final int H = SwitchGluTest.H;

  /** Zero-weight router whose logits are exactly {@code biases} for every token. */
  private static Linear biasRouter(MLXScope scope, float... biases) {
    return new Linear(
        scope,
        MLX.zeros(scope, new int[] {biases.length, H}, DType.FLOAT32),
        MLX.array(scope, biases, new int[] {biases.length}));
  }

  /** Input-dependent router, so different tokens pick different experts. */
  private static Linear patternRouter(MLXScope scope) {
    return new Linear(scope, MLX.array(scope, SwitchGluTest.pattern(E * H, 77), new int[] {E, H}),
        null);
  }

  private static void assertMatchesOracle(int tokens, int topK) {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(model);
      MoeMlp moe =
          new MoeMlp(model, patternRouter(model), SwitchGluTest.switchGlu(model, w,
              Activation.SILU), topK);
      DenseMoeReference oracle =
          new DenseMoeReference(model, patternRouter(model),
              SwitchGluTest.denseExperts(model, w, Activation.SILU), topK);
      MLXArray x =
          MLX.array(step, SwitchGluTest.pattern(tokens * H, 3), new int[] {1, tokens, H});
      assertArrayEquals(oracle.forward(x).toFloatArray(), moe.forward(x).toFloatArray(), EPS);
    }
  }

  @Test
  void rejectsInvalidTopK() {
    try (MLXScope scope = new MLXScope()) {
      SwitchGlu experts =
          SwitchGluTest.switchGlu(scope, SwitchGluTest.expertWeights(scope), Activation.SILU);
      assertThrows(IllegalArgumentException.class,
          () -> new MoeMlp(scope, biasRouter(scope, 0, 0, 0, 0), experts, 0));
      assertThrows(IllegalArgumentException.class,
          () -> new MoeMlp(scope, biasRouter(scope, 0, 0, 0, 0), experts, E + 1));
    }
  }

  @Test
  void matchesDenseOracleOnBothSidesOfSortThreshold() {
    assertMatchesOracle(3, 2); // 6 slots: unsorted path
    assertMatchesOracle(40, 2); // 80 slots >= SwitchGlu.SORT_THRESHOLD: sorted path
  }

  @Test
  void selectingEveryExpertMatchesDenseOracle() {
    assertMatchesOracle(5, E);
    assertMatchesOracle(40, E);
  }

  @Test
  void exactTiesChooseFirstExpertAcrossTokensAndRuns() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(model);
      MoeMlp moe = new MoeMlp(model, biasRouter(model, 0, 0, 0, 0),
          SwitchGluTest.switchGlu(model, w, Activation.SILU), 1);
      GatedMlp first = SwitchGluTest.denseExperts(model, w, Activation.SILU).get(0);
      MLXArray x = MLX.array(step, SwitchGluTest.pattern(2 * 6 * H, 4), new int[] {2, 6, H});
      float[] expected = first.forward(x).toFloatArray();
      for (int run = 0; run < 100; run++) {
        assertArrayEquals(expected, moe.forward(x).toFloatArray(), EPS);
      }
    }
  }

  @Test
  void unselectedInfiniteExpertNeverAffectsOutput() {
    try (MLXScope model = new MLXScope();
        MLXScope step = model.newChild()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(model);
      w[E - 1][2] = MLX.full(model, new int[] {H, SwitchGluTest.F}, Float.MAX_VALUE,
          DType.FLOAT32);
      MoeMlp moe = new MoeMlp(model, biasRouter(model, 2, 1, 0, -1),
          SwitchGluTest.switchGlu(model, w, Activation.SILU), 2);
      DenseMoeReference oracle = new DenseMoeReference(model, biasRouter(model, 2, 1, 0, -1),
          SwitchGluTest.denseExperts(model, w, Activation.SILU), 2);
      MLXArray x = MLX.array(step, SwitchGluTest.pattern(3 * H, 8), new int[] {1, 3, H});
      float[] actual = moe.forward(x).toFloatArray();
      for (float v : actual) {
        assertTrue(Float.isFinite(v));
      }
      assertArrayEquals(oracle.forward(x).toFloatArray(), actual, EPS);
    }
  }

  @Test
  void bf16OutputUsesHiddenStateDtype() {
    try (MLXScope scope = new MLXScope()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(scope);
      for (MLXArray[] e : w) {
        for (int kind = 0; kind < 3; kind++) {
          e[kind] = MLX.astype(e[kind], DType.BFLOAT16);
        }
      }
      Linear router = new Linear(scope,
          MLX.astype(MLX.zeros(scope, new int[] {E, H}, DType.FLOAT32), DType.BFLOAT16),
          MLX.astype(MLX.array(scope, new float[] {1, 0, 0, 0}, new int[] {E}), DType.BFLOAT16));
      MoeMlp moe = new MoeMlp(scope, router, SwitchGluTest.switchGlu(scope, w, Activation.SILU),
          2);
      MLXArray x = MLX.astype(MLX.array(scope, SwitchGluTest.pattern(H, 1), new int[] {1, 1, H}),
          DType.BFLOAT16);
      assertEquals(DType.BFLOAT16, moe.forward(x).dtype());
    }
  }

  @Test
  void registersRouterAndStackedExpertParameters() {
    try (MLXScope scope = new MLXScope()) {
      MoeMlp moe = new MoeMlp(scope, biasRouter(scope, 0, 0, 0, 0),
          SwitchGluTest.switchGlu(scope, SwitchGluTest.expertWeights(scope), Activation.SILU), 2);
      assertEquals(
          List.of("router.weight", "router.bias", "experts.gateWeight", "experts.upWeight",
              "experts.downWeight"),
          List.copyOf(moe.parameters().keySet()));
    }
  }

  /**
   * Review Focus 1: {@code ModuleGrad} through the gathered route must not hit GatherMM's
   * "Cannot calculate VJP with respect to indices" error, must give exactly zero gradient to
   * never-selected experts, and must equal the dense oracle's per-expert gradients -- on the
   * unsorted path (3 tokens x top-2 = 6 slots) AND the sorted path (40 x 2 = 80 slots >=
   * SORT_THRESHOLD), since every realistic training step takes the sorted one.
   */
  @Test
  void gradientsFlowOnlyToSelectedExperts() {
    assertGradientsMatchOracle(3);
    assertGradientsMatchOracle(40);
  }

  private static void assertGradientsMatchOracle(int tokens) {
    try (MLXScope model = new MLXScope()) {
      MLXArray[][] w = SwitchGluTest.expertWeights(model);
      // Biases 2, 1, 0, -1 with top-2: experts 0 and 1 always win; 2 and 3 are never selected.
      MoeMlp moe = new MoeMlp(model, biasRouter(model, 2, 1, 0, -1),
          SwitchGluTest.switchGlu(model, w, Activation.SILU), 2);
      DenseMoeReference oracle = new DenseMoeReference(model, biasRouter(model, 2, 1, 0, -1),
          SwitchGluTest.denseExperts(model, w, Activation.SILU), 2);
      try (ModuleGrad gathered =
              ModuleGrad.of(moe, (p, in) -> new MLXArray[] {MLXOps.sum(moe.forward(in[0]))});
          ModuleGrad dense = ModuleGrad.of(
              oracle, (p, in) -> new MLXArray[] {MLXOps.sum(oracle.forward(in[0]))});
          MLXScope step = model.newChild()) {
        MLXArray x =
            MLX.array(step, SwitchGluTest.pattern(tokens * H, 6), new int[] {1, tokens, H});
        MLXArray stacked = gathered.apply(step, new MLXArray[] {x}).grads()
            .get("experts.gateWeight");
        var denseGrads = dense.apply(step, new MLXArray[] {x}).grads();
        int f = SwitchGluTest.F;
        for (int e = 0; e < E; e++) {
          float[] slice = MLXShape.slice(stacked, new int[] {e, 0, 0}, new int[] {e + 1, f, H})
              .toFloatArray();
          assertArrayEquals(
              denseGrads.get("expert" + e + ".gateProj.weight").toFloatArray(), slice, EPS);
          if (e >= 2) {
            assertArrayEquals(new float[f * H], slice, 0f);
          }
        }
      }
    }
  }

  // ---- Absolute anchors: exact closed forms, independent of DenseMoeReference -------------

  /** Router picks logits from the first 3 input coordinates: {@code logits = x[..., :3]}. */
  private static Linear selectorRouter(MLXScope scope, DType dtype) {
    float[] w = new float[3 * 4];
    for (int e = 0; e < 3; e++) {
      w[e * 4 + e] = 1;
    }
    return new Linear(scope, MLX.astype(MLX.array(scope, w, new int[] {3, 4}), dtype), null);
  }

  /** 3 experts, H=F=4: gate = up = identity, down = (e + 1) * identity. */
  private static SwitchGlu diagonalExperts(MLXScope scope, DType dtype) {
    MLXArray[] gate = new MLXArray[3];
    MLXArray[] down = new MLXArray[3];
    for (int e = 0; e < 3; e++) {
      float[] identity = new float[16];
      float[] scaled = new float[16];
      for (int i = 0; i < 4; i++) {
        identity[i * 4 + i] = 1;
        scaled[i * 4 + i] = e + 1;
      }
      gate[e] = MLX.astype(MLX.array(scope, identity, new int[] {4, 4}), dtype);
      down[e] = MLX.astype(MLX.array(scope, scaled, new int[] {4, 4}), dtype);
    }
    MLXArray g = MLXShape.stack(gate, 0);
    return new SwitchGlu(scope, g, g, MLXShape.stack(down, 0), Activation.SILU);
  }

  /**
   * Exact expected output: per token, softmax over {@code x[:3]}, successive-argmax top-k
   * (lowest index on ties), renormalised weights w_j, and output
   * {@code sum_j w_j * (j + 1) * silu(x) * x} elementwise. Computed in double from {@code input}
   * as given, so half-precision callers pass the dtype-rounded input.
   */
  private static float[] closedForm(float[] input, int topK) {
    float[] out = new float[input.length];
    for (int token = 0; token < input.length / 4; token++) {
      double[] p = new double[3];
      double max = Double.NEGATIVE_INFINITY;
      for (int e = 0; e < 3; e++) {
        max = Math.max(max, input[token * 4 + e]);
      }
      double z = 0;
      for (int e = 0; e < 3; e++) {
        p[e] = Math.exp(input[token * 4 + e] - max);
        z += p[e];
      }
      boolean[] taken = new boolean[3];
      double selected = 0;
      double weighted = 0;
      for (int k = 0; k < topK; k++) {
        int best = -1;
        for (int e = 0; e < 3; e++) {
          if (!taken[e] && (best < 0 || p[e] > p[best])) {
            best = e;
          }
        }
        taken[best] = true;
        selected += p[best];
        weighted += (best + 1) * p[best];
      }
      double scale = weighted / selected;
      for (int h = 0; h < 4; h++) {
        double v = input[token * 4 + h];
        out[token * 4 + h] = (float) (scale * v * v / (1 + Math.exp(-v)));
      }
    }
    return out;
  }

  /** Inputs in [-2, 2] so routing varies per token. */
  private static float[] anchorInput(int tokens) {
    float[] x = new float[tokens * 4];
    for (int i = 0; i < x.length; i++) {
      x[i] = (float) (2 * Math.sin(0.37 * i + 3.9));
    }
    return x;
  }

  /** Replaces the dropped Phase 6.3 closed-form tests; float32, both paths, top-k 2 and 3. */
  @Test
  void float32MatchesClosedFormOnBothPaths() {
    for (int tokens : new int[] {2, 40}) {
      for (int topK : new int[] {2, 3}) {
        try (MLXScope scope = new MLXScope()) {
          MoeMlp moe = new MoeMlp(scope, selectorRouter(scope, DType.FLOAT32),
              diagonalExperts(scope, DType.FLOAT32), topK);
          float[] input = anchorInput(tokens);
          float[] actual =
              moe.forward(MLX.array(scope, input, new int[] {1, tokens, 4})).toFloatArray();
          assertArrayEquals(closedForm(input, topK), actual, 1e-5f,
              "tokens=" + tokens + " topK=" + topK);
        }
      }
    }
  }

  /**
   * Review Focus 3: bf16/f16 values, not just dtype. Bounds are Verified facts 8's (probe worst
   * case bf16 0.0076, f16 0.0011 on |a - r| / (|r| + 1)). The reference is evaluated on the
   * dtype-rounded input, so input quantisation is not counted as model error.
   */
  @Test
  void halfPrecisionMatchesClosedFormOnBothPaths() {
    for (DType dtype : new DType[] {DType.BFLOAT16, DType.FLOAT16}) {
      double bound = dtype == DType.BFLOAT16 ? 0.02 : 0.005;
      for (int tokens : new int[] {2, 40}) {
        for (int topK : new int[] {2, 3}) {
          try (MLXScope scope = new MLXScope()) {
            MoeMlp moe = new MoeMlp(scope, selectorRouter(scope, dtype),
                diagonalExperts(scope, dtype), topK);
            MLXArray x = MLX.astype(
                MLX.array(scope, anchorInput(tokens), new int[] {1, tokens, 4}), dtype);
            float[] rounded = x.toFloatArray();
            MLXArray y = moe.forward(x);
            assertEquals(dtype, y.dtype());
            float[] actual = y.toFloatArray();
            float[] expected = closedForm(rounded, topK);
            for (int i = 0; i < actual.length; i++) {
              double err = Math.abs(actual[i] - expected[i]) / (Math.abs(expected[i]) + 1);
              assertTrue(err <= bound, dtype + " tokens=" + tokens + " topK=" + topK
                  + " index " + i + ": " + actual[i] + " vs " + expected[i]);
            }
          }
        }
      }
    }
  }
}
```

The two closed-form tests restore what `selectingEveryExpertEqualsSoftmaxWeightedSum` and
`denseResultMatchesPerTokenReferenceWithHiddenFour` anchored: absolute values that don't depend
on `DenseMoeReference`. They now also cover the sorted path (40 × 2 = 80 and 40 × 3 = 120 slots)
and half precision. `topK = 3` with three experts is the old "select every expert" case.
`toFloatArray()` on a bf16/f16 array converts to float32 on read (it accepts any inexact
dtype), so `rounded` is the exact dtype-rounded input.

`SwitchGluTest`'s `E`, `H`, `F`, `pattern`, `expertWeights`, `switchGlu` and `denseExperts` are
package-private (Task 2), so this test uses them directly.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.nn.MoeMlpTest"`
Expected: compilation FAILS — no `MoeMlp(MLXScope, UnaryLayer, SwitchGlu, int)` constructor.

- [ ] **Step 4: Rewrite `MoeMlp`**

Keep the class javadoc's routing description, and replace "Dense mixture" with "Mixture of
gated experts with deterministic top-k routing; only the selected experts are evaluated (see
{@link SwitchGlu})". Fields and constructor:

```java
  private final UnaryLayer router;
  private final SwitchGlu experts;
  private final int topK;

  /** Registers the router and the stacked experts as children {@code router} and {@code experts}. */
  public MoeMlp(MLXScope scope, UnaryLayer router, SwitchGlu experts, int topK) {
    super(scope);
    this.router = child("router", Objects.requireNonNull(router, "router"));
    this.experts = child("experts", Objects.requireNonNull(experts, "experts"));
    if (topK < 1 || topK > experts.experts()) {
      throw new IllegalArgumentException("topK must be between 1 and the number of experts");
    }
    this.topK = topK;
  }
```

In `forward`, keep everything from the input-shape check through the loop that fills
`indices[i]`/`selectedProbabilities[i]`/`selectedSum` and the `weights[i]` cast, replacing
`experts.size()` with `experts.experts()` in the router-shape check and in `arange`. Delete
`result`, `zero` and the per-expert loop, and replace them with:

```java
    // [B, T, K]: argmaxAxis(keepdims=true) yields INT32 [B, T, 1] per slot.
    MLXArray selected = MLXShape.concatenate(indices, -1);
    MLXArray slotWeights = MLXShape.concatenate(weights, -1);
    MLXArray expertOutputs = experts.forward(x, selected); // [B, T, K, H]
    return MLXOps.sum(
        MLXOps.multiply(expertOutputs, MLXShape.expandDims(slotWeights, -1)),
        new int[] {2},
        false);
```

Remove the now-unused `java.util.List` import. Also drop the "where runs before multiplication"
comment: unselected experts are no longer computed, so an infinite expert output cannot reach
the sum.

- [ ] **Step 5: Run the MoE and probe tests**

Run: `./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.nn.MoeMlpTest" --tests "se.alipsa.jmlx.nn.MoeRoutingProbeTest" --tests "se.alipsa.jmlx.nn.SwitchGluTest" && ./gradlew spotlessApply :jmlx-core:checkstyleMain :jmlx-core:checkstyleTest`
Expected: PASS. `jmlx-models` does not compile yet (`DecoderAssembler` still builds
`List<GatedMlp>`); Task 4 fixes that. Don't run the root `build` until Task 4 is done.

- [ ] **Step 6: Do not commit yet**

This change breaks `jmlx-models` compilation until Task 4 updates `DecoderAssembler`, so it must
not be committed alone. Leave the files uncommitted; Task 4 Step 5 commits Tasks 3 and 4
together as one commit, so every commit on the branch builds.
A subagent-driven reviewer gates Task 3 on the working-tree diff (`git diff`) and the
`:jmlx-core` test run above.

### Task 4: Stack Mixtral expert tensors at load time

**Files:**
- Modify: `jmlx-models/src/main/java/se/alipsa/jmlx/models/DecoderAssembler.java` (`moeMlp`,
  ~lines 115-136; `assemble` javadoc; imports)
- Modify: `jmlx-models/src/test/java/se/alipsa/jmlx/models/DecoderAssemblerTest.java`

**Interfaces:**
- Consumes (Tasks 1-3): `MLXShape.stack(MLXArray[], int)`, `SwitchGlu(MLXScope, MLXArray,
  MLXArray, MLXArray, Activation)`, `MoeMlp(MLXScope, UnaryLayer, SwitchGlu, int)`. Existing:
  `DecoderAssembler.tensor`, `requireShape`, `projection`, `MLX.eval`, `MLXArray.close()`.
- Produces: `DecoderAssembler.assemble(...)` returns the same `Assembled`. **Behaviour change:**
  the per-expert `block_sparse_moe.experts.{e}.w{1,2,3}.weight` arrays in `tensors` are closed
  after being stacked (see Verified facts 6). The javadoc says so.

- [ ] **Step 1: Write the failing tests**

Add to `DecoderAssemblerTest` (the class already imports `MLX`, `MLXArray`, `MLXIO`, `DType`,
`MLXScope`, `ObjectMapper`, `assertThrows`, `assertTrue`; add `java.util.LinkedHashMap` if the
existing fully-qualified use is replaced):

```java
  private static final Path MIXTRAL =
      Path.of(System.getProperty("jmlx.repository.root"),
          "tools", "hf-reference", "goldens", "checkpoints", "mixtral");

  private static Map<String, MLXArray> mixtralTensors(MLXScope scope) {
    return new java.util.LinkedHashMap<>(
        MLXIO.loadSafetensors(scope, MIXTRAL.resolve("model.safetensors").toString()).tensors());
  }

  private static ArchitectureDescriptor mixtralDescriptor() throws IOException {
    return ArchitectureMappings.parse(
        new ObjectMapper().readTree(MIXTRAL.resolve("config.json").toFile()));
  }

  @Test
  @EnabledIfNativeAvailable
  void expertSourceTensorsAreClosedAfterStacking() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors = mixtralTensors(scope);
      MLXArray expert = tensors.get("model.layers.0.block_sparse_moe.experts.0.w1.weight");
      MLXArray router = tensors.get("model.layers.0.block_sparse_moe.gate.weight");
      DecoderAssembler.assemble(scope, mixtralDescriptor(), tensors);
      assertThrows(IllegalStateException.class, expert::shape);
      router.shape(); // non-expert tensors stay open: they are the live parameters
    }
  }

  @Test
  @EnabledIfNativeAvailable
  void mismatchedExpertShapeNamesTheTensorKey() throws Exception {
    String key = "model.layers.1.block_sparse_moe.experts.2.w2.weight";
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> tensors = mixtralTensors(scope);
      tensors.put(key, MLX.zeros(scope, new int[] {3, 3}, DType.FLOAT32));
      String message =
          assertThrows(
                  IllegalArgumentException.class,
                  () -> DecoderAssembler.assemble(scope, mixtralDescriptor(), tensors))
              .getMessage();
      assertTrue(message.contains(key), message);
    }
  }
```

`MixtralModelTest.prefillAndDecodeMatchHuggingFace` needs no change. It is the end-to-end
numeric gate against the Hugging Face tiny-checkpoint logits at `1e-4`, but only in **float32**
and only on the **unsorted** path (6-token prompt × top-2 = 12 slots; decode steps are 2 slots).
Don't claim it covers the sorted path. That path is anchored by Task 3's closed-form tests. A
sorted-path HF golden would need a prompt of ≥ 32 tokens (the fixture allows 128 positions)
and a regenerated reference via `tools/hf-reference`; that is deferred, since this plan
regenerates no goldens.

- [ ] **Step 2: Run the tests to verify they fail**

Start from Task 3's uncommitted working tree.

Run: `./gradlew :jmlx-models:test --tests "se.alipsa.jmlx.models.DecoderAssemblerTest"`
Expected: compilation FAILS in `DecoderAssembler.moeMlp` (`MoeMlp` no longer accepts
`List<GatedMlp>`).

- [ ] **Step 3: Implement stacking in `DecoderAssembler`**

Replace `moeMlp` with the following, and add `stackExperts` beside it:

```java
  private static UnaryLayer moeMlp(
      MLXScope scope,
      ArchitectureDescriptor descriptor,
      Map<String, MLXArray> tensors,
      String prefix) {
    int experts = descriptor.moe().experts();
    int hidden = descriptor.dimensions().hiddenSize();
    int intermediate = descriptor.dimensions().intermediateSize();
    // Validate all three kinds before stacking any, so a bad tensor fails before a source closes.
    for (String kind : new String[] {"w1", "w3", "w2"}) {
      boolean down = kind.equals("w2");
      for (int e = 0; e < experts; e++) {
        String key = prefix + "experts." + e + "." + kind + ".weight";
        requireShape(
            tensor(tensors, key), key, down ? hidden : intermediate, down ? intermediate : hidden);
      }
    }
    SwitchGlu stacked =
        new SwitchGlu(
            scope,
            stackExperts(tensors, prefix, "w1", experts),
            stackExperts(tensors, prefix, "w3", experts),
            stackExperts(tensors, prefix, "w2", experts),
            descriptor.mlp().activation());
    return new MoeMlp(
        scope, projection(scope, tensors, prefix + "gate", false), stacked,
        descriptor.moe().topK());
  }

  /**
   * Stacks one projection kind of every expert into {@code [E, rows, columns]}, materialises it,
   * then closes the per-expert sources. Evaluating per layer before closing keeps peak weight
   * memory near 1x instead of 2x (req/plans/phase6-3-performance.md, Verified facts 6).
   */
  private static MLXArray stackExperts(
      Map<String, MLXArray> tensors, String prefix, String kind, int experts) {
    MLXArray[] parts = new MLXArray[experts];
    for (int e = 0; e < experts; e++) {
      parts[e] = tensor(tensors, prefix + "experts." + e + "." + kind + ".weight");
    }
    MLXArray stacked = MLXShape.stack(parts, 0);
    MLX.eval(stacked);
    for (MLXArray part : parts) {
      part.close();
    }
    return stacked;
  }
```

Imports: add `se.alipsa.jmlx.nn.SwitchGlu`; remove `se.alipsa.jmlx.nn.GatedMlp` if no other use
remains (the dense-MLP path still uses it, so it probably stays). `MLX` and `MLXShape` are
already imported.

Add this sentence to `assemble`'s javadoc: "For mixture-of-experts layers, the per-expert
`block_sparse_moe.experts.*` arrays in {@code tensors} are stacked and then closed, so the caller
must not use them afterwards."

- [ ] **Step 4: Run the model tests and the checks**

Run:
```sh
./gradlew :jmlx-models:test --tests "se.alipsa.jmlx.models.DecoderAssemblerTest" \
  --tests "se.alipsa.jmlx.models.MixtralModelTest" \
  --tests "se.alipsa.jmlx.models.Phase63ModelTokenizerContractTest"
./gradlew spotlessApply
./gradlew :check :jmlx-core:check :jmlx-models:check
```
Expected: PASS, including the unchanged Mixtral golden at `1e-4` and root `:check` (inventory,
call sites), plus checkstyle and Spotless for both modules. Gate on `check`, not `build`:
`./gradlew build` currently fails in `:jmlx-tokenizer:javadoc` for a pre-existing reason
unrelated to this plan (see Task 5 Step 5).

- [ ] **Step 5: Commit Tasks 3 and 4 together**

```bash
git add jmlx-core/src/main/java/se/alipsa/jmlx/nn/MoeMlp.java \
  jmlx-core/src/test/java/se/alipsa/jmlx/nn/DenseMoeReference.java \
  jmlx-core/src/test/java/se/alipsa/jmlx/nn/MoeMlpTest.java \
  jmlx-models/src/main/java/se/alipsa/jmlx/models/DecoderAssembler.java \
  jmlx-models/src/test/java/se/alipsa/jmlx/models/DecoderAssemblerTest.java
git commit -m "Route MoE through gathered experts stacked at load time

MoeMlp evaluates only the selected experts via SwitchGlu; the dense
Phase 6.3 algorithm survives as the DenseMoeReference test oracle.
DecoderAssembler stacks Mixtral expert weights per layer and releases
the per-expert source tensors."
```

### Task 5: Opt-in benchmark, memory regression guard, and documentation

**Files:**
- Create: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/MoeMlpBenchmarkTest.java`
- Modify: `jmlx-core/build.gradle` (forward `jmlx.benchmark` to tests)
- Modify: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/SwitchGluTest.java` (leak guard)
- Modify: `jmlx-models/README.md`, `req/phase6-compatibility.md`, `req/plans/phase6-3-plan.md`,
  `CLAUDE.md`

**Interfaces:**
- Consumes: `MoeMlp`, `SwitchGlu`, `DenseMoeReference`, and the `SwitchGluTest` helpers (Tasks
  2-3); `NativeMemoryProbe.activeMemoryBytes()` (jmlx-ffi test fixture, already on jmlx-core's
  test classpath).
- Produces: no production API.

- [ ] **Step 1: Add the per-step leak guard (native, always on)**

Add this test to `SwitchGluTest`. It follows `LinearTest`'s leak-test structure: 50 warmup
iterations, 200 measured, 2 MB threshold. Each iteration runs in its own child scope under a
long-lived model scope. **Limitation (verified on MLX 0.31.2):** a retained `swapaxes` view
shares its source buffer, so active memory does not detect a view leaked into the model scope.
This test only guards against data-owning intermediates (gather outputs, stacked copies)
accumulating. The `x.scope()` rule for the views stays a code-review item (Task 2).

```java
  @Test
  void activeMemoryDoesNotGrowAcrossPerStepScopes() {
    try (MLXScope model = new MLXScope()) {
      SwitchGlu glu = switchGlu(model, expertWeights(model), Activation.SILU);
      Runnable step = () -> {
        try (MLXScope s = model.newChild()) {
          MLXArray x = MLX.array(s, pattern(40 * H, 2), new int[] {1, 40, H});
          MLXArray idx = MLX.array(s, indices(1, 40, 2), new int[] {1, 40, 2});
          MLX.eval(glu.forward(x, idx));
        }
      };
      for (int i = 0; i < 50; i++) {
        step.run();
      }
      long baseline = se.alipsa.jmlx.ffi.NativeMemoryProbe.activeMemoryBytes();
      for (int i = 0; i < 200; i++) {
        step.run();
      }
      long growth = se.alipsa.jmlx.ffi.NativeMemoryProbe.activeMemoryBytes() - baseline;
      org.junit.jupiter.api.Assertions.assertTrue(growth <= 2_000_000, "grew " + growth + " B");
    }
  }
```

- [ ] **Step 2: Add the opt-in benchmark**

In `jmlx-core/build.gradle`, append:

```groovy
tasks.withType(Test).configureEach {
    // Opt-in performance tests: ./gradlew :jmlx-core:test -Djmlx.benchmark=true --tests "*Benchmark*"
    systemProperty 'jmlx.benchmark', System.getProperty('jmlx.benchmark', 'false')
}
```

Create `MoeMlpBenchmarkTest.java`. It uses Mixtral-proportioned but scaled-down shapes (H=1024,
F=3584, E=8, K=2, bf16; about 176 MB of expert weights), so it runs on any Apple Silicon Mac.
It times the whole `MoeMlp.forward`, routing included, which is what a model pays. At decode
that is ≈2× in Java (1.88× to 2.29× across eight runs, Verified facts 7), **not** the probe's
expert-only ≈5×. The assertion floor is therefore **1.5×**. It still catches a regression back
to dense compute (≈1×), and it sits below every observed run. The measured ratio is always
reported, so drift shows up before it trips the floor. Prefill (T=128, sorted path) is reported,
not asserted: it ranged from 1.64× to 2.38× and sometimes fell well below decode, so any fixed
prefill floor would either be meaningless or flaky.

```java
package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestReporter;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXRandom;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Opt-in: gathered vs dense MoE wall time. Enable with {@code -Djmlx.benchmark=true}. */
@EnabledIfNativeAvailable
@EnabledIfSystemProperty(named = "jmlx.benchmark", matches = "true")
class MoeMlpBenchmarkTest {

  private static final int H = 1024;
  private static final int F = 3584;
  private static final int E = 8;
  private static final int K = 2;

  @Test
  void gatheredDecodeIsFasterThanDense(TestReporter reporter) {
    try (MLXScope model = new MLXScope()) {
      MLXArray[] gate = new MLXArray[E];
      MLXArray[] up = new MLXArray[E];
      MLXArray[] down = new MLXArray[E];
      java.util.List<GatedMlp> dense = new java.util.ArrayList<>();
      for (int e = 0; e < E; e++) {
        gate[e] = MLXRandom.normal(model, new int[] {F, H}, DType.BFLOAT16, 0, 0.02f);
        up[e] = MLXRandom.normal(model, new int[] {F, H}, DType.BFLOAT16, 0, 0.02f);
        down[e] = MLXRandom.normal(model, new int[] {H, F}, DType.BFLOAT16, 0, 0.02f);
        dense.add(new GatedMlp(model, new Linear(model, gate[e], null),
            new Linear(model, up[e], null), new Linear(model, down[e], null), Activation.SILU));
      }
      // One weight array, two Linear instances: Module.child forbids registering one module
      // instance under two parents (Module.java, child() javadoc: undefined results). Sharing
      // the underlying MLXArray is fine; Linear only reads it.
      MLXArray routerWeight = MLXRandom.normal(model, new int[] {E, H}, DType.BFLOAT16, 0, 0.02f);
      MoeMlp gathered = new MoeMlp(model, new Linear(model, routerWeight, null), new SwitchGlu(
          model, MLXShape.stack(gate, 0), MLXShape.stack(up, 0), MLXShape.stack(down, 0),
          Activation.SILU), K);
      DenseMoeReference reference =
          new DenseMoeReference(model, new Linear(model, routerWeight, null), dense, K);
      MLX.eval(gathered.parameters().values().toArray(MLXArray[]::new));

      for (int tokens : new int[] {1, 128}) {
        double g = millis(model, gathered, tokens);
        double d = millis(model, reference, tokens);
        reporter.publishEntry("T=" + tokens,
            String.format("dense %.2f ms, gathered %.2f ms, speedup %.2fx", d, g, d / g));
        if (tokens == 1) {
          // Floor, not target: Java measures ~2.1x for the whole forward (plan, Verified facts 7).
          assertTrue(d / g >= 1.5, "decode speedup " + d / g + "x < 1.5x");
        }
      }
    }
  }

  private static double millis(MLXScope model, UnaryLayer moe, int tokens) {
    for (int i = 0; i < 5; i++) {
      run(model, moe, tokens);
    }
    int n = 30;
    long start = System.nanoTime();
    for (int i = 0; i < n; i++) {
      run(model, moe, tokens);
    }
    return (System.nanoTime() - start) / 1e6 / n;
  }

  private static void run(MLXScope model, UnaryLayer moe, int tokens) {
    try (MLXScope step = model.newChild()) {
      MLXArray x = MLXRandom.normal(step, new int[] {1, tokens, H}, DType.BFLOAT16, 0, 1);
      MLX.eval(moe.forward(x));
    }
  }
}
```

(Each module gets its own router `Linear` over one shared weight array, the same way
`MoeMlpTest` builds its router twice. Sharing the `MLXArray` is safe; sharing the `Linear`
instance is not, per `Module.child`'s javadoc. The per-expert `GatedMlp`s and the stacked
`SwitchGlu` likewise share weight arrays but not module instances.)

- [ ] **Step 3: Run both tests**

Run:
```sh
./gradlew :jmlx-core:test --tests "se.alipsa.jmlx.nn.SwitchGluTest"
./gradlew :jmlx-core:test -Djmlx.benchmark=true --rerun \
  --tests "se.alipsa.jmlx.nn.MoeMlpBenchmarkTest"
grep -o 'value="[^"]*speedup[^"]*"' \
  jmlx-core/build/test-results/test/TEST-se.alipsa.jmlx.nn.MoeMlpBenchmarkTest.xml
```
Expected: the leak guard PASSES; the benchmark PASSES (decode speedup ≥ 1.5×, typically
≈2×). Two details:
- `--rerun` is needed because Gradle otherwise serves a cached result without running it. Test
  inputs don't change between runs, and `jmlx.benchmark` is only a system property.
- The timings are **not** printed to the console, even with `-i`. `TestReporter.publishEntry`
  writes to the JUnit XML report, so read the `T=1` and `T=128` entries from
  `jmlx-core/build/test-results/test/TEST-se.alipsa.jmlx.nn.MoeMlpBenchmarkTest.xml` (the
  `grep` above).

Without `-Djmlx.benchmark=true` the benchmark is reported as skipped.

- [ ] **Step 4: Update the docs**

- `jmlx-models/README.md`: the sentence wraps across lines 18-19, so an exact single-line match
  won't find it. Edit with the wrapped text as it stands in the file:

  ```text
  use grows with generation length until Phase 6.4. Mixtral computes every expert densely and masks
  unselected outputs.
  ```

  and replace it with:

  ```text
  use grows with generation length until Phase 6.4. Mixtral evaluates only the
  `num_experts_per_tok` experts selected per token (gathered matmul over expert weights stacked
  at load time).
  ```

  Check with `grep -n 'densely' jmlx-models/README.md` (expect no output).
- `req/phase6-compatibility.md`, Mixtral row: replace "dense MoE compute" with "gathered top-k
  expert compute".
- `req/plans/phase6-3-plan.md`: amend Scope decision 4 in place (don't rewrite it). Append:
  "**Superseded by `req/plans/phase6-3-performance.md`:** `MoeMlp` now routes through
  `SwitchGlu`/`mlx_gather_mm`; the dense algorithm survives as the test oracle
  `DenseMoeReference`." Update the line in "Explicitly deferred" that mentions dense MoE the
  same way.
- `CLAUDE.md`, Architecture block: add `SwitchGlu` to the `se.alipsa.jmlx.nn` list after
  `MultiHeadAttention, KVCache`.
- `req/plans/phase6-3-probe-findings.md`: append this section. The numbers below are from the
  probe's first run; replace them with Task 0 Step 6's output if they differ:

  ```markdown
  ## Gathered MoE (`mlx_gather_mm`) probe

  Status: **recorded 2026-09-30 on mlx 0.31.2** (the pinned `mlx-metal` wheel). Re-run with
  `./tools/mlx-oracle/.venv/bin/python tools/mlx-oracle/probes/moe_gather_probe.py` after any
  native pin change; it exits non-zero on a correctness regression.

  - Correctness (PASS): per-token equivalence; sorted == unsorted forward and backward for
    top-2 and top-k == E; float indices rejected; gradient through router-derived indices
    raises `[GatherMM] Cannot calculate VJP with respect to indices.` unless the indices go
    through `stop_gradient`, after which grads equal the dense implementation exactly.
  - **Not testable from Java (recorded, not asserted):** `sorted_indices=True` on unsorted
    indices gives the SAME result (the flag is only a hint); out-of-range indices raise NO error.
    If either flips on a future runtime, revisit `SwitchGlu`'s review items and bounds checks.
  - Half precision vs exact closed form, `|a - r| / (|r| + 1)`: bf16 worst 0.0076, f16 worst
    0.0011. Gathered and dense differ at top-3 (bf16 0.031 absolute) because of reduction order.
  - Load memory: per-layer stack + eval + close sources = 1.125x weights (peak and steady);
    keeping sources = 2.0x.
  - Approximate timing (bf16, H=1024 F=3584 E=8 K=2; varies by machine and run):
    - Python probe, expert path only: T=1 dense ≈1.67 ms vs gathered ≈0.32 ms; T=128 dense
      ≈5.51 ms, gathered ≈5.47 ms, gathered-sorted ≈2.85 ms.
    - Java `MoeMlpBenchmarkTest`, whole `MoeMlp.forward` including routing: T=1 ≈2.1 ms dense
      vs ≈1.0 ms gathered, 1.88-2.29× across eight runs. T=128 (256 slots, sorted path)
      1.64-2.38×, sometimes below decode. The probe's "unsorted ≈ dense at T=128" does not
      describe Java prefill, which always sorts at this size. The ≈0.65 ms/call gathered-side
      overhead against the probe is not yet explained; the decode floor is 1.5×, and prefill
      is reported only.
  ```

- [ ] **Step 5: Full verification and commit**

Run:
```sh
./gradlew spotlessApply
./gradlew :check :jmlx-core:check :jmlx-models:check
```
Expected: PASS. **Gate on `check`, not `build`.** In the current checkout `./gradlew build`
fails in `:jmlx-tokenizer:javadoc`: Javadoc `-Werror` reports missing `@param` tags on
`TokenizerEncoding`, a record last touched in `21816ae`. This is a pre-existing failure,
independent of this plan, which changes nothing in `jmlx-tokenizer`. `check` doesn't run
Javadoc, and root `:check` still covers the inventory and call-site guards. Also run
`./gradlew :jmlx-core:javadoc :jmlx-models:javadoc`, since this plan adds public javadoc in
those modules. Fixing the tokenizer Javadoc belongs in a separate PR; if it has already
landed when this plan runs, use `./gradlew spotlessApply build` instead.

```bash
git add jmlx-core/build.gradle \
  jmlx-core/src/test/java/se/alipsa/jmlx/nn/MoeMlpBenchmarkTest.java \
  jmlx-core/src/test/java/se/alipsa/jmlx/nn/SwitchGluTest.java \
  jmlx-models/README.md req/phase6-compatibility.md req/plans/phase6-3-plan.md \
  req/plans/phase6-3-probe-findings.md CLAUDE.md
git commit -m "Add MoE benchmark and leak guard; document gathered expert compute"
```

## Verification matrix

| Requirement | Evidence |
| --- | --- |
| Only selected experts computed | `SwitchGlu` code path (`gatherMatmul` only); benchmark decode speedup ≥ 1.5× (≈2× typical, whole forward) |
| Routing semantics unchanged | `MoeMlpTest` vs `DenseMoeReference` (ties, top-k = E, infinite expert); `MoeRoutingProbeTest` unchanged |
| Float32 end to end (unsorted path only) | `MixtralModelTest` HF golden at `1e-4`, goldens not regenerated |
| Absolute values, independent of the dense oracle | `MoeMlpTest.float32MatchesClosedFormOnBothPaths` (sorted + unsorted, top-k 2 and 3) |
| Sorted path correct | `SwitchGluTest.sortedPathMatchesDenseExperts`, `sortedAndUnsortedPathsAgreeAboveThreshold`; `MoeMlpTest.matchesDenseOracleOnBothSidesOfSortThreshold`, `float32MatchesClosedFormOnBothPaths` |
| Autograd works on both paths | `MoeMlpTest.gradientsFlowOnlyToSelectedExperts` (3 and 40 tokens) |
| Half precision: dtype and values | `MLXGatherOpsTest.bf16StaysBf16`, `SwitchGluTest.bf16StaysBf16`, `MoeMlpTest.bf16OutputUsesHiddenStateDtype`, `halfPrecisionMatchesClosedFormOnBothPaths` (bf16 ≤ 0.02, f16 ≤ 0.005) |
| Runtime facts no Java test can see (3, 5) | `tools/mlx-oracle/probes/moe_gather_probe.py`, summarised in `phase6-3-probe-findings.md` |
| Load memory ≈ 1× | `DecoderAssemblerTest.expertSourceTensorsAreClosedAfterStacking` |
| Clear load errors | `DecoderAssemblerTest.mismatchedExpertShapeNamesTheTensorKey` |
| Binding inventory | `generateMlxApiInventory` clean, `verifyMlxApiCallSites`, root `check` |

## Acceptance gate

- `./gradlew :check :jmlx-core:check :jmlx-models:check`, `./gradlew :jmlx-core:javadoc
  :jmlx-models:javadoc` and `./gradlew -p buildSrc check` pass on a bootstrapped macOS
  checkout, and on an unbootstrapped one (native tests skipped). This holds even when a
  `.claude/worktrees/*` checkout is present (Task 0).
- Full `./gradlew build` is **not** a gate while the pre-existing `:jmlx-tokenizer:javadoc`
  failure (`TokenizerEncoding` missing `@param`, unrelated to this plan) is unfixed. Note it in
  the PR description.
- `moe_gather_probe.py` exits 0 with fact 3 `SAME result` and fact 5 `NO error`.
- The opt-in benchmark passes its 1.5× decode floor on the implementer's machine. Paste the
  `T=1` and `T=128` entries from its XML report
  (`jmlx-core/build/test-results/test/TEST-se.alipsa.jmlx.nn.MoeMlpBenchmarkTest.xml`) into the
  PR description.
- No committed golden, fixture or `provenance.json` changes.

## Explicitly deferred

- **Quantized experts (`mlx_gather_qmm`)**: `ArchitectureMappings` still rejects quantized
  checkpoints. Wire `gather_qmm` into `SwitchGlu` when quantized loading lands.
- **Shared-expert MoE** (e.g. Qwen2-MoE, DeepSeek) and expert-parallel execution.
- **Closing the Java-vs-probe decode gap** (≈1.0 ms vs ≈0.32 ms gathered, Verified facts 7):
  profile which of the `swapaxes` views, the routing ops, or per-op FFM/scope/`eval` overhead
  accounts for it (for example, by timing `SwitchGlu.forward` alone against `MoeMlp.forward`)
  before optimizing. Raise the benchmark floor only when the Java distribution supports it.
- **Tuning `SORT_THRESHOLD`**: kept at mlx-lm's 64; the benchmark prints prefill numbers so a
  later change can be justified with data.
- **A sorted-path Hugging Face golden**: needs a ≥ 32-token Mixtral prompt and a regenerated
  `tools/hf-reference` reference. Until then the sorted path is anchored by the closed-form and
  dense-oracle tests, not by HF.
- **Bounds-checking router indices**: unnecessary while indices come only from the router's
  `argmax`; revisit if `SwitchGlu.forward(x, indices)` gets external callers.
