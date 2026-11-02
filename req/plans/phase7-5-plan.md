# Phase 7.5 — MLX paired-pin upgrade

- **Status:** delivered 2026-11-01 (PR #71; follow-up PR #72). The issue #73 timing
  investigation was recorded in PR #76, merged 2026-11-02 after Java/native CI passed;
  its remaining measurement limits belong to Phase 8.2, not an incomplete pin upgrade.
- **Parent:** `req/plans/phase7-plan.md` §7.5 (the reusable checklist) and `req/full-roadmap.md`
  § "Release and MLX upgrade plan".
- **Prerequisites:** milestone 7.3c and milestone 7.4 are each closed, delivered or deferred with a
  date. The Xcode Command Line Tools and `cmake` are installed locally; the hash-locked oracle
  environments (`tools/mlx-oracle`, `tools/qwen-vl-reference`) are installable; a `macos-26`
  GitHub-hosted runner is available (it is the only host that satisfies the oracle's GPU profile, see
  §6 and WP2).
- **Form:** one dedicated native-pin PR, never bundled with feature work. This document is also the
  checklist that the later upgrades (milestone 0 of Phases 9–15) reuse and update.

## 1. Goal, non-goals, prerequisites, and exit gate

**Goal.** Upgrade the paired native pins from `mlx-metal` 0.31.2 / mlx-c
`fba4470b89073180056c9ea46c443051375f7399` (tracks MLX v0.31.2, mlx-c v0.6.0) to the newest pairing
that `scripts/checkDependencies.zsh` reports at execution time. On **2026-11-01** that pairing is:

- **`mlx-metal` 0.32.4** — `mlx_metal-0.32.4-py3-none-macosx_26_0_arm64.whl`,
  SHA-256 `514df92d41a286895d2564b4b18f4da45096bd7a4a0c47e1042f345ad5a7f936`,
  `https://files.pythonhosted.org/packages/2a/d6/4340ed6e184f71693d7f531c9ae55d677451351245760d71f8dd714dcbe5/…`
- **mlx-c v0.8.0 = `a26f2d584a14ec619ae8db9a09be03becd47598a`** (its `CMakeLists.txt` pins MLX
  `v0.32.4`), the exact-commit pairing the bootstrap's version-pairing assertion requires.
- **Python frontend** for the oracle: `mlx-0.32.4-cp312-cp312-macosx_26_0_arm64.whl`,
  SHA-256 `7928869076db6267ab3ebdb253b2c80be1e738b3032c783170bd1d59d460c578`,
  `https://files.pythonhosted.org/packages/af/44/90bbdc3cdd120e1181f7c4f617ff996530159f2ea3ba3c08b83b08314d21/…`

These hash values are recorded so a reviewer can re-verify them without re-deriving them. **On the
day the upgrade is executed, re-run `./scripts/checkDependencies.zsh` first; if it reports a newer
mlx-metal and its paired mlx-c commit, that newer pairing wins and the pairing plus its wheel hashes
are recorded in the PR description before any edit.** The upgrade runs after 7.3c/7.4 and before any
7.6 code, so 7.6's oracle fixtures, goldens and Tier-B evidence are produced once, on the new
runtime, and the first native release (Phase 8) ships that runtime.

**Non-goals.**

- No **new** public Java API and no **new** call site for any mlx-c symbol. Two existing call sites
  are **mechanically retargeted** because v0.8.0 changes the signatures of the functions jmlx already
  calls (§4 WP1): `MLXOps.cumulativeSumAxis` moves from `mlx_cumsum` to `mlx_cumsum_axis` (the new
  no-axis form of `mlx_cumsum` drops `axis` and adds an `mlx_optional_dtype`; passing the absent
  dtype reproduces the old input-inferred dtype), and `MLXFast.scaledDotProductAttention` gains
  `force_fused=false`. Both retargets preserve current public behaviour: v0.31.2's
  `scaled_dot_product_attention` had no `force_fused` parameter, and MLX v0.32.4's `mlx/fast.h:60`
  declares `bool force_fused = false`, so `force_fused=false` reproduces the previous default call.
  The residual risk — an upstream change *inside* the default SDPA path between the pins — is caught
  by the WP3 goldens, not by this retarget.
- No version bump, and no publication — Phase 7 does not release.
- No loosening of any strict `full-float32` bound.
- No re-measuring of design bounds (they are invariants, not measurements — §6).
- No rewriting of dated historical evidence (probe findings, `req/data/phase6-4-1-baseline/*.json`,
  `req/benchmarks/`, dated report sections, the 7.3b/7.3c/7.4 evidence paragraphs and their pinned
  0.31.2 measurements). Only "current-state" references (pin tables, status lines, version links)
  are updated; new measurements are recorded next to the old ones.
- No feature work in the PR.

**Exit gate (from §7.5 and the roadmap procedure).** The PR regenerates the bindings, retargets the
two signature-changed call sites, diffs the inventory, regenerates the Python MLX oracle fixtures,
re-checks golden precision, re-runs the Tier-B suites, re-records the evidence jmlx recorded on MLX
(seeded sampling streams and the pin-gated Tier-B IDs), and adds a migration note covering every
skipped upstream release. Because this upgrade skips upstream releases 0.31.3, 0.31.4 … 0.32.3 and
mlx-c v0.6.0→v0.8.0, the inventory diff is taken against the previous pin (0.31.2 / `fba4470`) and
the migration note covers the upstream changes of every skipped release. The full checklist is §8.

## 2. Exact files changed (source of truth, as of 2026-11-01)

**Public Java API, call sites, and mlx-c declarations.** No **new** public Java API and no **new**
call site. Exactly **two existing call sites are retargeted** to the v0.8.0 signatures (behaviour
preserved): `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java:439` (`mlx_cumsum` →
`mlx_cumsum_axis` + absent `mlx_optional_dtype`) and
`jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXFast.java:199` (SDPA gains `force_fused=false`).
The generated bindings under `jmlx-ffi/src/main/generated/java` and the two API reports change
mechanically. The v0.8.0 inventory diff is **not** purely additive: it also **removes and renames**
symbols (listed in §3) — some of which jmlx does not call. Grouped below; "review" means the diff is
inspected in the PR, not blindly accepted.

**Pin, native, and the two retargets (WP1).**
- `scripts/bootstrap-native.sh` — `MLX_METAL_VERSION` (line 28), `MLX_METAL_WHEEL_URL` (line 29),
  `MLX_METAL_WHEEL_SHA256` (line 30), `MLX_C_COMMIT` (line 33), and the Decision-9 comment
  (lines 24–27). The three `MLX_METAL_*` lines are edited by hand; `MLX_C_COMMIT` is set by
  `scripts/updateMlx.zsh`. `native/install/lib/native-pin.properties` is regenerated by bootstrap
  (not hand-edited); it is the runtime-pin source for Tier-B tests and the oracle environment check.
- `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java` (line 439) and
  `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXFast.java` (line 199) — the two call-site
  retargets.
- `req/mlx-api-inventory-overrides.json` (line 160) — the `MLXOps` entry's `mlx_h.mlx_cumsum` is
  renamed to `mlx_h.mlx_cumsum_axis` (the SDPA entry at line 116 is unchanged: the symbol name is
  the same, only its descriptor gains a parameter).
- `jmlx-ffi/src/main/generated/java/**` — regenerated jextract output. Review the diff (usually
  small); it must remain byte-identical to `scripts/regen-bindings.sh` output.
- `req/mlx-api-inventory.md` and `req/mlx-api-header-coverage.md` — regenerated by
  `generateMlxApiInventory` / `generateMlxApiHeaderCoverage`. Review.

**Python MLX oracle (WP2).**
- `tools/mlx-oracle/provenance.json` — `mlx` {version, url, sha256}, `mlxMetal` {version, url,
  sha256}, `mlxCCommit`.
- `tools/mlx-oracle/requirements.lock` — the hash-locked `mlx==…` and `mlx-metal==…` frontend/backend
  wheel lines plus their hashes.
- `tools/mlx-oracle/fixtures/phase6-1-sampling.expected.json`,
  `phase6-tier-a-array.expected.json` (GPU family) and `phase7-1-core.expected.json` (CPU family) —
  regenerated (see the host constraint in §6 / WP2).
- `.github/workflows/ci.yml` — a **temporary** `generateMlxOracleFixtures` step added to the
  existing `macos-26` native job, removed in the final review round (WP2.4).

**jmlx-recorded goldens (WP4).**
- `jmlx-models/src/test/java/se/alipsa/jmlx/models/LlamaModelTest.java` — `EXPECTED_SEED_42` /
  `EXPECTED_SEED_7` (lines 39–42) **only if** the regenerated `phase6-1-sampling` oracle passes.
- `tools/release-smoke/goldens/mistral-sampled.properties` — rewritten by `run.sh ci --record`; it
  self-carries the `native.mlxMetalVersion` / `native.mlxcCommit` pin lines.
- `tools/tier-b/{lite-mistral-150m-v2-instruct, qwen2.5-0.5b-instruct, qwen3-0.6b,
  smollm2-135m-instruct}.json` — `recorded_pin`, `expected_token_ids`,
  `observed_peak_test_jvm_rss_kib`, `observed_run`.

**Converter environment (WP5).**
- `tools/qwen-vl-reference/requirements.in`, `tools/qwen-vl-reference/requirements.lock`,
  `tools/qwen-vl-reference/provenance.json` — `runtime.mlx` / `runtime.mlx_metal` and
  `requirements_lock_sha256` move to 0.32.4; `requirements.in` pins `mlx==0.32.4`.

**Tests whose memory bounds are measurements (WP6, only if an exceedance is documented).**
- `jmlx-models/src/tierB/java/se/alipsa/jmlx/models/TierBQwen3VlTest.java`,
  `TierBVisionLanguageTest.java` and `TierBPhi3VTest.java` — each with
  `NATIVE_PEAK_BOUND_BYTES` / `FOOTPRINT_BOUND_BYTES` (Phi-3.5-V's footprint bound,
  52,000,000,000 B, is the tightest: ~552 MB above the measured 51.42 GB).
- `jmlx-models/src/test/java/se/alipsa/jmlx/models/VisionLanguageMemoryTest.java`
  (`FIXED_MARGIN`); and `Qwen3VlMemoryTest.java` / `Phi3VMemoryTest.java` (same `src/test` set) only
  if the allocator-tier sweep (WP6) shows MLX's 16/32/64 KiB doubling tiers changed (their
  `CLEANUP_MARGIN` of 512 B stays untouched).
- `tools/hf-reference/provenance.json` — **only** the digests of manifests whose `local_evidence`
  is re-measured (see §6).

**Log-line change (WP4).**
- `jmlx-models/src/tierB/java/se/alipsa/jmlx/models/TierBReferences.java` — the log line
  (currently prints only `recorded_pin`, ~line 128) gains the running runtime pin, so it cannot be
  read as describing the new runtime.

**Documentation — "current-state" references only (WP7).**
- `AGENTS.md` — the "Current pin:" line (≈548–549). (The pinned-version line in the 7.3c WP8
  evidence paragraph, ≈947–948, is a dated 7.3c measurement and is **not** rewritten.)
- `README.md` (≈line 68), `jmlx-models/README.md` (≈line 152, the pinned-environment links),
  `req/phase6-inference-report.md` (pin table ≈54–55 **and** a new dated migration-note section),
  `req/full-roadmap.md` (Phase 7 pin line ≈745), `req/phase6-golden-precision.md` (re-measured
  TF32 bounds recorded next to the old ones).
- `req/plans/initial-plan.md` — Decision 9 prose (≈86–97, 557–558) and its research findings still
  name 0.31.2 / `fba4470`; updated by hand (no script edits markdown), kept as a historical record
  of *how* the original pairing was chosen with the new pairing noted.
- `req/plans/phase7-plan.md` §7.5 — a link to this sub-plan (`phase7-5-plan.md`).
- `jmlx-benchmarks/README.md` (≈line 74) and `req/phase6-compatibility.md` (≈line 85) are historical
  measurements pinned to 0.31.2; they are **left as-is**, not rewritten.

**Left untouched by design:** the `recorded_pin` value in the SmolVLM / Qwen3-VL / Phi-3.5-Vision
Tier-B manifests and their `tools/hf-reference/provenance.json` digests stay unchanged unless that
manifest's `local_evidence` is re-measured (then `recorded_pin` + digest move in the same commit).

## 3. Findings from required probes (WP0)

Run on the current pin before any edit; record the results in the PR description and carry the
upstream-change summary into the migration note:

- **Baseline.** `sw_vers` (host is macOS 27.0.1 on this machine — see the §6 host constraint),
  `git rev-parse HEAD`, `cat native/install/lib/native-pin.properties`, the full
  `./scripts/checkDependencies.zsh` output (target pairing and its mlx-c commit), and a green
  `./gradlew build` as the pre-change baseline.
- **Upstream survey (0.31.2 → 0.32.4 / mlx-c v0.6.0 → v0.8.0).** Read the MLX and mlx-c
  release notes / commit range for every skipped release (the mlx-c compare between `fba4470` and
  `a26f2d5`). Call out specifically:
  - **The two signature changes jmlx calls** (`mlx_cumsum`, `mlx_fast_scaled_dot_product_attention`,
    §1). The SDPA `force_fused=false` is already confirmed safe at the signature level (v0.31.2 had
    no `force_fused` parameter; v0.32.4's `mlx/fast.h:60` declares `bool force_fused = false`, so
    `false` reproduces the previous default call); WP0's SDPA task is limited to the residual risk —
    an upstream change *inside* the default SDPA path — which the WP3 goldens catch. Add both the
    SDPA kernel and the cumsum path to the list of possible causes of golden drift in case a golden
    moves in WP3.
  - **GPU reduced-precision (TF32) behaviour, the active-byte / allocator-tier accounting, the
    Threefry/Gumbel sampling streams, and allocator growth** — the evidence that a later bound
    change or golden shift follows MLX itself rather than jmlx.
- **Other changed mlx-c declarations (for the migration note).** v0.8.0 changes more than the two
  call-site symbols; list these separately in the migration note rather than lumping them into
  "added-and-unused". jmlx calls none of these today (only `mlx_cumsum` and the SDPA, both
  retargeted in WP1.4), so none is a retarget — the hazard is a later milestone calling an old name
  that still compiles but now means a different operation:
  - **removed names** (the symbol no longer exists): `mlx_distributed_group_rank` →
    `mlx_distributed_group_get_rank`; `mlx_distributed_group_size` →
    `mlx_distributed_group_get_size`;
  - **same name, different semantics** (the old name still compiles if the arity fits, but now means
    a no-axis / flattened scan; a new `_axis` / `_axes` variant carries the old per-axis behaviour):
    `mlx_cumsum` (+`mlx_cumsum_axis`), `mlx_cummax` (+`mlx_cummax_axis`), `mlx_cummin`
    (+`mlx_cummin_axis`), `mlx_cumprod` (+`mlx_cumprod_axis`), `mlx_logcumsumexp`
    (+`mlx_logcumsumexp_axis`), `mlx_median` (+`mlx_median_axes`), `mlx_trace` (+`mlx_trace_axes`);
  - **same name, new params (not called):** `mlx_gather_qmm` gains a `const mlx_array global_scale`
    before `sorted_indices` (matters for later quantized-MoE work);
    `mlx_detail_compile_clear_cache` and `mlx_detail_compile_erase` gain params;
    `mlx_fast_scaled_dot_product_attention` gains `force_fused` (this one *is* called and is
    retargeted in WP1.4);
  - **changed in place:** the `mlx_io_vtable` function pointers in `io_types.h`;
  - **added and used by this PR's retarget:** `mlx_cumsum_axis`;
  - **added and unused (later-milestone candidates), e.g. (illustrative, not exhaustive):**
    `mlx_cummax_axis`, `mlx_cummin_axis`, `mlx_cumprod_axis`, `mlx_logcumsumexp_axis`,
    `mlx_median_axis`, `mlx_median_axes`, `mlx_trace_axes`, and `mlx_fast_gated_delta_update` (the
    last is the Qwen3.8 Gated DeltaNet kernel — relevant to 7.6, not wired in here). v0.8.0's
    regenerated headers add many new symbols (the reviewing header diff counts 61 new function
    names); the authoritative, complete list is the regenerated `req/mlx-api-inventory.md` (reviewed
    in WP1.6), and §9 has reviewers check the migration note against that report.

## 4. Ordered work packages (one PR)

Ordered so the pin and native rebuild land before anything that consumes them. Each package names the
commit boundary.

### WP0 — baseline and upstream survey (no commit)
§3. Ends with the target pairing recorded, the `force_fused=false` SDPA default-call equivalence
confirmed against the v0.31.2 signature and v0.32.4's `mlx/fast.h:60`, the residual inside-kernel
risk handed to the WP3 goldens, the removed/renamed symbol list drafted, and the pre-change
`./gradlew build` green. Also run the WP6
allocator-tier **size sweep on the current 0.31.2 pin** (the probe described in WP6: allocate and
evaluate arrays at 8 KiB, 16 KiB, 16 KiB+1, 32 KiB+1, 128 KiB+1, 256 KiB+1 and record the
active-byte growth for each) so WP6 compares a **table against a table**, not a table against prose
(the repo documents the tiers only in prose — `phase7-3b-reference-findings.md`). Record the 0.31.2
table in the migration note.

### WP1 — pin, native rebuild, retargets, bindings and inventory (first commit)
1. Edit the three `MLX_METAL_*` lines in `scripts/bootstrap-native.sh` to 0.32.4 / new URL / new
   SHA-256, and the Decision-9 comment. **Do this before** running `updateMlx.zsh`: the script
   deliberately leaves the wheel lines untouched and only seds `MLX_C_COMMIT`. If the commit changed
   first while the wheel lines still said 0.31.2, bootstrap's step-4 pairing assertion (compares the
   new commit's `CMakeLists.txt` `GIT_TAG` against `v${MLX_METAL_VERSION}`) would die with
   "version pairing assertion failed" and the run would abort.
2. `./scripts/updateMlx.zsh a26f2d584a14ec619ae8db9a09be03becd47598a`. This re-checks out mlx-c,
   runs bootstrap (pairing assertion now passes), `regen-bindings.sh`, `generateMlxApiInventory`, and
   `generateMlxApiHeaderCoverage`. It keeps a `mktemp` backup of `scripts/bootstrap-native.sh` taken
   **after** the step-1 hand edits.
   **Recovery if this run fails:** the backup holds wheel 0.32.4 + commit `fba4470`, a **mismatched**
   pair, so the script's own "re-run bootstrap (now back on `$OLD_COMMIT`)" advice would fail the
   step-4 assertion. Recover explicitly instead:
   `git checkout -- scripts/bootstrap-native.sh && ./scripts/bootstrap-native.sh && ./scripts/regen-bindings.sh`
   (restores the committed 0.31.2 / `fba4470` pair and rebuilds), then start the WP1 sequence over.
3. Review the regenerated bindings: `git diff jmlx-ffi/src/main/generated/java` (must look like
   jextract output for the new commit).
4. **Retarget the two call sites** so the code compiles against the new signatures: `MLXOps.java:439`
   `mlx_h.mlx_cumsum(res, a, axis, reverse, inclusive, s)` →
   `mlx_h.mlx_cumsum_axis(res, a, axis, reverse, inclusive, <absent mlx_optional_dtype>, s)`; build
   the absent dtype with the existing helper `NativeOps.optDtype(Arena, DType)`
   (`NativeOps.java:332`, where `null` means absent) — do not hand-build the struct.
   `cumulativeSumAxis` has no temporary arena today, so this retarget also adopts the `opt*` arena
   pattern the other optional-argument call sites use (`try (Arena tmp = Arena.ofConfined())` and
   `NativeOps.optDtype(tmp, null)`). `MLXFast.java:199` adds `force_fused=false` before the stream
   argument. (If regen reveals additional
   signature changes, the same policy applies: mechanical retarget, no new public API, behaviour
   preserved.)
5. **Update the overrides JSON before verifying call sites:** in
   `req/mlx-api-inventory-overrides.json`, rename the `MLXOps` entry's `mlx_h.mlx_cumsum` to
   `mlx_h.mlx_cumsum_axis` (same `implemented`/facade/tests). Then re-run
   `./gradlew generateMlxApiInventory` so the inventory reflects the updated overrides.
6. Review `git diff req/mlx-api-inventory.md req/mlx-api-header-coverage.md` (the API reports — see
   §3 for the removed/renamed/added content), and run
   `./gradlew verifyMlxApiCallSites verifyMlxApiHeaderCoverage` — both must pass with the retargets
   in place (no unmapped call site; header coverage matches the new headers). Note: `verifyMlxApiCallSites` is not one of `updateMlx.zsh`'s own "next steps"
   (`verifyMlxApiHeaderCoverage` is, as its step 3) — the call-site verification is jmlx's, layered on
   top of the script.
7. `./gradlew build` green — this is the compile gate that proves the two retargets are sufficient.
8. Update the "Current pin" references in `AGENTS.md` and `req/plans/initial-plan.md` Decision 9
   prose.
9. **Commit.**

### WP2 — Python MLX oracle re-lock and fixture regeneration
1. Edit `tools/mlx-oracle/provenance.json` (`mlx`, `mlxMetal`, `mlxCCommit`) and
   `tools/mlx-oracle/requirements.lock` (the `mlx==`/`mlx-metal==` wheel lines and hashes).
2. `./tools/mlx-oracle/install.sh` (rebuilds the hash-locked venv on the new 0.32.4 wheels).
3. **Host constraint:** the oracle's `phase6` profile is
   `device: gpu, macOSMajor: 26, macOSMajorPolicy: exact` (and `profile_policy.py` forbids a GPU
   profile from using `minimum`), while this machine and the self-hosted `self-macos-xl` runner are
   macOS 27.0.1. So on a 27 host, `generateMlxOracleFixtures` regenerates only the CPU
   `phase7-1-core` fixture and **skips** `phase6-1-sampling` and `phase6-tier-a-array`. The GPU
   fixtures therefore must be regenerated on a **`macos-26` host**.
4. **Regenerate on `macos-26` via a temporary CI step** — *not* a new `workflow_dispatch` workflow:
   `workflow_dispatch` only fires for workflow files that exist on the **default branch**, and
   `main` is push-protected, so a workflow file that exists only on this branch cannot be dispatched.
   Instead, add a temporary step to `ci.yml`'s existing `macos-26` native job. **Placement:** put it
   **before** the "Run root verification" step (i.e. right after "Install hash-locked Python MLX
   oracle"), because that step runs `verifyMlxOracleFixtures -PmlxOracleRequireAllProfiles=true`,
   which fails on the stale 0.31.2 GPU fixtures and skips every later step. **Flag:** run
   `./gradlew generateMlxOracleFixtures -PmlxOracleRequireAllProfiles=true`, so a runner/profile
   mismatch hard-fails rather than silently skipping the GPU families and uploading a partial
   artifact. **Ordering:** `generateMlxOracleFixtures` `dependsOn verifyMlxOracle`, which checks the
   oracle provenance against the bootstrap pins — so the WP2.1 provenance/lock edits must be pushed
   in the **same push** as this temporary step, or `verifyMlxOracle` fails on the mismatched pins.
   **Restore the committed files before the step ends:** the step writes the regenerated
   `*.expected.json` into the CI workspace, and "Run root verification" — which runs
   `verifyMlxOracleFixtures` and, under `:check`, `SamplingOracleTest` — reads the files in that
   workspace. Leaving the regenerated files in place would make the oracle gate compare MLX's own
   fresh output against itself and pass no matter what is committed (a stale or hand-edited fixture
   would go unnoticed). Copy the regenerated files out, restore the committed ones, then upload from
   the copy:
   ```sh
   ./gradlew generateMlxOracleFixtures -PmlxOracleRequireAllProfiles=true
   mkdir -p "$RUNNER_TEMP/oracle" && cp tools/mlx-oracle/fixtures/*.expected.json "$RUNNER_TEMP/oracle/"
   git checkout -- tools/mlx-oracle/fixtures/
   ```
   then `actions/upload-artifact` from `$RUNNER_TEMP/oracle`. Push, download the artifact, stage the
   full fixture diff locally, and **remove the temporary step in the final review round.** (With the
   restore, `verifyMlxOracleFixtures` checks the *committed* fixtures on every intermediate run, so
   it stays red until the regenerated fixtures land and green once they do — the gate is never
   vacuous. Regenerating all three on 26 is preferred over mixing hosts: 26 satisfies every profile,
   so one path covers `phase6-1-sampling`, `phase6-tier-a-array` and `phase7-1-core`.)
5. Review the fixture diff (this is MLX's own output moving between versions — expected and
   reviewed, not a tolerance). `./gradlew verifyMlxOracle verifyMlxOracleFixtures` on this 27 host
   verifies only `phase7-1-core`; the full all-profile verification happens on the `macos-26` native
   CI run with `-PmlxOracleRequireAllProfiles=true` (§6).
6. **CI is expected to be red between the WP1 push and this WP2 commit:** the native job runs
   `verifyMlxOracleFixtures -PmlxOracleRequireAllProfiles=true`, which fails on the stale
   0.31.2-produced GPU fixtures until the regenerated fixtures land. That gap is expected and
   resolves with this commit. **Commit** the provenance, lock, and the three downloaded fixtures; the
   temporary `ci.yml` step (pushed together with the provenance/lock edits per the WP2.4 ordering)
   stays in place until the final review round (see §9), when it is removed.

### WP3 — independent references pass unchanged (no regeneration)
Run the evidence **not produced by MLX** on the new pins and require them to pass unchanged; a
failure is a finding to investigate per `req/phase6-golden-precision.md`, **never** a reason to
regenerate or loosen. If a golden moves, first check the WP0 suspects (the retargeted SDPA and cumsum
paths) before classifying it. In **both** precision modes (`:jmlx-core:test` +
`:jmlx-core:float32GoldenTest`, `:jmlx-models:test` + `:jmlx-models:float32GoldenTest`):
- the CPU Hugging Face goldens under `tools/hf-reference` (decoder-family, encoder/T5, the Qwen3-VL
  and Phi-3.5-Vision goldens);
- the tokenizer oracle (`verifyTokenizerOracle` + `verifyTokenizerOracleFixtures`) and the image
  oracle (`verifyImageOracle` + `verifyImageOracleFixtures`) fixtures;
- greedy IDs taken from those references (e.g. `mistral-window`);
- the HF-referenced Tier-B manifests (BERT SST-2, MiniLM, Flan-T5, SmolVLM, Qwen3-VL, Phi-3.5-Vision),
  which run in strict mode only (`tierBTest` sets `MLX_ENABLE_TF32=0`).

### WP4 — re-record jmlx-recorded goldens (evidence-gated)
These were produced by running on MLX, so an upgrade may legitimately change them. Re-record each
**only with the evidence that the change follows MLX itself**, and review each diff in the PR.
1. **Seeded sampling streams.** `LlamaModelTest.EXPECTED_SEED_42` / `EXPECTED_SEED_7`. Re-record
   **only after** the regenerated `phase6-1-sampling` oracle passes (`SamplingOracleTest` green on
   the new pins): that is the proof the stream move is MLX's, not a jmlx regression. The seeded
   stream is host-independent (fixed input of four equal-zero logits through a pinned runtime), so
   record the new constants **locally** and let the `macos-26` CI run confirm them
   (`SamplingOracleTest` is `@EnabledIfNativeAvailable`, not host-gated, and the current constants
   already pass on both hosts).
2. **Release-smoke golden.** `./tools/release-smoke/run.sh ci --record` (publishes the seven modules
   to the disposable smoke repo, resolves `jmlx-models` under a fresh `GRADLE_USER_HOME`, and rewrites
   `goldens/mistral-sampled.properties`, including its `native.*` pin lines). `--record` is `ci`-only.
3. **Pin-gated Phase 6 text Tier-B exact IDs.** `TierBSmokeTest` asserts `expected_token_ids` only
   while `recorded_pin` equals the running `runtimePin()`
   (`device|macOS|arch|mlx-metal|mlx-c|batch`); after the pin change it silently falls back to a
   "structural pass only", so exact-ID coverage is lost unless re-recorded. These are recorded on
   `macos-26` (the manifests currently read `Apple M1 (Virtual)|macOS=26.6.2`), so **dispatch the
   Tier-B workflow on the `macos-26` runner** (`gh workflow run tier-b.yml --ref <branch>` — that file
   lives on `main`, so it dispatches fine). **Note the dispatch runs all ten arms, including the three
   `self-macos-xl` arms (smolvlm, qwen3vl, phi3v) — the self-hosted runner must be online,** and the
   same run doubles as the WP3 HF-referenced Tier-B evidence. Per text manifest, set
   `recorded_pin`, `expected_token_ids`, `observed_peak_test_jvm_rss_kib` and `observed_run` (the run
   URL) from that run's output — do not record them from the local 27 host.
4. **8B mlx-vlm probe** (`tools/qwen-vl-reference/goldens/qwen3-vl-8b-wp1-mlx-probe.json`). Produced
   by Python MLX; re-run in the re-locked WP5 environment **only if** the opt-in 8B manual comparison
   (`Qwen3Vl8bManualProbeTest`) is repeated. Not a gate otherwise.
5. **Tier-B log line.** Add the running runtime pin to the `TierBReferences` log line (it currently
   prints only `recorded_pin`), so a log line is never misread as describing the new runtime.

### WP5 — converter environment re-lock (or confirmed)
`tools/qwen-vl-reference` pins the old MLX. Primary path: re-lock **only** `mlx` and `mlx-metal` to
0.32.4 in `requirements.in` / `requirements.lock`, and **hold** `mlx-vlm==0.3.9`,
`transformers==4.57.6`, `Pillow==12.3.0`, `numpy==2.5.3` — those shared pins are forced by
`verifyImageOracleDependencyPins` (the image, fast-comparison, Qwen-VL and HF-reference locks must
agree on Pillow/NumPy/Transformers). Also note `mlx-lm==0.29.1` in the lock: it is a transitive
dependency of `mlx-vlm`, requires `mlx>=0.29.2` with no upper bound, and re-locks cleanly against
mlx 0.32.4 — **hold it at 0.29.1** (do not let it move as part of this pin change). Update
`provenance.json` (`runtime.mlx`, `runtime.mlx_metal`, `requirements_lock_sha256`) in the same PR,
then `./tools/qwen-vl-reference/install.sh` and confirm the committed probe golden still matches
(or re-run it — WP4.4). If 0.32.4 cannot be adopted without moving one of the shared pins or
`mlx-vlm` proves incompatible, the fallback is a written **"confirmed compatible"** note — that
fallback requires explicit reviewer approval, not the re-lock.

### WP6 — memory bounds (findings-first)
- **Allocator-tier probe first.** Both `Qwen3VlMemoryTest` and `Phi3VMemoryTest` are built around
  MLX's allocator size tiers: the allocator is exact below 16 KiB and rounds each larger live buffer
  up to its 16/32/64 KiB doubling tier (e.g. `Qwen3VlMemoryTest`'s "mid ends with 16 KiB per K/V
  buffer, late with 32 KiB"; `Phi3VMemoryTest`'s `KV_TIER_SLACK`/`PROJECTED_TIER_SLACK`). A single
  array of one size **cannot** distinguish an accounting/tier change from a retention regression.
  **The probe** — allocate and evaluate arrays at sizes across the tier boundaries (8 KiB, 16 KiB,
  16 KiB+1, 32 KiB+1, 128 KiB+1, 256 KiB+1) and record the active-byte growth for each — is run
  twice: on the current 0.31.2 pin in WP0 (the baseline table) and on the new pin here. It is
  **scratch / report-only code, never part of `check`** (a throwaway `main`, or a report-only probe
  test in the pattern of `KVCacheQuantizationProbeTest`). Build the new-pin table and compare it
  **table-against-table** with the WP0 table; if they differ, adjust **both** memory tests' tier
  assumptions from that table (documenting the change in the migration note), not from a single
  per-buffer delta. If the two tables match, any exact-equality failure is a jmlx retention
  regression and must be fixed, not papered over.
- **Re-measure the measured bounds** (each recorded next to the old value; an exceedance is first a
  finding, a bound is raised only once its cause is documented — e.g. an upstream kernel/allocator
  change named in the migration note):
  - the separately named TF32 bounds in `req/phase6-golden-precision.md` (e.g. Qwen3-VL's `0.0025`,
    the cache-equivalence `0.003`, the family bounds `0.002` / `0.03` / `0.0012`);
  - the Tier-B native-memory gates: `TierBQwen3VlTest`
    (`NATIVE_PEAK_BOUND_BYTES` 9,828,890,788 B + 500 MiB; `FOOTPRINT_BOUND_BYTES` 12.5 GB against a
    sampled 11,979,116,480 B), `TierBVisionLanguageTest`
    (`NATIVE_PEAK_BOUND_BYTES` 5,737,416,464 B + 500 MiB; `FOOTPRINT_BOUND_BYTES` 8 GiB against a
    sampled 7,129,368,904 B), and `TierBPhi3VTest`
    (`NATIVE_PEAK_BOUND_BYTES` 31,100,376,165 B + 500 MiB; `FOOTPRINT_BOUND_BYTES` 52,000,000,000 B
    against a sampled 51,421,089,584 B — **the tightest margin of any bound, ~552 MB**, and the most
    likely to trip); plus the `local_evidence` blocks those constants cite;
  - `VisionLanguageMemoryTest`'s `FIXED_MARGIN` (16 KiB, sized from a measured warm overhead of at
    most 11,264 B).
  A strict `full-float32` bound is **never** loosened.
- **Never re-measure design bounds** (an exceedance is a regression): `Qwen3VlVisionPeakTest`
  `DEPTH_DELTA_BOUND` (64 MiB depth independence) and `PLANNING_BOUND` (the plan's 5 GiB), and the
  512 B `CLEANUP_MARGIN` in `VisionLanguageMemoryTest`, `Qwen3VlMemoryTest` and `Phi3VMemoryTest`.
- **Re-run the complete memory checklist** (all must pass; margin/slope checks absorb accounting
  change, no-change checks and design bounds stay regressions): `VisionLanguageMemoryTest`
  (`FIXED_MARGIN`), `Qwen3VlMemoryTest` and `Phi3VMemoryTest` (tier-based, adjusted per the sweep if
  the tiers moved), `DecoderCacheLifecycleTest` (8 MiB), `Seq2SeqMemoryTest` and `Phase71ContractsTest`
  (baseline + 64 KiB), `LlamaModelTest` (baseline + 4 KiB), `KVCacheMemoryPlateauTest` (3 × logical
  + 8 MiB), `BatchSchedulerMemoryTest` (`SLACK` 8 MiB + early/late slope), `KVCacheTest` (×4 linear
  growth, one row + 1 MiB after a reorder), `MLXMemoryLeakTest` / `LinearTest` /
  `QuantizedLinearTest` / `SwitchGluTest` (2,000,000 B leak thresholds), `MLXScopeTest` (500,000 B
  freed-detection slack), the `clearCache` and rejected-call "rejection must not allocate" checks in
  `MLXMemoryLeakTest`, `Qwen3VlVisionTowerValidationTest` and `VisualFeatureInjectionTest`, and the
  report-only `KVCacheQuantizationProbeTest`, `PackedAttentionProbeTest` and opt-in
  `Qwen3Vl8bManualProbeTest`.
- **Manifest re-measure rule.** When a manifest's `local_evidence` is re-measured, the manifest
  changes: its `recorded_pin` and its digest in `tools/hf-reference/provenance.json` are updated in
  the **same** commit, overriding the "leave unchanged" default in §2 for that manifest only.

### WP7 — migration note, documentation, final gate
1. **Migration note.** A new dated section in `req/phase6-inference-report.md` covering: the old and
   new pairing; every skipped upstream release (MLX 0.31.3–0.32.3, mlx-c v0.6.0→v0.8.0) and the
   upstream change that mattered; the inventory diff against the previous pin with **removed** and
   **renamed** symbols listed separately from the added-and-unused ones (per §3, including the
   `mlx_io_vtable`/`mlx_detail_compile_*` changes and the `mlx_fast_gated_delta_update` note for 7.6);
   the two call-site retargets and the `force_fused=false` SDPA finding; the oracle fixture diff
   rationale; every re-recorded golden and why it moved; and every memory-bound or accounting change
   with its documented cause (including the allocator-tier sweep table if the tiers moved).
2. **Current-state docs.** Update the pin tables and status lines in §2 (AGENTS.md, README.md,
   jmlx-models/README.md, req/phase6-inference-report.md, req/full-roadmap.md,
   req/phase6-golden-precision.md); link this sub-plan from `req/plans/phase7-plan.md` §7.5. Leave
   dated historical evidence untouched.
3. **Final gate.** Full `./gradlew build` (and `:jmlx-core`/`:jmlx-models` `check`, which pull in the
   `float32GoldenTest` tasks) on the new pins; the `macos-26` native CI run with
   `-PmlxOracleRequireAllProfiles=true` and its required-native-suites assertion (no skips);
   `./tools/release-smoke/run.sh ci` green against the re-recorded golden.
4. **Optional benchmarks** (§7): recorded evidence only.

### Note — bit-identity recordings start over
The exact before/after bit-identity comparison (the opt-in `exactBitsRecord`/`exactBitsVerify*`
tasks and the `exactBitsVerifyTf32On`/`Off` recordings, never part of `check`) refuses to compare
across native pins. Recordings made before this upgrade are therefore **not reused** and are not
re-run as a gate of this PR. Any later bit-identity claim records a fresh baseline on the new pins.
This is forward-looking guidance carried in the reusable checklist, not an active task here.

## 5. Compatibility decisions

- **Paired pin moves together** (Decision 9): `mlx-metal` and its mlx-c commit always change in one
  step, verified by the bootstrap's step-4 pairing assertion. Never bump one without the other.
- **No new call sites; two existing ones retargeted.** The upgrade adds no new mlx-c call and no new
  public Java API. It retargets exactly two existing call sites to the v0.8.0 signatures
  (`MLXOps` cumsum → `mlx_cumsum_axis` + absent dtype; `MLXFast` SDPA += `force_fused=false`), both
  preserving current public behaviour. `verifyMlxApiCallSites` and `verifyMlxApiHeaderCoverage` must
  pass; the overrides JSON is updated for the renamed symbol before the call-site check.
- **MLX semantics, dtype/shape/promotion, lazy/stream behaviour** are unchanged in jmlx; any change
  is upstream behaviour surfaced through the same facades. The single-thread/stream model,
  scope-confinement rules and `NativeLoader` contract are untouched.
- **Backward compatibility.** No published artifact changes — Phase 7 does not release. The staged
  runtime and the `jmlx-native-macos-arm64` payload move to 0.32.4 in-tree only; a later release
  (Phase 8) carries it.

## 6. Tests and verification matrix

Host note: the local and self-hosted `self-macos-xl` hosts are macOS 27.0.1; only the GitHub-hosted
`macos-26` runner satisfies the oracle's GPU profile (exact macOS 26). GPU-produced fixtures and the
text Tier-B IDs therefore come from `macos-26`.

| Evidence | Produced by | Action on upgrade | Precision mode | Host |
| --- | --- | --- | --- | --- |
| HF CPU goldens (`tools/hf-reference`) | Hugging Face, CPU | **Must pass unchanged** — never regenerate | both (`test` + `float32GoldenTest`) | local + macos-26 |
| Tokenizer / image oracle fixtures | HF `tokenizers`, Pillow/Transformers | **Must pass unchanged** | pure-Java / no native | local (27 ok) |
| Reference-derived greedy IDs (e.g. `mistral-window`) | HF references | **Must pass unchanged** | both | local + macos-26 |
| HF-referenced Tier-B (BERT/MiniLM/Flan-T5/SmolVLM/Qwen3-VL/Phi-3.5-Vision) | HF / sentence-transformers | **Must pass unchanged** | strict only (`tierBTest`, TF32=0) | macos-26 / self-macos-xl |
| Python MLX oracle fixtures (phase6-1-sampling, phase6-tier-a-array, phase7-1-core) | MLX | **Regenerate** (GPU on macos-26) and review diff | TF32=0 | macos-26 |
| Seeded sampling (`LlamaModelTest` 42/7) | MLX | **Re-record** (host-independent stream) only after `phase6-1-sampling` oracle passes | per test | recorded local; CI confirms on 26 |
| Release-smoke `mistral-sampled.properties` | MLX (jmlx runtime) | **Re-record** via `run.sh ci --record` | per golden | local (ci mode) |
| Phase 6 text Tier-B exact IDs (4 manifests) | MLX (jmlx runtime) | **Re-record** from a `macos-26` `tier-b.yml` run | strict only | macos-26 |
| 8B mlx-vlm probe golden | Python MLX | **Re-run** only if opt-in 8B comparison repeated | per probe | local (opt-in) |
| Allocator-tier accounting (`Qwen3VlMemoryTest`, `Phi3VMemoryTest`) | MLX allocator | **Size-sweep probe** across 16/32/64 KiB tiers; adjust tier assumptions from the table if changed | both | local + macos-26 |
| Measured bounds (TF32; Tier-B Qwen3-VL / SmolVLM / Phi-3.5-V peak+footprint; `FIXED_MARGIN`) | MLX measurement | **Re-measure**, record old + new, document cause | per bound | macos-26 / self-macos-xl |
| Design bounds (`DEPTH_DELTA_BOUND`, `PLANNING_BOUND`, `CLEANUP_MARGIN`) | jmlx design | **Never re-measure** — exceedance is a regression | both | local + macos-26 |
| Margin/slope + no-change memory suites (WP6 list) | MLX | **Re-run, must pass** | both | local + macos-26 |
| `BatchStepEquivalenceTest` / `Qwen3VlCacheEquivalenceTest` / `DecoderSeamTest` | MLX | **Re-run in both tasks** with mode-specific tolerance | both | local + macos-26 |

CI: the `macos-26` native job runs `:jmlx-core` and `:jmlx-models` in both precision tasks, the
oracle verification with `-PmlxOracleRequireAllProfiles=true`, and the "Assert required native
suites executed" step. The required-suite list is unchanged unless a suite is added or removed;
if one is, add it to that list (a skip otherwise passes CI unnoticed). `BatchStepEquivalenceTest`
and the dual-mode equivalence suites are asserted in both `test` and `float32GoldenTest` result
directories.

## 7. Benchmarks

Optional evidence only, never a gate. If the upstream survey (WP0) names a measured performance
change, or the reviewer wants it, re-run the decode benchmark and `benchmarkTf32` on the new pins and
record them with a full reproducibility tuple (clean commit, samples, warmups, medians with spread)
under `req/benchmarks/`, following the 7.3b WP6 one-off-record precedent. Do not treat the spread as
a confidence interval and do not gate the upgrade on it.

## 8. Exit gate (all-or-nothing)

- [ ] `./scripts/checkDependencies.zsh` re-run at execution; the pairing used (0.32.4 / v0.8.0 or a
      newer one) and its wheel hashes are recorded in the PR.
- [ ] Pins updated together; bootstrap step-4 pairing assertion passes; `native-pin.properties`
      regenerated to the new pin.
- [ ] Bindings regenerated and reviewed; `req/mlx-api-inventory.md` and
      `req/mlx-api-header-coverage.md` regenerated and reviewed against the **previous** pin;
      the two call-site retargets are in place and the overrides JSON updated, so
      `verifyMlxApiCallSites` and `verifyMlxApiHeaderCoverage` pass with **no new call site**
      (retargets listed); `./gradlew build` compiles.
- [ ] Oracle re-locked (provenance + lock); all three fixtures regenerated (GPU on a `macos-26` host,
      via the temporary `ci.yml` step) and committed; `verifyMlxOracleFixtures` passes **all** profiles
      on a `macos-26` native CI run with `-PmlxOracleRequireAllProfiles=true` — a run on which this
      gate counts only if the temporary step is removed or restores the committed fixtures (a run
      where the step leaves regenerated files in the workspace compares MLX's fresh output against
      itself and is vacuous); the temporary CI step is removed.
- [ ] Independent references (HF goldens, tokenizer/image oracles, reference-derived greedy IDs,
      HF-referenced Tier-B) pass **unchanged** in the required modes; none were regenerated; the
      `force_fused=false` SDPA default-call equivalence was confirmed (v0.31.2 signature vs v0.32.4
      `mlx/fast.h:60`), with the residual inside-kernel risk left to these goldens.
- [ ] jmlx-recorded goldens re-recorded with evidence (sampling only after the sampling oracle passes,
      recorded locally with CI confirmation; release-smoke via `run.sh ci --record`; text Tier-B IDs
      from a `macos-26` `tier-b.yml` run); every diff reviewed.
- [ ] Converter environment re-locked to 0.32.4 (shared pins and `mlx-lm` held) — or a "confirmed
      compatible" note with explicit reviewer approval; `provenance.json` updated in the same PR.
- [ ] Memory: allocator-tier size-sweep run and its table recorded; both tier-based memory tests
      adjusted from the table if the tiers moved (else confirmed unchanged); measured bounds
      (including Phi-3.5-V's tight 552 MB footprint bound) re-measured with old + new values and a
      documented cause for any raise; no strict `full-float32` bound loosened; no design bound
      re-measured; the full WP6 memory checklist passes; any re-measured manifest has its
      `recorded_pin` + `tools/hf-reference` digest updated in the same commit.
- [ ] `TierBReferences` log line now prints the running runtime pin.
- [ ] Migration note added (covers every skipped release, the removed/renamed symbols, and every
      change above); all current-state doc references updated; this sub-plan linked from
      `phase7-plan.md` §7.5; dated historical evidence untouched.
- [ ] Full `./gradlew build` green; `macos-26` native CI green (both precision tasks, required suites
      executed not skipped); `./tools/release-smoke/run.sh ci` green against the re-recorded golden.
- [ ] No pre-upgrade bit-identity recording is reused; the note that they start over on the new pins
      is stated in the migration note.
- [ ] One dedicated PR; no feature work bundled.

## 9. PR shape and reviewer attention

One PR, ordered commits: (1) pin + native + two retargets + overrides + inventory (WP1);
(2a) oracle provenance + lock + the temporary `ci.yml` step (pushed together — `generateMlxOracleFixtures`
depends on the new provenance/lock, WP2.4); (2b) the downloaded regenerated fixtures (WP2);
(3) re-recorded goldens + Tier-B log line (WP4); (4) converter re-lock (WP5); (5) memory bounds, if
any changed (WP6); (6) migration note + docs (WP7); and a final-round commit that removes the
temporary `ci.yml` generate step added in (2a). Reviewers should
focus on:

- the **two call-site retargets** (the only hand-written code change) and that `force_fused=false`
  truly reproduces the old SDPA behaviour (WP0 finding), and that the overrides JSON / inventory stay
  consistent with the `cumsum` → `cumsum_axis` rename;
- the **bindings diff** (large and mechanical — confirm it is the jextract output for the new commit,
  byte-identical to `regen-bindings.sh`);
- the **inventory diff** (which symbols were added, renamed, or removed; confirm the removed/renamed
  list in the migration note matches the regenerated report);
- the **oracle fixture diffs** (especially the GPU phase6 fixtures — MLX's own output moving between
  versions, reviewed not tolerated);
- the **seeded-sampling diff**, which must trace to MLX via the green `phase6-1-sampling` oracle
  (`SamplingOracleTest`), not to a jmlx regression;
- the **text Tier-B `recorded_pin` + IDs** (must come from the `macos-26` `tier-b.yml` run, not the
  local 27 host — a local-recorded `recorded_pin` would read macOS 27 and the exact-ID gate would
  never assert on CI);
- any **memory-bound change**, each with its documented upstream cause and the old value recorded
  next to the new one — and the **allocator-tier sweep table** if the 16/32/64 KiB tiers moved;
- the **migration note** for completeness (every skipped release, every moved number, the
  removed/renamed symbols);
- the invariant that **independent references are byte-unchanged** across the PR.
