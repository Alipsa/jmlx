# Phase 7.6 implementation checkpoint — 2026-11-03

Resumed on 2026-11-03 after the user's requested pause. The task remains to implement the
entire phase, verify it, then open a PR. Work is uncommitted on `phase7-6-implementation`.
No PR is open. This checkpoint is not a completion or acceptance report.

## Implemented so far

- Isolated `tools/qwen35-reference/{hf,mlx}` environments installed from the proposed hash locks.
  No existing reference environment, native pin, generated binding or version was changed.
- Reference tools cover tiny HF text/image execution, config/partial rotary probes, actual
  mlx-vlm/mlx-lm float and affine conversion, saved-output reloads, delta dispatch/boundary
  probes, normalization-only diagnostics, real-artifact inventories and reference benchmarks.
- Full pinned 0.8B and 27B artifacts were downloaded and hashed. Inventories are under
  `tools/qwen35-reference/evidence/`; both real configs execute in the pinned environments.
- Actual converter fixtures are under `tools/qwen35-reference/fixtures/`. The tiny tokenizer
  was changed from unsupported WordLevel to an in-range plain BPE tokenizer; all four actual
  converters were rerun through `run_converters.py` after that correction.
- Core `GatedDeltaRule` implements explicit token recurrence and bounded 64-position chunked
  prototypes. `RecurrentCache` retains evaluated FLOAT32 state and convolution history.
  `GatedDeltaNet` composes projections, causal convolution, HF Q/K normalization, decay/beta,
  ordinary gated norm and output projection. Prefill method is explicit; no default is selected.
- Added strict HF component goldens and `GatedDeltaRuleTest`, `GatedDeltaNetTest`,
  `RecurrentCacheTest`, `GatedDeltaMemoryTest`. These passed targeted native checks. The memory
  test covers repeated requests, exact parameter-scope handle stability and 4,128 positions.
- CI's required native suite assertion includes those four classes, with the two strict
  component suites assigned to `float32GoldenTest`.
- An opt-in `:jmlx-core:deltaBenchmark` and separate binding-count agent work. Counting uses
  runtime instrumentation only; generated bindings on disk are untouched. The agent smoke
  recorded nonempty actual native downcall counts. Timing and counted runs are separate.

## Findings and outstanding gate

- Stock mlx-vlm's normalization difference is measurable: tiny prefill logit differences from
  HF are about 0.01; the normalization diagnostic reduces them below 6e-5. Correct cached
  decode calls `model.language_model`, not the multimodal wrapper, to retain MRoPE positions.
- The 27B stock/diagnostic paired runs use identical pixels and eight teacher-forced stock
  histories. Greedy choices agree in the recorded cases. Prefill logit differences reach
  about 0.10. Attribution summaries are under `evidence/27b-normalization-attribution.json`.
  They are diagnostics, never acceptance goldens; Java/reference remainder is unmeasured.
- MLX's built-in delta op accepts widths the custom kernel does not, but misses the current
  strict 1e-4 component check at real head widths. It is explicitly marked ineligible in the
  benchmark rather than given a relaxed tolerance. Masked outputs differ between native and
  ops; active outputs and final states are checked separately.
- Reference-only 48-layer/1,280-position medians were roughly 323 ms chunked, 2,836 ms token
  recurrence, 112 ms custom Metal. Those early results used the older benchmark script.
- The benchmark was then amended to persist identical prepared inputs for reference and Java.
  The new reference single-layer results are recorded. Exploratory Java results using those
  exact inputs passed the real-shape 1e-4 comparison to token recurrence; 27B/1,280-position
  medians were about 54 ms chunked and 413 ms recurrence per layer. These are exploratory,
  dirty-tree results in `/tmp/qwen35-java-exploratory.json`; final clean-commit measurements,
  including 48-layer aggregate and binding counts, remain to be collected.
- An asynchronous WP1 review question is pending: approve chunked ops as default subject to
  Java validation, or review custom Metal scope. There is no approval yet. The earlier question
  quoted reference-only timings; present the actual Java aggregate measurement before selection.
  The plan explicitly requires review before selecting the default. Do not infer approval
  from elapsed time or the general implementation request.
- The 0.8B independent HF real-artifact run is recorded. HF's own cached/fresh text discrepancy
  reaches 3.19e-4; the failed 1e-4 measurement remains recorded as false. Strict tiny bounds
  were not changed. Real-artifact tolerances must be declared separately before Java acceptance.

## Resume priorities

1. Review the worktree and this checkpoint before further edits. No full build has been run yet.
2. Fix `probe_reload.py`'s mlx-lm observation coverage: its current delta/gated-norm hooks and
   source labels target mlx-vlm even for `--family lm`. mlx-lm imports `gated_delta_update` from
   `mlx_lm.models.gated_delta` into `mlx_lm.models.qwen3_5`, uses `normalize_qk`, `nn.Conv1d`
   for both phases, and `Qwen3NextRMSNormGated` from `qwen3_next`. Record its actual paths.
   Then rerun affected tiny probes/diagnostics and regenerate attribution summaries as needed.
3. Refresh tiny inventory JSON: older standalone tiny inventories predate the BPE correction.
   `converter-provenance.json` is from the final BPE conversion and includes current inventories.
4. Finish reference README, provenance and pure-Java/Groovy `verifyQwen35ReferenceGoldens` root
   check guard; neither README nor provenance.json nor Gradle guard exists yet. Preserve the
   distinction between smoke probes, diagnostics, independent goldens and pending acceptance.
   Record real capture files as external, not missing committed fixtures.
5. Finish WP0 findings and commit it separately from the WP1 core prototype commit, as planned.
   Commit verified prototype sources before final benchmark runs so measurements name a clean SHA.
   The plan/roadmap status still says planning; update accurately, not to completed.
6. Rerun reference 48-layer benchmarks with the new `--inputs-dir`, and collect final Java
   timings/counts. GPU dispatch counts are not observed; do not equate native downcalls with
   dispatches. Preserve host observations and describe their limits.
7. Obtain the required WP1 review before selecting a default or advancing past its gate.
8. WP2–WP8 remain: gated full attention/partial rotary integration and hybrid blocks; strict
   descriptor/layout preflight; generation-loop extraction with old-path exact-bit checks;
   text loader/model; vision reuse and family-specific DeepStack guards/tests; ownership and
   cancellation; full Tier-A/default-precision and real-artifact Java acceptance; docs/full CI.
   Do not open a completion PR until the user's requested work is completed and verified.

## Local resources and commands

- Artifacts: `/tmp/jmlx-phase76/artifacts/{hf-08b,mlx-27b}`.
- Large 27B captures: `/tmp/jmlx-phase76/captures/{27b-stock,27b-diagnostic}.npz`, about 2.6 GiB
  each. Their SHA-256s are recorded in the probe JSON; do not commit the capture binaries.
- Identical benchmark inputs: `/tmp/jmlx-phase76/benchmark-inputs/*.safetensors`, external.
- Selected reference wheels: `/tmp/jmlx-phase76/wheels/{hf,mlx}`, external; exact wheel hashes
  and tags are in `evidence/{hf,mlx}-environment.json`.
- Final BPE converter staging: `/tmp/jmlx-phase76/converters-bpe`.
- Logs: `/tmp/qwen35-*.log`; targeted tests passed in `core-test`, `cache-test`, `net-test`,
  `memory-test`, and latest `benchmark-lint` logs. `counter-smoke` passed. Earlier failed lint
  logs were corrected; the original HF real 1e-4 failure is preserved separately.
- Java benchmark example:
  `JMLX_DELTA_BENCHMARK_INPUT_DIR=/tmp/jmlx-phase76/benchmark-inputs ./gradlew :jmlx-core:deltaBenchmark`.
  Optional environment settings select shapes, lengths, sample count, layer repetitions and
  output file (`JMLX_DELTA_BENCHMARK_{SHAPES,LENGTHS,SAMPLES,LAYERS,OUTPUT}`).
  Add `-Pqwen35CountBindings=true` for the separate counting run; its timings are invalid.
- References require explicit venv Python. Native Gradle/MLX GPU commands need the existing
  sandbox escalation pattern; no auto-review rejection occurred during this task.

## Resume update — 2026-11-03

- Corrected mlx-lm delta/gated-norm hooks and source labels; both float/affine saved-output
  probes reran. The older executed probe is archived with a matching 27B source hash.
- Refreshed tiny inventories after the BPE correction; exact saved float conversion checks pass.
- Added the reference README, explicit provenance recorder and ordinary Gradle digest guard.
- `verifyQwen35ReferenceGoldens :jmlx-core:check` passes, including both precision tasks.
- Temporary files were removed between sessions. Artifacts and reproducible benchmark inputs
  are restored under ignored `.tier-b/phase7-6/`; large capture restoration is in progress.
- Remaining immediate work: reviewed provenance/docs commits, final clean-source Java/reference
  benchmarks and native-call counts, then the required WP1 default review. No default is selected.
- `resume.sh` is an unrelated untracked user file; leave it untouched and out of commits.
