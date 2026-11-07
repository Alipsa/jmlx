# Phase 8.2 — Reference findings (WP0+WP1)

- **Status:** WP0 readiness and comparison-contract record, 2026-11-06. All evidence below was
  collected on the measurement host named in §8 from locally verified inputs; no headline
  measurement was executed. WP1 adds §13 (frozen worker protocol and final report contract)
  and the two Python workers; its evidence is same-host smoke runs on 2026-11-06 — contract
  verification, not headline measurement.
- **Parent:** [Phase 8.2 plan](phase8-2-plan.md) — "WP0 — Record readiness and freeze the
  comparison contract". This document is the plan's required
  `req/plans/phase8-2-reference-findings.md`.
- **Gate check:** no unpinned artifact, unbudgeted MoE run or unreviewed timing/precision
  mismatch may enter measurement. §2 pins every input, §3 budgets the MoE case, §7 budgets
  every cell, and §7.3 records the one timing-semantics mismatch found (the 4-bit 2048
  page-cache regime) with a frozen mitigation. Missing capacity is stated explicitly in §11,
  not waived.

## 1. Readiness and the 8.1 freeze lift

- Phase 7 reconciliation is owned by
  [Phase 8.1 WP0](phase8-1-plan.md#wp0--confirm-readiness-and-record-the-release-inputs): its
  gate table passed on 2026-11-04, the last open row being the 7.6 self-hosted qwen35 Tier-B
  run (run `41841745005`). Phase 8.2 inherits that record and adds no new Phase 7 claims.
- The 8.1 publication freeze ran until the WP5 development-bump PR merged. PR #84
  (`phase8-1-dev-bump`) merged at **`42e6ff03dec95e589169973fd589a060ad226f28`** on
  **2026-11-05** — the freeze-lift evidence the plan requires. The 8.1 publication itself is
  recorded in `req/phase8-1-release-checklist.md` (`0.6.1` / `0.1.1` / `0.1.1`, one combined
  deployment, published 2026-11-05). This branch descends from that merge.
- Per the roadmap, no Central publication of any module is required to measure a local
  checkout; Phase 8.2 publishes nothing and changes no module version.

## 2. Frozen workload manifest

`tools/decode-baseline/workloads.json` (schema `jmlx-decode-baseline-workloads-1`) is the
single frozen input for both runtimes. WP0 replaced every candidate with exact revisions,
per-file SHA-256 and byte counts; the committed file's own digest is
`e8c7730178a2e8f1bf1ace461db14119d2394fff4f60fa90deca34ef4857a270` (WP1: carries the
`run_policy` finite limits of §7.4 and the `worker_sources` digest pin for the two
committed Python workers; the WP1 review rounds added `run_policy.first_requests`
and re-pinned `worker_sources` after the round-1 through round-4 worker fixes of
§13.4), which the guard pins as the frozen-contract backstop (§2.3).

### 2.1 Text cases

| Case | Role | Artifact (revision) | License | Stop set (union, resolved) | Stored dtypes | Identity claim |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-0.6b` | dense-unquantized | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` | Apache-2.0 | `{151643, 151645}` | 311 × BF16 | greedy |
| `tinymistral-6x248m` | MoE | `M4-ai/TinyMistral-6x248M` @ `2b4c0793c2e1dc58df51b9a6ce6852ba28e1a72c` | Apache-2.0 | `{2, 32001}` | 303 × F32 | greedy |
| `qwen3-8b-4bit` | quantized-4bit | `mlx-community/Qwen3-8B-4bit` @ `545dc4251c05440727734bcd94334791f6ab0192` | MIT | `{151645}` | 254 × U32 + 653 × BF16 | fixed-continuation |

Stop-set sources are recorded per case in the manifest (`stop_token_sources`): the union of
`generation_config.json` / `config.json` `eos_token_id` and the tokenizer's `eos_token_id`
as resolved by the pinned transformers 5.22.1 — the installed mlx-lm builds its stop set
from exactly these two sources (`tokenizer_utils.py` / `utils.py`). The Qwen tokenizer eos
is 151645 (`<|im_end|>`), not 151643 (`<|endoftext|>`), and the `phi3v`
tokenizer eos is 32000 (`<|endoftext|>`), not 32001 (`<|assistant|>`); both were
re-verified against the local digest-pinned artifact tokenizer files on 2026-11-06.
`qwen3-8b-4bit` has no `generation_config.json`. Both stock loops stop on exactly these
sets, so the stock comparison is stop-matched (plan §3.2 "Stock Java pair path").

The 4-bit case carries its existing pins forward: all 11 local file digests (including
`model.safetensors` 4,607,835,174 B and `model.safetensors.index.json`) were re-verified
against the recorded pins of `req/phase6-4-benchmark.md` on 2026-11-06; the case's remaining
gaps were the old execution pin (now matched, §9) and the absence of an independent output
comparison (closed by the qualification runs, §2.4).

`expected_dtype_histogram` is the **stored-tensor** count per safetensors dtype token, read
from each checkpoint header on 2026-11-06:

- `qwen3-0.6b`: 311 BF16 tensors; `lm_head.weight` is **not stored**
  (`tie_word_embeddings: true`). After loading, Python folds the tied head into the
  embedding (310 parameter arrays) while Java names it as an extra `lmHead.weight` (311);
  both read the same 311 stored tensors and compute in BF16 (TF32-immune matmuls).
- `tinymistral-6x248m`: 303 F32 tensors, not tied (both `lm_head.weight` and
  `model.embed_tokens.weight` stored). Both runtimes compute in float32 — **TF32-sensitive**;
  only this case can support a TF32 sensitivity interpretation (plan §3.2).
- `qwen3-8b-4bit`: 907 tensors — 254 U32 (4-bit packed, `bits: 4`, `group_size: 64`,
  including the U32-packed `embed_tokens` and `lm_head`) + 653 BF16 sidecars/norms/biases,
  not tied. Both runtimes load the identical packed layout; the 4-bit path is
  TF32-immune (no float32 matmul operands).

### 2.2 VLM follow-up cases

`qwen3vl-2b-instruct` (`Qwen/Qwen3-VL-2B-Instruct` @
`89644892e4d85e24eaac8bacfd4f463576704203`, Apache-2.0, stop `{151643, 151645}`) and
`phi3v-4b-instruct` (`microsoft/Phi-3.5-vision-instruct` @
`12b77fb40b63a2c73c68243d3f767aab688a1b2a`, MIT, stop `{2, 32000}`) pin **the manifest-file
digest** of the existing Tier-B manifests (`tools/tier-b/qwen3vl-2b-instruct.json`,
`tools/tier-b/phi3v-4b-instruct.json`); prompt, images, cache and generation policy are owned
by those manifests. The full artifact sets are re-verified at WP4 execution, not re-downloaded
in WP0. Known per-artifact mismatch recorded for WP4: the stock-resolved `phi3v` stop set is
`{2, 32000}` (the tokenizer eos is 32000 `<|endoftext|>`, 32001 is `<|assistant|>`), while the
Phase 7.3c Tier-B run pinned the jmlx-side set `{32000}`; the only difference is the extra
token 2 (the config eos). Each WP4 run mode must use one explicitly recorded set.

### 2.3 Pure-Java guard

`./gradlew verifyDecodeBaselineManifest` (wired into root `check`) verifies the manifest
without Python: schema, the frozen precision env values, the full controlled generation
policy (including the `prefill`/`cache`/`wired_limit`/`penalties` strings verbatim) and the
stock generation policy, per-file pin shape, the stop set equaling the **union of its
recorded sources** for every case, stop-set/prompt-exclusion agreement, the per-case
`identity_claim`, **full SplitMix64 re-derivation of all 3 prompt streams** (the committed
2048 arrays are the oracle for the JLS-long arithmetic — `Long.remainderUnsigned` semantics,
since the Python reference takes `z % vocab` over the unsigned 64-bit value), the canonical
prompt SHA-256s, the 0.6b shared pins against `tools/tier-b/qwen3-0.6b.json`, both VLM
manifest-file digests (with the VLM manifest path confined to the two declared repository
inputs) and repository/revision agreement, the Python lock (the 8 direct pins with PEP 503
name normalization, complete SHA-256 hash coverage including PEP 508 extras, malformed-pin
rejection, and the absence of `mlx-vlm`), the WP1 `run_policy` block (each frozen §7.4 limit, including `first_requests`,
as an exact nonpositive-rejecting value, `cache_regime`, `io_affected_cells`, and
`emitted_tokens` agreeing with `controlled_generation.max_new_tokens`) and the
`worker_sources` freeze (exact file set `controlled.py` + `worker.py`, each path confined
to `tools/decode-baseline/`, each re-hashed against the pinned SHA-256), and finally the
manifest's own SHA-256 as the
frozen-contract backstop for any field the structural checks do not name. Negative tests
executed on 2026-11-06: file-sha, prompt-array, controlled-policy, stock-policy,
stop-union, identity-claim, VLM-path, VLM-manifest, note-drift (digest backstop), malformed
requirements pin, extra-without-hashes, schema, run-policy-limit,
run-policy-first-requests, run-policy-regime,
io-affected-cells, worker-source-digest, worker-file-edit and worker-file-set tampers
each fail (the worker-file-edit and worker-source-digest tampers share the "worker
source SHA-256 differs" error by construction); the restored manifest passes. After the
round-1 re-pin the guard tampers (run-policy-limit, run-policy-first-requests,
run-policy-regime, note-drift, io-affected-cells, worker-source-digest, worker-file-set,
worker-file-edit) were re-executed against the re-pinned manifest and fail with distinct
errors; the remaining tampers cover structural checks whose manifest fields are
byte-identical in the re-pin (only `run_policy.first_requests` and `worker_sources`
changed), and the restored files are byte-identical after every round.

### 2.4 Qualification results (untimed, separate processes)

Both runtimes, headline `MLX_ENABLE_TF32=1` and strict `=0`, same prompt IDs:

| Case | Max abs logit diff (TF32=1 / =0) | Greedy-region min top-2 gap | Result |
| --- | --- | --- | --- |
| `qwen3-0.6b` | 0.0 / 0.0 | 0.5 (≫ 0.05 bound) | per-step argmax and 16-step greedy exact; 3 gap-0 steps all in the fixed-continuation region (tie-eligible) |
| `tinymistral-6x248m` | 5.30565e-3 / 1.52588e-5 | 0.116 (≥ 0.05) | exact in both modes; 4 below-bound steps (0.017–0.034) in the fixed-continuation region |
| `qwen3-8b-4bit` | 0.0 / 0.0 | **0.0 — exact top-2 ties at greedy steps 6 and 10** | greedy match rejected; the identity claim rests on the fixed continuation: 31 of 33 steps hard-matched, 2 tie-eligible |

Tolerances frozen per case in the manifest (`identity_claim` / `identity_evidence`). No
tolerance was widened and no golden regenerated. The 4-bit ties are a property of the
artifact's quantized logits, not a loader defect (diff 0.0).

### 2.5 Prompt generator

Frozen in the manifest (`prompt_generator`): SplitMix64 (golden gamma
`0x9E3779B97F4A7C15`); case initial state = the `z` output of one step from
`0x20261105 ^ (index * gamma)`, indices 0/1/2 for 0.6b/4bit/tinymistral; `token = z %
vocab_size` over the **unsigned** `z`, retrying while in the case stop set; per-length arrays
are nested prefixes 32 | 512 | 2048 of the 2048 stream; the fixed-continuation workload uses
`prefix512[32:64]`; canonical SHA-256 over the compact-JSON UTF-8 encoding of the array.
Re-derivation is enforced by the §2.3 guard, so a future worker reading the manifest cannot
silently use a different stream.

## 3. MoE candidate: TinyMistral-6x248M accepted

Qualified on 2026-11-06 from the on-disk artifact (not from name/size):

- `model_type: mixtral`, `MixtralForCausalLM` layout, 12 layers, hidden 1024, 32 attention
  heads / 8 KV heads (head_dim 32), 6 experts per layer with top-2 routing, vocab 32005,
  context 32768, not tied, **all 303 stored tensors F32**. jmlx loads it through the existing
  `mixtral` family (`Moe(6, 2)`, float32 softmax top-k in `MoeMlp`); Python loads it through
  `mlx_lm`'s `models/mixtral.py` (installed digest §4.3).
- Weight bytes 4,012,290,048 (file 4,012,327,872 B); float32 KV at P=2048+128 decode:
  53,477,376 B. No quantized Mixtral, no `qwen3_moe` — inside the jmlx MoE support gate.
- **Label:** every result for this case must read "TinyMistral-6x248M (6x248M real MoE,
  top-2)", never "Mixtral-8x7B performance". A standard Mixtral-8x7B was **not** qualified:
  the plan's planning estimate (~93.4 GB BF16 weights before overhead on a 128 GiB host,
  float32 exceeding RAM for weights alone) leaves no defensible headroom for a matched
  two-runtime comparison, and the feasibility gate is satisfied by a real, loader-compatible
  Mixtral layout at representative per-token expert work. This is a recorded scope decision,
  not a capacity blocker (plan §3.1).

## 4. Python environment

Dedicated CPython 3.12 environment at `tools/decode-baseline/` (venv gitignored; the other
oracle environments were not touched):

- `requirements.in`: 8 direct pins — `mlx-lm==0.32.2`, `mlx==0.32.4`, `mlx-metal==0.32.4`,
  `transformers==5.22.1`, `Pillow==12.3.0`, `numpy==2.5.3`, `safetensors==0.8.0`,
  `tokenizers==0.23.2`. `mlx-vlm` is deliberately **not** a pin.
- `requirements.lock`: complete hash lock (every wheel SHA-256) covering all 35 installed
  distributions (record: `tools/decode-baseline/installed-distributions.txt`).
- `install.sh`: macOS-arm64 + CPython 3.12 gate, `--require-hashes --only-binary=:all:`,
  `pip check`. `verify.sh`: installed-distribution pin check, imported-path assertions
  (`mlx_lm` from the local venv, default device `gpu`), `mx.device_info()` capture, and —
  since this WP0 — re-verification of all 202 installed `mlx_lm` source digests against the
  freeze below. Both scripts were executed successfully on 2026-11-06 (`VERIFIED`).
- mlx-lm 0.32.2 against MLX/Metal 0.32.4 qualified: the venv loads and generates on the GPU
  device with the same Metal version the Java side stages (§9).

### 4.1 Device (recorded by `verify.sh`, 2026-11-06)

`Apple M5 Max` (`applegpu_g17s`), `memory_size` 137,438,953,472 B,
`max_recommended_working_set_size` **120,259,084,288 B** (≈112 GiB),
`max_buffer_length` 86,586,540,032 B, `resource_limit` 499,000.

### 4.2 Wired-limit policy

The stock Python loop enters `wired_limit` and `maybe_set_recommended_wired_limit` sets the
device's recommended working set (120,259,084,288 B here) when available, restoring the
previous limit on exit. The controlled Python adapter enters no wired-limit context and calls
no setter; Java has no `mlx_set_wired_limit` call in core/models/benchmarks, so the controlled
policy is **`not set` on both sides** (plan §3.2). "Not set" means no explicit setter call;
the effective limit remains `unknown` unless observed independently, and no setter call is
made merely to query it. Stock-mode runs keep the shipped context and record
requested/previous/restored values where observable.

### 4.3 Installed source freeze

`tools/decode-baseline/installed-source-digests.json` records the SHA-256 of all **202**
`mlx_lm` `.py` files installed from the lock (recorded 2026-11-06; `verify.sh` re-checks
them after every install). Key files: `generate.py`
`9d46803799f99d6fc7c35fff2d4357c79b1a25ae98cc63efeaaebeb50a7b5f33`,
`generate_utils.py` `3dba1661ed2c1058…`, `utils.py` `2803ff8d2298b085…`,
`models/qwen3.py` `7e29b7055d8e526a…`, `models/mixtral.py` `a7d15990aa42b81b…`.
`generate.py` is **byte-identical** to the 0.32.2 source inspected in
`tools/qwen35-reference/mlx` on 2026-11-04 (same digest), so the loop-difference analysis in
§5 was re-verified, not re-derived. The hash-locked wheel makes the digest set reproducible
from `requirements.lock` alone.

## 5. Installed mlx-lm 0.32.2 loop semantics (verified facts)

Re-verified against the §4.3 `generate.py` digest; these are the stock-loop differences the
comparison must label (plan §3.2 table):

1. `generate_step` consumes all but the last prompt token as cache-only forwards, then steps
   the last token (default chunk 2048): for every planned P>1 there are ≥2 forwards — not
   Java's single full-prompt last-position prefill.
2. It constructs and `async_eval`s token n+1 before yielding token n: stock TTFT includes
   building/submitting token 2; a max-128 run constructs 128 cached forwards including one
   unused final result (evaluated work is drained at completion and reported, §10).
3. Host graph construction for n+1 overlaps GPU execution of n — a scheduling difference from
   Java's synchronous loop, a potential stock benefit, not an FFM saving.
4. `stream_generate` enters the `wired_limit` context (§4.2).
5. Cache clearing follows every prefill chunk and `n % 256 == 0` (including n=0 after the
   first yield): stock cached-byte counters and allocator reuse differ from a no-clear
   controlled loop.

## 6. Adapter contracts (frozen for WP1 implementation)

- **Python controlled:** custom synchronous loop over the pinned model/cache objects — one
  full-prompt forward with last-position-only output projection (matching Java `stepLogits`),
  then exactly 127 cached forwards for 128 emitted tokens; no next-token lookahead, no
  `async_eval` pipeline, no periodic cache clearing; required logits/cache state evaluated at
  matched boundaries before each step timer stops. Own-head rule: tied Qwen3-0.6B uses the
  embedding's `as_linear`; untied Qwen3-8B-4bit uses the loaded `lm_head` **preserving its
  U32-quantized module** (verified in WP0: the head is a U32-packed tensor with BF16
  sidecars, so no dense substitution is possible or permitted); the same rule applies to the
  MoE head. No upstream source monkeypatching; all custom source hashes are recorded in the
  manifest when implemented (WP1), and the adapter's logits/cache evolution is compared
  against the shipped model on tiny and real inputs before it may run.
- **Python stock:** installed `stream_generate` unmodified; exact prompt IDs, greedy, stop
  set §2.1, max 128 (a maximum, not a target); retains split prefill, lookahead, wired-limit
  context, cache clears and detokenization; the full-generation timer ends with an explicit
  `mx.synchronize()` after the iterator finishes (both `wired_limit` branches); the final
  `GenerationResponse` token is counted exactly once. Labeled `stock-default-loop`.
- **Java controlled:** benchmark-only same-package adapter over `DecoderModel`'s
  `stepLogits`-equivalent path (WP1), qualified against ordinary generation outputs; no
  production API is widened for timing.
- **Java stock:** public `DecoderModel.generate(GenerationRequest, Consumer)` with the
  checkpoint's own `HfTokenizer` attached (incremental detokenization runs and is timed,
  matching the stock loop's always-detokenizing behavior), pretokenized prompt IDs, full
  retention, `greedyDefaults(128, stopSet)`, no extra `stopTokenIds`, minimal event listener.
  The WP0 pilot and clean sweep (§7) exercised exactly this path.

Controlled and stock comparisons run in separate fresh workers/matrices and are never pooled.

## 7. Bounded pilots, cell budgets and the frozen cache policy

Pilot budget used: well under the 2 device-hour WP0 allowance (nine dedicated-process Java
cells + one dd-warm probe + the qualification runs + the original single-process pilot,
Python and Java sides).
Pilot results are planning/qualification evidence, not headline samples.

### 7.1 Clean Java sweep (stock path; one dedicated JVM per cell, TF32=1, warm + 2 measured)

| Case | L | load s | warm total s | measured total s (median) | ms/token (median) |
| --- | --- | --- | --- | --- | --- |
| `qwen3-0.6b` | 32 | 0.054 | 0.982 | 2.31 | 18.1 |
| `qwen3-0.6b` | 512 | 0.046 | 3.615 | 3.35 | 25.9 |
| `qwen3-0.6b` | 2048 | 0.046 | 5.797 | 5.46 | 41.4 |
| `tinymistral-6x248m` | 32 | 0.526 | 2.697 | 2.28 | 17.7 |
| `tinymistral-6x248m` | 512 | 0.284 | 2.884 | 2.42 | 18.6 |
| `tinymistral-6x248m` | 2048 | 0.265 | 3.544 | 3.12 | 22.2 |
| `qwen3-8b-4bit` | 32 | 0.069 | 3.341 | 6.93 | 52.9 |
| `qwen3-8b-4bit` | 512 | 0.064 | 3.817 | 7.64 | 56.9 |
| `qwen3-8b-4bit` | 2048 | 0.067 | 22.259 | **328.71** | **2189.8** |

All 128 tokens, `finish=MAX_TOKENS`, event count = generated count (per-step event emission
confirmed by the event probe, §7.3). Loads are lazy mmap (0.05–0.5 s); first-request lazy
materialization is carried by the warm row, not by `load`.

### 7.2 Python stock pilot (original single-process pilot, TF32=1; external wall totals, medians)

| Case | L | load s | measured total s | stock tps field | peak mem GB |
| --- | --- | --- | --- | --- | --- |
| `qwen3-0.6b` | 32 / 512 / 2048 | 0.355 | 1.65 / 1.81 / 6.37 | 79 / 151 / 21.7 | 1.25 / 1.57 / 2.14 |
| `tinymistral-6x248m` | 32 / 512 / 2048 | 0.381 | 1.34 / 0.51 / 3.33 | 101 / 266 / 53.0 | 5.08 / 5.08 / 5.67 |
| `qwen3-8b-4bit` | 32 / 512 / 2048 | 0.720 | 4.34 / 4.72 / 5.32 | 30.8 / 30.2 / 33.1 | 4.90 / 5.22 / 5.61 |

All `finish=length` (no early EOS on the frozen prompts). The `stock tps` column is the
installed loop's own throughput field, retained as an observation only — it does not replace
the external timing contract. Two Python rows (0.6b-512 m2, tinymistral-2048 m1) show single
outlier runs (3.10 s, 4.44 s) in an otherwise tight pair: expected noise from the
stock loop's n+1 pipeline, recorded, not explained away.

### 7.3 The 4-bit 2048 page-cache regime (finding + frozen policy)

**Finding.** The `qwen3-8b-4bit` L=2048 cell is I/O-sensitive in a way no other cell is:

- Clean sweep (dedicated JVM): the two *measured* requests per cell took 321/336 s
  (TTFT 50.5 s, decode 2190 ms/token), while the *first* request in the same process took
  22.3 s (TTFT 1.47 s, 163 ms/token) and the L=32/512 cells of the same artifact are
  compute-bound (6.9/7.6 s). The fast first request inherited resident checkpoint pages the
  sweep's preceding 4bit-512 cell had just faulted in; the 2048-length run's larger
  KV-cache + prefill-activation footprint then reclaims those 4.6 GB of file pages during
  the run, so each subsequent decode step re-faults at SSD speed. `tinymistral`-2048 (4.0 GB)
  stays compute-bound because its MoE per-step weight traffic and footprint are lower — the
  trigger is the 2048 footprint on this host, not prompt length alone.
- `dd` probes during a slow run read the 4.6 GB checkpoint at 6.05 GB/s then 3.0 GB/s
  (bandwidth halving = concurrent re-faulting from SSD), versus 0.35 s ≈ 13 GB/s for the
  same file when resident.
- A `dd` warm pass immediately before the run dropped TTFT to 1.78 s and started decode at
  ~64 ms/token, which then **degraded to ~285 ms/token within the same request** — the
  checkpoint is not stably resident even inside a primed request.
- The same cell measured 4.75–5.89 s in the original Python pilot (2026-11-05 night), i.e.
  the regime is host/cache-state dependent, not runtime dependent: both loaders mmap the
  same files, and host file-cache trim policy on this machine (≈7 GiB inactive file cache
  observed while 65% of RAM was free and 40 GiB wired) does not keep 4.6 GB stably
  resident. The exact macOS cache-trim cause is a host behavior; it is recorded as a
  limitation, not chased further in WP0, and it is never interpreted as FFM cost.
- `MLX_ENABLE_TF32=0` reproduces the identical slowness: not a precision effect.

**Frozen policy (effective from WP1).** To make the pair comparison cache-regime-matched:

1. Both workers (all 9 cells, both adapters, both runtimes) perform a declared
   **checkpoint-priming read** — one full sequential read of every checkpoint file, its
   duration recorded as `prime_s` and **excluded from every measured interval** —
   immediately before each of the 26 requests. Regime label: `cache_regime:
   primed-per-request`. Priming is uniform across cells so no cell silently runs in a
   different regime; for the 1.5/4.0 GB files the prime is 0.1–0.4 s and changes nothing
   measurable.
2. The `qwen3-8b-4bit` L=2048 cell is additionally labeled `io_affected: true`: even a
   primed request re-faults mid-run (§7.3), so its absolute times are I/O-contaminated on
   both runtimes identically; pair ratios from it are reported with that label, and the
   cell's budget uses the primed measurement (21.1 s/request, dd-warm probe median) rather
   than the unprimed 328 s.
3. One **observed-cache reference pair** (no priming, both runtimes, 1 request each) per
   cell is optional in WP3 to document the raw regime; it is budgeted separately and can
   never substitute for a primed pair.
4. No cold-disk comparison is claimed (plan §3.1); load and first-request times are labeled
   I/O-bound where the evidence says so.

**Event-timing note.** The original single-process pilot's Java `event-step` values were
non-physically consistent at longer prompts (e.g. 0.47 ms/token for 0.6b-2048; 2.7 ms/token
for 4bit-512 — below the memory-bound floor for both artifacts) and are **superseded** by
the dedicated-process sweep and the event probe (128 per-step events, monotonic
timestamps, count = generated). The public `generate` event stream is sound for the
streaming-decode metric of plan §4; WP1 workers must still assert event count = generated
count per request.

### 7.4 Per-cell budgets (committed)

Per adapter, per cell: 12 processes (6 pairs × 2 runtimes) × (1 load + 26 requests);
per-request cost = §7.1 measured median (primed regime), except 4bit-2048 = 21.1 s primed
(§7.3). Process-launch overhead and priming reads are inside the per-request budget.

| Cell | Per-request s | Cell wall (s) | Cell wall (h) |
| --- | --- | --- | --- |
| 0.6b-32 / -512 / -2048 | 2.31 / 3.35 / 5.46 | 720 / 1045 / 1706 | 0.20 / 0.29 / 0.47 |
| tinymistral-32 / -512 / -2048 | 2.28 / 2.42 / 3.12 | 717 / 758 / 978 | 0.20 / 0.21 / 0.27 |
| 4bit-32 / -512 / -2048 | 6.93 / 7.64 / **21.1 (primed)** | 2163 / 2383 / **6585** | 0.60 / 0.66 / **1.83** |

- Per adapter, 9 cells: **≈4.7 device-hours**. Both adapter comparisons: **≈9.5
  device-hours** of headline text generation alone.
- Without the §7.3 priming policy the 4bit-2048 cell alone would be 28.5 h per adapter
  (57 h both) — the policy is what keeps the matrix inside a finite WP3 budget.
- On top of the headline matrix, WP3 additionally budgets (finite, from the same pilot
  rates): qualification re-runs in separate processes (≈1 h/adapter), the fixed-continuation
  workload for 4bit (32 steps × 32 requests/cell, ≈0.3 h/adapter), memory-counter runs
  (≈1 h/adapter, separate processes) and the optional observed-cache reference pairs
  (≈1.2 h both adapters, dominated by 4bit-2048 unprimed 330 s × 2). **Total WP3 text
  reservation: ≤16 device-hours**, to be refined with the implemented controlled adapters
  in WP2 before execution, per plan §4.1.
- Finite limits to carry into the WP1 manifest (launcher rejects missing/nonpositive):
  `worker_timeout = 1800 s` (primed worst cell 21.1 s × 26 ≈ 9.3 min + load/prime/launch
  headroom; unprimed reference pairs get the separate 720 s limit), `cell_time_budget =
  4.0 h`, `session_time_budget = 12 h` (one session ≈ one adapter's 4.7 h plus the
  re-run headroom of §4.1's three-session rule).

## 8. Host and environment record (measurement host)

- Apple M5 Max (128 GiB unified; `memory_size` 137,438,953,472 B), macOS **27.2**
  (build 26B123), `applegpu_g17s`.
- Free disk at recording: 1.0 TiB on a 1.8 TiB volume (43% used). Checkpoint footprint:
  1.5 + 4.0 + 4.6 = 10.1 GB across the three text cases (plus Tier-B VLM artifacts at WP4);
  no hardlink/cache double-count assumption needed at this scale.
- Java: Temurin 25.0.4 (`jdk-25.0.4.1+1`, aarch64). Python: CPython 3.12.16
  (Clang 21), the dedicated venv of §4 only.
- Host state at pilot time: 65% system-wide memory free, largest consumer `omlx-server`
  10.4 GiB, no benchmark/build/serving contention; §4's `hostrec.sh`-convention records are
  still mandatory per worker in WP3 — quiet-host evidence here covers the pilots only.

## 9. Native pin compatibility

- Java staged runtime (`native/install/lib/native-pin.properties`): `mlxMetalVersion=0.32.4`,
  `mlxcCommit=a26f2d584a14ec619ae8db9a09be03becd47598a`.
- Python venv: `mlx==0.32.4`, `mlx-metal==0.32.4` (the same Metal wheel version the Java
  bootstrap stages against).
- **Same MLX/Metal version on both sides — no repin is required** for the main comparison.
  Workers must still read the staged pin properties and imported library paths at start
  (plan §3.2), not trust these strings. The Phase 7.5 old-pin pair (Metal 0.31.2 / mlx-c
  `fba4470`) is WP4-only, in isolated worktrees outside this checkout.

## 10. Proposed output schema `jmlx-decode-baseline-1`

Distinct from the existing `jmlx-decode-benchmark-4` and seam schemas (preserved untouched).
Versioned JSON; every timer monotonic with explicit inclusion/exclusion notes; controlled
and stock samples carry separate `adapter` labels and are never pooled.

```jsonc
{
  "schema": "jmlx-decode-baseline-1",
  "session": {
    "utc_start": "", "utc_end": "",
    "host": {"chip": "", "ram_bytes": 0, "os": "", "os_build": "", "disk_free_bytes": 0,
             "hostrec": {"pre": "", "post": ""}},          // full hostrec outputs, missing-sample status
    "code": {"sha": "", "branch": "", "dirty": false},
    "manifest_sha256": "", "workload": "", "precision_env": "MLX_ENABLE_TF32=1",
    "cache_regime": "primed-per-request",
    "java": {"vendor": "", "version": "", "flags": ""},
    "python": {"venv_python": "", "pip_freeze_sha256": "", "source_digests_sha256": ""},
    "native_pins": {"java": {"metal": "", "mlxc_commit": ""},
                    "python": {"mlx": "", "mlx_metal": ""}},
    "wired_limit_policy": "not set | stock-shipped-context",
    "instrumented": false,                                  // true => ineligible for headline
    "fork_order": ["java", "python", ""]                    // alternating; equal leads
  },
  "workers": [{
    "runtime": "java | python",
    "adapter": "controlled | stock",
    "case": "", "prompt_length": 0,
    "load_s": 0.0, "load_state": "lazy-mmap | materialized",
    "requests": [{
      "kind": "first | warmup | measured",
      "prime_s": 0.0, "prime_excluded": true,
      "ttft_s": 0.0,
      "prefill_tps": 0.0,                       // controlled: last-position projection only
      "steps": [ {"position": 0, "s": 0.0} ],   // controlled: one entry per cached step
      "streaming_decode_tps": 0.0,              // (N-1)/(last_event - first_event); token 1 excluded
      "full_generation_s": 0.0, "emitted": 0, "finish_reason": "",
      "unused_lookahead_forwards": 0,           // stock only; drained at completion
      "memory": {"active_bytes": 0, "peak_active_bytes": 0, "cached_bytes": 0,
                 "rss_peak_bytes": 0},          // separate diagnostic processes; not headline
      "io_affected": false
    }]
  }],
  "summary": {
    "pairs": [ { "pair": 0, "leader": "java|python",
                 "per_metric": { "metric": "", "java_p10": 0, "java_median": 0, "java_p90": 0,
                                 "python_p10": 0, "python_median": 0, "python_p90": 0,
                                 "ratio_median": 0 } } ],
    "first_request": { "metric": "", "java": [], "python": [] }   // separate distribution
  }
}
```

Quantile convention (frozen): linear interpolation (numpy `R`/type-7) over the per-process
warm medians, i.e. percentiles are computed **over the six pair medians, not over 120
requests**; raw per-request values stay in the archived report and are never resampled into
headline statistics. `instrumented: true` reports (WP2 counters/tracing) are structurally
excluded from headline aggregation.

## 11. Deferrals and open items (R3)

1. **Mixtral-8x7B not qualified** — recorded scope decision (§3): no defensible headroom on
   the 128 GiB host; the MoE row is satisfied by TinyMistral-6x248M with its mandatory
   label. Reopening requires a dated owner scope amendment.
2. **VLM artifact sets** are pinned at manifest-file level only; full re-verification and
   the stop-set mismatch resolution happen at WP4 execution.
3. **4bit-2048 root cause** of the host file-cache non-residency is not fully root-caused
   (host behavior, §7.3); the frozen priming policy neutralizes it for the pair comparison
   and the `io_affected` label bounds the claim. If WP3 host records show the primed regime
   is unstable on this host, the cell follows §4.1's unstable/incomplete rules — it is not
   dropped.
4. **Python-side per-cell rates** come from the single original pilot (Python was run once
   per case); WP1's stock worker re-measures every cell, and the §7.4 budget refresh in WP2
   uses the implemented controlled adapters before the WP3/WP4 budgets are committed.
5. **Stock-loop throughput field** (`stock tps`) is retained as observation only; the
   external timing contract is the only headline source.
6. Raw pilot/sweep outputs are archived outside the verified checkout
   (`~/wp0-scratch/`, digests §12); a later documentation PR records their SHAs without
   editing the measured tree.

## 12. Raw evidence

Dedicated-process Java sweep (stock path), `~/wp0-scratch/budget/*.json`:

| File | SHA-256 |
| --- | --- |
| `qwen3-0.6b-32.json` | `669fdcfedd742be51eb508c10ce133f54bce70ff39f37a71c8eb11a25fd08e82` |
| `qwen3-0.6b-512.json` | `7222cd03ee7ea18f2c82383023ede6ad7c535978838ea75177c75bb2cae13da6` |
| `qwen3-0.6b-2048.json` | `5b70b0f803478b27c980b55ba925013bf06b0bef165cec42c53a4531fb9c5484` |
| `tinymistral-6x248m-32.json` | `46b04e80af7382f86eef3d4982628e4430a0afac0c78518963cf22d2a133fad4` |
| `tinymistral-6x248m-512.json` | `9b960cf48e3afd65c2c420036789291b40daf351f2680826fd759243eb7edebf` |
| `tinymistral-6x248m-2048.json` | `3507d163f6d001fe05d0c149e61c5f3cd3749b301341254a60993cd0ab96b165` |
| `qwen3-8b-4bit-32.json` | `cb167d53da739b8a4bd332dac99752de5577239eef51cce6143cfccb45fe8743` |
| `qwen3-8b-4bit-512.json` | `307f8f67d3bd8f40682e659dddbf4f15a363d25bb3da879bdaf2573ea7cbf676` |
| `qwen3-8b-4bit-2048.json` | `3c790fbf23a8c2bf2c89013a74524499bb244844132196c6dd3cc56baad8d805` |
| dd-warm probe log (`4bit-2048-ddwarm-probe.log`) | `89976b4e12bc166bc06f42e13b6d02513113e28fc1008bcfb1e80ab62e798acd` |
| rerun2048 / tf320 / ddprobe re-runs | `58a02faa349cd3bf9c60bb91037d54978e0568c66ed712f452b3abfa989fe004` / `44a91102ce7e76a9a8229967561728cf674fba9b4446e86b2487f7d4a28debdb` / `26c8897ad7ae5c521e2f85247d6b2bf1506a6dbcb92709da7cd2bcab81d3422c` |

## 13. WP1 — Worker protocol and final report contract (frozen)

This section freezes the launcher/worker handoff and the final
`jmlx-decode-baseline-1` record shapes, superseding the §10 sketch where they
differ (amendments listed in §13.4). The Python side of the contract is
implemented in `tools/decode-baseline/worker.py` + `controlled.py` (both
digest-pinned in the manifest's `worker_sources` and re-hashed by the §2.3
guard) and was smoke-verified on this host on 2026-11-06 against the
digest-pinned 0.6b artifact with the full policy request mix (1 first + 5
warmups + 20 measured); the Java side implements the same contract in
`jmlx-benchmarks` (launcher, statistics, validation) and `jmlx-models` test
sources (benchmark-only adapters).

### 13.1 Job file (`jmlx-decode-baseline-job-1`, launcher → worker)

The launcher creates a unique per-worker output directory and writes one job
file per worker. Fields (all required unless noted):

| Field | Value / rule |
| --- | --- |
| `schema` | `jmlx-decode-baseline-job-1` |
| `adapter` | `controlled` \| `stock` |
| `case` | text-case name from the manifest (VLM cases rejected until WP4) |
| `prompt_length` | member of `controlled_generation.prompt_lengths` |
| `mode` | `headline` \| `fixed-continuation` \| `reference-pair` |
| `requests` | `{first, warmups, measured}` nonnegative ints; headline/fixed-continuation must equal `run_policy.first_requests`/`warmups`/`measured_requests`; reference-pair must be exactly `{0, 0, 1}` |
| `timeout_s` | `run_policy.worker_timeout_s` (1800) or `run_policy.reference_pair_worker_timeout_s` (720) for reference-pair; must match exactly |
| `precision_env` | `"1"` \| `"0"` — becomes `MLX_ENABLE_TF32`, set **before any mlx import** |
| `artifact_dir` / `manifest_path` / `manifest_sha256` / `repo_dir` / `out_dir` | paths; `out_dir` is created by the launcher and must not already contain a `report.json` (overwrite rejected) |
| `fixed_continuation` | `{offset: 32, count: 32}` only in fixed-continuation mode; the fed ids are `prefix512[32:64]` per the manifest `prompt_generator` rule |

Worker validation, before any model work: `manifest_path` resolves to the
repository's canonical `tools/decode-baseline/workloads.json` (anchored on the
worker file's own location, so a hand-written job cannot point at a matching
copy); `repo_dir` resolves to the repository the worker runs from (anchored
on the worker file's own location the same way, so the report's code
identity can never name a different checkout); manifest SHA-256 equals the
job; every `files` pin re-verified; the
artifact's `*.safetensors` set equals the case's pinned set exactly (no
unpinned weight file is primed or loaded); venv pins re-checked (installed
`mlx`/`mlx-metal`/`mlx-lm` versions and the full installed `mlx_lm` source
freeze — all 202 pinned files, not just `generate.py` — against
`installed-source-digests.json`); outside reference-pair mode
`requests.first`/`warmups`/`measured` equal `run_policy.first_requests`/
`warmups`/`measured_requests`; the frozen prompt's canonical SHA-256 recomputed
and compared to the manifest pin. Any failure → failed report (§13.3) and exit
1, never a partial success — and never an overwrite of a report the directory
already holds (§13.2).

### 13.2 Worker protocol and output layout

- Before any validation, the worker claims the launcher-created output
  directory with an exclusive-create `worker-claim` file (`{utc, pid}`); if
  the claim already exists, the worker writes nothing into the directory
  (no report, no temp file), reports the refusal on stderr
  (`failure_reason: output folder already claimed by another worker`) and
  exits 1. The claim is sticky: a killed worker leaves it, so a claimed
  directory without a report is dead and a launcher must treat it as such.
  A worker that fails validation later still writes its failed report into
  the directory it claimed itself.
- The worker writes `worker-ready` (`{utc, pid, adapter, case}`) with
  exclusive create after validation, **before** the model load — it is the
  process-startup metric's handshake endpoint (plan §4: parent launch to
  worker-ready); a pre-existing marker fails the run (the folder was
  claimed by this run, so a failed report is written).
- The worker writes `report.json` atomically (a per-process temp file,
  `report.json.tmp.<pid>`, plus a hard link, which fails rather than
  replaces), exactly once, and never overwrites an existing `report.json`:
  a write into an already-reported directory is refused and reported on
  stderr only, leaving the existing report byte-identical — except that a
  *success* whose report link is refused is downgraded to `failed` (exit 1:
  its results are not recorded, so it must not present the run as a
  success). The refusal message is built from the final state and printed
  after that downgrade, so it never reports an `ok` status the run was
  downgraded away from. The per-process temp name keeps a concurrent worker
  in the same directory from truncating it mid-write; a temp-write failure
  removes the temp file before propagating (nothing left behind). The
  check-then-write is guarded by a lock shared with the watchdog backstop.
- Exit codes: 0 ok, 1 failed (validation, contract or unexpected error),
  3 timeout. A timeout or failure never presents a partial result as
  success: the report carries `status` (`ok`/`failed`/`timeout`) and
  `failure_reason`, plus whatever complete request records exist.
- Wall-clock watchdog: a 1 s-poll daemon thread sets a fired event at the
  deadline; the main loop breaks at the next request boundary and writes the
  timeout report. A 30 s grace backstop then writes the report best-effort
  and `os._exit(3)`, so a wedged GPU call cannot hold the cell open. The
  backstop skips the status overwrite when the report is already written (the
  main thread's report stays authoritative); the exit is unconditional.
- The fired deadline is re-checked after the final request: if it fired while
  the last request was running, the run reports `timeout` (exit 3) even though
  that request finished within the grace, with all complete request records.
- Priming (§7.3): one sequential full read of every checkpoint safetensors
  file immediately before each request; `prime_s` is recorded per request and
  excluded from every measured interval (`prime_excluded: true`).
  Reference-pair workers prime nothing (`prime_s: 0.0`, `cache_regime:
  "observed"`).
- The model is loaded once per worker process after the worker-ready marker
  (`load_s`, `load_state`); requests run sequentially over a fresh
  per-request cache.

### 13.3 Worker report record (per-worker file; schema `jmlx-decode-baseline-1`)

```jsonc
{
  "schema": "jmlx-decode-baseline-1",
  "worker": {
    "runtime": "python",                  // or "java"
    "pid": 0,                             // worker process (matches worker-claim / worker-ready)
    "adapter": "controlled | stock",
    "case": "", "prompt_length": 0,
    "mode": "headline | fixed-continuation | reference-pair",
    "status": "ok | failed | timeout",
    "failure_reason": null,
    "utc_start": "", "utc_end": "",
    "identity": {                          // per-worker (moved from the §10 session block)
      "code": {"sha": "", "branch": "", "dirty": false},
      "manifest_sha256": "",
      "run_policy": { "worker_timeout_s": 1800, "reference_pair_worker_timeout_s": 720,
                      "cell_time_budget_s": 14400, "session_time_budget_s": 43200,
                      "first_requests": 1, "warmups": 5, "measured_requests": 20,
                      "emitted_tokens": 128, "cache_regime": "primed-per-request" },
      "precision_env": "MLX_ENABLE_TF32=1",
      "python": {"version": "", "venv": "", "mlx": "", "mlx_metal": "", "mlx_lm": ""},
      "device": {"device_name": "", "memory_size_bytes": 0,
                 "max_recommended_working_set_bytes": 0},
      "wired_limit_policy": "",            // manifest controlled_generation/stock_generation.wired_limit verbatim
      "cache_regime": "primed-per-request | observed",
      "io_affected": false,
      "instrumented": false
    },
    "stop_token_ids": [],
    "load_s": 0.0, "load_state": "materialized",
    "requests": [{
      "index": 0, "kind": "first | warmup | measured",
      "prime_s": 0.0, "prime_excluded": true,
      "ttft_s": null, "prefill_s": null,
      "steps": [ {"position": 0, "s": 0.0} ],   // controlled: one entry per cached step;
                                                // fixed-continuation adds argmax_id, top2_margin
      "event_s": [], "tokens": [],              // host-visible token times / emitted ids
      "streaming_decode_tps": null,             // (N-1)/(event_s[N-1] - event_s[0]); token 1 excluded
      // one value per mode, the other null (uniform key set):
      "full_generation_s": 0.0,                 // headline / reference-pair: wall incl. prefill
      "step_region_s": null,                    // fixed-continuation: step-region wall (below)
      "emitted": 0,
      "finish_reason": "controlled-max-tokens | stop | length | fixed-continuation",
      "unused_lookahead_forwards": 0,           // stock: 1 for any non-empty run
      "memory": null, "io_affected": false
    }],
    "notes": []
  }
}
```

Boundary semantics (frozen): controlled emits all `emitted_tokens`
regardless of stop-set membership (`stop_termination: false`, finish
`controlled-max-tokens`) with a single full-prompt forward, last-position
projection, and exactly `emitted - 1` cached steps (step `position` = the
1-based position of the fed token — the cache length after the step — so
`prompt_length + 1` … `prompt_length + 127`). Controlled `event_s` records
every token's host-visible time after its argmax (token 1 included);
`prefill_s` stops at the evaluated prefill logits. Stock runs the shipped loop with the manifest stop
set (early `stop` possible), reports host-visible event times only (its
internal prompt/decode fields stay stock metrics), and drains the one queued
lookahead forward with an explicit `mx.synchronize()` after the iterator —
counted, not hidden. Stock records `prefill_s`/`steps` as null; the
controlled-only measurements are the prefill/decode contract of plan §4.
Fixed-continuation feeds `prefix512[32:64]` ids one per step (no sampling,
no lookahead pipeline; stock uses the shipped model call with a
last-position slice) and records per-step `argmax_id`/`top2_margin`; it has
no `event_s`/`tokens`/`ttft_s` and `emitted: 0`. In both adapters the
per-step `s` stops at the evaluated logits (the argmax and the
full-vocabulary sort run outside the measured interval), the prompt cache is
created before the timer starts, and fixed-continuation records
`step_region_s` (with `full_generation_s` null) — the monotonic wall time
over the step region only (from the prefill evaluation to the end of the
final step's host work), prefill excluded, one definition on both paths.
The step region is **not** decode time: it includes the per-step
full-vocabulary sort-to-host and argmax check work (about 39-46% of the
value on the 0.6b L=512 cell, item 14), so decode comparisons use the
per-step `s` values. The other modes keep `full_generation_s` with its
prefill-inclusive meaning (`step_region_s` null). Fixed-continuation
`prefill_s` is per-adapter and not comparable
across adapters: the stock shipped call projects the output head at every
prompt position (the full [L, vocab] score tensor, about 155 MB at L=512),
while the controlled prefill projects only the last position. `memory` stays
null in
WP1 (WP2 diagnostic processes fill it; `instrumented` workers are
ineligible for headline aggregation regardless).

The **session report** is assembled by the launcher (Java side): the §10
`session` block **minus** the identity fields that moved per-worker
(`code`, `manifest_sha256`, `run_policy`, `precision_env`, `python`,
`device`, `wired_limit_policy`, `cache_regime`, `io_affected`,
`instrumented` — the session keeps `host`, `java`, `native_pins`,
`workload`), plus `fork_order`, the embedded per-worker reports, and the
`summary` with its exclusion list (failed/timeout/instrumented workers and
requests are excluded by construction, with the exclusion recorded). The
§10 quantile convention is unchanged.

### 13.4 Amendments to §10 and verified WP1 facts (2026-11-06, this host)

1. Identity block moves from the session to each worker report; the session
   retains only host/fork-order/native-pin/workload fields (§13.3).
2. Per-request fields gain `index`, `event_s`, `tokens`, `ttft_s`,
   `prefill_s`, `full_generation_s`, `finish_reason`,
   `unused_lookahead_forwards`, `io_affected`; `prefill_tps` is replaced by
   `prefill_s` (tps is derived in the summary, never stored per request).
3. `load_state` is recorded at the worker (Python: `materialized` — the
   pinned `load()` evaluates all parameters; Java records its own state).
4. Stop-set resolution (correcting an earlier probe that called
   `load_tokenizer` directly): the pinned `mlx_lm.load()` merges
   `generation_config.json`'s `eos_token_id` into the tokenizer config
   (`utils.py:376`), so all three text cases already resolve to exactly the
   manifest sets (0.6b → {151643, 151645}, tinymistral → {2, 32001}, 4bit →
   {151645}); verified by direct load on 2026-11-06. The stock worker still
   completes any missing ids via the shipped `add_eos_token` API (the
   `--extra-eos-token` mechanism) and **asserts** the resolved set equals
   the manifest set, so a future drift fails the worker instead of
   changing behavior silently.
5. Greedy-stream agreement is **not** an invariant between the two adapters:
   on the frozen 0.6b L=32 prompt the controlled and stock streams match
   for the first 102 tokens and diverge at token 103 (top-2 margin 0.125 at
   the diverging step; 4 of 127 steps had margin < 1e-2, including exact
   0-ties). The cause is the different forward structure (single full-prompt
   forward vs shipped split-prefill + lookahead), not a load defect — the
   §2.4 logit-level identity (max diff 0.0, greedy-region min gap 0.5) is
   unaffected. Headline tables therefore compare per-adapter timings, never
   pooled, per plan §4.
6. Fixed-continuation cross-check on the same artifact: 26 × 32 steps per
   adapter, positions 513–544, **zero argmax mismatches** between controlled
   and stock including the exact-tie steps (first-maximum argmax tie-break
   is deterministic in both paths).
7. Metal first-use warm-up: the first process to exercise a shape pays a
   one-time compile cost (observed: fixed-continuation L=512 request-0
   prefill 0.332 s vs median 0.022 s; the later stock process's request-0
   was 0.036 s). This is why the 5 discarded warmups exist; it is a note,
   not a contract number.
8. `mx.sort` in the pinned mlx 0.32.4 sorts ascending with no order keyword;
   `mx.default_device() == mx.gpu` holds on this host (asserted by the
   worker before load). The worker runs under `python -I` (isolated mode),
   which keeps the script directory off `sys.path` — the worker appends its
   own directory (not prepends, so a stray sibling file cannot shadow stdlib
   or installed packages) for the sibling `controlled` import.
9. Review round 1 (2026-11-06, PR review of head `4ef0503`): (a) a refused
   job can no longer destroy the existing report — `write_report` hard-links
   the temp file (`FileExistsError` instead of replace) and the watchdog
   skips the status overwrite once the report is written, the
   check-then-write being lock-guarded; (b) the fired deadline is re-checked
   after the final request (§13.2); (c) the artifact's safetensors set must
   equal the pinned set exactly (§13.1); (d) `run_policy.first_requests: 1`
   is frozen in the manifest, the guard and the worker (§13.1/§13.3);
   (e) the full 202-file installed-source freeze is re-hashed per run;
   (f) `manifest_path` is pinned to the canonical repository file; (g) the
   fixed-continuation step timers and `full_generation_s` are defined
   identically in both adapters, and controlled token 1's event time is
   taken after its argmax (§13.3); (h) the worker appends its directory to
   `sys.path` (item 8). The manifest was re-pinned (new digest in §2;
   `worker_sources` moved with the worker edits) and the guard backstop
   updated.
10. The T2 timeout probe (item 12's method) found a real bug in the
    round-1 watchdog rewrite: the fired-event set sat behind a loop whose
    only exit condition was the event itself, so no timeout could ever fire
    (every timed run would silently report `ok`). After the fix
    (set-and-break inside the poll loop), the probe — a temporary
    real-manifest tamper with the full 26-request mix on the 0.6b L=512
    cell — placed the deadline inside request 25 (in-loop break, 25
    complete records, exit 3) and inside the final request (post-loop
    check, 26 complete records, `status: timeout`, exit 3); the manifest
    was restored byte-identical (SHA-256 verified) and the guard re-passed.
11. Worker-level negative tests (this host, 2026-11-06; 13 assertions,
    validation failures only): a pre-seeded `report.json` survives a refused
    job byte-identical (exit 1, the refusal reason on stderr, no temp file
    left behind); a validation failure into an already-reported directory
    leaves the report intact; a stray `model-extra.safetensors` in a cloned
    artifact dir is rejected by name; `requests.first ≠ 1` is rejected; a
    non-canonical manifest path with a matching digest is rejected.
12. Smoke re-run after the fixes (0.6b, both adapters, full 26-request mix,
    2026-11-06, warm on-disk Metal cache): fixed-continuation L=512 —
    26 × 32 steps per adapter, positions 513–544, zero argmax mismatches,
    max top-2 margin difference < 1e-6; step-s medians 4.46 ms (controlled)
    / 3.95 ms (stock) with the sort outside the timer; `full_generation_s`
    medians 0.2489 s (controlled) / 0.2401 s (stock). Headline L=32 —
    controlled: 26/26 emitted 128, measured-request medians ttft 5.33 ms,
    prefill 4.92 ms, 255.5 tok/s streaming, 0.5027 s full; stock: 26/26
    emitted 128 (no early stop this run), 0.4253 s full median. Item 5's
    stream-divergence fact and the §2.4 logit-level identity are unaffected
    (same artifact, same adapters, same frozen prompts).
13. Review round 2 (2026-11-07, PR review of head `a2f580d`): (a) the
    report's code identity can no longer name a different checkout —
    `repo_dir` must resolve to the repository the worker runs from,
    anchored on the worker file's own location exactly like the
    `manifest_path` check (§13.1); the medium finding was reproduced with
    a throwaway `repo_dir` (the run finished normally and reported the
    other repository's commit as clean) and is now rejected (T7, with an
    empty `identity.code`). (b) Fixed-continuation `full_generation_s` is
    now the step-region wall time with the prefill excluded, one
    definition on both adapters, the stock prompt cache is created before
    the timer starts, and `prefill_s` is documented per-adapter / not
    cross-adapter comparable (§13.3, superseding item 12's wall-time
    wording for this mode). (c) Report writing hardened: per-process temp
    name, a success whose report link is refused is downgraded to
    `failed` (exit 1), and a temp-write failure removes the temp before
    propagating (§13.2). W1-W4 on the worker module: refusal into an
    already-reported directory returns False with the old report
    byte-identical and no temp behind; a success returns True; a second
    call returns False with the report unchanged; a write failure
    propagates with no temp and no report. The manifest was re-pinned
    twice in this round (final digest in §2; the second after the
    refusal-message wording); the guard backstop, the negative tampers
    (8/8, byte-identical restore) and the worker negative suites (13 +
    14 assertions) re-passed on the final head. The T2 timeout probe
    (item 10's method, with self-correcting deadline placement) re-ran on
    the final head: in-loop break at T=7 s (22 complete records, exit 3)
    on the fixed-continuation cell and post-loop check at T=15 s (26
    complete records, `status: timeout`, exit 3) on the headline cell;
    both exits via the main thread (wall < T + 15 s; the 30 s backstop
    would be at ≥ T + 29 s). The probe also measured the fired lag on
    this host: the fired event lands 0.5-1.4 s after the integer-second
    deadline, because the watchdog thread's first poll is delayed by the
    main thread's `mlx_lm` import — the 1 s-poll design bound is
    unchanged, the lag re-rolls per process, and the placement
    self-corrects on top of it. The manifest was restored byte-identical
    (SHA-256 verified) and the guard re-passed. Round-2 smoke numbers
    (0.6b, both adapters, full mix, warm Metal cache):
    fixed-continuation L=512 step-s medians 4.24 ms (controlled) / 3.90 ms
    (stock), `full_generation_s` (step region) medians 0.2219 s (sum of
    steps 0.1356 s) / 0.2316 s (sum 0.1255 s); headline L=32 controlled
    measured medians ttft 11.81 ms, prefill 11.55 ms, 257.4 tok/s; stock
    26/26 emitted 128; fixed-continuation still zero argmax mismatches
    with positions 513–544 unchanged. `verify.sh` and root `check` green
    on the final head.
 14. Review round 3 (2026-11-07, PR review of head `602016e`):
    (a) Reproduced: a second worker failing validation could take over a
    running worker's output folder — a valid worker A started, then an
    invalid job B (`first: 0`) pointed at A's folder wrote its failed
    report in, and A exited 1 after finishing all 26 requests with its
    results lost, while its refusal message still said "run ended as ok"
    (the downgrade happens after the print). Fix (§13.2/§13.3): each
    worker claims the launcher-created directory with an exclusive-create
    `worker-claim` (`{utc, pid}`) **before** validation; on an existing
    claim the worker writes nothing into the directory, reports
    `failure_reason: output folder already claimed by another worker` on
    stderr and exits 1; the claim is sticky (a killed worker leaves it —
    a claimed directory without a report is dead for the launcher). The
    report's worker block gains `pid` (a launcher can now tell the report
    came from the process it started), `worker-ready` is created with
    exclusive create, and the report-link refusal message is built from
    the final state and printed after the success downgrade. (b)
    Fixed-continuation's mode-dependent `full_generation_s` is renamed
    `step_region_s` (with `full_generation_s` null there; the other modes
    gain `step_region_s` null — uniform key set, one value per mode), and
    §13.3 states that decode comparisons use the per-step `s` values,
    since the step region includes the per-step full-vocabulary
    sort-to-host/argmax check work (about 39-46% of the 0.6b L=512
    value), not decode. Verification: the manifest was re-pinned (digest
    in §2; `worker_sources` moved with the worker edits), the guard
    backstop updated; the 8 tamper negatives re-passed with
    byte-identical restore; the worker negative suites re-passed (13 +
    14 assertions — the round-1/2 stderr assertions survive the
    post-downgrade message, which now quotes the final
    `failure_reason`); the new round-3 suite passed 26 assertions: R1 the
    reviewer repro (A exit 0 with 26 records, report `pid` == claim `pid`
    == ready `pid`; B exit 1 "already claimed", no report, no temp, claim
    byte-identical); R2 pre-seeded claim + invalid job (exit 1, nothing
    written, claim byte-identical); R3 pre-seeded `worker-ready` + valid
    job (exit 1, `FileExistsError` failure reason, failed report written
    since the folder was claimed by the run itself, ready byte-identical);
    R4 module-level (refusal sets the flag and prints nothing, a stale
    flag is reset at the call start, the message carries the
    post-downgrade final status). The T2 timeout probe re-ran 11/11,
    which exposed and fixed a **probe** placement bug (the worker was
    correct): the probe aimed the integer-second deadline as if it sat at
    `T` in the `utc_start` origin, but the worker sets the deadline right
    after the `worker-ready` marker (before the `mlx_lm` import and model
    load), so the deadline is at `ooff + T` (`ooff` ≈ 0.58 s here), and
    the mid-run fired event lands essentially **at** the deadline — the
    watchdog polls are phase-locked to the anchor (polls at
    anchor+1, anchor+2, …; deadline = anchor+T), and item 13's 0.5-1.4 s
    lag applies only to the first poll, which the main thread's import
    delays. The headline MIX cell's final-request window (~0.58 s, at
    [15.8, 16.4] s this session) therefore contained no integer-second
    deadline at all (T=15 → 15.57 s, T=16 → 16.57 s), so the round-2
    T=15 pass was a host-speed phase coincidence, not a stable placement.
    The probe now aims in T-space (windows shifted by `ooff`) and
    exercises the post-loop branch on the reference-pair
    single-request cell, whose request starts at ~1.4 s so T=1 sits
    inside its window across normal host-speed variation (the post-loop
    branch under test is identical): in-loop break at T=7 s (22 complete
    records, exit 3, wall 7.83 s) on the fixed-continuation cell and
    post-loop check at T=1 s (1 complete record, `status: timeout`, exit
    3, wall 2.05 s) on the reference-pair cell; both via the main thread
    (wall < T + 15 s; the 30 s backstop would be at ≥ T + 29 s). The
    manifest was restored byte-identical (SHA-256 verified) and the guard
    re-passed. Round-3 smoke (0.6b, both adapters, full mix, warm Metal
    cache, 30 cross-checks): fixed-continuation L=512 step-s medians
    4.24 ms (controlled) / 3.89 ms (stock); `step_region_s` medians
    0.2178 s (sum of steps 0.1355 s, host-work share ≈ 38%) / 0.2307 s
    (sum 0.1248 s, ≈ 46%); headline L=32 controlled measured medians
    ttft 5.24 ms, prefill 4.89 ms, 255.6 tok/s; stock 26/26 emitted 128;
    fixed-continuation zero argmax mismatches with positions 513–544;
    report `pid` == claim `pid` == ready `pid` on all four reports.
    `verify.sh` VERIFIED and root `check` green on the final head.
 15. Review round 4 (2026-11-07, PR review of head `94fea72`):
    (a) Reproduced: with the round-3 claim created before validation, a
    missing output folder (or an out_dir that is a file) failed with a
    raw `FileNotFoundError` from the claim create in the catch-all
    handler, and the old `out_dir must be created by the launcher` check
    in `validate` became dead code. Fix: `main` now checks
    `Path(raw_out_dir).is_dir()` before claiming and raises the old
    message as a `WorkerError` (stderr "worker validation failed: out_dir
    must be created by the launcher", exit 1, nothing written — the
    directory is absent or not a directory, so there is no report target);
    the dead `validate` check was removed and the §13.2 claim bullet
    records the rejection. Verified by new negative tests T8 (missing
    dir: exit 1, the launcher message on stderr, no `FileNotFoundError`,
    nothing written) and T9 (file in the way: exit 1, same message, the
    file byte-identical, no report). (b) Nit: two comments still called
    the step region "the comparable quantity" (controlled.py and the
    stock fixed-continuation branch in worker.py), contradicting the
    updated §13.3, and the controlled.py docstring still named the value
    `full_s` while the report field is `step_region_s` — all reworded/
    renamed in the same round (the `controlled.py` rewording rides this
    round's re-pin rather than waiting for a later worker change).
    Verification: the manifest was re-pinned (digest in §2;
    `worker_sources` moved with both worker edits), the guard backstop
    updated; the 8 tamper negatives re-passed with byte-identical
    restore; the worker negative suites re-passed (20 + 14 assertions);
    the round-3 suite re-passed (26 assertions); the T2 timeout probe
    re-ran 11/11 (in-loop break at T=8 s — landed on the first attempt
    under the corrected T-space aim — 25 complete records, exit 3;
    post-loop check at T=1 s, 1 complete record, `status: timeout`, exit
    3; both via the main thread, wall < T + 15 s); the manifest was
    restored byte-identical and the guard re-passed. Round-4 smoke
    (0.6b, both adapters, full mix, 30 cross-checks) re-ran green:
    fixed-continuation zero argmax mismatches with positions 513–544,
    report `pid` == claim `pid` == ready `pid` on all four reports.
    `verify.sh` VERIFIED and root `check` green on the final head.

### 13.5 Manifest re-pin flow (frozen)

The manifest now carries `run_policy` (the frozen §7.4 limits, exact
nonpositive-rejecting values) and `worker_sources` (SHA-256 of
`worker.py` + `controlled.py`; path confined to `tools/decode-baseline/`).
Any worker edit must be followed by: regenerate the manifest (the
`worker_sources` pins move), update the guard's digest backstop and the §2
digest citation, re-run `verifyDecodeBaselineManifest` and the negative
tampers (a worker-file edit without manifest regeneration must fail with
"worker source SHA-256 differs"). The current frozen manifest digest is the
one cited in §2.
