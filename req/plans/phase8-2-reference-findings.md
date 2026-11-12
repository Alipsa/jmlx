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
| `instrumentation` | optional (WP2): `none` \| `binding-count` \| `tracing`; absent means `none`, and the launcher writes it only when not `none`, so headline job files stay byte-identical to the WP1 shape; python workers are never instrumented (the frozen Python workers carry no diagnostics) |
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
    "instrumentation": "none",             // echo of the job field (WP2): none |
                                           // binding-count | tracing
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
      "instrumented": false,               // == instrumentation != none; instrumented workers
                                           // are excluded from headline aggregation by
                                           // construction
      "host": {                            // present IFF instrumented (WP2): the per-worker
                                           // host record, captured by the diagnostic process
                                           // itself; frozen Python and uninstrumented Java
                                           // workers never carry it
        "hw_model": "", "os": "", "os_build": "",
        "disk_free_bytes": 0, "system_load_average": 0.0,
        "vm_stat": "", "top_processes": []
      }
    },
    "binding_signatures": {},             // binding-count only (WP2): binding name -> the
                                          // generated static wrapper's JVM descriptor
                                          // (agent-recorded at class-load time; the FFM
                                          // transition estimate groups counts by these
                                          // signature classes)
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
      "binding_counts": null,             // binding-count only (WP2):
        // {phases: [{phase, position, counts}], total} — counts are the
        // attempted downcalls per binding (the total over every window
        // must be non-empty or the run fails)
      "tracing": null,                    // tracing only (WP2):
        // {request_wall_s, request_cpu_s, windows:
        // [{phase, position, wall_s, cpu_s}],
        //  residual_wall_s, residual_cpu_s} — residual is the adapter
        // call's span minus the sum of the windows; the windows never
        // change the work, scope lifetimes or timer boundaries
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
while the controlled prefill projects only the last position. `memory` is
filled for instrumented requests only (WP2): `{active_bytes,
peak_active_bytes, cached_bytes, rss_peak_bytes}` — the first three are
sampled after the request (the pre-request `MLXMemory.resetPeak()`
excludes model load), and `rss_peak_bytes` is the cumulative maximum of
the post-request RSS samples (a documented lower bound on the true peak);
uninstrumented requests keep it null. The WP2 instrumentation windows
(plan section 5.1) are strictly additive: binding-count mode records each
window's attempted-downcall counts as `binding_counts`, tracing mode each
window's wall and caller-thread CPU with the unassigned residual as
`tracing`; a mid-run counting-agent failure fails the run, and
`instrumented` workers are ineligible for headline aggregation
regardless.

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
 16. Review round 1 (2026-11-08, PR #89 review of the Java WP1 head
     `c80cdfa`; one Medium, five Low, nits and two suggestions — each
     closed with a test; the implementation record is section 14.7). The
     round amended the frozen contract in four places:
     (a) **Medium, reproduced**: the session summary pooled adapters — the
     reviewer's mixed fake-worker reproduction (the e2e mini case, not a
     0.6b run) had a headline median TTFT of 0.535 s, between
     controlled's ~0.03 s and stock's ~1.03 s, because the cell key was
     `<mode>/<case>/<length>`. The key is now adapter- and mode-scoped on
     both `per_runtime` and `first_request`
     (`<adapter>/<mode>/<case>/<length>`), so no two adapters — and no two
     modes — ever pool.
     (b) The stock stop-set assertion (item 4) now unions the tokenizer's
     own special-token eos id: the tokenizer's eos is part of the
     artifact-resolved set because the manifest derives its frozen set
     from the artifact's stop sources, tokenizer included (the stock path
     feeds the generation loop only the manifest frozen set — the loop
     checks only that, not the tokenizer); the tokenizer loads before the
     assertion.
     (c) The stock full-generation boundary is the `generate` return
     itself: the synchronous Java loop evaluates every step at its
     selection boundary and queues no lookahead, so nothing is pending at
     return — no drain evaluate exists or is needed (item 12's drain
     wording applies to the Python stock path only).
     (d) Fixed-continuation cells gain `step_s` — the contract's decode
     comparison (§13.3) — quantiled per step over the measured
     requests' 32 frozen positions each; first-request FC cells carry
     only the request-level metrics.
     Plus the worker/launcher hardening the review required:
     (e) the worker's hard watchdog is per-run — each `run` increments a
     process token under the report lock together with the fired/finished
     reset, and the watchdog re-checks its token before every remaining
     action (loop exit, fire, post-grace status, report write, exit
     hook), dropping silently when a later run owns the shared state;
     verified by a two-run in-process test (run A wedged past its
     deadline with its watchdog in the grace period, run B started in the
     same process: B exits 0 with its own ok report, the exit hook never
     fires, and A — whose resumed request hits the thread-confined scope
     the later run owns — leaves no report, the expected in-process
     artifact of the one-run-per-process design).
     (f) The launcher cross-checks every ok report against the job it
     launched (runtime/adapter/case/prompt_length/mode, the report's own
     `manifest_sha256`, `precision_env`, and the request-kind sequence —
     an ok report must carry exactly the job's mix, a failed report at
     most a complete prefix); a mismatch is `invalid-report` with the
     violations, never silently aggregated.
     (g) Launcher limits: the fork wait limit is
     `min(timeout_s + 90 s backstop, cell_time_budget_s)` with a 10 s kill
     settle (a killed worker is classified from its left artifacts with
     the kill in the exclusion note); the session loop checks elapsed
     against `session_time_budget_s` before each fork and records the
     remainder `not-started` with the budget reason; `run_policy` is
     validated without defaults (all eight fields integers, the four
     timeouts/budgets > 0, first/measured/emitted ≥ 1, warmups ≥ 0) and
     the job builder reads it default-free; the launcher pins the frozen
     manifest digest and refuses a checkout whose `workloads.json` digest
     differs. The Python side is untouched (frozen, PR #87).
  17. Review round 2 (2026-11-08, PR #89): two test-only findings and
     two wording nits — no contract, schema or measurement-path change,
     so no opt-in session re-run is required (the section 14.6 session
     'd' numbers, recorded on the pre-round-2 head, stay valid):
     (a) The stale-watchdog two-run test relied on a sleep margin (run A
     wedged in a 10 s request delay; on a slow runner A's resumed
     request woke before B wrote its report, and A's failed report from
     its error handler — targeted at the process-static out_dir, i.e.
     B's directory — poisoned B's cell, so B's own report write was
     refused and B returned failed). A's wedge is now a one-shot
     `requestHold` latch consumed by the held run's first request: the
     test orders everything by bounded polls (worker-ready, latch
     consumption, watchdog fire) and releases A only after B's report
     is on disk, so no part of A's post-release failure handling can
     race B's write.
     Two read seams (`requestHoldConsumedForTest`, `hasFiredForTest`)
     make the ordering observable without timing assumptions.
     (b) `resetRunState()` ran before the run-token increment, outside
     the lock: a stale watchdog whose grace action landed in that gap
     saw the fresh reset state under the old token and could mark a
     healthy new run timeout. The reset now runs inside the same
     `REPORT_LOCK` critical section, after the increment. In-process
     only — production is one run per process.
     Plus wording nits:
     (c) The stop-set record stated the tokenizer's eos is in the
     resolved set because the stock path feeds the `HfTokenizer` stop
     set to the generation loop; the stock path feeds the loop only the
     manifest frozen set (item 16(b) and the assertion javadoc are
     corrected in place).
     (d) The section 14.7 Medium bullet attributed the pooling
     reproduction to "the mixed 0.6b run"; it was the fake-worker e2e
     reproduction over the tiny mini case (item 16(a) and the bullet
     are corrected in place, numbers unchanged).
     The session budget overrun (a worker forked just before
     `session_time_budget_s` runs to its own wait limit, so the session
     wall can exceed the budget by at most that limit) is the
     documented behavior and stays as designed; the launcher javadoc
     now says so explicitly.
  18. Review round 3 (2026-11-08, PR #89): two test-only findings — no
     contract, schema or measurement-path change, and no session
     re-run is required:
     (a) Item 17(a) mischaracterized the round-2 race twice (corrected
     in place): on a slow runner A's resumed request woke before B
     wrote its report, and A's poisoned report was a failed report
     from its error handler (its resumed request throws on B's live
     scope, which is confined to B's thread, so the post-loop timeout
     path is never reached), not a timeout report — the failed report
     landed in B's directory, B's own write was refused, and B
     returned failed.
     (b) A failed check before the latch release left run A blocked on
     the latch for the rest of the test JVM: the cross-run test now
     releases the hold in a try/finally (and run A is a daemon
     thread), and `resetTestSeams` releases any outstanding latch, so
     a failed check can no longer strand a held thread between tests.
  19. Review round 4 (2026-11-08, PR #89): one wording finding — no
     code, contract, schema or measurement-path change, and no
     session re-run is required:
     (a) Item 18(a) and section 14.9 said A's resumed request in the
     old race "throws on the closed scope"; that is true only in the
     current test, where B has already finished. In the old race B
     was still running, so the scope was B's live one and only B's
     thread could use it (thread confinement, as the round-1 test
     comment said). Corrected in place; the conclusion (A's request
     throws, the post-loop timeout path is never reached) is
     unchanged.

  20. WP2 (2026-11-08): the instrumentation, host-observation and
     FFM-calibration contract amendments (plan section 5.1; the
     implementation and gate evidence are in section 15):
     (a) The job file gains the optional `instrumentation` field
     (`none` | `binding-count` | `tracing`; absent means `none`; the
     launcher writes it only when it is not `none`, so headline job
     files stay byte-identical to the WP1 shape). Python workers are
     never instrumented — the frozen Python workers carry no
     diagnostics — and a binding-count Java worker without the
     jmlx-core binding-agent jar is a spec failure. The worker report
     echoes the field in its `worker` block.
     (b) The worker identity gains, present iff instrumented, the
     `host` block (hw_model, os, os_build, disk_free_bytes,
     system_load_average, vm_stat, top_processes) — the diagnostic
     processes capture their own host state; frozen Python and
     uninstrumented Java workers never carry it.
     (c) The worker block gains `binding_signatures` (binding-count
     only): binding name -> the generated static wrapper's JVM
     descriptor, recorded by the binding-count agent at class-load
     time.
     (d) The request record gains `binding_counts` (binding-count
     only) and `tracing` (tracing only), null in the other modes;
     uninstrumented requests carry neither field. The binding-count
     total over every phase window must be non-empty or the run
     fails.
     (e) `memory` is filled for instrumented requests only (the WP2
     memory rule replacing the WP1 "stays null" sentence);
     uninstrumented requests keep it null.
     (f) The instrumentation windows (the
     `BaselineAdapters.PhaseObserver` seam) are strictly additive —
     they never change the work, scope lifetimes or timer
     boundaries. Controlled and fixed-continuation expose
     `prefill-construction` / `prefill-sync` /
     `prefill-selection` at position 0 and `construction` /
     `sync` / `selection` per cached step at `prompt_length + k + 1`;
     stock exposes one `request` window at position 0 (a documented
     controlled-first limitation).
     (g) The session block gains `orchestration` and `native_cache`,
     and `fork_order` carries the full cell identifiers (the worker
     output directory names, the instrumentation suffix included)
     instead of the runtimes. The worker output directory name gains
     the `-<instrumentation>` suffix when instrumented, every forked
     Java worker runs with `-Djmlx.native.cache.path` at the
     session's native cache, and the cache is retained on a
     successful session and deleted on any failure.
     (h) A `ok` instrumented worker is excluded from headline
     aggregation by construction, with the exclusion reason
     `instrumented` recorded.
     (i) The FFM calibration (a calibrated estimate, not a causal
     partition): the no-op C library
     (jmlx-benchmarks/src/main/resources/ffm_cal.c) carries one
     non-elidable no-op per dominant generated-wrapper shape and is
     compiled under the output directory with `cc -shared -O2` (no
     production native artifact); `FfmCalibration` records the
     compiler, flags and source/binary SHA-256s in build-info.json
     (schema jmlx-ffm-cal-build-1) and pools the fresh-JVM raw
     samples (schema jmlx-ffm-cal-raw-1) into per-shape
     p10/median/p90 with min/max and every raw sample retained
     (ffm-calibration.json, schema jmlx-ffm-cal-1);
     `FfmTransitionEstimate` maps a binding-count report plus a
     calibration summary to the per-decode-step FFM transition budget
     (schema jmlx-ffm-transition-estimate-1) with coverage, the
     uncovered signatures and the explicit assumptions.

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

## 14. WP1 — Java workers, launcher and session report (2026-11-07)

`phase8-2-wp1-java` implements the Java side of the frozen section 13 contract:
benchmark-only adapters and the process worker in `jmlx-models` test sources
(`BaselineAdapters`, `BaselineWorker`, `BaselineWorkerTestSupport`, driven through the
`GenerationRequest.pretokenized` seam), and the launcher plus the job/report/statistics
contract in `jmlx-benchmarks` (`BaselineBenchmark`, `BaselineJob`,
`BaselineReportValidator`, `BaselineStatistics`), exposed through the opt-in
`:jmlx-benchmarks:benchmarkBaseline` task (documented in `jmlx-benchmarks/README.md`).
The Python workers of PR #87 are the reference implementation; the Java code mirrors the
frozen contract, and the deviations below are the only intentional departures.

### 14.1 Java-specific deviations from the Python worker

- (a) JDK 25 removed `System.setenv` (deprecated for removal): the Java worker cannot
  set its own `MLX_ENABLE_TF32` after JVM start. The launcher therefore owns the
  environment on both worker processes (`builder.environment().put` before the fork),
  and the Java worker *verifies* the process environment equals the job's
  `precision_env` before any native initialization, failing the run on a mismatch
  (the Python worker keeps setting it from the job file, where it is a no-op
  equivalence).
- (b) `Files.move(ATOMIC_MOVE)` is `rename(2)` and silently replaces an existing
  destination: the no-replace report guarantee the Python worker gets from `os.link`
  is instead an explicit `Files.exists(report.json)` refusal under the worker's
  `REPORT_LOCK`, around the per-process temp-file-then-move (the report is written at
  most once; a success whose report write is refused is downgraded to failed).
- (c) Identity/label mapping: `identity.device` comes from sysctl (`hw.model`,
  `hw.memsize`) with `max_recommended_working_set_bytes: 0` plus a note (the Java
  worker has no MLX device-info facade); `identity.java` is
  `{version, vendor, flags}` with `flags` the space-joined
  `ManagementFactory.getRuntimeMXBean().getInputArguments()` — the same source
  the launcher's session block uses (review round 1, §13.4 item 16); the
  frozen finish-reason labels map
  EOS/STOP_TOKEN→`stop`, MAX_TOKENS→`length`, CANCELLED→`cancelled`; Java stock
  reports `unused_lookahead_forwards: 0` (the Java loop has no queued lookahead) and
  `load_state: "materialized"`; `instrumented: false` and `wired_limit_policy` are
  carried from the job.

### 14.2 Validator contract rules pinned by the launcher tests

The pure-Java `BaselineReportValidator` mirrors frozen section 13.3 and, in addition,
these adapter-aware shapes — each was caught by a fixture-driven test before the rule
existed:

- stock headline/reference-pair requests carry `steps: null` and `prefill_s: null`
  (the shipped loop does not expose the controlled prefill/decode contract; the
  Python worker writes the same nulls, worker.py:516-517); controlled requests
  require a numeric `prefill_s` and a `steps` array of exactly `emitted - 1`
  entries;
- the python-stock "drain one queued lookahead iff the run is non-empty" rule
  applies to zero-emission runs too — the emitted==0 early return must not skip it;
- `streaming_decode_tps` is checked only when `event_s.size() == emitted` (Jackson
  3's `get(int)` throws on an out-of-range index, so an unguarded check turns a
  count mismatch into an NPE instead of a violation);
- `step_region_s` must be null outside fixed-continuation — the schema's "one value
  per mode, the other null (uniform key set)" (section 13.3, schema line 654).

### 14.3 Session report shape (launcher decision)

`session.json` (schema `jmlx-decode-baseline-1`): root
`{schema, session, workers[], summary}`.

- the session block is the section 10 block plus `fork_order` (the full cell
  identifiers — each worker's output directory name, the instrumentation
  suffix included — in fork order), `orchestration` (`spec-order`, the
  default, or `alternating` — which requires consecutive runtimes to
  strictly differ) and `native_cache` (the session's isolated
  classpath-extraction directory for the Java workers, retained on a
  successful session and deleted on any failure; a guard, not a live
  path — every worker command sets `jmlx.library.path` first and
  `NativeLoader` resolves it before any classpath-extraction fallback,
  so in a bootstrapped checkout no extraction ever runs and the
  directory stays empty, §15.6 item 8), and minus
  `manifest_sha256` (it lives in each worker's identity; the section 10
  `workload` digest is retained as `workload`); `host.hostrec.pre/post`
  and `python` are null until WP3; `native_pins.java` is read from
  `native/install/lib/native-pin.properties` (`mlxMetalVersion` → `metal`,
  `mlxcCommit` → `mlxc_commit`, empty strings when the file is absent),
  `native_pins.python` is null until WP3;
- `workers[]`: a valid report embeds the full worker block plus the worker's `dir`;
  any other outcome embeds the stub
  `{dir, runtime, adapter, case, prompt_length, mode, outcome}`, and an
  `invalid-report` additionally embeds the raw `report` so its violations can be
  reviewed;
- `summary`: `quantile_convention "linear-interpolation-type-7"` (h = (n-1)q,
  0-based, linear interpolation), `pairs: []` (WP3 fills),
  `exclusions[] {worker, reason, violations?}`,
  `per_runtime[runtime]["<adapter>/<mode>/<case>/<length>"][metric] = {p10, median, p90, n}`
  and `first_request[runtime]["<adapter>/<mode>/<case>/<length>"][metric]`. The
  cells are **adapter- and mode-scoped** (review round 1, §13.4 item 16(a)):
  headline and fixed-continuation measure different quantities and never pool,
  controlled and stock never pool (a mixed run's pooled median TTFT landed
  between the two adapters' own medians — the finding that forced the key
  change), and reference-pair samples run the observed (unprimed) cache
  regime, so they are excluded from `per_runtime` entirely — they belong to
  the WP3 `pairs` aggregation. Warmups are never aggregated. Fixed-continuation
  cells additionally carry `step_s` — the contract's decode comparison
  (§13.3) — quantiled per step over the measured requests' 32 frozen
  positions each (32 × measured samples); the first request's FC steps are
  not pooled into it, its first-request cell carrying only the request-level
  metrics. (An early unmodeled run pooled a
  reference-pair sample into the headline cell — n=21 instead of 20 on the 0.6b
  headline; the mode-scoped rule and its fixture tests are what caught it.)

### 14.4 Outcomes and exit codes

Worker outcomes in the session report: `ok` / `failed` / `timeout` (valid report,
its `status`), `invalid-report` (report present, structural violations — with the
raw report), `dead` (claim, no report), `crashed` (no claim, no report),
`not-started` (the launcher never reached it). Launcher exit: 0 = all workers ok;
1 = spec validation failed (nothing forked, no `session.json`); 2 = any worker
non-ok (`session.json` is still written, the failed cells are recorded as
exclusions, never a partial success presented as success). A starter exception
stops the session: the remaining workers are recorded `not-started` and the exit
is 2. Worker process exit: 0 ok, 1 failed, 3 timeout (the same codes as the
Python worker).

### 14.5 Jackson 3 (tools.jackson 3.1.2) API pitfalls (verified by javap + probe)

- `ObjectNode.putArray(String)` returns `ArrayNode`, not `ObjectNode`;
- `isTextual()` / `asText()` / `textValue()` are deprecated → `isString()` /
  `asString()`;
- `asStringOpt()` coerces *any* node type (a number yields `Optional[123]`) — it
  is not a textual test;
- `ObjectMapper` has no `createTextualNode` (use `put(String, String)` or
  `createArrayNode()`);
- a naive `find -name "*.jar"` over the Gradle cache puts `*-sources.jar` on the
  classpath and breaks standalone javac runs (exclude `-sources` jars).

### 14.6 Local evidence

- `:jmlx-benchmarks:check` green with 73 launcher tests — 6 statistics
  (hand-computed type-7 values), 28 validator (every valid adapter shape plus
  every structural rejection), 39 end-to-end over a hermetic mini repository
  (a temp-dir checkout with a tiny manifest and one artifact directory, and a
  fake worker starter that simulates every protocol outcome — pure Java, no
  native bootstrap, runs on any platform). The review-round-1 e2e additions
  cover the adapter-scoped key shape, the report-vs-job violations (mismatch,
  no-requests, short prefix, extra requests), the kill and session-budget
  paths, the run_policy rejections, and the digest-pin mismatch plus the
  frozen-manifest backstop. `:jmlx-models:check` green with
  `BaselineAdaptersTest` (5), `BaselineWorkerProtocolTest` (20 — the two
  review-round-1 additions: the tokenizer-eos stop-set failure and the
  stale-watchdog two-run test) and `BaselineWorkerProcessTest` (9); no new
  suite classes were added in the round, so the macOS CI required-suite list
  in `.github/workflows/ci.yml` is unchanged.
- Opt-in 0.6b session on the final head (real frozen manifest, M5 Max,
  precision 1): both workers `ok` in ~20 s wall. Headline (round-1 key
  `controlled/headline/qwen3-0.6b/32`): 26 requests (1 first + 5 warmups +
  20 measured), 128 emitted per request, finish `controlled-max-tokens`,
  `load_state materialized`, primed-per-request;
  `per_runtime.java["controlled/headline/qwen3-0.6b/32"].ttft_s = {p10
  0.0057996631, median 0.006027583, p90 0.0068274292, n 20}`, `prefill_s`
  median 0.0054023335, `full_generation_s` median 0.569823333,
  `streaming_decode_tps` ≈ 223.3 / 225.4 / 228.1; first request ttft
  0.119919417 s, prefill 0.115580583 s, full 0.842221583 s, 175.8 tok/s.
  Reference-pair: `ok`, `observed`, one measured request, excluded from
  `per_runtime`. Session identity: workload sha `e8c7730178a2e8f1…` (the
  frozen manifest digest), native pins metal 0.32.4 / mlx-c
  `a26f2d584a14ec619ae8db9a09be03becd47598a`, code sha `c80cdfa` on branch
  `phase8-2-wp1-java` (dirty — with the review-round-1 fixes this section
  records, committed with this record), `fork_order ["java", "java"]`.
  The numbers agree with the pre-round session within run-to-run variation
  (median TTFT 0.0070 s → 0.0060 s, streaming ≈ 228.4 → 225.4 tok/s).

### 14.7 Review round 1 (2026-11-08, PR #89): launcher and worker hardening

The round's findings (one Medium reproduced, five Low, nits, two
suggestions), each closed with a test; the contract-level amendments are
recorded in §13.4 item 16:

- **Medium (reproduced)**: the summary pooled controlled and stock
  headlines — the reviewer's reproduction (the mixed fake-worker e2e run
  over the tiny mini case, not a 0.6b run) had a median TTFT of 0.535 s,
  between controlled's ~0.03 s and stock's ~1.03 s. Fixed with the
  adapter- and mode-scoped cell keys on `per_runtime` and `first_request`
  (§14.3), asserted by the mixed-adapter e2e fixture.
- **Low**: the launcher session block's and the worker identity's
  `java.flags` now share one source — space-joined
  `ManagementFactory.getRuntimeMXBean().getInputArguments()` (§14.1(c)).
- **Low**: the stock "drain evaluate" note was wrong — the synchronous
  Java loop evaluates every step at its selection boundary and queues no
  lookahead, so nothing is pending at `generate` return. The explicit
  `MLX.eval()` drain was removed from `BaselineAdapters.runStock` and the
  worker's stock note reworded to say no drain exists or is needed
  (§13.4 item 16(c)).
- **Low**: the worker watchdog is per-run (the run token, §13.4 item
  16(e)); the two-run in-process test executes the stale path end to end.
- **Low**: the launcher cross-checks every ok report against the launched
  job (`jobViolations`: runtime/adapter/case/prompt_length/mode, the
  report's own `identity.manifest_sha256`, `identity.precision_env`, and
  the request-kind sequence — ok carries exactly the job's mix, failed at
  most a complete prefix); a mismatch is `invalid-report` with the
  violations in the exclusion (§13.4 item 16(f)).
- **Low**: the launcher limits (§13.4 item 16(g)) — the fork wait limit is
  `min(timeout_s + 90 s, cell_time_budget_s)` (`WAIT_LIMIT_BACKSTOP_S`)
  with a 10 s kill settle after `destroyForcibly`; a starter returning
  `WORKER_KILLED = -1` (the starter-level sentinel, not an outcome)
  classifies the cell from its left artifacts and records the kill with
  the limit in the exclusion note; the session loop checks elapsed
  against `session_time_budget_s` before each fork and records the
  remainder `not-started` with `REASON_BUDGET`; `validateRunPolicy`
  rejects a missing, non-integer or below-minimum `run_policy` field
  before anything forks, and `buildJob` reads the policy default-free.
- **Nits**: the stock stop-set assertion loads the `HfTokenizer` first and
  unions its special-token eos into the resolved set (§13.4 item 16(b));
  `benchmarkBaseline` depends on `:jmlx-models`' test *runtime classpath*
  FileCollection (so the test-fixtures jars build, not just `testClasses`);
  the `BaselineStatistics` javadoc now cites the right findings file and
  says empty samples carry NaNs.
- **Suggestions**: fixed-continuation `step_s` in the session summary
  (§14.3); the frozen-manifest digest pin
  (`PINNED_MANIFEST_SHA256 = e8c7730178a2e8f1…`, the same digest the root
  `verifyDecodeBaselineManifest` guard backstops) with a test-only seam
  (`setPinnedManifestSha256ForTest`) — the e2e mini repositories re-pin per
  test and restore the production pin after each, and the backstop test
  walks up from the test class's code source to the repository root and
  asserts the pin equals this checkout's `workloads.json` digest.

### 14.8 Review round 2 (2026-11-08, PR #89): deterministic cross-run test and lock ordering

Two test-only findings — the stale-watchdog test's sleep margin and the
`resetRunState`/run-token lock-ordering gap — plus two wording nits (the
stop-set assertion javadoc's stated reason, the §14.7 Medium
attribution); all recorded in §13.4 item 17. The cross-run test now
holds run A on a one-shot `requestHold` latch and observes every
ordering step (worker-ready, latch consumption, watchdog fire) with
bounded polls instead of fixed sleeps; A is released only after run
B's report is on disk, so A's post-release failure handling can never
race B's report write. `resetRunState()` moved inside the `REPORT_LOCK`
critical section after the run-token increment, so a stale watchdog
can no longer observe the fresh state under the old token. No
contract, schema, golden or measurement-path change: the §14.6
session 'd' numbers stay valid and no opt-in session re-run was
needed.

### 14.9 Review round 3 (2026-11-08, PR #89): race-description correction and latch release on failure

Two test-only findings, recorded in §13.4 item 18. Item 17(a)'s
description of the round-2 race was corrected in place: A's resumed
request woke before B wrote its report, and A's poisoned report was a
failed report from the error handler (the resumed request throws on
B's live scope, which is confined to B's thread, so the post-loop
timeout path is never reached), not a timeout report. The cross-run
test now releases the hold latch in a try/finally, run A is a daemon
thread, and `resetTestSeams` releases any outstanding latch, so a
failed check can no longer leave a held thread blocked for the rest
of the test JVM. No contract, schema or measurement-path change; the
§14.6 session 'd' numbers stay valid.

### 14.10 Review round 4 (2026-11-08, PR #89): scope-state wording in the old-race description

One wording finding, recorded in §13.4 item 19: item 18(a) and §14.9
attributed A's resumed-request failure in the old race to the closed
scope — true only in the current test, where B has already finished.
In the old race B was still running, so the scope was B's live one
and only B's thread could use it (thread confinement, as the round-1
test comment said). Corrected in place; no code, contract or
measurement change and no session re-run.

## 15. WP2 — Instrumentation, host observations and FFM calibration (2026-11-08, this host)

Phase 8.2 WP2 (plan §5.1 and §6) landed on `phase8-2-wp2` as five
commits over the WP1 merge head `ce4e193`: the agent generalization,
the WP1 boundary-bug fix, the worker instrumentation, the launcher
plumbing and the FFM calibration. The frozen contract was amended in
place (§13.1, §13.3, §14.3; the amendment list is §13.4 item 20). No
native pin, generated binding, published API or golden changed, and
ordinary `check` gains no download, Python or C-compiler
prerequisite.

### 15.1 Agent generalization (`e935464`)

The binding-count agent moved from the jmlx-models tests into the
jmlx-core testFixtures (`se.alipsa.jmlx.nn.BindingCounterAgent`),
generalized for any native-call-counting consumer:

- system-property prefix `jmlx.binding.*` (`installed`,
  `count.<name>`, `recording`, `failure`, `signatures`);
- `signatures()`: binding name -> the generated static wrapper's JVM
  descriptor (`MethodModel.methodType().stringValue()`), recorded at
  class-load time — the FFM estimate groups counts by these signature
  classes without re-reading the generated bytecode;
- `failureReason()`: the mid-run transform failure message;
  `start()`/`stop()` throw it, so an instrumented run fails rather
  than counting silently;
- the `:jmlx-core:bindingAgent` task (classifier `binding-agent`)
  builds the `-javaagent` jar; the opt-in `:jmlx-core:bindingCoverage`
  test asserts `mlx_array_new_data >= 2`, `mlx_array_free >= 2` and
  `mlx_matmul == 1` in one window.

The agent instruments the generated static wrappers (name
`mlx_[a-z0-9_]+` with an `invokeExact` in the code), not the holder
classes; the generated sources stay byte-identical. Fresh local run
(this host, native staged): `BindingCoverageTest` 1 test, executed
and green.

### 15.2 WP1 boundary-bug fix (`e852118`)

Verified against the frozen Python reference
(`tools/decode-baseline/controlled.py`): it creates the prompt cache
**before** `t0`. The Java adapters (`runControlled` and
`runFixedContinuation`) started `t0` before `freshCaches`, so the old
Java `ttft_s` included the per-request prompt-cache construction and
the per-step `s` included the per-step cache-construction work. Fix:
`t0` (and the event-time origin) now comes after `freshCaches` in
both paths; `stepS` stays right after `evalStep`. `step_region_s` and
`streaming_decode_tps` are unchanged under both definitions (the tps
is an event-time difference — the origin cancels; the FC region is
prefill-eval to final-step host work). The frozen §13.3 wording ("the
prompt cache is created before the timer starts") was already
correct, so this was a Java bug, not a recorded deviation. Caveat: the
§14.6 session 'd' 0.6b numbers were measured under the old Java
semantics — their `ttft_s` includes prefill cache creation and their
per-step `s` includes per-step cache creation; the WP3 headline
baseline runs under the fixed semantics and must not be compared
against those numbers.

### 15.3 Instrumentation and host observations (`57a9319`, `ad07170`)

**Adapters.** `BaselineAdapters.PhaseObserver` (package-private):
`void window(String phase, int position, ModelWork work)` +
`ModelWork { void run(); }` + `NONE`. The windows never change the
work, scope lifetimes or timer boundaries:

- controlled: `prefill-construction` / `prefill-sync` /
  `prefill-selection` at position 0 (prompt construction; `evalStep`;
  token-1 argmax) and `prefill-release` at position 0 (the prefill
  activation scope's `close` — its per-array `mlx_array_free` downcalls
  are observed work, §15.6 item 1); per cached step k: `construction` /
  `sync` / `selection` at `prompt_length + k + 1` (decode forward;
  `evalStep`; argmax) and `release` at the step's position (the step
  activation scope's `close`); the request scope's `close` (the
  per-request caches) runs in a `request-release` window at position 0,
  after every frozen timer has stopped;
- fixed-continuation: the two prefill windows
  `prefill-construction` / `prefill-sync` (no `prefill-selection` —
  the pinned loop has no token-1 argmax) plus the same `prefill-release`
  window, then per fed step `construction` / `sync` / `selection` /
  `release`, with `selection` carrying the argmax **and** the
  full-vocabulary top-2-margin readback, and the same `request-release`
  window;
- stock: one `request` window at position 0 around the shipped
  generation call — a documented controlled-first limitation (stock's
  internal boundaries are not observable without modifying the
  shipped loop). The release windows never change the work, the scope
  lifetimes or the frozen timer boundaries: each `close` runs at the
  same wall-clock position try-with-resources gave it (an observing run
  produces the identical tokens and the same frozen timings as a
  `NONE` run). The release windows run through
  `BaselineAdapters.releaseWindow` (§15.7 item D): a release-window
  failure does not replace an in-flight exception — it is attached to
  it with `addSuppressed` (and propagates as-is when no exception is
  in flight); the catch is `Throwable`, and if the window dies before
  the close runs (the binding-count agent's `start()` failing, for
  example), the close is called directly rather than skipped —
  `MLXScope.close()` is idempotent, so a window that already ran it
  pays nothing — and a fallback-close failure is attached to the
  release failure, never masking it (review round 3, 2).
  Fixed-continuation's `prefill_s` is measured inside the
  prefill try, after `prefill-sync` (the evaluated prefill logits) and
  before the `prefill-release` window, so the release window's own
  overhead never enters `prefill_s` on either adapter (§15.7 item C).

**Worker.** The job field `instrumentation` (absent -> `none`) gates
before model load: binding-count requires the agent installed on the
worker JVM (`isInstalled`) with no premain transform failure
(`failureReason`); tracing requires the thread CPU time API.
`identity.instrumented` is present in every worker report; the
`identity.host` block (hw_model, os, os_build, disk_free_bytes,
system_load_average, vm_stat and the top 16 of
`ps -Ao pid,pcpu,rss,comm -r`) is present iff instrumented. Per
instrumented request: `binding_counts` (the phase windows plus
`total`; a non-empty `total` is a `require` gate — an empty total
means the agent silently stopped counting, which fails the run; each
window's `counts` lists hit bindings only); `tracing` (the span wall
and caller-thread CPU, the per-window wall/CPU, and the residual =
span − Σwindows); `memory` = `{active_bytes, peak_active_bytes,
cached_bytes, rss_peak_bytes}`, with `MLXMemory.resetPeak()` before
every instrumented request (model load excluded) and
`rss_peak_bytes` the cumulative maximum of the post-request
`ps -o rss=` samples — a documented lower bound on the true peak; the
request's wall/CPU clocks are taken **before** the RSS sample, because
the sampler forks `ps` and the fork must not land inside the measured
request span (§15.6 item 5). The binding-count window hook keeps the
agent's lifecycle apart from the window's work: `start()`, `work.run()`
and `stop()` each run in their own error boundary — an agent failure
fails the run with a named phase ("failed to start the binding-count
agent for the '<phase>' window" / "failed to stop the binding-count
agent in the '<phase>' window"), while a `RuntimeException` *or an
`Error`* (an `OutOfMemoryError`, an `ExceptionInInitializerError` when
a lazy holder class loads) from inside `work.run()` (a model error, an
`MLXException`) propagates as itself after a best-effort agent
`stop()`, never re-labelled as an agent failure (§15.6 item 6; the
`Error` boundary is §15.7 item E). Both boundaries catch
`Throwable`, and a `stop()` failure is attached to the work failure
with `addSuppressed` instead of swallowed — the agent is already
broken; the window's own failure is what the report must carry.
Uninstrumented workers carry none of the diagnostic fields and pay for
two `nanoTime` reads only. The
host-observation command runner degrades to a missing sample when its
10 s bounded wait times out (§15.6 item 12).

**Launcher.** `BaselineJob` gains `instrumentation`,
`bindingAgentJar` and `nativeCacheDir`; `dirName` appends the
`-<instrumentation>` suffix when instrumented (the cell identity);
`command()` adds `-Djmlx.native.cache.path` for every forked Java
worker and the `-javaagent:` for java + binding-count workers. Spec
validation: `orchestration` is `spec-order` (default) or
`alternating` (consecutive runtimes must strictly differ); a
per-worker `instrumentation` is never applied to Python workers; a
binding-count Java worker without the
`jmlx.baseline.binding.agent.jar` property is a spec failure, and a
property that names a file that is not a regular file is a spec
failure too (the forked JVM would otherwise die at `-javaagent` load
time, mid-session, §15.6 item 11); the duplicate-cell key includes the
instrumentation, so the headline cell and the instrumented cell it
measures are distinct cells and both run (§15.6 item 4). The
launcher's own host-observation commands (`git`, `df`) wait at most
30 s and degrade to a missing value (§15.6 item 12). Session
block: `orchestration`, `native_cache` (`session/native-cache`;
retained on a successful session, `deleteTree` on exit != OK; a guard
— the `jmlx.library.path` set first on every worker command wins in
`NativeLoader`, so no extraction ever runs in a bootstrapped checkout
and the directory stays empty, §15.6 item 8);
`fork_order` carries the full cell identifiers. Validator:
`INSTRUMENTATIONS`, the memory rule (memory is null iff the worker is
uninstrumented, non-null iff instrumented) and the cross-mode absence
(`binding_counts` only in binding-count, `tracing` only in tracing).
The `benchmarkBaseline` task sets the agent-jar property and depends
on `:jmlx-core:bindingAgent`.

### 15.4 FFM calibration (`135c9b3`)

**C source** (`jmlx-benchmarks/src/main/resources/ffm_cal.c`, a
classpath resource): one non-elidable no-op per shape the controlled
binding-count workloads actually call — the 27 keys of §15.7 item A
(round-1's 16 representative classes are gone), symbols
`cal_<param-keys>_<return-key>` (`cal_M_I` … `cal_P_P_I_I_M`; P =
void* parameter, M = 8-byte struct carrier by value, I/J/F/D/Z/V the
scalars, the no-parameter shape `cal__M`). Each body folds its
arguments into a `volatile uint64_t` sink (an observable side
effect, so `-O2` cannot eliminate the call) and never dereferences an
input; `struct cal_eight { void *ctx; }` mirrors the 8-byte opaque
struct. **Carrier fidelity:** in the pinned mlx-c header every opaque
struct is passed **by value** (`int mlx_array_free(mlx_array arr)` —
jextract maps such a parameter to the struct's 8-byte layout, not to
a pointer; verified in the committed generated sources, e.g.
`mlx_array_free`'s `FunctionDescriptor.of(C_INT, mlx_array_.layout())`);
on AArch64 an 8-byte struct parameter is carried in one 64-bit
register — the same C ABI as a `void*` — and 8-byte struct returns by
value use the same register convention, so the no-op carriers are
register-equivalent to the real ones **and** the Java-side
marshalling matches (the 8-byte struct read from the argument
segment, not a bare pointer pass). The C functions take `struct
cal_eight` parameters by value accordingly (review round 1, §15.6
item 7; the previous `void *` parameter form under-measured the
Java-side struct read the generated wrappers pay). C `size_t` is
modeled by `C_LONG` (key `J`), `int` by `C_INT` (`I`), `bool` by
`C_BOOL` (`Z`).

**Worker** (`FfmCalWorker`, fresh JVM): `SymbolLookup.libraryLookup`
+ `Linker.nativeLinker().downcallHandle` exactly like the generated
holders — one private holder class per shape
(`Handle0` … `Handle26`, `static final MethodHandle HANDLE =
downcall(N)`, bound in `<clinit>` through a shared lookup installed by
`run()` before `initializeHandles()` pre-touches all 27), so each
measured call site references a JIT-trusted constant field — the
jextract wrappers' own form (678 `invokeExact` sites in
`mlx_h.java`), not the address-indexed dispatch a `MethodHandle[]`
array load would force (review round 1, §15.6 item 7); one dedicated
typed static call site per shape (`call0` … `call26`) invoking
`invokeExact` on its holder's handle. Every M parameter slot is
declared the 8-byte struct layout (by value, as in the generated
wrappers); the pointer-return shape `(M)P` keeps its `ADDRESS` return
layout, and each of the four struct-return shapes (`()M`, `(F)M`,
`(PJ)M`, `(PPII)M`) takes a `SegmentAllocator` as its first Java
argument (JEP 451: the allocator is the handle's first parameter,
consumed JVM-side; the 8-byte struct is returned in a register, no
hidden C out-parameter). The carriers are production's kind, not a
bare confined arena (§15.7 item A, calibration side): the ten P/M
argument carriers (the maximally-arity shapes need ten 8-byte
arguments) are process-lifetime `Arena.ofShared()` segments — the
same arena kind production `MLXScope.Holder` uses for every native
handle — and the struct-return shapes allocate their results through
a `CalScope` that mirrors a production step scope's lifecycle
(synchronized shared-arena allocation plus per-scope handle tracking,
created before every sample's timed batch and closed after it,
outside the timed region, exactly as production closes a step scope
after every step — review round 3, 1; a fork-lifetime scope would
have run its tracking set's doubling rehashes and the growing live
heap's GC pauses inside the later samples' timed batches); the
dylib's `libraryLookup` keeps its own confined arena open for the run
(the dylib must stay loaded). The shape order rotates per iteration; one
sample is the wall time of `calls` consecutive downcalls; the timing
floor is a private `floor(calls)` no-op loop — a linear congruential
recurrence `sink = sink * C + (k + 1)` with a loop-carried
dependency, so a vectorizer cannot reassociate the iterations away
(review round 1's xor-fold was reassociatable, §15.7 item F) —
warmed up like the batches and folded into the same volatile sink;
fixed argument constants `INT_ARG = 7`, `LONG_ARG = 4096`,
`BOOL_ARG = true`; raw JSON schema `jmlx-ffm-cal-raw-1`.

**Calibration** (`FfmCalibration`, task `benchmarkFfmCalibration`
with `--args 'output-directory [iterations calls forks warmup]'`,
defaults 200 2000 5 2000): extracts the source into the output
directory (the dylib and the extracted source are the only build
artifacts — under `build/`, no production native artifact), builds it
with `cc -shared -O2` (a clear error when `cc` is absent; the compile
wait is bounded at 300 s, the version probe at 30 s with a "cc"
degradation, and each worker fork at 300 s plus one second per
million downcalls the fork makes — a stuck child fails the run
instead of hanging it, §15.6 item 12), records compiler/flags/
`source_sha256`/`binary_sha256` in `build-info.json` (schema
`jmlx-ffm-cal-build-1`), forks fresh JVMs (the Tf32Benchmark pattern,
`--enable-native-access=ALL-UNNAMED`) and pools the raw samples into
per-shape `n/p10/median/p90/min/max/raw` (type-7 quantiles from
`BaselineStatistics`) plus the `loop` floor entry (schema
`jmlx-ffm-cal-1`). `aggregate` is a pure method over the fork outputs.

**Estimate** (`FfmTransitionEstimate`): pure
`estimate(workerReport, calibration)` -> schema
`jmlx-ffm-transition-estimate-1`. Refuses up front any
non-controlled worker report (the stock adapter's single
whole-request window cannot be split into prefill and per-step
phases), any report with no measured request (§15.6 items 2 and
3), any report without a nonempty `binding_descriptors` object
(predating the native-descriptor recording — a clear re-run error,
§15.7 item A) and any calibration summary without a positive
`workload.calls` (the per-call division below would be undefined,
§15.7 item B). `shapeKey` (round 1's `normalize` is gone) parses the
*native* descriptor grammar of the worker-recorded
`binding_descriptors` into a shape key — `(params)return`; `a8`
(optionally named, optionally carrying the `:[*:b1]` element
annotation) -> P; `[body](name)` -> M iff the body's byte size is 8
(padding counts as `xN`; brackets are matched, so nested structs and
named members parse), unmodeled otherwise; `i4`/`j8`/`f4`/`d8`/`z1`/
`v` -> I/J/F/D/Z/V — with null (unparsable or unmodeled, e.g. a
by-value struct that is not 8 bytes) -> uncovered with the recorded
descriptor; the JVM-form `binding_signatures` are provenance only and
never classify (§15.7 item A). Only `kind == "measured"` requests
enter every aggregate (totals, per_shape, uncovered, prefill,
per-step, request_total — warmups and firsts are excluded, §15.6
item 2); the output carries `measured_requests`. Windows whose phase
starts with `prefill-` -> the request's prefill budget; phases in
{construction, sync, selection, release} with position > 0 -> that
request's step-phase budget at the position; the per-position output
is the type-7 quantiles over the requests' own per-position samples
(one sample per (measured request, position) — never a sum across
requests, the N-times-too-large pooling bug of §15.6 item 2), and the
top-level `per_decode_step` is the pooled quantiles over all
(request, position) samples; plus the request total (one sample per
measured request; the `request-release` window enters it but neither
prefill nor any step). Every per-call cost divides the per-sample
calibration quantile by `workload.calls` (type-7 quantiles commute
with a positive scale factor, so the divided quantiles are exactly
the quantiles of the per-call samples). Every budget accumulates its
`count × per-call cost` terms in double and rounds once at its
output value — rounding each (binding, window) term to whole
nanoseconds would bias a ~50-binding step by up to ±25 ns at
per-call costs of tens of nanoseconds (review round 3, 3). Round
1's `median_ns_per_call` emitted the per-*sample* median as if it
were per-call, overstating every budget by the factor `calls`
(§15.7 item B) — so `per_shape` carries `median_ns_per_call` with
`p10_ns_per_call`/`p90_ns_per_call` alongside, and the budgets
(`budget_ns` plus `budget_p10_ns`/`budget_p90_ns` — the p10/p90
accumulators round 1 computed but never emitted) are per-call. Every
summary node (each per-position entry, `per_decode_step`, `prefill`,
`request_total`) reports two ranges: `p10_ns`/`median_ns`/`p90_ns`
keep the round-1 meaning (the request-to-request spread of the
per-request budgets computed from each shape's calibration median),
and the new `calibration_p10_ns`/`calibration_p90_ns` are the
calibration's own sampling range — the same per-request and
per-position budgets computed from each shape's calibration p10 and
p90 samples (§15.7 item B). The output also carries the
classified-call coverage, the uncovered signatures (name ->
descriptor + calls) and four explicit assumptions: the holder-form
constant handles plus the production arena fidelity (the P/M carriers
on a process-lifetime shared arena, the struct-return results
through the per-sample `CalScope` that mirrors a production step
scope's lifecycle — round 1's confined-arena caveat is gone), the
by-value carrier
fidelity with the shape keys now the native descriptor classes of
the called bindings (round 1's pointer-kind-M-indistinguishable
caveat is resolved by the native descriptor), the count semantics
(attempted downcalls through the generated wrappers, no Metal/GPU
work), and "A calibrated estimate, not a causal partition" with the
two-range reporting named.

**Local measurement baseline** (M5 Max, Apple clang 21.0.0,
review-round-3 source SHA-256
`cb30e80b9824eb51113efca5a608f704d9503421e9e7ee331a0eb3d5bc60e2d4`,
binary SHA-256
`f1e2e7d76b4ff25d610c4284f38b856044eee604e3b844c45f539f0d30901ad9`
— bit-identical to the round-2 binary, the round-3 C change being
comment-only — default workload 5 forks × 200 iterations × 2000
calls, warmup 2000, n = 1000 pooled samples per shape, re-measured
after review round 3, 1 re-scoped the result allocator per sample —
these numbers supersede and are not comparable to the review-round-2
values recorded above; the struct-return shapes drop ~10× and that
drop is the expected correction: the round-2 medians carried exactly
the artifact of review round 3, 1 (the fork-lifetime tracking set's
doubling rehashes and the growing live heap's GC pauses landing in
the later samples' timed batches), while the allocation-free int
shapes show only the environmental variance of the +10% floor shift
(1.19 → 1.31 ns/iteration); the artifact is gitignored under
`build/ffm-cal/`): no-op floor median 2625 ns per 2000 calls
(1.31 ns/iteration; p10 2416, p90 3125); the 23 int-return shapes
4.56–13.17 ns/call (median ns/call: `(M)I` 4.56, `(M)J` 4.77,
`(PDDDIM)I` 8.98, `(PMM)I` 9.04, `(PMIIM)I` 9.08, `(PM)I` 9.15,
`(PMMMM)I` 9.17, `(PMMM)I` 9.31, `(PMIM)I` 9.31, `(PMMIM)I` 9.31,
`(PMMFM)I` 9.38, `(PMIZMFIMM)I` 9.48, `(PMMMMZM)I` 9.48, `(PMZM)I`
9.56, `(PMIZM)I` 9.56, `(PPJMIM)I` 10.40, `(PMMMFPMMZM)I` 10.71,
`(PMPJM)I` 10.75, `(PMPJZM)I` 10.90, `(PMMMMZMMPM)I` 10.98,
`(PMMMMMPMMM)I` 11.54, `(M)P` 12.28, `(PMPJPJPJM)I` 13.17); **the
four struct-return shapes 52.2–55.1 ns/call** (`()M` 52.2, `(F)M`
54.1, `(PJ)M` 54.1, `(PPII)M` 55.1) — the per-call result-segment
allocation through the per-sample `CalScope` dominates, exactly as in
production (round 1's `(AM)M` confined-arena figure of ~183 ns/call
and the round-2 fork-lifetime figure of 509.0–517.9 ns/call are
superseded; the round-2 figure measured the same allocation plus the
artifact the re-scoping removed). The estimate re-run end-to-end
against the same two binding-count session reports (TinyMistral-6x248M
and Qwen3-0.6B) gives coverage 1.0 with zero uncovered bindings in
both, a per-decode-step FFM budget of 76.65 µs / 106.68 µs against
measured decode-step wall times of 14,437.2 µs / 20,162.5 µs
(≈ 0.53%), prefill budgets 85.54 µs / 100.23 µs and request totals
9.82 ms / 13.65 ms (calibration ranges 63.26–98.74 / 88.11–137.33
µs per step); the top per-shape contributors are now `()M` (118.2 /
165.7 ms per request), `(M)P` (24.0 / 34.6 ms) and `(M)I` (13.2 /
18.8 ms), and the counted-only descriptor restriction (review round
3, 4) reproduces the full-descriptor estimates exactly (minus
`created_at`) on both workloads. The large max/p90 spikes are OS
noise (which is why p10/median/p90 are reported).

### 15.5 Test evidence

`:jmlx-benchmarks:check` green with 96 tests (94 after review
round 1, §15.6) — `BaselineBenchmarkTest`
51, `BaselineReportValidatorTest` 28, `BaselineStatisticsTest` 6,
`FfmCalibrationTest` 1 (executed when `cc` is present: the full
pipeline, ordered per-shape quantiles above the loop floor; skipped
when `cc` is absent, so ordinary `check` gains no compiler
prerequisite), `FfmTransitionEstimateTest` 8 (pure Java: the
native-descriptor `shapeKey` mapping including the unparsable
(`(L)I`, `(i4`, `int(int)`, null), the unmodeled non-8-byte struct
and the scalar-only `(II)I` parse; the hand-computed budget/coverage
over the controlled adapter's full window set including the release
windows — step 33 = 210 ns, step 34 = 270, pooled per-step
p10/median/p90 216/240/264 over the two (request, position) samples
with the calibration range 177/323, prefill 480 (range 410/590),
request total 1080 (range 910/1330), coverage 1.0, per-shape
`(PMMM)I` 8 calls / 720 ns budget with per-call 90/80/110 and
`(M)I` 9 / 360 with per-call 40 — the synthetic calibration carries
`calls = 10` with per-sample quantiles 10× the per-call values, so
the division by `workload.calls` is pinned, §15.7 item B); the
two-measured-requests pooling test (position-33 samples [90, 90] →
the request-to-request spread collapses to 90 while the
calibration range does not, 80/110; prefill p10/median/p90
45/65/85 over [40, 90] with calibration range 35/104, request total
median 155 with range 115/214; warmup request with 200 matmul calls
excluded from every aggregate, including the calibration-range
fields); the non-controlled, no-measured-request and
no-native-descriptor refusals; the calibration-without-a-positive-
`workload.calls` refusal; the uncovered case at coverage 3/7 lists
both the unparsable `(L)I` signature and the missing-descriptor
binding) and `Tf32BenchmarkTest` 2.
The `:jmlx-models` baseline suites green: `BaselineAdaptersTest` 9
(four more: the release-window failure paths — the observer dying
before the close, the observer dying after a clean close, an `Error`
instead of a `RuntimeException`, and no original in flight — direct
unit tests over a throwing `PhaseObserver` and a real scope close,
review round 3, 6), `BaselineWorkerProcessTest` 9,
`BaselineWorkerProtocolTest` 25 (its
tracing window-count assertion now expects the 33-window layout with
the release windows, §15.6 item 1; three more: a fake-agent
binding-count run whose report falls back to recording
`binding_descriptors` for every instrumented name, the counted-only
descriptor restriction, and a window work `Error` propagating
unwrapped with the recording stopped — review round 3, 4 and 6).
The opt-in
`:jmlx-core:bindingCoverage` (`BindingCoverageTest` 1) executed
against the staged native runtime. Every WP2 suite is pure Java or a
cc-gated optional, so the macOS CI required-suite list in
`.github/workflows/ci.yml` is unchanged.

### 15.6 Review round 1 (2026-11-09, PR #90): instrumentation, observation and calibration fixes

Twelve findings (seven bugs, five smaller issues), each closed with a
test where the behavior is observable; the contract amendments are
the in-place edits to §14.3, §15.3 and §15.4 above. No report or job
schema changed; the estimate output schema stays
`jmlx-ffm-transition-estimate-1` (one new field,
`measured_requests`); the frozen timers are untouched (every release
close runs at the same wall-clock position try-with-resources gave
it).

1. **Bug: release work invisible to the estimator.** The scope
   `close()` calls (a scope close issues its per-array
   `mlx_array_free` downcalls) ran in try-with-resources, outside
   every phase window, so a binding-count run's per-step budget
   dropped the step's own release. Closed: the controlled and
   fixed-continuation adapters now wrap each close in its own window
   — `prefill-release` (position 0), `release` (the step's position)
   and `request-release` (position 0, after every frozen timer) — via
   explicit try/finally with `observer.window(…, scope::close)`
   (`MLXScope.close()` is idempotent, so the finally path is safe).
   The estimator charges `prefill-release` to the prefill budget and
   `release` to the per-position step budget; `request-release` enters
   only the request total, per-shape and totals. The protocol test's
   window count moved 24 → 33 for the 8-token controlled layout.
2. **Bug: per-step budget scaled with the request count.** The
   per-position step accumulator summed across requests, so N
   measured requests made every per-step figure N× too large, and
   warmup/first requests leaked into every aggregate. Closed: all
   aggregates are restricted to `kind == "measured"` requests; the
   per-position output is the type-7 quantiles over the requests' own
   per-position samples (one sample per (measured request, position)),
   the top-level `per_decode_step` the pooled quantiles over all
   (request, position) samples; a report with zero measured requests
   is refused. New tests: the two-measured-requests pooling case
   ([40, 90] → median 65, never 130) with a warmup request carrying
   200 matmul calls that appears in no aggregate, and the
   zero-measured refusal.
3. **Bug: the estimate accepted stock reports and mislabeled their
   window.** The stock adapter's single whole-request `request` window
   (position 0) was classified as prefill. Closed: non-controlled
   worker reports are refused up front (with the reason), and prefill
   is classified by `phase.startsWith("prefill-")` instead of
   `position == 0` (which also keeps `request-release` out of
   prefill). New refusal test.
4. **Bug: the duplicate-cell key omitted the instrumentation.** A
   headline cell and its instrumented twin (different `dirName`,
   different cells) were refused as duplicates. Closed: the key
   includes the instrumentation. New test: same cell, different
   instrumentation, both job files present.
5. **Bug: the `ps` fork landed inside the request span.** The RSS
   sample ran before the wall/CPU clocks, so the `ps` fork's cost
   charged the request. Closed: clocks first, then the sample
   (§15.3).
6. **Bug: a window's own failure was misattributed to the agent.**
   The binding-count window wrapped `start` + work + `stop` in one
   `catch (IllegalStateException)`, so a model error (an
   `MLXException`, an ISE from a scope check) surfaced as "binding
   count agent failed". Closed: three error boundaries — start
   failure and stop failure fail the run with the phase named, a
   `RuntimeException` from the work propagates as itself after a
   best-effort agent `stop()` whose own failure is swallowed
   (§15.3).
7. **Bug: the calibration did not mirror the generated wrappers.**
   (a) The measured call sites invoked `invokeExact` on a
   `MethodHandle[]` array load — the generated wrappers bind each
   handle in a private holder class's `static final` field, which C2
   constant-folds; the array form forces address-indexed dispatch the
   wrappers do not pay. (b) The M parameter slots were declared
   `ADDRESS` (a pointer pass), but the pinned header passes every
   opaque struct **by value** — jextract maps it to the struct's
   8-byte layout, so the generated wrappers read the 8 bytes out of
   the argument segment on every call; the pointer form
   under-measured that read. (c) The empty-loop floor could be folded
   away per iteration and was never warmed up. Closed: 16 private
   holder classes (`Handle0` … `Handle15`, bound in `<clinit>` through
   a shared lookup, pre-touched by `initializeHandles()` which
   converts the `ExceptionInInitializerError` to a clean
   `IllegalStateException`); every M parameter slot declared the
   8-byte struct layout (C side: `struct cal_eight` parameters by
   value — the AArch64 C ABI is register-identical; return layouts
   unchanged); a `floor(calls)` method with a per-iteration xor-fold
   (two ALU ops, not foldable), called in warmup and in the measured
   loop. The assumption texts were rewritten to state the holder form,
   the confined-arena-vs-`MLXScope` allocator difference for the
   struct-return shapes, the by-value carrier fidelity through the
   struct-by-value declarations, and the pointer-kind caveat
   (indistinguishable in the recorded JVM descriptor). The calibration
   was re-measured; the new baseline and source SHA-256 are in §15.4
   (pre-round-1 values: floor 1542 ns/500, `(M)I` ~31.8 ns/call,
   4-5 carriers ~40-47, `(M)M` ~41, `(AM)M` ~107 — superseded).
8. **Native-cache directory is a no-op in a bootstrapped checkout.**
   `-Djmlx.native.cache.path` only guards the classpath-extraction
   fallback, which never runs because every worker command sets
   `jmlx.library.path` first and `NativeLoader.resolveLibraryDir()`
   resolves it before the extraction. The session contract (isolated
   directory, retained on success / deleted on failure) is kept; the
   guard wording was added in §14.3, §15.3, the `BaselineJob` and
   `BaselineBenchmark` docs and `jmlx-benchmarks/README.md`.
9. **Floor not comparable to the measured calls.** The empty loop
   (item 7c) — closed with the `floor(calls)` no-op baseline that
   does the same order of ALU work per iteration as the C no-op body
   and is warmed up like the batches.
10. **`asInt` on the per-window counts.** A window count above
    `Integer.MAX_VALUE` silently wrapped. Closed: `asLong` (and a
    `long` accumulator).
11. **Agent-jar property pointing at a missing file.** Spec
    validation only checked the property's absence; a path that is
    not a regular file passed and the forked worker JVM would die at
    `-javaagent` load time. Closed: `Files.isRegularFile` check with
    a violation naming the path and the failure mode; the
    `bindingCountJobRunsInstrumentedCell` test now creates a real
    temp file (its former `/agent/binding-agent.jar` string would now
    be refused), plus a new refusal test.
12. **Unbounded `waitFor` in every external-process path.** A stuck
    child (wedged `ps`, `df`, `cc`, a hung worker fork) hung the run
    forever. Closed with bounded waits + `destroyForcibly`: the
    worker's host probes (10 s, degrade to a missing sample), the
    launcher's `git`/`df` probes (30 s, degrade to a missing value),
    `FfmCalibration.build` (300 s, fail), `FfmCalibration.
    compilerVersion` (30 s, degrade to "cc") and
    `FfmCalibration.runWorker` (300 s + 1 s per million downcalls the
    fork makes, fail).

### 15.7 Review round 2 (2026-11-09, PR #90): calibration classification, ranges and adapter exception paths

Six findings (A–F). The contract amendments are the in-place edits to
§15.3 and §15.4 above, plus the worker-report amendment below. No job
schema changed; the worker report schema gains one additive field; the
estimate output schema stays `jmlx-ffm-transition-estimate-1` and gains
additive fields only (`calibration_p10_ns`/`calibration_p90_ns` on
every summary node and per-position entry, `budget_p10_ns`/
`budget_p90_ns`, `p10_ns_per_call`/`p90_ns_per_call` on `per_shape`).

**Amendment — worker report: `binding_descriptors` (finding A).** The
frozen §13.3 text is untouched; this is recorded here as the amendment.
In binding-count mode the worker block now also carries
`binding_descriptors`: a name -> string object mapping each recorded
binding name to the `toString` of the generated downcall's own
`FunctionDescriptor` (the *native* descriptor form, e.g. `mlx_eval` ->
`([a8(ctx):[*:b1]](mlx_vector_array_))i4`). Source:
`se.alipsa.jmlx.ffi.mlx_h.<name>$descriptor` via reflection — jmlx-ffi
is an `implementation`-scoped dependency of `jmlx-core`, on the
worker's runtime classpath but not its compile classpath, so the worker
cannot name the type. A binding whose descriptor cannot be read is
omitted rather than failing the run (the estimate records it as
uncovered). `binding_signatures` (the agent-recorded JVM-form
descriptors) is still written, now provenance only. The validator
requires a nonempty object of nonempty strings for both fields in
binding-count mode and their absence otherwise (Python workers are
never instrumented, so the requirement cannot fire for them).

**A. The estimate classified the counts by the JVM descriptor, not the
native one — and the calibration measured classes the workloads never
call.** Round 1's `normalize` keyed the JVM-form
`binding_signatures` (e.g. `mlx_matmul` ->
`(Ljava/lang/foreign/MemorySegment;...)I`) to the shape keys, treating
every `MemorySegment` as an 8-byte by-value struct `M`. That
classification conflates two register classes: a by-value 8-byte struct
and a raw pointer are both `MemorySegment` in JVM form, so the recorded
`M` carried a pointer-shaped binding's counts (or vice versa) into the
struct-shaped calibration number. Closed on three fronts:

1. *Worker recording* (the amendment above): `binding_descriptors`
   carries each called binding's native descriptor.
2. *Estimator classification* (native-descriptor-only): `normalize` is
   gone; `shapeKey` parses the native descriptor grammar —
   `(params)return`; `a8` (optionally named, optionally carrying the
   `:[*:b1]` element annotation) -> P; `[body](name)` -> M when the
   body's byte size is 8 (padding counts as `xN`; brackets are matched,
   so nested structs and named members parse), unmodeled otherwise;
   `i4`/`j8`/`f4`/`d8`/`z1`/`v` -> I/J/F/D/Z/V. Unparsable or
   unmodeled descriptors (a by-value struct that is not 8 bytes, an
   unknown token) classify as uncovered with the recorded descriptor,
   and a report with no nonempty `binding_descriptors` object (i.e.
   predating the recording) is refused with a clear re-run error. The
   JVM-form `binding_signatures` never drives the estimate again.
3. *Calibration shape set*: the 16 round-1 representative classes are
   replaced by the union of the native descriptor classes of the
   bindings the controlled binding-count workloads actually call — 51
   distinct called bindings across the three runs (TinyMistral-6x248M,
   Qwen3-0.6B, Qwen3-8B-4bit; the per-binding native descriptors are
   the worker-recorded `binding_descriptors` / agent class-load
   recording) collapse to exactly **27 keys**:

   | key | called bindings |
   | --- | --- |
   | `()M` | `mlx_array_new` |
   | `(F)M` | `mlx_array_new_float32` |
   | `(M)I` | `mlx_array_dtype`, `mlx_array_eval`, `mlx_array_free`, `mlx_eval`, `mlx_vector_array_free` |
   | `(M)J` | `mlx_array_ndim`, `mlx_array_size` |
   | `(M)P` | `mlx_array_data_int32`, `mlx_array_shape` |
   | `(PDDDIM)I` | `mlx_arange` |
   | `(PJ)M` | `mlx_vector_array_new_data` |
   | `(PM)I` | `mlx_array_set` |
   | `(PMIIM)I` | `mlx_flatten`, `mlx_swapaxes` |
   | `(PMIM)I` | `mlx_argsort_axis`, `mlx_astype`, `mlx_concatenate_axis`, `mlx_expand_dims` |
   | `(PMIZM)I` | `mlx_argmax_axis`, `mlx_softmax_axis` |
   | `(PMIZMFIMM)I` | `mlx_fast_rope` |
   | `(PMM)I` | `mlx_sigmoid`, `mlx_stop_gradient`, `mlx_transpose` |
   | `(PMMFM)I` | `mlx_fast_rms_norm` |
   | `(PMMIM)I` | `mlx_take_along_axis`, `mlx_take_axis` |
   | `(PMMM)I` | `mlx_add`, `mlx_divide`, `mlx_equal`, `mlx_greater_equal`, `mlx_less`, `mlx_less_equal`, `mlx_logical_and`, `mlx_matmul`, `mlx_multiply` |
   | `(PMMMFPMMZM)I` | `mlx_fast_scaled_dot_product_attention` |
   | `(PMMMM)I` | `mlx_where` |
   | `(PMMMMMPMMM)I` | `mlx_dequantize` |
   | `(PMMMMZM)I` | `mlx_gather_mm` |
   | `(PMMMMZMMPM)I` | `mlx_quantized_matmul` |
   | `(PMPJM)I` | `mlx_broadcast_to`, `mlx_reshape`, `mlx_transpose_axes` |
   | `(PMPJPJPJM)I` | `mlx_slice` |
   | `(PMPJZM)I` | `mlx_sum_axes` |
   | `(PMZM)I` | `mlx_contiguous` |
   | `(PPII)M` | `mlx_array_new_data` |
   | `(PPJMIM)I` | `mlx_full` |

   The pinned-header fact backing the M class: in mlx-c v0.8.0 every
   by-value struct — the opaque handles and the `mlx_optional_*`
   scalar wrappers alike — is a single 8-byte pointer member
   (`mlx_vector_array` at
   `native/install/include/mlx/c/vector.h:26`, so `mlx_eval` is
   `(M)I` and `mlx_vector_array_new_data` is `(PJ)M`); any by-value
   struct that is not 8 bytes is unmodeled and uncovered by
   construction (the pinned header has none). Consequence in the
   re-run: coverage is 1.0000 with zero uncovered bindings on both
   e2e reports (§15.4).

**B. One range where two were owed, and per-sample costs charged as
per-call.** Round 1's output carried a single p10/median/p90 triple
(the request-to-request spread over the per-request budgets) — the
calibration's own sampling range was dead weight: the per-shape p10/p90
budget accumulators were computed but never emitted, and
`median_ns_per_call` emitted the per-*sample* median (one sample is
`workload.calls` consecutive downcalls) as if it were per-call, so
every budget overstated the per-shape cost by the factor `calls`
(2000× at the default workload). Closed:

1. *Per-call costs*: every per-call cost divides the per-sample
   quantile by `workload.calls` (type-7 quantiles commute with a
   positive scale factor, so the divided quantiles are exactly the
   quantiles of the per-call samples); a calibration without a
   positive `workload.calls` is refused up front instead of dividing
   by zero. `median_ns_per_call` is now per-call, with
   `p10_ns_per_call`/`p90_ns_per_call` alongside.
2. *Two ranges*: every summary node (each per-position entry,
   `per_decode_step`, `prefill`, `request_total`) now reports both —
   `p10_ns`/`median_ns`/`p90_ns` keep the round-1 meaning (the
   request-to-request spread of the per-request budgets computed from
   each shape's calibration median; one sample per measured request,
   per (measured request, position) for the step nodes), and the new
   `calibration_p10_ns`/`calibration_p90_ns` are the calibration's own
   sampling range — the same per-request/per-position budgets computed
   from each shape's calibration p10 and p90 samples. `per_shape`
   emits the formerly dead p10/p90 accumulators as
   `budget_p10_ns`/`budget_p90_ns`.
3. *Assumption labels*: the assumption texts were rewritten for the
   native-descriptor classification, the shared-arena carriers and
   `CalScope` — both round-1 caveats are gone (the "confined arena"
   one, resolved by the `CalScope` mirror in item A, calibration
   side; the "pointer-kind M indistinguishable" one, resolved by the
   native descriptor in item A) — and for the two-range reporting.

**A (calibration side). The carriers and result allocator were not
production's.** (a) The P/M argument carriers lived on the same
*confined* arena as the dylib lookup; production
`MLXScope.Holder` allocates every native handle on a process-lifetime
`Arena.ofShared()`. (b) The struct-return shapes took their results
from that same bare confined arena, while production pays a
*synchronized* shared-arena allocation plus per-scope
`LinkedHashSet` tracking (`MLXScope.Holder.allocate`) for every
struct-return downcall result. Closed: the ten P/M carriers (the
maximally-arity shapes need ten 8-byte arguments) are
process-lifetime shared-arena segments; the four struct-return shapes
allocate their results through a `CalScope` that mirrors
`MLXScope.Holder.allocate` (synchronized shared-arena allocation plus
the per-scope handle set); the dylib's `libraryLookup` keeps its own
confined arena open for the run (the dylib must stay loaded). The
measured effect is visible in §15.4: the struct-return shapes land at
~509–518 ns/call against 4.3–12.6 ns/call for the int-return shapes —
the per-call result allocation dominates, exactly as in production.

**C. The fixed-continuation `prefill_s` included its release window.**
In `runFixedContinuation`, `prefillS` was computed after the prefill
try/finally, so it absorbed the `prefill-release` window (the
activation scope's `close` and, in binding-count mode, its per-array
`mlx_array_free` downcalls) — while the controlled loop's prefill
timer already stopped at the evaluated prefill logits (frozen §13.3
boundary; the controlled path needed no change). Closed: `prefillS` is
now measured inside the prefill try, after `prefill-sync` (the
evaluated prefill logits) and before the release window runs, so the
release window's own overhead never enters `prefill_s` on either
adapter; the frozen timer boundaries are otherwise untouched.

**D. A release-window failure replaced the original exception.**
Round 1's explicit try/finally (round 1, item 1) ran
`observer.window(..., scope::close)` in a `finally`: when the
protected block threw *and* the release window threw, the
release-window failure propagated out of the `finally` and *replaced*
the original exception — try/finally does not attach, only
try-with-resources does, and the adapter no longer uses it. Closed:
the six release sites (pre / step / request, both adapters) now run
through `releaseWindow(observer, phase, position, scope, original)`:
the close still runs at the same wall-clock position try-with-resources
gave it; a release-window failure is attached to the in-flight
exception with `addSuppressed` (and propagates as-is when no
exception is in flight, as before).

**E. An `Error` from the window's work left the agent recording.**
The binding-count window hook caught `RuntimeException` from
`work.run()`; an `Error` (an `OutOfMemoryError`, an
`ExceptionInInitializerError` when a lazy holder class loads) bypassed
the catch and propagated with the agent still recording, and a
`stop()` failure was swallowed outright. Closed: both boundaries catch
`Throwable` — an `Error` from the work stops the recording like a
`RuntimeException` — and a `stop()` failure is attached to the work
failure with `addSuppressed` instead of swallowed (the agent is
already broken; the window's own failure is what the report must
carry).

**F. The floor loop was vectorizable.** Round 1's floor was
`sink ^= (k + 1) * C` — an XOR reduction whose per-iteration terms are
independent of each other, so a vectorizer can reassociate it into
independent even/odd chains (or across SIMD lanes) and fold the loop.
Closed: the floor is now a linear congruential recurrence
`sink = sink * C + (k + 1)` — every iteration depends on the previous
result (the multiply-add cannot be reassociated), so the loop cannot
be vectorized, while the per-iteration ALU work stays in the same
order as the C no-op bodies; the result lands in the volatile sink
like the batch results.

**Local evidence.** `:jmlx-benchmarks:test` green with 96 tests
(§15.5, round-2 counts); the `:jmlx-models` baseline suites green
(`BaselineAdaptersTest` 5, `BaselineWorkerProcessTest` 9,
`BaselineWorkerProtocolTest` 22). The calibration was re-measured at
the default workload and the estimate re-run end-to-end against two
real binding-count session reports with the worker-recorded
`binding_descriptors` injected — coverage 1.0000, zero uncovered,
per-decode-step FFM budget ≈ 3.5% of the measured step wall time
(full numbers and the source/binary SHA-256 in §15.4). The
`FfmTransitionEstimateTest` per-call test pins the division
(workload `calls = 10`, per-sample quantiles 10× the per-call values),
so the round-1 conflation cannot return silently.

### 15.8 Review round 3 (2026-11-09, PR #90): measurement fidelity, rounding and failure paths

Six findings (one medium, four low, one nit), each closed with a
test where the behavior is observable; the contract amendments are
the in-place edits to §15.3, §15.4 and §15.5 above. No report or job
schema changed; the estimate output schema stays
`jmlx-ffm-transition-estimate-1` (the first assumption's text was
amended to the per-sample allocator lifecycle — same schema, same
four-assumption array); the calibration summary schema and the
27-key shape set are unchanged.

1. **CalScope per sample (medium).** The struct-return shapes
   allocated their results through one fork-lifetime `CalScope`: the
   measured batches alone would have kept 4 × iterations × calls =
   1.6M segments (plus one more per struct shape per warmup round) in
   one tracking set and one arena for the whole fork. The set's
   doubling rehashes run on later insertions, and the growing live
   heap could put GC pauses into any shape's samples — both inside
   timed batches, biasing the p90 tails the calibration reads.
   Closed: every sample — warmup and measured — gets a fresh
   `CalScope` (the per-scope shared arena plus per-scope
   `LinkedHashSet`), created before its `t0` and closed after the
   sample, outside the timed region. This mirrors the production
   step-scope lifecycle exactly: production closes its step scope
   (arena release included) after every step, so the
   first-allocation reservation cost inside the timed region is
   faithful. The non-allocating shapes create and close a scope per
   sample too (the allocator goes unused; the create/close are
   off-timed), so no shape's timed batch differs in structure from
   production's.
2. **`releaseWindow` (low).** The release-window catch was
   `RuntimeException`, and a window that died before `scope::close`
   ran (the binding-count agent's `start()` failing) left the
   activation and request scopes to the Cleaner instead of closing
   them. Closed: the catch is `Throwable`; on any release failure the
   close is called directly (a nested try) — `MLXScope.close()` is
   idempotent, so a window that already ran the close pays nothing —
   and a fallback-close failure is attached to the release failure
   with `addSuppressed`, never masking the in-flight original.
   Package-private for the direct unit tests (6).
3. **Rounding bias (low).** The estimate rounded every
   (binding, window) `count × per-call cost` term to whole
   nanoseconds before summing — at per-call costs of tens of ns a
   ~50-binding step biases by up to ±25 ns. Closed: the per-shape
   split is now `perShapeCalls` (long) + `perShapeBudgets`
   (double[3]), and every window/request/pool accumulator is double;
   `Math.round` applies once, at each per-shape output value.
   `stepSummary`/`calibrationP10`/`calibrationP90` take
   `List<Double>`; the `doubleList` helper is gone. The
   `FfmTransitionEstimateTest` hand-computed values are exact on
   integer costs and stay green unchanged.
4. **Descriptor lookups (low perf).** The worker looked up the
   native descriptors of every instrumented binding (670 names in the
   Tier-A workloads) when assembling the report — each lookup
   initialises the binding's holder (`findOrThrow` plus
   `downcallHandle`) and links its downcall, so the report path
   linked every instrumented downcall for a classification set of at
   most 49 names. Closed: `countedBindingNames()` is the union
   of every measured request's per-name `total` with count > 0, and
   `nativeBindingDescriptors(signatures, counted)` (package-private)
   looks up only the counted ∩ signatures. A run that failed before
   any window completed carries no counted names (failed runs add no
   request records) and falls back to every instrumented name,
   keeping the nonempty-`binding_descriptors` invariant the report
   validator enforces for failed runs too. End-to-end the counted-only
   descriptors (49 of 670 for TinyMistral-6x248M, 39 of 670 for
   Qwen3-0.6B) reproduce the full-descriptor estimates exactly (minus
   `created_at`) on both workloads.
5. **`ffm_cal.c` comment (nit).** The carrier-fidelity comment
   described every by-value struct as "a single 8-byte pointer
   member": the opaque handles are, but the `mlx_optional_*` scalar
   wrappers are a value/has-value pair padded to 8 bytes. Comment
   fixed; the rebuilt dylib is bit-identical to the round-2 binary
   (comment-only change; the binary SHA-256 in §15.4 is unchanged,
   the source SHA-256 is the new one).
6. **Tests (low).** The D/E release-window and window-work error
   paths of §15.6/§15.7 had no direct tests (the end-to-end adapter
   runs never fail their windows). Closed: `BaselineAdaptersTest`
   +4 — a throwing `PhaseObserver` around a real scope close:
   observer dies before the close (the fallback close must have run),
   observer dies after a clean close (the fallback is the idempotent
   no-op), an `OutOfMemoryError` instead of a `RuntimeException`
   (caught, attached to the original, never propagated in its place),
   and no original in flight (the failure propagates); closed
   asserted via `newChild`'s `IllegalStateException`.
   `BaselineWorkerProtocolTest` +3 — a fake-agent binding-count run
   (the agent's state faked through its system properties, no
   `-javaagent` in-process: the model work executes real, uncounted
   downcalls, every window's counts come out empty, and the run fails
   at the non-empty-counts require) whose report must still carry a
   nonempty `binding_descriptors` — exactly {`mlx_add`, `mlx_eval`}
   with `(`-prefixed descriptor values through the real reflective
   `$descriptor` lookup; a direct counted-only-restriction test of
   `nativeBindingDescriptors` (a never-counted name is not looked
   up); and a window work `OutOfMemoryError` propagating unwrapped
   with the recording stopped (`jmlx.binding.recording` false).

**Local evidence.** `:jmlx-benchmarks:test` green with 96 tests (the
`FfmTransitionEstimateTest` hand-computed values unchanged under
double accumulation); the `:jmlx-models` baseline suites green
(`BaselineAdaptersTest` 9, `BaselineWorkerProcessTest` 9,
`BaselineWorkerProtocolTest` 25 — §15.5). The calibration was
re-measured at the default workload (source SHA-256
`cb30e80b9824eb51113efca5a608f704d9503421e9e7ee331a0eb3d5bc60e2d4`,
binary SHA-256
`f1e2e7d76b4ff25d610c4284f38b856044eee604e3b844c45f539f0d30901ad9`
— bit-identical to the round-2 pin, confirming the comment-only C
change, Apple clang 21.0.0): floor 2625 ns per 2000 calls (1.31
ns/iteration), int shapes 4.56–13.17 ns/call, struct shapes
52.2–55.1 ns/call. The ~10× struct drop against the round-2
509.0–517.9 ns/call is the expected correction (item 1's artifact
removed); the int shapes move only +5–8%, consistent with the
floor's +10% environmental shift. The estimate re-run end-to-end
against the same two binding-count session reports gives
per-decode-step budgets 76.65 µs / 106.68 µs against measured step
walls 14,437.2 µs / 20,162.5 µs (≈ 0.53%, vs round 2's ≈ 3.5%),
prefill 85.54 µs / 100.23 µs, request totals 9.82 ms / 13.65 ms
(calibration ranges 63.26–98.74 / 88.11–137.33 µs per step),
coverage 1.0 with zero uncovered — full numbers in §15.4.

### 15.9 Review round 4 (2026-11-09, PR #90): counted-name filtering, lookup seam and test hygiene

Four findings (all low), all in the `jmlx-models` test sources.
No job, report or estimate schema changed (the estimate output
schema stays `jmlx-ffm-transition-estimate-1`), and no calibration
artifact moved (binary SHA-256
`f1e2e7d76b4ff25d610c4284f38b856044eee604e3b844c45f539f0d30901ad9`,
source SHA-256
`cb30e80b9824eb51113efca5a608f704d9503421e9e7ee331a0eb3d5bc60e2d4`).

a. **`countedBindingNames` measured filter (low).** The Javadoc
   said "some measured request" while the loop iterated every
   request record — `requests` holds one `requestRecord` per
   index, whatever its kind (first + warmups + measured). The
   superset lookup set was harmless (the estimator classifies
   measured windows only) but the wording was false. Closed by
   filtering on `"measured".equals(record.get("kind"))` in
   `countedBindingNames()` — the stricter of the two offered
   fixes, so the Javadoc now holds. No estimate output changes:
   a name that appears only in a first/warmup window is neither
   classified nor uncovered, so the round-3 result (counted-only
   descriptors reproduce the full-descriptor estimates exactly)
   stands without a re-run.

b. **Descriptor-restriction test checked less than its comment
   (low).** `bindingDescriptorsAreRestrictedToTheCountedNames`
   asserted only the output keys; it would have passed even if
   the uncounted `mlx_array_new_data` holder had been
   initialised and its result then dropped. Closed by checking
   the restriction directly through a counting lookup seam:
   `nativeBindingDescriptors` carries a `descriptorLookups`
   counter incremented once per attempted reflective
   `$descriptor` invoke, with a package-private
   `descriptorLookupCountForTest()` accessor and a reset in
   `resetTestSeams()` (the file's existing `*ForTest` seam
   pattern). The test now asserts exactly 2 lookups for counted
   = {`mlx_add`, `mlx_eval`} while the signature map also
   carries `mlx_array_new_data`, so the "must not be looked up"
   comment is true.

c. **Agent properties left behind (hygiene).** Both fake-agent
   protocol tests cleared `jmlx.binding.installed` (and
   `jmlx.binding.signatures`) but left `jmlx.binding.recording`
   set to `false` — the agent's `start()` sets it `true` and its
   `stop()` sets it back to `false` and leaves it. Nothing reads
   it today; the finally blocks now clear it alongside the other
   properties. `jmlx.binding.count.*` is never created in a fake
   run (no transformer installed), so nothing else needs
   clearing.

d. **Positional 19-argument `Validated` constructor in a test
   (low).** `bindingCountWindowWorkErrorPropagatesUnwrapped`
   built the `Validated` record from 19 positional arguments,
   mostly null — a record-shape change would have broken it
   silently-but-noisily. Closed with a record factory
   `Validated.forInstrumentation(String instrumentation)` that
   fills the rest with the same null / `Map.of()` / `List.of()`
   values; the test calls it.

**Local evidence.** The `:jmlx-models` baseline suites are green
unchanged at `BaselineAdaptersTest` 9, `BaselineWorkerProcessTest`
9 and `BaselineWorkerProtocolTest` 25 (§15.5's counts — the tests
were strengthened, none added). `:jmlx-models:spotlessCheck` and
`:jmlx-models:checkstyleTest` are clean; `git diff --check` is
clean. No `jmlx-benchmarks` change, and no golden, schema, pin or
version change; the round-3 estimate numbers stand.

## 16. WP3 — Qualification and the headline text baseline (2026-11-09/10)

Cross-references: "plan section N" is `req/plans/phase8-2-plan.md`; a bare
"section N" is this findings file.

### 16.1 Code and host state

- Measured code: clean tree at cc0c603 (branch phase8-2-wp3, no untracked files —
  a local scratch script was moved outside the checkout so every worker record
  carries dirty: false; verified in all session.json code blocks). No code
  changed between the pilot, qualification, headline and diagnostics sessions;
  this section records the results in one pass afterwards.
- Host: M5 Max (128 GiB), macOS 27.2, Temurin 25.0.4.1 worker JVMs, the
  dedicated CPython 3.12.16 venv of tools/decode-baseline (mlx 0.32.4 /
  mlx-metal 0.32.4, matching the staged Java runtime, section 9).
- Timestamps: all hostrec and session.json times are UTC (the recorder uses
  date -u); the machine's local time is UTC+1.
- Host boots and quieting: the pilot session (19:27:50-19:48:19 UTC) ran on
  the pre-reboot host (a sustained ~1-core load from RemoteManagement
  screensharingd, 87-138% CPU with no visible remote session, was resolved by
  the owner with a reboot at 20:13 UTC; `last reboot` records 21:13 local).
  The qualification sessions (20:29:12-20:32:37 UTC) and headline session A
  ran on the post-reboot host (the post-boot mediaanalysisd reindex had
  finished before the first fork). The extra controlled session (A2) ran after
  the owner's next morning reboot (05:10 UTC; 06:10 local). The pre-reboot
  host evidence covers the planning pilot only; every qualification, headline
  and diagnostic sample comes from the post-reboot hosts.
- hostrec convention (plan section 4): pre/post records plus a 60 s poll per
  launcher session; the launcher's session.json carries the hostrec.pre/post
  nulls by design (WP1 shape), the operator records are hostrec.log in each
  session directory. hostrec records memfree, not a memory-pressure level.
- Host-record deviation from plan section 4 (recorded here; the plan lists
  a missing host record as a further-session trigger): hostrec captures, per
  launcher session, only the load averages, the therm-warnings count, omlx
  CPU and memfree — not the per-worker before/after records of memory
  pressure/swap and resident GPU processes with full command outputs. The
  full `pmset -g therm` outputs that tools/gpu-burn/hostrec.sh calls the
  primary thermal record exist only for the B3 and diagnostics blocks
  (codex-*-thermal-*.txt under ~/wp3-scratch); for sessions A through B2,
  "no thermal warnings" rests on the therm-warnings count signal (2 in every
  poll). No further session was triggered by this gap (the session caps were
  consumed or the cells settled); the gap is recorded for the owner's review
  in section 16.9.
- Launcher summary defect (fixed in section 16.8, after the measurement): at
  cc0c603 every session.json `summary.first_request` block holds only the last
  cell of each runtime (the per-runtime object was re-created for every cell).
  `summary.per_runtime` is unaffected. Every first-request figure in this
  section is recomputed from the raw worker records, never from that block.
- Raw-data digests: `req/plans/phase8-2-wp3-raw-digests.json` (schema v2,
  458 files, paths relative to `~/wp3-scratch`) records the SHA-256 of
  every session.json (33), report.json (344) and spec.json (33) under
  `sessions/`, the 33 per-session `hostrec.log` operator records (each
  byte-identical to the hostrec tool's own `hostrec/<session>/host.log`
  copy, verified 2026-11-11), `hostrec/a3-block/rsswatch.log`, the four
  `codex-{b3,diag}-thermal-{before,after}.txt` records, the
  pilot/qualification/diagnostic `hostrec/<block>/host.log` and
  `launcher.log` files, and the rejected first launch of session A
  (`hostrec/session-a-controlled/host.log` and `launcher.log`, section
  16.4) — tying this section, including its host-side claims, to the exact
  raw tree; every session.json code block is cc0c603 / dirty: false.

### 16.2 Budget refresh (controlled-adapter pilot, planning evidence)

One dedicated process per (case, length, runtime), controlled adapter, headline
mode, 1 load + 26 requests each (18 workers, one pair per cell, pair leaders
alternating): planning/qualification evidence only, never headline samples
(plan section 4.1). Session wall 20m30s, exit 0, 18/18 ok, 0 exclusions.

| Case | L | java req-s (median) | python req-s (median) | java step-ms | python step-ms | cell wall (6 java + 6 python worker walls) |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b | 32 | 0.565 | 0.500 | 3.989 | 3.641 | 0.08 h |
| qwen3-0.6b | 512 | 0.661 | 0.530 | 4.660 | 3.817 | 0.06 h |
| qwen3-0.6b | 2048 | 0.977 | 2.229* | 6.773 | 16.596* | 0.16 h |
| qwen3-8b-4bit | 32 | 1.399 | 4.350* | 10.267 | 32.884* | 0.27 h |
| qwen3-8b-4bit | 512 | 1.843 | 1.516 | 12.490 | 10.424 | 0.17 h |
| qwen3-8b-4bit | 2048 | 9.797* | 2.149 | 26.977* | 10.525 | 1.08 h |
| tinymistral-6x248m | 32 | 0.583 | 0.517 | 4.127 | 3.733 | 0.07 h |
| tinymistral-6x248m | 512 | 0.637 | 0.540 | 4.439 | 3.817 | 0.07 h |
| tinymistral-6x248m | 2048 | 0.880 | 0.666 | 5.542 | 4.078 | 0.09 h |

\* Degraded pilot workers on the pre-reboot host, not representative of the
clean regime: java 4bit/2048 9.80 s/request and 27.0 ms/step (worker wall
587 s) against the headline clean ~3.1 s and 17.5 ms; python 4bit/32 4.35 s
against ~1.27 s; python 0.6b/2048 2.23 s against ~0.61 s.

- The cell wall is 6 x (java worker wall + python worker wall) from the pilot's
  own worker records, so it includes load, the first request and the warmups.
  (An earlier draft used 12 x 26 x the median request per runtime, which
  counted 12 processes per runtime instead of 6 and left out load and the
  slower first/warmup requests; it happened to give a similar 4bit/2048 total.)
- Frozen manifest limits (worker 1800 s / reference-pair 720 s / cell 4.0 h /
  session 12 h) remain valid unchanged: the binding cell is 4bit-2048 at
  ~1.08 h (java 6 x 587 s = 0.98 h + python 6 x 61 s = 0.10 h), under the 4.0 h
  cell budget with ~3.7x margin even with the degraded pilot worker. The pilot
  java 4bit/2048 worker was degraded but stayed far from the section 7.3
  unprimed regime (27 ms/step against ~285 ms).
- Actual walls measured: pilot 20m30s; qualification 3m14s of worker sessions
  (1m46s TF32=1 + 1m28s TF32=0); session A 1h29m; all far inside the
  section 7.4 reservation.

### 16.3 Qualification re-run (separate processes, untimed)

Fixed-continuation mode (prefix512[32:64], 32 steps, per-step argmax_id/
top2_margin — the plan section 3.3 qualification workload), both runtimes, all
three cases, both TF32 modes (12 workers). Each worker ran 1 first + 5 warmup +
20 measured requests; the table compares the first and measured requests
(21 x 32 = 672 steps per runtime). The 5 warmups (160 further steps per
runtime) also match exactly, and their tie-eligible counts scale in the same
proportion (832/832 steps over all 26 requests).

| Case | TF32 | argmax match (java vs python) | tie-eligible (<0.05) | max margin diff | vs section 2.4 |
| --- | --- | --- | --- | --- | --- |
| qwen3-0.6b | 1 | 672/672 | 21 | 0.0 | re-held (ties fc-region only) |
| qwen3-0.6b | 0 | 672/672 | 21 | 0.0 | re-held |
| qwen3-8b-4bit | 1 | 672/672 | 0 | 0.0 | improved (2.4: 2/33 tie-eligible) |
| qwen3-8b-4bit | 0 | 672/672 | 0 | 0.0 | improved |
| tinymistral-6x248m | 1 | 672/672 | 63 | 8.926e-4 | re-held |
| tinymistral-6x248m | 0 | 672/672 | 84 | 4.768e-6 | re-held |

Divergence policy unchanged from section 2.4: the 4bit identity claim is
fixed-continuation (the matched-greedy claim was rejected there because of the
greedy-region top-2 ties); the fixed continuation re-verified with zero
tie-eligible steps at the 0.05 bound in both modes. 0.6b and tinymistral keep
their greedy claims (tie-eligible steps lie in the fixed-continuation region
only, as in 2.4). No golden regenerated, no tolerance widened. This
qualification covers the controlled adapters; the stock greedy comparison is
recorded with the stock sessions (section 16.5).

### 16.4 Headline session A — controlled adapter, TF32=1, primed-per-request

108 workers (9 cells x 6 pairs x 2 runtimes). The launcher's cell identity
(runtime, adapter, case, length, mode, instrumentation) carries no pair index,
so the execution unit was one launcher session per pair: six spec-order
sessions of 18 workers, pair leaders alternating by pair index (java-led
pairs 0/2/4, python-led 1/3/5 — 3/3 parity per cell). Block wall
20:44:52-22:13:39 UTC (1h29m), all 108 workers ok, 0 exclusions, every record
clean cc0c603 / dirty: false. Before pair0, a first launch of session A was
rejected (20:36:50-20:36:51, hostrec/session-a-controlled): a single
108-worker spec — one session for all six pairs — failed with "duplicate
cell" spec errors, the consequence of the pair-index-free cell identity
above; BUILD FAILED in 526 ms, no worker ran. It has no measurement impact
and does not count against plan section 4.1's restart rule.

Per-cell greedy-ID agreement (headline greedy, 128 tokens, 20 measured requests
x 6 workers per runtime, paired by pair and request index): 120/120 identical
ID sequences in all 9 cells (1080 comparisons). The 0.6b/tinymistral greedy
identity claims are confirmed by the headline runs themselves; the 4bit
greedy match re-holds as observation only (its frozen identity claim is
fixed-continuation, section 2.4).

Pairs aggregation (the launcher leaves summary.pairs empty for WP3; frozen
convention: per-pair warm median over the 20 measured requests, type-7
quantiles over the six pair medians, ratio = median over the pairs of the
per-pair java/python median ratio). full_generation_s and
streaming_decode_tps, java p10/median/p90 then python, then ratio(java/python):

| Cell | full_gen java (p10/med/p90) | full_gen python | ratio | tps java | tps python | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3-0.6b/32 | 0.557/0.558/0.559 | 0.491/0.494/0.503 | 1.1318 | 230.8/232.9/233.0 | 255.3/260.2/261.7 | 0.8912 |
| qwen3-0.6b/512 | 0.654/0.657/0.658 | 0.516/0.518/0.526 | 1.2655 | 199.3/200.4/201.4 | 248.1/251.5/252.9 | 0.7963 |
| qwen3-0.6b/2048 | 0.968/0.971/0.982 | 0.605/0.608/1.343 | 1.5963 | 139.2/141.7/141.9 | 145.9/227.1/227.8 | 0.6246 |
| qwen3-8b-4bit/32 | 1.373/1.377/3.370 | 1.267/1.271/2.930 | 1.0844 | 58.9/95.2/95.5 | 65.1/103.3/103.6 | 0.9217 |
| qwen3-8b-4bit/512 | 1.790/1.793/4.428 | 1.402/1.533/1.567 | 1.2783 | 39.3/77.9/78.0 | 91.6/92.9/101.6 | 0.7669 |
| qwen3-8b-4bit/2048 (io_affected) | 3.069/3.283/10.057 | 1.987/2.160/2.208 | 1.5372 | 14.0/49.1/52.1 | 87.9/89.3/94.1 | 0.5429 |
| tinymistral/32 | 0.580/0.582/0.585 | 0.516/0.517/0.517 | 1.1278 | 223.4/224.6/225.2 | 253.0/253.3/253.7 | 0.8851 |
| tinymistral/512 | 0.631/0.632/0.634 | 0.538/0.538/0.541 | 1.1734 | 209.8/210.6/211.0 | 246.9/248.1/248.3 | 0.8487 |
| tinymistral/2048 | 0.874/0.876/0.876 | 0.664/0.664/0.664 | 1.3185 | 170.1/170.3/170.8 | 233.3/233.5/233.6 | 0.7301 |

(per-step decode medians: java 4bit 10.1/12.2/17.5 ms at 32/512/2048 vs python
9.4/10.4/10.9 ms; 0.6b java 6.8 vs python 4.2 ms at 2048; tinymistral java
4.1-5.5 vs python 3.7-4.1 ms — python is faster per decode step on 4bit and
0.6b-2048, while java's per-request overhead is lower at short prompts.)

First-request (first-use, a separate distribution, never pooled with warm):
java vs python full_generation medians over the 6 workers per cell — 0.6b:
0.76/0.51, 0.85/0.54, 1.27/0.63 s; 4bit: 1.65/1.28, 2.03/1.42, 3.61/1.91 s;
tinymistral: 0.74/0.53, 0.80/0.55, 1.08/0.68 s (at 32/512/2048). Median load
per cell: java 0.03-0.20 s, python 0.17-0.43 s.

Host anomalies — three degraded windows tripped the plan section 4 >10%
adjacent-fork-median rule (an investigation trigger, not a discard):

- W1 20:46:43-21:04:32 (pair0), both runtimes, 5 workers: 0.6b/2048-py
  2.07 vs ~0.61 s, 4bit/32-ja 5.35 vs 1.38 s, 4bit/32-py 4.58 vs 1.27 s
  (decode-step medians 3.3-3.9x), 4bit/512-ja 2.72 vs 1.79 s (median step
  only 13.7 vs 12.2 ms), and 4bit/2048-ja (20:54:50-21:04:32): first request
  19.6 s, warmups 287.6 / 175.0 / 10.0 s — the section 7.3 degraded regime —
  then measured median 3.42 vs pair1's 3.08 s (+11.1%). Concurrent omlx-server
  CPU spike to 88.7% at 20:46:52; load stayed <=3.5. memfree 68->58% through
  20:54, then 28-39% from 20:55:53 to 21:02:53 with omlx-server idle
  (0.2-0.3%), recovering to 62% at 21:03:53 and 86% at 21:04:53 — the deepest
  trough coincides with the 4bit/2048-ja worker's own window.
- W2 21:21:18-21:38:21 (pair2), java only: 4bit/512-ja 6.14 vs 1.79 s
  (steps 42.0 vs 12.2 ms; prefill 0.433 vs 0.162 s) and 4bit/2048-ja 10.60 vs
  3.08 s (steps 28.3 vs 17.5 ms; prefill 1.019 vs 0.637 s); the adjacent
  python workers were normal. The 4bit/512 worker (21:21:18-21:23:21) ran at
  54-55% memfree (77% at 21:20:54, before it started), before the trough
  began. The 4bit/2048 worker's (21:24:23-21:38:21) first request and
  warmups ran inside the trough: memfree 28-50% sustained 21:24:54-21:33:55
  (the 21:28:54 sample read 50%), recovering from 21:34:55; load <=2.4. But
  its 20 measured requests (placement window 21:34:40-21:34:42 to
  21:38:21, section 16.4b) ran at 81-82% memfree — every hostrec sample
  covering the measured phase read 81-82% — uniformly 9.923-10.768 s
  (median 10.60, 3.4x the adjacent pair's 3.08 s), so that measured
  slowdown is not explained by the trough or by page-cache churn; whatever
  slowed them outlasted the memory pressure (an unverified guess).
- W3 22:08:06-22:11:21 (pair5), java only: 4bit/2048-ja 9.51 vs 3.06 s per
  request (3.1x). The median step (18.3 vs 17.5 ms) hides a heavy tail: the
  per-request step sums are 8.3-8.8 s against 2.4 s clean; prefill 0.840 vs
  0.624 s. memfree stayed at 75-78%, so W3 carries no memory signature.

The 10% rule also fires in session A without a degraded-window signature, on
the controlled python 4bit cells' per-pair regime (the same bimodal pattern
as the stock python cells of section 16.5): python 4bit/512 1.572->1.405
(-10.7%), 1.405->1.561 (+11.2%), 1.400->1.541 (+10.1%); python 4bit/2048
1.939->2.192 (+13.0%). These cells are already blocked by W1/W2/W3, so the
set of cells that needs a further session is unchanged.

No thermal warnings in any of the six host logs (therm-warnings=2 throughout),
and no jetsam/memorystatus events in the unified log for the windows. W1's
later part and W2's 4bit/2048 worker's first-request/warmup phase coincide
with memfree troughs (W1's deepest trough is inside the java 4bit/2048
worker's window; section 16.4b places the measured phases at or after the
recoveries); W2's 4bit/512 worker ran before its trough began, and W3 has
no memory signature. The W2/W3 pattern (java-only, prefill and decode steps
both inflated) is consistent with checkpoint page-cache churn of the
section 7.3 io_affected mechanism, but that attribution remains unconfirmed
(W2's measured-phase slowdown, which ran after the recovery, is a further
unexplained case; see section 16.4b for the candidate causes).

10%-rule outcome: the affected cells (qwen3-8b-4bit/32 both runtimes, /512
both runtimes, /2048 both runtimes; qwen3-0.6b/2048 python) cannot receive a
runtime attribution from session A alone — plan section 4 requires one further
alternating session before attribution. No sample was discarded; the
contaminated pair medians stay in the quantiles with these host observations
attached. The clean cells (tinymistral x3, 0.6b/32, 0.6b/512) are settled;
their pair-median spreads are under 1% in every cell except python 0.6b/32
(3.72%) and python 0.6b/512 (2.91%), both far below the 10% trigger and
without a degraded-window signature.

### 16.4a Extra controlled session (A2) for the four blocked cells

Three further alternating pairs (pairs 6-8; leaders java/python/java), the
four affected cells x 2 runtimes = 8 workers per session, 24 workers total, on
the post-morning-reboot host (boot 05:10 UTC). Block wall 05:23:56-05:59:58
UTC (36 min), 24/24 ok, 0 exclusions, clean cc0c603.

Deviation from plan section 4.1 (recorded here, not covered by a plan
revision): plan section 4.1 requires "the same sample counts and complete pairs
in each session" and plan section 4 requires each runtime to lead equally. A2
ran 3 pairs, not the cell's fixed 6, with 2 java-led and 1 python-led pair;
the cumulative leaders over sessions A + A2 are java 5 / python 4. Section
16.7 lists the outcomes that rest on this session as pending the owner's
ratification.

Per-pair full_generation_s medians (with decode step medians):
- qwen3-8b-4bit/32: pair6 1.380/1.334 (10.2/10.0 ms), pair7 1.377/1.268,
  pair8 1.370/1.270 — no degraded window; java shifts <1%, python pair6->7
  1.334->1.268 (-4.9%), below the 10% trigger.
- qwen3-8b-4bit/512: pair7 1.797/1.448 (12.2/9.7 ms), pair8 1.785/1.405 —
  clean; pair6 degraded, both runtimes (java 6.046, steps 43.5 ms; python
  4.597, steps 33.2 ms; 05:25:26-05:30:01).
- qwen3-8b-4bit/2048: pair7 3.126/2.235 (17.5/11.1 ms), pair8 3.082/2.168 —
  java clean; python pair6->7 1.970->2.235 (+13.5%); pair6 java measured
  median 3.580 (+14% vs pair7), its warmup requests showing the full section
  7.3 degraded regime decaying toward clean: first 25.1 s, warmups 310.2 /
  237.3 / 110.0 / 10.6 / 3.6 s (a 310 s request matches the section 7.1
  degraded 328 s cell wall); all 20 measured requests 3.48-5.40 s.
- qwen3-0.6b/2048: all three pairs clean (java ~0.97, python ~0.61).

Host evidence for the pair6 window: memfree 67% (05:27:56) -> 21%
(05:30:57), holding 31-40% through 05:40:57 with one 65% sample (05:31:57),
recovered from 05:41:57 (63%) — the same ~40-60 GiB consumer signature as
W1/W2, ~15 min after this morning's boot; load normal (<=2.9); no thermal
warnings. The 4bit/2048-ja worker's (05:30:01-05:43:08) first request and
warmups ran inside the trough (hence the 310 s warmup requests); its
measured requests began 7-9 s before the 05:41:57 recovery (estimated from
05:41:48-05:41:50, section 16.4b), at 63-68% memfree. The 4bit/512 workers
(both runtimes, 43-67% memfree) ran before the trough began.

Outcome per the plan section 4 rule:
- qwen3-8b-4bit/32 and qwen3-0.6b/2048: the further alternating session has
  no >10% shift -> the attribution block lifts; both cells aggregate 9 pair
  medians (each cell's contaminated session-A pair0 remains in the quantiles
  with the W1 host observations attached), subject to the A2 deviation above.
- qwen3-8b-4bit/512 and qwen3-8b-4bit/2048: A2 itself contains >10%
  adjacent-fork shifts (the pair6 window; for 2048 also the python
  pair6->7 shift), so they remain variable after this further session.

### 16.4b Second further session (A3) — recorded owner-approved extension

Per plan section 4.1, a session beyond the one the 10% rule required needs a
recorded run-policy/plan revision, not an automatic retry. This extension was
approved by the owner on 2026-11-10 before the fork, but it is recorded only
here, after the run, not in a plan revision: two further alternating pairs
(pairs 9-10; leaders python/java) over the two still-blocked cells
(qwen3-8b-4bit/512, /2048) x 2 runtimes = 8 workers, inside the frozen
per-cell budgets (no manifest or run_policy change; the manifest stays
digest-pinned and untouched). A3, like A2, ran fewer than the cell's fixed 6
pairs (plan section 4.1 deviation; cumulative leaders over A + A2 + A3:
java 6 / python 5). The stock B3 session of section 16.5b is the same kind
of third session and was run on the owner's handoff without a separately
recorded extension; both are listed for ratification in section 16.7. A
scratch-only host consumer watcher (top-10 RSS processes + memfree every
60 s, outside the verified checkout) runs during the block to identify the
recurring memory consumer if it strikes again. Rule: if A3 is clean (no >10%
adjacent-fork shift within it), both cells settle over 11 pair medians; if it
strikes again, both cells close as `unstable` (plan section 4.1) with the
ranges and host observations, and no further rerun happens.

A3 results. pair9 (python-led, 06:29:01-07:02:39 UTC) overlapped a fresh
strike of the memory consumer, which began at 06:33:01 (70 -> 31% memfree).
The python/512 worker (06:29:01-06:30:39) ran before the strike began, at
55-60% memfree; its measured median is degraded, 3.551 s (first 4.84 s) vs
the clean ~1.42 s, but the degradation is not attributable to the strike.
java/512 1.813 s (clean), python/2048 1.915 s (clean). **java/2048
(06:32:33-07:02:38) hit the 1800 s worker wall-clock timeout inside the
strike**: first request 14.8 s, warmups 264.8 / 155.9 / 10.9 / 3.4 / 3.5 s
(the section 7.3 degraded regime decaying toward clean), measured requests
1-6 clean at 3.5-3.7 s, then a second degradation from measured request 7
(11.1, 12.0, 14.7, 238.2, 319.3, 283.0, 282.7, 124.5, 11.7 s) until the
timeout after 15 measured requests. pair10 (java-led, 07:03:03-07:07:25)
was fully clean: java/512 1.799, python/512 1.417, java/2048 3.205,
python/2048 2.253 s.
A3 therefore contains >10% adjacent-fork shifts (python/512 pair9 vs pair10:
150%; python/2048 1.915->2.253, +17.7%; the java/2048 timeout), so per the
recorded rule above **both cells close as `unstable` (plan section 4.1) and
no further rerun happens** — except that plan section 4.1's own text makes a
cell with a timed-out worker `incomplete`, not `unstable` success, so
4bit/2048 defaults to `incomplete` there until the owner's dated scope
amendment (section 16.7, item 4). The timed-out worker is retained as an
incomplete pair with its reason (worker wall-clock timeout exceeded), never
silently rerun. Plan section 4.1 says a timeout "stops that session"; pair10
was a separate launcher session and still ran after the timeout (07:02:38 ->
07:03:03), a further recorded deviation.

Watcher findings on the memory consumer: the RSS watcher log starts at
06:44:01, missing the strike's onset (06:33:01) and its first 11 minutes;
from 06:44:01 until the worker timeout its memfree samples ranged 21-41%
and the top-10-RSS snapshots showed **no user process above 11 GB**
(omlx-server, the resident agent model); vm_stat showed ~47.6 GB wired
(16 KB pages) and swap had grown from ~13 GB (2026-11-09 evening) to ~48 GB
total / ~46 GB used. Wired growth of that size on this host is most
consistent with IOAccelerator/GPU buffer memory, but per-process wired
attribution is not available without root, so the owner of the growth is not
identified.

The hostrec memfree troughs align with the java 4bit/2048 worker windows
(memfree is host-wide, so this is correlation, not proof of cause):

| Window | java 4bit/2048 worker | memfree crash | recovery |
| --- | --- | --- | --- |
| A pair0 (W1) | 20:54:50-21:04:32 | 20:55:53 (58 -> 31%) | 21:03:53-21:04:53 |
| A pair2 (W2) | 21:24:23-21:38:21 | 21:24:54 (68 -> 32%) | 21:34:55 |
| A2 pair6 | 05:30:01-05:43:08 | 05:30:57 (63 -> 21%) | 05:41:57 |
| A3 pair9 | 06:32:33-07:02:38 | 06:33:01 (70 -> 31%) | (worker timeout) |
| B pair0 | 07:23:36-07:31:10 | 07:24:23 (71 -> 31%) | 07:30:23-07:31:23 |
| B2 pair6 | 17:08:27-17:16:57 | 17:08:31-17:09:31 (45 -> 35%) | 17:15:31-17:17:32 |

Every crash starts within about 60 s of a java 4bit/2048 worker starting.
In all five recovered cases the recovery begins 0.7-3.4 min before the
worker's own end (39 s for A pair0, 47 s for B pair0, 71 s for A2 pair6,
86 s for B2 pair6, 3.4 min for A pair2; A3 pair9's trough never recovered
in-window — the worker timed out). Placing each worker's requests on the
clock from its report (summing full_generation_s plus the recorded prime_s
— the primed-per-request mode excludes prime time from full_generation_s —
forward from utc_start and backward from utc_end) leaves only 2.0-2.4 s
unaccounted per worker (model load ~0.05 s), so the forward and backward
placements agree to within ~2 s. The recovery sample falls 7-36 s after
the estimated end of that worker's first-request plus warmup phase — the
start of the measured phase (B2 pair6's trough ended 9 s before it) — so
the measured phases of A pair0, A2 pair6 and B pair0 began 7-36 s before
their recovery sample (B2 pair6's began 9-12 s after it), with the hostrec
samples (60 s cadence) covering those phases reading 56-69%; the 60 s
cadence leaves at
most the first 28-37 s of A pair0's and B pair0's measured phases possibly
still in the trough's final transition. The same placement shows A pair2's
20 measured requests (10.6 s each, 3.4x clean) began 12-15 s before its
recovery sample (21:34:40-21:34:42 vs 21:34:55), and every hostrec sample
covering their run read 81-82% (section 16.4). A python
4bit/2048 worker cannot overlap a trough in the first place (the workers
run one at a time and every trough sits inside a java 4bit/2048 window),
and its ~60 s window admits at most one hostrec sample, so the useful
evidence is the python-led pairs, where the python 4bit/2048 worker ran
first at normal memfree: A3 pair9's python/2048 ran 06:31:35-06:32:33 (its
in-window sample 70%), 28 s before the 06:33:01 strike, clean at 1.915 s
(A2 pair7's ran 05:49:04-05:50:09, its in-window sample 88%, clean at
2.235 s). The 4bit/512 slowdowns form a different pattern: W2's java worker ran
at 54-55% memfree (77% at 21:20:54, before it started), A2 pair6's workers
(both runtimes) at 43-67%, and A3 pair9's python worker at 55-60% — all
before their troughs began. The data thus suggests two mechanisms rather
than one memory consumer.

Five of the seven degradation windows (section 16.4/16.5/16.5a windows,
including the non-trough W3) overlapped periods of heavy omlx-server
generation at some point, but W1's deepest trough (20:55-21:03), B-pair0
and B2-pair6 occurred while the serving agent was idle (all three inside a
java 4bit/2048 worker's window). An 8.5 h idle window between the two
afternoon blocks showed zero strike activity (0/515 watcher samples below
50% memfree). If the trough mechanism is the elevated wired pool, its
plausible effect is to churn the page cache and evict the 4-bit checkpoint's
(~4.6 GB) cached pages, degrading workers that run inside the window into
the section 7.3 SSD regime — a candidate explanation, not an established
one.

Two candidate causes remain for the trough mechanism, and the data now
weighs clearly toward the second:
- A host-level consumer, with concurrent omlx-server generation as a
  contributor where present. No trough sample showed a user process above
  11 GB, and the three troughs that occurred while the serving agent was
  idle (W1's deepest, B-pair0, B2-pair6) all sit inside java 4bit/2048
  worker windows.
- The benchmark workers' own Metal buffer pools (unverified hypothesis, now
  the favored one for the troughs). The section 16.6 memory blocks show the
  java controlled worker's pooled free device buffers (allocator-cached,
  mlx_get_cache_memory) at ~103 GB (96 GiB) at 4bit/2048 and ~28 GB at
  4bit/512, on a 128 GiB host, on top of a ~21.6 GB peak. Clean java
  4bit/2048 workers alone pull memfree down (A pair1 64% at 21:14:54, B
  pair1 55% at 07:40:24, B2 pair7 67% at 17:23:32). The RSS watcher cannot
  see pooled Metal buffers (the same worker's rss_peak is ~5.9 GB), so "no
  process above X GB" does not exclude the workers. If this cause holds, the
  4bit/2048 instability is a property of the java adapter's
  no-periodic-clearing allocator policy, not host noise, and 4bit/2048 is
  partly runtime-attributable — which is what owner item 7 of section 16.7
  now carries.
- The pre-trough 4bit/512/W3 slowdowns are the second mechanism, distinct
  from the troughs: A3-pair9's python/512 was degraded from its first
  request at 55-60% memfree, before its session's strike began (and before
  any 2048 worker of that session ran), and W3 degraded at 75-78% memfree.
  For them, the page-cache-churn hypothesis of section 7.3 remains the
  leading candidate; they are not evidence against the pool hypothesis for
  the troughs.

Per-process wired attribution is not available without root, so neither
cause is proven. Python memory counters were not collected (section 16.6),
so the two runtimes' pools cannot be compared from this data.

Consequence: 4bit/512 closes as unstable, with four degraded pairs;
4bit/2048 closes as unstable per the recorded run rule, with five degraded
pairs including the timeout, but defaults to `incomplete` under plan
section 4.1's text pending the owner's dated scope amendment (section 16.7,
item 4). All degraded pairs carry host observations:
- 4bit/512: A pair0 java, A pair2 java, A2 pair6 both runtimes, A3 pair9
  python.
- 4bit/2048: A pairs 0/2/5 java, A2 pair6 java, the A3 pair9 java timeout.

The clean clusters stay tight: 4bit/512 java 1.785-1.813 s (within 1.2% of
their median), 4bit/2048 java 3.059-3.205 s (within 3.3%). The 4bit/2048
io_affected label applies throughout.

### 16.5 Headline session B — stock adapter, TF32=1, primed-per-request

Same shape as 16.4 (6 launcher sessions x 18 workers, leaders alternating by
pair index, block wall 07:18:22-08:24:26 UTC, 66 min). 107/108 workers ok;
one exclusion: java stock 4bit/2048 pair0 was rejected by the report
validator (first request's full_generation_s led its last host-visible event
by +352 ms against the 50 ms tolerance; all 25 later requests passed at
1-7 ms) — the session-B strike below inflated its first request's final
processing. The report's own status is ok; the worker is retained with its
exclusion reason, never rerun in place (its re-run, if any, comes as a
complete pair, see 16.5a).

Strike: memfree 71% (07:23:23) -> 31% (07:24:23) holding 32-40% through
07:29, recovered 07:30:23-07:31:23 (56 -> 82%) (the sixth degradation
window; the watcher again showed no process above ~1.4 GB, which is the size
of the worker JVM itself). No other pair0 worker ran inside the window;
python 4bit/2048 ran 07:31:11-07:32:10, after recovery, fully clean. No
thermal warnings in any of the six host logs.

Pairs aggregation (type-7 over pair medians; full_generation_s and
streaming_decode_tps, java p10/med/p90 | python | ratio):

| Cell | full_gen java | python | ratio | tps ratio |
| --- | --- | --- | --- | --- |
| qwen3-0.6b/32 (6 pairs) | 0.532/0.533/0.535 | 0.395/0.395/0.397 | 1.3485 | 0.7372 |
| qwen3-0.6b/512 (6) | 0.632/0.633/0.635 | 0.423/0.424/0.426 | 1.4935 | 0.6592 |
| qwen3-0.6b/2048 (6) | 0.946/0.947/0.949 | 0.509/0.512/0.519 | 1.8516 | 0.5150 |
| qwen3-8b-4bit/32 (6) | 1.356/1.378/1.463 | 1.185/1.212/1.381 | 1.1238 | 0.8709 |
| qwen3-8b-4bit/512 (6) | 1.784/1.823/1.923 | 1.384/1.482/1.505 | 1.2728 | 0.7462 |
| qwen3-8b-4bit/2048 io_affected (5 complete pairs) | 3.082/3.144/3.159 | 2.029/2.136/2.147 | 1.4780 | 0.5451 |
| tinymistral/32 (6) | 0.561/0.561/0.563 | 0.457/0.458/0.458 | 1.2272 | 0.7754 |
| tinymistral/512 (6) | 0.611/0.612/0.612 | 0.467/0.468/0.468 | 1.3078 | 0.7402 |
| tinymistral/2048 (6) | 0.855/0.856/0.857 | 0.600/0.600/0.602 | 1.4245 | 0.6387 |

Every stock request emitted exactly 128 tokens with finish_reason length
(4,056 stock requests over sessions B, B2 and B3; no early-EOS occurrences;
the per-case stop set never fired on the fixed headline prompts).

**Stock greedy outputs diverge in five of the nine cells.** Pairing the
measured requests by pair and request index (as for the controlled check of
section 16.4):

| Stock cell | java vs python identical sequences | first divergent token, 0-based (B pair1) |
| --- | --- | --- |
| qwen3-0.6b/32 | 0/120 | 102 |
| qwen3-0.6b/512 | 0/120 | 6 |
| qwen3-0.6b/2048 | 0/120 | 5 |
| qwen3-8b-4bit/32 | 0/120 (B), 0/120 (B2), 0/120 (B3) | 6 |
| qwen3-8b-4bit/512 | 0/120 (B), 0/120 (B2) | 16 |
| qwen3-8b-4bit/2048 | 120/120 (B), 120/120 (B2) | — |
| tinymistral x3 | 120/120 each | — |

Within each runtime every request emits the same sequence. The stock java
sequences are identical to the controlled sequences, while the stock python
sequences differ from both, so the divergence lies in the stock mlx-lm loop.
The cause has not been investigated; it is not a token-count difference
(both runtimes emit 128 tokens). Plan section 3.3: "If greedy outputs
diverge, reject the matched-generation ratio for that cell pending review;
retain the mismatch". The five divergent cells' ratios are therefore
rejected as matched-generation ratios, pending review (section 16.7). Their
timing stays recorded as stock-default-loop observations.

10%-rule findings (per-pair medians, adjacent shifts; pairs 1-5 ran on a
quiet host, memfree 54-88% with single-poll dips only):
- qwen3-8b-4bit/32 python: a bimodal per-pair regime — 1.187 s in the three
  python-led pairs (1/3/5), 1.380 s in java-led pairs 2/4 (pair0 1.236
  intermediate); within-pair request distributions are flat (<1%). A
  systematic per-pair effect, not a degradation window (no strike in
  pairs 1-5). Four adjacent shifts of 14-16%.
- qwen3-8b-4bit/512 python: one +10.9% adjacent shift (pair1 low at 1.343 s,
  the other pairs 1.43-1.52 s), same per-pair-regime phenomenon, weaker.
- qwen3-8b-4bit/2048: no >10% shift among the five complete pairs (max
  +3.4% java, +9.0% python), but pair0's pair is incomplete (the excluded
  java member). Counting that member's validated measured median (3.504 s)
  fires the rule on pair0 -> 1 (-13.0%); the same is true in B2 (pair6 -> 7,
  -12.5%, section 16.5a), so the trigger fires whatever the owner decides
  about the memory warnings (section 16.7, item 5).
- All other cells: adjacent shifts <=2.6%.

Memory-warning trigger: plan section 4 also lists "a thermal/memory warning"
as a trigger, alongside the >10% shift. hostrec records memfree, not a
pressure level. If the strikes count as memory warnings, stock 4bit/2048
(the strike fell inside its pair0 java window) also needs a further session
before attribution. Section 16.5a applies the same reading to B2.

Outcome: per plan section 4, the two triggered cells and the incomplete 2048
pair require one further alternating session (2nd of the 3-session cap)
before runtime attribution, and the affected 2048 pair's re-run includes both
members. The clean stock cells (0.6b x3, tinymistral x3) have no timing
trigger; the 0.6b cells' ratios are nonetheless rejected under the greedy
divergence above.

### 16.5a Stock re-session (B2) for the three 4bit stock cells

Six fresh complete pairs (pairs 6-11, leaders continuing the alternation
java/python), 3 cells x 2 runtimes = 6 workers per session, 36 workers total
(the plan's fixed per-cell sample count; the owner chose the strict scope on
2026-11-10). Block wall 17:04:31-17:48:33 UTC (44 min). 35/36 ok, clean
cc0c603.

Strike (the seventh degradation window): memfree 61% (17:07:31) -> 45%
(17:08:31) -> 35% (17:09:31), holding 32-39% through 17:14, recovered
17:15:31-17:17:32 (56 -> 86%), bracketing the java stock 4bit/2048
pair6 worker's window (17:08:27-17:16:57). Its first request again violated
the 50 ms event consistency tolerance (+188 ms) -> invalid-report exclusion,
the second occurrence on exactly this worker (B pair0 was the first). The
watcher again showed no process above the ~1.4 GB worker JVM. Notably, this
strike (and B pair0's) occurred while the serving agent session was idle —
the owner parked the remaining blocks on a different (remote-inference) agent
so omlx-server stays quiet.

Results (per-pair measured medians, full_generation_s):
- qwen3-8b-4bit/32: java pair6 1.850, pairs 7-11 1.355-1.361; python
  1.187-1.195 in ALL six pairs (both leaders). The session-B python bimodal
  regime (1.187 in python-led vs 1.380 in java-led pairs) did NOT
  reproduce. pair6's java worker is the block's first worker after the
  8.5 h idle gap (memfree 57-71%, no strike): first request 4.90 s, warmups
  4.6-4.7 s, measured requests 1-7 at 4.54-4.74 s, then 2.34 / 3.77 /
  3.70 s, and the clean 1.36 s from measured request 11 (the block-start
  cold-cache regime, a mild version of the section 7.3 decay; the python
  worker starting after it was clean from its first request at 1.22 s). Its
  measured median 1.850 is a contaminated mixture, retained with this host
  observation.
- qwen3-8b-4bit/512: java 1.776-1.786 (all six, uniform), python 1.325-1.342
  (all six, uniform) — the session-B high cluster (1.48-1.52 in pairs 2-5)
  did not reproduce; no >10% adjacent shift inside B2 (max +0.8%).
- qwen3-8b-4bit/2048: java pairs 7-11 3.040-3.069 (tight; pair6 java
  excluded by the strike), python 1.902-2.035, max adjacent shift +7.0%
  among the included members. Counting the excluded pair6 java member's
  validated measured median (3.475 s) fires the >10% rule on pair6 -> 7
  (-12.5%); the same is true in session B (pair0 -> 1, 3.504 -> 3.050 s,
  -13.0%, section 16.5).

Outcomes per the recorded rules:
- qwen3-8b-4bit/512 (stock): the further session is clean -> the timing
  attribution block lifts; settled over 12 pair medians (the non-reproduced
  session-B high cluster stays in the quantile ranges, both runtimes). The
  cell's matched-generation ratio is nonetheless rejected under the greedy
  divergence of section 16.5.
- qwen3-8b-4bit/2048 (stock, io_affected): the >10% trigger fires on
  pair6 -> 7 counting the excluded member's validated measured median
  (-12.5%, above), so a third alternating session is required before
  attribution whatever the owner decides about the memory warnings — the
  section 4.1 cap allows one and it was not run, so the cell is blocked
  (section 16.7). Timing is reported over the 10 complete pairs (>= the
  6-pair minimum); the two strike-excluded java pair members (B pair0, B2
  pair6, same worker, same signature) are retained incomplete with their
  host observations. Two open points remain for owner review (section
  16.7):
  - Memory warnings: if the strikes count as memory warnings, B and B2 both
    carry one for this cell — an independent further-session trigger that
    changes nothing about the blocked status.
  - Exclusion bias: both excluded members were the slow ones (measured
    medians 3.504 / 3.475 s against 3.04-3.16 s, pair ratios 1.743 / 1.813).
    They were excluded by the first-request event check only; their 20
    measured requests passed validation. Over the 10 complete pairs the
    ratio is 1.5021 (tps 0.5313); including both pairs it is 1.5226
    (tps 0.5272).
- qwen3-8b-4bit/32 (stock): B2's block-start contaminated java pair median
  (1.850 vs 1.358, a -27% adjacent shift) means the further session is not
  fully clean, so the cell stays blocked after its second session. One
  further (third, final, cap-bound) session B3 (pairs 12-17, 12 workers)
  then ran under the remote-inference agent per the owner's handoff so the
  local serving agent stays quiet (section 16.5b): B3 is clean, and the
  cell's timing settles over the 18 pair medians. Its matched-generation
  ratio is rejected under the greedy divergence of section 16.5.

### 16.5b Final stock session (B3) for qwen3-8b-4bit/32

Six fresh complete pairs (pairs 12-17, leaders continuing the alternation —
java-led 12/14/16, python-led 13/15/17, 3/3 parity), 1 cell x 2 runtimes,
12 workers, run by the remote-inference agent per the owner's handoff so the
local serving agent stays quiet (block wall 18:21:24-18:32:50 UTC, 11 min).
12/12 ok, 0 exclusions, every record clean cc0c603 / dirty: false, the
preflight meeting the quiet-host thresholds without a wait. One benign runner
deviation, no measurement impact: the initial detached nohup shell died
before executing the loop (no partial data, no runner/launcher/worker
process, no session directory); the loop restarted under a persistent
session, no measurement was retried, no worker interrupted, no repository
edit.

- java pair medians 1.373 / 1.377 / 1.437 / 1.377 / 1.389 / 1.375 s
  (adjacent shifts -4.2 to +4.4%); python 1.199 / 1.195 / 1.264 / 1.252 /
  1.199 / 1.213 s (-4.2 to +5.8%). No shift over 10%; no decay signature
  (largest first-five/last-five measured-request ratio in the block:
  medians 1.030, means 1.028). pair14 is the most elevated pair in both
  runtimes; python pair15 (1.252 s) is also elevated.
- Host: memfree 90-92% in the 12 watcher samples of the window (89-96%
  across the 18 hostrec polls), no strike, therm-warnings 2 in every
  hostrec poll.
- Greedy output: 0/120 identical java/python sequences, as in B and B2
  (section 16.5).

B3 is clean, so the timing attribution block lifts and qwen3-8b-4bit/32
(stock) settles over the 18 pair medians (B pairs 0-5, B2 pairs 6-11, B3
pairs 12-17; B2-pair6's contaminated java member, 1.850 s, stays in the
quantile with its block-start cold-cache observation of section 16.5a). The
ratio below is a stock-loop timing observation, not an accepted
matched-generation ratio (section 16.5). Final aggregation (type-7 over the
18 pair medians):

| Metric | java (p10/med/p90) | python | ratio |
| --- | --- | --- | --- |
| ttft_s | 0.0430 / 0.0432 / 0.0445 | 0.0577 / 0.0579 / 0.0649 | 0.7429 |
| full_generation_s | 1.3564 / 1.3740 / 1.4525 | 1.1869 / 1.1953 / 1.2992 | 1.1375 |
| streaming_decode_tps | 90.33 / 95.54 / 96.79 | 103.58 / 112.17 / 112.95 | 0.8604 |

### 16.6 Memory and plan section 5.1 diagnostics (ineligible for headline)

The diag-5-1 spec: 18 java workers — every cell x {binding-count,
tracing}, controlled adapter, headline mode, 1 first + 5 warmup + 20
measured requests per worker, TF32=1. Block wall 18:33:47-18:50:58 UTC
(17m11s), run by the same remote-inference agent immediately after B3,
on the same quiet host. 18/18 ok; exclusions=18 reason=instrumented, by
design — instrumented workers are ineligible for the headline
(plan section 5.1), and no headline sample comes from this block. Loads
0.034-0.198 s (warm page cache). The payload (91,353,713 bytes, the 18
worker objects) verifies exactly against the original worker objects
after removing only the request fields tokens/event_s/steps.

Host: 17 watcher samples, memfree 44-96%. Two mild low-memory samples
(18:40:31 46%, 18:46:31 44% — a mild event, not a full strike: the full
strikes of section 16.4b reached trough floors of 21-32% and held below
50% for 6-11 min, while these never fell below 44% and each lasted one 60
s cadence sample; onset value alone would not separate them, since B2
pair6's onset sample read 45%) both overlap
the qwen3-8b-4bit/2048 binding-count worker's window (18:39:43.732-
18:46:48.309 UTC); the 60 s cadence does not establish exact onset/end
(bracketing samples 82% at 18:39:31 and 79% at 18:45:31). Swap used
grew 305.5 MB (pre-block) to 1,505.7 MB (18:41:47) and 1,158.6 MB
(block end) — the same pressure signature, milder than a full strike.
No worker was interrupted; no thermal warnings. Consequence: that
worker's measured median is 12.362 s against 3.136 s for the same
cell's tracing worker (18:46:49-18:48:23, after the window, clean). The
count-based FFM budget of that worker is unaffected — the downcall
counts are deterministic graph properties, identical across all three
prompt lengths within a model — but its wall times and its memory block
(rss_peak 1,290 MB, far below the tracing worker's 5,864 MB and the
other 4bit cells) are suspect and excluded from the evidence; the
4bit/2048 memory block is taken from the tracing worker.

**FFM transition estimate.** The committed FfmTransitionEstimate (schema
jmlx-ffm-transition-estimate-1) runs over each binding-count report with
the WP2 calibration (build/ffm-cal/ffm-calibration.json;
workload.calls=2000, so every per-call cost is the per-sample quantile
divided by 2000). Conventions: the prefill budget is the request's
prefill-* windows; the per-step budget is the four step-phase windows
(construction/sync/selection/release) at one (request, position > 0),
pooled over the 20 measured requests x 127 positions (n=2540); the
request total adds request-release; the calibration range is the type-7
p10 of the budgets computed from the calibration p10 samples and the
type-7 p90 of the budgets from its p90 samples; measured requests only.
It is a calibrated estimate, not a causal partition (the section 15.4
assumptions apply). Coverage is 1.000000 in all 9 cells, with zero
uncovered bindings: every one of the 39 descriptor forms maps into the
27 calibration shapes (across the three models the workload touches all
27), including the 4bit additions mlx_dequantize (PMMMMMPMMM)I (128
calls/request) and mlx_quantized_matmul (PMMMMZMMPM)I (32,384
calls/request) and tinymistral's mlx_array_new_float32 (F)M (1,536
calls/request). The estimate reproduces the section 15.4 end-to-end
baseline exactly on the two non-quantized models — per-step 76.65 /
106.68 µs, prefill 85.54 / 100.23 µs, request total 9.820 / 13.649 ms —
so the trimmed payload and the estimate path are identical to WP2's.

The per-request FFM budget is length-independent (identical at
32/512/2048 within a model: the graph's downcall count is a fixed
property and the 127 decode steps are fixed by the controlled
max-tokens, not the prompt length):

| Case | per-decode-step (cal p10-p90) | prefill (cal p10-p90) | request total (cal p10-p90) |
| --- | --- | --- | --- |
| qwen3-0.6b | 106.68 µs (88.11-137.33) | 100.23 µs (82.68-129.00) | 13.649 ms (11.273-17.571) |
| qwen3-8b-4bit | 111.12 µs (91.72-143.74) | 102.37 µs (84.37-132.45) | 14.214 ms (11.733-18.388) |
| tinymistral-6x248m | 76.65 µs (63.26-98.74) | 85.54 µs (70.53-110.21) | 9.820 ms (8.105-12.650) |

Downcall counts per measured request (identical across the 20):
prefill-construction 4,948 / 4,567 / 3,980; prefill-sync 3;
prefill-selection 14; prefill-release 1,145 / 1,215 / 1,009; per decode
step construction 5,315 / 5,062 / 3,719, sync 3, selection 14, release
1,183 / 1,269 / 859; request-release 56 / 72 / 24 (0.6b / 4bit /
tinymistral). Totals 16,671,420 / 16,241,340 / 11,771,900 calls per
request. The budget is dominated by the ()M no-parameter struct-return
shape (the array constructor, whose per-call result-segment allocation
is the 52.24 ns/call one): 8.28 / 8.96 / 5.91 ms of the per-request
budget (61 / 63 / 60%), then the (M)P, (M)I and (M)J array-handle
shapes (1.20-1.73, 0.66-0.94 and 0.53-0.77 ms); the 4bit budget adds
the quantized-matmul shape (10.98 ns/call, 0.36 ms/request) on top of
the same graph shapes.

**Tracing.** Caller-thread CPU/wall windows per phase (median over the
20 measured requests); the unassigned residual is the request wall/cpu
minus the sum of the phase windows:

| Cell | wall/cpu ms | residual wall/cpu ms | prefill-sync µs | step constr/sync/sel/rel µs | req-release µs |
| --- | --- | --- | --- | --- | --- |
| 0.6b/32 | 559.5 / 236.9 | +1.97 / +1.61 | 4,818.6 | 366.0 / 3,557.2 / 252.7 / 116.6 | 811.2 |
| 0.6b/512 | 662.9 / 243.3 | +1.88 / +1.55 | 14,393.7 | 386.5 / 4,264.4 / 254.4 / 124.9 | 829.0 |
| 0.6b/2048 | 975.9 / 266.2 | +2.25 / +1.80 | 72,328.9 | 394.7 / 6,286.5 / 262.1 / 127.0 | 842.5 |
| 4bit/32 | 1,555.5 / 320.3 | +2.47 / +1.90 | 48,764.7 | 475.9 / 10,847.3 / 295.0 / 161.2 | 1,083.2 |
| 4bit/512 | 1,865.6 / 342.6 | +2.88 / +2.24 | 172,164.8 | 482.6 / 12,129.0 / 268.7 / 175.8 | 1,087.0 |
| 4bit/2048 | 3,138.3 / 402.4 | +4.10 / +3.09 | 669,804.2 | 552.9 / 16,980.4 / 275.0 / 224.0 | 1,112.6 |
| tinymistral/32 | 586.7 / 213.9 | +1.62 / +1.35 | 15,320.9 | 334.1 / 3,804.9 / 209.9 / 100.0 | 356.1 |
| tinymistral/512 | 637.5 / 211.5 | +1.63 / +1.37 | 27,473.9 | 320.0 / 4,108.5 / 212.1 / 101.1 | 352.9 |
| tinymistral/2048 | 883.9 / 224.5 | +1.88 / +1.57 | 128,775.8 | 328.2 / 5,220.4 / 230.0 / 100.3 | 354.3 |

The per-step sync (the per-token eval/sync) dominates the step wall
(3,557 µs at 0.6b/32 to 16,980 µs at 4bit/2048); the per-step
construction (graph building) is a fixed 0.32-0.55 ms and the
selection/release phases 0.10-0.23 and 0.10-0.22 ms. prefill-sync grows
with the prompt (0.6b: 4.8 / 14.4 / 72.3 ms at 32/512/2048; 4bit:
48.8 / 172.2 / 669.8 ms). The unassigned residual (the span between
phase windows: inter-phase bookkeeping and token emission) is a small
positive +1.6 to +4.1 ms wall per request, growing with length.

**Memory blocks** (medians over the 20 measured requests, the section
10 shape; MB = 10^6 bytes). active is identical in the two workers of
every cell. peak differs by at most 1,024 bytes (tinymistral/32 and
/512) and allocator-cached by at most 0.25% (0.6b/32, 0.6b/512,
tinymistral/32 and /512; 0.6b/512 7,338.6 vs 7,320.2 MB); the table
shows the binding-count worker's values. rss_peak differs by the counting
agent's JVM overhead (~0.3 GB for 0.6b/4bit, ~0.4 GB for tinymistral).
The 4bit/2048 binding-count rss is starred (suspect, above).

| Cell | active MB | peak MB | allocator-cached MB | rss_peak MB (binding-count / tracing) |
| --- | --- | --- | --- | --- |
| 0.6b/32 | 1,503 | 1,574 | 794 | 3,032 / 2,740 |
| 0.6b/512 | 1,503 | 2,660 | 7,339 | 3,047 / 2,731 |
| 0.6b/2048 | 1,503 | 5,870 | 26,190 | 3,030 / 2,727 |
| 4bit/32 | 4,608 | 5,040 | 4,174 | 6,166 / 5,868 |
| 4bit/512 | 4,608 | 8,846 | 28,043 | 6,157 / 5,871 |
| 4bit/2048 | 4,608 | 21,570 | 103,258 | 1,290* / 5,864 |
| tinymistral/32 | 4,012 | 4,111 | 716 | 2,070 / 1,662 |
| tinymistral/512 | 4,012 | 5,726 | 3,805 | 2,067 / 1,659 |
| tinymistral/2048 | 4,012 | 11,035 | 14,563 | 2,073 / 1,660 |

active is the resident GPU working set (the 4bit checkpoint at 4,608
MB in all three lengths, plus the small 128-token KV remainder) and is
flat in length within a model; peak carries the prefill/decode
transients (21,570 MB at 4bit/2048); allocator-cached
(mlx_get_cache_memory, the pooled free device buffers) grows with the
prefill transient size (0.6b: 794 -> 26,190 MB; 4bit: 4,174 -> 103,258
MB, i.e. ~96 GiB of the host's 128 GiB; tinymistral: 716 -> 14,563 MB).

Memory-run gap: these Java blocks come from the instrumented diagnostic
workers. No Python worker recorded a memory block in any session, and no
headline worker records one (memory is null by design, plan section 4).
WP3 calls for separate memory runs, with Python's counters matched against
the native counter semantics (plan section 4). Those runs are not done; the
gap is listed in section 16.7.

### 16.7 Session outcomes

Final status per plan section 4.1, with the review-round-1 corrections of
section 16.8. Items marked "pending" need an owner decision (a dated scope
amendment or plan revision) before WP3's gate can pass on them.

| Adapter | Cell | Status |
| --- | --- | --- |
| controlled | 0.6b/32, 0.6b/512, tinymistral x3 | `stable` (session A, 6 pairs) |
| controlled | 0.6b/2048, 4bit/32 | `stable`, pending ratification of the A2 reduced-session deviation (section 16.4a) |
| controlled | 4bit/512 | `unstable` (A + A2 + A3) |
| controlled | 4bit/2048 io_affected | `incomplete` by default (one timed-out worker; plan section 4.1: a timeout "is incomplete, not unstable success"); the recorded run rule of section 16.4b closes it as `unstable` — pending the owner's dated scope amendment (item 4 below) |
| stock | tinymistral x3 | `stable` (greedy-identical) |
| stock | 4bit/2048 io_affected | the >10% trigger fires counting the excluded members' validated medians (sections 16.5/16.5a) whatever the memory-warning decision; timing reported over the 10 complete pairs; blocked (unrun third session — pending owner decision, item 5 below) |
| stock | 0.6b x3, 4bit/32, 4bit/512 | greedy-divergent: matched-generation ratio rejected pending review (plan section 3.3, section 16.5); timing recorded as stock-loop observations |

One cell remains blocked by an unrun session: stock 4bit/2048. The >10%
adjacent-fork trigger fires in B2, the cell's second session, counting the
excluded members' validated measured medians (section 16.5a), whatever the
owner decides about the memory warnings; plan section 4 then requires one
further alternating session before attribution, the section 4.1 cap allows
a third, and none was run — so stock 4bit/2048 is blocked unless the owner
decides otherwise (item 5 below). The single timed-out controlled worker
(controlled 4bit/2048, A3 pair9 java) is retained as the incomplete member
of its pair: the recorded run rule of section 16.4b closes that cell as
`unstable` over the 10 complete pairs, but plan section 4.1's timeout rule
defaults it to `incomplete` (item 4 below), and no further session is
proposed for it. The two strike-excluded stock java members (B pair0, B2
pair6) are retained in the 4bit/2048 stock cell (section 16.5a).

**Controlled adapter (TF32=1, primed per request).**
full_generation_s, java p10/median/p90 then python, then
ratio(java/python); final pair count per cell (6 pairs = session A only;
9 pairs = A + A2 — the 9-pair rows supersede the session-A values of
section 16.4 for those two cells).

| Cell | pairs | full_gen java | full_gen python | ratio | tps ratio |
| --- | --- | --- | --- | --- | --- |
| qwen3-0.6b/32 | 6 | 0.557/0.558/0.559 | 0.491/0.494/0.503 | 1.1318 | 0.8912 |
| qwen3-0.6b/512 | 6 | 0.654/0.657/0.658 | 0.516/0.518/0.526 | 1.2655 | 0.7963 |
| qwen3-0.6b/2048 (pending A2) | 9 | 0.969/0.969/0.976 | 0.606/0.612/0.906 | 1.5863 | 0.6241 |
| qwen3-8b-4bit/32 (pending A2) | 9 | 1.372/1.377/2.184 | 1.267/1.270/1.983 | 1.0841 | 0.9220 |
| tinymistral/32 | 6 | 0.580/0.582/0.585 | 0.516/0.517/0.517 | 1.1278 | 0.8851 |
| tinymistral/512 | 6 | 0.631/0.632/0.634 | 0.538/0.538/0.541 | 1.1734 | 0.8487 |
| tinymistral/2048 | 6 | 0.874/0.876/0.876 | 0.664/0.664/0.664 | 1.3185 | 0.7301 |

Unstable/incomplete cells: no ratio is attributed. Pair-median ranges,
type-7 p10-p90, over the 11 complete pairs for 4bit/512 (`unstable`) and
the 10 complete pairs for 4bit/2048 (`incomplete` by default, `unstable`
per the section 16.4b run rule — the incomplete A3-pair9 is excluded from
the 4bit/2048 range): full_generation_s java 1.789-6.046 s / python
1.405-3.551 s for
4bit/512, java 3.077-9.623 s / python 1.967-2.237 s for 4bit/2048
(io_affected throughout); streaming_decode_tps java 22.4-78.0 / python
38.3-101.4 and java 14.5-52.2 / python 87.6-95.2 respectively. Including
A3-pair9 (its partial java median over 15 measured requests, 11.682 s, and
python 1.915 s), the 4bit/2048 full_generation_s range is java
3.079-10.600 s / python 1.939-2.235 s. The clean clusters stay tight
(4bit/512 java within 1.2%, 4bit/2048 java within 3.3% of their medians);
every degraded member carries its host observation (section 16.4b).

**Stock adapter (TF32=1, primed per request).** Final pair counts
(6 / 18 / 12 / 10 complete pairs). The ratios of the five greedy-divergent
cells are stock-loop timing observations, not accepted matched-generation
ratios:

| Cell | pairs | full_gen java | full_gen python | ratio | tps ratio |
| --- | --- | --- | --- | --- | --- |
| qwen3-0.6b/32 (greedy-divergent) | 6 | 0.532/0.533/0.535 | 0.395/0.395/0.397 | 1.3485 | 0.7372 |
| qwen3-0.6b/512 (greedy-divergent) | 6 | 0.632/0.633/0.635 | 0.423/0.424/0.426 | 1.4935 | 0.6592 |
| qwen3-0.6b/2048 (greedy-divergent) | 6 | 0.946/0.947/0.949 | 0.509/0.512/0.519 | 1.8516 | 0.5150 |
| qwen3-8b-4bit/32 (greedy-divergent) | 18 | 1.356/1.374/1.453 | 1.187/1.195/1.299 | 1.1375 | 0.8604 |
| qwen3-8b-4bit/512 (greedy-divergent) | 12 | 1.779/1.785/1.915 | 1.329/1.342/1.489 | 1.3269 | 0.7207 |
| qwen3-8b-4bit/2048 io_affected (pending) | 10 | 3.046/3.063/3.156 | 1.923/2.032/2.140 | 1.5021 | 0.5313 |
| tinymistral/32 | 6 | 0.561/0.561/0.563 | 0.457/0.458/0.458 | 1.2272 | 0.7754 |
| tinymistral/512 | 6 | 0.611/0.612/0.612 | 0.467/0.468/0.468 | 1.3078 | 0.7402 |
| tinymistral/2048 | 6 | 0.855/0.856/0.857 | 0.600/0.600/0.602 | 1.4245 | 0.6387 |

The 4bit/32 row is the 18-pair aggregation of section 16.5b
(superseding the 6-pair 1.1238/0.8709 of section 16.5); the 4bit/512 and
4bit/2048 rows supersede the session-B values 1.2728/0.7462 and
1.4780/0.5451 of section 16.5. With the two excluded 4bit/2048 pairs
included, that cell's ratio is 1.5226 (tps 0.5272).

Session-count cap (at most 3 alternating sessions per cell): fully
consumed by stock 4bit/32 (B + B2 + B3, settled in B3). Controlled
4bit/512 and 4bit/2048 used A + A2 + A3 (A3 is the owner-approved
extension recorded in section 16.4b), and counting A3 pair10 — run after
the pair9 timeout — as its own session under plan section 4.1's "any
restart consumes one of the remaining sessions" (owner item 3), they used
four, over the cap; pair10's medians feed the 11-pair (4bit/512) and
10-pair (4bit/2048) ranges. Every other cell used 1 session; controlled
0.6b/2048 and 4bit/32 and stock 4bit/512 and 4bit/2048 used 2.

Optional observed-cache reference pairs (section 7.3): not run, recorded
as skipped with reason — the strikes of sections 16.4-16.5b re-created
the unprimed ~285 ms/step regime around every big 4-bit window, and the
per-cell cap was consumed where needed, so an unprimed reference pair
could not have run on the same host state as the primed workers; the
primed-regime headline question (the settled ratios above) does not
depend on it.

Correctness and no-discard: the controlled greedy-ID identity
(120/120 per cell, 1080 comparisons, session A, section 16.4; A2 and A3
also identical) holds in all 9 cells; the 4bit identity claim is the
frozen-manifest fixed-continuation one (the greedy match is
observation-only, section 2.4). Every stock request emitted 128 tokens with
finish_reason length, but stock greedy outputs diverge between the runtimes
in five cells (section 16.5). No headline sample was discarded: the
W1/W2/W3-contaminated pair medians, B2-pair6's block-start java member
(1.850 s), the strike-excluded B-pair0 and B2-pair6 java workers and the
A3-pair9 java timeout are all retained with their reasons and host
observations, and the 10% adjacent-fork rule acted as an investigation
trigger, never as a discard.

Open items for the owner before WP3's gate (plan section 6, WP3):
1. Stock greedy divergence (five cells): review the stock mlx-lm loop
   divergence and either accept the stock ratios with a dated scope
   amendment or keep them rejected (plan section 3.3).
2. Reduced-size further sessions: ratify or reject A2 (3 pairs, 2/1
   leaders) and A3 (2 pairs), and the cumulative leader imbalance, against
   plan section 4.1's same-sample-count rule. A2 decides controlled
   0.6b/2048 and 4bit/32.
3. Third sessions: record the A3 extension and the B3 third session as
   plan/run-policy revisions, and record that A3 pair10 ran after the pair9
   timeout. Under plan section 4.1 ("any restart consumes one of the
   remaining sessions"), the post-timeout A3 pair10 is a fourth session for
   the two controlled 4bit cells, over the three-session cap; its pair
   medians feed the 11-pair (4bit/512) and 10-pair (4bit/2048) ranges.
4. Controlled 4bit/2048: the status defaults to `incomplete` (plan section
   4.1: a timeout "is incomplete, not unstable success", and "incomplete
   required cells need completion or a dated owner scope amendment"); the
   recorded run rule of section 16.4b closes it as `unstable` over the 10
   complete pairs. Either complete the cell or confirm `unstable` with a
   dated scope amendment.
5. Stock 4bit/2048: the >10% adjacent-fork trigger fires counting the two
   excluded members' validated measured medians (B pair0 -> 1, -13.0%; B2
   pair6 -> 7, -12.5%), so it fires whatever the owner decides about the
   memfree strikes counting as plan section 4 memory warnings. Decide (a)
   whether the strikes count as memory warnings, and (b) whether the two
   first-request event-check exclusions stand, given their bias (1.5021 vs
   1.5226 over the 10 complete pairs).
6. Memory runs: the separate memory runs and Python memory counters WP3
   requires are not done (section 16.6).
7. 4bit memory consumer: the worker-allocator-pool hypothesis of section
   16.4b is open; the hostrec troughs align with the java 4bit/2048 worker
   windows (the section 16.4b table), which weighs toward it, and the
   question bears on whether the 4bit/2048 instability is host noise or
   runtime-attributable.

### 16.8 Review round 1 (2026-11-10, PR #91)

The review recomputed every table from the raw session records under
`~/wp3-scratch/sessions` (per-pair medians, type-7 quantiles, per-pair ratio
medians, the qualification table, the B3 table and the diagnostics windows).
The tables reproduce exactly apart from the 3-decimal rounding slips fixed
in review round 2 (section 16.9). This round amends section 16 in place:

- Stock greedy identity, which the first version did not check: five stock
  cells diverge (section 16.5). Their ratios are relabeled per plan section
  3.3, and section 16.7 now gives per-cell statuses and open owner items.
- W1 now includes the session-A pair0 java 4bit/2048 worker, which the first
  version called clean. It was degraded (warmups 287.6 / 175.0 s, measured
  +11.1%) and coincides with the deepest memfree trough. The memfree
  correlation statement is corrected accordingly.
- Section 16.4b records the worker-allocator-pool hypothesis for the memory
  consumer, with the evidence for and against it.
- Recorded deviations: A2/A3 sample counts and leader parity (A2 is 2/1, not
  3/3), the post-timeout A3 pair10, the timeout-versus-`incomplete` question,
  and the memory-warning trigger with the stock 4bit/2048 exclusion bias.
- Previously unreported >10% shifts in controlled python (session A 4bit/512
  and 4bit/2048, A2 and A3 4bit/2048). The set of cells that needed further
  sessions is unchanged.
- The unstable 4bit/2048 range is now stated as over the 10 complete pairs,
  with the pair9-inclusive range added. The degraded-pair counts are
  corrected from "five and two" to four and five.
- Memory blocks: the unit is MB (10^6 bytes), not MiB, and the
  binding-count/tracing difference is stated exactly instead of "identical".
  The Python memory-run gap is recorded.
- Pilot: the cell-wall column is recomputed from worker walls (the first
  version counted 12 processes per runtime), and degraded pilot cells are
  marked.
- Qualification ran on the post-reboot host, not the pre-reboot one.
- Smaller corrections: B3 has six pairs, not three; the A2 4bit/32 python
  shift is -4.9%; the B3 shift ranges; first-request 4bit/2048 python median
  1.91 s; per-cell load ranges; the qualification request mix; W3 is no
  longer called mild; W1's step ratios; the A3-pair9 request sequence;
  clean-cluster spreads; the plan-versus-findings cross-reference convention;
  µs units.
- Launcher defect (code change after the measurement; the measured code
  remains cc0c603): `BaselineBenchmark.finishFirstRequest` re-created each
  runtime's `summary.first_request` block for every cell (Jackson's
  `putObject` replaces an existing property), so only the last cell
  survived. It now reuses the runtime block. The new
  `BaselineBenchmarkTest.firstRequestSummaryKeepsEveryCellOfARuntime` runs two
  cells of one runtime and fails without the fix. The archived session.json
  files keep the defective block; section 16 never used it.

### 16.9 Review round 2 (2026-11-11, PR #91)

The review recomputed the tables from the raw records again (344 reports,
33 session.json files, all cc0c603 / dirty: false; the qualification and
B3 tables, greedy identity and first-request medians reproduce) and checked
hostrec memfree against the worker windows. This round amends section 16
in place:

- Code (after the measurement; the measured code remains cc0c603): the
  round-1 `finishFirstRequest` fix made that method functionally identical
  to `finishQuantiles`; the method is deleted and the first-request block
  now goes through `finishQuantiles`. The regression test fails against
  origin/main's launcher and passes with the change.
- Two mechanisms (section 16.4b): the hostrec memfree troughs line up with
  the java 4bit/2048 worker windows — every crash starts within about 60 s
  of a java 4bit/2048 worker starting, every recovery begins within about 4
  min of the worker ending (new table in section 16.4b; review round 3
  refines the recovery alignment and shows the measured phases ran at or
  after the recoveries, section 16.10), and no python 4bit/2048 worker
  coincides with a trough (A pair0 python ran 21:04:33-21:05:36 at 86%
  memfree; the weak overlap argument is replaced by the python-led-pair
  evidence in section 16.10). The 4bit/512 slowdowns are a different
  pre-trough pattern (W2 java 54-55% free, A2 pair6 both runtimes 43-67%,
  A3 pair9 python 55-60%). The trough evidence now weighs toward the
  worker-allocator-pool hypothesis, which would make 4bit/2048 partly
  runtime-attributable (owner item 7); the two claims that had been stated
  as fact (the owner of the wired growth, and the page-cache-eviction
  mechanism) are hedged.
- A3 pair9 narrative (section 16.4b): the python/512 worker ran
  06:29:01-06:30:39, before the 06:33:01 strike, so its 3.551 s
  degradation is not attributable to that strike; the RSS watcher log
  starts at 06:44:01 and missed the strike's onset and first 11 minutes
  (its in-strike samples ranged 21-41%, not 28-41%).
- Host-record deviation from plan section 4 recorded in section 16.1:
  per-worker before/after host state with full command outputs was not
  captured; the full `pmset -g therm` outputs (the primary thermal record
  per tools/gpu-burn/hostrec.sh) exist only for B3 and the diagnostics
  block, so the "no thermal warnings" statements for sessions A through B2
  rest on the therm-warnings count signal.
- Controlled 4bit/2048 defaults to `incomplete`, not `unstable` (plan
  section 4.1: a timeout "is incomplete, not unstable success"; incomplete
  required cells need completion or a dated owner scope amendment), until
  the owner amends — section 16.4b's consequence line, the section 16.7
  status row and owner item 4 are aligned. Owner item 3 now states that,
  under plan section 4.1's "any restart consumes one of the remaining
  sessions", the post-timeout A3 pair10 is a fourth session over the
  three-session cap, yet its data feeds the 11-pair and 10-pair ranges.
- Stock 4bit/2048: counting the two excluded members' validated measured
  medians, the >10% adjacent-fork rule fires in both sessions (B pair0 -> 1
  java 3.504 -> 3.050 s, -13.0%; B2 pair6 -> 7 java 3.475 -> 3.040 s,
  -12.5%), so the trigger fires whatever the owner decides about the
  memory warnings — sections 16.5/16.5a, the section 16.7 status row and
  owner item 5 say so.
- Rounding slips (root cause: rounding to 4 places and then to 3, or 1
  place): section 16.4 A 0.6b/2048 java p90 0.983 -> 0.982 (exact
  0.982483); section 16.5 tinymistral/2048 python median 0.601 -> 0.600
  (0.600496, the slip also appears in the section 16.7 table) and 4bit/512
  python p10 1.385 -> 1.384 (1.384483); section 16.7 stock 4bit/512 java
  p10 1.780 -> 1.779 (1.779495), stock 4bit/2048 python p90 2.141 -> 2.140
  (2.140497) and controlled 4bit/512 java p10 1.790 -> 1.789 (1.789465);
  section 16.4 python load median range 0.18-0.43 -> 0.17-0.43 s (min
  0.174650); section 16.4b A3 python/2048 shift +17.6% -> +17.7%
  (17.6514%).
- Smaller corrections: section 16.4's "spreads <1%" is restated — the
  python 0.6b/32 and 0.6b/512 pair-median spreads are 3.72% and 2.91%; the
  clean 4bit/512 java cluster is 1.785-1.813 s, not 1.786-1.813 (pair8
  1.78546); the section 16.6 "canary convention" reference is reworded
  against the recorded full strikes (21-35% crash-onset samples; the
  onset-based threshold is corrected to the trough floor in section 16.10);
  W2's
  sustained trough is 28-50% (the 21:28:54 sample read 50%), and W2's
  4bit/512 slowdown is placed before its trough; the divergence-index
  column is marked 0-based; B3's 12 watcher samples are 90-92% memfree
  (89-96% is the hostrec range); the section 16.5/16.5a strike
  descriptions are restated from the raw hostrec samples (71 -> 31% at
  07:24:23, recovering 07:30:23-07:31:23; 61% at 17:07:31 -> 45% at
  17:08:31 -> 35% at 17:09:31, recovering 17:15:31-17:17:32); the A2 pair6
  host evidence gains the
  05:31:57 65% sample and the 05:41:57 recovery point.
- Raw-data traceability: `req/plans/phase8-2-wp3-raw-digests.json` records
  the SHA-256 of every session.json (33) and report.json (344) under
  `~/wp3-scratch/sessions` (referenced from section 16.1; extended to
  schema v2 — 452 files including spec.json, the per-session hostrec.log,
  rsswatch.log, the thermal records and the diagnostic-block host logs — in
  review round 3, section 16.10).

### 16.10 Review round 3 (2026-11-11, PR #91)

The review re-placed each degraded worker's requests on the hostrec clock
from the raw reports (forward from utc_start, backward from utc_end) and
re-checked the session-outcome text against the round-2 changes. This
round amends section 16 in place:

- Code (after the measurement; the measured code remains cc0c603):
  `finishQuantiles`'s parameters are renamed `perRuntime` -> `target` and
  `measured` -> `samplesByCell`, since the method now builds both the
  measured-request block and the first-request block. No behavior change.
- Trough alignment (section 16.4b): in all five recovered cases the
  recovery begins 39-206 s (0.7-3.4 min) before the worker's own end, not
  "within about 4 min of the worker ending"; it falls 7-36 s after the
  estimated end of that worker's first-request plus warmup phase (B2
  pair6's trough ended 9 s before its estimated end). Review round 4
  (section 16.11) tightens the placement: the 14.7-21.2 s apparent
  inter-request gaps are mostly the recorded prime time, which the
  primed-per-request mode excludes from full_generation_s; including
  prime_s leaves only 2.0-2.4 s unaccounted per worker. Review round 5
  (section 16.12) corrected the range.
- W2 (sections 16.4/16.4b): the W2 bullet had said the worker "ran
  inside" the trough; only its first request and warmups did. Its 20
  measured requests (9.923-10.768 s each, 3.4x the clean 3.08 s) ran at
  81-82% free — every hostrec sample covering the measured phase read
  81-82% (the measured phase began 12-15 s before the 21:34:55 recovery
  sample; placement tightened in section 16.11), so that slowdown is not
  explained by the trough or by page-cache churn — whatever slowed them
  outlasted the memory pressure (an unverified guess).
- The python-overlap statement carried little weight: the workers run one
  at a time and every trough sits inside a java 4bit/2048 window, so a
  python worker could not overlap a trough, and a ~60 s python window
  admits at most one hostrec sample. It is replaced by the python-led-pair
  evidence: A3 pair9's python/2048 ran 06:31:35-06:32:33 (its in-window
  sample 70% — the 59% sample at 06:31:01 predates the worker; corrected
  in section 16.11), 28 s before the strike, clean at 1.915 s (A2 pair7's
  in-window sample 88%, clean at 2.235 s).
- Section 16.7: "No cell remains blocked by an unrun session" was false
  since round 2 — the stock 4bit/2048 trigger fires in B2, the cell's
  second session, plan section 4 requires one further alternating session,
  the cap allows a third, and none was run; stock 4bit/2048 is now stated
  as blocked unless the owner decides otherwise (item 5). The
  session-count cap paragraph now records that controlled 4bit/512 and
  4bit/2048 used four sessions counting A3 pair10 (owner item 3), over the
  cap; "Unstable cells" is relabeled "Unstable/incomplete cells" and the
  "the cell that A3 closed" wording is replaced.
- Section 16.6: the mild-event threshold is now based on the trough floor
  (21-32%) and duration, not the onset sample — the old "21-35%
  crash-onset samples" wording was contradicted by the table's own B2
  pair6 onset sample (45%).
- Raw-data digests (section 16.1): the manifest is extended to schema v2,
  452 files — adding the 33 spec.json, the 33 per-session hostrec.log
  (each byte-identical to hostrec/<session>/host.log),
  hostrec/a3-block/rsswatch.log, the four codex-*-thermal-*.txt records
  and the pilot/qualification/diagnostic hostrec/<block>/host.log files —
  so the host-side claims tie to fixed data too. (Review round 4, section
  16.11, adds the block launcher.log files and the rejected
  session-a-controlled directory, bringing the total to 458.)

### 16.11 Review round 4 (2026-11-11, PR #91)

This round amends section 16 in place:

- Section 16.5a outcome: the stock 4bit/2048 bullet no longer says "no
  >10% trigger" — the trigger fires on pair6 -> 7 counting the excluded
  member's validated measured median (-12.5%), so a third alternating
  session is required before attribution whatever the memory-warning
  decision, it was not run, and the cell is blocked; the section 16.7
  status row is labeled "blocked (unrun third session)".
- Placement precision (sections 16.4/16.4a/16.4b/16.10): the "unattributed
  inter-request gaps" (14.7-21.2 s per worker) are mostly the recorded
  prime_s — the primed-per-request mode excludes prime time from
  full_generation_s, so placing requests on full_generation_s + prime_s
  leaves only 2.0-2.4 s unaccounted per worker, and the forward and
  backward placements agree to within ~2 s. Quoted windows tightened: W2
  21:34:40-21:34:42 (was 21:34:29-21:34:50), A2 pair6 05:41:48-05:41:50
  (was 05:41:38-05:41:55), the trough-tail hedge is now "at most the first
  28-37 s of A pair0's and B pair0's measured phases". The conclusions are
  unchanged: with prime time included, A pair0's measured phase starts
  34-36 s before its recovery sample and B2 pair6's 9-12 s after it.
- A3 pair9's python 4bit/2048 worker (06:31:35-06:32:33): the only hostrec
  sample inside its window is 06:32:01 (70%); the 59% sample (06:31:01)
  predates the worker, so "at 59-70% memfree" is restated as "its
  in-window sample 70%" (sections 16.4b/16.10).
- Rejected first launch of session A recorded (section 16.4): at
  20:36:50-20:36:51 a single 108-worker spec — one session for all six
  pairs — was rejected with "duplicate cell" spec errors (the cell
  identity carries no pair index), BUILD FAILED in 526 ms, no worker ran
  (hostrec/session-a-controlled). No measurement impact; it does not
  count against plan section 4.1's restart rule.
- Raw-data digests (section 16.1): the manifest gains the
  hostrec/session-a-controlled host.log and launcher.log and the
  pilot/qualification/diagnostic launcher.log files — 458 files.

### 16.12 Review round 5 (2026-11-11, PR #91)

This round amends section 16 in place; both findings are consistency
fixes in the section 16.4b placement paragraph, verified against the
raw report.json placements and the 60 s hostrec samples:

- Range unification (sections 16.4b/16.10): "the recovery sample
  falls 9-36 s after the estimated end of that worker's
  first-request plus warmup phase" and "the measured phases of A
  pair0, A2 pair6 and B pair0 began 7-36 s before their recovery
  sample" describe the same moment — the end of the warmup phase is
  the start of the measured phase. The verified per-worker offsets
  (backward placement / forward placement) are A pair0 34/36 s, B
  pair0 28/30 s and A2 pair6 7/9 s before the recovery sample, so
  both clauses now read 7-36 s.
- In-phase sample range (section 16.4b): "the hostrec samples (60 s
  cadence) covering the measured phases reading 56-82%" — the named
  workers' in-phase samples are 62 (A pair0), 63 and 68 (A2 pair6),
  56 (B pair0) and 69 (B2 pair6), so the range is 56-69%. The
  81-82% readings are A pair2's measured phase, which the next
  sentence covers separately.
- No code, measurement or raw-data change; the digest manifest
  remains at 458 entries.

## 17. WP4 — Both-pin, both-VLM first inference (2026-11-11, this host)

Cross-references: "plan section N" is `req/plans/phase8-2-plan.md`; a bare
"section N" is this findings file.

### 17.1 Code and host state

- Measured code: clean tree at e4cb1f0 (branch phase8-2-wp4; tracked-clean
  — the only untracked file is the operator scratch script `resume.sh`,
  recorded as untracked in every session report and never committed). The
  WP4 harness (worker, launcher, launcher tests, `benchmarkVlmFirstUse`
  task) landed in this one commit. The commit was amended once, from
  a9b5abc, after the rejected first smoke (the TTFT stamping fix, section
  17.2); the post-fix smoke and the full session both ran from the
  amended SHA, so no measured code changed between them.
- Label (plan section 5.2): this is a **code-and-pin comparison, not an
  isolated native-pin effect**. The old tree runs the framework code
  committed at 46cb325 (156 commits before the new tree) against the
  0.31.2/fba4470 dylib; the new tree runs e4cb1f0 against
  0.32.4/a26f2d5. What is shared and verified between the trees is the
  measurement logic — `VlmFirstUseWorker.java`, byte-identical in both
  trees (SHA-256
  `405a839c2f0399391b67e399bfbbb83218a4ee44bab9c2c56d9631665ed02130`,
  re-verified from disk by the launcher before the first fork and recorded
  in both session reports; in the old tree it is an untracked file) — and
  the workload bytes (digest-pinned manifests and images, re-verified per
  fork by both launcher and worker). The launcher itself is new-tree only
  (it forks and does not measure).
- Trees (compat worktrees outside the checkout, resolved absolute paths
  recorded in both session reports): new = `/Users/per/project/jmlx` at
  e4cb1f0f43ddd90242237f7e155a399ca5c84cfd; old =
  `/Users/per/jmlx-wp4-oldpin` at 46cb325bec9b4878534cc5c6bda1a569dfe4aa4c
  (detached head; an ancestor of the new tree's HEAD). The old worktree
  sits outside the checkout (a sibling of the project directory), so the
  root inventory scans (`verifyMlxApiCallSites` and the others) see only
  the current checkout's generated bindings, never the old tree's. The
  main checkout was not repinned and its staged runtime was not touched;
  the current generated bindings were never pointed at the old dylib —
  each fork runs one tree's installDist against that tree's own staged
  `native/install/lib` (each tree's `class_dir` and `native_dir` are
  recorded in the session report's tree blocks, and the worker report
  records the pin loaded from its own `-Djmlx.library.path`).
- Pin identities, staged (read from disk by the launcher) and re-verified
  at the worker's own `-Djmlx.library.path` (the property the loader
  resolves first; this is the staged pin file at that path, not a runtime
  verification of which dylib the process actually loaded): old 0.31.2 /
  mlx-c fba4470b89073180056c9ea46c443051375f7399; new 0.32.4 /
  mlx-c a26f2d584a14ec619ae8db9a09be03becd47598a. Launcher-side and
  worker-side staged identities agree for all 24 workers.
- Dist dependency sets (recorded, not a controlled variable): the old
  tree's installDist resolves jackson 3.1.2, jmlx-jinja 0.6.0,
  jmlx-tokenizer/jmlx-vision 0.1.0; the new tree's resolves jackson 3.2.4,
  jmlx-jinja 0.6.2, jmlx-tokenizer/jmlx-vision 0.1.2. The worker compiles
  against both (it uses only API stable across the two).
- Case and settings (frozen, plan section 5.2; WP0 section 2.2): per
  workload, the digest-pinned Tier-B manifest's case 0 (qwen3vl-2b:
  manifest `5ebfaf7a…`, phi3v-4b: manifest `faeae7ad…`), the manifest's
  own images (digests re-verified per fork), max_new_tokens and
  expanded/unexpanded IDs from the manifest, `MLX_ENABLE_TF32=0` (the
  worker report records the precision env per worker). Stop sets, recorded
  explicitly per workload: qwen3vl-2b `{151643, 151645}`, phi3v-4b
  `{32000}`. They are inert for the frozen cases (both finish
  MAX_TOKENS with no generated ID in either set) but are carried in every
  job and worker report as the declared setting.
- Per-process request sequence (one fixed case per process, plan section
  5.2): model load, first image+text generation captured immediately after
  load in the fresh JVM (before any correctness forward, warmup, probe or
  other case; lazy loader work stays inside the first request — the load
  wall records the lazy portion, the report notes it), correctness
  verification of the first request (IDs vs the manifest's recorded
  target IDs, finish reason, prompt positions vs expanded-ID size,
  rendered prompt vs the manifest), then 3 warmups and 5 measured
  repetitions of the same request. Worker timeout 900 s, session time
  budget 5400 s.
- Host: M5 Max (128 GiB), macOS 27.2 (build 26B123), Temurin 25.0.4.1
  worker JVMs. The host was rebooted 09:03 local (08:03 UTC) the morning
  of the run, per the operator's `last reboot` (recorded here; not part
  of the raw tree); both the smoke and the full session ran on the
  post-reboot host. The primary thermal records (full `pmset -g therm`
  at session start and end, no warning or performance level) are the
  `thermal-start.txt` / `thermal-end.txt` of the session directory in the
  raw tree. Timestamps: hostrec and session-report times are UTC; local
  time is UTC+1.
- Fork order (recorded in the session report): workload-major,
  pair-major, tree-minor, with the leading tree alternating per pair (even
  pairs lead old, odd pairs lead new) — every fork is on a different tree
  from its predecessor and each pin leads 12 of the 24 forks.
- hostrec convention (plan section 4): pre/post appends per fork by the
  launcher, plus the session wrapper's 60 s poll loop; full
  `pmset -g therm` start/end records in the session directory; the
  wrapper's `host.log` is copied into the session directory as
  `hostrec.log`. hostrec records memfree as a percentage, not a pressure
  level (same convention and limitation as section 16.1).

### 17.2 Smoke (1 pair per VLM, rejected first run and the TTFT fix)

- First smoke (1 pair, 4 workers, from pre-amend tree a9b5abc): all four
  workers `ok`, all IDs matched, but the report's warm `measured_ttft_s`
  arrays were `-1e-09` — the worker stamped the first-token time only when
  recording the full token-event list (first request only), so warm
  iterations reported the "no event" sentinel. The run was rejected, its
  session directory removed, and the worker fixed so every generation
  stamps the first-token time (the TTFT) while only the first request
  records the token-event list. The fix is in the amended commit e4cb1f0;
  the worker source digest quoted in section 17.1 is the post-fix one.
- Post-fix smoke (1 pair, 4 workers, from e4cb1f0; kept in the raw tree
  under `sessions/wp4-smoke`): 4/4 `ok`, all IDs matched, measured TTFTs
  now realistic (qwen3vl warm TTFT ~0.31-0.35 s, phi3v ~1.6-2.4 s),
  per-pair warm-median ratios present for both workloads; tree/pin
  verification, hostrec bracketing and the session-report schema all
  verified end-to-end against real model loads on both pins.

### 17.3 Full session `wp4-observed-cache-1` (6 alternating pairs per VLM)

- Session 15:38:49-15:46:07 UTC (7 m 18 s wall for 24 workers), out root
  `~/wp4-scratch/sessions/wp4-observed-cache-1`. All 24 workers `ok`; no
  dead, crashed, timeout, job-verification-failed or not-started
  classification. Correctness-after-capture held for every worker: first
  request IDs equal the manifest's recorded target IDs, finish reason
  MAX_TOKENS, prompt positions equal the expanded-ID size, rendered prompt
  equal to the manifest's, and all 5 measured repetitions of each worker
  reproduced the first request's IDs (greedy determinism under TF32=0).
- Host records, restated from both record sets (the wrapper-only
  statement that originally stood here is withdrawn by section 17.5,
  P2): the launcher's 48 per-fork records (one per pre/post fork,
  `hostrec/host.log`, digest-pinned) show therm-warnings count 2 in
  every record (no new warnings), 1-min load 1.24-4.20, memfree 38-64%,
  and omlx (hostrec.sh's "leading contention candidate" — the resident
  serving-model server) at or below 0.6% only from 15:43:29 on (the
  phi3v phase): 16 of the 48 records read above 15%, and four bursts
  above 50% (97.9 / 95.2 / 83.2 / 75.6%) all fall in
  15:38:49-15:40:14, bracketing the qwen3vl pairs 2-5 and the anomalous
  qwen3vl-2b new-pair-2 dip. The wrapper's coarser 9 records (pre, 7
  polls, post; `hostrec.log`) agree in trend but sample only every 60 s
  (omlx at most 76.6%, in the pre record; 0.4-14.5% at every poll and
  the post; load 1.40-3.32; memfree 35-59%) and miss the bursts between
  samples. Two caveats: `ps` %CPU on macOS is a decaying average, so a
  high reading covers earlier activity and no causal link between the
  omlx bursts and the p2 dip is claimed; and the session did not run on
  a quiet host — the serving agent (this session's own model,
  omlx-server) was actively working during the qwen3vl phase, against
  the section 16.4b convention of parking the agent's work on a
  different agent so omlx-server stays quiet. The session therefore does
  not qualify as the plan section 4 quiet-host baseline, and the section
  4 trigger rule (a >10% adjacent-fork-median shift, or a surprising
  host state) calls for another alternating session before runtime
  attribution: session 2, section 17.6. No gpu-burn contention was
  applied (plan section 5.2: controlled contention is optional follow-up
  only). Thermal start and end: no warning or performance level
  recorded.
- First-in-process (first request; raw per-worker values in the session
  report, MB below are 10^6):
  - qwen3vl-2b: load wall old 0.793-1.306 s, new 1.070-1.409 s (medians
    1.090 / 1.360); first TTFT old 0.249-0.691 s (median 0.449), new
    0.475-0.673 s (median 0.509); first full generation old 0.613-1.139 s
    (median 0.776), new 0.777-1.047 s (median 0.827).
  - phi3v-4b: load wall old 0.836-1.924 s, new 0.822-1.937 s (medians
    1.082 / 0.846); first TTFT old 1.340-2.470 s (median 1.573), new
    1.270-2.184 s (median 1.354); first full generation old
    2.422-5.776 s (median 3.124), new 2.371-3.856 s (median 2.448).
- Warm measurements (post-warmup, 5 measured per worker, pooled n=30 per
  tree per workload; type-7 linear-interpolation quantiles, as in section
  16):
  - qwen3vl-2b TTFT: old p10/median/p90 0.252/0.329/0.353 s, new
    0.145/0.324/0.352 s; full generation old 0.495/0.605/0.627 s, new
    0.266/0.598/0.630 s.
  - phi3v-4b TTFT: old 1.381/1.572/2.416 s, new 1.396/1.508/2.103 s; full
    generation old 2.268/2.448/4.023 s, new 2.255/2.397/3.670 s.
- Paired per-pair warm-median ratios (new/old, the paired statistic; pairs
  are process pairs, not independent samples):
  - qwen3vl-2b: TTFT 0.525/1.117 range, median 0.951; full generation
    0.478/1.066 range, median 0.939.
  - phi3v-4b: TTFT 0.850/1.082 range, median 0.958; full generation
    0.909/1.050 range, median 0.975.
- Per-pair trajectory (warm full-generation median, old/new by pair):
  qwen3vl-2b old 0.616/0.549/0.596/0.590/0.618/0.614, new
  0.612/0.586/0.285/0.534/0.601/0.484; phi3v-4b old
  3.998/4.015/2.268/2.454/2.442/2.301, new 3.635/3.666/2.383/2.390/2.384/
  2.255. Two observations, recorded without attribution (plan section 5.2:
  repeat before attributing; OS/Metal cache state is observed/uncontrolled
  in this regime):
  - phi3v-4b pairs 0-1 (forks 13-16, directly after the 12 qwen3vl
    forks) are slow in **both trees at the same session position**
    (warm full ~3.6-4.0 s) and step down to ~2.3-2.4 s from pair 2
    (forks 17-18) in both trees. A synchronized
    two-tree step at one session position is consistent with a host
    cache-regime transition (OS page cache / system shader cache
    warming across the fresh processes) rather than a pin effect; the
    controlled shader-cache
    experiment that would attribute it was not run (plan section 5.2
    marks it additional work), so shader-cache attribution is left
    unresolved, as the plan allows. It does not invalidate the paired
    comparison, and the pooled statistics deliberately include the early
    pairs (the p90 values carry the transition).
  - qwen3vl-2b new pair 2 is a single low pair (0.285 s vs ~0.5-0.6 s
    around it; the paired TTFT ratio 0.525 is the session minimum). The
    10% adjacent-fork delta is an investigation trigger, not a gate; it
    is recorded here and the pooled statistics include it. No further
    session was triggered (one session per cell, under the plan section 4
    cap of three).
- Memory (process-wide MLX counters sampled after each phase's eval; the
  report's note on lagging Metal frees applies; the v1 worker never
  reset the peak counter, so every peak below is lifetime-cumulative —
  the after_first peak includes the load and after_load phases — and is
  not directly comparable to the phase-local peaks of session 2,
  section 17.6):
  - qwen3vl-2b: after load, peak 8,514.86 MB and active 8,510.14 MB in
    both trees (span across the 12 workers: peak 8,514,863,104-
    8,514,865,152 B, active 8,510,144,512-8,510,146,560 B). After the
    first forward, peak 9,828.89 MB in all 12 workers (9,828,890,684-
    9,828,892,836 B; both trees inside a 2.1 KB span), consistent with
    the 7.3b WP8 measurement of 9,828,890,788 B for the same checkpoint
    and case.
  - phi3v-4b: after load, peak 16,643.12 MB and active 16,536.11 MB in
    both trees (peak 16,643,123,200-16,643,127,296 B, active
    16,536,107,008-16,536,111,104 B). After the first forward, the
    trees diverge: old 31,100.38 MB (31,100,376,165-31,100,380,261 B),
    new 30,202.75 MB (30,202,745,961-30,202,748,009 B) — the new tree
    peaks ~897.6 MB lower on the identical checkpoint and case. The
    after-load peaks agree between trees (the same lazy loader), so the
    divergence appears with the first forward. Per the section 17.1
    label this is a code-and-pin effect, not attributable to the native
    pin alone; the old tree's number matches the 7.3c WP8 measurement
    (31,100,376,165 B) within one page, the new tree's reflects the
    framework memory changes in the 156 commits between the trees.
  - First-request TTFT and full generation include the lazy loader work
    inside the first request, as declared (the report's load block notes
    "load is lazy (no eval during load); weight materialization lands
    inside the first forward"); the load wall itself records only the
    (mostly lazy) loader.
- Session-report and per-worker artifacts: `session-report.json` (schema
  `jmlx-vlm-first-use-session-1`), 24 cells each with `job.json`,
  `report.json` (schema `jmlx-vlm-first-use-1`) and
  `worker-stderr.log` (empty for every worker), the per-fork hostrec
  directory, `hostrec.log` (wrapper host log) and the two thermal
  records.

### 17.4 Raw-data digests and open items

- Raw-data digests: `req/plans/phase8-2-wp4-raw-digests.json` (schema v2,
  181 files, paths relative to `~/wp4-scratch`) records the SHA-256 of
  both full sessions (`wp4-observed-cache-1`: 24 worker cells with
  job/report/stderr each, session report, per-fork hostrec records,
  `hostrec.log`, thermal records; `wp4-observed-cache-2`: same layout),
  the post-fix smoke session (same layout, 4 cells), the wrapper's
  `hostrec/<session>/{host.log,launcher.log}` for both sessions and all
  three session specs — tying this section, including its host-side
  claims, to the exact raw tree. All 100 v1 files are byte-identical in
  v2 (session 1's raw tree is unchanged). Session 1's report carries
  tree new e4cb1f0 / tree old 46cb325 and the v1 worker digest quoted in
  section 17.1; session 2's carries tree new ecc69a5 / tree old 46cb325
  and the review commit's worker digest quoted in section 17.5, plus the
  executed-jar digests per tree. The rejected first-smoke session was
  removed before the post-fix smoke ran and is not in the raw tree (its
  only evidentiary role was exposing the TTFT stamping bug, section
  17.2).
- Open items: the phi3v-4b early-pair transition (section 17.3) remains
  unattributed — attributing it needs either back-to-back repeats or the
  controlled shader-cache experiment, both follow-up work per plan section
  5.2; session 2 (section 17.6) is the back-to-back repeat: the
  early-pair slowness did not recur on the quiet host, so the state is
  consistent with a first-use-after-reboot host effect, still not
  attributed. The qwen3vl-2b new-pair-2 dip (section 17.3) was the other
  section 4 trigger that ordered session 2; it did not recur (section
  17.6) and its bracketing by session 1's omlx burst is recorded without
  causal attribution. The phi3v-4b after-first-forward allocator peak
  difference (~897.6 MB, new tree lower) is a code-and-pin observation
  under the section 17.1 label, unchanged in session 2; attributing it to
  specific framework changes would need a commit-range memory bisect,
  which is not part of this work package. The old-pin worktree
  (`/Users/per/jmlx-wp4-oldpin`) and its staged runtime stay in place for
  any repeat session the owner orders; nothing in this section repins or
  modifies the main checkout.

### 17.5 Review round 1 (2026-11-11, PR #92)

Sixteen findings (1 high, 5 medium, 10 low), all addressed in the review
commit; the worker report schema is bumped `jmlx-vlm-first-use-1` -> `-2`
and the session report schema `jmlx-vlm-first-use-session-1` -> `-2`
(the job schema is unchanged — its `manifest_path` now always carries
the launcher-verified resolved path). The review commit's worker source
digest is `fede0c00ac0528f044cb322ad24ed0d93c03068582cd47ec8e20ad04eabbccac`
(byte-identical in both trees, as before). Session 1's results stand as
recorded (all 24 workers ok, all IDs matched; the high-severity finding
did not affect it — every `first_request.correct.*` field in the raw
reports is true), but the harness it ran on changed, and the section 4
triggers fired, so session 2 (section 17.6) ran the fixed harness from
the review commit's SHA. Its first qwen3vl forks showed residual omlx
activity, so only its later qwen3vl pairs and phi3v phase were quiet.

- P1 (high): a first-request correctness failure wrote
  `correctness_failed` and then fell through to an `ok` report with
  exit 0 (missing `return`), and the launcher's classify trusted
  `status` alone. Fixed: the post-capture decision is now the static,
  side-effect-free `postCaptureVerdict(idsMatch, finishMatch,
  positionsMatch, divergences)` returning `Verdict(status, exitCode,
  failureReason)`, and the verdict's exit code is the process exit code;
  classify accepts `ok` only when the exit code is 0 and every
  `first_request.correct.*` field is boolean true
  (`firstRequestCorrect`), else `invalid-report` with a detail.
- P2: the section 17.3 host bullet restated from both record sets (the
  48 per-fork records, not only the wrapper's 9); "quiet" withdrawn;
  the serving agent's own omlx activity during the qwen3vl phase
  recorded. See the restated bullet.
- P3: the worker's second `put("workload")` (the inputs object)
  overwrote the workload name (Jackson 3 `putObject` replaces an
  existing property). The object is now `workload_inputs`.
- P4: a warm/measured ID divergence threw, was misclassified as a
  runtime exit-2 failure with the captured first request lost, and
  `measuredIdsMatch.add(true)` was tautological. Fixed: a divergence
  records `WarmDivergence(label, generatedIds)` in the report's
  `divergences` array, stops the remaining iterations, and the post-
  capture verdict finishes the worker as `correctness_failed` (exit 3)
  with the first-request data retained; `measured_ids_match` records
  real per-iteration results.
- P5: `token_event_times_s` held only the first token (the early return
  skipped every later event). Every first-request token event is now
  recorded; the TTFT stamping (first event only) is unchanged.
- P6: the worker had no tests. `VlmFirstUseWorkerTest` (pure Java, no
  native) covers the verdict matrix (all-ok, each single first-request
  mismatch, labeled warm/measured divergences, combined problems) and
  the job schema gate; the launcher test gained the two inconsistent-
  ok classify cells (ok report with exit 3; ok report with a false
  `correct` field) and the new out_root refusal scenarios.
- P7: "actually loaded" pin identity reworded to "staged pin file re-
  read from the worker's own `-Djmlx.library.path`" (the property the
  loader resolves first; no runtime dylib-load verification); the
  report now carries the same note in `native_pin.note` (section 17.1
  bullet restated).
- P8: the phi3v step-down wording softened to "consistent with a host
  cache-regime transition rather than a pin effect" (section 17.3).
- P9: the executed jars are now digested per tree (`jar_sha256` in each
  session-report tree block).
- P10: hostrec failures and timeouts are no longer invisible: 30 s
  timeout per call (killed and recorded), all failures listed in the
  session report's `hostrec_failures` array (a missing host record is a
  plan section 4 trigger).
- P11: the session report's hostrec string corrected (per-fork records
  in `hostrec/host.log`; the wrapper's 60 s poll log is `hostrec.log`).
- P12: the backstop race fixed with a write-once `reportWritten`
  guard: only the first caller of `finish` writes the report, and the
  backstop exits 4 only if it itself wrote.
- P13: the job now carries the launcher-verified resolved manifest path
  (`workload.manifestAt(spec.repoRoot)`), not the spec's raw string.
- P14: `out_root` (and every spec path) is normalized before the
  containment check, which now also covers both trees.
- P15: the worker now calls `MLXMemory.resetPeak()` after the
  after_load and after_first samples (phase-local peaks, plan section 7
  "peaks reset at stated phases"); session 1's peaks were
  lifetime-cumulative and are labeled as such (section 17.3 memory
  bullet).
- P16 (minors): `Job.parse` rejects a missing or wrong schema;
  `generate` is void (the unused `long[]` return is gone); the worker
  class Javadoc order matches the code (verification before the
  handshake claim); non-string, non-integer manifest content-part
  fields are rejected instead of passed into the chat template as raw
  JsonNode.

Session 2 (the section 4 trigger): the plan's trigger rule is "a >10%
shift between adjacent fork medians, a thermal/memory warning, a
missing host record, or a surprising VLM prefill triggers another
alternating session before runtime attribution" (cap three sessions per
cell; one used). Two triggers fire in session 1: the qwen3vl-2b
new-pair-2 dip (0.285 s vs ~0.5-0.6 s neighbors) and the phi3v-4b
early-pair step-down; and session 1 does not qualify as a quiet-host
baseline (the serving agent worked on omlx during the run, section
17.3). Session 2 reuses the frozen case, config and fork-order
convention (spec `wp4-observed-cache-2`: same pairs 6, warmups 3,
measured 5, timeout 900 s, budget 5400 s, TF32=0, stop sets qwen3vl
{151643, 151645} / phi3v {32000}) from the review commit's SHA, with
the serving agent fully idle during the run (the section 16.4b
convention restored).

### 17.6 Session 2 `wp4-observed-cache-2` (host activity during qwen3vl, fixed harness)

- Session 21:00:17-21:07:00 UTC (6 m 43 s wall, 24 workers) from the
  review commit ecc69a5 (worker digest
  `fede0c00ac0528f044cb322ad24ed0d93c03068582cd47ec8e20ad04eabbccac`
  byte-identical in both trees, re-verified by the launcher; executed-jar
  digests in each session-report tree block; `hostrec_failures` empty).
  All 24 workers `ok` under the `-2` schemas; every first request
  matches the manifest reference (all `correct.*` true), no warm or
  measured divergence (all `divergences` arrays empty; all 5 measured
  repetitions of every worker reproduce the first request's IDs), and
  every report carries `workload_inputs` and the full
  `token_event_times_s` array.
- Host: the 48 per-fork records show omlx at 89.4% in the pre record,
  decaying monotonically (34.6 / 20.3 / 29.9 / 17.9 / 15.1 / 13.1 /
  12.5 / 12.6 / 7.4%) across the first four qwen3vl forks and at or
  below 0.7% from qwen3vl old:p4 (21:01:24) to the end of the session —
  the entire phi3v phase ran on the quiet host of the section 16.4b
  convention, with no bursts between the readings (unlike session 1).
  The decaying readings of the first ~70 s carry the owner's pre-launch
  activity (the agent was idle from the moment the background run was
  launched); per the decaying-average caveat they are recorded without
  causal attribution. therm-warnings 2 in all 48 records; thermal start
  and end: no warning or performance level; 1-min load 3.37-7.08 (the
  phi3v forks themselves push the load); memfree 28-69%.
- First-in-process (first request; separate from the warm statistics,
  plan section 4; ranges over the 6 workers per tree):
  - qwen3vl-2b: load wall old 0.786-1.459 s, new 0.819-2.845 s (new p2's
    load is an outlier); first TTFT old 0.204-0.765 s, new 0.233-0.922
    s; first full old 0.335-1.327 s, new 0.366-1.350 s.
  - phi3v-4b: load wall old 0.896-1.351 s, new 0.937-1.315 s; first
    TTFT old 1.355-1.655 s, new 1.316-1.817 s; first full old
    2.472-6.568 s (p0 4.959, p1 6.568, p3 5.384), new 2.447-9.755 s
    (p0 9.755, p4 8.685). Slow first requests occur in both trees (the
    lazy loader work inside the first request; recorded without
    attribution) and the first-request distribution stays separate from
    the warm statistics.
- Warm measurements (5 measured per worker, pooled n=30 per tree per
  workload; type-7 quantiles as in section 16):
  - qwen3vl-2b TTFT: old p10/median/p90 0.110/0.239/0.378 s, new
    0.111/0.338/0.389 s; full generation old 0.224/0.392/0.654 s, new
    0.226/0.616/0.674 s.
  - phi3v-4b TTFT: old 1.416/1.617/1.869 s, new 1.348/1.733/1.855 s;
    full generation old 2.332/2.491/2.794 s, new 2.234/2.672/2.864 s.
- Paired per-pair warm-median ratios (new/old, the paired statistic):
  - qwen3vl-2b full generation
    1.190/1.063/0.994/2.379/1.079/1.005 (TTFT 1.303/1.068/0.986/2.328/
    1.036/1.001): pairs 1, 2, 4 and 5 sit in 0.994-1.079. The p3 ratio
    is an intra-pair ordering artifact: the host's fast-regime boundary
    fell between new:p3 and old:p3, so the pair's new member measured in
    the slow regime and its old member in the fast one.
  - phi3v-4b full generation 0.902/1.026/0.992/1.120/1.058/1.018 (TTFT
    0.830/1.013/0.964/1.167/1.027/1.034), median 1.022, on the fully
    quiet host.
- Regime structure and session-to-session comparison:
  - qwen3vl-2b runs two warm-full regimes in session 2, ~0.563-0.669 s
    (pairs 0-2 and new p3) vs ~0.224-0.272 s (old p3, pairs 4-5); the
    boundary coincides with the omlx readings crossing below 1%
    (21:01:24-21:01:28). Session 1's qwen3vl phase ran entirely inside
    the omlx-burst window and its warm full was 0.549-0.618 s (old) and
    0.484-0.612 s (new, apart from the new-p2 dip of 0.285 s) — the same
    slow regime; session 1's new-p2 dip did not recur (session 2 new p2
    is 0.629 s), consistent with its bracketing by session 1's 95.2%
    omlx burst (recorded without causal attribution, decaying-average
    caveat).
  - The phi3v-4b early-pair slowness (session 1: ~3.6-4.0 s for pairs
    0-1 in both trees, stepping down from pair 2) did not recur:
    session 2's warm full is flat at 2.136-2.815 s across all pairs on
    the quiet host, and the pooled medians (old 2.491 / new 2.672 s)
    match session 1's quiet-phase values (2.448 / 2.397 s). This
    is consistent with a host-side state tied to the first use of the
    checkpoint after the 09:03 reboot; this does not establish a
    code-and-pin runtime attribution;
    the controlled experiment that would attribute it was not run (plan
    section 5.2 marks it additional work), so attribution stays
    "consistent with".
  - The fully quiet phi3v phase has paired ratios near 1.0 (median
    1.022 in session 2). For qwen3vl, only pairs 4-5 ran after the
    observed omlx activity decayed; their full-generation ratios are
    1.079 and 1.005. These two quiet pairs do not support a workload-wide
    code-and-pin attribution. The earlier qwen3vl forks and the p3
    ratio of 2.379 remain part of the reported data.
- Memory (phase-local peaks, schema -2; session 1's cumulative peaks,
  section 17.3, agree within the quoted spans, confirming the peak is
  reached in the first forward, not during load):
  - qwen3vl-2b: after load, peak 8,514,863,104-8,514,865,152 B in both
    trees (session 1's span). After the first forward, peak
    9,828,890,684-9,828,892,836 B — byte-identical to session 1's
    after-first span and to the 7.3b WP8 measurement
    (9,828,890,788 B). After the warm/measured phases, peak
    9,828,890,685-9,828,892,733 B: the warm phase never exceeds the
    first forward's peak, it re-touches the same maximum buffer.
  - phi3v-4b: after load, peak 16,643,123,200-16,643,125,248 B in both
    trees. After the first forward: old 31,100,376,165-31,100,378,213
    B, new 30,202,744,781-30,202,748,009 B (session 1 spans: old
    31,100,376,165-31,100,380,261, new 30,202,745,961-30,202,748,009 —
    the ~897.6 MB tree-to-tree gap is unchanged, a code-and-pin
    observation under the section 17.1 label, not attributable to the
    pin alone); after the warm/measured phases, the peaks equal the
    after-first peaks (same maximum buffer re-touched).
- Cell closure: **qwen3vl-2b = `unstable`; phi3v-4b = `unstable`** under
  plan section 4.1. Both sessions have all required worker samples and
  are retained with their ranges and host records, but neither cell has
  a settled performance ratio or runtime attribution. Session 2 has
  >10% adjacent-fork shifts in qwen3vl, and phi3v pair 0 (TTFT ratio
  0.830) and pair 3 (full/TTFT ratios 1.120/1.167) also meet that
  trigger. The qwen3vl regime boundary coincides with decaying omlx
  activity; the phi3v host phase is quiet but still variable. Session 3
  was not run. Section 4 requires another alternating session **before
  runtime attribution**; no such attribution is made here. Ending
  collection after two sessions, before budget exhaustion or the
  three-session limit, is a recorded deviation from section 4.1's
  stopping rule. Any later runtime attribution needs another complete
  alternating session or a recorded plan revision.
- Artifacts: `session-report.json` (schema `jmlx-vlm-first-use-session-2`),
  24 cells each with `job.json`, `report.json` (schema
  `jmlx-vlm-first-use-2`) and `worker-stderr.log` (empty for every
  worker), the per-fork hostrec directory (48 records), `hostrec.log`
  (wrapper host log) and the two thermal records; digested in the v2
  raw-digest manifest (section 17.4).

### 17.7 Review rounds 2–3 (2026-11-12, PR #92)

The round-2 harness checks each installed JMLX jar against the newest
tracked main-source or build-file modification in its own module and
the root Gradle inputs (including `gradle.properties` and the Gradle
wrapper version); it also checks the benchmarks jar against the
worker source, which can be untracked in the old worktree. A Git checkout
can refresh input mtimes while Gradle keeps content-identical jars; a
freshness refusal now directs the operator to rerun `installDist` with
`--rerun-tasks`. The launcher compares worker reports with the expected
cell and verified pin, accepts only known statuses and matching exit codes,
retains a timeout classification when the launcher kills a worker after its
backstop wrote a timeout report, and records three native-binary digests.
Worker input-verification failures exit 1. Report and session schemas are
now `jmlx-vlm-first-use-3` and `jmlx-vlm-first-use-session-3`;
`job_verification_failed` has one spelling with or without a report.

At the end of review round 3, this committed harness had **not had a
native smoke or full session**. The macOS CI native job runs module checks
and release smoke, not this two-tree VLM launcher. Section 17.8 records
the subsequent one-pair native smoke of this harness, including real
`exit_code`, `native_pin.library_path` and jar freshness acceptance.
Its worker-source SHA-256 is
`76c053b8f1a52dbfaa8b27983e81d223c794f1585e380a0286f25b5411efa439`,
different from both measured worker versions. The session 1 and 2
reports retain their historical worker digests and
schemas; their raw artifacts have not been rewritten. Before any next
session, copy the current `VlmFirstUseWorker.java` into the old worktree's
untracked worker path and rebuild that tree's `installDist`. Otherwise the
launcher rejects the byte-different worker source (or a stale benchmarks
jar). The new checkout also needs its `installDist` rebuilt from this
commit. The section 4 trigger applies to both workloads: session 2's
phi3v pair 0 and pair 3 gaps exceed 10% as well as the qwen3vl shifts,
so both cells close as `unstable`, without a settled ratio or runtime
attribution from session 2.

### 17.8 CI one-pair native VLM smoke (2026-11-12, PR #92)

The labeled [GitHub Actions run](https://github.com/Alipsa/jmlx/actions/runs/43307319179)
`43307319179` (attempt 1) executed head
`76dac46c323c54d99e8bcd6596218c0207703a32` against old tree
`46cb325bec9b4878534cc5c6bda1a569dfe4aa4c`. Both trees used the
byte-identical worker source SHA-256
`76c053b8f1a52dbfaa8b27983e81d223c794f1585e380a0286f25b5411efa439`
and rebuilt `installDist` from their source. The worker reports record
Java 25.0.5; the two full sessions used 25.0.4.1.

| Tree | `mlxMetalVersion` | `mlxcCommit` | `mlx.metallib` SHA-256 | `libmlxc.dylib` SHA-256 | `libmlx.dylib` SHA-256 |
| --- | --- | --- | --- | --- | --- |
| old | 0.31.2 | `fba4470b89073180056c9ea46c443051375f7399` | `8c8bfcece8c0610745b68879771e5aa1b92b29fa5e17172e5508e4f5153d8d15` | `f63fe68131e23876859497369069ef2ad3c869569563fd70bab8ab65ec0b8847` | `2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd` |
| new | 0.32.4 | `a26f2d584a14ec619ae8db9a09be03becd47598a` | `c6d7a3facb83568f6e4cb2cf4a8c736ee29302571a48343cb9a3a4d1cd3e3893` | `3b17e14ba9fffc430983c9f216935542e72abc6872a6d0ec103b4a838bbf63ce` | `d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6` |

The protocol gate passed **4/4 real `ok` workers**: one old/new pair for
each workload, with matching exit codes, tree/cell identities, pin values,
and two measured iterations per worker. The after-first-forward MLX peaks
reproduce the prior shape: qwen3vl-2b old/new both 9,828,890,788 B;
phi3v-4b old 31,100,376,165 B and new 30,202,745,961 B. These are
phase-local peaks. The first pre-fork host record has `omlx=40.0%`, so
the host was **not quiet**. The smoke's per-pair warm ratios, based on
only two measured iterations per worker, are protocol observations, **not
performance evidence**; they do not change either cell's `unstable`
closure in section 17.6 or support runtime attribution.

The downloaded 18-file Actions artifact is archived as
`req/plans/evidence/wp4-native-smoke-43307319179-1.zip` (SHA-256
`a85a1e2b44452a9dbe246d107a849349618cb0a32b18c1f9ff58a0a4b33e12b6`).
`req/plans/phase8-2-wp4-raw-digests.json` schema v3 records the archive
and every contained file's digest, independently of the Actions artifact's
14-day retention. Its 181 historical scratch-tree digests remain unchanged.
