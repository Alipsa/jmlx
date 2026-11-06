# Phase 8.2 — Reference findings (WP0)

- **Status:** WP0 readiness and comparison-contract record, 2026-11-06. All evidence below was
  collected on the measurement host named in §8 from locally verified inputs; no headline
  measurement was executed.
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
`e48041c7d1f671d5b99847410c99893f290605ee055062039cac9091be1073dc`, which the guard pins
as the frozen-contract backstop (§2.3).

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
is 151645 — the `
</think>

` special token — not 151643 (`</s>`), and the `phi3v`
tokenizer eos is 32000 (`endoftext`), not 32001 (`<|assistant|>`); both were
re-verified against the local digest-pinned artifact tokenizer files on 2026-11-06. `qwen3-8b-4bit` has no
`generation_config.json`. Both stock loops stop on exactly these sets, so the stock
comparison is stop-matched (plan §3.2 "Stock Java pair path").

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
`{2, 32000}` (the tokenizer eos is 32000 `endoftext`, 32001 is `<|assistant|>`), while the
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
rejection, and the absence of `mlx-vlm`), and finally the manifest's own SHA-256 as the
frozen-contract backstop for any field the structural checks do not name. Negative tests
executed on 2026-11-06: file-sha, prompt-array, controlled-policy, stock-policy,
stop-union, identity-claim, VLM-path, VLM-manifest, note-drift (digest backstop), malformed
requirements pin, extra-without-hashes and schema tampers each fail with a distinct error;
the restored manifest passes.

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
