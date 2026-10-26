# Phase 6 Tier-B artifact manifest

Tier-B checks use named real Hugging Face artifacts and are manual or scheduled only. They are not
required pull-request checks, never download credentials or multi-gigabyte weights implicitly, and
must record every field below before an artifact is accepted as compatibility evidence.

| Architecture | Repository candidate | Revision | File hashes | License / access | Expected output and runtime pin | Download / peak memory | Trigger owner | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama | [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | `12fd25f77366fa6b3b4b768ec3050bf629380bac` | all consumed files in [`tools/tier-b/smollm2-135m-instruct.json`](../tools/tier-b/smollm2-135m-instruct.json) | Apache-2.0; public | `model_type=llama`; exact 16 chat-generated IDs, chat text/IDs, and runtime pin in machine manifest | 271,169,733 bytes consumed; 300,000,000-byte cap; highest sampled test-JVM RSS 461,424 KiB | project maintainer | verified-with-real-artifact; [exact-token run](https://github.com/Alipsa/jmlx/actions/runs/36640308134) |
| Qwen2 | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | `7ae557604adf67be50417f59c2c2f167def9a775` | all consumed files in [`tools/tier-b/qwen2.5-0.5b-instruct.json`](../tools/tier-b/qwen2.5-0.5b-instruct.json) | Apache-2.0; public | `model_type=qwen2`; exact 16 chat-generated IDs, chat text/IDs, and runtime pin in machine manifest | 995,137,433 bytes consumed; 1,100,000,000-byte cap; highest sampled test-JVM RSS 1,018,112 KiB | project maintainer | verified-with-real-artifact; [exact-token run](https://github.com/Alipsa/jmlx/actions/runs/36640308134) |
| Qwen3 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `c1899de289a04d12100db370d81485cdf75e47ca` | all consumed files in [`tools/tier-b/qwen3-0.6b.json`](../tools/tier-b/qwen3-0.6b.json) | Apache-2.0; public | `model_type=qwen3`; exact 16 chat-generated IDs, chat text/IDs, and runtime pin in machine manifest | 1,514,733,440 bytes consumed; 1,520,000,000-byte cap; highest sampled test-JVM RSS 867,520 KiB | project maintainer | verified-with-real-artifact; [observed run](https://github.com/Alipsa/jmlx/actions/runs/37301827462) |
| Mistral | [OuteAI/Lite-Mistral-150M-v2-Instruct](https://huggingface.co/OuteAI/Lite-Mistral-150M-v2-Instruct) | `6e2f90cbf312921289b3b8c29cb34cbdb6cb708e` | all consumed files in [`tools/tier-b/lite-mistral-150m-v2-instruct.json`](../tools/tier-b/lite-mistral-150m-v2-instruct.json) | Apache-2.0; public | `model_type=mistral`; exact 16 chat-generated IDs, chat text/IDs, and runtime pin in machine manifest | 627,886,087 bytes consumed; 650,000,000-byte cap; highest sampled test-JVM RSS 802,064 KiB | project maintainer | verified-with-real-artifact; [exact-token run](https://github.com/Alipsa/jmlx/actions/runs/36640308134) |
| Gemma v1 | [google/gemma-2b](https://huggingface.co/google/gemma-2b) | — | — | Gemma terms acceptance and token required | — | about 5.02 GB weights plus runtime; peak unmeasured | project maintainer | candidate pending |
| Phi-3 | [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | — | — | MIT; public | — | index reports 7,642,159,104 bytes of weights, exceeding hosted runner RAM before runtime overhead | self-hosted runner owner pending | candidate pending |
| Mixtral | [M4-ai/TinyMistral-6x248M](https://huggingface.co/M4-ai/TinyMistral-6x248M) evaluated; smaller runnable candidate pending | — | — | Apache-2.0; public | — | 4.01 GB weight file; hosted-runner peak unmeasured, dense expert execution cost unmeasured | project maintainer | candidate pending |

On 2026-09-29, [GitHub's hosted runner specification](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
listed `macos-26` arm64 as an M1 runner with **7 GB RAM and 14 GB SSD**. The Phi-3 weight total above
comes from the [model's safetensors index](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/model.safetensors.index.json).
The [pinned Qwen Instruct weight file](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/model.safetensors) is
988,097,824 bytes and publishes the SHA-256 recorded in the machine-readable manifest. The three
small consumed JSON files were downloaded at that revision and hashed locally. Gemma's [gated repository](https://huggingface.co/google/gemma-2b) requires accepting its
terms; its file listing shows a 4.95 GB shard and a 67.1 MB shard. Qwen's exact-token run used
`Apple M1 (Virtual)`, macOS 26.6.2, aarch64, mlx-metal 0.31.2, and mlx-c
`fba4470b89073180056c9ea46c443051375f7399`; it produced the 16 IDs in the machine
manifest. Its highest sampled **test JVM RSS** across recorded runs was 1,018,112 KiB. That
measurement does not include separate Gradle processes or all unified GPU allocations; the
successful hosted-runner load is the practical fit evidence. Llama's exact-token run used the
same recorded Apple M1/macOS/MLX pin; the highest sampled test-JVM RSS across its recorded runs
was 461,424 KiB. Mistral's exact-token run used the same runner pin, asserted the 16 IDs in
its machine manifest, and sampled a peak test-JVM RSS of 802,064 KiB. Qwen3's hosted run logged the 16 IDs in its machine manifest on the Apple M1 (Virtual),
macOS 26.6.2 runner pin, with a peak sampled test-JVM RSS of 867,520 KiB; that run
predated this manifest's pin update, so it passed structurally, and the recorded pin now
enables the exact-ID assertion on subsequent matching hosted runs. A local Apple M5 Max run
on the same checkpoint asserted the identical 16 IDs on its recorded pin. The Tier-B generation
requests use the same rendered chat IDs checked before model loading. The remaining rows are
candidate evaluations, not Tier-B evidence.

The opt-in `tierBTest` source set accepts a local checkpoint. The scheduled Tier-B workflow
downloads only pinned public Llama, Qwen, Qwen3, and Mistral files with per-file and total hard caps,
checks every SHA-256,
bootstraps native MLX, and asserts exact greedy IDs when its measured device/macOS/MLX pin
matches. On a different pin it reports a structural pass and does not claim token parity.

Adding a row requires an immutable repository revision and hashes for every consumed config,
tokenizer, index, and weight file. Record whether authentication or license acceptance is required,
the exact model metadata and prompt/token output asserted, the maximum download/cache size, and who
owns manual/scheduled execution. A passing Tier-A synthetic fixture is not Tier-B evidence.

`tierBTest` explicitly sets `MLX_ENABLE_TF32=0` and declares that mode as a task input.
Exact greedy-ID comparisons therefore use full float32 independently of the launching shell.
The recorded artifacts were verified on the documented device; this does not establish
identical greedy IDs across GPU generations.

## Phase 7.2 task manifests (2026-10-05)

| Task | Manifest | Immutable revision | Consumed bytes | License |
| --- | --- | --- | ---: | --- |
| embedding | tools/tier-b/minilm-l6-v2.json | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 | 91346791 | Apache-2.0 |
| classification | tools/tier-b/bert-base-uncased-sst2.json | ce4cfd087e0c988beae0699f77a610bf7916742c | 438431729 | Apache-2.0 |
| seq2seq | tools/tier-b/flan-t5-small.json | 0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab | 310308220 | Apache-2.0 |

The Intel MRPC candidate at `844e804944328e45c3f3bf69b35467e0ce0b26af` has no safetensors payload;
the planned licensed fallback is SST-2. Pair tokenization remains independently covered.
Use `./gradlew downloadTierBArtifact -PtierBManifest=<manifest> -PtierBTarget=<directory>`, then
set JMLX_TIER_B_MODEL_DIR and JMLX_TIER_B_MANIFEST for `:jmlx-models:tierBTest`.
The manifest task selects a separate suite; both BERT artifacts can therefore use model_type=bert.
Optional JMLX_TIER_B_TASK asserts the task. Legacy decoder manifests retain generation behavior.
Every file is SHA-pinned, including nested pooling metadata. The downloader is Java buildSrc;
its tests run in the existing `./gradlew -p buildSrc check` CI step. Python download.py was removed.
Actual sentence-transformers and HF CPU references, source hashes, full logits, gaps, final local
measurements and host are recorded in manifests. Numerical comparisons always run; exact IDs
require stable margin eligibility. Initial macOS 26 workflow acceptance remains pending.

## Phase 7.3a WP7 image/text candidate (2026-10-11)

`tools/tier-b/smolvlm-256m-instruct.json` pins public Apache-2.0
`HuggingFaceTB/SmolVLM-256M-Instruct` at `7e3e67edbbed1bf9888184d9df282b700a323964`, all
13 consumed files (517,903,046 bytes), a committed MIT synthetic RGB image, structured chat,
16 unexpanded / 347 expanded prompt IDs and four exact greedy IDs `[378, 2443, 21559, 354]`.
Task `image-text-generation` dispatches `TierBVisionLanguageTest` in the existing TF32-off profile.
The weekly/on-demand row is owned by the project maintainer and uploads numerical/memory evidence.

Local macOS 27 measured maximum all-vocabulary logit error `5.2928924560546875e-5` (allowed
`1e-4`), minimum top-two gap `0.26448631286621094` (required `> 0.0003`), native allocator peak
5,737,416,464 bytes and highest 20-ms sampled process phys_footprint 7,129,368,904 bytes. The
suite gates the process footprint at 8 GiB — the RAM a hosted runner must actually hold — and
the native allocator peak at 6,261,704,464 bytes; test-JVM RSS is not asserted because it misses
the MLX/Metal allocations. This is local eligibility evidence.

WP8 (2026-10-13, `phase7-3a-implementation-report.md`) re-ran this exact manifest on the merged
tree: identical greedy IDs, identical maximum logit error, and a fresh margin-eligible run, closing
the seven-artifact consumer and capability-documentation acceptance. The row remains local
eligibility evidence until the first CI execution lands — the weekly Tier-B row now targets
the self-hosted `self-macos-xl` runner (GitHub's larger `macos-26-xlarge` tier is billed per
minute even in public repos, so the row does not target it; `tools/tier-b/README.md`).

## Phase 7.3b Qwen3-VL Tier-B (2026-10-26)

`tools/tier-b/qwen3vl-2b-instruct.json` pins `Qwen/Qwen3-VL-2B-Instruct` at revision
`89644892e4d85e24eaac8bacfd4f463576704203` (Apache-2.0, license evidence URL in the manifest):
the 11-file WP0 inventory, 4,266,647,442 bytes in aggregate, each file SHA-256- and
size-capped, downloaded through the Java `downloadTierBArtifact` manifest/SHA path (verified
locally, 4,266,647,442 bytes).

**Request.** One multi-image request (the plan's required multi-image prompt): a user message
carrying two images — `tools/tier-b/images/qwen3vl-wide-200x40.png`
(sha `61f3af15…`, copied unchanged from the image-oracle `resize/wide-200x40.png` fixture, whose
source hash is provenance-pinned) and `tools/tier-b/images/qwen3vl-square-96x96.png`
(sha `44d78692…`, same provenance) — with the text "Describe both images.",
`max_new_tokens: 8`. Both images are below the artifact's declared 65,536-pixel minimum, so the
declared min-pixel clamp drives the grids: `[[1, 8, 36], [1, 16, 16]]` (merged-row counts 72 and
64). The chat template is the artifact's own (`tokenizer_config.json` string form, verified
byte-identical to `chat_template.json`); the recorded unexpanded prompt is 18 tokens and the
expanded prompt is 152 tokens (RoPE delta −110).

**Reference.** `tools/hf-reference/qwen3vl_tier_b.py` (hash-locked CPU float32, single-threaded
eager, transformers 4.57.6; the 4.57.6 slow-processor size-override bug compensated with
explicit `min_pixels`/`max_pixels` exactly as the WP1 probe documents) renders the prompt,
expands the two images, and records the full-vocabulary logit vector at every expected greedy
history step. The vectors are stored as a little-endian float32 binary
(`tools/hf-reference/goldens/qwen3vl-tier-b-logits.bin`, 4,861,952 bytes, sha
`2b44f9f2…9ae76afb`; one full 151,936-vector per step — a documented size-driven deviation from
the JSON logits goldens, following the shared Qwen3-VL patch-bin precedent). Provenance pins the
manifest, generator, both images and the binary in
`tools/hf-reference/provenance.json` (`qwen3vl_tier_b_files` + the `files` entry), verified by
root `verifyHfReferenceGoldens`.

Reference greedy IDs: `[785, 1378, 5335, 3897, 525, 19516, 13, 2379]` (finish MAX_TOKENS; no EOS
in 8 steps). Measured minimum top-two logit gap across the eight steps: 0.10282325744628906
(step 6, gap 19516 vs 7009).

**Java suite.** `TierBQwen3VlTest` (selected by task `qwen3-vl-generation`, part of the
`tierBTest` suites map): re-verifies the manifest's image digests, re-processes each image
through `jmlx-vision`'s Qwen3-VL processor and asserts the reference's per-image
`pixel_values_sha256` (byte-for-byte preprocessing on the real artifact), asserts the rendered
prompt, unexpanded IDs, expanded IDs, delta and per-image token runs, compares every cached
decode/prefill logit vector against the reference binary under the recorded epsilon, re-derives
the reference's own recorded gap/top-two from the reference vectors (pin of the reference's
internal consistency), and applies the stability gate to both the reference vectors and the
Java run's own measured margins before asserting the exact greedy IDs, the exact decoded
output (the case's `expected_text`, the reference's
`tokenizer.decode(target_ids, skip_special_tokens=True)`), the prompt positions,
finish reason and the end-to-end `generate` result.

**Measured local evidence** (manifest `local_evidence`; M5 Max / macOS 27.0.1, TF32 off, pinned
mlx-metal 0.31.2 / mlx-c fba4470): maximum Java-vs-reference absolute logit error
**1.0228157043457031e-4** (stable across repeated runs), hence the recorded
ε = 2e-4 / margin = 1e-4 (eligibility bound 2ε + margin = 5e-4, two orders of magnitude below
the 0.1028 minimum gap). MLX allocator peak 9,828,890,788 bytes (suite bound: peak + 500 MiB);
sampled process footprint peak 11,979,116,480 bytes (suite bound: 12,500,000,000 bytes,
~520 MB above the measured peak). Load wall
~843 ms; one-request generation wall ~247 ms.

**CI status: pending, not verified.** A `qwen3vl` row was added to
`.github/workflows/tier-b.yml` on the self-hosted `self-macos-xl` runner (the standard 7 GB
runner cannot hold the 11.98 GB measured footprint, and GitHub's larger `macos-26-xlarge`
tier is billed per minute even in public repos, so the row does not target it). The suite's
footprint bound fails the suite on a memory regression, but it is not a guarantee the
runner's RAM survives a large regression (the OS and the Gradle daemon share that RAM, and
macOS may swap or kill first), and the local M5 Max measurement is the only runner-fit
evidence so far — the first CI run is acceptance. While the self-hosted runner is offline
the row's jobs queue; the other rows run on the standard hosted runner
(`tools/tier-b/README.md`).
