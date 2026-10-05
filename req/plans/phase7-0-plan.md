# jmlx Phase 7.0 — Qwen3 text decoders

**Plan for:** `req/plans/phase7-plan.md` §7.0 (milestone added by the dated roadmap amendment of
2026-10-04).

## 1. Goal, non-goals, prerequisites

**Goal.** Support the dense Qwen3 decoder family (`model_type: qwen3`) end to end in the existing
Phase 6 stack: config parsing and `ArchitectureDescriptor` mapping, tensor-plan validation, assembly,
generation, sampling, tokenizer/chat-template rendering through the real Qwen3 template,
affine-quantized (4-bit) checkpoints, Tier-A synthetic goldens, a pinned Tier-B artifact, and a
recorded manual 8B run.

**Non-goals.** Anything 7.1–7.4 owns (new `nn` layers, `MLXOps`/`MLXShape` slices, encoders,
multimodal). `qwen3_moe` is rejected by name. Generated thinking content is returned as raw
text; parsing it is deferred (Phase 7 master plan deferrals).

**Prerequisites:** step 0 roadmap amendment (merged on this branch). No native API additions are
required; every binding this milestone touches is already `implemented`.

**Exit gate (from §7.0):** Qwen3 synthetic logits match the HF float32 reference at the strict
`1e-4` bound in `float32GoldenTest`; the default-TF32 mode records its measured error under a
separately named bound; both chat-template goldens pass; Qwen3-0.6B exact greedy IDs pass Tier-B;
the 8B 4-bit run and benchmark are recorded; the compatibility matrix gains the Qwen3 row and its
Quantization column is reconciled with the affine support recorded below the matrix (PR #31).

## 2. Verified architecture facts (checked 2026-10-04)

Verified against live artifacts at pinned revisions and `transformers` 4.57.6 sources, not from
plan prose (a prior review found plan prose wrong on facts 1, 3 and 8):

1. **There is no `Qwen/Qwen3-*-Instruct` repository.** Qwen3 ships one checkpoint per size;
   thinking mode is toggled by the chat-template variable `enable_thinking`. The Tier-B
   reference is **`Qwen/Qwen3-0.6B`** at revision
   `c1899de289a04d12100db370d81485cdf75e47ca`.
2. Qwen3-0.6B `config.json` SHA-256 `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`;
   complete key list is in the mapping table (§6).
3. **`layer_types` is absent** (not explicit `null`) from every live dense size checked
   (0.6B, 1.7B, 4B, 8B, 14B, 32B). It is accepted when absent or null; a present array must be
   all `full_attention` and match `num_hidden_layers`.
4. Qwen3-8B `config.json`@`b968826d9c46dd6066d109eabc6255188de91218`:
   `tie_word_embeddings: false`, `head_dim: 128`, no `layer_types`. Qwen3-0.6B:
   `tie_word_embeddings: true`, `head_dim: 128`, `num_attention_heads: 16`,
   `hidden_size: 1024` — **`head_dim × heads ≠ hidden` is normal for Qwen3** and the existing
   non-Gemma product check must not reject the family.
5. `transformers/models/qwen3/modeling_qwen3.py`: `q_norm`/`k_norm` are
   `Qwen3RMSNorm(head_dim, eps=config.rms_norm_eps)` applied to the `[..., -1, head_dim]`
   reshape of the projection output **before** `transpose(1, 2)` and before RoPE — i.e. RMS over
   the last axis. Applying them after jmlx `AttentionHeads.toHeads` (which lands in
   `[B, H, T, D]`) is numerically identical, since `MLXFast.rms_norm` reduces the last axis with
   a `[D]` weight. They have **no bias** parameter: the existing weight-only `RMSNorm` matches.
6. The attention scale is `head_dim**-0.5` — the hard-coded `1/sqrt(headDim)` in
   `DecoderAttention` already matches once `head_dim` is the configuration's explicit value.
7. All four projections use `bias=config.attention_bias` (false at 0.6B/8B), so
   `Attention.qkvBias == Attention.outBias == attention_bias` for this family.
8. The chat template lives in `tokenizer_config.json` (`tokenizer_class: Qwen2Tokenizer`; no
   separate `chat_template.jinja`). Template SHA-256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
   Its `enable_thinking` semantics are **inverted** relative to the master plan's paraphrase:
   `enable_thinking is defined and enable_thinking is false` appends an empty thinking block
   (think-open tag, newline, think-close tag, blank line) after the assistant role-open token;
   `enable_thinking` true or undefined leaves the turn open-ended. The `transformers`
   `Qwen3` processor passes `enable_thinking` explicitly, so the pinned golden sets record the
   actual rendered text rather than the master plan's assumed prefix.

## 3. Exact public API and files

**`ArchitectureDescriptor.Attention`** gains a fifth component `boolean qkNorm` (record, so a
breaking change to the canonical constructor). Exactly two production call sites are updated:
`ArchitectureMappings.dense` and the test helper `TestDescriptors.dense`. No builder is added; the
flag is set by the family, not by a config key (Qwen3 always has qk-norm; there is no
`qk_norm` config field in the live artifacts).

**`DecoderAttention`** (jmlx-core) gains two nullable `UnaryLayer` constructor parameters
`qNorm`/`kNorm` (an overload; the existing constructor delegates with `null`). When present they
are registered as children `qNorm`/`kNorm` and applied after the three `toHeads` calls and before
either `rope.apply` branch — i.e. exactly the HF ordering from fact 5. The
`parameters()` key set gains `qNorm.weight`/`kNorm.weight` only when enabled.

**`ArchitectureMappings`**:
- New `FAMILIES` entry `"qwen3"`: `WindowPolicy.IGNORE`, `honorsAttentionBias()`, `qkNorm()`,
  `acceptsMaxWindowLayers()`, `acceptsLayerTypes()`.
- New `Family`/`Builder` capability flag `qkNorm` (default off).
- `dense()`: the explicit-`head_dim` branch (currently Gemma-only) becomes a family capability.
  Qwen3 honors an explicit `head_dim` without the `head_dim × heads == hidden` product check,
  keeping `headDim = head_dim` and `rotaryDims = head_dim` (partial factor 1).
- `tensorPlan`: when `qkNorm`, require `self_attn.q_norm.weight`/`k_norm.weight` per layer and
  forbid them otherwise. They are never in the `addQuantizedCompanions` allow-list, so a
  `q_norm.scales`/`k_norm.scales` tensor is `unexpected` and fails.
- `parse`: an explicit named rejection of `qwen3_moe` (deferred), mirroring the `gemma2`/`gemma3`
  pattern.

**`DecoderAssembler.assemble`**: when `qkNorm`, construct two `RMSNorm(scope, tensor(tensors,
"…q_norm.weight"), eps, weightOffset=false)` (and `k_norm`) and pass them into the
`DecoderAttention` overload. Norm weights stay float; a packed/`UINT32` norm weight is rejected by
the existing `validatePackedTensors`.

**`TextGenerationModels`**: `case "qwen3" -> QwenModel.create(scope, descriptor, directory)`.
`QwenModel.load` keeps asserting `model_type qwen2`; `TextGenerationModels.load` is the public
entry that dispatches qwen3. `QwenModel`'s javadoc is updated to say it serves both `qwen2` and
`qwen3`.

**`ModelMetadata`**: unchanged this milestone (no `inputModalities` — that is 7.3a). No images
support is added; a qwen3 model given `GenerationRequest` images would be a 7.3a concern.

## 4. Ordered implementation tasks

1. `ArchitectureDescriptor.Attention` + the two constructor call sites (compiles with the flag
   present but unused).
2. `DecoderAttention` qk-norm overload + `parameters()`/javadoc; jmlx-core unit tests (hand-
   computed RMS over the last axis before RoPE).
3. `ArchitectureMappings` family/table/head_dim/tensorPlan/`qwen3_moe` rejection; mapping tests.
4. `DecoderAssembler` wiring; `TextGenerationModels` dispatch; `QwenModel` javadoc.
5. `TinyCheckpoints` qwen3 writer + `Qwen3ModelTest` (native): load, greedy, tied/untied,
   missing-norm rejection, quantized-norm rejection, bias-on variant.
6. `tools/hf-reference` `qwen3` family + runner-generated goldens + provenance.
7. Qwen3 tokenizer bundle + `jmlx-jinja` real-template rendering + chat golden + oracle fixtures.
8. `DecoderRefactorGoldenTest`/`BatchStepEquivalenceTest`/scheduler `@ValueSource` add `qwen3`;
   CI required-suite list + `verifyHfReferenceGoldens` update.
9. Tier-B `qwen3-0.6b.json` manifest + `.github/workflows/tier-b.yml` entry + local run.
10. 8B 4-bit manual run, peak-memory and benchmark recording; compatibility-matrix + Quantization
    column reconciliation.
11. `req/phase6-4-benchmark.md` and `req/phase6-compatibility.md` updates.

## 5. Native probes

No new probe is required: qk-norm reuses `MLXFast.rmsNorm` (last-axis RMS, already oracle-tested)
and the existing `toHeads`/`rope.apply` path. No new `mlx_h` downcall, no new struct layout, no
new error-convention surface. This is recorded here to satisfy Rule 2 explicitly.

## 6. Config-key mapping table (Qwen3-0.6B live artifact, revision c1899de)

Every key of the live `config.json` and its handling in `ArchitectureMappings.parse`. No key may
fall through to the unknown-key warning.

| Key | Value (0.6B) | Handling |
| --- | --- | --- |
| `model_type` | `qwen3` | family lookup; dispatch in `TextGenerationModels` |
| `architectures` | `[Qwen3ForCausalLM]` | `IGNORED` (metadata only) |
| `attention_bias` | `false` | honored: qkv and o biases (family flag `honorsAttentionBias`) |
| `attention_dropout` | `0.0` | `IGNORED` (training only) |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | — | `IGNORED` (tokenizer/generation settings) |
| `head_dim` | `128` | new family capability: explicit head dim, no `× heads == hidden` check; `rotaryDims = head_dim` |
| `hidden_act` | `silu` | existing activation resolver |
| `hidden_size` / `intermediate_size` | `1024` / `3072` | `CONSUMED` |
| `initializer_range` | `0.02` | `IGNORED` (training only) |
| `max_position_embeddings` | `40960` | `CONSUMED` (cache/validation) |
| `max_window_layers` | `28` | accepted, inert (`acceptsMaxWindowLayers`, `use_sliding_window` false) |
| `num_attention_heads` / `num_key_value_heads` / `num_hidden_layers` | `16` / `8` / `28` | `CONSUMED` |
| `rms_norm_eps` | `1e-06` | `CONSUMED`; also the `q_norm`/`k_norm` eps |
| `rope_scaling` | `null` | `CONSUMED`; null accepted; scaled-RoPE paths reused |
| `rope_theta` | `1000000` | `CONSUMED` |
| `sliding_window` | `null` | `CONSUMED`; `WindowPolicy.IGNORE` accepts absent/null |
| `tie_word_embeddings` | `true` (0.6B) / `false` (8B) | `CONSUMED`; `Head.tied`; lm_head required iff untied |
| `torch_dtype` | `bfloat16` | `IGNORED` (storage dtype) |
| `transformers_version` | `4.51.0` | `IGNORED` (export metadata) |
| `use_cache` | `true` | `IGNORED` (inference always caches) |
| `use_sliding_window` | `false` | `CONSUMED`; `true` rejected by the existing family-independent check |
| `vocab_size` | `151936` | `CONSUMED` |
| `layer_types` | **absent** on all live sizes | accepted when absent/null; present must be all `full_attention`, length == `num_hidden_layers` |
| `qwen3_moe` (`model_type`) | — | explicit named rejection, deferred to a dated follow-on |

The `generation_config.json` (revision c1899de) carries `do_sample: true`, `temperature: 0.6`,
`top_k: 20`, `top_p: 0.95`, `eos_token_id: [151645, 151643]`, `pad_token_id: 151643`; jmlx reads
`generation_config.json` only for the stop tokens, as it does today.

## 7. Tests

**jmlx-core (`:jmlx-core:test` / `float32GoldenTest`):**
- `DecoderAttention` qk-norm: hand-computed case (random weights, identity projection, known
  `RMSNorm` over the last axis applied to Q/K before RoPE) at the strict `1e-4` bound under
  `full-float32`, plus a default-mode bound; flag-off path has unchanged `parameters()` keys;
  scope/confinement for the two norm temporaries.

**jmlx-models (pure-Java):**
- `ArchitectureMappingsTest`: the live 0.6B config parses to the descriptor facts in §6
  (`headDim == 128`, `qkNorm`, no biases, `tieWordEmbeddings`, `ropeTheta 1e6`); live 8B config
  parses with `tie_word_embeddings: false`; `qwen3_moe` rejected by name; `use_sliding_window:
  true` rejected; per-family maps gain the `qwen3` entry.
- `TensorPlanTest`: `q_norm`/`k_norm` required when enabled, forbidden otherwise, and a
  `q_norm.scales`/`.biases` tensor is `unexpected`.

**jmlx-models (native, `@EnabledIfNativeAvailable`):**
- `Qwen3ModelTest` (new): tiny `TinyCheckpoints` qwen3 checkpoint (head_dim ≠ hidden/heads to
  exercise the explicit path), tied and untied variants, load + exact greedy IDs, seeded
  sampling, missing `q_norm.weight` rejection, `attention_bias: true` variant requires all four
  projection biases, packed `q_norm.weight` rejected.
- `QuantizedDecoderTest`: affine-quantized qwen3 synthetic case (norm weights stay float,
  dequantized equivalence against the float checkpoint).
- `DecoderRefactorGoldenTest` and `BatchStepEquivalenceTest` gain the `qwen3` `@ValueSource`
  entry; the scheduler `@ValueSource` lists gain `qwen3` where they enumerate families.

**jmlx-tokenizer / jmlx-jinja (pure Java):**
- `Phase63FamilyTokenizerTest` + `ChatTemplateRendererTest` gain the `qwen3` family: the bundled
  real template renders the enable_thinking true/false, system/no-system, multi-turn-with-assistant
  and tool-call cases; output matches `tools/hf-reference/goldens/chat-qwen3.json` (rendered text
  and token IDs).
- `jmlx-jinja` `InterpreterTest` (or the model-template corpus suite) gains the committed real
  template with the same case matrix and the pinned expected text.
- `tools/tokenizer-oracle`: encode/decode fixtures for the committed qwen3 bundle, regenerated via
  `generateTokenizerOracleFixtures` and verified by `verifyTokenizerOracleFixtures`.

**CI:** `.github/workflows/ci.yml` "Assert required native suites executed" gains
`Qwen3ModelTest`; `float32_suites` is unchanged unless the new qk-norm test is tagged
`full-float32` in `:jmlx-core` (it is: it goes on the core float32 list).

## 8. Fixtures and goldens

**Tier-A (in-repo, every PR):** `tools/hf-reference` gains `--family qwen3`:
- `FAMILIES` appends `"qwen3"` (last, so no existing seed changes);
- `config_for`: `Qwen3Config(**common, head_dim=32)` — explicitly different from
  `hidden/heads == 16` so the golden exercises the explicit-head-dim path; `tie_word_embeddings`
  stays `False` (lm_head present);
- `expected_tensor_names`: q/k/v/o weights, **no projection biases**, plus
  `self_attn.q_norm.weight`/`k_norm.weight` per layer;
- output `goldens/qwen3.json` (prefill + 2 greedy decode steps) and
  `goldens/checkpoints/qwen3/{config.json,generation_config.json,model.safetensors}`, all
  SHA-256-pinned in `provenance.json` by `verifyHfReferenceGoldens`.

Generation host: the pinned environment (`torch 2.6.0+cpu`, `transformers 4.57.6`,
`safetensors 0.6.2`, CPython 3.12) is only available on a Linux x86_64 host — this machine has
Python 3.14 and no matching torch. The goldens are produced by a **temporary `workflow_dispatch`
job** on a `ubuntu-24.04` runner (the same class of host that produced the existing
`Linux/x86_64` provenance), run as `generate.py --family qwen3 --out goldens`, and the
`qwen3.json`, `checkpoints/qwen3/*` and rewritten `provenance.json` are downloaded back as
artifacts and committed. The temporary workflow is deleted in a follow-up commit. No existing
golden is regenerated, so all other file hashes in `provenance.json` stay byte-identical.

**Chat goldens:** `tools/hf-reference/generate.py --chat` is extended with `--family qwen3`:
`CHAT_FAMILIES` appends `qwen3`; the `qwen3` case set renders through the committed bundle with
`enable_thinking` true/false × (no system / system) × plain/multi-turn, plus one tool-call
conversation, each at `add_generation_prompt` false/true. `generate_chat` gains an optional
per-family `extra` keyword-argument map passed to `apply_chat_template`. The committed bundle is
a small synthetic byte-level BPE vocabulary (per the existing `families/` convention, not the
11 MB production vocab) whose added tokens cover the template's special tokens, paired with the
**real** Qwen3 chat template text (SHA-256 `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`), also committed to the
`jmlx-jinja/src/test/resources/model-templates/` corpus. `chat-qwen3.json` plus the bundle hashes
in `provenance.json` `chat_sources` are produced by the same runner job.

**Tier-B:** `tools/tier-b/qwen3-0.6b.json` pins `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca` (files: `config.json`, `generation_config.json`,
`tokenizer.json`, `tokenizer_config.json`, `model.safetensors`, with per-file and total size caps),
records the recorded_pin/reproducibility tuple, the exact greedy chat prompt IDs and expected
generated token IDs from a local M5 run, and the observed peak test-JVM RSS. `.github/workflows/
tier-b.yml` runs it on schedule; the local run is the evidence recorded in the manifest.

**8B 4-bit manual run:** `mlx-community/Qwen3-8B-4bit` (revision
`545dc4251c05440727734bcd94334791f6ab0192`, MLX affine 4-bit — the exact shape PR #31 loads) is
downloaded locally, loaded through `TextGenerationModels.load`, run for a short greedy decode,
and peak native memory + download size are measured against the recorded CI runner budget in
`req/phase6-tier-b-artifacts.md` (M1, 7 GB RAM, 14 GB SSD) to record whether it fits with margin.
It stays a manual, evidence-only run recorded in `req/phase6-4-benchmark.md`.

## 9. Precision gates and benchmarks

- **Strict:** the qwen3 `DecoderRefactorGoldenTest` case asserts the CPU float32 HF reference at
  `1e-4` under `float32GoldenTest` (`MLX_ENABLE_TF32=0`), same as the other families.
- **Default mode:** the ordinary `test` task asserts the same goldens under a separately named
  default-mode bound recorded next to the Phase 6 measurements; the measured maximum error is
  written down in `req/phase6-golden-precision.md`. No golden is regenerated to accommodate it.
- **Benchmarks:** `req/phase6-4-benchmark.md` gains the Qwen3-8B 4-bit local run (tokens/s and
  peak memory) alongside the Phase 6 baselines.

## 10. Documentation, inventory, deferrals

- `req/phase6-compatibility.md`: add the Qwen3 row (Tier-A synthetic + Tier-B Qwen3-0.6B + manual
  8B 4-bit) and reconcile the existing rows' Quantization column (still "float weights only")
  with the MLX affine support recorded below the matrix, which landed in PR #31.
- `req/phase6-4-benchmark.md`: 8B 4-bit run and benchmark; fit-margin note for the CI runner.
- Inventory: no new bindings, so `req/mlx-api-inventory.md` is unchanged; this is recorded here
  to satisfy Rule 1. `verifyMlxApiCallSites` stays green.
- `jmlx-models` published-POM description and `DecoderModel`/`QwenModel` javadocs gain the family.
- Deferred (from the master plan): `qwen3_moe`, video input, parsing of generated thinking
  output, and batched scheduling of qwen3 requests (the scheduler's decoder-only contract is
  unchanged by this milestone).

## 11. Verification checklist

1. `./gradlew build` (spotless, checkstyle, tests, inventory freshness, `verifyMlxApiCallSites`,
   `verifyHfReferenceGoldens`).
2. `./gradlew :jmlx-core:float32GoldenTest :jmlx-models:float32GoldenTest` — the qwen3 row of
   `DecoderRefactorGoldenTest` and the qk-norm strict case pass at `1e-4`.
3. `./gradlew :jmlx-models:tierBTest` with a locally downloaded `Qwen/Qwen3-0.6B`
   (`JMLX_TIER_B_MODEL_DIR`, `JMLX_TIER_B_MANIFEST=tools/tier-b/qwen3-0.6b.json`).
4. 8B 4-bit manual run recorded in `req/phase6-4-benchmark.md`.
5. `git diff --check`; diff review of every golden/provenance change.
## 12. Status (2026-10-05, complete pending merge)

Done and green on branch `phase7-0-qwen3`:

- Step 0 roadmap amendment (its own commit, per the master plan).
- This sub-plan, with all live-artifact verifications (fact 8's inverted
  `enable_thinking` included).
- `DecoderAttention` qk-norm (jmlx-core) + strict/cache/dimension tests
  (default and `float32GoldenTest` green).
- `ArchitectureDescriptor.Attention.qkNorm`, the `qwen3` family in
  `ArchitectureMappings` (explicit head_dim, honored `attention_bias`,
  `qwen3_moe` named deferral, tensor plan), `DecoderAssembler` wiring,
  `TextGenerationModels` dispatch, `QwenModel` javadoc.
- Mapping tests (live 0.6B/8B configs, per-family pins), `TinyCheckpoints
  .randomQwen3`, `Qwen3ModelTest` (8 tests), `QuantizedDecoderTest` qwen3
  cases. `:jmlx-models:test` (228), `:jmlx-models:float32GoldenTest` (22),
  `:jmlx-core:test` (389) and `:jmlx-core:float32GoldenTest` (13) all green.
- `tools/hf-reference/generate.py`: `qwen3` appended to `FAMILIES`/
  `CHAT_FAMILIES`, `Qwen3Config(head_dim=32)` fixture config, tensor-name
  manifest with float-only `q_norm`/`k_norm`.

Items 1-10 of the final checklist:

1. Done (`generate.py --chat qwen3` matrix: `enable_thinking` true/false,
   tool call, multi-turn with assistant) — commit `cae2455`.
2. Done (`families/qwen3/` bundle with the real template, the jinja
   `model-templates/` copy, `InterpreterTest` cases) — commit `cae2455`.
3. Done (tokenizer-oracle fixtures, provenance-pinned) — commit `cae2455`.
4. Done: goldens generated on the pinned ubuntu-24.04 runner (workflow
   run 37290082996), `provenance.json` diff reviewed (new entries only),
   `verifyHfReferenceGoldens` green — commit `1d86244`; the temporary
   workflow lived on main (PR #34) and is deleted by this branch.
5. Done: `qwen3` in `DecoderRefactorGoldenTest`, `BatchStepEquivalenceTest`
   (verified in both precision modes), `BatchGenerationSchedulerTest`,
   `BatchSchedulerFailureTest` value sources, and the quantized
   dequantized-weights case — commit `1d86244`.
6. Done: `Qwen3ModelTest` on the required native-suite list in
   `.github/workflows/ci.yml` — commit `1d86244`.
7. Tier-B manifest + `tier-b.yml` entry committed (`41ea666`); the local
   run asserted the exact 16 greedy IDs on the recorded Apple M5 Max pin
   (peak test-JVM RSS 1,910,976 KiB). The hosted-runner run happens after
   this branch merges: the workflow checks out the default branch, whose
   `ArchitectureMappings` rejects `qwen3` until the decoder code lands, so
   the arm cannot pass earlier (an early arm+manifest on main failed the
   qwen3 arm exactly that way and was reverted, PR #36). After the merge,
   dispatch the Tier-B workflow, then set the manifest's `recorded_pin`,
   `observed_peak_test_jvm_rss_kib` and `observed_run` from the qwen3 arm
   (and update the `req/phase6-tier-b-artifacts.md` row's status and link).
8. Done: `mlx-community/Qwen3-8B-4bit` (revision
   `545dc4251c05440727734bcd94334791f6ab0192`) local run recorded in
   `req/phase6-4-benchmark.md` (86.92 tokens/s median, 4,699,272,000 peak
   active native bytes, 4,619,268,473 bytes downloaded, budget-fit
   analysis); qwen3 golden-suite and batch-equivalence measured errors in
   `req/phase6-golden-precision.md` and `req/phase6-batch-equivalence.md`.
9. Done: `req/phase6-compatibility.md` Qwen3 row + Quantization-column
   reconciliation, `req/phase6-tier-b-artifacts.md` Qwen3 row,
   `jmlx-models` POM description, `DecoderModel` javadoc and README.
10. Final gate: `./gradlew build` plus the §11 checklist (see the merge
   commit's CI run for the full matrix).
