# Phase 6.3 implementation plan — architecture and checkpoint capability matrix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Llama/Qwen branches in `DecoderModel` with explicit, validated capability
descriptors, add RoPE scaling, sliding-window attention and the Mistral, Gemma, Phi and Mixtral
families, and make every claimed capability provable with a Tier-A fixture and every claimed family
provable with Tier-B evidence.

**Architecture:** A pure-Java `ArchitectureDescriptor` (records/enums, no native state) is parsed
from `config.json` by a per-`model_type` mapping, validates every required and forbidden tensor key
against the checkpoint's tensor names *before* any layer is constructed, and drives a generalized
decoder built from small `jmlx-core` components (`Attention` with a `RopeSpec`, a mask policy,
configurable norms and MLPs, a `MoeMlp`). `DecoderModel` keeps the generation loop from 6.1/6.2
unchanged and only stops knowing which family it is running.

**Tech Stack:** Java 25 (`jmlx-models`, `jmlx-core`), MLX via `jmlx-core` facades, safetensors
(`MLXIO`), Jackson 3 for `config.json`, the offline Hugging Face reference tool (`tools/hf-reference/`, committed and
hash-verified goldens) for numeric ground truth, JUnit 5.

**Spec:** `req/plans/phase6-plan.md` §6.3 (and `req/full-roadmap.md` §6.3). Related:
`req/phase6-compatibility.md`, `req/phase6-tier-a-fixtures.md`, `req/phase6-tier-b-artifacts.md`,
`req/plans/phase6-2-plan.md` (tokenizer contract this plan extends), `req/plans/phase5-m3-retrospective.md`.

**Prerequisite:** Phase 6.2, merged in PR #24 (`77b40cc`). **Branch:** `phase6-3`.

## Global Constraints

- Platform: macOS on Apple Silicon, Java 25 toolchain for `jmlx-core`/`jmlx-models`; the pinned MLX
  runtime (`mlx-metal==0.31.2`, mlx-c `fba4470`). `jmlx-tokenizer` and `jmlx-jinja` stay Java 21.
- No new module, no HTTP server, no new runtime dependency. `jmlx-models` keeps depending only on
  `jmlx-core`, `jmlx-tokenizer` and `jackson-databind` (`verifyPublishedDependencies`).
- Fail early and specifically: an unsupported config field, tensor, tokenizer feature or generation
  setting is rejected before any partial model is returned, naming the config key or tensor key and the
  missing capability. `config.json` keys follow the "Config field policy" below: nothing that affects
  numerics is ever ignored, and unlisted keys are logged, not silently dropped.
- Reference goldens must come from an implementation independent of the Java code and of the
  Java author's own port. `transformers`/`torch` are allowed **only** in the offline, generate-only
  `tools/hf-reference/` tool (Task 0); CI verifies committed golden hashes and never runs it. They are
  not added to `tools/tokenizer-oracle/` or `tools/mlx-oracle/` environments.
- Google Java Style, 2-space indent, 100 columns; `./gradlew spotlessCheck checkstyleMain
  checkstyleTest` and `git diff --check` clean before each commit. Checkstyle warning baseline must
  not grow.
- Every new native call is a `jmlx-core` facade with an inventory entry (`req/mlx-api-inventory-overrides.json`,
  then `./gradlew generateMlxApiInventory`); `jmlx-models` never reaches `jmlx-ffi`.
- Native-dependent tests use `@EnabledIfNativeAvailable`; pure-Java parsing/validation tests must run
  on Ubuntu CI (`:jmlx-models:check`) without `native/install/lib`.
- Never commit to `main` directly. Commit trailer: `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- Legacy entry points (`LlamaModel.load`, `QwenModel.load`, `DecoderModel.generate(int[],...)`,
  `generateText`) keep their signatures and greedy output; their goldens must not change.
- Reproducibility contract from Phase 6 is unchanged: no claim across changed MLX pin, device or batch
  shape.
- An architecture row in `req/phase6-compatibility.md` moves to `verified-with-real-artifact` only
  with Tier-B evidence; a Tier-A fixture only earns `verified-with-synthetic-fixture`.

## Review Focus

Inputs and failure modes the roadmap implies but no single task's happy-path tests exercise, most
likely first. Each line has a pinning test in the owning task.

1. **`config.json` fields that are present but unsupported** (`rope_scaling` with an unknown
   `rope_type`, `sliding_window` with `use_sliding_window=false`, `layer_types`, `quantization`,
   `num_local_experts` on a dense family): expected is a load-time `IllegalArgumentException` naming the
   key, never a model that computes something else. (Tasks 1, 3, 7, 8, 9)
2. **Checkpoint with an extra, unexpected tensor** (for example `model.layers.0.self_attn.rotary_emb.inv_freq`
   from older Llama exports, or a forbidden bias): expected is either an explicit allow-listed ignore
   or a named rejection, never silent use or silent drop. (Task 2)
3. **Sliding window smaller than the prompt, and cache offset past the window**: expected is output
   identical to a full-attention reference restricted to the last `window` positions, for both prefill
   and one-token decode. (Tasks 4, 7)
4. **RoPE scaling at position 0, at positions beyond `original_max_position_embeddings`, and with
   `partial_rotary_factor < 1`**: expected is numerical agreement with the oracle at multiple offsets,
   not just position 0 where every scaling scheme is the identity. (Tasks 3, 6)
5. **MoE routing ties and `top_k == num_local_experts`**: expected is deterministic expert choice
   (first-index tie-break per 6.0b probe) and a router whose weights still normalise. (Task 9)
6. **Sharded checkpoints where the index names a tensor missing from every shard, or a tensor present
   in a shard but absent from the index**: expected is a named error before layer construction. (Task 2)
7. **Tokenizer/model vocabulary mismatch for a new family** (Gemma's large vocab, Phi's padded head):
   expected is the existing 6.2 bounds check, plus a per-family tokenizer golden. (Task 10b)

8. **Checkpoints that load today but violate the new tensor plan** (`attention_bias=false` with a
   `q_proj.bias`; Qwen2 with an `o_proj.bias`): today's loader uses such a bias; the plan rejects it by
   name. Expected is the named rejection, pinned by a regression test. (Task 2)
9. **Freshly saved configs carrying harmless keys** (`layer_types` all `full_attention`, `torch_dtype`,
   `rope_parameters` with `rope_theta` inside): expected is a successful load, not a rejection. (Task 1)
10. **Dynamic-NTK across a prefill/decode boundary**: cached keys keep their old rotation, so output
    depends on how the prompt was chunked; expected is a documented, tested behavior. (Task 3)

---

## Baseline

- `DecoderModel` builds Llama and Qwen2 from `DecoderConfig` (12 fields) and two constructor booleans
  (`qkvBiasRequired`, `outBiasRequired`). Tensor names, `GroupedQueryAttention`, `SwiGLU`, `RMSNorm`
  and the RoPE base are hard-wired; `head_dim` is derived as `hidden/heads`.
- `DecoderConfig.fromFile` rejects `rope_scaling`, `quantization(_config)`, `use_sliding_window` and any
  `hidden_act` other than silu/swish. `TextGenerationModels` dispatches on `model_type` (`llama`, `qwen2`).
- `MLXFast.rope(x, dims, traditional, base, scale, offset, freqs)` already exposes `freqs` and a
  partial `dims`; `mlx_fast_rope_dynamic` is `unplanned` in the inventory. `MLXFast.scaledDotProductAttention`
  accepts `causal` **or** an explicit BOOL/additive `maskArr`, not both.
- `KVCache` is append-only (no capacity, no eviction); 6.4 owns eviction. `Linear` and
  `QuantizedLinear` are both `UnaryModule`. `MLXIO.loadGguf` exists; nothing in `jmlx-models` uses it.
- `CheckpointLoader` reads shards and the index but does not cross-check index entries against shard
  contents, and `DecoderModel` reports one missing tensor at a time while building layers.
- **Behavior changes this milestone makes to checkpoints/configs that load today** (each has a
  regression test): `DecoderModel.bias()` returns any present bias regardless of config, so a Llama
  checkpoint with `attention_bias=false` plus a `q_proj.bias`, or a Qwen2 checkpoint with an
  `o_proj.bias`, currently loads *and applies* the bias; Task 2 rejects both by name.
  `DecoderConfig.fromFile` currently accepts any `model_type`; after Task 1 an unregistered type throws.
  A stray `*.rotary_emb.inv_freq` tensor is now an explicit allow-listed ignore.
- The Node/`@huggingface/jinja` oracle in `jmlx-jinja` (`nodeCorpusVerify`, `tools/corpus/`) already
  renders real model templates (`mistral-7b-instruct-v0.3.jinja`, Qwen, Step) from
  `jmlx-jinja/src/test/resources/model-templates`; the tokenizer oracle (`tools/tokenizer-oracle`) has
  no chat/Jinja support and pins only `tokenizers==0.23.2`.

## Scope decisions

These fix what "Mistral, Gemma, Phi and one MoE family" means so the milestone is finishable.

| Family | `model_type` in 6.3 | Capabilities added | Explicitly deferred |
| --- | --- | --- | --- |
| Llama / Qwen2 | `llama`, `qwen2` | descriptor form of today's behavior; RoPE scaling (linear, dynamic-NTK, llama3, YaRN) | Qwen2 `use_sliding_window` / `max_window_layers` stays rejected |
| Mistral | `mistral` | sliding-window attention on every layer | per-layer window schedules |
| Gemma | `gemma` (v1) | `(1 + weight)` RMSNorm, sqrt(hidden) embedding scale, explicit `head_dim`, `gelu_pytorch_tanh` GeGLU, tied head | `gemma2`/`gemma3` (softcapping, extra norms, alternating windows) |
| Phi | `phi3` | fused `qkv_proj` and `gate_up_proj` checkpoint mapping, optional sliding window | `phi` (Phi-2: LayerNorm, parallel block), `longrope` scaling |
| MoE | `mixtral` | router softmax/top-k/renormalise, per-expert SwiGLU, MoE MLP | expert-parallel or gather-based kernels (6.4), shared experts |

Other decisions:

1. **Descriptors are pure Java.** `ArchitectureDescriptor` and its parts are immutable records/enums
   with no native handles, so parsing and tensor validation run on Ubuntu CI.
2. **`DecoderConfig` stays public and source-compatible.** It remains the common-dimension record
   returned by `DecoderModel.config()`; `fromFile` delegates to the descriptor mapping so its existing
   rejections and messages are preserved (`DecoderConfigTest` is the guard).
3. **Projections are `UnaryLayer`, not `Linear`, and stay in the module tree.** `Module.child` is
   `<M extends Module> M child(String, M)` and `UnaryModule` is a plain interface, so a bare
   `UnaryModule` parameter could not be registered (its parameters would vanish from `parameters()`,
   `rebind` and `ModuleGrad`). Add `public abstract class UnaryLayer extends Module implements
   UnaryModule` in `jmlx-core`; `Linear` and `QuantizedLinear` change from `extends Module implements
   UnaryModule` to `extends UnaryLayer` (source-compatible). Likewise `public abstract class
   CachedAttention extends Module { public abstract MLXArray forward(MLXArray x, KVCache cache); }`
   is the shared attention abstraction; `GroupedQueryAttention` and `DecoderAttention` extend it, and
   `DecoderBlock` holds `CachedAttention` and `UnaryLayer mlp`. Child names stay `inputNorm`,
   `attention`, `postAttentionNorm`, `mlp`, so existing parameter paths do not change (pinned by a
   test that asserts the exact `parameters()` key set of a tiny Llama before and after).
4. **MoE computes every expert densely** and selects with `where(selected, expertOutput, 0)` (never a
   multiply: `inf * 0` is NaN, plausible for a bf16 expert on tokens it was not routed).
   This is correct and simple; the cost (E/top_k times the MLP FLOPs) is a documented 6.4 optimization
   target, not a 6.3 defect. Only Tier-A-sized MoE is asserted for speed-insensitive tests.
5. **No cache eviction in 6.3.** Sliding-window layers mask by absolute position over the full cache;
   memory stays unbounded per generation until 6.4, and the compatibility matrix says so.
6. **Config field policy (replaces the old deny-list).** Each family mapping owns two tables in
   `ArchitectureMappings`: `consumed` (keys it reads) and `ignored` (key → one-line reason, for keys
   with no effect on inference: `architectures`, `torch_dtype`, `dtype`, `transformers_version`,
   `bos/eos/pad_token_id`, `initializer_range`, `attention_dropout`, `*_pdrop`, `use_cache`,
   `max_position_embeddings` (recorded, not enforced), `pretraining_tp`, `router_aux_loss_coef`,
   `router_jitter_noise`, `output_router_logits` only when `false`, `sliding_window` only when the
   family does not read it and the value is `null`). A key in neither table is **not** rejected (that
   would break every new transformers release) but is `System.Logger` WARNING-logged once per load with
   the key name; a key that *changes numerics* is handled by an explicit named rejection rule instead
   (`quantization(_config)`, unknown `rope_type`, `layer_types` containing anything but
   `full_attention` for a family without schedules, `use_sliding_window=true` on Qwen2, `gemma2`-only
   fields such as `attn_logit_softcapping`, `final_logit_softcapping`, `query_pre_attn_scalar`).
   `layer_types` that is uniformly `full_attention` (freshly saved Qwen2/Llama configs) is accepted.
   `rope_scaling` sub-keys follow the same rule: each variant consumes its documented keys and rejects
   an unknown sub-key that could change frequencies by name.
7. **RoPE parameters.** Read `rope_theta` from the top level or from `rope_parameters.rope_theta`
   (transformers ≥ 5 layout; top-level wins only if they agree, otherwise reject naming both), and
   `rope_scaling` or `rope_parameters` for the variant. YaRN's `attention_factor` (overrides the
   computed mscale) and `truncate` (default `true`) are parsed; `mscale`/`mscale_all_dim` are parsed
   for the DeepSeek-style form and used only in the mscale ratio HF uses.
8. **Quantization and GGUF are probe-gated (Task 10).** The matrix's "float weights only" stays true
   unless the probe proves a variant loads into `QuantizedLinear` without a new native surface.

## File structure

Created in `jmlx-models/src/main/java/se/alipsa/jmlx/models/`:

- `ArchitectureDescriptor.java` — immutable capability record (attention, norm, mlp, rope, head, moe).
- `ArchitectureMappings.java` — `model_type` → descriptor parsing; owns every `config.json` rejection.
- `TensorPlan.java` — required/optional/forbidden/ignored tensor keys; `validate(Set<String>)`.
- `DecoderAssembler.java` — builds `jmlx-core` components from a validated descriptor and tensors.

Created in `jmlx-core/src/main/java/se/alipsa/jmlx/`:

- `nn/RopeSpec.java` — rope variants and inverse-frequency construction (pure Java math).
- `nn/UnaryLayer.java`, `nn/CachedAttention.java` — module-tree-safe abstractions (Decision 3).
- `nn/DecoderAttention.java`, `nn/AttentionMask.java` — explicit `head_dim`, `RopeSpec`, causal or
  sliding-window masking.
- `nn/GatedMlp.java`, `nn/MoeMlp.java` — gated MLP with selectable activation; routed experts.
- `nn/RMSNorm.java` gains an optional weight offset (Gemma); `nn/DecoderBlock.java` gains a
  component-based constructor.

Created in `jmlx-models/src/test/java/se/alipsa/jmlx/models/`: `TestDescriptors.java` (descriptor
builders), `TinyCheckpoints.java` (seeded non-zero checkpoint writer, Task 5). Created under `tools/`:
`hf-reference/` (Task 0). Modified: `Linear`, `QuantizedLinear` (extend `UnaryLayer`), `DecoderConfig`, `DecoderModel`, `TextGenerationModels`, `CheckpointLoader`, `LlamaModel`,
`QwenModel`, `MLXOps` (logical-and), `req/mlx-api-inventory-overrides.json`, `req/phase6-compatibility.md`,
`req/phase6-tier-a-fixtures.md`, `req/phase6-tier-b-artifacts.md`, `jmlx-models/README.md`,
`build.gradle` (`verifyHfReferenceGoldens`, `tierB` Spotless/Checkstyle wiring), `jmlx-models/build.gradle`
(`tierB` source set), `.github/workflows/tier-b.yml`.

---

### Task 0: Offline Hugging Face reference tool and committed goldens

Task 6, 8 and 9 need reference values that a Java transcription error cannot agree with. A second
implementation written by the same author from the same HF source in the same task can share one
misreading, so the goldens come from Hugging Face's own code, run once offline and committed.

**Files:**
- Create: `tools/hf-reference/README.md`, `requirements.in`, `requirements.lock` (hash-locked CPU
  `torch` + `transformers` + `safetensors`, Python 3.12), `generate.py`, `provenance.json`
- Create: `tools/hf-reference/goldens/{rope,mistral,phi3,gemma,mixtral}.json` and the tiny
  `*.safetensors` checkpoints they were computed from (`goldens/checkpoints/<family>/`)
- Modify: `build.gradle` — `verifyHfReferenceGoldens` (pure Gradle/Java: recompute SHA-256 of every file
  under `goldens/` and compare with `provenance.json`); wired into root `check`. There is deliberately
  **no** Gradle task that runs `generate.py`, and CI never installs `torch`.

**Interfaces:**
- Produces: `goldens/rope.json` — for each `rope_type` and offset, the HF-computed `inv_freq` and
  `attention_scaling` from `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS`, plus the rotated
  output of HF's own rotary application on a fixed input; `goldens/<family>.json` — prompt IDs, full
  prefill logits, and two greedy decode steps from `transformers`' `AutoModelForCausalLM` (CPU, float32,
  `attn_implementation="eager"`) over a committed tiny random checkpoint (weights seeded with
  `torch.manual_seed`, saved with `save_pretrained`, so Java reads exactly these safetensors).
- Consumes: nothing from Java.

- [ ] **Step 1:** Write `generate.py` taking `--family` and `--out`; it builds a tiny config per family
  (hidden ≥ 64 so the same checkpoints serve the quantization probe, GQA `kv < heads`, 2 layers), saves
  the model, runs prefill and greedy decode, and writes JSON with `float` values rounded to 7 digits.
- [ ] **Step 2:** `python -m venv` + `pip install --require-hashes -r requirements.lock`; run once per
  family; record Python, `torch`, `transformers` and `safetensors` versions, the commit of
  `modeling_rope_utils.py` semantics used, host OS/arch, and file hashes in `provenance.json`.
- [ ] **Step 3:** Add `verifyHfReferenceGoldens` and a Java test `HfReferenceProvenanceTest` (pure Java,
  runs on Ubuntu) that fails on any hash drift. Document in the README that regenerating is a reviewed,
  manual step and that a repin changes the goldens' expected diff.
- [ ] **Step 4:** Run `./gradlew verifyHfReferenceGoldens :check`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add tools/hf-reference build.gradle
git commit -m "Add offline Hugging Face reference tool and committed tiny-model goldens"
```

If a family's tiny model cannot be produced by `transformers` at the pinned version (for example a
config field removed upstream), stop and record it in the plan's probe notes; the family's numeric claim
then falls back to `verified-with-synthetic-fixture` without an independent reference and is labelled so.

---

### Task 1: Architecture descriptor and mappings for Llama and Qwen2

**Files:**
- Create: `jmlx-models/src/main/java/se/alipsa/jmlx/models/ArchitectureDescriptor.java`
- Create: `jmlx-models/src/main/java/se/alipsa/jmlx/models/ArchitectureMappings.java`
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/RopeSpec.java` (sealed interface with `Base` only; Task 3 completes it)
- Modify: `jmlx-models/src/main/java/se/alipsa/jmlx/models/DecoderConfig.java`
- Test: `jmlx-models/src/test/java/se/alipsa/jmlx/models/ArchitectureMappingsTest.java`

**Interfaces:**
- Produces:
  - `record ArchitectureDescriptor(DecoderConfig dimensions, int headDim, Norm norm, Mlp mlp, Attention attention, Head head, Embedding embedding, Moe moe)` with nested records/enums
    `Norm(NormKind kind, float eps, boolean weightOffset)`, `Mlp(MlpLayout layout, Activation activation, boolean bias)`,
    `Attention(boolean qkvBias, boolean outBias, boolean fusedQkv, Integer slidingWindow)`, `Head(boolean tied)`,
    `Embedding(boolean scaleBySqrtHidden)`, `Moe(int experts, int topK)` (null when dense).
    `MlpLayout` is `SEPARATE_GATE_UP`, `FUSED_GATE_UP`; `Activation` is `SILU`, `GELU_TANH`.
  - `static ArchitectureDescriptor ArchitectureMappings.parse(JsonNode config)` — throws
    `IllegalArgumentException` naming the key for anything unsupported.
  - `static Set<String> ArchitectureMappings.supportedModelTypes()`.
  - `ArchitectureDescriptor.modelType()` is a convenience accessor for `dimensions().modelType()` (no
    duplicated component).
  - Per-family `consumed`/`ignored` key tables implementing the "Config field policy" (Scope decisions
    6–7), and a `List<String> ArchitectureMappings.lastWarnings()`-free design: unknown keys are logged
    through `System.Logger` (`ArchitectureMappings` logger), and `parse` takes an optional
    `Consumer<String> unknownKeySink` overload so tests can assert on them without log capture.
- Consumes: `DecoderConfig`'s canonical constructor (unchanged).

- [ ] **Step 1: Write the failing tests**

```java
class ArchitectureMappingsTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private static JsonNode json(String text) {
    return MAPPER.readTree(text);
  }

  @Test
  void llamaMapsToTodaysBehavior() {
    ArchitectureDescriptor d = ArchitectureMappings.parse(json("""
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,
         "tie_word_embeddings":true}"""));
    assertEquals(2, d.headDim());
    assertEquals(NormKind.RMS, d.norm().kind());
    assertFalse(d.attention().qkvBias());
    assertTrue(d.head().tied());
    assertNull(d.attention().slidingWindow());
    assertNull(d.moe());
  }

  @Test
  void qwen2HardcodesQkvBiasAndNoOutBias() {
    ArchitectureDescriptor d = ArchitectureMappings.parse(json("""
        {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
         "attention_bias":false}"""));
    assertTrue(d.attention().qkvBias());
    assertFalse(d.attention().outBias());
  }

  @Test
  void freshlySavedConfigWithHarmlessKeysLoads() {
    // Review Focus 9: transformers >= 5 writes layer_types and rope_parameters.
    List<String> unknown = new ArrayList<>();
    ArchitectureDescriptor d = ArchitectureMappings.parse(json("""
        {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
         "layer_types":["full_attention","full_attention"],"torch_dtype":"bfloat16",
         "transformers_version":"5.0.0","use_cache":true,"bos_token_id":1,"eos_token_id":2,
         "rope_parameters":{"rope_type":"default","rope_theta":1000000.0},
         "brand_new_key":7}"""), unknown::add);
    assertEquals(1_000_000f, ((RopeSpec.Base) d.rope()).theta());
    assertEquals(List.of("brand_new_key"), unknown);
  }

  @Test
  void nonUniformLayerTypesAreRejectedByName() {
    var e = assertThrows(IllegalArgumentException.class, () -> ArchitectureMappings.parse(json("""
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":2,"num_attention_heads":2,
         "layer_types":["full_attention","sliding_attention"]}""")));
    assertTrue(e.getMessage().contains("layer_types"));
  }

  @Test
  void conflictingRopeThetaLocationsAreRejected() {
    assertThrows(IllegalArgumentException.class, () -> ArchitectureMappings.parse(json("""
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"rope_theta":10000,
         "rope_parameters":{"rope_type":"default","rope_theta":500000}}""")));
  }

  @Test
  void unknownModelTypeNamesTheKey() {
    var e = assertThrows(IllegalArgumentException.class,
        () -> ArchitectureMappings.parse(json("{\"model_type\":\"falcon\"}")));
    assertTrue(e.getMessage().contains("model_type") && e.getMessage().contains("falcon"));
  }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `./gradlew :jmlx-models:test --tests "*ArchitectureMappingsTest"`
Expected: FAIL — `ArchitectureMappings` does not exist.

- [ ] **Step 3: Implement**

Create the descriptor records exactly as listed under Interfaces, with compact-constructor validation
(positive `headDim`, `slidingWindow > 0` when present, `experts >= topK >= 1`). In
`ArchitectureMappings.parse`, read `model_type` and dispatch through a `Map<String, Function<JsonNode,
ArchitectureDescriptor>>`; `llama` and `qwen2` reuse the checks that today live in
`DecoderConfig.fromFile` (moved, not copied: `quantization(_config)`, `use_sliding_window`,
`hidden_act` in {silu, swish}, `head_dim * heads == hidden`). Keep `rope_scaling` rejected here with the
existing message until Task 3 replaces it. Change `DecoderConfig.fromFile` to read the tree, call
`ArchitectureMappings.parse`, and return `descriptor.dimensions()`. **Behavior change:** `fromFile`
previously accepted any `model_type` (and only `TextGenerationModels` rejected unknown ones); it now
throws for an unregistered type. Note it in the class javadoc and the README.
The mapping cannot compile until `rope()` exists, so this task adds a `Rope rope` component holding
`RopeSpec.Base(theta)` only (parsing `rope_theta` from the top level or `rope_parameters`); Task 3
extends the parsing to scaling variants. `RopeSpec` (a tiny sealed interface with `Base`) is therefore
created here in `jmlx-core` and completed in Task 3.

- [ ] **Step 4: Run the new and existing config tests**

Run: `./gradlew :jmlx-models:test --tests "*ArchitectureMappingsTest" --tests "*DecoderConfigTest"`
Expected: PASS (every existing `DecoderConfigTest` assertion and message unchanged).

- [ ] **Step 5: Commit**

```bash
git add jmlx-models/src/main/java/se/alipsa/jmlx/models/ArchitectureDescriptor.java \
  jmlx-models/src/main/java/se/alipsa/jmlx/models/ArchitectureMappings.java \
  jmlx-models/src/main/java/se/alipsa/jmlx/models/DecoderConfig.java \
  jmlx-models/src/test/java/se/alipsa/jmlx/models/ArchitectureMappingsTest.java
git commit -m "Add architecture descriptor and model_type mappings for Llama and Qwen2"
```

### Task 2: Tensor plans and checkpoint validation before layer construction

**Files:**
- Create: `jmlx-models/src/main/java/se/alipsa/jmlx/models/TensorPlan.java`
- Create: `jmlx-models/src/main/java/se/alipsa/jmlx/models/SafetensorsHeaders.java` (pure Java)
- Create (test): `jmlx-models/src/test/java/se/alipsa/jmlx/models/TestDescriptors.java` — static builders
  `llama(int layers, boolean attentionBias)`, `qwen2(int layers)`, etc. returning descriptors without JSON
- Modify: `jmlx-models/src/main/java/se/alipsa/jmlx/models/CheckpointLoader.java` (header pre-check)
- Modify: `jmlx-models/src/main/java/se/alipsa/jmlx/models/ArchitectureMappings.java` (`tensorPlan(descriptor)`)
- Test: `jmlx-models/src/test/java/se/alipsa/jmlx/models/TensorPlanTest.java`,
  `jmlx-models/src/test/java/se/alipsa/jmlx/models/CheckpointIndexTest.java`

**Interfaces:**
- Produces:
  - `record TensorPlan(Set<String> required, Set<String> optional, Set<String> forbidden, Set<Pattern> ignored)`.
  - `void TensorPlan.validate(Set<String> available)` — collects **every** problem and throws one
    `IllegalArgumentException` listing them all: `missing tensor 'K'`, `unexpected tensor 'K'`,
    `forbidden tensor 'K' (capability: <text>)`. Keys matching `ignored` are skipped.
  - `static TensorPlan ArchitectureMappings.tensorPlan(ArchitectureDescriptor d)` — built from the
    descriptor (per-layer keys expanded for `0..numHiddenLayers-1`).
  - `static Map<Path, Set<String>> SafetensorsHeaders.tensorNames(List<Path> files)` — reads only the
    8-byte little-endian header length and the JSON header of each file (no tensor data, no native
    code), rejecting a header length above 100 MB or beyond the file size.
  - `CheckpointLoader.load` runs a **pre-load** check using those names, *before* any
    `MLXIO.loadSafetensors` call: it throws when an index `weight_map` key is absent from every shard,
    when a shard tensor is absent from the index, or when the index maps a key to a different shard
    than the one containing it. In the no-index path it ignores `consolidated.safetensors`
    (Mistral repos ship it beside sharded weights) only when a `model*.safetensors` set is also
    present, and names the ignored file in a `System.Logger` message; if it is the only file it loads.
- Consumes: `ArchitectureDescriptor` (Task 1).

- [ ] **Step 1: Write the failing tests (pure Java, no native)**

```java
class TensorPlanTest {
  private static TensorPlan llamaPlan(boolean bias) {
    return ArchitectureMappings.tensorPlan(TestDescriptors.llama(1, bias));
  }

  @Test
  void reportsEveryProblemNotJustTheFirst() {
    Set<String> have = new HashSet<>(llamaPlan(false).required());
    have.remove("model.norm.weight");
    have.add("model.layers.0.self_attn.q_proj.bias");
    var e = assertThrows(IllegalArgumentException.class, () -> llamaPlan(false).validate(have));
    assertTrue(e.getMessage().contains("missing tensor 'model.norm.weight'"));
    assertTrue(e.getMessage().contains("forbidden tensor 'model.layers.0.self_attn.q_proj.bias'"));
  }

  @Test
  void ignoresLegacyRotaryInvFreq() {
    Set<String> have = new HashSet<>(llamaPlan(false).required());
    have.add("model.layers.0.self_attn.rotary_emb.inv_freq");
    llamaPlan(false).validate(have);
  }

  @Test
  void rejectsAnUnknownExtraTensor() {
    Set<String> have = new HashSet<>(llamaPlan(false).required());
    have.add("model.layers.0.mystery.weight");
    var e = assertThrows(IllegalArgumentException.class, () -> llamaPlan(false).validate(have));
    assertTrue(e.getMessage().contains("unexpected tensor 'model.layers.0.mystery.weight'"));
  }

  @Test
  void explicitLmHeadIsAllowedWhenTied() {
    // Existing behavior: an explicit lm_head.weight wins even when tie_word_embeddings=true.
    Set<String> have = new HashSet<>(llamaPlan(false).required());
    have.add("lm_head.weight");
    llamaPlan(false).validate(have);
  }

  @Test
  void untiedModelWithoutLmHeadFails() {
    TensorPlan untied = ArchitectureMappings.tensorPlan(TestDescriptors.llama(1, false, /*tied*/ false));
    Set<String> have = new HashSet<>(untied.required());
    have.remove("lm_head.weight");
    var e = assertThrows(IllegalArgumentException.class, () -> untied.validate(have));
    assertTrue(e.getMessage().contains("missing tensor 'lm_head.weight'"));
  }

  @Test
  void biasThatLoadsTodayIsNowRejectedByName() {
    // Review Focus 8 / Baseline: previously DecoderModel.bias() applied any bias present.
    Set<String> llama = new HashSet<>(llamaPlan(false).required());
    llama.add("model.layers.0.self_attn.q_proj.bias");
    assertThrows(IllegalArgumentException.class, () -> llamaPlan(false).validate(llama));
    TensorPlan qwen = ArchitectureMappings.tensorPlan(TestDescriptors.qwen2(1));
    Set<String> q = new HashSet<>(qwen.required());
    q.add("model.layers.0.self_attn.o_proj.bias");
    var e = assertThrows(IllegalArgumentException.class, () -> qwen.validate(q));
    assertTrue(e.getMessage().contains("o_proj.bias"));
  }
}
```

`CheckpointIndexTest` is **pure Java** (runs on Ubuntu CI): it writes minimal safetensors files by hand
(8-byte length + JSON header, zero-length data section is enough for header parsing) and an index that
(a) names a tensor in no shard, (b) omits a shard tensor, (c) maps a key to the wrong shard; each must fail
with a message naming the tensor key **without any native library loaded** (assert by running the test
class in a JVM where `NativeLoader` is never touched, i.e. no `@EnabledIfNativeAvailable`). A fourth case
places `consolidated.safetensors` beside `model-00001-of-00002.safetensors` files with no index and expects
the consolidated file to be ignored; a fifth has it alone and expects it to load its header.

- [ ] **Step 2: Run to verify failure**

Run: `./gradlew :jmlx-models:test --tests "*TensorPlanTest" --tests "*CheckpointIndexTest"`
Expected: FAIL — `TensorPlan` missing.

- [ ] **Step 3: Implement**

`TensorPlan.validate` is a pure set comparison: `missing = required − available`;
`forbidden = available ∩ forbidden`; `unexpected = available − required − optional − forbidden −
ignored`. `ignored` defaults to `.*\.rotary_emb\.inv_freq$` (non-persistent buffer written by older
exports; HF ignores it too). `tensorPlan` puts the biases the descriptor does not call for into
`forbidden` with capability text (for example `attention_bias=false`), the tied-head `lm_head.weight`
into `optional`, and Qwen2's `o_proj.bias` into `forbidden`. `CheckpointLoader.load` compares the index
`weight_map` keys with the union of header tensor names *before loading any shard* and throws on either
difference (fail before allocating native memory).
Do not call `plan.validate` from the loader; `DecoderAssembler` (Task 5) calls it once with
`tensors.keySet()` before constructing anything.

- [ ] **Step 4: Run tests**

Run: `./gradlew :jmlx-models:test --tests "*TensorPlanTest" --tests "*CheckpointIndexTest" --tests "*LlamaModelTest" --tests "*QwenModelTest"`
Expected: PASS. Note any existing native test whose checkpoint carries a tensor the plan now calls
unexpected; fix the plan (not the test) only if HF's own modeling code loads that tensor.

- [ ] **Step 5: Commit**

```bash
git add jmlx-models/src/main/java/se/alipsa/jmlx/models/TensorPlan.java \
  jmlx-models/src/main/java/se/alipsa/jmlx/models/CheckpointLoader.java \
  jmlx-models/src/main/java/se/alipsa/jmlx/models/ArchitectureMappings.java \
  jmlx-models/src/test/java/se/alipsa/jmlx/models/TensorPlanTest.java \
  jmlx-models/src/test/java/se/alipsa/jmlx/models/CheckpointIndexTest.java
git commit -m "Validate checkpoint tensors against a per-architecture plan"
```

---

### Task 3: RoPE scaling — linear, dynamic-NTK, Llama 3, YaRN

**Files:**
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/RopeSpec.java` (created in Task 1; add the scaling variants)
- Modify: `jmlx-models/src/main/java/se/alipsa/jmlx/models/ArchitectureDescriptor.java` (`Rope rope`)
- Modify: `jmlx-models/src/main/java/se/alipsa/jmlx/models/ArchitectureMappings.java` (parse `rope_scaling`)
- Modify: `req/mlx-api-inventory-overrides.json` only if the probe below selects `mlx_fast_rope_dynamic`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/RopeSpecTest.java` (pure Java math),
  `jmlx-core/src/test/java/se/alipsa/jmlx/core/RopeFreqsProbeTest.java` (native),
  `jmlx-models/src/test/java/se/alipsa/jmlx/models/RopeScalingConfigTest.java`

**Interfaces:**
- Produces (`se.alipsa.jmlx.nn`, public):
  - `sealed interface RopeSpec` with records `Base(float theta)`, `Linear(float theta, float factor)`,
    `DynamicNtk(float theta, float factor, int maxPositions)`, `Llama3(float theta, float factor,
    float lowFreqFactor, float highFreqFactor, int originalMaxPositions)`, `Yarn(float theta, float
    factor, int originalMaxPositions, float betaFast, float betaSlow, float mscale, float
    mscaleAllDim)`.
  - `Yarn` additionally carries `Float attentionFactor` (overrides the computed mscale when present)
    and `boolean truncate` (default `true`), both parsed from `rope_scaling`.
  - `double[] RopeSpec.frequencies(int rotaryDims, int sequenceLength)` — the *period* array MLX's
    `freqs` argument expects (`base^(2i/dims)` scaled per variant), length `rotaryDims/2`. Only
    `DynamicNtk` reads `sequenceLength`.
  - `boolean RopeSpec.isStatic()` — `true` for `Base`, `Linear`, `Llama3`, `Yarn`; their `freqs` array
    is built **once** in the model scope (by `DecoderAssembler`, shared by every layer) and never per
    layer per step. Only `DynamicNtk` (`false`) rebuilds per call, in the step scope. `Linear` needs no
    `freqs` at all: it calls `rope(base=theta, scale=1/factor)`.
  - `float RopeSpec.scale()` (the `scale` argument; `1/factor` for `Linear`, else `1`) and
    `float RopeSpec.attentionScaling()` (HF's `attention_scaling`: YaRN's factor, else `1`).
  - `static MLXArray RopeSpec.apply(MLXArray x, int rotaryDims, int offset, MLXArray staticFreqs)` —
    the **single production code path** that applies RoPE to `[.., T, headDim]`. It rotates only
    `x[..., :rotaryDims]`, multiplies **only that rotated slice** by `attentionScaling()` (HF scales
    cos/sin, which only touches rotated dimensions; mlx-lm's `YarnRoPE` likewise does
    `x[..., :dims] *= mscale`), and concatenates the untouched pass-through dims. `DecoderAttention`
    and every test call this method; no test re-implements the call.
- Consumes: `ArchitectureDescriptor` (Task 1); `MLXFast.rope` unchanged.

- [ ] **Step 1: Probe the native contract first (record findings in the test's javadoc)**

`RopeFreqsProbeTest` (`@EnabledIfNativeAvailable`) answers, against pinned MLX: (a) does
`MLXFast.rope` with `base=null` and a `freqs` array reproduce `base=theta` when
`freqs = theta^(2i/dims)`; (b) is `freqs` a period (divisor) or an inverse frequency; (c) does
`scale != 1` compose with `freqs` as `offset*scale/freqs`. Only if a dynamic-NTK or offset case cannot
be expressed by recomputing `freqs` per call, add `mlx_fast_rope_dynamic` as a `MLXFast.ropeDynamic`
facade with an inventory entry; otherwise leave it `unplanned` and say so in the test.

- [ ] **Step 2: Write the failing math tests (pure Java)**

```java
@Test void baseMatchesThetaPower() {
  double[] f = new RopeSpec.Base(10000f).frequencies(4, 1);
  assertArrayEquals(new double[] {1.0, 100.0}, f, 1e-9);
}

@Test void linearDividesPositionsByFactorViaScale() {
  var s = new RopeSpec.Linear(10000f, 4f);
  assertEquals(0.25f, s.scale(), 0f);
  assertArrayEquals(new RopeSpec.Base(10000f).frequencies(4, 1), s.frequencies(4, 1), 1e-9);
}

@Test void dynamicNtkIsIdentityUntilContextExceedsMaxPositions() {
  var s = new RopeSpec.DynamicNtk(10000f, 2f, 8);
  double[] base = new RopeSpec.Base(10000f).frequencies(4, 1);
  assertArrayEquals(base, s.frequencies(4, 8), 1e-9);
  // HF: theta' = theta * ((factor*L/max) - (factor-1))^(dim/(dim-2)); L=16, max=8, factor=2,
  // dim=4 -> theta' = 10000 * 3^2 = 90000, and period[i] = theta'^(2i/dim).
  double[] scaled = s.frequencies(4, 16);
  assertEquals(1.0, scaled[0], 1e-9);
  assertEquals(Math.pow(90000.0, 0.5), scaled[1], 1e-6);
}

@Test void llama3KeepsHighFrequenciesAndDividesLowOnes() { /* compare with HF _compute_llama3_parameters
  values computed offline in the oracle fixture (Task 6); here assert the three regimes on a 64-dim head */ }

@Test void yarnReportsAttentionFactor() {
  assertEquals((float) (0.1 * Math.log(4.0) + 1.0),
      new RopeSpec.Yarn(10000f, 4f, 4096, 32f, 1f, 0f, 0f, null, true).attentionScaling(), 1e-6f);
  assertEquals(1.5f,
      new RopeSpec.Yarn(10000f, 4f, 4096, 32f, 1f, 0f, 0f, 1.5f, true).attentionScaling(), 0f);
}

@Test void yarnScalingLeavesPassThroughDimensionsUntouched() {   // native
  // headDim 8, rotaryDims 4: x[..., 4:] must equal the input exactly after apply().
}

@Test void staticSpecsAreStaticAndDynamicIsNot() {
  assertTrue(new RopeSpec.Llama3(10000f, 8f, 1f, 4f, 8).isStatic());
  assertFalse(new RopeSpec.DynamicNtk(10000f, 2f, 8).isStatic());
}
```

Fill the `dynamicNtk` and `llama3` expectations from the formulas in the Hugging Face
`modeling_rope_utils.py` reference (cite the file and revision in the test); Task 6 then checks the same
variants against the committed Hugging Face goldens from Task 0, which are the authority. If a formula cannot be reproduced to 1e-6 the task stops and the
discrepancy goes into the plan's probe notes.

- [ ] **Step 3: Run to verify failure**

Run: `./gradlew :jmlx-core:test --tests "*RopeSpecTest"`
Expected: FAIL — `RopeSpec` missing.

- [ ] **Step 4: Implement `RopeSpec` and the config mapping**

`frequencies` returns the period array `theta^(2i/dims)` for `Base`/`Linear`; `DynamicNtk` recomputes
`theta'` from `sequenceLength` when it exceeds `maxPositions`; `Llama3` and `Yarn` port HF's
`_compute_llama3_parameters` and `_compute_yarn_parameters` (Yarn: `find_correction_range`, linear
ramp mask, `mscale` from `get_mscale`). `attentionScaling` is the YaRN attention factor.
`RopeSpec.apply` multiplies the rotated slice of both `q` and `k` by it (equivalent to HF scaling cos/sin
because the logits see the factor on both sides), and **never** the pass-through dims when
`rotaryDims < headDim` — the rejected alternative of scaling all of `q`/`k` was checked against this
requirement and is wrong for `partial_rotary_factor < 1`. If the probe shows `MLXFast.rope` cannot rotate
a strict prefix of the head dimension with `dims < x.shape[-1]`, `apply` slices, rotates, scales and
concatenates explicitly.

`ArchitectureMappings` parses `rope_scaling` (or the newer `rope_parameters`, whose `rope_theta` is read
per Scope decision 7): `rope_type` (fall back to
`type`) in {`default`, `linear`, `dynamic`, `llama3`, `yarn`}; every other value, and `longrope`, throws
`config.json rope_scaling.rope_type 'X' is not supported`. Require the variant's fields
(`factor`, and `low_freq_factor`/`high_freq_factor`/`original_max_position_embeddings` for llama3;
`original_max_position_embeddings` for yarn/dynamic falling back to `max_position_embeddings`) and reject
a missing or non-numeric one by name; an unknown sub-key that could change frequencies is rejected by
name (Scope decision 6). Also parse `partial_rotary_factor` (default 1.0) into
`rotaryDims = (int)(headDim * factor)`, which must be even and `<= headDim`.
`RopeScalingConfigTest` covers each accepted type, each rejection, and the missing-field messages.

- [ ] **Step 5: Run tests**

Add `RopeFreqsProbeTest.dynamicNtkChunkingDependence` (Review Focus 10): with `DynamicNtk(max=8)`, prefill
6 tokens then decode 6 one at a time (crossing 8) yields a *different* final-position rotation than
prefilling all 12 at once, because cached keys keep the rotation they were written with (this matches
HF's cache behavior). The test asserts the difference is non-zero and that the same chunking twice is
identical, and its javadoc records "output depends on prefill chunking for dynamic NTK". The limitation
goes into the README's RoPE section (Task 10d).

Run: `./gradlew :jmlx-core:test --tests "*RopeSpecTest" --tests "*RopeFreqsProbeTest" && ./gradlew :jmlx-models:test --tests "*RopeScalingConfigTest" --tests "*DecoderConfigTest"`
Expected: PASS. Update `DecoderConfigTest`'s old "rejects rope_scaling" case to assert an *unsupported*
type is still rejected (`longrope`), and add the accepted-`llama3` case.

- [ ] **Step 6: Commit**

```bash
git add jmlx-core/src/main/java/se/alipsa/jmlx/nn/RopeSpec.java jmlx-core/src/test \
  jmlx-models/src/main/java/se/alipsa/jmlx/models jmlx-models/src/test
git commit -m "Add RoPE scaling specs and rope_scaling config mapping"
```

---

### Task 4: Configurable attention — explicit `head_dim`, `RopeSpec`, sliding-window mask

**Files:**
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/DecoderAttention.java`
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/AttentionMask.java`
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/core/MLXOps.java` (`logicalAnd`), `req/mlx-api-inventory-overrides.json`
  (`mlx_logical_and` → implemented), then `./gradlew generateMlxApiInventory`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/DecoderAttentionTest.java`,
  `jmlx-core/src/test/java/se/alipsa/jmlx/nn/AttentionMaskTest.java`

**Interfaces:**
- Consumes: `RopeSpec` (Task 3), `KVCache`, `Linear`/`QuantizedLinear` via `UnaryModule`.
- Produces:
  - `MLXOps.logicalAnd(MLXArray a, MLXArray b)` — BOOL, broadcast-compatible.
  - `AttentionMask.slidingWindow(MLXScope scope, int queryLength, int keyLength, int window)` — BOOL
    `[queryLength, keyLength]`, `true` = attend. Query row `i` is absolute position
    `keyLength - queryLength + i`; it attends key `j` iff `j <= pos && pos - j < window`.
  - `DecoderAttention extends CachedAttention` (Decision 3), constructed as
    `DecoderAttention(MLXScope scope, int numHeads, int numKeyValueHeads, int headDim, RopeSpec rope,
    int rotaryDims, MLXArray staticFreqs /*nullable*/, Integer slidingWindow, UnaryLayer q, UnaryLayer k,
    UnaryLayer v, UnaryLayer out)`, each projection registered with `child("queryProj"|"keyProj"|
    "valueProj"|"outProj", …)` so parameter paths match `GroupedQueryAttention`'s. It delegates every
    rotation to `RopeSpec.apply` (Task 3) and exposes a package-private
    `MLXArray attend(MLXArray q, MLXArray k, MLXArray v, int offset)` seam used by the window test.
  - `GroupedQueryAttention` is changed to `extends CachedAttention` (behavior untouched).
  - **Window semantics (pinned):** a query at absolute position `p` attends key `j` iff
    `j <= p && p - j < window` — exactly `window` keys, including itself. This matches Hugging Face
    `transformers`' `sliding_window_overlay` mask (`kv_idx > q_idx - sliding_window`), cited by file and
    revision in `AttentionMask`'s javadoc and in `tools/hf-reference/provenance.json`; the offline
    reference (Task 0) produces the goldens, not our reading of it. HF's flash-attention path has
    historically differed by one token at the boundary (Phi-3 ships `sliding_window: 2047` for this
    reason); the eager/SDPA form above is the one pinned. Unlike `GroupedQueryAttention`, `headDim` is
    explicit (Gemma: `numHeads * headDim != hidden`), so the output projection maps
    `numHeads*headDim → hidden`.
  - `GroupedQueryAttention` is left untouched and unused by new code; Task 5 stops using it, and it is
    deleted in Task 5 only if no test or example references it (otherwise kept as-is).

- [ ] **Step 1: Write the failing tests**

`AttentionMaskTest` (native, small arrays read back with `toFloatArray` on an `astype(FLOAT32)` copy):
window 2, `queryLength == keyLength == 4` gives rows `[1,0,0,0],[1,1,0,0],[0,1,1,0],[0,0,1,1]`;
one-token decode (`queryLength=1, keyLength=5, window=3`) gives `[0,0,1,1,1]`; `window >= keyLength`
equals the plain causal mask; `window <= 0` throws `IllegalArgumentException`.

`DecoderAttentionTest` (native): (1) with `slidingWindow=null`, `DecoderAttention` equals
`GroupedQueryAttention` on the same seeded weights to `1e-5` (guards the refactor), and `parameters()`
key sets are identical; (2) **Review Focus 3** — with `slidingWindow=2` and a 6-token prompt, compare the
attention output at the last position with a **direct** call of
`MLXFast.scaledDotProductAttention(qLast, kLast2, vLast2, scale, false, null, null)`, where `qLast` is the
last query rotated at position 5 and `kLast2`/`vLast2` are the last two rows of the already-rotated,
already-cached keys/values (read through the `attend` seam). A fresh 2-entry `KVCache` is **not** used: it
would have `offset=2` and rotate `q` at position 2, giving different output. Repeat for a one-token decode
step whose cache offset is 5. (2b) Boundary: for `window=3` and a query at position 5, key 3 (distance 2)
must contribute and key 2 (distance 3) must not — two inputs differing only in token 3 give different
last-position output, two differing only in token 2 give identical output; repeat at `window+1`.
(3) explicit `headDim=3` with `hidden=4, numHeads=2` (`q_proj` is `[6,4]`, `o_proj` is `[4,6]`) runs and
returns `[1, T, 4]`. (4) `rotaryDims < headDim` leaves the trailing dimensions un-rotated
(compare against manually concatenated rotated/unrotated slices), including with a `Yarn` spec whose
`attentionScaling != 1` (pass-through dims must be unscaled).

- [ ] **Step 2: Run to verify failure**

Run: `./gradlew :jmlx-core:test --tests "*AttentionMaskTest" --tests "*DecoderAttentionTest"`
Expected: FAIL — classes missing.

- [ ] **Step 3: Implement**

Mask: build `pos = arange(keyLength-queryLength, keyLength)` as `[q,1]` and `j = arange(0,keyLength)` as
`[1,k]` (INT32), `causal = lessEqual(j, pos)`, `inWindow = less(subtract(pos, j), window)`,
`logicalAnd`. Allocate every temporary in the caller's step scope, never the model scope (the
`GELU`/`Linear` scope rule in `CLAUDE.md`).

`DecoderAttention.forward`: project, reshape to heads, multiply `q`/`k` by
`rope.queryKeyMultiplier()` when it is not `1`, call `MLXFast.rope(x, rotaryDims, false, base?, rope.scale(),
cache.offset(), freqsOrNull)` where `Base` passes `theta` and no freqs (bit-identical to today's call)
and every other variant passes `base=null` plus the `freqs` array from `rope.frequencies(rotaryDims,
offset + sequence)`; append to the cache; call `scaledDotProductAttention` with `causal=true` when
`slidingWindow == null`, else `causal=false` and `maskArr = AttentionMask.slidingWindow(...)`. Keep
key/value head repetition exactly as `GroupedQueryAttention` does it. Cache the `freqs` array per
`(rotaryDims, offset+sequence)` only if the probe in Task 3 shows a measurable cost; otherwise rebuild
per call in the step scope.

- [ ] **Step 4: Run tests**

Run: `./gradlew :jmlx-core:test --tests "*AttentionMaskTest" --tests "*DecoderAttentionTest" --tests "*GroupedQueryAttentionTest" && ./gradlew verifyMlxApiCallSites generateMlxApiInventory && git diff --stat req/`
Expected: PASS; inventory diff shows only `mlx_logical_and` becoming implemented.

- [ ] **Step 5: Commit**

```bash
git add jmlx-core req/mlx-api-inventory-overrides.json req/mlx-api-inventory.md
git commit -m "Add DecoderAttention with explicit head_dim, RoPE specs and sliding-window masks"
```

---

### Task 5: Component-based decoder and assembler — Llama/Qwen2 through the descriptor

**Files:**
- Create: `jmlx-models/src/main/java/se/alipsa/jmlx/models/DecoderAssembler.java`
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/GatedMlp.java`, `nn/UnaryLayer.java`,
  `nn/CachedAttention.java`
- Create (test): `jmlx-models/src/test/java/se/alipsa/jmlx/models/TinyCheckpoints.java` — seeded,
  **non-zero** checkpoint writer (moved here from Task 6 so the refactor is checked by something that can
  fail), and `GoldenCapture.java` (one-off capture helper)
- Modify: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/DecoderBlock.java`, `nn/RMSNorm.java`,
  `nn/Linear.java`, `nn/QuantizedLinear.java`, `nn/GroupedQueryAttention.java`, `nn/SwiGLU.java`
- Modify: `jmlx-models/.../DecoderModel.java`, `LlamaModel.java`, `QwenModel.java`, `TextGenerationModels.java`
- Test: `jmlx-models/src/test/java/se/alipsa/jmlx/models/LlamaModelTest.java` and `QwenModelTest.java`
  (unchanged goldens), new `DecoderAssemblerTest.java`, `DecoderRefactorGoldenTest.java`

**Interfaces:**
- Consumes: `ArchitectureDescriptor`, `TensorPlan` (Tasks 1–2), `DecoderAttention` (Task 4).
- Produces:
  - `GatedMlp extends UnaryLayer`: `GatedMlp(MLXScope, UnaryLayer gate, UnaryLayer up, UnaryLayer down,
    Activation activation)`, children named `gateProj`/`upProj`/`downProj` as `SwiGLU`'s are;
    `down(act(gate(x)) * up(x))`. `Activation.SILU` is exactly today's `SwiGLU`, which stays
    source-compatible (extends `UnaryLayer`, behavior unchanged).
  - `RMSNorm(MLXScope, MLXArray weight, float eps, boolean weightOffset)`; the three-argument
    constructor stays and means `weightOffset=false`. The offset form stores `weight + 1` **once at
    construction** (computed in float32 and cast back to the checkpoint dtype, as HF does) as the norm's
    parameter value, so `forward` is a plain `rmsNorm` with no per-call op.
  - `DecoderBlock`'s fields become `CachedAttention attention` and `UnaryLayer mlp`; a new constructor
    `(MLXScope, RMSNorm, CachedAttention, RMSNorm, UnaryLayer)` is added and the existing
    `(…, GroupedQueryAttention, …, SwiGLU)` constructor is kept and delegates. Child names stay
    `inputNorm`, `attention`, `postAttentionNorm`, `mlp`.
  - `MoeMlp` (Task 9) also extends `UnaryLayer`. `UnaryLayer`/`CachedAttention` are introduced by
    whichever of Task 4/5 compiles first; the other reuses them.
  - `TinyCheckpoints.randomLlama(dir, seed, kvHeads, attentionBias, tiedHead)` and
    `randomQwen2(dir, seed)`: hidden ≥ 64 (so the same files serve the Task 10a quantization probe),
    `num_attention_heads=4`, GQA with `kv < heads`, weights drawn from a fixed `MLXRandom` key and stored
    with `MLXIO.saveSafetensors`.
  - `static Assembled DecoderAssembler.assemble(MLXScope scope, ArchitectureDescriptor d, Map<String,MLXArray> tensors)`
    where `Assembled(Embedding embedding, List<DecoderBlock> layers, RMSNorm finalNorm, Linear lmHead /*null when tied*/)`.
    First statement: `ArchitectureMappings.tensorPlan(d).validate(tensors.keySet())`.
  - `DecoderModel`'s protected constructor becomes `(MLXScope, ArchitectureDescriptor, Map<String,MLXArray>)`;
    the old `(scope, DecoderConfig, tensors, boolean, boolean)` constructor is removed (it is
    `protected` in an abstract class with only `LlamaModel`/`QwenModel` subclasses, both final and both
    updated here). `DecoderModel.config()` still returns `DecoderConfig`.

- [ ] **Step 1: Write the failing test**

`DecoderAssemblerTest`: a checkpoint missing `model.layers.0.mlp.down_proj.weight` throws before any
layer is built (validate through `TensorPlan`; assert no `Module` child was registered).

`DecoderRefactorGoldenTest` is the real refactor guard. The existing tiny fixtures are all zeros
(`zeros(scope, …)` in `LlamaModelTest`/`QwenModelTest`), so every logit is equal and they cannot detect a
wrong weight mapping, a swapped q/k, a lost bias or a wrong head repetition. They still guard event and
scope behavior and stay unchanged, but they are **not** the numeric golden. Write `TinyCheckpoints`
**first**, on the untouched code, and capture goldens from four seeded, non-zero configurations:
(a) Llama, GQA (`kv=2 < heads=4`), tied head; (b) Llama, GQA, **untied** head; (c) Llama with
`attention_bias=true`; (d) Qwen2 with q/k/v biases and GQA. For each, commit
`jmlx-models/src/test/resources/golden/refactor-<name>.json` holding the full-prompt logits (8-token
prompt), two greedy decode IDs with their logits, and the exact `parameters()` key set. After the refactor
the test asserts logits to `1e-6`, identical greedy IDs and an identical parameter key set (Decision 3).

- [ ] **Step 2: Capture goldens on the untouched code, run tests to verify the new ones fail**

Run: `./gradlew :jmlx-models:test --tests "*LlamaModelTest" --tests "*QwenModelTest"` (baseline green), run
`GoldenCapture` **before any `DecoderModel` edit**, review the JSON, and commit it on its own
(`Capture pre-refactor decoder goldens`). Then
`./gradlew :jmlx-models:test --tests "*DecoderAssemblerTest"` — Expected: FAIL (`DecoderAssembler` missing).

- [ ] **Step 3: Implement**

`DecoderAssembler` reproduces today's `DecoderModel` constructor body from the descriptor: tensor names
`model.layers.{i}.…` for the separate-projection layout, `Embedding`, per-layer `DecoderBlock`, final
norm, and an explicit `lm_head.weight` preferred over tying. `DecoderModel.generate`,
`forward`, and `normalizedHiddenStates` change only in where `embedding`/`layers`/`norm`/`lmHead` come
from; the generation loop, sampling, decoder, event and abort code are not touched. Add the Gemma
embedding scale now as a no-op branch on `descriptor.embedding().scaleBySqrtHidden()` **inside
`normalizedHiddenStates`**, the one method `forward` and `generate` both already call for the embedding
and blocks (only the head projection is duplicated between them), so Task 8 reopens neither. `LlamaModel`/`QwenModel` call `ArchitectureMappings.parse` through
`TextGenerationModels` and pass the descriptor.

- [ ] **Step 4: Run tests**

Run: `./gradlew :jmlx-models:check :jmlx-core:check`
Expected: PASS with unchanged Llama/Qwen goldens (`[1,0,0]`, seeded streams, tokenizer-backed text).
Checkstyle warnings must not exceed the baseline.

- [ ] **Step 5: Commit**

```bash
git add jmlx-core jmlx-models
git commit -m "Build the decoder from architecture descriptors instead of Llama/Qwen branches"
```

---

### Task 6: RoPE reference goldens and Llama 3.1 end-to-end

**Files:**
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/RopeReferenceTest.java`,
  `jmlx-models/src/test/java/se/alipsa/jmlx/models/RopeScalingModelTest.java`
- Consumes (from Task 0): `tools/hf-reference/goldens/rope.json`. The Python **MLX** oracle
  (`tools/mlx-oracle/`) is not used for RoPE values: it would be a second implementation ported by the
  same author from the same HF source, so the two could share one misreading.

**Interfaces:**
- Consumes: `RopeSpec` including `RopeSpec.apply(x, rotaryDims, offset, staticFreqs)` (Task 3).
- Produces: nothing new. `RopeReferenceTest` calls **`RopeSpec.apply` itself** — the production path — and
  does not re-implement the `MLXFast.rope` call.

Cases in `rope.json` (produced offline by HF `ROPE_INIT_FUNCTIONS` + HF's own rotary application):
`{linear, dynamic, llama3, yarn}` × offsets `{0, 1, 7, 100, original_max_position_embeddings + 3}` ×
`(headDim=8, rotaryDims ∈ {8, 4})`, plus **YaRN with `rotaryDims=4 < headDim=8` and a non-unit
`attention_factor`** (Review Focus 4: pass-through dims unscaled), YaRN `truncate=false`, and a YaRN case
with an explicit `attention_factor`. Each entry stores HF's `inv_freq`, `attention_scaling`, the input `x`
`[1,2,T,8]` and HF's rotated output.

- [ ] **Step 1: Write the failing test.** `RopeReferenceTest` (a) checks `RopeSpec.frequencies` against HF's
  `1/inv_freq` and `attentionScaling()` against HF's `attention_scaling` to `1e-6` (pure Java, runs on
  Ubuntu), and (b) natively runs `RopeSpec.apply` and compares to HF's rotated output to `1e-5`.
- [ ] **Step 2: Run to verify failure**

Run: `./gradlew :jmlx-core:test --tests "*RopeReferenceTest"`
Expected: FAIL until Task 3's implementation matches HF. Any disagreement is a real bug in `RopeSpec` or in
the plan's reading of HF; fix `RopeSpec`, never the golden.
- [ ] **Step 3: Fix until green.** Note in the test javadoc the one place HF and MLX are known to differ
  (MLX `freqs` are periods, HF `inv_freq` are reciprocals) and how the test converts.
- [ ] **Step 4: End-to-end Llama 3.1-style tiny model.** `RopeScalingModelTest` uses
  `TinyCheckpoints.randomLlama` (Task 5; seeded, non-zero, hidden ≥ 64) with
  `"rope_scaling":{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,"high_freq_factor":4.0,
  "original_max_position_embeddings":8}` and asserts (a) it loads (rejected before 6.3), (b) a 12-token
  prompt's logits differ from the same checkpoint without scaling, (c) generation is deterministic across
  two runs, and (d) logits match a `transformers` golden for the same tiny checkpoint if Task 0 produced
  one for this configuration (it should: add a `llama31` entry to `goldens/`).
- [ ] **Step 5: Run tests and commit**

Run: `./gradlew :jmlx-core:test :jmlx-models:test verifyHfReferenceGoldens`
Expected: PASS.

```bash
git add jmlx-core/src/test jmlx-models/src/test tools/hf-reference
git commit -m "Pin RoPE scaling against Hugging Face reference goldens and load a Llama 3.1-style config"
```

---

### Task 7: Mistral (sliding window) and Phi-3 (fused projections)

**Files:**
- Modify: `jmlx-models/.../ArchitectureMappings.java` (`mistral`, `phi3`), `TensorPlan` builders,
  `DecoderAssembler.java` (fused split), `TextGenerationModels.java` (dispatch)
- Create: `jmlx-models/.../MistralModel.java`, `Phi3Model.java` (thin `final` subclasses mirroring `LlamaModel`)
- Test: `MistralModelTest.java`, `Phi3ModelTest.java`, additions to `ArchitectureMappingsTest.java`

**Interfaces:**
- Consumes: descriptor, `DecoderAttention` with `slidingWindow`, `GatedMlp`.
- Produces: `MistralModel.load(MLXScope, Path)` / `Phi3Model.load(MLXScope, Path)` (same shape as
  `LlamaModel.load`); `model_type` `mistral` and `phi3` in `TextGenerationModels`.
  `DecoderAssembler` splits a fused `self_attn.qkv_proj.weight` `[(H+2*KV)*headDim, hidden]` into
  q/k/v with `MLXShape.slice` **once at load** (owned by the model scope) and a fused
  `mlp.gate_up_proj.weight` `[2*inter, hidden]` into gate/up. Splitting is a load-time copy, not a
  forward-time op.

- [ ] **Step 1: Write the failing tests**

Pure-Java (`ArchitectureMappingsTest`): `mistral` config with `sliding_window: 4` maps to
`attention().slidingWindow() == 4`; `sliding_window: null` maps to `null` (Mistral v0.2/0.3 ship `null`);
`phi3` maps `FUSED_GATE_UP` and `fusedQkv`; `phi3` with `rope_scaling.rope_type=longrope` is rejected
naming the key and value; a `mistral` config with `layer_types` is rejected (per-layer schedules are
deferred). Native (`MistralModelTest`, `Phi3ModelTest`, seeded non-zero tiny checkpoints):

1. **Sliding-window behavior** — (a) property: with `sliding_window=4` and **one** decoder layer, two
   prompts that differ only in tokens older than the window give identical next-token logits (with several
   layers the receptive field grows to `layers * window`, so the property test uses one layer; the
   multi-layer case asserts only the HF golden); (b) reference: Mistral tiny logits for a 10-token prompt
   and two decode steps equal the committed `transformers` golden
   (`tools/hf-reference/goldens/mistral.json`, Task 0), which pins the window boundary to HF's mask;
   (c) `sliding_window=null` (Mistral v0.2/0.3) equals full causal attention.
2. **Phi-3 fused mapping** — build the same weights once as separate `q/k/v`, `gate/up` (loaded as
   `llama`) and once fused (loaded as `phi3`); logits agree to `1e-6`, pinning the split order (`q,k,v` and
   `gate,up`). The Phi-3 tiny model's logits also match `goldens/phi3.json`. Both tests read the **same
   committed safetensors** the `transformers` reference used
   (`tools/hf-reference/goldens/checkpoints/<family>/`), so Java and HF share weights by construction.
3. Missing/extra tensor cases from Task 2 for each family (forbidden separate `q_proj` on `phi3`,
   forbidden fused `qkv_proj` on `mistral`).

- [ ] **Step 2: Run to verify failure**

Run: `./gradlew :jmlx-models:test --tests "*MistralModelTest" --tests "*Phi3ModelTest" --tests "*ArchitectureMappingsTest"`
Expected: FAIL — unsupported `model_type`.

- [ ] **Step 3: Implement**

Mistral is Llama's descriptor with `slidingWindow` from `sliding_window` and no bias (its `consolidated.safetensors` handling is Task 2's). Phi-3 reads
`num_key_value_heads`, `rope_scaling` (only `default`/absent accepted until `longrope` is scheduled),
`partial_rotary_factor`, and optional `sliding_window`; tensor keys are `self_attn.qkv_proj.weight`,
`self_attn.o_proj.weight`, `mlp.gate_up_proj.weight`, `mlp.down_proj.weight`, `input_layernorm.weight`,
`post_attention_layernorm.weight`. Validate fused shapes before slicing (rows must equal
`(numHeads + 2*numKvHeads) * headDim`) and name the tensor on mismatch.

- [ ] **Step 4: Run tests**

Run: `./gradlew :jmlx-models:check`
Expected: PASS; Llama/Qwen goldens unchanged.

- [ ] **Step 5: Commit**

```bash
git add jmlx-models
git commit -m "Add Mistral sliding-window and Phi-3 fused-projection decoders"
```

---

### Task 8: Gemma (v1)

**Files:**
- Modify: `ArchitectureMappings.java`, `DecoderAssembler.java`, `DecoderModel.java` (embedding scale),
  `nn/RMSNorm.java` (offset already added in Task 5), `nn/GatedMlp.java` (`GELU_TANH`), `MLXOps.java` only if a
  tanh-GELU needs ops not present (`tanh`, `power`/`square` exist)
- Create: `jmlx-models/.../GemmaModel.java`
- Test: `GemmaModelTest.java`, `jmlx-core/src/test/java/se/alipsa/jmlx/nn/ActivationsTest.java` additions

**Interfaces:**
- Produces: `Activation.GELU_TANH` = `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))` (HF
  `gelu_pytorch_tanh`, what Gemma v1's `hidden_activation`/`hidden_act` uses); `GemmaModel.load`.

- [ ] **Step 1: Write the failing tests**

- Activation golden: `GELU_TANH` at `x ∈ {-3,-1,0,0.5,2}` against values from
  `torch.nn.functional.gelu(x, approximate="tanh")` produced by the Task 0 tool and committed in
  `goldens/gemma.json`; it differs from the exact `GELU` by the tanh approximation error (assert both).
- Config: `gemma` requires `head_dim` (Gemma v1 7B has `head_dim=256`, `hidden=3072`, 16 heads, so
  `head_dim*heads != hidden`). The check to relax is the **mapping's** `head_dim * heads == hidden` rule
  (moved from `DecoderConfig.fromFile` to `ArchitectureMappings` in Task 1), *not* `DecoderConfig`'s
  constructor, which has no `headDim` field and whose `hidden % heads == 0` check real Gemma v1 configs
  already satisfy; the Tier-A tiny Gemma must satisfy it too. The relaxation applies only to families whose
  mapping declares an explicit head dimension; a Llama config with a mismatched `head_dim` stays rejected.
  **Activation precedence** follows HF: `hidden_activation` if present, else `hidden_act`. A null/absent
  `hidden_activation` means `gelu_pytorch_tanh` (HF's Gemma default); an explicit `gelu_pytorch_tanh` or
  `gelu_new` maps to `GELU_TANH`; an explicit `gelu` maps to the **exact** `GELU`; anything else is
  rejected by name. `tie_word_embeddings` defaults **true** for Gemma (unlike Llama's false).
  `gemma2`/`gemma3` `model_type` are rejected with "deferred to a later milestone".
- Behavior (native, seeded non-zero tiny checkpoint): (1) the `RMSNorm` offset form equals
  `rmsNorm(x, 1+w)`, adds no per-call op (assert the stored parameter is `w+1`), and with a bfloat16 weight
  the stored value equals the float32 sum cast back; (2) embedding output is scaled by `sqrt(hidden)` **in
  the working dtype** (HF casts the normalizer to the hidden dtype: assert on the float32 fixture and note
  bf16 rounding in a comment for Tier B); (3) a tiny Gemma prefill/decode result (`GemmaModelTest`) equal to
  `goldens/gemma.json` from Hugging Face `transformers` (Task 0) over the committed safetensors both sides
  read, so the Java model is checked against Hugging Face's own implementation, not a port of it.

- [ ] **Step 2: Run to verify failure**

Run: `./gradlew :jmlx-core:test --tests "*ActivationsTest" && ./gradlew :jmlx-models:test --tests "*GemmaModelTest"`
Expected: FAIL.

- [ ] **Step 3: Implement**

Descriptor: `Norm(RMS, eps, weightOffset=true)`, `Mlp(SEPARATE_GATE_UP, GELU_TANH, bias=false)`,
`Embedding(scaleBySqrtHidden=true)`, `Head(tied=true)`, explicit `headDim`. `DecoderAssembler` builds
`DecoderAttention` with `q: hidden→heads*headDim`, `o: heads*headDim→hidden`. The embedding scale lives in
`normalizedHiddenStates` (added as a no-op in Task 5), which `forward` and `generate` both already call, so
there is nothing to keep in sync; only the head projection is duplicated between them and it is not
affected. Multiply in the working dtype (HF casts the normalizer to the hidden dtype).

- [ ] **Step 4: Run tests**

Run: `./gradlew :jmlx-core:check :jmlx-models:check verifyHfReferenceGoldens`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jmlx-core jmlx-models tools/hf-reference
git commit -m "Add Gemma v1 decoder: offset RMSNorm, scaled embeddings, GeGLU and explicit head_dim"
```

---

### Task 9: Mixtral (mixture of experts)

**Files:**
- Create: `jmlx-core/src/main/java/se/alipsa/jmlx/nn/MoeMlp.java`
- Modify: `ArchitectureMappings.java`, `TensorPlan` builders, `DecoderAssembler.java`, `TextGenerationModels.java`;
  `MLXOps.java`/`MLXShape.java` only for ops the probe below shows missing (`mlx_topk_axis`,
  `mlx_argpartition_axis` are `planned`; 6.1's `SamplingPipeline` already uses top-k selection — reuse its facade)
- Create: `jmlx-models/.../MixtralModel.java`
- Test: `jmlx-core/src/test/java/se/alipsa/jmlx/nn/MoeMlpTest.java`, `MoeRoutingProbeTest.java`,
  `jmlx-models/src/test/java/se/alipsa/jmlx/models/MixtralModelTest.java` (reference: Task 0's `mixtral.json`)

**Interfaces:**
- Produces: `MoeMlp extends UnaryLayer`, `MoeMlp(MLXScope, UnaryLayer router, List<GatedMlp> experts,
  int topK)`, children `router` and `expert0..expertN-1`. `forward(x [B,T,H])`:
  `logits = router(x)` `[B,T,E]`; `probs = softmax(logits, axis=-1)` in float32; select the `topK`
  indices; `weights = probs[top] / sum(probs[top])` (Mixtral renormalises); output
  `Σ_e where(selected_e, weight_e · expert_e(x), 0)` computed densely. Unselected experts are masked with
  **`where`, never multiplied by a zero weight**: `inf * 0` is NaN, plausible for a bf16 expert on tokens it
  was not routed to. Constructor rejects `topK < 1 || topK > experts.size()`.
- **Tie-break (must be probed, not assumed).** `phase6-0b-probe-findings.md` establishes first-index ties
  only for `argmaxAxis`; its `argsort` case only checked a sorted permutation, and `MLXOps.argsortAxis`'s
  javadoc says "Stable" without a probe behind it. Step 1 adds `MoeRoutingProbeTest` recording, on the
  pinned MLX, whether `argsortAxis` (on negated probabilities) is stable for exact ties and whether
  `topk`/`argpartition` (from 6.1's facade, if present) break ties by lowest index. **If `argsort` is not
  provably stable, top-1 uses `argmaxAxis` (probed, first index) and top-k>1 builds its selection by
  `topK` successive `argmax` + mask passes**, which inherit the probed rule. HF uses `torch.topk`, whose
  tie order is unspecified, so Review Focus 5 asserts *determinism* (identical result across 100 runs and
  across batch positions) and lowest-index preference on exact ties, and documents that HF may differ on
  exact ties only.
- Tensor keys: `model.layers.{i}.block_sparse_moe.gate.weight`,
  `…block_sparse_moe.experts.{e}.w1.weight` (gate), `.w2.weight` (down), `.w3.weight` (up).
  `w1`/`w3` map to gate/up, `w2` to down; pin this with a test (swap detection).

- [ ] **Step 1: Write the failing tests**

`MoeRoutingProbeTest` (native, first): records the tie behavior described above in its javadoc and pins it.

`MoeMlpTest`: (1) **top_k == experts** — output equals the softmax-weighted sum of all experts (Review
Focus 5); (2) **top_k=1, tied router logits** — chooses expert 0, identically across 100 runs and for every
batch/token position; (3) **weights renormalise** — for `top_k=2` of 4, selected weights sum to 1 within
`1e-6`; (4) an expert with zero weights contributes zero; (5) **NaN safety** — an unselected expert whose
weights produce `inf` on the input does not poison the output (build one with a huge weight and assert a
finite result equal to the routed experts' sum); (6) the dense formulation equals a per-token loop reference
in plain `float[]` math on a 2-token, 3-expert, hidden-4 case.

`MixtralModelTest`: config maps `num_local_experts`, `num_experts_per_tok`. **Sliding window:** HF's
`MixtralConfig` defines `sliding_window` (Mixtral-8x7B ships `null`); `null` is accepted, a positive value
is honored through the same `DecoderAttention` window as Mistral (the family shares that capability), and
`use_sliding_window`-style schedules are not applicable. `router_aux_loss_coef`, `router_jitter_noise` and
`output_router_logits=false` are in the mapping's explicit `ignored` table (training-only; Scope decision
6); `output_router_logits=true` is rejected by name; a dense family carrying `num_local_experts` is
rejected (Review Focus 1). The reference is `tools/hf-reference/goldens/mixtral.json`: a 2-layer, 4-expert,
top-2 tiny `transformers` Mixtral saved to `goldens/checkpoints/mixtral/`, with prefill logits and two greedy
decode steps; Java reads the **same safetensors** and asserts `1e-4`.

- [ ] **Step 2: Run to verify failure, then implement**

Run: `./gradlew :jmlx-core:test --tests "*MoeMlpTest" && ./gradlew :jmlx-models:test --tests "*MixtralModelTest"`
Expected: FAIL. Implement `MoeMlp` with only ops already exposed (`softmaxAxis`, `argsortAxis` descending
via negation, `takeAlongAxis`, `where`, `sum`, `divide`, `multiply`); add `topk`-style facades only if
`SamplingPipeline` does not already expose one reusable from `jmlx-core` — check first and reuse.
Build the `[B,T,E]` routing tensor by comparing an `arange(E)` against the selected indices (`equal` +
`where`), avoiding scatter ops that are `unplanned`.

- [ ] **Step 3: Run tests**

Run: `./gradlew :jmlx-core:check :jmlx-models:check verifyHfReferenceGoldens verifyMlxApiCallSites`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add jmlx-core jmlx-models tools/hf-reference req/mlx-api-inventory-overrides.json req/mlx-api-inventory.md
git commit -m "Add Mixtral mixture-of-experts decoder with dense routed MLP"
```

---

### Task 10: Quantization and GGUF probe, tokenizer goldens per family, Tier-B, docs and CI

This task closes the gate. It has four independently committable parts; do them in this order.

**10a — Quantization/GGUF probe and decision**

**Files:** Create `jmlx-models/src/test/java/se/alipsa/jmlx/models/QuantizedCheckpointProbeTest.java`;
create `req/plans/phase6-3-probe-findings.md`.

- [ ] Probe (native): quantize the **seeded Task 5 `TinyCheckpoints.randomLlama` (hidden ≥ 64, intermediate
  ≥ 128)** — MLX quantization needs the last dimension divisible by `group_size`, so the old hidden-4 tiny
  model cannot use group 64 (use group 32 only if a dimension must stay smaller, and record it). Quantize its
  linear weights with `MLXQuant.quantize` (affine, group 64, 4-bit), write `weight/scales/biases` triplets to safetensors, and load them into `QuantizedLinear`
  through a spike assembler; compare logits to the float model within a recorded tolerance. Record: the
  MLX-community safetensors key layout (`*.weight`, `*.scales`, `*.biases`) and the `config.json`
  `quantization: {group_size, bits}` block; whether embeddings/`lm_head` are quantized (they are in
  MLX-community exports, so `Embedding.project` needs a quantized path); which GGUF quantizations
  (`Q4_0`, `Q8_0`, `Q4_K`, …) `MLXIO.loadGguf` dequantizes on read versus returns packed.
- [ ] **Decision gate (write it in the findings file, do not skip):** if MLX-community quantized safetensors
  load with no new native surface, add a Task 10a′ that maps the `quantization` block to `QuantizedLinear`
  via the descriptor (`TensorPlan` requires `scales`/`biases` per quantized weight; embeddings use a
  `QuantizedEmbedding` only if the probe shows `MLXQuant.dequantize` per lookup is acceptable). If it does
  not, the compatibility matrix keeps "float weights only" and `config.json` `quantization` stays a named
  rejection. GGUF is claimed **only** for variants the probe shows load correctly; otherwise it is listed
  as unsupported with the exact key/metadata error. Either outcome satisfies the gate; an unrecorded
  outcome does not.
- [ ] **GGUF gap, recorded explicitly.** The spec (§6.3 item 4) asks for "chosen GGUF quantizations"; the
  gate above is satisfied even if none is chosen, and the plan says so rather than hiding it. If GGUF *is*
  chosen, the work is larger than a loader call: a `blk.N.attn_q.weight` → `model.layers.N.self_attn.q_proj`
  name mapping table, llama.cpp's q/k **un-permutation** (GGUF stores q/k in an interleaved rotary layout
  that differs from HF's half-split layout), tokenizer/vocab metadata keys, and per-type dequantization
  parity tests. Choosing it requires its own plan amendment.
- [ ] Commit: `git add jmlx-models/src/test req/plans/phase6-3-probe-findings.md && git commit -m "Record the quantized-checkpoint and GGUF probe findings"`

**10b — Per-family tokenizer/template goldens (Review Focus 7)**

**Where the goldens come from.** `tools/tokenizer-oracle/runner.py` has no chat, template or Jinja support
and its lock pins only `tokenizers==0.23.2` (no `jinja2`, no `transformers`), so it cannot produce rendered
chat text. Repinning it is rejected: it would change the environment that verifies every existing
tokenizer fixture. Instead, rendered chat text and token IDs come from the **offline Task 0 tool**
(`tools/hf-reference/`, `transformers` + `jinja2`): `generate.py --chat` loads each family's committed
`tokenizer.json` + `tokenizer_config.json` with `AutoTokenizer`, calls `apply_chat_template(...,
tokenize=False, add_generation_prompt=...)` and `tokenizer(rendered, add_special_tokens=False)`, and writes
`goldens/chat-<family>.json` (hash-pinned by `verifyHfReferenceGoldens`). Where a family's template is
already in the Node/`@huggingface/jinja` corpus (`jmlx-jinja/src/test/resources/model-templates/
mistral-7b-instruct-v0.3.jinja`), `Phase63FamilyTokenizerTest` additionally asserts that the golden's
rendered text equals what `jmlx-jinja` renders for the same context — a second, independent renderer as a
cross-check, not the source of truth. Byte-level *encode* parity for these tokenizers keeps using the
existing `tokenizers` oracle (`generateTokenizerOracleFixtures`), which needs no template support.

**Files:** Create `jmlx-tokenizer/src/test/resources/families/{mistral,gemma,phi3,mixtral}/` (small
`tokenizer.json` + `tokenizer_config.json` derived from the family's real files where the license allows
redistribution, otherwise a synthetic file with the same component shape and the family's real template,
recorded in the manifest); extend `tools/tokenizer-oracle/fixtures/*.input.json` for encode parity; test
`Phase63FamilyTokenizerTest.java` in `jmlx-tokenizer`; goldens under `tools/hf-reference/goldens/`.

- [ ] For each family: `HfTokenizer.fromDirectory` loads; `renderChat` on a plain, a system+user and a
  multi-turn conversation matches `goldens/chat-<family>.json` for text and IDs. Include the family
  quirks: Mistral's `[INST]` template with no system role (assert whatever HF does: fail or fold),
  Gemma's `<start_of_turn>` turns and `assistant→model` role rename, Phi-3's `<|user|>` markers, Mixtral
  sharing Mistral's template.
- [ ] `GenerationRequest.chat` for each family adds no duplicate BOS (6.2's `OMIT` invariant).
- [ ] The 6.2 bounds check stays (`tokenizer.vocabSize() <= config.vocabSize()`); add a padded-head test
  (`vocab_size` larger than the tokenizer) per family.
- [ ] Commit: `git commit -m "Add tokenizer and chat-template goldens for Mistral, Gemma, Phi-3 and Mixtral"`

**10c — Tier-B evidence and workflow**

**Files:** Modify `req/phase6-tier-b-artifacts.md`; create `.github/workflows/tier-b.yml`; create
`jmlx-models/src/tierB/java/se/alipsa/jmlx/models/TierBSmokeTest.java` on a dedicated `tierB` source set
(see "Build wiring" below), gated on `JMLX_TIER_B_MODEL_DIR`.

**Build wiring for the new source set.** `jmlx-models/build.gradle` declares `sourceSets.tierB` (compile
classpath = `main` + `test` runtime, `testFixtures(project(':jmlx-ffi'))`) and a `tierBTest` `Test` task
using JUnit Platform with `jmlx.library.path` and `--enable-native-access=ALL-UNNAMED` (copy the root
`tasks.withType(Test)` settings; a hand-registered task does not inherit them automatically unless it is a
`Test`, which it is). It is **not** wired into `check`. Root `build.gradle`'s Spotless target and the
Checkstyle configuration must be told about the new directory: add `src/tierB/java/**/*.java` to the
Spotless `googleJavaFormat` target and register `checkstyleTierB`, then include both in the root
`spotlessCheck`/`check` so the new sources are formatted and linted like every other source set. Verify with
`./gradlew spotlessCheck checkstyleTierB` before the first commit.

- [ ] **Feasibility first.** For each candidate record the download size *and* the peak resident memory
  needed to load it on the target runner: hosted Apple-Silicon runners have limited RAM (verify the
  current `macos-26` allocation from GitHub's runner documentation at selection time and record the
  figure and date), and weights load in their stored dtype (for example Phi-3-mini in bf16 is on the order
  of 7–8 GB; Gemma-2B is borderline). A candidate that cannot fit is either replaced by a smaller
  same-architecture artifact or moved to a self-hosted runner recorded as the trigger owner; otherwise it
  stays `candidate pending`. Do not put an unverified memory assumption in the manifest.
- [ ] Select **one accessible, licensed** artifact per family and fill the manifest columns: repository,
  immutable revision, SHA-256 of every consumed file, license, access requirement, expected metadata,
  exact asserted output, maximum download size, **the device and macOS/MLX pins the output was recorded
  on**, and trigger owner. Candidates to evaluate (verify license and size at selection time; record the
  evaluation): a small Llama-architecture model, Qwen2.5-0.5B (Apache-2.0), a Mistral-architecture ≤1B
  model, a Gemma-architecture 2B (gated: record the license-acceptance requirement), Phi-3-mini (MIT), and
  a tiny Mixtral-architecture test model. A family whose only candidate is gated with no credentials
  available, or that cannot fit, is recorded `candidate pending` and its matrix row stays
  `implemented / verified-with-synthetic-fixture`.
- [ ] **Assertions respect the reproducibility contract** (no claim across device, MLX pin or batch
  shape). `TierBSmokeTest` asserts exact greedy token IDs **only when the running device and pins equal
  those recorded in the manifest row**; otherwise it asserts structural properties (loads, prompt
  renders, 16 tokens generated, no NaN logits, generated text non-empty and decodes) and reports the
  mismatch as a warning, never a silent pass.
- [ ] `tier-b.yml`: `workflow_dispatch` plus a weekly `schedule`, `macos-26` (or the recorded self-hosted
  runner), `actions/cache` keyed on the manifest revision, `HF_TOKEN` from a repository secret used
  **only** for gated rows, a hard size cap enforced before download. It is never a required check.
- [ ] Commit: `git commit -m "Record Tier-B artifacts and add the opt-in real-artifact workflow"`

**10d — Documentation and matrix**

**Files:** Modify `req/phase6-compatibility.md`, `req/phase6-tier-a-fixtures.md`, `jmlx-models/README.md`,
`jmlx-models/build.gradle` (POM description), `CLAUDE.md` (supported architectures sentence, RoPE
scaling now supported), `req/plans/phase5-m3-retrospective.md` is left as history.

- [ ] Matrix: for each family, set status/verification/tokenizer/checkpoint/quantization/RoPE/cache
  columns to exactly what was proven — `implemented` + `verified-with-synthetic-fixture` per Tier-A
  fixture, `verified-with-real-artifact` only with a Tier-B run recorded, "sliding-window masking; cache
  unbounded until 6.4" for Mistral/Phi-3, "dense MoE compute" for Mixtral. Remove the `planned (6.3)`
  rows that shipped; leave `gemma2`/`gemma3`, `phi` (Phi-2), `longrope`, and shared-expert MoE as
  explicit `planned`/unsupported rows.
- [ ] `phase6-tier-a-fixtures.md`: add rows for the RoPE oracle, sliding-window equivalence, Phi-3 fused
  mapping equivalence, Gemma and Mixtral tiny goldens, family tokenizer goldens, and `TensorPlan` tests.
- [ ] README: supported-architectures list, `rope_scaling` support, an "unsupported and how it fails"
  table (quantization unless 10a′, `gemma2`, `longrope`, unknown tensors), and the cache-memory caveat.
- [ ] Commit: `git commit -m "Update the compatibility matrix and docs for Phase 6.3"`

---

## Verification matrix

Platform-independent (Ubuntu CI job, no native):

```text
./gradlew -p buildSrc check
./gradlew :check verifyHfReferenceGoldens
./gradlew :jmlx-tokenizer:check :jmlx-models:check
./gradlew verifyTokenizerOracle verifyTokenizerOracleFixtures
```

macOS Apple Silicon after `./scripts/bootstrap-native.sh` and `./tools/mlx-oracle/install.sh` (the
MLX oracle still verifies the 6.1 sampling fixtures; `tools/hf-reference/` needs no install to verify):

```text
./gradlew --no-build-cache :jmlx-core:check :jmlx-models:check
./gradlew :check verifyMlxApiHeaderCoverage verifyMlxOracle verifyMlxOracleFixtures
```

Run Spotless, Checkstyle, Javadoc and `git diff --check` before each task's commit. Tier-B runs only by
`workflow_dispatch`/schedule and is reported in the PR, never required.

## Self-review against the spec (`phase6-plan.md` §6.3)

| Spec requirement | Task |
| --- | --- |
| Capability descriptors replace Llama/Qwen branches (attention layout/bias, norm, MLP, tied output, RoPE, sliding window, name mapping) | 0, 1, 5, 7, 8 |
| Descriptor validates every required and forbidden tensor before constructing any layer | 2, 5 |
| RoPE scaling before accepting configs that declare it; probe `mlx_fast_rope_dynamic`/YaRN factor; reference agreement at multiple positions | 0, 3, 6 |
| Mistral (sliding window), Gemma, Phi, one MoE family, each with mapping table, tiny safetensors fixture, golden prefill/decode IDs | 0, 7, 8, 9 |
| Tokenizer/template golden before a family's row becomes supported | 10b |
| One Tier-B accessible licensed artifact per family | 10c |
| Common indexed safetensors forms; chosen GGUF quantizations; errors name the key/tensor and capability | 2, 10a |
| Gate: every implemented capability has a Tier-A fixture; unsupported artifacts fail before a partial model | 2, 5, all tasks' rejection tests |

**Gaps recorded rather than hidden:**

- The "mapping table from `config.json`/tensor names to capabilities" is delivered as code
  (`ArchitectureMappings` + `TensorPlan`, including the `consumed`/`ignored` key tables) and rendered into
  the compatibility matrix and README in 10d; a generated Markdown table is an optional extra.
- **GGUF may end with no chosen variants** (10a): the gate is satisfied by a recorded decision, and a real
  GGUF path needs its own plan amendment (name mapping, q/k un-permutation, metadata).
- **Independence of numeric ground truth.** RoPE, sliding-window, Mistral, Phi-3, Gemma and Mixtral goldens
  and every chat-template golden come from Hugging Face `transformers` through the **offline, generate-only
  Task 0 tool**, verified in CI only by committed SHA-256 (`verifyHfReferenceGoldens`); CI never installs
  `torch`. `transformers` is deliberately **not** added to `tools/tokenizer-oracle/` or `tools/mlx-oracle/`,
  whose environments verify existing fixtures. A family whose tiny model `transformers` cannot produce is
  labelled `verified-with-synthetic-fixture` without an independent reference.
- Dynamic-NTK output depends on prefill chunking (cache keeps old rotation, matching HF); documented, not
  "fixed".
- Cache memory stays unbounded and MoE compute stays dense until 6.4.
- Behavior changes to checkpoints/configs that load today (extra biases rejected, unknown `model_type`
  from `DecoderConfig.fromFile`) are listed in the Baseline and pinned by regression tests.


## Acceptance gate

Accept Phase 6.3 only when:

- Llama and Qwen2 goldens (`[1,0,0]`, seeded streams, tokenizer-backed text) are unchanged after the
  descriptor refactor, and `DecoderModel` contains no `model_type` branch;
- every `rope_type` in {linear, dynamic, llama3, yarn} matches the committed Hugging Face reference at
  offsets including one beyond `original_max_position_embeddings`, with YaRN also checked for
  `partial_rotary_factor < 1` (pass-through dims unscaled), and `longrope` fails by name;
- sliding-window attention matches the pinned window semantics (`p - j < window`) at `window` and
  `window+1` distance for prefill and decode, and the one-layer older-token-independence property;
- Mistral, Gemma, Phi-3 and Mixtral each load a Tier-A tiny checkpoint and match committed
  prefill/decode goldens from Hugging Face over shared committed safetensors, and each unsupported
  field/tensor in Review Focus 1, 2, 6 and 8 fails with the key named before any layer exists (the
  index/shard check runs on Ubuntu without native code);
- each family has chat-template text/ID goldens from the offline Hugging Face tool (cross-checked against
  `jmlx-jinja`'s renderer where a template exists there) and encode parity from the pinned `tokenizers`
  oracle;
- the quantization/GGUF probe outcome is recorded, and the matrix claims only what it proved;
- every family the matrix marks `verified-with-real-artifact` has a recorded Tier-B run; the rest say
  `verified-with-synthetic-fixture` or `planned`;
- inventory, call-site guard, Spotless, Checkstyle (no new warnings), Ubuntu Java/oracle CI and macOS
  native CI are green.

## Explicitly deferred

- `gemma2`/`gemma3`, Phi-2 (`phi`), `longrope`, shared-expert or expert-parallel MoE, per-layer window
  schedules, Qwen2 `use_sliding_window`.
- Cache capacity, sliding-window *eviction*, cache quantization, fork/reorder (6.4); dense MoE
  performance work (6.4); batching (6.5).
- Running `transformers`/`torch` in CI or adding them to the pinned oracle environments (they stay in the
  offline Task 0 tool only).
- Model downloading, caching and license acceptance inside the library; multimodal and encoder-decoder
  models (Phase 7).
