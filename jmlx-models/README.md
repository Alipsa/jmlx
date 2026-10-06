# jmlx-models

Reference local decoder-model implementations built on jmlx. The module loads Hugging Face
safetensors checkpoints and provides inference-only Llama, Qwen2, Qwen3, Mistral, Gemma v1, Phi-3,
and Mixtral decoders with a pure-Java Hugging Face tokenizer. The native runtime supports macOS on
Apple Silicon. Every supported family has committed Hugging Face tiny-checkpoint references; Qwen3
adds per-head query/key normalization, an explicit head dimension, and tied or untied output heads.

The API supports greedy generation and explicitly seeded sampling, synchronous token/text events,
raw-text and configured-chat requests, penalties, and top-k/top-p/min-p filtering. Additional model
serving infrastructure remains later Phase 6 work; see
the [compatibility matrix](../req/phase6-compatibility.md).

MLX-community affine-quantized checkpoints (`quantization: {group_size, bits}` in `config.json`,
tensors stored as `.weight`/`.scales`/`.biases`) load for Llama, Qwen2, Qwen3, Mistral, Gemma v1
and Phi-3.
The projections stay packed and run through the fused quantized matmul. The input-embedding lookup
stays packed too, but gathers and dequantizes only the selected rows, never expanding the full
table; the same table serves as the tied output head through the fused quantized matmul. Norm
weights and projection biases stay float. Layers without `.scales` remain float, including layers
whose input width is not compatible with the group size. Packing parameters are checked against the
model's configured input dimensions, packed weight columns and scales columns at load time;
inconsistent `group_size` or `bits` values fail with the offending tensor named.

The descriptor validates supported `config.json` capabilities and checkpoint tensor names before
constructing decoder layers. RoPE supports base, linear, dynamic NTK, Llama 3, and YaRN scaling.
Dynamic NTK uses the sequence length at each call; cached keys retain their earlier rotation, so
output can depend on prefill chunking. Every generation request defaults to a FULL cache, including
checkpoints with a sliding attention window. A checkpoint with a non-null `sliding_window` may opt
into post-attention cache eviction with
`request.withCachePolicy(GenerationCachePolicy.slidingWindowFromModel())`; a FULL cache keeps all
keys and still applies the model's attention mask. `GenerationCachePolicy.full(capacity)` rejects
requests that cannot fit before prefill. A sliding request is rejected if the loaded checkpoint has
no window. Mixtral evaluates only the
`num_experts_per_tok` experts selected per token (gathered matmul over expert weights stacked
at load time).

| Unsupported input | Load-time result |
| --- | --- |
| GGUF, GPTQ/AWQ and other non-MLX quantization | The checkpoint format or `quant_method` is rejected; only float safetensors and MLX affine quantization are loaded |
| MLX quantization with per-layer overrides, a non-affine `mode`, or on Mixtral | The offending `quantization` key is named |
| `gemma2`, `gemma3`, `qwen3_moe`, Phi-2 `phi` | Unsupported `model_type` named |
| Phi-3 `longrope` | Unsupported `rope_scaling.rope_type` named |
| Unknown checkpoint tensor or forbidden projection bias | Tensor key named by the preflight validator |
| `use_sliding_window=true` (any family) | Config key named |

## Load and generate text

Add `jmlx-models` and the native companion at runtime, then keep the model scope open for the model's
lifetime:

```java
try (MLXScope scope = new MLXScope()) {
  TextGenerationModel model = TextGenerationModels.load(scope, modelDirectory);
  HfTokenizer tokenizer = HfTokenizer.fromDirectory(modelDirectory);
  int eos = tokenizer.eosTokenId(tokenizer.metadata().eosToken().orElseThrow()).orElseThrow();
  GenerationConfig config = GenerationConfig.greedyDefaults(64, Set.of(eos));

  GenerationResult result = model.generate(
      GenerationRequest.text(
          tokenizer, "Hello", PromptSpecialTokens.ADD, config, CancellationToken.NONE),
      event -> {
        if (event.textDelta() != null) {
          System.out.print(event.textDelta());
        }
      });
  System.out.println("\ncomplete: " + result.generatedText());
}
```

For chat, the template owns special tokens and the request enforces omission automatically:

```java
GenerationRequest request = GenerationRequest.chat(
    tokenizer,
    List.of(Map.of("role", "user", "content", "Hello")),
    ChatTemplateOptions.defaults(true),
    config,
    CancellationToken.NONE);
```

Use the existing `GenerationRequest(int[], ...)` constructor for pretokenized input. Its event
`textDelta` and result `generatedText` remain null. Tokenizer-backed requests produce non-null,
possibly empty deltas; the terminal event can carry the decoder's final UTF-8 flush. Concatenating
all token and terminal deltas equals `GenerationResult.generatedText()`.

For sampling, use a positive temperature and explicit request-local seed. Selected-token log
probabilities describe the final filtered and renormalized distribution; greedy log probability is
`0.0` by API convention. Record the jmlx/MLX pins, checkpoint, prompt IDs, complete generation
policy, and seed when reproducibility matters.

`DecoderModel.forward(tokenIds, caches)` now evaluates logits and cache tensors before returning;
an exception after a layer advances poisons the whole cache set until every cache is reset.
`forward(tokenIds, caches, validLengths)` accepts left-padded batched rows and a positive valid
length per row. Dynamic NTK rejects unequal-position batches. Cache quantization is a named
unsupported capability: the pinned runtime rejects packed K/V in SDPA, and full-cache
dequantization raises decode peak memory. `GenerationCachePolicy.quantized(bits, groupSize)` fails
immediately, without a float-cache fallback. The native probe and benchmark contract are documented
in [Phase 6.4 benchmark](../req/phase6-4-benchmark.md).

## Precision in Java and Groovy applications

Choose the precision mode when launching the application's process. MLX's
`MLX_ENABLE_TF32` setting controls reduced-precision paths for float32 matrix
operations on supported hardware; arrays still have the `FLOAT32` dtype.
The pinned MLX runtime defaults to `1` and caches this setting on first use.
See the [MLX precision guide](https://ml-explore.github.io/mlx/build/html/usage/precision.html)
and the [pinned environment implementation](https://github.com/ml-explore/mlx/blob/v0.31.2/mlx/utils.h).

For normal inference, start with the default performance mode and validate it
against representative prompts and your quality requirements. For comparisons
with full-float32 reference logits, or applications that need tighter numerical
agreement, launch with `MLX_ENABLE_TF32=0`:

```sh
MLX_ENABLE_TF32=0 java --enable-native-access=ALL-UNNAMED -jar application.jar
MLX_ENABLE_TF32=0 JAVA_OPTS='--enable-native-access=ALL-UNNAMED' groovy inference.groovy
```

These examples assume the application's dependencies and native runtime are
configured. An IDE, service launcher or deployment environment should set the
same environment variable for the application process. A Java or Groovy parent
process can select the mode for a new JVM with `ProcessBuilder`:

```java
ProcessBuilder launcher = new ProcessBuilder(
    "java", "--enable-native-access=ALL-UNNAMED", "-jar", "application.jar");
launcher.environment().put("MLX_ENABLE_TF32", "0");
launcher.inheritIO().start().waitFor();
```

This is a native process setting, not a jmlx per-model option. A JVM `-D` flag
or `System.setProperty("MLX_ENABLE_TF32", "0")` does not change it. Choose the
mode before starting Java/Groovy; use separate processes to compare modes or
serve workloads that need different settings.

Full float32 can cost GPU throughput. The TF32 benchmark measured a 2.73x cost
for a 2048-square float32 matmul on M5 Max, while tiny-Llama timings were similar;
that does not predict the cost for a particular production model. Use
[`jmlx-benchmarks`](../jmlx-benchmarks/README.md) with your checkpoint and workload
to evaluate the tradeoff. Record the precision setting alongside hardware,
native pin, checkpoint, prompt IDs, generation policy, batch shape and seed when
reproducibility matters. Full float32 reduces numerical discrepancies but does
not promise identical logits or generated tokens across hardware or batch shapes.

## Model test precision

`./gradlew :jmlx-models:check` automatically runs the ordinary tests with
`MLX_ENABLE_TF32=1` and the CPU reference suites in a separate JVM with
`MLX_ENABLE_TF32=0`. To run a reference suite alone, use for example:

```sh
./gradlew :jmlx-models:float32GoldenTest --tests '*GemmaModelTest'
```

New strict float32 reference tests use the JUnit tag `full-float32`; Gradle
selects their precision mode automatically. Core attention/MoE reference tests
use the same policy in `:jmlx-core:float32GoldenTest`. Batch-step equivalence
runs in both model tasks. These build settings apply to test
JVMs only; external applications choose their own launch environment.

## Errors, cancellation, and ownership

Prompt tokenization happens before native generation begins. A generated-ID decode failure is
reported as `GenerationAbortedException` with prompt/generated evidence and the failing ID; its token
event and terminal event are not emitted. Listener failures have the same contextual wrapper.
Cancellation is observed before prefill and between decode steps.

Models own checkpoint tensors in their supplied `MLXScope`. Generation owns and closes child scopes,
caches, sampler state, and incremental decoder state; callers do not manage those intermediates.
Only the cancellation token may be changed from another thread.

## Serving many requests: the batch scheduler

`BatchGenerationScheduler` is the opt-in way to serve concurrent callers. One worker thread owns
MLX, builds the model through your factory (so the weights live in a worker-owned scope), and
decodes compatible requests together as a cohort. Any thread may `submit` and `cancel`:

```java
BatchSchedulerConfig config = BatchSchedulerConfig.defaults(); // 4 rows, 16 queued
try (BatchGenerationScheduler scheduler =
    BatchGenerationScheduler.start(config, scope -> TextGenerationModels.load(scope, modelDirectory))) {
  BatchRequestHandle a = scheduler.submit(requestA, eventA -> {});
  BatchRequestHandle b = scheduler.submit(requestB, eventB -> {});
  b.cancel();
  GenerationResult result = a.stage().toCompletableFuture().join();
}
```

MLX is used from **at most one thread at a time per process**, so while a scheduler runs, do not call
the direct `generate` API or any other MLX code on another thread; `start` rejects a second running
scheduler in the same classloader. Always close the scheduler (its threads are non-daemon). Callbacks
run on the worker and must return promptly. See the `BatchGenerationScheduler` Javadoc for failure
semantics, and the `se.alipsa.jmlx.core` package Javadoc for the threading rule and its one known
exception (the `Cleaner` backstops).

Each scheduler worker carries its own scheduler stream (about 60 KB) for the process's life --
mlx-c cannot free streams -- so close-then-restart accumulates one stream per worker thread.
Keep one scheduler running rather than churning them: the per-thread stream cost also makes the
long-lived worker the recommended shape for any direct-`generate` server.

When you need a guaranteed batch shape, start with a cohort gate: `start(config, factory,
waiting -> waiting >= 2, Duration.ofSeconds(30))` holds the worker until two requests are queued
or 30 s elapse, whichever first, so both share one cohort. The gate runs on the worker under the
admission lock: keep it fast and side-effect free. See the `BatchGenerationScheduler` Javadoc for
the full contract.

## Models, downloads, and caching

`jmlx-models` loads **local directories only**; it never downloads anything, and neither do the
required PR checks. You manage the artifacts:

- **Layout.** A model directory holds `config.json`, the float `*.safetensors` shards (and their
  index), and the tokenizer files (`tokenizer.json`, optionally `tokenizer_config.json` and
  `chat_template.jinja`). Pair the checkpoint with *its own* tokenizer from the same revision; a
  mismatched tokenizer fails at the first out-of-range id or produces wrong text.
- **Fetching.** Use any tool you like (for example `huggingface-cli download <repo> --revision <sha>
  --local-dir <dir>`). Pin an exact commit `--revision`, not a branch name, and record the
  resulting file SHA-256s so a re-download can be verified.
- **Licenses.** Many checkpoints are gated or carry use restrictions; accept the license on the
  model's hub page before downloading, and follow it when redistributing. jmlx ships no weights.
- **Size and eviction.** Weights are loaded fully into unified memory (float or MLX affine-quantized; GGUF
  checkpoints are rejected), so budget roughly the safetensors size plus KV cache. Keep downloads in a
  directory you own and delete old revisions yourself; jmlx has no cache manager.
- **Native library cache.** Separately, the packaged native binaries are extracted to a per-pin cache
  directory; override it with `-Djmlx.native.cache.path=<dir>` on slow or shared storage.
- **Real-artifact tests.** The opt-in Tier-B checks that run against real downloaded models are
  listed in [`req/phase6-tier-b-artifacts.md`](../req/phase6-tier-b-artifacts.md).

## BERT encoders and classifiers

```java
try (MLXScope scope = new MLXScope()) {
  HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
  TokenizerEncoding input = tokenizer.encode(text, EncodingOptions.unbounded(true));
  TextEncoderModel encoder = TextEncoderModels.load(scope, directory);
  EncoderResult result = encoder.encode(input); // artifact pooling and Normalize defaults
  float[] embedding = result.embedding();
  // Override explicitly: encoder.encode(input, new Pooling(Pooling.Mode.MEAN, true)).
}
```

`SequenceClassifiers.load(scope, directory)` and `TokenClassifiers.load(scope, directory)` select
BertForSequenceClassification and BertForTokenClassification respectively; call `classify(input)`.
Results copy every array on construction/access. Single-label scores use softmax; explicitly
multi-label heads use sigmoid. Regression, decoder/cross-attention BERT variants, non-absolute
positions, pruned heads and quantized checkpoints are rejected. BERT input IDs, type IDs, masks,
column lengths and capacities are checked before native array creation.

Pooling supports CLS (attended index 0), MEAN and MAX plus optional L2 normalization. Mean/max
include attended special tokens and omit padding. Padded hidden-state rows are retained and use
HF's key-only mask. Supported sentence-transformers modules are Transformer, one Pooling mode,
and optional Normalize; unsafe module paths and unsupported modules fail. Without module metadata,
CLS without normalization is the default. Unlike sentence-transformers, over-limit inputs fail
rather than silently truncating: apply explicit tokenizer truncation first.

## T5 and Flan-T5

```java
try (MLXScope scope = new MLXScope()) {
  HfTokenizer tokenizer = HfTokenizer.fromDirectory(directory);
  TextGenerationModel model = TextGenerationModels.load(scope, directory, new T5LoadOptions(512));
  GenerationRequest request = GenerationRequest.text(tokenizer, "translate English to German: Hello",
      PromptSpecialTokens.ADD, GenerationConfig.greedyDefaults(16, Set.of(1)), CancellationToken.NONE);
  GenerationResult result = model.generate(request, event -> {});
}
```

Use ADD for HF-compatible source EOS insertion. OMIT is allowed as an explicit caller deviation.
The source limit defaults to 512; T5 has no absolute-position capacity. Source IDs are attended;
pretokenized requests do not infer padding. Cache capacity counts target decoding positions,
including the decoder start; the source is separately limited. The start token is absent from
results/events. Penalties count decoder start and emitted targets, never source tokens. EOS is
included; explicit non-EOS stop tokens are excluded. Sampling, log probabilities, cancellation
and listeners follow the existing generation API.

Only FULL cache retention is supported; both sliding modes and the batch scheduler reject T5
explicitly. Encoder output and per-layer cross K/V are request-owned, evaluated once, and released
at request completion/failure. Supported FFNs are ReLU and gated-GELU; tied heads rescale hidden
states, untied Flan-T5 heads do not. Beam search, training, quantized weights and decoder prefixes
are outside this release.
