# jmlx-models

Reference local decoder-model implementations built on jmlx. The module loads Hugging Face
safetensors checkpoints and provides inference-only Llama and Qwen2-style decoder models with a
pure-Java Hugging Face tokenizer.

It is intentionally a focused reference-model module, not a general model-serving framework. The
current public API supports deterministic greedy generation and explicitly seeded sampling with
synchronous token events. Batching, additional architectures, and RoPE scaling are planned in later
Phase 6 milestones. In particular, configurations declaring `rope_scaling` are rejected, so Llama 3.1+
checkpoints are not supported yet. See the [Phase 6 compatibility matrix](../req/phase6-compatibility.md)
for the precise support boundary.

## Use

Add `jmlx-models` and the native runtime companion at runtime:

```groovy
dependencies {
  implementation("se.alipsa:jmlx-models:<version>")
  runtimeOnly("se.alipsa:jmlx-native-macos-arm64:<version>")
}
```

For sampling, supply a positive temperature and explicit request-local seed. The complete policy
also supports repetition, frequency, and presence penalties plus top-k, top-p, and min-p filters:

```java
GenerationConfig sampled = new GenerationConfig(
    64, OptionalLong.of(42), 0.8f, 40, 0.95f, 0.05f,
    1.1f, 0.0f, 0.0f, Set.of(eosId), Set.of(), true);
GenerationResult result = model.generate(
    new GenerationRequest(prompt, sampled, CancellationToken.NONE),
    event -> {
      if (event.tokenId() != null) {
        System.out.printf("token=%d logp=%f%n", event.tokenId(), event.logProbability());
      }
    });
```

Selected-token log probabilities describe the final, filtered and renormalized distribution; they
do not expose alternative-token probabilities. Greedy requests may also ask for log probabilities,
which are reported as `0.0` by API convention. `textDelta` remains unsupported and null.

The native artifact supports macOS on Apple Silicon. It is extracted automatically unless
`jmlx.library.path` or `JMLX_LIBRARY_PATH` points to a locally staged runtime.

Load a local Hugging Face model directory containing `config.json`, safetensors files (including
indexed shards), and `tokenizer.json`:

```java
try (MLXScope scope = new MLXScope()) {
  TextGenerationModel model = TextGenerationModels.load(scope, modelDirectory);
  HfTokenizer tokenizer = HfTokenizer.fromFile(modelDirectory.resolve("tokenizer.json"));
  int[] prompt = tokenizer.encode("Hello", true).stream().mapToInt(Integer::intValue).toArray();
  GenerationResult result = model.generate(
      new GenerationRequest(
          prompt,
          GenerationConfig.greedyDefaults(64, Set.of(tokenizer.eosTokenId().orElseThrow())),
          CancellationToken.NONE),
      event -> {
        if (event.tokenId() != null) {
          System.out.println("generated token " + event.tokenId());
        }
      });
  System.out.println(tokenizer.decode(result.generatedTokenIds(), true));
}
```

`TextGenerationModel.metadata()` exposes architecture-neutral fields such as `modelType`, vocabulary
size, and layer count. Decoder-specific settings remain available from `DecoderModel.config()` when
working with the current Llama/Qwen implementations directly.

Byte-level BPE may split one Unicode code point across tokens, so do not decode individual event
tokens. Decode the complete generated-ID sequence as above; a tokenizer-aware streaming decoder is
planned for a later Phase 6 milestone. `LlamaModel.load` and `QwenModel.load` remain compatibility
entry points; both delegate to the common loader. For chat models, render the model's chat template
through `HfTokenizer`/`ChatTemplateRenderer`, then encode that rendered text with
`addSpecialTokens=false`: templates ordinarily include their own BOS token. The event callback runs
synchronously on the thread which owns the generation scope. A cancelling thread may only change
the `CancellationToken`; it must never touch model or native resources. Cancellation is observed
before prefill and between decode steps, not during an in-progress prompt prefill.

To reproduce sampled output, record the jmlx commit, native MLX pins, macOS/device, model/checkpoint,
prompt token IDs and batch shape, the complete generation policy, and seed. A request owns its key
chain: another request, cancellation, or process-global `MLXRandom.seed` state cannot perturb it.

## Resource ownership

Models own checkpoint tensors in the supplied `MLXScope`. Keep that scope open for the entire model
and generation lifetime, then close it to release native memory. Generation creates and closes its
own child scopes and KV caches; callers do not need to manage those intermediates.
