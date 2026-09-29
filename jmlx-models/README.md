# jmlx-models

Reference local decoder-model implementations built on jmlx. The module loads Hugging Face
safetensors checkpoints and provides inference-only Llama and Qwen2-style decoder models with a
pure-Java Hugging Face tokenizer. The native runtime supports macOS on Apple Silicon.

The API supports greedy generation and explicitly seeded sampling, synchronous token/text events,
raw-text and configured-chat requests, penalties, and top-k/top-p/min-p filtering. Additional model
architectures, quantization, RoPE scaling, and serving infrastructure remain later Phase 6 work; see
the [compatibility matrix](../req/phase6-compatibility.md).

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

## Errors, cancellation, and ownership

Prompt tokenization happens before native generation begins. A generated-ID decode failure is
reported as `GenerationAbortedException` with prompt/generated evidence and the failing ID; its token
event and terminal event are not emitted. Listener failures have the same contextual wrapper.
Cancellation is observed before prefill and between decode steps.

Models own checkpoint tensors in their supplied `MLXScope`. Generation owns and closes child scopes,
caches, sampler state, and incremental decoder state; callers do not manage those intermediates.
Only the cancellation token may be changed from another thread.
