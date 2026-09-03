# jmlx-models

Reference local decoder-model implementations built on jmlx. The module loads Hugging Face
safetensors checkpoints and provides inference-only Llama and Qwen2-style decoder models with a
pure-Java Hugging Face tokenizer.

It is intentionally a focused reference-model module, not a general model-serving framework. The
current API supports greedy generation; sampling, streaming, batching, additional architectures,
and RoPE scaling are planned in Phase 6. In particular, configurations declaring `rope_scaling` are
rejected, so Llama 3.1+ checkpoints are not supported yet.

## Use

Add `jmlx-models` and the native runtime companion at runtime:

```groovy
dependencies {
  implementation("se.alipsa:jmlx-models:<version>")
  runtimeOnly("se.alipsa:jmlx-native-macos-arm64:<version>")
}
```

The native artifact supports macOS on Apple Silicon. It is extracted automatically unless
`jmlx.library.path` or `JMLX_LIBRARY_PATH` points to a locally staged runtime.

Load a local Hugging Face model directory containing `config.json`, safetensors files (including
indexed shards), and `tokenizer.json`:

```java
try (MLXScope scope = new MLXScope()) {
  LlamaModel model = LlamaModel.load(scope, modelDirectory);
  HfTokenizer tokenizer = HfTokenizer.fromFile(modelDirectory.resolve("tokenizer.json"));
  String completion =
      model.generateText(tokenizer, "Hello", 64, Set.of(tokenizer.eosTokenId().orElseThrow()));
}
```

`QwenModel.load(scope, modelDirectory)` has the same shape. For chat models, render the model's
chat template through `HfTokenizer`/`ChatTemplateRenderer` before calling the overload that disables
automatic special tokens for an already-rendered prompt.

## Resource ownership

Models own checkpoint tensors in the supplied `MLXScope`. Keep that scope open for the entire model
and generation lifetime, then close it to release native memory. Generation creates and closes its
own child scopes and KV caches; callers do not need to manage those intermediates.
