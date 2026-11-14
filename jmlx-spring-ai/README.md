# jmlx Spring AI adapter

Java 25, Spring AI 2.0.2, macOS on Apple Silicon. Development coordinate:
se.alipsa:jmlx-spring-ai:0.1.0-SNAPSHOT. Publication is planned for Phase 8.4.
The native runtime jar remains an explicit runtime dependency.

Create one scheduler and one shared client for the application:

~~~java
Path decoderDirectory = Path.of("/models/decoder");
Path encoderDirectory = Path.of("/models/sentence-bert");
HfTokenizer tokenizer = HfTokenizer.fromDirectory(decoderDirectory);
BatchGenerationScheduler scheduler = BatchGenerationScheduler.start(
    BatchSchedulerConfig.defaults(),
    root -> TextGenerationModels.load(root, decoderDirectory));
try (SchedulerInferenceClient client = new SchedulerInferenceClient(
    scheduler, tokenizer, "local-decoder",
    GenerationConfig.greedyDefaults(128, Set.of(/* checkpoint EOS IDs */)),
    encoderDirectory, null)) {
  ChatModel chat = new JmlxChatModel(client);
  EmbeddingModel embeddings = new JmlxEmbeddingModel(client);
  String answer = chat.call(new Prompt("Hello")).getResult().getOutput().getText();
  Flux<ChatResponse> stream = chat.stream(new Prompt("Hello"));
  float[] vector = embeddings.embed("Hello");
}
~~~

Use se.alipsa.jmlx.models.*, se.alipsa.jmlx.tokenizer.HfTokenizer and
se.alipsa.jmlx.springai.* imports, together with Spring AI's chat/embedding
interfaces and Reactor's Flux. The example's encoder must be a supported BERT
sentence encoder, not the decoder's token embedding table. Null pooling uses
the encoder artifact's pooling/normalization defaults; an explicit Pooling
overrides them. A null encoder directory disables embeddings.

Both views borrow the client. Close the client once all consumers have
finished; it owns the scheduler and both models' native scopes. Creating the
tokenizer before starting the scheduler avoids leaving a live worker if
tokenizer loading fails. Do not make direct MLX calls while the client runs.
Framework calls can originate on virtual threads; all native work stays on the
scheduler worker. Encoder work runs between generation cohorts, so an encoder
request can wait behind a long generation cohort.

Chat accepts ordered system/user/assistant text messages and uses the
checkpoint's chat template with add_generation_prompt=true. Requests support
max tokens, temperature, top-k/top-p and frequency/presence penalties. Sampling
requires a seed in the client's configured GenerationConfig; greedy is the
default in this example. Stop strings, tool calling and media are rejected.
Model-name overrides must match the configured client name. Embedding model
and dimension overrides are rejected; use a separate encoder configuration.
Encoder inputs are not silently truncated.

Streaming is cold (one generation per subscription). Text chunks contain only
new text, including the final tokenizer flush; the last response has empty
text plus usage and the native finish reason. Usage counts prompt positions
and generated IDs, including a retained EOS. Cancellation signals the native
request at a safe boundary. Deltas are delivered serially off the MLX worker,
with a bounded 64-delta queue; a consumer that falls behind fails and cancels
its generation. Reactor rejects emission without downstream demand rather
than buffering indefinitely. Dispose of unused subscriptions.

Run ./gradlew :jmlx-spring-ai:check to verify mapping and stream lifecycle
without native code. tools/release-smoke/run.sh ci --profile all additionally
verifies both frameworks against locally published Maven jars and real native
inference. Spring Boot auto-configuration, tools, media, structured output and
a Quarkus extension are outside this milestone.

Text delivery also has a client-wide admission bound equal to the scheduler's
total admission capacity. A completed native generation keeps its delivery
slot until its consumer finishes, so blocked callbacks cannot create an
unbounded number of delivery threads through repeated submissions.
