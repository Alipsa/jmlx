# jmlx Spring AI adapter

Java 25, Spring AI 2.0.2, macOS on Apple Silicon. Coordinate:
se.alipsa:jmlx-spring-ai:0.1.0. Publication is planned for Phase 8.4.
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
Encoder inputs are not silently truncated; the entire batch is validated on the
caller before native submission.

Streaming is cold (one generation per subscription). Text chunks contain only
new text, including the final tokenizer flush; the last response has empty
text plus usage and the native finish reason. Scheduler cancellation is an error
in both blocking and streaming calls, rather than a success with partial text. Usage counts prompt positions
and generated IDs, including a retained EOS. Cancellation signals the native
request at a safe boundary. Deltas are delivered serially off the MLX worker,
with a bounded 64-delta queue; a consumer that falls behind fails and cancels
its generation. The delivery thread waits for downstream demand without an
unbounded Reactor buffer. An independent failure signal wakes that wait on
overflow or cancellation, so no further downstream request is needed to fail
the stream and return its slot. Each demand wait, including for the terminal
response, has a 30-second timeout by default. Configure another positive timeout
with new JmlxChatModel(client, Duration). Timeout signals an error, disposes the
subscription and cancels native work. Dispose unused subscriptions promptly.

Run ./gradlew :jmlx-spring-ai:check to verify mapping and stream lifecycle
without native code. tools/release-smoke/run.sh ci --profile all additionally
verifies both frameworks against locally published Maven jars and real native
inference. Spring Boot auto-configuration, tools, media, structured output and
a Quarkus extension are outside this milestone.

Text delivery also has a client-wide admission bound equal to the scheduler's
total admission capacity. A completed native generation keeps its delivery
slot until its consumer finishes, so blocked callbacks cannot create an
unbounded number of delivery threads through repeated submissions.

Blocking chat and embedding calls wait interruptibly. Interruption cancels the
request, restores the interrupt flag and throws CancellationException. Runtime
failures are unwrapped instead of exposed as CompletionException. Blocking
chat does not start a streaming delivery thread.

Embedding batches have a maximum of 128 texts. Larger ingestion batches must
be split by the caller. Null text lists/elements are IllegalArgumentException.
Every text is validated against vocabulary/type-ID ranges, mask and CLS
requirements and sequence capacity, including special tokens,
before any native work. Valid texts run as separate worker operations so
generation can run between them. A batch keeps one scheduler admission permit
and queue slot until its last text finishes; continuation never reacquires
admission. Concurrent embedding batches are capped at max(1, maxQueuedRequests / 4),
leaving generation headroom in queues with at least four slots. Tiny queues
may have no waiting capacity left while a batch owns their only slot.
Embedding latency depends on generation load: a large batch can wait behind
many generation cohorts across its texts, even though it was admitted once.
Size ingestion jobs with that latency trade-off in mind. The shared client's embedCall method returns
an EmbeddingCall with cancel() and interruptible await(). Cancellation skips
queued/remaining texts but cannot interrupt a native forward already running.
Cancelling a queued batch removes it and returns its sequence permit, waiting
slot and admission permit immediately, so retry does not wait for a generation
cohort. A running text retains that capacity until its next safe boundary.
The first use loads encoder weights on the worker; that load temporarily
blocks generation. Failed loading is sticky for the client's lifetime; repair
the checkpoint and create a new client to retry. Sentence-only encoding pools
on the native worker and reads back only a vector.

Ownership transfers when constructing SchedulerInferenceClient. Invalid
arguments, invalid defaults (including exceeding the scheduler token limit)
or CPU encoder configuration failure close the supplied scheduler.

Sampling deliberately reuses the configured seed for every request. Identical
prompts and policies therefore produce repeatable samples. A positive
temperature too small to represent as float32 is rejected; use explicit zero
to select greedy generation.

Arbitrary user callbacks must return. Spring's own downstream-demand wait is
bounded and wakes on native failure; it cannot safely terminate user code
that blocks inside an onNext/onError callback. Such a callback keeps its delivery thread, result and admission slot even after
overflow cancels native generation. Java cannot safely stop arbitrary user
code; enough permanently blocked consumers exhaust the bounded delivery
capacity. Dispose subscriptions and make callbacks cooperate with timeouts.

The automatic JPMS module name is se.alipsa.jmlx.springai, matching the package.
Embedding responses identify the encoder by its configured directory basename.

Streaming failure is fail-fast: cancellation, overflow or a failed generation
discards queued text that has not reached the consumer. Already delivered text
remains visible, and an in-progress callback may finish. Queued deltas are not
flushed before reporting the error.
