# jmlx LangChain4j adapter

Java 25, LangChain4j 1.23.0, macOS on Apple Silicon. Development coordinate:
se.alipsa:jmlx-langchain4j:0.1.0-SNAPSHOT. Publication is planned for Phase 8.4.
The native runtime jar remains an explicit runtime dependency.

Configure a shared SchedulerInferenceClient as in the
[Spring AI example](../jmlx-spring-ai/README.md), then borrow these views:

~~~java
ChatModel chat = new JmlxChatModel(client);
StreamingChatModel streaming = new JmlxStreamingChatModel(client);
EmbeddingModel embeddings = new JmlxEmbeddingModel(client);

String answer = chat.chat("Hello");
float[] vector = embeddings.embed("Hello").content().vector();
streaming.chat("Hello", new StreamingChatResponseHandler() {
  @Override
  public void onPartialResponse(PartialResponse text, PartialResponseContext context) {
    System.out.print(text.text());
    // context.streamingHandle().cancel() stops callbacks and requests cancellation.
  }
  @Override
  public void onCompleteResponse(ChatResponse response) {
    System.out.println(response.tokenUsage());
  }
  @Override
  public void onError(Throwable failure) {
    failure.printStackTrace();
  }
});
~~~

Import adapter classes from se.alipsa.jmlx.langchain4j, interfaces from
dev.langchain4j.model.chat / .embedding, and response types from
dev.langchain4j.model.chat.response. Close the shared client only after
streaming finishes. These views never close a borrowed client and can share
one client with Spring AI.

Chat supports unnamed system/user/assistant text messages and uses the
checkpoint's chat template. Max output tokens, temperature, top-k/top-p,
frequency and presence penalties map to jmlx. Sampling requires an explicit
seed in the client's defaults. A model-name override must match its configured
name. Tool calls/results, reasoning history, media, response formats and stop
strings are rejected. EOS/stop-token maps to STOP, token limit maps to LENGTH;
runtime cancellation without explicit stream cancellation is an error.
Usage includes special prompt tokens and retained generated EOS IDs.

Streaming callbacks run serially off the MLX worker. Cancellation through a
partial-response context silences further callbacks, including completion and
error, and cancels the native request at a safe boundary. Callback failures
cancel that request. Slow consumers have at most 64 queued deltas before the
request fails; they do not block MLX or other requests.

Embeddings use a separate supported BERT sentence encoder on the same worker,
retain input order and report encoded input-token usage. Pooling and L2
normalization belong to the shared client's encoder configuration.
Inputs are not silently truncated.

Run ./gradlew :jmlx-langchain4j:check to verify mappings and cancellation
without native code. tools/release-smoke/run.sh ci --profile all runs the
published-artifact native integration. This module provides LangChain4j
interfaces; Quarkus extension wiring and version compatibility are not yet
verified.

Text delivery also has a client-wide admission bound equal to the scheduler's
total admission capacity. A completed native generation keeps its delivery
slot until its consumer finishes, so blocked callbacks cannot create an
unbounded number of delivery threads through repeated submissions.
