package smoke;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.springframework.ai.chat.prompt.Prompt;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.models.BatchGenerationScheduler;
import se.alipsa.jmlx.models.BatchSchedulerConfig;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.Pooling;
import se.alipsa.jmlx.models.SchedulerInferenceClient;
import se.alipsa.jmlx.models.TextGenerationModels;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/** Public-framework integration against Maven artifacts, never project class directories. */
final class AdapterSmoke {
  private AdapterSmoke() {}

  static void run(Path decoder, Path encoder) throws Exception {
    long before = settledActive();
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(decoder);
    try (SchedulerInferenceClient client =
        new SchedulerInferenceClient(
            BatchGenerationScheduler.start(
                BatchSchedulerConfig.defaults(), root -> TextGenerationModels.load(root, decoder)),
            tokenizer,
            "tiny-mistral",
            GenerationConfig.greedyDefaults(6, Set.of()),
            encoder,
            new Pooling(Pooling.Mode.MEAN, true))) {
      org.springframework.ai.chat.model.ChatModel spring =
          new se.alipsa.jmlx.springai.JmlxChatModel(client);
      org.springframework.ai.embedding.EmbeddingModel springEmbedding =
          new se.alipsa.jmlx.springai.JmlxEmbeddingModel(client);
      dev.langchain4j.model.chat.ChatModel langchain =
          new se.alipsa.jmlx.langchain4j.JmlxChatModel(client);
      dev.langchain4j.model.chat.StreamingChatModel streaming =
          new se.alipsa.jmlx.langchain4j.JmlxStreamingChatModel(client);
      dev.langchain4j.model.embedding.EmbeddingModel langchainEmbedding =
          new se.alipsa.jmlx.langchain4j.JmlxEmbeddingModel(client);
      Prompt prompt = new Prompt("the fox and the hen");
      ChatRequest request =
          ChatRequest.builder().messages(UserMessage.from("the fox and the hen")).build();
      // Concurrent framework access from virtual threads, including encoder loading alongside chat.
      try (var callers = java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()) {
        var springCall = callers.submit(() -> spring.call(prompt));
        var langchainCall = callers.submit(() -> langchain.chat(request));
        var embeddings =
            callers.submit(() -> springEmbedding.embed(List.of("hello world", "another sentence")));
        var a = springCall.get(60, TimeUnit.SECONDS);
        var b = langchainCall.get(60, TimeUnit.SECONDS);
        require(
            a.getResult().getOutput().getText().equals(b.aiMessage().text()),
            "framework chat mismatch");
        require(a.getMetadata().getUsage().getCompletionTokens() == 6, "Spring usage mismatch");
        require(b.tokenUsage().outputTokenCount() == 6, "LangChain4j usage mismatch");
        StringBuilder springText = new StringBuilder();
        var chunks = spring.stream(prompt).collectList().block(java.time.Duration.ofSeconds(60));
        require(chunks != null && chunks.size() > 1, "Spring stream did not deliver incrementally");
        chunks.forEach(chunk -> springText.append(chunk.getResult().getOutput().getText()));
        require(
            springText.toString().equals(b.aiMessage().text()),
            "Spring delta/terminal duplication");
        require(
            chunks.getLast().getMetadata().getUsage().getCompletionTokens() == 6,
            "stream usage mismatch");
        CompletableFuture<ChatResponse> complete = new CompletableFuture<>();
        StringBuilder langchainText = new StringBuilder();
        streaming.chat(
            request,
            new StreamingChatResponseHandler() {
              @Override
              public void onPartialResponse(String text) {
                langchainText.append(text);
              }

              @Override
              public void onCompleteResponse(ChatResponse response) {
                complete.complete(response);
              }

              @Override
              public void onError(Throwable failure) {
                complete.completeExceptionally(failure);
              }
            });
        require(
            complete.get(60, TimeUnit.SECONDS).aiMessage().text().equals(langchainText.toString()),
            "LangChain4j delta/terminal mismatch");
        List<float[]> vectors = embeddings.get(60, TimeUnit.SECONDS);
        var other =
            langchainEmbedding.embedAll(
                List.of(
                    dev.langchain4j.data.segment.TextSegment.from("hello world"),
                    dev.langchain4j.data.segment.TextSegment.from("another sentence")));
        require(other.tokenUsage().inputTokenCount() == 8, "embedding token count mismatch");
        require(vectors.size() == 2 && vectors.getFirst().length == 16, "embedding shape mismatch");
        for (int i = 0; i < vectors.size(); i++) {
          require(
              java.util.Arrays.equals(vectors.get(i), other.content().get(i).vector()),
              "embedding order or values mismatch");
          double norm = 0;
          for (float value : vectors.get(i)) {
            require(Float.isFinite(value), "nonfinite embedding");
            norm += value * value;
          }
          require(Math.abs(norm - 1) < 1e-4, "embedding was not normalized");
        }
        require(
            !java.util.Arrays.equals(vectors.get(0), vectors.get(1)),
            "different texts have identical vectors");
      }
      // LangChain4j cancellation must silence every subsequent callback, including completion.
      CompletableFuture<Void> cancelled = new CompletableFuture<>();
      java.util.concurrent.atomic.AtomicInteger callbacks =
          new java.util.concurrent.atomic.AtomicInteger();
      streaming.chat(
          request,
          new StreamingChatResponseHandler() {
            @Override
            public void onPartialResponse(
                PartialResponse response, PartialResponseContext context) {
              callbacks.incrementAndGet();
              context.streamingHandle().cancel();
              cancelled.complete(null);
            }

            @Override
            public void onCompleteResponse(ChatResponse response) {
              callbacks.incrementAndGet();
            }

            @Override
            public void onError(Throwable error) {
              cancelled.completeExceptionally(error);
            }
          });
      cancelled.get(60, TimeUnit.SECONDS);
      // A later request drains through the same worker; cancellation cannot poison it.
      langchain.chat(request);
      require(callbacks.get() == 1, "callback after streaming cancellation");
    }
    long after = settledActive();
    require(
        after == before,
        "adapter close retained native bytes: before=" + before + ", after=" + after);
    System.out.println(
        "adapter smoke OK: chat, streaming, embeddings, cancellation, virtual threads and close");
  }

  private static long settledActive() throws InterruptedException {
    long value = MLXMemory.activeBytes();
    for (int i = 0; i < 20; i++) {
      Thread.sleep(5);
      long next = MLXMemory.activeBytes();
      if (next == value) {
        return next;
      }
      value = next;
    }
    return value;
  }

  private static void require(boolean condition, String message) {
    if (!condition) {
      throw new AssertionError(message);
    }
  }
}
