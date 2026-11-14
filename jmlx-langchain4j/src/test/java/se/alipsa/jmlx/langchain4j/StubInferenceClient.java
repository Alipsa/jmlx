package se.alipsa.jmlx.langchain4j;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import se.alipsa.jmlx.models.EmbeddingBatch;
import se.alipsa.jmlx.models.EmbeddingCall;
import se.alipsa.jmlx.models.FinishReason;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationResult;
import se.alipsa.jmlx.models.InferenceCall;
import se.alipsa.jmlx.models.InferenceClient;

final class StubInferenceClient implements InferenceClient {
  int calls;
  final java.util.concurrent.CountDownLatch called = new java.util.concurrent.CountDownLatch(1);
  CompletableFuture<EmbeddingBatch> embeddingPending;
  volatile boolean cancelled;
  final java.util.concurrent.CountDownLatch cancelledSignal =
      new java.util.concurrent.CountDownLatch(1);
  List<Map<String, Object>> messages;
  GenerationConfig policy;
  CompletableFuture<GenerationResult> pending;
  Consumer<String> listener;
  RuntimeException failure;

  @Override
  public String modelName() {
    return "tiny";
  }

  @Override
  public String embeddingModelName() {
    return "tiny-bert";
  }

  @Override
  public GenerationConfig defaults() {
    return GenerationConfig.greedyDefaults(8, Set.of(2));
  }

  @Override
  public InferenceCall chat(
      List<Map<String, Object>> input, GenerationConfig config, Consumer<String> delta) {
    calls++;
    messages = input;
    policy = config;
    listener = delta;
    called.countDown();
    if (failure != null) {
      throw failure;
    }
    if (pending != null) {
      return new InferenceCall(
          pending,
          () -> {
            cancelled = true;
            cancelledSignal.countDown();
          });
    }
    delta.accept("h");
    delta.accept("é");
    return new InferenceCall(
        CompletableFuture.completedFuture(result()),
        () -> {
          cancelled = true;
          cancelledSignal.countDown();
        });
  }

  static GenerationResult result() {
    return new GenerationResult(
        List.of(1, 4, 3), List.of(5, 6), FinishReason.MAX_TOKENS, List.of(), "hé");
  }

  @Override
  public EmbeddingCall embedCall(List<String> texts) {
    called.countDown();
    return new EmbeddingCall(
        embeddingPending == null
            ? CompletableFuture.completedFuture(
                new EmbeddingBatch(
                    texts.stream().map(text -> new float[] {text.length(), 1}).toList(),
                    texts.size() * 4))
            : embeddingPending,
        () -> {
          cancelled = true;
          cancelledSignal.countDown();
        });
  }

  @Override
  public void close() {
    throw new AssertionError("adapter must borrow the client");
  }
}
