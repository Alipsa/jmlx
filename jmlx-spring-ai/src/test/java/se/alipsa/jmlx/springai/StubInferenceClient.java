package se.alipsa.jmlx.springai;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;
import java.util.function.Consumer;
import se.alipsa.jmlx.models.EmbeddingBatch;
import se.alipsa.jmlx.models.FinishReason;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationResult;
import se.alipsa.jmlx.models.InferenceCall;
import se.alipsa.jmlx.models.InferenceClient;

final class StubInferenceClient implements InferenceClient {
  int calls;
  boolean cancelled;
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
    if (failure != null) {
      throw failure;
    }
    if (pending != null) {
      return new InferenceCall(pending, () -> cancelled = true);
    }
    delta.accept("h");
    delta.accept("é");
    return new InferenceCall(CompletableFuture.completedFuture(result()), () -> cancelled = true);
  }

  static GenerationResult result() {
    return new GenerationResult(
        List.of(1, 4, 3), List.of(5, 6), FinishReason.MAX_TOKENS, List.of(), "hé");
  }

  @Override
  public CompletionStage<EmbeddingBatch> embed(List<String> texts) {
    return CompletableFuture.completedFuture(
        new EmbeddingBatch(
            texts.stream().map(text -> new float[] {text.length(), 1}).toList(), texts.size() * 4));
  }

  @Override
  public void close() {
    throw new AssertionError("adapter must borrow the client");
  }
}
