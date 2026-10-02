package se.alipsa.jmlx.examples;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import se.alipsa.jmlx.models.BatchGenerationScheduler;
import se.alipsa.jmlx.models.BatchRequestHandle;
import se.alipsa.jmlx.models.BatchSchedulerConfig;
import se.alipsa.jmlx.models.CancellationToken;
import se.alipsa.jmlx.models.FinishReason;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationRequest;
import se.alipsa.jmlx.models.GenerationResult;
import se.alipsa.jmlx.models.TextGenerationModels;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/**
 * Opt-in batch-scheduler demo against an already-downloaded local checkpoint; nothing is fetched.
 *
 * <p>Three chat requests share one scheduler: the first streams sampled text, the second is
 * cancelled after a few tokens, and the third runs to its limit. Run with {@code ./gradlew
 * :jmlx-examples:runSchedulerExample --args="<model-dir> [max-new-tokens]"}.
 */
public final class SchedulerExample {
  private static final int DEFAULT_MAX_NEW_TOKENS = 48;
  private static final int CANCEL_AFTER_TOKENS = 5;

  private SchedulerExample() {}

  /**
   * Runs with {@code model-dir [max-new-tokens]}.
   *
   * @param args a local model directory (weights plus tokenizer) and an optional token limit
   * @throws Exception if loading or generation fails
   */
  public static void main(String[] args) throws Exception {
    if (args.length < 1 || args.length > 2) {
      throw new IllegalArgumentException("usage: SchedulerExample model-dir [max-new-tokens]");
    }
    Path modelDirectory = Path.of(args[0]);
    int maxNewTokens = args.length > 1 ? Integer.parseInt(args[1]) : DEFAULT_MAX_NEW_TOKENS;
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(modelDirectory);
    Set<Integer> eos =
        tokenizer.eosTokenId().isPresent() ? Set.of(tokenizer.eosTokenId().getAsInt()) : Set.of();

    // The worker builds the model inside a scope it owns and closes it again on close().
    try (BatchGenerationScheduler scheduler =
        BatchGenerationScheduler.start(
            BatchSchedulerConfig.defaults(),
            scope -> TextGenerationModels.load(scope, modelDirectory))) {
      BatchRequestHandle[] handles = new BatchRequestHandle[3];
      AtomicInteger secondTokens = new AtomicInteger();

      handles[0] =
          scheduler.submit(
              chat(tokenizer, "Name three colors.", sampled(maxNewTokens, 7, eos)),
              event -> {
                if (event.textDelta() != null) {
                  System.out.print(event.textDelta());
                }
              });
      handles[1] =
          scheduler.submit(
              chat(
                  tokenizer,
                  "Write a long story.",
                  GenerationConfig.greedyDefaults(maxNewTokens, eos)),
              event -> {
                // Callbacks run on the worker; cancelling only flips a flag, so this is safe.
                if (event.tokenId() != null
                    && secondTokens.incrementAndGet() == CANCEL_AFTER_TOKENS) {
                  handles[1].cancel();
                }
              });
      handles[2] =
          scheduler.submit(
              chat(tokenizer, "What is 2 + 2?", GenerationConfig.greedyDefaults(maxNewTokens, eos)),
              event -> {});

      for (int i = 0; i < handles.length; i++) {
        GenerationResult result = handles[i].stage().toCompletableFuture().join();
        FinishReason reason = result.finishReason();
        System.out.printf(
            "%nrequest %d: %d tokens, finish=%s%n", i, result.generatedTokenIds().size(), reason);
      }
    }
  }

  private static GenerationConfig sampled(int maxNewTokens, long seed, Set<Integer> eos) {
    return GenerationConfig.samplingDefaults(maxNewTokens, seed, 0.8f, eos);
  }

  private static GenerationRequest chat(
      HfTokenizer tokenizer, String user, GenerationConfig config) {
    return GenerationRequest.chat(
        tokenizer,
        List.of(Map.of("role", "user", "content", user)),
        new ChatTemplateOptions("", true, Map.of()),
        config,
        CancellationToken.NONE);
  }
}
