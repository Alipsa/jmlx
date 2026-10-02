package se.alipsa.jmlx.models;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

/** Shared helpers for the batch scheduler's native tests. */
final class SchedulerFixtures {
  static final int TIMEOUT_SECONDS = 60;

  private SchedulerFixtures() {}

  static Path checkpoint(String family) {
    return Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens")
        .resolve("checkpoints")
        .resolve(family);
  }

  static BatchGenerationScheduler start(String family, BatchSchedulerConfig config) {
    return BatchGenerationScheduler.start(
        config, root -> TextGenerationModels.load(root, checkpoint(family)));
  }

  static BatchGenerationScheduler start(
      String family, BatchSchedulerConfig config, BatchGenerationScheduler.Hooks hooks) {
    return BatchGenerationScheduler.start(
        config, root -> TextGenerationModels.load(root, checkpoint(family)), hooks);
  }

  /**
   * Hooks that hold the worker until {@code n} requests are waiting, so exactly those batch. The
   * gate is one-shot: once it opens it stays open, so a later single request is not held forever.
   */
  static BatchGenerationScheduler.Hooks gated(int n) {
    AtomicBoolean opened = new AtomicBoolean();
    return BatchGenerationScheduler.Hooks.NONE.withCohortGate(
        waiting -> opened.get() || (waiting >= n && opened.compareAndSet(false, true)));
  }

  static BatchSchedulerConfig config(int batch, int queued) {
    return new BatchSchedulerConfig(batch, queued, 4096, 256);
  }

  static GenerationRequest greedy(int[] prompt, int maxNewTokens) {
    return greedy(prompt, maxNewTokens, Set.of(), Set.of());
  }

  static GenerationRequest greedy(
      int[] prompt, int maxNewTokens, Set<Integer> eos, Set<Integer> stop) {
    return new GenerationRequest(
        prompt, GenerationConfig.greedyDefaults(maxNewTokens, eos, stop), CancellationToken.NONE);
  }

  static GenerationRequest sampled(int[] prompt, int maxNewTokens, long seed) {
    return new GenerationRequest(
        prompt,
        GenerationConfig.samplingDefaults(maxNewTokens, seed, 0.9f, Set.of()),
        CancellationToken.NONE);
  }

  static GenerationResult await(BatchRequestHandle handle) throws Exception {
    try {
      return handle.stage().toCompletableFuture().get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
    } catch (ExecutionException e) {
      throw e.getCause() instanceof Exception cause ? cause : e;
    }
  }

  /** The exception a failed stage actually carries, unwrapped from the stage's wrapper. */
  static Throwable failureOf(BatchRequestHandle handle) throws InterruptedException {
    try {
      handle.stage().toCompletableFuture().get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
      return null;
    } catch (ExecutionException e) {
      return e.getCause();
    } catch (TimeoutException e) {
      throw new AssertionError("stage did not complete", e);
    } catch (CompletionException e) {
      return e.getCause();
    }
  }

  static List<Integer> tokens(GenerationResult result) {
    return new ArrayList<>(result.generatedTokenIds());
  }
}
