package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/**
 * Benchmark-only generation/step adapters for the Phase 8.2 decode baseline
 * (req/plans/phase8-2-plan.md sections 3.2/3.3/4; frozen report contract in
 * req/plans/phase8-2-reference-findings.md section 13.3).
 *
 * <p>Package-private and test-source only: the controlled and fixed-continuation adapters drive the
 * model's own package-private seam entries ({@link DecoderModel#stepLogits} prefill, {@link
 * DecoderModel#decode} single-row steps), which already implement the own-head rule ({@code
 * tiedOutput ? embedding.project : lmHead.forward}) without ever substituting or reconstructing a
 * head. The stock adapter is the public {@link DecoderModel#generate(GenerationRequest,
 * java.util.function.Consumer)} path with the checkpoint's own tokenizer attached through {@link
 * GenerationRequest#pretokenized}; no production public API is added for timing.
 *
 * <p>Every timer is monotonic and starts and stops at the frozen boundaries (section 13.3): the
 * timers start after the per-request prompt caches are created; the prefill timer stops at the
 * evaluated prefill logits; each cached-step timer stops at the evaluated step logits, and the
 * argmax (and, for fixed continuation, the full-vocabulary readback and sort) run outside the
 * measured intervals. The controlled loop has no next-token lookahead, no async pipeline, no
 * periodic cache clearing and no stop termination: all {@code emitted} tokens are produced,
 * stop-set members included.
 */
final class BaselineAdapters {
  private static final double NANOS_PER_SECOND = 1_000_000_000d;

  private BaselineAdapters() {}

  /** One cached-step measurement; the fixed-continuation fields are null in headline runs. */
  record Step(int position, double s, Integer argmaxId, Double top2Margin) {}

  /** Controlled headline / reference-pair request (frozen section 13.3 boundaries). */
  record ControlledResult(
      List<Integer> tokens, double ttftS, double prefillS, List<Step> steps, double[] eventS) {}

  /** Fixed-continuation request: no sampling, no emission, per-step argmax/margin recorded. */
  record FixedContinuationResult(double prefillS, List<Step> steps, double stepRegionS) {}

  /** Stock (public generate) headline / reference-pair request. */
  record StockResult(
      List<Integer> tokens,
      double[] eventS,
      double ttftS,
      double fullGenerationS,
      FinishReason finishReason) {}

  /**
   * Observes the adapter's phase-local work windows (plan section 5.1; findings section 15). Each
   * window names the phase and the 1-based position of the token the phase acts on (0 for the
   * prefill phases and the stock whole-request window) and runs that phase's work between the
   * observer's before/after hooks. The per-scope release work is observed too: each activation
   * scope's {@code close} runs in its own window ({@code prefill-release} at position 0, {@code
   * release} at the step's position), and the request scope's {@code close} in the {@code
   * request-release} window at position 0 — the estimator charges their downcall counts (the
   * per-array {@code mlx_array_free} of a scope close) instead of dropping them. The windows never
   * change the work, the scope lifetimes or the frozen timer boundaries: an observing run produces
   * the identical tokens and the same frozen timings as a {@link #NONE} run.
   */
  @FunctionalInterface
  interface PhaseObserver {
    /** Runs one phase's work; implementations measure only around {@code work.run()}. */
    void window(String phase, int position, ModelWork work);

    /**
     * The work inside one phase window. Unchecked exceptions (for example {@code MLXException})
     * propagate to the adapter.
     */
    @FunctionalInterface
    interface ModelWork {
      void run();
    }

    /** The no-op observer the adapters use when no diagnostics run. */
    PhaseObserver NONE = (phase, position, work) -> work.run();
  }

  /** One fresh FULL-retention cache per decoder layer, owned by {@code scope}. */
  static List<KVCache> freshCaches(DecoderModel model, MLXScope scope) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.metadata().numHiddenLayers(); i++) {
      caches.add(new KVCache(scope));
    }
    return caches;
  }

  /**
   * The controlled loop: one full-prompt forward with last-position-only projection through the
   * model's own output head, then exactly {@code emitted - 1} cached one-token steps on a fresh
   * per-request cache. All {@code emitted} tokens are produced regardless of stop-set membership
   * (finish {@code controlled-max-tokens}); token 1 comes from the prefill projection.
   *
   * <p>Every step's required logits and cache state are evaluated before its timer stops; each
   * token's {@code eventS} entry (seconds from the timer origin, token 1 included) is taken after
   * that token's argmax, matching the pinned Python controlled loop.
   */
  static ControlledResult runControlled(
      DecoderModel model, MLXScope owner, int[] prompt, int emitted) {
    return runControlled(model, owner, prompt, emitted, PhaseObserver.NONE);
  }

  static ControlledResult runControlled(
      DecoderModel model, MLXScope owner, int[] prompt, int emitted, PhaseObserver observer) {
    if (emitted < 1) {
      throw new IllegalArgumentException("emitted must be >= 1");
    }
    MLXScope request = owner.newChild();
    ControlledResult result;
    try {
      List<KVCache> caches = freshCaches(model, request);
      // The prompt caches are created before the timer starts (frozen section 13.3 boundary);
      // t0 is also the event-time origin, as in the pinned Python loop.
      long t0 = System.nanoTime();
      int promptLength = prompt.length;
      double prefillS;
      int token1;
      MLXScope activation = request.newChild();
      try {
        final MLXArray[] logitsRef = new MLXArray[1];
        final int[] token1Ref = new int[1];
        observer.window(
            "prefill-construction",
            0,
            () -> {
              MLXArray ids = MLX.array(activation, prompt, new int[] {1, promptLength});
              logitsRef[0] = model.stepLogits(ids, caches, new int[] {promptLength}, null);
            });
        observer.window("prefill-sync", 0, () -> evalStep(logitsRef[0], caches));
        prefillS = (System.nanoTime() - t0) / NANOS_PER_SECOND;
        // Token 1's event time is taken after its argmax, like every later token's; the
        // prefill timer above already stopped at the evaluated logits.
        observer.window("prefill-selection", 0, () -> token1Ref[0] = argmax(logitsRef[0]));
        token1 = token1Ref[0];
      } catch (Throwable t) {
        releaseWindow(observer, "prefill-release", 0, activation, t);
        throw t;
      }
      // The release of the prefill's activation scope is observed work too (its per-array
      // mlx_array_free downcalls); the close itself runs at the same wall-clock position it
      // always did, outside every frozen timer.
      releaseWindow(observer, "prefill-release", 0, activation, null);
      double[] eventS = new double[emitted];
      eventS[0] = (System.nanoTime() - t0) / NANOS_PER_SECOND;
      List<Integer> tokens = new ArrayList<>(emitted);
      tokens.add(token1);
      List<Step> steps = new ArrayList<>(emitted - 1);
      for (int k = 0; k < emitted - 1; k++) {
        int position = promptLength + k + 1;
        long stepStart = System.nanoTime();
        double stepS;
        int next;
        final int lastToken = tokens.get(tokens.size() - 1);
        MLXScope stepActivation = request.newChild();
        try {
          final MLXArray[] logitsRef = new MLXArray[1];
          final int[] nextRef = new int[1];
          observer.window(
              "construction",
              position,
              () -> {
                MLXArray ids = MLX.array(stepActivation, new int[] {lastToken}, new int[] {1, 1});
                logitsRef[0] = model.decode(ids, caches, null, null);
              });
          observer.window("sync", position, () -> evalStep(logitsRef[0], caches));
          // The per-step timer stops at the evaluated logits, before the argmax (frozen
          // section 13.3 boundary); the argmax host work runs outside the measured interval.
          stepS = (System.nanoTime() - stepStart) / NANOS_PER_SECOND;
          observer.window("selection", position, () -> nextRef[0] = argmax(logitsRef[0]));
          next = nextRef[0];
        } catch (Throwable t) {
          releaseWindow(observer, "release", position, stepActivation, t);
          throw t;
        }
        releaseWindow(observer, "release", position, stepActivation, null);
        steps.add(new Step(position, stepS, null, null));
        tokens.add(next);
        eventS[k + 1] = (System.nanoTime() - t0) / NANOS_PER_SECOND;
      }
      result =
          new ControlledResult(
              List.copyOf(tokens), eventS[0], prefillS, List.copyOf(steps), eventS);
    } catch (Throwable t) {
      releaseWindow(observer, "request-release", 0, request, t);
      throw t;
    }
    // The request scope owns the per-request caches; its close (and their per-array frees)
    // is the request-release window, after every frozen timer has stopped.
    releaseWindow(observer, "request-release", 0, request, null);
    return result;
  }

  /**
   * The fixed-continuation workload (plan section 3.3): the given continuation IDs are fed one per
   * step through the model's own seam path with identical context growth, each step recording the
   * 1-based position of the fed token (the cache length after the step), its seconds, the greedy
   * argmax ID and the top-2 logit margin. No sampling and no emission. {@code stepRegionS} is the
   * wall time over the step region only (from the prefill evaluation to the end of the final step's
   * host work); it includes the per-step argmax/sort host work, so it is not decode time (the
   * per-step {@code s} values are the decode comparison; findings section 13.3).
   */
  static FixedContinuationResult runFixedContinuation(
      DecoderModel model, MLXScope owner, int[] prompt, int[] fedIds) {
    return runFixedContinuation(model, owner, prompt, fedIds, PhaseObserver.NONE);
  }

  static FixedContinuationResult runFixedContinuation(
      DecoderModel model, MLXScope owner, int[] prompt, int[] fedIds, PhaseObserver observer) {
    MLXScope request = owner.newChild();
    List<Step> steps;
    double stepRegionS;
    double prefillS;
    try {
      List<KVCache> caches = freshCaches(model, request);
      // The prompt caches are created before the timer starts (frozen section 13.3 boundary).
      long t0 = System.nanoTime();
      int promptLength = prompt.length;
      MLXScope activation = request.newChild();
      try {
        final MLXArray[] logitsRef = new MLXArray[1];
        observer.window(
            "prefill-construction",
            0,
            () -> {
              MLXArray ids = MLX.array(activation, prompt, new int[] {1, promptLength});
              logitsRef[0] = model.stepLogits(ids, caches, new int[] {promptLength}, null);
            });
        observer.window("prefill-sync", 0, () -> evalStep(logitsRef[0], caches));
        // The prefill timer stops at the evaluated prefill logits, before the activation
        // scope's release (the frozen section 13.3 boundary; the controlled loop does the
        // same). The release window's own overhead (and its downcalls in binding-count
        // mode) never enter prefill_s.
        prefillS = (System.nanoTime() - t0) / NANOS_PER_SECOND;
      } catch (Throwable t) {
        releaseWindow(observer, "prefill-release", 0, activation, t);
        throw t;
      }
      // Same release-window accounting as the controlled loop (findings section 15.6).
      releaseWindow(observer, "prefill-release", 0, activation, null);
      steps = new ArrayList<>(fedIds.length);
      for (int k = 0; k < fedIds.length; k++) {
        int position = promptLength + k + 1;
        long stepStart = System.nanoTime();
        double stepS;
        int argmaxId;
        double top2Margin;
        final int fedId = fedIds[k];
        MLXScope stepActivation = request.newChild();
        try {
          final MLXArray[] logitsRef = new MLXArray[1];
          final int[] argmaxIdRef = new int[1];
          final double[] marginRef = new double[1];
          observer.window(
              "construction",
              position,
              () -> {
                MLXArray ids = MLX.array(stepActivation, new int[] {fedId}, new int[] {1, 1});
                logitsRef[0] = model.decode(ids, caches, null, null);
              });
          observer.window("sync", position, () -> evalStep(logitsRef[0], caches));
          // The step timer stops at the evaluated logits, before the argmax and the
          // full-vocabulary readback and sort (the cross-adapter check), exactly as in the
          // pinned Python loop.
          stepS = (System.nanoTime() - stepStart) / NANOS_PER_SECOND;
          observer.window(
              "selection",
              position,
              () -> {
                argmaxIdRef[0] = argmax(logitsRef[0]);
                marginRef[0] = top2Margin(logitsRef[0]);
              });
          argmaxId = argmaxIdRef[0];
          top2Margin = marginRef[0];
        } catch (Throwable t) {
          releaseWindow(observer, "release", position, stepActivation, t);
          throw t;
        }
        releaseWindow(observer, "release", position, stepActivation, null);
        steps.add(new Step(position, stepS, argmaxId, top2Margin));
      }
      // Wall time over the step region only (prefill excluded, one definition on both paths).
      stepRegionS = (System.nanoTime() - t0) / NANOS_PER_SECOND - prefillS;
    } catch (Throwable t) {
      releaseWindow(observer, "request-release", 0, request, t);
      throw t;
    }
    releaseWindow(observer, "request-release", 0, request, null);
    return new FixedContinuationResult(prefillS, List.copyOf(steps), stepRegionS);
  }

  /**
   * The stock Java path: the public {@code generate} with pretokenized prompt IDs, the checkpoint's
   * own tokenizer (so the loop's incremental detokenization runs and is timed), full retention and
   * the frozen greedy policy. Host-visible token events are observed without inserting per-token
   * synchronizations; the synchronous loop evaluates every step at its selection boundary and
   * queues no lookahead, so nothing is pending when the call returns — there is no drain to run,
   * and the full-generation timer stops at the return.
   */
  static StockResult runStock(
      DecoderModel model,
      HfTokenizer tokenizer,
      int[] prompt,
      int maxTokens,
      Set<Integer> eosTokenIds) {
    return runStock(model, tokenizer, prompt, maxTokens, eosTokenIds, PhaseObserver.NONE);
  }

  static StockResult runStock(
      DecoderModel model,
      HfTokenizer tokenizer,
      int[] prompt,
      int maxTokens,
      Set<Integer> eosTokenIds,
      PhaseObserver observer) {
    GenerationRequest request =
        GenerationRequest.pretokenized(
            tokenizer,
            prompt,
            GenerationConfig.greedyDefaults(maxTokens, eosTokenIds),
            CancellationToken.NONE);
    long t0 = System.nanoTime();
    List<Integer> tokens = new ArrayList<>();
    List<Double> eventRel = new ArrayList<>();
    final GenerationResult[] resultRef = new GenerationResult[1];
    // The stock path has no phase split (a documented limitation: plan section 5.1 is
    // controlled-first) — one whole-request window around the public generate call.
    observer.window(
        "request",
        0,
        () ->
            resultRef[0] =
                model.generate(
                    request,
                    event -> {
                      if (event.tokenId() != null) {
                        eventRel.add((System.nanoTime() - t0) / NANOS_PER_SECOND);
                        tokens.add(event.tokenId());
                      }
                    }));
    // No drain: the Java loop evaluates every step at its selection boundary and queues no
    // lookahead forward, so all generated work is evaluated by the time generate returns.
    double[] eventS = new double[eventRel.size()];
    for (int i = 0; i < eventS.length; i++) {
      eventS[i] = eventRel.get(i);
    }
    return new StockResult(
        List.copyOf(tokens),
        eventS,
        eventS.length > 0 ? eventS[0] : 0.0,
        (System.nanoTime() - t0) / NANOS_PER_SECOND,
        resultRef[0].finishReason());
  }

  /**
   * Evaluates a step's lazy logits together with the cache arrays the step advanced — the required
   * logits/cache state at the frozen timer boundary (section 13.3).
   */
  private static void evalStep(MLXArray logits, List<KVCache> caches) {
    MLXArray[] cacheArrays = DecoderModel.stepCacheArrays(caches);
    MLXArray[] all = new MLXArray[cacheArrays.length + 1];
    all[0] = logits;
    System.arraycopy(cacheArrays, 0, all, 1, cacheArrays.length);
    MLX.eval(all);
  }

  /**
   * Runs one scope's release window while preserving the exception in flight (if any): the close
   * still runs at the same wall-clock position the try-with-resources it replaced gave it, but a
   * release-window failure is attached to {@code original} as a suppressed exception instead of
   * replacing it — try-with-resources used to do the attaching (review round 2, D). With no
   * original exception, a release-window failure propagates, as before.
   */
  private static void releaseWindow(
      PhaseObserver observer, String phase, int position, MLXScope scope, Throwable original) {
    try {
      observer.window(phase, position, scope::close);
    } catch (RuntimeException releaseFailure) {
      if (original == null) {
        throw releaseFailure;
      }
      original.addSuppressed(releaseFailure);
    }
  }

  /** Native greedy argmax of the {@code [1,1,V]} row, read back to the host. */
  private static int argmax(MLXArray logits) {
    MLXArray selected = MLXOps.argmaxAxis(logits, 2, false);
    MLX.eval(selected);
    return selected.toIntArray()[0];
  }

  /** Margin between the two largest logits of the flattened row (first-maximum tie-break). */
  private static double top2Margin(MLXArray logits) {
    float[] row = logits.toFloatArray();
    Arrays.sort(row);
    return (double) row[row.length - 1] - row[row.length - 2];
  }
}
