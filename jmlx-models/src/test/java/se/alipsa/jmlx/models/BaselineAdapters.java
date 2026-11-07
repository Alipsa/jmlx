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
 * <p>Every timer is monotonic and stops at the frozen boundaries: the prefill timer stops at the
 * evaluated prefill logits; each cached-step timer stops at the evaluated step logits, and the
 * argmax (and, for fixed continuation, the full-vocabulary readback and sort) run outside the
 * measured interval. The controlled loop has no next-token lookahead, no async pipeline, no
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
   * token's {@code eventS} entry (seconds from entry, token 1 included) is taken after that token's
   * argmax, matching the pinned Python controlled loop.
   */
  static ControlledResult runControlled(
      DecoderModel model, MLXScope owner, int[] prompt, int emitted) {
    if (emitted < 1) {
      throw new IllegalArgumentException("emitted must be >= 1");
    }
    long t0 = System.nanoTime();
    try (MLXScope request = owner.newChild()) {
      List<KVCache> caches = freshCaches(model, request);
      int promptLength = prompt.length;
      int token1;
      double prefillS;
      try (MLXScope activation = request.newChild()) {
        MLXArray ids = MLX.array(activation, prompt, new int[] {1, promptLength});
        MLXArray logits = model.stepLogits(ids, caches, new int[] {promptLength}, null);
        evalStep(logits, caches);
        prefillS = (System.nanoTime() - t0) / NANOS_PER_SECOND;
        // Token 1's event time is taken after its argmax, like every later token's; the
        // prefill timer above already stopped at the evaluated logits.
        token1 = argmax(logits);
      }
      double[] eventS = new double[emitted];
      eventS[0] = (System.nanoTime() - t0) / NANOS_PER_SECOND;
      List<Integer> tokens = new ArrayList<>(emitted);
      tokens.add(token1);
      List<Step> steps = new ArrayList<>(emitted - 1);
      for (int k = 0; k < emitted - 1; k++) {
        long stepStart = System.nanoTime();
        int next;
        try (MLXScope activation = request.newChild()) {
          MLXArray ids =
              MLX.array(activation, new int[] {tokens.get(tokens.size() - 1)}, new int[] {1, 1});
          MLXArray logits = model.decode(ids, caches, null, null);
          evalStep(logits, caches);
          next = argmax(logits);
        }
        // The step timer stops at the evaluated logits; the argmax host work above ran after
        // the stop, exactly as in the pinned Python loop.
        steps.add(
            new Step(
                promptLength + k + 1,
                (System.nanoTime() - stepStart) / NANOS_PER_SECOND,
                null,
                null));
        tokens.add(next);
        eventS[k + 1] = (System.nanoTime() - t0) / NANOS_PER_SECOND;
      }
      return new ControlledResult(
          List.copyOf(tokens), eventS[0], prefillS, List.copyOf(steps), eventS);
    }
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
    long t0 = System.nanoTime();
    try (MLXScope request = owner.newChild()) {
      List<KVCache> caches = freshCaches(model, request);
      int promptLength = prompt.length;
      double prefillS;
      try (MLXScope activation = request.newChild()) {
        MLXArray ids = MLX.array(activation, prompt, new int[] {1, promptLength});
        MLXArray logits = model.stepLogits(ids, caches, new int[] {promptLength}, null);
        evalStep(logits, caches);
      }
      prefillS = (System.nanoTime() - t0) / NANOS_PER_SECOND;
      List<Step> steps = new ArrayList<>(fedIds.length);
      for (int k = 0; k < fedIds.length; k++) {
        long stepStart = System.nanoTime();
        int argmaxId;
        double top2Margin;
        try (MLXScope activation = request.newChild()) {
          MLXArray ids = MLX.array(activation, new int[] {fedIds[k]}, new int[] {1, 1});
          MLXArray logits = model.decode(ids, caches, null, null);
          evalStep(logits, caches);
          // The step timer stops at the evaluated logits; the argmax and the full-vocabulary
          // readback and sort (the cross-adapter check) run outside the measured interval,
          // as in the pinned Python loop.
          argmaxId = argmax(logits);
          top2Margin = top2Margin(logits);
        }
        steps.add(
            new Step(
                promptLength + k + 1,
                (System.nanoTime() - stepStart) / NANOS_PER_SECOND,
                argmaxId,
                top2Margin));
      }
      // Wall time over the step region only (prefill excluded, one definition on both paths).
      return new FixedContinuationResult(
          prefillS, List.copyOf(steps), (System.nanoTime() - t0) / NANOS_PER_SECOND - prefillS);
    }
  }

  /**
   * The stock Java path: the public {@code generate} with pretokenized prompt IDs, the checkpoint's
   * own tokenizer (so the loop's incremental detokenization runs and is timed), full retention and
   * the frozen greedy policy. Host-visible token events are observed without inserting per-token
   * synchronizations; after the call returns, all pending work is drained with an explicit evaluate
   * before the full-generation timer stops.
   */
  static StockResult runStock(
      DecoderModel model,
      HfTokenizer tokenizer,
      int[] prompt,
      int maxTokens,
      Set<Integer> eosTokenIds) {
    GenerationRequest request =
        GenerationRequest.pretokenized(
            tokenizer,
            prompt,
            GenerationConfig.greedyDefaults(maxTokens, eosTokenIds),
            CancellationToken.NONE);
    long t0 = System.nanoTime();
    List<Integer> tokens = new ArrayList<>();
    List<Double> eventRel = new ArrayList<>();
    GenerationResult result =
        model.generate(
            request,
            event -> {
              if (event.tokenId() != null) {
                eventRel.add((System.nanoTime() - t0) / NANOS_PER_SECOND);
                tokens.add(event.tokenId());
              }
            });
    // The Java loop evaluates every step at its selection boundary and queues no lookahead
    // forward; the explicit drain makes the full-completion boundary unambiguous either way.
    MLX.eval();
    double[] eventS = new double[eventRel.size()];
    for (int i = 0; i < eventS.length; i++) {
      eventS[i] = eventRel.get(i);
    }
    return new StockResult(
        List.copyOf(tokens),
        eventS,
        eventS.length > 0 ? eventS[0] : 0.0,
        (System.nanoTime() - t0) / NANOS_PER_SECOND,
        result.finishReason());
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
