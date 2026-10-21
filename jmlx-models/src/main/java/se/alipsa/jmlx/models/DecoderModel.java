package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.AttentionMask;
import se.alipsa.jmlx.nn.DecoderBlock;
import se.alipsa.jmlx.nn.EmbeddingLayer;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.MultiAxisRotation;
import se.alipsa.jmlx.nn.RMSNorm;
import se.alipsa.jmlx.nn.UnaryLayer;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.IncrementalTokenDecoder;
import se.alipsa.jmlx.tokenizer.TokenizerException;

/**
 * Inference-only pre-norm decoder shared by the Llama, Qwen2, Qwen3, Mistral, Gemma, Phi-3 and
 * Mixtral checkpoints.
 */
public abstract class DecoderModel extends Module implements TextGenerationModel {
  private static final System.Logger LOGGER = System.getLogger(DecoderModel.class.getName());
  private final DecoderConfig config;
  private final ArchitectureDescriptor descriptor;
  private final ModelMetadata metadata;
  private final EmbeddingLayer embedding;
  private final List<DecoderBlock> layers;
  private final RMSNorm norm;
  private final UnaryLayer lmHead;
  private final boolean tiedOutput;
  private StepBoundaryEvaluator stepBoundaryEvaluator = StepBoundaryEvaluator.NATIVE;

  /** Replaces the evaluation boundary for package tests without affecting other model instances. */
  final void setStepBoundaryEvaluatorForTest(StepBoundaryEvaluator evaluator) {
    stepBoundaryEvaluator = Objects.requireNonNull(evaluator, "evaluator");
  }

  /** Builds a decoder from checkpoint tensors validated against the architecture's tensor plan. */
  protected DecoderModel(
      MLXScope scope, ArchitectureDescriptor descriptor, Map<String, MLXArray> tensors) {
    super(scope);
    this.descriptor = Objects.requireNonNull(descriptor, "descriptor");
    this.config = descriptor.dimensions();
    metadata =
        new DecoderMetadata(config.modelType(), config.vocabSize(), config.numHiddenLayers());
    DecoderAssembler.Assembled assembled = DecoderAssembler.assemble(scope, descriptor, tensors);
    embedding = child("embedding", assembled.embedding());
    List<DecoderBlock> built = new ArrayList<>();
    for (int i = 0; i < assembled.layers().size(); i++) {
      built.add(child("layer" + i, assembled.layers().get(i)));
    }
    layers = List.copyOf(built);
    norm = child("norm", assembled.finalNorm());
    tiedOutput = assembled.lmHead() == null;
    lmHead = tiedOutput ? null : child("lmHead", assembled.lmHead());
  }

  /**
   * Returns this decoder architecture's detailed configuration. Use {@link #metadata()} from code
   * that supports multiple model architectures; it deliberately exposes only stable common fields.
   */
  public final DecoderConfig config() {
    return config;
  }

  @Override
  public final ModelMetadata metadata() {
    return metadata;
  }

  /**
   * Returns logits shaped {@code [batch, sequence, vocab]} and advances one cache per layer. Each
   * cache must be non-null and live in this model scope or a descendant scope that outlives this
   * call; generation creates such caches automatically. This method synchronizes logits and changed
   * cache arrays before returning, so a lazy native failure poisons the caches instead of surfacing
   * after the positions were committed. Logits are computed for <em>every</em> position: {@link
   * #generate} avoids that cost by projecting only the last hidden state, so timings and peak
   * memory of this method are not comparable to generation's prefill. A failure after cache
   * mutation poisons all supplied caches; reset every cache before reuse.
   */
  public final MLXArray forward(MLXArray tokenIds, List<KVCache> caches) {
    int[] before = preflight(tokenIds, caches);
    try {
      MLXArray normalized = normalizedHiddenStates(tokenIds, caches);
      requirePostflight(caches);
      MLXArray logits = tiedOutput ? embedding.project(normalized) : lmHead.forward(normalized);
      MLXArray[] arrays = new MLXArray[1 + 2 * caches.size()];
      arrays[0] = logits;
      System.arraycopy(cacheArrays(caches), 0, arrays, 1, 2 * caches.size());
      stepBoundaryEvaluator.evaluate(arrays);
      return logits;
    } catch (RuntimeException | Error failure) {
      poisonIfMutated(caches, before);
      throw failure;
    }
  }

  /**
   * Evaluates a left-padded {@code [B,T]} batch with one positive valid length per row. Each row's
   * valid tokens occupy its rightmost positions. The resulting cache set must be reset after a
   * failure that occurs after any layer advances.
   */
  public final MLXArray forward(MLXArray tokenIds, List<KVCache> caches, int[] validLengths) {
    int[][] before = preflightBatch(tokenIds, caches, validLengths);
    try {
      MLXArray normalized = normalizedHiddenStatesBatch(tokenIds, caches, validLengths, null);
      requirePostflightBatch(caches, validLengths.length);
      MLXArray logits = tiedOutput ? embedding.project(normalized) : lmHead.forward(normalized);
      MLXArray[] arrays = new MLXArray[1 + 2 * caches.size()];
      arrays[0] = logits;
      System.arraycopy(cacheArrays(caches), 0, arrays, 1, 2 * caches.size());
      stepBoundaryEvaluator.evaluate(arrays);
      return logits;
    } catch (RuntimeException | Error failure) {
      poisonBatchIfMutated(caches, before, validLengths.length);
      throw failure;
    }
  }

  /**
   * Worker-thread-only access to this model's weight scope, which the inherited {@code
   * Module.scope()} (protected in another package) hides from the scheduler. The scheduler checks
   * {@code root.isAncestorOf(modelScope())} before accepting a factory result.
   */
  final MLXScope modelScope() {
    return scope();
  }

  /**
   * The model facts a scheduler snapshots on its worker so {@code submit} can validate without it.
   */
  record BatchFacts(
      String modelType,
      int vocabSize,
      Integer slidingWindow,
      boolean dynamicNtk,
      int layerCount,
      boolean tiedOutput) {}

  final BatchFacts batchFacts() {
    return new BatchFacts(
        config.modelType(),
        config.vocabSize(),
        descriptor.attention().slidingWindow(),
        descriptor.rope() instanceof se.alipsa.jmlx.nn.RopeSpec.DynamicNtk,
        layers.size(),
        tiedOutput);
  }

  /**
   * Transforms embeddings immediately after token lookup, before the optional sqrt-hidden scale.
   * Composing vision models merge image features here; scheduler tests also perturb selected rows.
   * Any supplied features receive the same subsequent embedding scale as the token embeddings.
   */
  @FunctionalInterface
  interface EmbeddingHook {
    MLXArray apply(MLXArray embedded);
  }

  /**
   * Injects a transform after each decoder block's residual output and before the final
   * normalization, invoked with the block's 0-based layer index. Qwen3-VL DeepStack (WP6/WP7) uses
   * this seam to add DeepStack level {@code i} after text block {@code i}, where the level's
   * features come from the vision tower's {@code deepstack_visual_indexes[i]} (findings §6.6/§6.9).
   * Those are different counters that must never be mixed: the vision extraction indices address
   * blocks of the 24-block vision stack (pinned 2B: {@code [5, 11, 17]}), while the injection index
   * addresses blocks of the 28-block text stack. The hook is invoked after <em>every</em> block's
   * residual output; selecting which layers actually receive an injection is the hook's own job.
   * Production ordinary decoders pass no-op or null hooks. Implementations must return a non-null
   * array with exactly the same shape as the hidden state they were given; {@code decoderStack}
   * checks this immediately after each call and throws {@link IllegalArgumentException} naming the
   * block index, so a bad result is reported at the hook rather than later inside the next block's
   * projection.
   */
  @FunctionalInterface
  interface PostBlockHook {
    MLXArray apply(int layerIndex, MLXArray hidden);
  }

  /**
   * Lazy batched step shared with the scheduler: runs one forward over a left-padded {@code [B,T]}
   * batch and returns the {@code [B,1,V]} logits of every row's last valid column, with <em>no</em>
   * evaluation and no cache poisoning bookkeeping beyond {@link #poisonBatchIfMutated}. Because
   * preflight requires the padded width to equal the longest valid row, every row's last valid
   * column is column {@code T-1}, so one slice (taken before the output projection, like the
   * single-row prefill) suffices -- no per-row gather. A one-column decode batch is the same call.
   *
   * <p>Unlike the public {@link #forward(MLXArray, List, int[])}, this neither evaluates the logits
   * and cache arrays (which would cost a second synchronization per step once the scheduler's joint
   * selection evaluate runs) nor projects all {@code T} positions.
   */
  final MLXArray stepLogits(
      MLXArray tokenIds, List<KVCache> caches, int[] validLengths, EmbeddingHook hook) {
    return stepLogits(tokenIds, caches, validLengths, hook, null, null);
  }

  /**
   * The seam-carrying form of {@link #stepLogits(MLXArray, List, int[], EmbeddingHook)}: the
   * embedding hook keeps its fourth, post-lookup/pre-scale position; a non-null {@code rotation}
   * replaces the stack's derived step frequencies with the request's own multi-axis rotation, and
   * {@code postBlockHook} fires after every block's residual output, before the final
   * normalization. The preflight, postflight, evaluation and poisoning invariants are the batched
   * ones, unchanged. A non-null {@code rotation} is validated against the batch, the sequence
   * width, the head dimension and the configured rotary width ({@link
   * MultiAxisRotation#validateAgainst(int, int, int, int)}) before the embedding lookup builds any
   * graph node. The existing 4-argument overload delegates with {@code (null, null)} and keeps the
   * identical graph.
   */
  final MLXArray stepLogits(
      MLXArray tokenIds,
      List<KVCache> caches,
      int[] validLengths,
      EmbeddingHook hook,
      MultiAxisRotation rotation,
      PostBlockHook postBlockHook) {
    int[][] before = preflightBatch(tokenIds, caches, validLengths);
    if (rotation != null) {
      // Reject a mismatched rotation before the embedding lookup builds any graph node.
      rotation.validateAgainst(
          tokenIds.shape()[0], tokenIds.shape()[1], descriptor.headDim(), descriptor.rotaryDims());
    }
    try {
      MLXArray normalized =
          normalizedHiddenStatesBatch(
              tokenIds, caches, validLengths, hook, rotation, postBlockHook);
      requirePostflightBatch(caches, validLengths.length);
      int[] shape = normalized.shape();
      MLXArray last =
          MLXShape.slice(
              normalized, new int[] {0, shape[1] - 1, 0}, new int[] {shape[0], shape[1], shape[2]});
      return tiedOutput ? embedding.project(last) : lmHead.forward(last);
    } catch (RuntimeException | Error failure) {
      poisonBatchIfMutated(caches, before, validLengths.length);
      throw failure;
    }
  }

  /** The cache arrays a joint step evaluation must include, in layer order (keys, values). */
  static MLXArray[] stepCacheArrays(List<KVCache> caches) {
    return cacheArrays(caches);
  }

  /** The evaluation boundary a scheduler must route its joint evaluation through. */
  final StepBoundaryEvaluator stepBoundaryEvaluator() {
    return stepBoundaryEvaluator;
  }

  /** Poisons every cache if any row of any layer advanced since {@code before} was recorded. */
  static void poisonBatchIfMutated(List<KVCache> caches, int[][] before, int batch) {
    for (int layer = 0; layer < caches.size(); layer++) {
      KVCache cache = caches.get(layer);
      for (int row = 0; row < batch; row++) {
        if (cache.nextPosition(row) != before[layer][row]) {
          caches.forEach(KVCache::poison);
          return;
        }
      }
    }
  }

  private int[][] preflightBatch(MLXArray tokenIds, List<KVCache> caches, int[] validLengths) {
    Objects.requireNonNull(tokenIds, "tokenIds");
    Objects.requireNonNull(validLengths, "validLengths");
    Objects.requireNonNull(caches, "caches");
    if (tokenIds.ndim() != 2 || tokenIds.shape()[0] <= 0 || tokenIds.shape()[1] <= 0) {
      throw new IllegalArgumentException(
          "batched forward requires nonempty [B,T], B lengths and one cache per layer");
    }
    return preflightBatch(tokenIds.shape()[0], tokenIds.shape()[1], caches, validLengths);
  }

  /**
   * The shape-keyed core of the batched preflight, run by the token-ID entries ({@link
   * #forward(MLXArray, List, int[])} and {@link #stepLogits}) through the token-ID overload and, by
   * embedding-start callers (the SmolVLM work, WP6, this package), directly with {@code
   * (embeddings.shape()[0], sequenceWidth)}. It is pure int arithmetic over the caches and valid
   * lengths: it builds no graph node, evaluates nothing and changes no captured bit. It validates
   * the valid-length invariants (one positive length per row, each within {@code 1..width}, and the
   * padded width equal to the longest valid row), the cache count and batch size against {@code
   * batch}, policy consistency across the layers, the per-row and padded capacity of each policy,
   * matching row positions across the layer caches, and the DynamicNtk requirement of equal batch
   * positions. A poisoned cache or a wrong cache batch size throws {@link IllegalStateException};
   * every other violation throws {@link IllegalArgumentException}. It returns the {@code before}
   * snapshot of every layer and row's next position, which the caller passes to {@link
   * #poisonBatchIfMutated} in its own catch.
   */
  int[][] preflightBatch(int batch, int width, List<KVCache> caches, int[] validLengths) {
    Objects.requireNonNull(validLengths, "validLengths");
    Objects.requireNonNull(caches, "caches");
    if (batch <= 0
        || width <= 0
        || caches.size() != layers.size()
        || validLengths.length != batch) {
      throw new IllegalArgumentException(
          "batched forward requires nonempty [B,T], B lengths and one cache per layer");
    }
    int maxValid = 0;
    for (int valid : validLengths) {
      if (valid <= 0 || valid > width) {
        throw new IllegalArgumentException("each batch row needs 1..T valid tokens");
      }
      maxValid = Math.max(maxValid, valid);
    }
    if (maxValid != width) {
      throw new IllegalArgumentException("left-padded width must equal longest valid row");
    }
    int[][] before = new int[caches.size()][batch];
    int[] starts = new int[batch];
    int[] positions = new int[batch];
    KVCachePolicy expectedPolicy = Objects.requireNonNull(caches.getFirst(), "cache 0").policy();
    for (int layer = 0; layer < caches.size(); layer++) {
      KVCache cache = Objects.requireNonNull(caches.get(layer), "cache " + layer);
      requireCompatiblePolicy(cache.policy(), expectedPolicy);
      if (cache.isPoisoned() || (cache.batchSize() != 0 && cache.batchSize() != batch)) {
        throw new IllegalStateException("decoder batch cache is poisoned or has wrong batch size");
      }
      validateCacheShape(cache, batch);
      int layerWidth = 0;
      for (int row = 0; row < batch; row++) {
        int next = cache.nextPosition(row);
        int start = cache.startPosition(row);
        if (layer > 0 && (positions[row] != next || starts[row] != start)) {
          throw new IllegalArgumentException("decoder layer caches have mismatched row positions");
        }
        if (layer == 0) {
          positions[row] = next;
          starts[row] = start;
        }
        cache.policy().requireCapacity((long) next + validLengths[row], "decoder batch cache");
        before[layer][row] = next;
        layerWidth = Math.max(layerWidth, next - start + validLengths[row]);
      }
      cache.policy().requireCapacity(layerWidth, "decoder padded cache width");
    }
    if (descriptor.rope() instanceof se.alipsa.jmlx.nn.RopeSpec.DynamicNtk) {
      int ending = positions[0] + validLengths[0];
      for (int row = 1; row < batch; row++) {
        if (positions[row] != positions[0] || positions[row] + validLengths[row] != ending) {
          throw new IllegalArgumentException("DynamicNtk does not support unequal batch positions");
        }
      }
    }
    return before;
  }

  private static void requirePostflight(List<KVCache> caches) {
    int next = caches.getFirst().nextPosition();
    int start = caches.getFirst().startPosition();
    for (KVCache cache : caches) {
      if (cache.nextPosition() != next || cache.startPosition() != start) {
        throw new IllegalStateException("decoder layers diverged after forward");
      }
    }
  }

  private static void requirePostflightBatch(List<KVCache> caches, int batch) {
    KVCache first = caches.getFirst();
    for (KVCache cache : caches) {
      for (int row = 0; row < batch; row++) {
        if (cache.nextPosition(row) != first.nextPosition(row)
            || cache.startPosition(row) != first.startPosition(row)) {
          throw new IllegalStateException("decoder batch layers diverged after forward");
        }
      }
    }
  }

  /**
   * The post-block hook's contract: it must hand back the block's own hidden state, non-null and
   * with the exact shape it was given, so the next block's projection sees the same tensor
   * contract. Enforced immediately after the call, so a violation names the block instead of
   * surfacing later inside the next block's projection.
   */
  private static void requirePostBlockResult(int blockIndex, MLXArray hidden, int[] expected) {
    if (hidden == null || !Arrays.equals(expected, hidden.shape())) {
      throw new IllegalArgumentException(
          "postBlockHook for block "
              + blockIndex
              + " must return the block's hidden state with the same shape "
              + Arrays.toString(expected)
              + ": got "
              + (hidden == null ? "null" : Arrays.toString(hidden.shape())));
    }
  }

  /**
   * The embedding-input half of the decoder stack: the embedding lookup, the optional hook invoked
   * at the identical post-lookup/pre-scale point, and the optional sqrt-hidden scale. Both the
   * single-row and the batched stack call it, so the hook and the scale are written once, in one
   * order.
   */
  private MLXArray decoderEmbeddings(MLXArray tokenIds, EmbeddingHook hook) {
    MLXArray x = embedding.forward(tokenIds);
    if (hook != null) {
      x = hook.apply(x);
    }
    if (descriptor.embedding().scaleBySqrtHidden()) {
      MLXArray scale =
          MLX.array(x.scope(), new float[] {(float) Math.sqrt(config.hiddenSize())}, new int[] {1});
      x = MLXOps.multiply(x, MLX.astype(scale, x.dtype()));
    }
    return x;
  }

  /**
   * The internal embedding-start entry of the batched decoder stack: the left-padded attention
   * mask, the RoPE step frequencies, every block and the final normalization, run over
   * already-embedded activations. It takes the explicit padded sequence width and valid lengths
   * that the token-ID entry derived, preserves the graph operation order, dtypes and
   * lazy-evaluation boundaries of the stack it was factored from, and it neither evaluates nor
   * poisons: an embedding-start caller (the SmolVLM work, WP6, this package) must run {@link
   * #preflightBatch(int, int, List, int[])} itself with {@code (embeddings.shape()[0],
   * sequenceWidth)} and pass the returned snapshot to {@link #poisonBatchIfMutated} in its own
   * catch, and keeps the postflight and evaluation responsibilities. This entry itself validates
   * the batch shape of {@code embeddings} against {@code validLengths}, {@code sequenceWidth} and
   * {@code caches} (shape {@code [B, sequenceWidth, hiddenSize]} with one cache per layer) before
   * building any graph node. Production token-ID callers keep the identical stack via {@link
   * #normalizedHiddenStatesBatch}.
   */
  final MLXArray decoderStack(
      MLXArray embeddings, List<KVCache> caches, int[] validLengths, int sequenceWidth) {
    return decoderStack(embeddings, caches, validLengths, sequenceWidth, null, null);
  }

  /**
   * The seam-carrying form of {@link #decoderStack(MLXArray, List, int[], int)}: a non-null {@code
   * rotation} skips the step frequency derivation and rotates each block's query and key heads from
   * the request's own multi-axis rotation instead; a non-null {@code postBlockHook} is invoked
   * after every block's residual output and before the final {@code norm.forward} (see {@link
   * PostBlockHook}). Its result is checked immediately: a null or differently shaped return is
   * rejected as an {@link IllegalArgumentException} naming the block index. The batched mask is the
   * same in both cases. A non-null {@code rotation} is validated against the batch, the sequence
   * width, the head dimension and the configured rotary width ({@link
   * MultiAxisRotation#validateAgainst(int, int, int, int)}) before the mask or any block builds a
   * graph node. With {@code (null, null)} this runs the byte-identical stack the 4-argument
   * overload always ran: one frequency derivation at the same graph position, the same block
   * forwards and the same final normalization. This entry itself still neither evaluates nor
   * poisons; embedding-start callers keep the {@link #preflightBatch(int, int, List, int[])}/{@link
   * #poisonBatchIfMutated} and postflight/evaluation responsibilities of the 4-argument entry.
   */
  final MLXArray decoderStack(
      MLXArray embeddings,
      List<KVCache> caches,
      int[] validLengths,
      int sequenceWidth,
      MultiAxisRotation rotation,
      PostBlockHook postBlockHook) {
    int[] shape = embeddings.shape();
    if (shape.length != 3
        || shape[0] != validLengths.length
        || shape[1] != sequenceWidth
        || shape[2] != config.hiddenSize()) {
      throw new IllegalArgumentException(
          "decoderStack requires embeddings shaped ["
              + validLengths.length
              + ", "
              + sequenceWidth
              + ", "
              + config.hiddenSize()
              + "]: got "
              + Arrays.toString(shape));
    }
    if (caches.size() != layers.size()) {
      throw new IllegalArgumentException(
          "one KVCache is required per decoder layer: got "
              + caches.size()
              + " caches for "
              + layers.size()
              + " layers");
    }
    MLXArray x = embeddings;
    KVCache first = caches.getFirst();
    int batch = validLengths.length;
    if (rotation != null) {
      // Reject a mismatched rotation before the batched mask or any block builds a graph node.
      rotation.validateAgainst(batch, sequenceWidth, descriptor.headDim(), descriptor.rotaryDims());
    }
    int[] queryStarts = new int[batch];
    int[] keyStarts = new int[batch];
    int keyWidth = 0;
    int maxEnding = 0;
    for (int row = 0; row < batch; row++) {
      queryStarts[row] = first.nextPosition(row);
      keyStarts[row] = first.startPosition(row);
      keyWidth = Math.max(keyWidth, queryStarts[row] - keyStarts[row] + validLengths[row]);
      maxEnding = Math.max(maxEnding, queryStarts[row] + validLengths[row]);
    }
    Integer window = descriptor.attention().slidingWindow();
    MLXArray mask =
        AttentionMask.batched(
            x.scope(),
            queryStarts,
            keyStarts,
            validLengths,
            sequenceWidth,
            keyWidth,
            window == null ? 0 : window);
    // The null-rotation path derives the step frequencies exactly once, in the same scope and at
    // the same graph position as before the seam existed; the rotation path skips the derivation
    // and rotates from the request's own cos/sin tables.
    MLXArray frequencies =
        rotation == null
            ? descriptor.rope().stepFrequencies(x.scope(), descriptor.rotaryDims(), maxEnding)
            : null;
    for (int i = 0; i < layers.size(); i++) {
      x =
          rotation == null
              ? layers.get(i).forward(x, caches.get(i), mask, frequencies, validLengths)
              : layers.get(i).forward(x, caches.get(i), mask, validLengths, rotation);
      if (postBlockHook != null) {
        // MLXArray.shape() returns a fresh array on every call, so the hook cannot alias it.
        int[] expected = x.shape();
        x = postBlockHook.apply(i, x);
        requirePostBlockResult(i, x, expected);
      }
    }
    return norm.forward(x);
  }

  private MLXArray normalizedHiddenStatesBatch(
      MLXArray tokenIds, List<KVCache> caches, int[] validLengths, EmbeddingHook hook) {
    return normalizedHiddenStatesBatch(tokenIds, caches, validLengths, hook, null, null);
  }

  private MLXArray normalizedHiddenStatesBatch(
      MLXArray tokenIds,
      List<KVCache> caches,
      int[] validLengths,
      EmbeddingHook hook,
      MultiAxisRotation rotation,
      PostBlockHook postBlockHook) {
    MLXArray x = decoderEmbeddings(tokenIds, hook);
    return decoderStack(x, caches, validLengths, tokenIds.shape()[1], rotation, postBlockHook);
  }

  private int[] preflight(MLXArray tokenIds, List<KVCache> caches) {
    Objects.requireNonNull(tokenIds, "tokenIds");
    if (tokenIds.ndim() != 2 || tokenIds.shape()[0] <= 0 || tokenIds.shape()[1] <= 0) {
      throw new IllegalArgumentException("tokenIds must have nonempty shape [batch, sequence]");
    }
    Objects.requireNonNull(caches, "caches");
    if (caches.size() != layers.size()) {
      throw new IllegalArgumentException("one KVCache is required per decoder layer");
    }
    int[] before = new int[caches.size()];
    KVCache first = Objects.requireNonNull(caches.get(0), "cache 0");
    if (first.isPoisoned()) {
      throw new IllegalStateException("decoder cache set is poisoned; reset every layer");
    }
    requireUnpadded(first);
    int position = first.nextPosition();
    int start = first.startPosition();
    for (int i = 0; i < caches.size(); i++) {
      KVCache cache = Objects.requireNonNull(caches.get(i), "cache " + i);
      requireCompatiblePolicy(cache.policy(), first.policy());
      if (cache.isPoisoned()) {
        throw new IllegalStateException("decoder cache set is poisoned; reset every layer");
      }
      requireUnpadded(cache);
      if (cache.nextPosition() != position || cache.startPosition() != start) {
        throw new IllegalArgumentException("decoder layer caches have mismatched positions");
      }
      if (cache.keys() != null && cache.keys().shape()[0] != tokenIds.shape()[0]) {
        throw new IllegalArgumentException("decoder cache batch size differs from tokenIds");
      }
      validateCacheShape(cache, tokenIds.shape()[0]);
      cache.policy().requireCapacity((long) position + tokenIds.shape()[1], "decoder cache");
      before[i] = position;
    }
    return before;
  }

  private static void requireUnpadded(KVCache cache) {
    if (!cache.isUniform() || cache.rowLength(0) != cache.length()) {
      throw new IllegalArgumentException("left-padded cache requires validLengths overload");
    }
  }

  private void requireCompatiblePolicy(KVCachePolicy policy, KVCachePolicy expected) {
    if (!policy.equals(expected)) {
      throw new IllegalArgumentException("decoder layer caches have mismatched policies");
    }
    if (policy.evicts()
        && !Objects.equals(descriptor.attention().slidingWindow(), policy.limit())) {
      throw new IllegalArgumentException("SLIDING_WINDOW cache must match checkpoint window");
    }
  }

  private void validateCacheShape(KVCache cache, int batch) {
    if (cache.keys() == null) {
      return;
    }
    int[] keyShape = cache.keys().shape();
    int[] valueShape = cache.values().shape();
    if (keyShape.length != 4
        || valueShape.length != 4
        || keyShape[0] != batch
        || valueShape[0] != batch
        || keyShape[1] != config.numKeyValueHeads()
        || valueShape[1] != config.numKeyValueHeads()
        || keyShape[3] != descriptor.headDim()
        || valueShape[3] != descriptor.headDim()
        || keyShape[2] != valueShape[2]) {
      throw new IllegalArgumentException("decoder cache key/value shape differs from architecture");
    }
  }

  private static void poisonIfMutated(List<KVCache> caches, int[] before) {
    for (int i = 0; i < caches.size(); i++) {
      if (caches.get(i).nextPosition() != before[i]) {
        caches.forEach(KVCache::poison);
        return;
      }
    }
  }

  private static MLXArray[] cacheArrays(List<KVCache> caches) {
    MLXArray[] arrays = new MLXArray[2 * caches.size()];
    for (int i = 0; i < caches.size(); i++) {
      arrays[2 * i] = caches.get(i).keys();
      arrays[2 * i + 1] = caches.get(i).values();
    }
    return arrays;
  }

  private MLXArray normalizedHiddenStates(MLXArray tokenIds, List<KVCache> caches) {
    return normalizedHiddenStates(tokenIds, caches, null, null);
  }

  /**
   * The single-row seam-carrying form of {@link #normalizedHiddenStates(MLXArray, List)}. With
   * {@code (null, null)} it runs the old single-row graph byte for byte: the 2-D windowed mask (or
   * the implicit causal one), the derived step frequencies and the non-batch block forwards. A
   * non-null rotation or hook routes the row through the batch-aware overloads instead: the {@link
   * #decoderStack(MLXArray, List, int[], int, MultiAxisRotation, PostBlockHook)} entry with one
   * uniform valid length per row. With only a hook, that route is the same arithmetic as the old
   * single-row path for a batch-1 unpadded row: the uniform append and the {@code [1,1,T,L]}
   * batched mask match the single-row ones element for element, windowed and evicted included
   * (pinned by {@code SingleRowDecodeEquivalenceTest}), the per-row RoPE form delegates to the
   * offset form for a uniform row, and the explicit mask yields the same SDPA result as the
   * implicit causal one -- pinned bit-exact at the logits, end to end, by {@code
   * DecoderSeamTest.decodeWithNoOpPostBlockHookMatchesOldSingleRowPath} -- plus the hook firing
   * after every block's residual output, before the final normalization. A non-null rotation
   * deliberately changes the arithmetic instead: it rotates from the request's own cos/sin tables
   * rather than the rope fast kernel, within the 1e-5 that {@code
   * DecoderSeamTest.equalAxisRotationMatchesRopePathWithinKernelTolerance} pins.
   */
  private MLXArray normalizedHiddenStates(
      MLXArray tokenIds,
      List<KVCache> caches,
      MultiAxisRotation rotation,
      PostBlockHook postBlockHook) {
    for (int i = 0; i < caches.size(); i++) {
      Objects.requireNonNull(caches.get(i), "cache " + i);
    }
    MLXArray x = decoderEmbeddings(tokenIds, null);
    if (rotation == null && postBlockHook == null) {
      Integer window = descriptor.attention().slidingWindow();
      int keyLength = caches.get(0).length() + tokenIds.shape()[1];
      MLXArray mask =
          window != null && (window < keyLength || caches.get(0).startPosition() > 0)
              ? AttentionMask.slidingWindow(
                  x.scope(),
                  caches.get(0).nextPosition(),
                  caches.get(0).startPosition(),
                  tokenIds.shape()[1],
                  keyLength,
                  window)
              : null;
      MLXArray stepFrequencies =
          descriptor
              .rope()
              .stepFrequencies(
                  x.scope(),
                  descriptor.rotaryDims(),
                  caches.get(0).nextPosition() + tokenIds.shape()[1]);
      for (int i = 0; i < layers.size(); i++) {
        x = layers.get(i).forward(x, caches.get(i), mask, stepFrequencies);
      }
      return norm.forward(x);
    }
    int sequence = tokenIds.shape()[1];
    int[] validLengths = new int[tokenIds.shape()[0]];
    Arrays.fill(validLengths, sequence);
    return decoderStack(x, caches, validLengths, sequence, rotation, postBlockHook);
  }

  /**
   * Resolves a request-level cache policy against this checkpoint's effective sliding window -- the
   * same resolution {@link #generate(GenerationRequest, Consumer)} applies, so callers that build
   * their own caches for {@link #forward} cannot disagree with it.
   */
  public final KVCachePolicy resolveCachePolicy(GenerationCachePolicy requested) {
    return Objects.requireNonNull(requested, "requested")
        .resolve(descriptor.attention().slidingWindow());
  }

  /** Greedily generates up to {@code maxNewTokens}; prompt tokens are included in the result. */
  public final List<Integer> generate(int[] prompt, int maxNewTokens, Set<Integer> eosTokenIds) {
    if (prompt == null || prompt.length == 0) {
      throw new IllegalArgumentException("prompt must not be empty");
    }
    GenerationResult result =
        generate(
            new GenerationRequest(
                prompt,
                GenerationConfig.greedyDefaults(maxNewTokens, eosTokenIds),
                CancellationToken.NONE),
            ignored -> {});
    return result.tokenIds();
  }

  /**
   * Runs the requested greedy or explicitly seeded sampling policy and emits token plus terminal
   * events synchronously on the calling thread.
   */
  @Override
  public final GenerationResult generate(
      GenerationRequest request, Consumer<GenerationEvent> listener) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(listener, "listener");
    if (!request.images().isEmpty()) {
      throw new IllegalArgumentException(
          "model_type " + config.modelType() + " does not accept images");
    }
    GenerationConfig policy = request.config();
    int[] prompt = request.promptTokenIds();
    validateTokenIds(prompt, policy, config.vocabSize());
    KVCachePolicy cachePolicy = resolveCachePolicy(request.cachePolicy());
    long required =
        policy.maxNewTokens() == 0 ? 0 : (long) prompt.length + policy.maxNewTokens() - 1;
    // Clamped: an effectively unbounded token budget ("until EOS") is legal on unbounded and
    // sliding caches; only a bounded FULL capacity can be exceeded up front.
    cachePolicy.requireCapacity(Math.min(required, Integer.MAX_VALUE), "generation");
    HfTokenizer tokenizer = request.tokenizer();
    if (tokenizer != null && tokenizer.vocabSize() > config.vocabSize()) {
      throw new IllegalArgumentException(
          "tokenizer vocabulary size "
              + tokenizer.vocabSize()
              + " exceeds checkpoint vocabulary size "
              + config.vocabSize());
    }
    return runStepLoop(
        request,
        policy,
        prompt,
        prompt,
        prompt.length,
        cachePolicy,
        listener,
        (activation, stepCaches) -> {
          MLXArray ids = MLX.array(activation, prompt, new int[] {1, prompt.length});
          int[][] before = preflightBatch(ids, stepCaches, new int[] {prompt.length});
          MLXArray logits = stepLogits(ids, stepCaches, new int[] {prompt.length}, null);
          return new FirstStepResult(
              logits, cs -> poisonBatchIfMutated(cs, before, before[0].length));
        });
  }

  /**
   * Step 0 of {@link #runStepLoop}, caller-supplied so a composing model (the SmolVLM work, WP6,
   * this package) can run its own vision-merge first step. It receives the step's activation scope,
   * in which it must build any token-ID array it needs, and the request's fresh cache set.
   */
  @FunctionalInterface
  interface FirstStep {
    FirstStepResult apply(MLXScope activation, List<KVCache> caches);
  }

  /**
   * Step 0's lazy {@code [1,1,V]} logits plus the guard the loop applies when a failure happens
   * after the step's lazy graph has already advanced the caches.
   */
  record FirstStepResult(MLXArray logits, StepFailureGuard guard) {}

  /**
   * Poisons the caches on a first-step failure, using the position snapshot the first step captured
   * before it advanced them.
   */
  @FunctionalInterface
  interface StepFailureGuard {
    void onStepFailure(List<KVCache> caches);
  }

  /**
   * The decode step {@link #runStepLoop} uses for every step at 1 or higher, returning the lazy
   * {@code [1,1,V]} logits of the new token. The 8-argument entry's shared default is this model's
   * own single-row {@code decode} path; a composing model (the Qwen3-VL work, WP6/WP7, this
   * package) passes a step that calls the seam-carrying {@code decode} overload with its request's
   * rotary state. The loop's own preflight, poisoning and evaluation ordering stay around the call.
   */
  @FunctionalInterface
  interface DecodeStep {
    MLXArray apply(MLXArray ids, List<KVCache> caches);
  }

  /**
   * The shared generation step loop behind {@link #generate(GenerationRequest, Consumer)}: the
   * per-step activation scope, the per-request generation scope, sampler, caches, cancellation
   * checks, token and terminal events and the result construction. Step 0 is caller-supplied
   * through {@code firstStep} -- the decoder passes a first step that runs the plain batched {@link
   * #stepLogits} prefill, while a composing model (the SmolVLM work, WP6, this package) passes its
   * vision-merge step; every step at 1 or higher always uses this model's own single-row {@code
   * decode} path with its own preflight and poisoning. {@code prompt} (the unexpanded IDs) is what
   * the result and abort reports carry; {@code frequencySeed} may differ from it, letting a vision
   * model run the penalties over its expanded prompt; {@code promptPositions} becomes the result's
   * {@link GenerationResult#promptPositions()}. All request validation stays with the caller,
   * including the image rejection, the tokenizer-vocabulary check and the {@code maxNewTokens == 0}
   * capacity accounting, so a zero-step run performs no step at all.
   */
  final GenerationResult runStepLoop(
      GenerationRequest request,
      GenerationConfig policy,
      int[] prompt,
      int[] frequencySeed,
      int promptPositions,
      KVCachePolicy cachePolicy,
      Consumer<GenerationEvent> listener,
      FirstStep firstStep) {
    return runStepLoop(
        request,
        policy,
        prompt,
        frequencySeed,
        promptPositions,
        cachePolicy,
        listener,
        firstStep,
        this::decode);
  }

  /**
   * The seam-carrying form of {@link #runStepLoop(GenerationRequest, GenerationConfig, int[],
   * int[], int, KVCachePolicy, Consumer, FirstStep)}: steps at 1 or higher call {@code
   * decodeStep.apply(ids, caches)} where the 8-argument entry calls the model's own single-row
   * {@code decode}. Everything else -- the per-step activation scope, the first step, the sampler,
   * cancellation, token and terminal events, the preflight, poisoning, evaluation ordering and the
   * result construction -- is the identical loop body. The existing 8-argument entry delegates with
   * {@code this::decode}, so its path is unchanged.
   */
  final GenerationResult runStepLoop(
      GenerationRequest request,
      GenerationConfig policy,
      int[] prompt,
      int[] frequencySeed,
      int promptPositions,
      KVCachePolicy cachePolicy,
      Consumer<GenerationEvent> listener,
      FirstStep firstStep,
      DecodeStep decodeStep) {
    HfTokenizer tokenizer = request.tokenizer();
    List<Integer> generated = new ArrayList<>();
    List<Double> logProbabilities = new ArrayList<>();
    StringBuilder generatedText = tokenizer == null ? null : new StringBuilder();
    IncrementalTokenDecoder decoder = null;
    LinkedHashMap<Integer, Integer> frequencies = PenaltyInputs.frequencies(frequencySeed);
    FinishReason reason = FinishReason.MAX_TOKENS;
    if (!request.cancellationToken().isCancelled() && policy.maxNewTokens() > 0) {
      decoder = tokenizer == null ? null : tokenizer.newIncrementalDecoder(true);
      try (MLXScope generation = scope().newChild();
          SamplingPipeline sampler =
              new SamplingPipeline(generation, policy, config.vocabSize(), stepBoundaryEvaluator)) {
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
          caches.add(new KVCache(generation, cachePolicy));
        }
        int[] input = prompt;
        for (int step = 0; step < policy.maxNewTokens(); step++) {
          if (request.cancellationToken().isCancelled()) {
            reason = FinishReason.CANCELLED;
            break;
          }
          try (MLXScope activation = generation.newChild()) {
            int[] before = null;
            StepFailureGuard firstStepGuard = null;
            SamplingPipeline.Selection selection;
            try {
              MLXArray lastLogits;
              if (step == 0) {
                FirstStepResult first = firstStep.apply(activation, caches);
                lastLogits = first.logits();
                firstStepGuard = first.guard();
              } else {
                MLXArray ids = MLX.array(activation, input, new int[] {1, input.length});
                before = preflight(ids, caches);
                lastLogits = decodeStep.apply(ids, caches);
              }
              selection =
                  sampler.select(
                      lastLogits,
                      PenaltyInputs.from(frequencies, config.vocabSize()),
                      step,
                      cacheArrays(caches));
            } catch (RuntimeException | Error failure) {
              if (step == 0 && firstStepGuard != null) {
                firstStepGuard.onStepFailure(caches);
              } else if (before != null) {
                poisonIfMutated(caches, before);
              }
              // A firstStep.apply failure is guarded by the first step's own preflight/catch.
              throw failure;
            }
            int next = selection.tokenId();
            boolean eos = policy.eosTokenIds().contains(next);
            if (!eos && policy.stopTokenIds().contains(next)) {
              reason = FinishReason.STOP_TOKEN;
              break;
            }
            generated.add(next);
            frequencies.merge(next, 1, Integer::sum);
            if (selection.logProbability() != null) {
              logProbabilities.add(selection.logProbability());
            }
            String textDelta = null;
            if (decoder != null) {
              try {
                textDelta = decoder.append(next);
                generatedText.append(textDelta);
              } catch (TokenizerException e) {
                throw new GenerationAbortedException(
                    toList(prompt), generated, "output decoder", next, e);
              }
            }
            try {
              listener.accept(tokenEvent(next, textDelta, selection.logProbability()));
            } catch (RuntimeException e) {
              throw new GenerationAbortedException(toList(prompt), generated, e);
            }
            if (eos) {
              reason = FinishReason.EOS;
              break;
            }
            input = new int[] {next};
          }
        }
      }
    } else if (request.cancellationToken().isCancelled()) {
      reason = FinishReason.CANCELLED;
    }
    String terminalDelta = null;
    if (tokenizer != null) {
      terminalDelta = "";
      if (decoder != null) {
        try {
          terminalDelta = decoder.finish();
        } catch (TokenizerException e) {
          throw new GenerationAbortedException(
              toList(prompt), generated, "output decoder finish", null, e);
        }
      }
      generatedText.append(terminalDelta);
    }
    GenerationResult result =
        new GenerationResult(
            toList(prompt),
            generated,
            reason,
            logProbabilities,
            generatedText == null ? null : generatedText.toString(),
            promptPositions);
    try {
      listener.accept(
          terminalDelta == null
              ? GenerationEvent.finished(reason)
              : GenerationEvent.finished(reason, terminalDelta));
    } catch (RuntimeException e) {
      LOGGER.log(System.Logger.Level.WARNING, "generation terminal listener failed", e);
    }
    return result;
  }

  private MLXArray decode(MLXArray ids, List<KVCache> caches) {
    return decode(ids, caches, null, null);
  }

  /**
   * The single-row decode step with request-supplied rotary state and post-block injection: the
   * same one-token and populated-cache validation as {@link #decode(MLXArray, List)}, then the
   * seam-carrying stack and the same postflight and tied projection. A non-null {@code rotation} is
   * validated against the batch, the sequence width, the head dimension and the configured rotary
   * width ({@link MultiAxisRotation#validateAgainst(int, int, int, int)}) before the embedding
   * lookup builds any graph node. Package-private (like the other seam entries) so a {@link
   * DecodeStep} and the in-package seam tests can call it directly; production ordinary decoders
   * reach it through the 8-argument {@link #runStepLoop} delegation with {@code (null, null)} and
   * keep the old graph.
   */
  final MLXArray decode(
      MLXArray ids, List<KVCache> caches, MultiAxisRotation rotation, PostBlockHook postBlockHook) {
    if (ids.shape()[1] != 1 || caches.get(0).nextPosition() == 0) {
      throw new IllegalArgumentException("decode requires one token and a populated cache");
    }
    if (rotation != null) {
      // Reject a mismatched rotation before the embedding lookup builds any graph node.
      rotation.validateAgainst(
          ids.shape()[0], ids.shape()[1], descriptor.headDim(), descriptor.rotaryDims());
    }
    MLXArray normalized = normalizedHiddenStates(ids, caches, rotation, postBlockHook);
    requirePostflight(caches);
    return tiedOutput ? embedding.project(normalized) : lmHead.forward(normalized);
  }

  static GenerationEvent tokenEvent(int tokenId, String textDelta, Double logProbability) {
    if (textDelta == null) {
      return logProbability == null
          ? GenerationEvent.token(tokenId)
          : GenerationEvent.token(tokenId, logProbability);
    }
    return logProbability == null
        ? GenerationEvent.token(tokenId, textDelta)
        : GenerationEvent.token(tokenId, textDelta, logProbability);
  }

  private static List<Integer> toList(int[] ids) {
    return Arrays.stream(ids).boxed().toList();
  }

  static void validateTokenIds(int[] prompt, GenerationConfig policy, int vocabularySize) {
    for (int token : prompt) {
      requireTokenId("prompt", token, vocabularySize);
    }
    for (int token : policy.eosTokenIds()) {
      requireTokenId("EOS", token, vocabularySize);
    }
    for (int token : policy.stopTokenIds()) {
      requireTokenId("stop", token, vocabularySize);
    }
    if (policy.topK() > vocabularySize) {
      throw new IllegalArgumentException(
          "topK " + policy.topK() + " exceeds vocabulary size " + vocabularySize);
    }
  }

  private static void requireTokenId(String kind, int token, int vocabularySize) {
    if (token < 0 || token >= vocabularySize) {
      throw new IllegalArgumentException(
          kind + " token ID " + token + " outside vocabulary [0, " + vocabularySize + ")");
    }
  }

  /**
   * Encodes {@code prompt}, greedily generates text, then decodes only the newly generated token
   * ids, adding the tokenizer's normal special tokens. For an already-rendered chat prompt, use the
   * overload with {@code addSpecialTokens=false}.
   */
  public final String generateText(
      HfTokenizer tokenizer, String prompt, int maxNewTokens, Set<Integer> eosTokenIds) {
    return generateText(tokenizer, prompt, true, maxNewTokens, eosTokenIds);
  }

  /**
   * Encodes and greedily generates text. Pass {@code false} for a prompt already rendered by a chat
   * template, because Hugging Face chat templates include their own BOS token.
   */
  public final String generateText(
      HfTokenizer tokenizer,
      String prompt,
      boolean addSpecialTokens,
      int maxNewTokens,
      Set<Integer> eosTokenIds) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    Objects.requireNonNull(prompt, "prompt");
    GenerationRequest request =
        GenerationRequest.text(
            tokenizer,
            prompt,
            addSpecialTokens ? PromptSpecialTokens.ADD : PromptSpecialTokens.OMIT,
            GenerationConfig.greedyDefaults(maxNewTokens, eosTokenIds),
            CancellationToken.NONE);
    return generate(request, ignored -> {}).generatedText();
  }
}
