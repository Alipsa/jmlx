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
import se.alipsa.jmlx.nn.Embedding;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.RMSNorm;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.tokenizer.IncrementalTokenDecoder;
import se.alipsa.jmlx.tokenizer.TokenizerException;

/** Inference-only pre-norm decoder shared by Llama and Qwen2 checkpoints. */
public abstract class DecoderModel extends Module implements TextGenerationModel {
  private static final System.Logger LOGGER = System.getLogger(DecoderModel.class.getName());
  private final DecoderConfig config;
  private final ArchitectureDescriptor descriptor;
  private final ModelMetadata metadata;
  private final Embedding embedding;
  private final List<DecoderBlock> layers;
  private final RMSNorm norm;
  private final Linear lmHead;
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
      MLXArray normalized = normalizedHiddenStatesBatch(tokenIds, caches, validLengths);
      requirePostflightBatch(caches, validLengths.length);
      MLXArray logits = tiedOutput ? embedding.project(normalized) : lmHead.forward(normalized);
      MLXArray[] arrays = new MLXArray[1 + 2 * caches.size()];
      arrays[0] = logits;
      System.arraycopy(cacheArrays(caches), 0, arrays, 1, 2 * caches.size());
      stepBoundaryEvaluator.evaluate(arrays);
      return logits;
    } catch (RuntimeException | Error failure) {
      for (int layer = 0; layer < caches.size(); layer++) {
        KVCache cache = caches.get(layer);
        for (int row = 0; row < validLengths.length; row++) {
          if (cache.nextPosition(row) != before[layer][row]) {
            caches.forEach(KVCache::poison);
            throw failure;
          }
        }
      }
      throw failure;
    }
  }

  private int[][] preflightBatch(MLXArray tokenIds, List<KVCache> caches, int[] validLengths) {
    Objects.requireNonNull(tokenIds, "tokenIds");
    Objects.requireNonNull(validLengths, "validLengths");
    Objects.requireNonNull(caches, "caches");
    if (tokenIds.ndim() != 2
        || tokenIds.shape()[0] <= 0
        || tokenIds.shape()[1] <= 0
        || caches.size() != layers.size()
        || validLengths.length != tokenIds.shape()[0]) {
      throw new IllegalArgumentException(
          "batched forward requires nonempty [B,T], B lengths and one cache per layer");
    }
    int batch = tokenIds.shape()[0];
    int width = tokenIds.shape()[1];
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

  private MLXArray normalizedHiddenStatesBatch(
      MLXArray tokenIds, List<KVCache> caches, int[] validLengths) {
    MLXArray x = embedding.forward(tokenIds);
    if (descriptor.embedding().scaleBySqrtHidden()) {
      MLXArray scale =
          MLX.array(x.scope(), new float[] {(float) Math.sqrt(config.hiddenSize())}, new int[] {1});
      x = MLXOps.multiply(x, MLX.astype(scale, x.dtype()));
    }
    KVCache first = caches.getFirst();
    int batch = validLengths.length;
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
            tokenIds.shape()[1],
            keyWidth,
            window == null ? 0 : window);
    MLXArray frequencies =
        descriptor.rope().stepFrequencies(x.scope(), descriptor.rotaryDims(), maxEnding);
    for (int i = 0; i < layers.size(); i++) {
      x = layers.get(i).forward(x, caches.get(i), mask, frequencies, validLengths);
    }
    return norm.forward(x);
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
    for (int i = 0; i < caches.size(); i++) {
      Objects.requireNonNull(caches.get(i), "cache " + i);
    }
    MLXArray x = embedding.forward(tokenIds);
    if (descriptor.embedding().scaleBySqrtHidden()) {
      MLXArray scale =
          MLX.array(x.scope(), new float[] {(float) Math.sqrt(config.hiddenSize())}, new int[] {1});
      x = MLXOps.multiply(x, MLX.astype(scale, x.dtype()));
    }
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
    List<Integer> generated = new ArrayList<>();
    List<Double> logProbabilities = new ArrayList<>();
    StringBuilder generatedText = tokenizer == null ? null : new StringBuilder();
    IncrementalTokenDecoder decoder = null;
    LinkedHashMap<Integer, Integer> frequencies = PenaltyInputs.frequencies(prompt);
    FinishReason reason = FinishReason.MAX_TOKENS;
    if (!request.cancellationToken().isCancelled()) {
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
            MLXArray ids = MLX.array(activation, input, new int[] {1, input.length});
            int[] before = preflight(ids, caches);
            SamplingPipeline.Selection selection;
            try {
              MLXArray lastLogits = step == 0 ? prefill(ids, caches) : decode(ids, caches);
              selection =
                  sampler.select(
                      lastLogits,
                      PenaltyInputs.from(frequencies, config.vocabSize()),
                      step,
                      cacheArrays(caches));
            } catch (RuntimeException | Error failure) {
              poisonIfMutated(caches, before);
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
    } else {
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
            generatedText == null ? null : generatedText.toString());
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

  private MLXArray prefill(MLXArray ids, List<KVCache> caches) {
    MLXArray normalized = normalizedHiddenStates(ids, caches);
    requirePostflight(caches);
    int[] shape = normalized.shape();
    MLXArray last =
        MLXShape.slice(
            normalized, new int[] {0, shape[1] - 1, 0}, new int[] {shape[0], shape[1], shape[2]});
    return tiedOutput ? embedding.project(last) : lmHead.forward(last);
  }

  private MLXArray decode(MLXArray ids, List<KVCache> caches) {
    if (ids.shape()[1] != 1 || caches.get(0).nextPosition() == 0) {
      throw new IllegalArgumentException("decode requires one token and a populated cache");
    }
    MLXArray normalized = normalizedHiddenStates(ids, caches);
    requirePostflight(caches);
    return tiedOutput ? embedding.project(normalized) : lmHead.forward(normalized);
  }

  private static GenerationEvent tokenEvent(int tokenId, String textDelta, Double logProbability) {
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

  private static void validateTokenIds(int[] prompt, GenerationConfig policy, int vocabularySize) {
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
