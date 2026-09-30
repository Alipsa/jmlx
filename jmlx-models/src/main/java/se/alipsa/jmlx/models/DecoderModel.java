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
   * call; generation creates such caches automatically.
   */
  public final MLXArray forward(MLXArray tokenIds, List<KVCache> caches) {
    Objects.requireNonNull(tokenIds, "tokenIds");
    if (tokenIds.ndim() != 2) {
      throw new IllegalArgumentException("tokenIds must have shape [batch, sequence]");
    }
    Objects.requireNonNull(caches, "caches");
    if (caches.size() != layers.size()) {
      throw new IllegalArgumentException("one KVCache is required per decoder layer");
    }
    MLXArray normalized = normalizedHiddenStates(tokenIds, caches);
    return tiedOutput ? embedding.project(normalized) : lmHead.forward(normalized);
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
    MLXArray mask =
        window != null && window < caches.get(0).offset() + tokenIds.shape()[1]
            ? AttentionMask.slidingWindow(
                x.scope(),
                tokenIds.shape()[1],
                caches.get(0).offset() + tokenIds.shape()[1],
                window)
            : null;
    MLXArray stepFrequencies =
        descriptor
            .rope()
            .stepFrequencies(
                x.scope(), descriptor.rotaryDims(), caches.get(0).offset() + tokenIds.shape()[1]);
    for (int i = 0; i < layers.size(); i++) {
      x = layers.get(i).forward(x, caches.get(i), mask, stepFrequencies);
    }
    return norm.forward(x);
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
          SamplingPipeline sampler = new SamplingPipeline(generation, policy, config.vocabSize())) {
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
          caches.add(new KVCache(generation));
        }
        int[] input = prompt;
        for (int step = 0; step < policy.maxNewTokens(); step++) {
          if (request.cancellationToken().isCancelled()) {
            reason = FinishReason.CANCELLED;
            break;
          }
          try (MLXScope activation = generation.newChild()) {
            MLXArray ids = MLX.array(activation, input, new int[] {1, input.length});
            MLXArray normalized = normalizedHiddenStates(ids, caches);
            int[] shape = normalized.shape();
            MLXArray lastHiddenState =
                MLXShape.slice(
                    normalized,
                    new int[] {0, shape[1] - 1, 0},
                    new int[] {shape[0], shape[1], shape[2]});
            MLXArray lastLogits =
                tiedOutput ? embedding.project(lastHiddenState) : lmHead.forward(lastHiddenState);
            SamplingPipeline.Selection selection =
                sampler.select(
                    lastLogits, PenaltyInputs.from(frequencies, config.vocabSize()), step);
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
