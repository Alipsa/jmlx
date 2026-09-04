package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Arrays;
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
import se.alipsa.jmlx.nn.DecoderBlock;
import se.alipsa.jmlx.nn.Embedding;
import se.alipsa.jmlx.nn.GroupedQueryAttention;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.Module;
import se.alipsa.jmlx.nn.RMSNorm;
import se.alipsa.jmlx.nn.SwiGLU;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/** Inference-only pre-norm decoder shared by Llama and Qwen2 checkpoints. */
public abstract class DecoderModel extends Module implements TextGenerationModel {
  private static final System.Logger LOGGER = System.getLogger(DecoderModel.class.getName());
  private final DecoderConfig config;
  private final ModelMetadata metadata;
  private final Embedding embedding;
  private final List<DecoderBlock> layers;
  private final RMSNorm norm;
  private final Linear lmHead;
  private final boolean tiedOutput;

  /**
   * Builds every layer from {@code tensors}, keyed by their Hugging Face checkpoint names. {@code
   * qkvBiasRequired}/{@code outBiasRequired} are constructor arguments, not methods a subclass
   * overrides: reading them from an overridable hook here would run before the subclass's own field
   * initializers (the classic Java construction-order hazard) -- see {@link QwenModel}'s and {@link
   * LlamaModel}'s private constructors for how each architecture derives them. Neither simply reads
   * {@link DecoderConfig#attentionBias()} directly: Qwen2 never defines that config.json field
   * (hardcoding q/k/v bias -- but never o_proj bias -- in HF's modeling code instead), so a
   * hand-edited Qwen2 config that happened to set it would otherwise reject an o_proj-bias-less
   * checkpoint that is, in fact, perfectly valid.
   */
  protected DecoderModel(
      MLXScope scope,
      DecoderConfig config,
      Map<String, MLXArray> tensors,
      boolean qkvBiasRequired,
      boolean outBiasRequired) {
    super(scope);
    this.config = Objects.requireNonNull(config, "config");
    metadata =
        new DecoderMetadata(config.modelType(), config.vocabSize(), config.numHiddenLayers());
    embedding =
        child("embedding", new Embedding(scope, tensor(tensors, "model.embed_tokens.weight")));
    boolean mlpBiasExpected = config.mlpBias();
    List<DecoderBlock> built = new ArrayList<>();
    for (int i = 0; i < config.numHiddenLayers(); i++) {
      String p = "model.layers." + i + ".";
      RMSNorm inputNorm =
          new RMSNorm(scope, tensor(tensors, p + "input_layernorm.weight"), config.rmsNormEps());
      GroupedQueryAttention attention =
          new GroupedQueryAttention(
              scope,
              config.numAttentionHeads(),
              config.numKeyValueHeads(),
              config.ropeTheta(),
              tensor(tensors, p + "self_attn.q_proj.weight"),
              bias(tensors, p + "self_attn.q_proj.bias", qkvBiasRequired),
              tensor(tensors, p + "self_attn.k_proj.weight"),
              bias(tensors, p + "self_attn.k_proj.bias", qkvBiasRequired),
              tensor(tensors, p + "self_attn.v_proj.weight"),
              bias(tensors, p + "self_attn.v_proj.bias", qkvBiasRequired),
              tensor(tensors, p + "self_attn.o_proj.weight"),
              bias(tensors, p + "self_attn.o_proj.bias", outBiasRequired));
      RMSNorm postNorm =
          new RMSNorm(
              scope, tensor(tensors, p + "post_attention_layernorm.weight"), config.rmsNormEps());
      SwiGLU mlp =
          new SwiGLU(
              scope,
              tensor(tensors, p + "mlp.gate_proj.weight"),
              bias(tensors, p + "mlp.gate_proj.bias", mlpBiasExpected),
              tensor(tensors, p + "mlp.up_proj.weight"),
              bias(tensors, p + "mlp.up_proj.bias", mlpBiasExpected),
              tensor(tensors, p + "mlp.down_proj.weight"),
              bias(tensors, p + "mlp.down_proj.bias", mlpBiasExpected));
      built.add(child("layer" + i, new DecoderBlock(scope, inputNorm, attention, postNorm, mlp)));
    }
    layers = List.copyOf(built);
    norm =
        child(
            "norm", new RMSNorm(scope, tensor(tensors, "model.norm.weight"), config.rmsNormEps()));
    MLXArray headWeight = tensors.get("lm_head.weight");
    // Prefer an explicit output head: some converted fine-tunes leave the source model's
    // tie_word_embeddings flag set after untying and training lm_head.
    tiedOutput = headWeight == null && config.tieWordEmbeddings();
    if (!tiedOutput && headWeight == null) {
      throw new IllegalArgumentException("checkpoint missing lm_head.weight");
    }
    lmHead = tiedOutput ? null : child("lmHead", new Linear(scope, headWeight, null));
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
    MLXArray x = embedding.forward(tokenIds);
    for (int i = 0; i < layers.size(); i++) {
      x = layers.get(i).forward(x, Objects.requireNonNull(caches.get(i), "cache " + i));
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
   * Runs the currently supported deterministic greedy policy and emits token plus terminal events.
   */
  @Override
  public final GenerationResult generate(
      GenerationRequest request, Consumer<GenerationEvent> listener) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(listener, "listener");
    GenerationConfig policy = request.config();
    requireGreedy(policy);
    int[] prompt = request.promptTokenIds();
    List<Integer> generated = new ArrayList<>();
    FinishReason reason = FinishReason.MAX_TOKENS;
    if (!request.cancellationToken().isCancelled()) {
      try (MLXScope generation = scope().newChild()) {
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
            int next = MLXOps.argmaxAxis(lastLogits, 2, false).toIntArray()[0];
            boolean eos = policy.eosTokenIds().contains(next);
            if (!eos && policy.stopTokenIds().contains(next)) {
              reason = FinishReason.STOP_TOKEN;
              break;
            }
            generated.add(next);
            try {
              listener.accept(GenerationEvent.token(next));
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
    GenerationResult result = new GenerationResult(toList(prompt), generated, reason, List.of());
    try {
      listener.accept(GenerationEvent.finished(reason));
    } catch (RuntimeException e) {
      LOGGER.log(System.Logger.Level.WARNING, "generation terminal listener failed", e);
    }
    return result;
  }

  private static List<Integer> toList(int[] ids) {
    return Arrays.stream(ids).boxed().toList();
  }

  private static void requireGreedy(GenerationConfig config) {
    if (config.temperature() != 0
        || config.topK() != 0
        || config.topP() != 1
        || config.minP() != 0
        || config.repetitionPenalty() != 1
        || config.frequencyPenalty() != 0
        || config.presencePenalty() != 0
        || config.seed().isPresent()
        || config.logProbabilities()) {
      throw new UnsupportedOperationException(
          "this release supports only greedy policy values: temperature=0, topK=0, topP=1, "
              + "minP=0, repetitionPenalty=1, frequencyPenalty=0, presencePenalty=0, no seed, "
              + "and no log probabilities");
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
    List<Integer> promptIds = tokenizer.encode(prompt, addSpecialTokens);
    int[] input = promptIds.stream().mapToInt(Integer::intValue).toArray();
    List<Integer> allIds = generate(input, maxNewTokens, eosTokenIds);
    return tokenizer.decode(allIds.subList(promptIds.size(), allIds.size()), true);
  }

  private static MLXArray tensor(Map<String, MLXArray> tensors, String name) {
    MLXArray tensor = tensors.get(name);
    if (tensor == null) {
      throw new IllegalArgumentException("checkpoint missing tensor '" + name + "'");
    }
    return tensor;
  }

  /**
   * Returns the bias tensor {@code name}, or {@code null} if this architecture's config does not
   * call for one. Throws if {@code expected} is {@code true} but the checkpoint lacks it -- a
   * silently-null bias here would otherwise compute wrong attention/MLP output with no diagnostic.
   */
  private static MLXArray bias(Map<String, MLXArray> tensors, String name, boolean expected) {
    MLXArray bias = tensors.get(name);
    if (expected && bias == null) {
      throw new IllegalArgumentException("checkpoint missing required bias tensor '" + name + "'");
    }
    return bias;
  }
}
