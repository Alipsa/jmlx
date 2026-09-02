package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
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
public abstract class DecoderModel extends Module {
  private final DecoderConfig config;
  private final Embedding embedding;
  private final List<DecoderBlock> layers;
  private final RMSNorm norm;
  private final Linear lmHead;
  private final boolean tiedOutput;

  protected DecoderModel(MLXScope scope, DecoderConfig config, Map<String, MLXArray> tensors) {
    super(scope);
    this.config = Objects.requireNonNull(config, "config");
    embedding =
        child("embedding", new Embedding(scope, tensor(tensors, "model.embed_tokens.weight")));
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
              tensors.get(p + "self_attn.q_proj.bias"),
              tensor(tensors, p + "self_attn.k_proj.weight"),
              tensors.get(p + "self_attn.k_proj.bias"),
              tensor(tensors, p + "self_attn.v_proj.weight"),
              tensors.get(p + "self_attn.v_proj.bias"),
              tensor(tensors, p + "self_attn.o_proj.weight"),
              tensors.get(p + "self_attn.o_proj.bias"));
      RMSNorm postNorm =
          new RMSNorm(
              scope, tensor(tensors, p + "post_attention_layernorm.weight"), config.rmsNormEps());
      SwiGLU mlp =
          new SwiGLU(
              scope,
              tensor(tensors, p + "mlp.gate_proj.weight"),
              tensors.get(p + "mlp.gate_proj.bias"),
              tensor(tensors, p + "mlp.up_proj.weight"),
              tensors.get(p + "mlp.up_proj.bias"),
              tensor(tensors, p + "mlp.down_proj.weight"),
              tensors.get(p + "mlp.down_proj.bias"));
      built.add(child("layer" + i, new DecoderBlock(scope, inputNorm, attention, postNorm, mlp)));
    }
    layers = List.copyOf(built);
    norm =
        child(
            "norm", new RMSNorm(scope, tensor(tensors, "model.norm.weight"), config.rmsNormEps()));
    MLXArray headWeight = tensors.get("lm_head.weight");
    tiedOutput = headWeight == null && config.tieWordEmbeddings();
    if (headWeight == null && !tiedOutput) {
      throw new IllegalArgumentException("checkpoint missing lm_head.weight");
    }
    lmHead = tiedOutput ? null : child("lmHead", new Linear(scope, headWeight, null));
  }

  public final DecoderConfig config() {
    return config;
  }

  /** Returns logits shaped {@code [batch, sequence, vocab]} and advances one cache per layer. */
  public final MLXArray forward(MLXArray tokenIds, List<KVCache> caches) {
    Objects.requireNonNull(tokenIds, "tokenIds");
    if (tokenIds.ndim() != 2)
      throw new IllegalArgumentException("tokenIds must have shape [batch, sequence]");
    if (caches.size() != layers.size())
      throw new IllegalArgumentException("one KVCache is required per decoder layer");
    MLXArray x = embedding.forward(tokenIds);
    for (int i = 0; i < layers.size(); i++) x = layers.get(i).forward(x, caches.get(i));
    MLXArray normalized = norm.forward(x);
    return tiedOutput ? embedding.project(normalized) : lmHead.forward(normalized);
  }

  /** Greedily generates up to {@code maxNewTokens}; prompt tokens are included in the result. */
  public final List<Integer> generate(int[] prompt, int maxNewTokens, Set<Integer> eosTokenIds) {
    if (prompt == null || prompt.length == 0)
      throw new IllegalArgumentException("prompt must not be empty");
    if (maxNewTokens < 0) throw new IllegalArgumentException("maxNewTokens must be non-negative");
    Objects.requireNonNull(eosTokenIds, "eosTokenIds");
    List<Integer> result = new ArrayList<>();
    for (int id : prompt) result.add(id);
    try (MLXScope generation = scope().newChild()) {
      List<KVCache> caches = new ArrayList<>();
      for (int i = 0; i < layers.size(); i++) caches.add(new KVCache(generation));
      int[] input = prompt;
      for (int step = 0; step < maxNewTokens; step++) {
        try (MLXScope activation = generation.newChild()) {
          MLXArray ids = MLX.array(activation, input, new int[] {1, input.length});
          MLXArray logits = forward(ids, caches);
          int[] shape = logits.shape();
          MLXArray last =
              MLXShape.slice(
                  logits, new int[] {0, shape[1] - 1, 0}, new int[] {1, shape[1], shape[2]});
          int next = MLXOps.argmaxAxis(last, 2, false).toIntArray()[0];
          result.add(next);
          if (eosTokenIds.contains(next)) break;
          input = new int[] {next};
        }
      }
    }
    return List.copyOf(result);
  }

  /**
   * Encodes {@code prompt}, greedily generates text, then decodes only the newly generated token
   * ids. For chat models, render the chat prompt with M2's {@code ChatTemplateRenderer} first.
   */
  public final String generateText(
      HfTokenizer tokenizer, String prompt, int maxNewTokens, Set<Integer> eosTokenIds) {
    Objects.requireNonNull(tokenizer, "tokenizer");
    Objects.requireNonNull(prompt, "prompt");
    List<Integer> promptIds = tokenizer.encode(prompt, true);
    int[] input = promptIds.stream().mapToInt(Integer::intValue).toArray();
    List<Integer> allIds = generate(input, maxNewTokens, eosTokenIds);
    return tokenizer.decode(allIds.subList(promptIds.size(), allIds.size()), true);
  }

  private static MLXArray tensor(Map<String, MLXArray> tensors, String name) {
    MLXArray tensor = tensors.get(name);
    if (tensor == null)
      throw new IllegalArgumentException("checkpoint missing tensor '" + name + "'");
    return tensor;
  }
}
