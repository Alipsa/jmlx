package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.CachedAttention;
import se.alipsa.jmlx.nn.DecoderAttention;
import se.alipsa.jmlx.nn.DecoderBlock;
import se.alipsa.jmlx.nn.Embedding;
import se.alipsa.jmlx.nn.GatedMlp;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.MoeMlp;
import se.alipsa.jmlx.nn.RMSNorm;
import se.alipsa.jmlx.nn.UnaryLayer;

/** Constructs the registered decoder modules from a validated architecture and checkpoint. */
public final class DecoderAssembler {
  private DecoderAssembler() {}

  /** Components to register under the decoder's stable child names. */
  public record Assembled(
      Embedding embedding, List<DecoderBlock> layers, RMSNorm finalNorm, Linear lmHead) {
    /** Copies the layer list so callers cannot change the assembled component set. */
    public Assembled {
      layers = List.copyOf(layers);
    }
  }

  /** Builds a decoder after confirming that loaded tensor keys still satisfy the header plan. */
  public static Assembled assemble(
      MLXScope scope, ArchitectureDescriptor descriptor, Map<String, MLXArray> tensors) {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(descriptor, "descriptor");
    Objects.requireNonNull(tensors, "tensors");
    ArchitectureMappings.tensorPlan(descriptor).validate(tensors.keySet());
    MLXArray staticFreqs = staticRopeFrequencies(scope, descriptor);
    Embedding embedding = new Embedding(scope, tensor(tensors, "model.embed_tokens.weight"));
    List<DecoderBlock> layers = new ArrayList<>();
    for (int i = 0; i < descriptor.dimensions().numHiddenLayers(); i++) {
      String prefix = "model.layers." + i + ".";
      RMSNorm input = norm(scope, descriptor, tensors, prefix + "input_layernorm.weight");
      String attentionPrefix = prefix + "self_attn.";
      Linear[] qkv =
          descriptor.attention().fusedQkv()
              ? fusedQkv(scope, descriptor, tensors, attentionPrefix + "qkv_proj.weight")
              : new Linear[] {
                projection(
                    scope, tensors, attentionPrefix + "q_proj", descriptor.attention().qkvBias()),
                projection(
                    scope, tensors, attentionPrefix + "k_proj", descriptor.attention().qkvBias()),
                projection(
                    scope, tensors, attentionPrefix + "v_proj", descriptor.attention().qkvBias())
              };
      CachedAttention attention =
          new DecoderAttention(
              scope,
              descriptor.dimensions().numAttentionHeads(),
              descriptor.dimensions().numKeyValueHeads(),
              descriptor.headDim(),
              descriptor.rope(),
              descriptor.rotaryDims(),
              staticFreqs,
              descriptor.attention().slidingWindow(),
              qkv[0],
              qkv[1],
              qkv[2],
              projection(
                  scope, tensors, attentionPrefix + "o_proj", descriptor.attention().outBias()));
      RMSNorm post = norm(scope, descriptor, tensors, prefix + "post_attention_layernorm.weight");
      UnaryLayer mlp =
          descriptor.moe() == null
              ? denseMlp(scope, descriptor, tensors, prefix + "mlp.")
              : moeMlp(scope, descriptor, tensors, prefix + "block_sparse_moe.");
      layers.add(new DecoderBlock(scope, input, attention, post, mlp));
    }
    RMSNorm finalNorm = norm(scope, descriptor, tensors, "model.norm.weight");
    MLXArray headWeight = tensors.get("lm_head.weight");
    if (headWeight == null && !descriptor.head().tied()) {
      throw new IllegalArgumentException("checkpoint missing lm_head.weight");
    }
    Linear lmHead = headWeight == null ? null : new Linear(scope, headWeight, null);
    return new Assembled(embedding, layers, finalNorm, lmHead);
  }

  private static RMSNorm norm(
      MLXScope scope, ArchitectureDescriptor d, Map<String, MLXArray> tensors, String key) {
    return new RMSNorm(scope, tensor(tensors, key), d.norm().eps(), d.norm().weightOffset());
  }

  private static UnaryLayer denseMlp(
      MLXScope scope,
      ArchitectureDescriptor descriptor,
      Map<String, MLXArray> tensors,
      String prefix) {
    Linear[] gateUp =
        descriptor.mlp().layout() == ArchitectureDescriptor.MlpLayout.FUSED_GATE_UP
            ? fusedGateUp(scope, descriptor, tensors, prefix + "gate_up_proj.weight")
            : new Linear[] {
              projection(scope, tensors, prefix + "gate_proj", descriptor.mlp().bias()),
              projection(scope, tensors, prefix + "up_proj", descriptor.mlp().bias())
            };
    return new GatedMlp(
        scope,
        gateUp[0],
        gateUp[1],
        projection(scope, tensors, prefix + "down_proj", descriptor.mlp().bias()),
        descriptor.mlp().activation());
  }

  private static UnaryLayer moeMlp(
      MLXScope scope,
      ArchitectureDescriptor descriptor,
      Map<String, MLXArray> tensors,
      String prefix) {
    List<GatedMlp> experts = new ArrayList<>();
    for (int expert = 0; expert < descriptor.moe().experts(); expert++) {
      String p = prefix + "experts." + expert + ".";
      experts.add(
          new GatedMlp(
              scope,
              projection(scope, tensors, p + "w1", false),
              projection(scope, tensors, p + "w3", false),
              projection(scope, tensors, p + "w2", false),
              descriptor.mlp().activation()));
    }
    return new MoeMlp(
        scope,
        projection(scope, tensors, prefix + "gate", false),
        experts,
        descriptor.moe().topK());
  }

  private static MLXArray staticRopeFrequencies(MLXScope scope, ArchitectureDescriptor d) {
    if (!(d.rope() instanceof se.alipsa.jmlx.nn.RopeSpec.Llama3)
        && !(d.rope() instanceof se.alipsa.jmlx.nn.RopeSpec.Yarn)) {
      return null;
    }
    double[] periods = d.rope().frequencies(d.rotaryDims(), 1);
    float[] data = new float[periods.length];
    for (int i = 0; i < data.length; i++) {
      data[i] = (float) periods[i];
    }
    return MLX.array(scope, data, new int[] {data.length});
  }

  private static Linear[] fusedQkv(
      MLXScope scope, ArchitectureDescriptor d, Map<String, MLXArray> tensors, String key) {
    MLXArray weight = tensor(tensors, key);
    int hidden = d.dimensions().hiddenSize();
    int queryRows = d.dimensions().numAttentionHeads() * d.headDim();
    int kvRows = d.dimensions().numKeyValueHeads() * d.headDim();
    requireShape(weight, key, queryRows + 2 * kvRows, hidden);
    return new Linear[] {
      new Linear(
          scope, MLXShape.slice(weight, new int[] {0, 0}, new int[] {queryRows, hidden}), null),
      new Linear(
          scope,
          MLXShape.slice(weight, new int[] {queryRows, 0}, new int[] {queryRows + kvRows, hidden}),
          null),
      new Linear(
          scope,
          MLXShape.slice(
              weight,
              new int[] {queryRows + kvRows, 0},
              new int[] {queryRows + 2 * kvRows, hidden}),
          null)
    };
  }

  private static Linear[] fusedGateUp(
      MLXScope scope, ArchitectureDescriptor d, Map<String, MLXArray> tensors, String key) {
    MLXArray weight = tensor(tensors, key);
    int intermediate = d.dimensions().intermediateSize();
    int hidden = d.dimensions().hiddenSize();
    requireShape(weight, key, 2 * intermediate, hidden);
    return new Linear[] {
      new Linear(
          scope, MLXShape.slice(weight, new int[] {0, 0}, new int[] {intermediate, hidden}), null),
      new Linear(
          scope,
          MLXShape.slice(weight, new int[] {intermediate, 0}, new int[] {2 * intermediate, hidden}),
          null)
    };
  }

  private static void requireShape(MLXArray weight, String key, int rows, int columns) {
    if (weight.ndim() != 2 || weight.shape()[0] != rows || weight.shape()[1] != columns) {
      throw new IllegalArgumentException(
          "checkpoint tensor '" + key + "' must have shape [" + rows + ", " + columns + "]");
    }
  }

  private static Linear projection(
      MLXScope scope, Map<String, MLXArray> tensors, String prefix, boolean biasRequired) {
    MLXArray bias = tensors.get(prefix + ".bias");
    if (biasRequired && bias == null) {
      throw new IllegalArgumentException(
          "checkpoint missing required bias tensor '" + prefix + ".bias'");
    }
    return new Linear(scope, tensor(tensors, prefix + ".weight"), bias);
  }

  private static MLXArray tensor(Map<String, MLXArray> tensors, String name) {
    MLXArray result = tensors.get(name);
    if (result == null) {
      throw new IllegalArgumentException("checkpoint missing tensor '" + name + "'");
    }
    return result;
  }
}
