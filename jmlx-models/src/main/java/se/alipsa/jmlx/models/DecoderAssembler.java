package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import se.alipsa.jmlx.core.DType;
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
import se.alipsa.jmlx.nn.SwitchGlu;
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

  /**
   * Builds a decoder after confirming that loaded tensor keys still satisfy the header plan. For
   * mixture-of-experts layers, the per-expert {@code block_sparse_moe.experts.*} arrays in {@code
   * tensors} are stacked and then closed, so the caller must not use them afterwards. Every layer's
   * expert tensors are checked for shape and one shared dtype before the first is closed, so a
   * checkpoint that fails <em>that</em> check leaves {@code tensors} untouched. Other failures (a
   * malformed non-expert tensor, a truncated shard that only surfaces while stacking, running out
   * of memory) can occur after earlier layers' expert tensors were closed, so treat {@code tensors}
   * as unusable after any exception.
   */
  public static Assembled assemble(
      MLXScope scope, ArchitectureDescriptor descriptor, Map<String, MLXArray> tensors) {
    Objects.requireNonNull(scope, "scope");
    Objects.requireNonNull(descriptor, "descriptor");
    Objects.requireNonNull(tensors, "tensors");
    ArchitectureMappings.tensorPlan(descriptor).validate(tensors.keySet());
    if (descriptor.moe() != null) {
      // All layers are validated before any is stacked: stacking closes the per-expert sources.
      for (int i = 0; i < descriptor.dimensions().numHiddenLayers(); i++) {
        validateExperts(descriptor, tensors, "model.layers." + i + ".block_sparse_moe.");
      }
    }
    MLXArray staticFreqs = descriptor.rope().staticFrequencies(scope, descriptor.rotaryDims());
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
    // An explicit lm_head.weight wins over tie_word_embeddings: fine-tunes may leave the flag set.
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
    int experts = descriptor.moe().experts();
    MLXArray[][] sources = {
      expertTensors(tensors, prefix, "w1", experts),
      expertTensors(tensors, prefix, "w3", experts),
      expertTensors(tensors, prefix, "w2", experts)
    };
    MLXArray gate = MLXShape.stack(sources[0], 0);
    MLXArray up = MLXShape.stack(sources[1], 0);
    MLXArray down = MLXShape.stack(sources[2], 0);
    // One sync per layer, then release the per-expert sources: peak weight memory stays near 1x
    // plus one layer instead of 2x (req/plans/phase6-3-performance.md, Verified facts 6).
    MLX.eval(gate, up, down);
    for (MLXArray[] kind : sources) {
      for (MLXArray part : kind) {
        part.close();
      }
    }
    SwitchGlu stacked = new SwitchGlu(scope, gate, up, down, descriptor.mlp().activation());
    return new MoeMlp(
        scope,
        projection(scope, tensors, prefix + "gate", false),
        stacked,
        descriptor.moe().topK());
  }

  /**
   * Checks every expert tensor of one layer by key: its shape, and that all of them share one dtype
   * so stacking cannot silently promote the layer (gate, up and down are multiplied together).
   */
  private static void validateExperts(
      ArchitectureDescriptor descriptor, Map<String, MLXArray> tensors, String prefix) {
    int experts = descriptor.moe().experts();
    int hidden = descriptor.dimensions().hiddenSize();
    int intermediate = descriptor.dimensions().intermediateSize();
    DType expected = null;
    for (String kind : new String[] {"w1", "w3", "w2"}) {
      boolean down = kind.equals("w2");
      for (int e = 0; e < experts; e++) {
        String key = prefix + "experts." + e + "." + kind + ".weight";
        MLXArray weight = tensor(tensors, key);
        requireShape(weight, key, down ? hidden : intermediate, down ? intermediate : hidden);
        if (expected == null) {
          expected = weight.dtype();
        } else if (weight.dtype() != expected) {
          throw new IllegalArgumentException(
              "checkpoint tensor '"
                  + key
                  + "' has dtype "
                  + weight.dtype()
                  + " but this layer's other expert weights are "
                  + expected);
        }
      }
    }
  }

  private static MLXArray[] expertTensors(
      Map<String, MLXArray> tensors, String prefix, String kind, int experts) {
    MLXArray[] parts = new MLXArray[experts];
    for (int e = 0; e < experts; e++) {
      parts[e] = tensor(tensors, prefix + "experts." + e + "." + kind + ".weight");
    }
    return parts;
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
