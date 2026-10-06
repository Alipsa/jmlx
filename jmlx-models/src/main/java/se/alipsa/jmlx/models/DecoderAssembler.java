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
import se.alipsa.jmlx.nn.EmbeddingLayer;
import se.alipsa.jmlx.nn.GatedMlp;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.MoeMlp;
import se.alipsa.jmlx.nn.QuantizedEmbedding;
import se.alipsa.jmlx.nn.QuantizedLinear;
import se.alipsa.jmlx.nn.RMSNorm;
import se.alipsa.jmlx.nn.SwitchGlu;
import se.alipsa.jmlx.nn.UnaryLayer;

/** Constructs the registered decoder modules from a validated architecture and checkpoint. */
public final class DecoderAssembler {
  private static final System.Logger LOGGER = System.getLogger(DecoderAssembler.class.getName());

  private DecoderAssembler() {}

  /** Components to register under the decoder's stable child names. */
  public record Assembled(
      EmbeddingLayer embedding, List<DecoderBlock> layers, RMSNorm finalNorm, UnaryLayer lmHead) {
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
    validatePackedTensors(descriptor, tensors);
    MLXArray staticFreqs = descriptor.rope().staticFrequencies(scope, descriptor.rotaryDims());
    ArchitectureDescriptor.Quantization quantization = descriptor.quantization();
    EmbeddingLayer embedding =
        !tensors.containsKey("model.embed_tokens.scales")
            ? new Embedding(scope, tensor(tensors, "model.embed_tokens.weight"))
            : new QuantizedEmbedding(
                scope,
                tensor(tensors, "model.embed_tokens.weight"),
                tensor(tensors, "model.embed_tokens.scales"),
                tensor(tensors, "model.embed_tokens.biases"),
                quantization.groupSize(),
                quantization.bits());
    List<DecoderBlock> layers = new ArrayList<>();
    for (int i = 0; i < descriptor.dimensions().numHiddenLayers(); i++) {
      String prefix = "model.layers." + i + ".";
      RMSNorm input = norm(scope, descriptor, tensors, prefix + "input_layernorm.weight");
      String attentionPrefix = prefix + "self_attn.";
      UnaryLayer[] qkv =
          descriptor.attention().fusedQkv()
              ? fusedQkv(scope, descriptor, tensors, attentionPrefix + "qkv_proj")
              : new UnaryLayer[] {
                projection(
                    scope,
                    descriptor,
                    tensors,
                    attentionPrefix + "q_proj",
                    descriptor.attention().qkvBias()),
                projection(
                    scope,
                    descriptor,
                    tensors,
                    attentionPrefix + "k_proj",
                    descriptor.attention().qkvBias()),
                projection(
                    scope,
                    descriptor,
                    tensors,
                    attentionPrefix + "v_proj",
                    descriptor.attention().qkvBias())
              };
      RMSNorm queryNorm =
          descriptor.attention().qkNorm()
              ? norm(scope, descriptor, tensors, attentionPrefix + "q_norm.weight")
              : null;
      RMSNorm keyNorm =
          descriptor.attention().qkNorm()
              ? norm(scope, descriptor, tensors, attentionPrefix + "k_norm.weight")
              : null;
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
                  scope,
                  descriptor,
                  tensors,
                  attentionPrefix + "o_proj",
                  descriptor.attention().outBias()),
              queryNorm,
              keyNorm);
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
    UnaryLayer lmHead =
        headWeight == null ? null : projection(scope, descriptor, tensors, "lm_head", false);
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
    UnaryLayer[] gateUp =
        descriptor.mlp().layout() == ArchitectureDescriptor.MlpLayout.FUSED_GATE_UP
            ? fusedGateUp(scope, descriptor, tensors, prefix + "gate_up_proj")
            : new UnaryLayer[] {
              projection(scope, descriptor, tensors, prefix + "gate_proj", descriptor.mlp().bias()),
              projection(scope, descriptor, tensors, prefix + "up_proj", descriptor.mlp().bias())
            };
    return new GatedMlp(
        scope,
        gateUp[0],
        gateUp[1],
        projection(scope, descriptor, tensors, prefix + "down_proj", descriptor.mlp().bias()),
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
        projection(scope, descriptor, tensors, prefix + "gate", false),
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

  private static UnaryLayer[] fusedQkv(
      MLXScope scope, ArchitectureDescriptor d, Map<String, MLXArray> tensors, String prefix) {
    int hidden = d.dimensions().hiddenSize();
    int queryRows = d.dimensions().numAttentionHeads() * d.headDim();
    int kvRows = d.dimensions().numKeyValueHeads() * d.headDim();
    checkFusedShape(tensors, prefix, queryRows + 2 * kvRows, hidden);
    return new UnaryLayer[] {
      rowSlice(scope, d, tensors, prefix, 0, queryRows),
      rowSlice(scope, d, tensors, prefix, queryRows, queryRows + kvRows),
      rowSlice(scope, d, tensors, prefix, queryRows + kvRows, queryRows + 2 * kvRows)
    };
  }

  private static UnaryLayer[] fusedGateUp(
      MLXScope scope, ArchitectureDescriptor d, Map<String, MLXArray> tensors, String prefix) {
    int intermediate = d.dimensions().intermediateSize();
    checkFusedShape(tensors, prefix, 2 * intermediate, d.dimensions().hiddenSize());
    return new UnaryLayer[] {
      rowSlice(scope, d, tensors, prefix, 0, intermediate),
      rowSlice(scope, d, tensors, prefix, intermediate, 2 * intermediate)
    };
  }

  /** Float weights are {@code [rows, hidden]}; quantized ones are packed along the columns. */
  private static void checkFusedShape(
      Map<String, MLXArray> tensors, String prefix, int rows, int hidden) {
    MLXArray weight = tensor(tensors, prefix + ".weight");
    if (!tensors.containsKey(prefix + ".scales")) {
      requireShape(weight, prefix + ".weight", rows, hidden);
    } else if (weight.ndim() != 2 || weight.shape()[0] != rows) {
      throw new IllegalArgumentException(
          "checkpoint tensor '" + prefix + ".weight' must have " + rows + " rows");
    }
  }

  /** Rows {@code [from, to)} of a fused projection, as its own layer (no bias). */
  private static UnaryLayer rowSlice(
      MLXScope scope,
      ArchitectureDescriptor d,
      Map<String, MLXArray> tensors,
      String prefix,
      int from,
      int to) {
    MLXArray weight = rows(tensor(tensors, prefix + ".weight"), from, to);
    ArchitectureDescriptor.Quantization q = d.quantization();
    if (!tensors.containsKey(prefix + ".scales")) {
      return new Linear(scope, weight, null);
    }
    return new QuantizedLinear(
        scope,
        weight,
        rows(tensor(tensors, prefix + ".scales"), from, to),
        rows(tensor(tensors, prefix + ".biases"), from, to),
        null,
        q.groupSize(),
        q.bits());
  }

  private static MLXArray rows(MLXArray array, int from, int to) {
    return MLXShape.slice(array, new int[] {from, 0}, new int[] {to, array.shape()[1]});
  }

  private static void requireShape(MLXArray weight, String key, int rows, int columns) {
    if (weight.ndim() != 2 || weight.shape()[0] != rows || weight.shape()[1] != columns) {
      throw new IllegalArgumentException(
          "checkpoint tensor '" + key + "' must have shape [" + rows + ", " + columns + "]");
    }
  }

  private static UnaryLayer projection(
      MLXScope scope,
      ArchitectureDescriptor d,
      Map<String, MLXArray> tensors,
      String prefix,
      boolean biasRequired) {
    MLXArray bias = tensors.get(prefix + ".bias");
    if (biasRequired && bias == null) {
      throw new IllegalArgumentException(
          "checkpoint missing required bias tensor '" + prefix + ".bias'");
    }
    ArchitectureDescriptor.Quantization q = d.quantization();
    if (!tensors.containsKey(prefix + ".scales")) {
      return new Linear(scope, tensor(tensors, prefix + ".weight"), bias);
    }
    return new QuantizedLinear(
        scope,
        tensor(tensors, prefix + ".weight"),
        tensor(tensors, prefix + ".scales"),
        tensor(tensors, prefix + ".biases"),
        bias,
        q.groupSize(),
        q.bits());
  }

  /**
   * Validates dimensions before slicing packed arrays or constructing any decoder layers. Package
   * visible rather than private so tests can reach the defensive branches every public path
   * preempts: {@link #assemble} runs the tensor plan validation first, which already rejects a
   * {@code .scales} tensor on a descriptor that declares no quantization and any weight name the
   * architecture does not know.
   */
  static void validatePackedTensors(ArchitectureDescriptor d, Map<String, MLXArray> tensors) {
    boolean sawPackedTensor = false;
    for (String key : tensors.keySet()) {
      if (!key.endsWith(".weight")) {
        continue;
      }
      String prefix = key.substring(0, key.length() - ".weight".length());
      MLXArray scales = tensors.get(prefix + ".scales");
      MLXArray offsets = tensors.get(prefix + ".biases");
      if (scales == null) {
        if (offsets != null || tensors.get(key).dtype() == DType.UINT32) {
          throw new IllegalArgumentException("checkpoint missing tensor '" + prefix + ".scales'");
        }
        continue;
      }
      sawPackedTensor = true;
      ArchitectureDescriptor.Quantization q = d.quantization();
      if (q == null) {
        throw new IllegalArgumentException(
            "checkpoint tensor '"
                + prefix
                + ".scales' present but config declares no quantization");
      }
      int input = packedInputWidth(d, prefix);
      MLXArray weight = tensors.get(key);
      if (scales.ndim() != 2 || (long) scales.shape()[1] * q.groupSize() != input) {
        throw new IllegalArgumentException(
            "checkpoint tensor '"
                + prefix
                + ".scales' is incompatible with group_size "
                + q.groupSize()
                + " and input size "
                + input);
      }
      if (weight.ndim() != 2 || (long) weight.shape()[1] * 32 != (long) input * q.bits()) {
        throw new IllegalArgumentException(
            "checkpoint tensor '"
                + key
                + "' is incompatible with bits "
                + q.bits()
                + " and input size "
                + input);
      }
      requireShape(scales, prefix + ".scales", weight.shape()[0], input / q.groupSize());
      requireShape(
          tensor(tensors, prefix + ".biases"),
          prefix + ".biases",
          weight.shape()[0],
          input / q.groupSize());
    }
    if (d.quantization() != null && !sawPackedTensor) {
      // Every .scales-less weight above fell back to a float layer, so a checkpoint that was
      // never actually quantized loads silently as float. The reverse mismatches (packed data
      // without a declared block, missing .scales for packed data) are hard errors; this one
      // is only silently wrong in the sense that the declared packing is ignored -- hence a
      // warning rather than a rejection, which would break legitimately mixed float checkpoints.
      LOGGER.log(
          System.Logger.Level.WARNING,
          "config declares quantization (group_size {0}, bits {1}) but the checkpoint has no"
              + " packed .scales tensors; every layer loads as float",
          d.quantization().groupSize(),
          d.quantization().bits());
    }
  }

  /**
   * The input width a packed projection's group count must match, keyed by the projection name. An
   * unrecognized {@code .scales}-bearing name fails loudly here rather than defaulting to {@code
   * hiddenSize}, so a future family with differently named projections cannot be silently validated
   * against the wrong width.
   */
  private static int packedInputWidth(ArchitectureDescriptor d, String prefix) {
    if (prefix.endsWith(".down_proj")) {
      return d.dimensions().intermediateSize();
    }
    if (prefix.endsWith(".o_proj")) {
      return d.dimensions().numAttentionHeads() * d.headDim();
    }
    if (prefix.endsWith(".q_proj")
        || prefix.endsWith(".k_proj")
        || prefix.endsWith(".v_proj")
        || prefix.endsWith(".qkv_proj")
        || prefix.endsWith(".gate_proj")
        || prefix.endsWith(".up_proj")
        || prefix.endsWith(".gate_up_proj")
        || prefix.endsWith(".embed_tokens")
        || prefix.equals("lm_head")) {
      return d.dimensions().hiddenSize();
    }
    throw new IllegalArgumentException(
        "checkpoint tensor '"
            + prefix
            + ".scales' names no projection with a known input width for this architecture");
  }

  private static MLXArray tensor(Map<String, MLXArray> tensors, String name) {
    MLXArray result = tensors.get(name);
    if (result == null) {
      throw new IllegalArgumentException("checkpoint missing tensor '" + name + "'");
    }
    return result;
  }
}
