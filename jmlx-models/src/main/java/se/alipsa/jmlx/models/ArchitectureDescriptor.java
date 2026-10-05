package se.alipsa.jmlx.models;

import java.util.Objects;
import se.alipsa.jmlx.core.MLXQuant;
import se.alipsa.jmlx.nn.Activation;
import se.alipsa.jmlx.nn.RopeSpec;

/** Pure-Java description of the numerical capabilities required by a decoder checkpoint. */
public record ArchitectureDescriptor(
    DecoderConfig dimensions,
    int headDim,
    int rotaryDims,
    RopeSpec rope,
    Norm norm,
    Mlp mlp,
    Attention attention,
    Head head,
    Embedding embedding,
    Moe moe,
    Quantization quantization) {

  /** Validates dimensions and required component specifications. */
  public ArchitectureDescriptor {
    Objects.requireNonNull(dimensions, "dimensions");
    Objects.requireNonNull(rope, "rope");
    Objects.requireNonNull(norm, "norm");
    Objects.requireNonNull(mlp, "mlp");
    Objects.requireNonNull(attention, "attention");
    Objects.requireNonNull(head, "head");
    Objects.requireNonNull(embedding, "embedding");
    if (headDim <= 0 || rotaryDims <= 0 || rotaryDims > headDim || (rotaryDims & 1) != 0) {
      throw new IllegalArgumentException("invalid head_dim or rotary dimensions");
    }
    if (rope instanceof RopeSpec.DynamicNtk && rotaryDims <= 2) {
      throw new IllegalArgumentException("dynamic rope_scaling requires rotary dimensions > 2");
    }
  }

  /** Returns the model type represented by the common dimensions. */
  public String modelType() {
    return dimensions.modelType();
  }

  /** Supported normalization families. */
  public enum NormKind {
    RMS
  }

  /** Normalization epsilon and optional learned-weight offset. */
  public record Norm(NormKind kind, float eps, boolean weightOffset) {
    /** Validates normalization parameters. */
    public Norm {
      Objects.requireNonNull(kind, "kind");
      if (!Float.isFinite(eps) || eps <= 0) {
        throw new IllegalArgumentException("rms_norm_eps must be positive and finite");
      }
    }
  }

  /** Checkpoint projection layouts for a gated MLP. */
  public enum MlpLayout {
    SEPARATE_GATE_UP,
    FUSED_GATE_UP
  }

  /** MLP projection layout, activation, and bias policy. */
  public record Mlp(MlpLayout layout, Activation activation, boolean bias) {
    /** Validates MLP parameters. */
    public Mlp {
      Objects.requireNonNull(layout, "layout");
      Objects.requireNonNull(activation, "activation");
    }
  }

  /**
   * Attention projection layout, bias policy, optional sliding window, and optional per-head QK
   * normalization: {@code qkNorm} requires float {@code self_attn.q_norm.weight} and {@code
   * self_attn.k_norm.weight} tensors shaped {@code [head_dim]} and applies them over the head
   * dimension after projection and before RoPE (transformers' Qwen3 ordering).
   */
  public record Attention(
      boolean qkvBias, boolean outBias, boolean fusedQkv, Integer slidingWindow, boolean qkNorm) {
    /** Validates the optional window. */
    public Attention {
      if (slidingWindow != null && slidingWindow <= 0) {
        throw new IllegalArgumentException("sliding_window must be positive");
      }
    }
  }

  /** Whether the output head shares the embedding weights. */
  public record Head(boolean tied) {}

  /** Whether embeddings are scaled by the square root of hidden size. */
  public record Embedding(boolean scaleBySqrtHidden) {}

  /**
   * Affine weight quantization declared by an MLX-style {@code quantization} config block. Packing
   * is decided per layer by the checkpoint itself: a layer whose tensors include {@code .scales}
   * (and {@code .biases}) alongside {@code .weight} loads packed, and every other layer — norm
   * weights, biases, and projections whose width is not group-size compatible — stays float, so one
   * checkpoint can mix packed and float layers. A null {@code quantization} on the descriptor means
   * the config declares none; packed tensors in that case are rejected by {@code DecoderAssembler}.
   */
  public record Quantization(int groupSize, int bits) {
    /** Validates the group size and bit width against the sets the native runtime supports. */
    public Quantization {
      MLXQuant.checkGroupSize("quantization group_size", groupSize);
      MLXQuant.checkBits("quantization bits", bits);
    }
  }

  /** Expert count and number selected per token. */
  public record Moe(int experts, int topK) {
    /** Validates expert routing dimensions. */
    public Moe {
      if (experts < 1 || topK < 1 || topK > experts) {
        throw new IllegalArgumentException("invalid num_local_experts or num_experts_per_tok");
      }
    }
  }
}
