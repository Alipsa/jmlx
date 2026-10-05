package se.alipsa.jmlx.models;

import se.alipsa.jmlx.nn.Activation;
import se.alipsa.jmlx.nn.RopeSpec;

/** Small architecture descriptors for pure-Java model tests. */
final class TestDescriptors {
  private TestDescriptors() {}

  static ArchitectureDescriptor llama(int layers, boolean attentionBias) {
    return llama(layers, attentionBias, false);
  }

  static ArchitectureDescriptor llama(int layers, boolean attentionBias, boolean tiedHead) {
    return dense("llama", layers, attentionBias, attentionBias, tiedHead, null, false);
  }

  static ArchitectureDescriptor qwen2(int layers) {
    return dense("qwen2", layers, true, false, false, null, false);
  }

  /**
   * Qwen3: no projection biases, per-head QK normalization, explicit head_dim path exercised by the
   * checkpoint-level tests.
   */
  static ArchitectureDescriptor qwen3(int layers) {
    return dense("qwen3", layers, false, false, false, null, true);
  }

  /** One llama layer declaring an MLX-style quantization block. */
  static ArchitectureDescriptor llamaWithQuantization(int groupSize, int bits) {
    return dense(
        "llama",
        1,
        false,
        false,
        false,
        new ArchitectureDescriptor.Quantization(groupSize, bits),
        false);
  }

  private static ArchitectureDescriptor dense(
      String type,
      int layers,
      boolean qkvBias,
      boolean outBias,
      boolean tiedHead,
      ArchitectureDescriptor.Quantization quantization,
      boolean qkNorm) {
    DecoderConfig dimensions =
        new DecoderConfig(type, 4, 4, 8, layers, 2, 1, 1e-6f, 10000, tiedHead, outBias, false);
    return new ArchitectureDescriptor(
        dimensions,
        2,
        2,
        new RopeSpec.Base(10000),
        new ArchitectureDescriptor.Norm(ArchitectureDescriptor.NormKind.RMS, 1e-6f, false),
        new ArchitectureDescriptor.Mlp(
            ArchitectureDescriptor.MlpLayout.SEPARATE_GATE_UP, Activation.SILU, false),
        new ArchitectureDescriptor.Attention(qkvBias, outBias, false, null, qkNorm),
        new ArchitectureDescriptor.Head(tiedHead),
        new ArchitectureDescriptor.Embedding(false),
        null,
        quantization);
  }
}
