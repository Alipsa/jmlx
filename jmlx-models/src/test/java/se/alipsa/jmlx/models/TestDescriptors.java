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
    return dense("llama", layers, attentionBias, attentionBias, tiedHead);
  }

  static ArchitectureDescriptor qwen2(int layers) {
    return dense("qwen2", layers, true, false, false);
  }

  private static ArchitectureDescriptor dense(
      String type, int layers, boolean qkvBias, boolean outBias, boolean tiedHead) {
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
        new ArchitectureDescriptor.Attention(qkvBias, outBias, false, null),
        new ArchitectureDescriptor.Head(tiedHead),
        new ArchitectureDescriptor.Embedding(false),
        null);
  }
}
