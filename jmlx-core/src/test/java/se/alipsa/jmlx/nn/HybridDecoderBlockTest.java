package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** HF differential checks for partial MRoPE, gated attention and both hybrid block variants. */
@EnabledIfNativeAvailable
@Tag("full-float32")
class HybridDecoderBlockTest {
  private static final float EPSILON = 1e-4f;

  @Test
  void partialRotaryPrefixMatchesHfAndTailPassesThroughExactly() {
    try (MLXScope scope = new MLXScope()) {
      Map<String, MLXArray> f = fixture(scope);
      for (String variant : new String[] {"base", "wide"}) {
        MultiAxisRotation rotation = rotation(scope, f, variant, false);
        assertArrayEquals(
            f.get(variant + ".cos").toFloatArray(), rotation.cos().toFloatArray(), 1e-6f);
        assertArrayEquals(
            f.get(variant + ".sin").toFloatArray(), rotation.sin().toFloatArray(), 1e-6f);
        for (String kind : new String[] {"query", "key"}) {
          MLXArray before = f.get(variant + "." + kind + "_normalized");
          MLXArray after = rotation.rotate(before);
          assertNear(f.get(variant + "." + kind + "_rotated"), after);
          int[] end = before.shape();
          int[] start = new int[end.length];
          start[3] = 16;
          assertArrayEquals(
              MLXShape.slice(before, start, end).toFloatArray(),
              MLXShape.slice(after, start, end).toFloatArray());
        }
      }
    }
  }

  @Test
  void perHeadGatedAttentionAndCachedDecodeMatchHfWithConfiguredHeadWidth() {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> f = fixture(weights);
      for (String variant : new String[] {"base", "wide"}) {
        GatedDecoderAttention attention = attention(weights, f, variant + ".attention.weights.");
        try (MLXScope request = weights.newChild()) {
          KVCache cache = new KVCache(request);
          try (MLXScope step = request.newChild()) {
            MLXArray input = inScope(f.get(variant + ".input"), step);
            MultiAxisRotation rotation = rotation(step, f, variant, false);
            MLXArray fresh = attention.forward(input, null, null, rotation);
            assertNear(f.get(variant + ".attention.output"), fresh);
            MLXArray output =
                attention.forward(
                    input, cache, AttentionMask.slidingWindow(step, 5, 5, 5), rotation);
            MLX.eval(output, cache.keys(), cache.values());
            assertNear(f.get(variant + ".attention.output"), output);
          }
          assertKeys(f, variant + ".attention.", cache);
          try (MLXScope step = request.newChild()) {
            MLXArray output =
                attention.forward(
                    inScope(f.get(variant + ".decode_input"), step),
                    cache,
                    null,
                    rotation(step, f, variant, true));
            MLX.eval(output, cache.keys(), cache.values());
            assertNear(f.get(variant + ".attention.decode_output"), output);
          }
          assertKeys(f, variant + ".attention.decode_", cache);
          assertEquals(6, cache.length());
        }
      }
    }
  }

  @Test
  void hybridFullAndLinearBlocksMatchHfPrefillAndDecode() {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> f = fixture(weights);
      for (String variant : new String[] {"base", "wide"}) {
        String full = variant + ".full_block.";
        String linear = variant + ".linear_block.";
        GatedDecoderAttention attention = attention(weights, f, full + "weights.self_attn.");
        HybridDecoderBlock fullBlock =
            new HybridDecoderBlock(
                weights,
                norm(weights, f, full + "weights.input_layernorm.weight"),
                attention,
                norm(weights, f, full + "weights.post_attention_layernorm.weight"),
                mlp(weights, f, full + "weights.mlp."));
        GatedDeltaNet delta = delta(weights, f, linear + "weights.linear_attn.");
        HybridDecoderBlock linearBlock =
            new HybridDecoderBlock(
                weights,
                norm(weights, f, linear + "weights.input_layernorm.weight"),
                delta,
                norm(weights, f, linear + "weights.post_attention_layernorm.weight"),
                mlp(weights, f, linear + "weights.mlp."));
        assertFalse(fullBlock.isLinear());
        assertTrue(linearBlock.isLinear());
        try (MLXScope request = weights.newChild();
            RecurrentCache recurrent = delta.newCache(request, 1)) {
          KVCache kv = new KVCache(request);
          for (boolean decode : new boolean[] {false, true}) {
            String suffix = decode ? "decode_" : "";
            try (MLXScope step = request.newChild()) {
              MLXArray input = inScope(f.get(variant + "." + suffix + "input"), step);
              MLXArray y = fullBlock.forward(input, kv, null, rotation(step, f, variant, decode));
              MLXArray z =
                  linearBlock.forward(input, recurrent, GatedDeltaNet.PrefillMethod.CHUNKED);
              MLX.eval(y, z, kv.keys(), kv.values(), recurrent.convolution(), recurrent.state());
              assertNear(f.get(full + suffix + "output"), y);
              assertNear(f.get(linear + suffix + "output"), z);
            }
            assertKeys(f, full + suffix, kv);
            assertNear(f.get(linear + suffix + "state"), recurrent.state());
            assertNear(f.get(linear + suffix + "convolution"), recurrent.convolution());
          }
          assertEquals(6, kv.length());
          assertEquals(6, recurrent.position());
        }
      }
    }
  }

  @Test
  void splitPrefillMatchesWholePrefixAndRejectsInvalidMaskBeforeCacheMutation() {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> f = fixture(weights);
      GatedDecoderAttention attention = attention(weights, f, "wide.attention.weights.");
      for (int split : new int[] {1, 3, 4}) {
        try (MLXScope request = weights.newChild()) {
          KVCache cache = new KVCache(request);
          for (int[] range : new int[][] {{0, split}, {split, 5}}) {
            try (MLXScope step = request.newChild()) {
              MLXArray input = slice(inScope(f.get("wide.input"), step), range[0], range[1]);
              int[][] positions = positions(f.get("wide.positions"), range[0], range[1]);
              MultiAxisRotation rotation =
                  new MultiAxisRope(10_000_000f, new int[] {3, 3, 2})
                      .prepare(step, 1, range[1] - range[0], positions);
              MLXArray output = attention.forward(input, cache, null, rotation);
              MLX.eval(output, cache.keys(), cache.values());
              assertNear(slice(f.get("wide.attention.output"), range[0], range[1]), output);
            }
          }
          assertKeys(f, "wide.attention.", cache);
        }
      }
      try (MLXScope request = weights.newChild();
          MLXScope step = request.newChild()) {
        KVCache cache = new KVCache(request);
        MLXArray input = inScope(f.get("wide.input"), step);
        MultiAxisRotation rotation = rotation(step, f, "wide", false);
        assertThrows(
            IllegalArgumentException.class,
            () -> attention.forward(input, cache, f.get("wide.input"), rotation));
        assertEquals(0, cache.length());
        assertFalse(cache.isPoisoned());
        MultiAxisRotation wrong =
            new MultiAxisRope(10_000_000f, new int[] {2, 2, 2})
                .prepare(step, 1, 5, positions(f.get("wide.positions"), 0, 5));
        assertThrows(
            IllegalArgumentException.class, () -> attention.forward(input, cache, null, wrong));
        assertEquals(0, cache.length());
      }
    }
  }

  private static Map<String, MLXArray> fixture(MLXScope scope) {
    return MLXIO
        .loadSafetensors(
            scope,
            Path.of("..", "tools", "qwen35-reference", "goldens", "gated-attention.safetensors")
                .toString())
        .tensors();
  }

  private static GatedDecoderAttention attention(MLXScope s, Map<String, MLXArray> f, String p) {
    return new GatedDecoderAttention(
        s,
        2,
        1,
        64,
        16,
        linear(s, f, p + "q_proj"),
        linear(s, f, p + "k_proj"),
        linear(s, f, p + "v_proj"),
        linear(s, f, p + "o_proj"),
        shifted(f.get(p + "q_norm.weight")),
        shifted(f.get(p + "k_norm.weight")),
        1e-6f);
  }

  private static GatedDeltaNet delta(MLXScope s, Map<String, MLXArray> f, String p) {
    return new GatedDeltaNet(
        s,
        2,
        6,
        32,
        64,
        linear(s, f, p + "in_proj_qkv"),
        linear(s, f, p + "in_proj_z"),
        linear(s, f, p + "in_proj_b"),
        linear(s, f, p + "in_proj_a"),
        MLXShape.transpose(f.get(p + "conv1d.weight"), new int[] {0, 2, 1}),
        f.get(p + "A_log"),
        f.get(p + "dt_bias"),
        f.get(p + "norm.weight"),
        1e-6f,
        linear(s, f, p + "out_proj"));
  }

  private static GatedMlp mlp(MLXScope s, Map<String, MLXArray> f, String p) {
    return new GatedMlp(
        s,
        linear(s, f, p + "gate_proj"),
        linear(s, f, p + "up_proj"),
        linear(s, f, p + "down_proj"),
        Activation.SILU);
  }

  private static Linear linear(MLXScope scope, Map<String, MLXArray> f, String p) {
    return new Linear(scope, f.get(p + ".weight"), null);
  }

  private static RMSNorm norm(MLXScope scope, Map<String, MLXArray> f, String p) {
    return new RMSNorm(scope, shifted(f.get(p)), 1e-6f);
  }

  private static MLXArray shifted(MLXArray a) {
    return MLXOps.add(a, MLX.array(a.scope(), new float[] {1f}, new int[] {1}));
  }

  private static MultiAxisRotation rotation(
      MLXScope scope, Map<String, MLXArray> f, String variant, boolean decode) {
    return new MultiAxisRope(10_000_000f, new int[] {3, 3, 2})
        .prepare(
            scope,
            1,
            decode ? 1 : 5,
            positions(
                f.get(variant + (decode ? ".decode_positions" : ".positions")), 0, decode ? 1 : 5));
  }

  private static int[][] positions(MLXArray a, int start, int end) {
    int[] all = a.toIntArray();
    int length = a.shape()[2];
    int[][] result = new int[3][end - start];
    for (int axis = 0; axis < 3; axis++) {
      System.arraycopy(all, axis * length + start, result[axis], 0, end - start);
    }
    return result;
  }

  private static MLXArray inScope(MLXArray a, MLXScope scope) {
    return MLXShape.reshape(MLXShape.expandDims(a, scope, 0), a.shape());
  }

  private static MLXArray slice(MLXArray a, int start, int end) {
    return MLXShape.slice(a, new int[] {0, start, 0}, new int[] {1, end, a.shape()[2]});
  }

  private static void assertKeys(Map<String, MLXArray> f, String stem, KVCache cache) {
    assertNear(f.get(stem + "keys"), cache.keys());
    assertNear(f.get(stem + "values"), cache.values());
  }

  private static void assertNear(MLXArray expected, MLXArray actual) {
    assertArrayEquals(expected.shape(), actual.shape());
    assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), EPSILON);
  }
}
