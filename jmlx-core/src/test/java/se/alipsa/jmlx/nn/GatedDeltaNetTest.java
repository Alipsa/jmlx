package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** Complete DeltaNet projection/convolution/normalization/gate and retained-state HF reference. */
@EnabledIfNativeAvailable
@Tag("full-float32")
class GatedDeltaNetTest {
  private static final float EPSILON = 1e-4f;

  @Test
  void explicitPrefillMethodsAndDecodeMatchHfAcrossAllBoundaries() {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> fixture = fixture(weights);
      GatedDeltaNet net = net(weights, fixture, null);
      for (GatedDeltaNet.PrefillMethod method : GatedDeltaNet.PrefillMethod.values()) {
        for (int length : new int[] {1, 3, 4, 5, 63, 64, 65, 127, 128, 129}) {
          String stem = "t" + length + ".";
          try (MLXScope request = weights.newChild();
              RecurrentCache cache = net.newCache(request, 1)) {
            try (MLXScope step = request.newChild()) {
              MLXArray output =
                  net.forward(inScope(fixture.get(stem + "input"), step), cache, method);
              assertArrayEquals(
                  fixture.get(stem + "output").toFloatArray(),
                  output.toFloatArray(),
                  EPSILON,
                  stem + method);
            }
            assertState(fixture, stem, cache);
            assertEquals(length, cache.position());
            try (MLXScope step = request.newChild()) {
              MLXArray output =
                  net.forward(inScope(fixture.get(stem + "decode_input"), step), cache, method);
              assertArrayEquals(
                  fixture.get(stem + "decode_output").toFloatArray(),
                  output.toFloatArray(),
                  EPSILON);
            }
            assertState(fixture, stem + "decode_", cache);
            assertEquals(length + 1, cache.position());
          }
        }
      }
    }
  }

  @Test
  void splitPrefillsKeepRealConvolutionHistoryAndWholeRecurrentState() {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> fixture = fixture(weights);
      GatedDeltaNet net = net(weights, fixture, null);
      for (int split : new int[] {1, 3, 4, 63, 64, 65, 128}) {
        try (MLXScope request = weights.newChild();
            RecurrentCache cache = net.newCache(request, 1)) {
          MLXArray source = fixture.get("t129.input");
          try (MLXScope step = request.newChild()) {
            net.forward(
                positions(inScope(source, step), 0, split),
                cache,
                GatedDeltaNet.PrefillMethod.CHUNKED);
          }
          try (MLXScope step = request.newChild()) {
            MLXArray output =
                net.forward(
                    positions(inScope(source, step), split, 129),
                    cache,
                    GatedDeltaNet.PrefillMethod.CHUNKED);
            assertArrayEquals(
                positions(fixture.get("t129.output"), split, 129).toFloatArray(),
                output.toFloatArray(),
                EPSILON);
          }
          assertState(fixture, "t129.", cache);
          assertEquals(129, cache.position());
        }
      }
    }
  }

  @Test
  void outputFailurePoisonsCacheWithoutCommittingPositions() {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> fixture = fixture(weights);
      UnaryLayer failing =
          new UnaryLayer(weights) {
            @Override
            public MLXArray forward(MLXArray x) {
              throw new IllegalStateException("injected output failure");
            }
          };
      GatedDeltaNet net = net(weights, fixture, failing);
      try (MLXScope request = weights.newChild();
          RecurrentCache cache = net.newCache(request, 1);
          MLXScope step = request.newChild()) {
        MLXArray x = inScope(fixture.get("t3.input"), step);
        assertThrows(
            IllegalStateException.class,
            () -> net.forward(x, cache, GatedDeltaNet.PrefillMethod.CHUNKED));
        assertTrue(cache.isPoisoned());
        assertEquals(0, cache.position());
        assertThrows(
            IllegalStateException.class,
            () -> net.forward(x, cache, GatedDeltaNet.PrefillMethod.CHUNKED));
      }
    }
  }

  private static Map<String, MLXArray> fixture(MLXScope scope) {
    return MLXIO
        .loadSafetensors(
            scope,
            Path.of("..", "tools", "qwen35-reference", "goldens", "delta-net.safetensors")
                .toString())
        .tensors();
  }

  private static GatedDeltaNet net(MLXScope scope, Map<String, MLXArray> f, UnaryLayer output) {
    return new GatedDeltaNet(
        scope,
        2,
        6,
        32,
        64,
        linear(scope, f, "in_proj_qkv"),
        linear(scope, f, "in_proj_z"),
        linear(scope, f, "in_proj_b"),
        linear(scope, f, "in_proj_a"),
        MLXShape.transpose(f.get("weights.conv1d.weight"), new int[] {0, 2, 1}),
        f.get("weights.A_log"),
        f.get("weights.dt_bias"),
        f.get("weights.norm.weight"),
        1e-6f,
        output == null ? linear(scope, f, "out_proj") : output);
  }

  private static Linear linear(MLXScope scope, Map<String, MLXArray> fixture, String name) {
    return new Linear(scope, fixture.get("weights." + name + ".weight"), null);
  }

  private static void assertState(
      Map<String, MLXArray> fixture, String stem, RecurrentCache cache) {
    assertArrayEquals(
        fixture.get(stem + "state").toFloatArray(),
        cache.state().toFloatArray(),
        EPSILON,
        stem + "state");
    assertArrayEquals(
        fixture.get(stem + "convolution").toFloatArray(),
        cache.convolution().toFloatArray(),
        EPSILON,
        stem + "convolution");
  }

  private static MLXArray inScope(MLXArray a, MLXScope scope) {
    return MLXShape.reshape(MLXShape.expandDims(a, scope, 0), a.shape());
  }

  private static MLXArray positions(MLXArray a, int start, int end) {
    return MLXShape.slice(a, new int[] {0, start, 0}, new int[] {1, end, 128});
  }
}
