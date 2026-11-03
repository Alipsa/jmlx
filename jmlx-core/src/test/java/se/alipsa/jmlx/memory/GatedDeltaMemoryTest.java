package se.alipsa.jmlx.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.nn.GatedDeltaNet;
import se.alipsa.jmlx.nn.Linear;
import se.alipsa.jmlx.nn.RecurrentCache;

/** Native allocation and model-scope handle regression for evaluated hybrid recurrent inference. */
@EnabledIfNativeAvailable
class GatedDeltaMemoryTest {
  @Test
  void repeatedAndLongRequestsReleaseStateAndNeverAllocateInTheParameterScope() {
    try (MLXScope weights = new MLXScope()) {
      Map<String, MLXArray> fixture =
          MLXIO
              .loadSafetensors(
                  weights,
                  Path.of("..", "tools", "qwen35-reference", "goldens", "delta-net.safetensors")
                      .toString())
              .tensors();
      GatedDeltaNet net = net(weights, fixture);
      MLXArray shortInput = fixture.get("t64.input");
      MLXArray longInput = MLXShape.tile(fixture.get("t129.input"), new int[] {1, 32, 1});
      MLX.eval(fixture.values().toArray(MLXArray[]::new));
      MLX.eval(longInput);
      int parameterHandles = weights.handleCount();
      run(weights, net, shortInput);
      long baseline = MLXMemory.activeBytes();
      for (int i = 0; i < 16; i++) {
        run(weights, net, shortInput);
        assertEquals(
            parameterHandles, weights.handleCount(), "forward retained parameter-scope handles");
        assertTrue(
            MLXMemory.activeBytes() - baseline <= 1024 * 1024,
            "completed requests retained native arrays");
      }
      MLXMemory.resetPeak();
      run(weights, net, longInput);
      long peak = MLXMemory.peakBytes() - baseline;
      assertEquals(parameterHandles, weights.handleCount());
      assertTrue(
          MLXMemory.activeBytes() - baseline <= 1024 * 1024, "long request retained native arrays");
      // Whole-prompt projections/output scale with T. The bounded solver graph must not scale
      // with the number of 64-position chunks; allow its fixed overhead separately.
      long activationAllowance = 16L * longInput.shape()[1] * (128 + 512 + 384);
      assertTrue(
          peak <= activationAllowance + 32L * 1024 * 1024,
          "temporary state exceeded the bounded-chunk allowance: " + peak);
    }
  }

  private static void run(MLXScope weights, GatedDeltaNet net, MLXArray input) {
    try (MLXScope request = weights.newChild();
        RecurrentCache cache = net.newCache(request, 1)) {
      try (MLXScope step = request.newChild()) {
        MLXArray x = MLXShape.reshape(MLXShape.expandDims(input, step, 0), input.shape());
        net.forward(x, cache, GatedDeltaNet.PrefillMethod.CHUNKED);
      }
      long logical = cache.logicalStorageBytes();
      assertEquals(4L * (6 * 32 * 64 + 3 * 512), logical);
      assertTrue(MLXMemory.activeBytes() >= logical);
    }
  }

  private static GatedDeltaNet net(MLXScope scope, Map<String, MLXArray> f) {
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
        linear(scope, f, "out_proj"));
  }

  private static Linear linear(MLXScope scope, Map<String, MLXArray> f, String name) {
    return new Linear(scope, f.get("weights." + name + ".weight"), null);
  }
}
