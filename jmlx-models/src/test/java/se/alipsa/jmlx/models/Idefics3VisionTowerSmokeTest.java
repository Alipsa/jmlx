package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/**
 * Native smoke for {@link Idefics3VisionTower#forward} and {@link Idefics3Connector#forward}: a
 * tiny deterministic tower (hidden 16, one layer, two heads, patch 8, P = 4, pixel-shuffle scale 2)
 * runs end to end on hand-made float32 tiles, asserting shapes, dtype, finiteness, the scope
 * contract and the input validation. No reference values are compared: the pure math vectors and
 * the pixel-shuffle permutation are venv-verified in {@link Idefics3VisionMathTest}.
 */
@EnabledIfNativeAvailable
class Idefics3VisionTowerSmokeTest {

  private static final int HIDDEN = 16;
  private static final int PATCH = 8;
  private static final int TILE = 32;
  private static final int SCALE = 2;

  private static Idefics3VisionTower buildTower(MLXScope root) {
    Random random = new Random(42);
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    weights.put("patchEmbedding.weight", randomTensor(root, random, HIDDEN, PATCH, PATCH, 3));
    weights.put("patchEmbedding.bias", randomTensor(root, random, HIDDEN));
    weights.put("positionEmbedding.weight", randomTensor(root, random, PATCH * PATCH, HIDDEN));
    for (String name : new String[] {"query", "key", "value", "output"}) {
      weights.put(
          "layer0.attention." + name + ".weight", randomTensor(root, random, HIDDEN, HIDDEN));
      weights.put("layer0.attention." + name + ".bias", randomTensor(root, random, HIDDEN));
    }
    for (String norm : new String[] {"attentionNorm", "outputNorm"}) {
      weights.put("layer0." + norm + ".weight", randomTensor(root, random, HIDDEN));
      weights.put("layer0." + norm + ".bias", randomTensor(root, random, HIDDEN));
    }
    weights.put("layer0.feedForward.0.weight", randomTensor(root, random, 2 * HIDDEN, HIDDEN));
    weights.put("layer0.feedForward.0.bias", randomTensor(root, random, 2 * HIDDEN));
    weights.put("layer0.feedForward.2.weight", randomTensor(root, random, HIDDEN, 2 * HIDDEN));
    weights.put("layer0.feedForward.2.bias", randomTensor(root, random, HIDDEN));
    weights.put("postLayernorm.weight", randomTensor(root, random, HIDDEN));
    weights.put("postLayernorm.bias", randomTensor(root, random, HIDDEN));
    Idefics3Connector connector =
        new Idefics3Connector(
            root, SCALE, randomTensor(root, random, HIDDEN, HIDDEN * SCALE * SCALE));
    return new Idefics3VisionTower(root, 1, 2, 1e-6f, weights, connector);
  }

  private static MLXArray randomPixels(MLXScope scope, Random random, int height) {
    return MLX.array(
        scope, randomFloats(random, 2 * height * TILE * 3), new int[] {2, height, TILE, 3});
  }

  private static float[] randomFloats(Random random, int length) {
    float[] values = new float[length];
    for (int i = 0; i < length; i++) {
      values[i] = random.nextFloat() - 0.5f;
    }
    return values;
  }

  private static MLXArray randomTensor(MLXScope scope, Random random, int... shape) {
    int length = 1;
    for (int size : shape) {
      length *= size;
    }
    return MLX.array(scope, randomFloats(random, length), shape);
  }

  private static boolean allFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  @Test
  void forwardAndConnectorProduceFiniteShapes() {
    int[] mask = new int[TILE / PATCH * (TILE / PATCH)];
    Arrays.fill(mask, 1);
    try (MLXScope root = new MLXScope()) {
      Idefics3VisionTower tower = buildTower(root);
      assertTrue(tower.parameters().containsKey("connector.modalityProjection.weight"));
      Random random = new Random(43);
      try (MLXScope step = root.newChild()) {
        MLXArray out = tower.forward(randomPixels(step, random, TILE), mask);
        assertArrayEquals(new int[] {2, TILE / PATCH * (TILE / PATCH), HIDDEN}, out.shape());
        assertEquals(DType.FLOAT32, out.dtype());
        assertTrue(allFinite(out.toFloatArray()));
        MLXArray features = tower.connector().forward(out);
        assertArrayEquals(
            new int[] {2, (TILE / PATCH / SCALE) * (TILE / PATCH / SCALE), HIDDEN},
            features.shape());
        assertTrue(allFinite(features.toFloatArray()));
      }
      // A fresh child scope with fresh pixels: activations allocate into the pixels' scope.
      try (MLXScope other = root.newChild()) {
        MLXArray out = tower.forward(randomPixels(other, random, TILE), mask);
        assertArrayEquals(new int[] {2, TILE / PATCH * (TILE / PATCH), HIDDEN}, out.shape());
        assertTrue(allFinite(out.toFloatArray()));
      }
    }
  }

  @Test
  void forwardRejectsInvalidPixelsAndMasks() {
    int[] mask = new int[TILE / PATCH * (TILE / PATCH)];
    Arrays.fill(mask, 1);
    try (MLXScope root = new MLXScope()) {
      Idefics3VisionTower tower = buildTower(root);
      Random random = new Random(44);
      try (MLXScope step = root.newChild()) {
        MLXArray good = randomPixels(step, random, TILE);
        MLXArray wrongChannels =
            MLX.array(
                step, randomFloats(random, 2 * TILE * TILE * 4), new int[] {2, TILE, TILE, 4});
        assertThrows(IllegalArgumentException.class, () -> tower.forward(wrongChannels, mask));
        int[] notBinary = mask.clone();
        notBinary[3] = 2;
        assertThrows(IllegalArgumentException.class, () -> tower.forward(good, notBinary));
        assertThrows(
            IllegalArgumentException.class, () -> tower.forward(good, new int[] {1, 1, 1}));
        MLXArray shortHeight = randomPixels(step, random, 30);
        assertThrows(IllegalArgumentException.class, () -> tower.forward(shortHeight, mask));
      }
    }
  }
}
