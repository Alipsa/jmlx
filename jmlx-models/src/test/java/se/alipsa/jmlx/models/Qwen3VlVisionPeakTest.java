package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXMemoryTestSupport;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.TwoAxisRope;
import se.alipsa.jmlx.nn.TwoAxisRotaryAttention;
import se.alipsa.jmlx.nn.TwoAxisRotation;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessor;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessorResult;
import se.alipsa.jmlx.vision.Qwen3VlProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;

/**
 * Measured per-image peak of the WP5 vision tower ({@code
 * req/plans/phase7-3b-reference-findings.md} §11.4). Runs the real-2B tower geometry — hidden 1024,
 * 16 heads, head_dim 64, intermediate 4096, out_hidden 2048, 2304 position embeddings (48×48),
 * patch/temporal 16/2, merge 2 — at reduced depth 3 with DeepStack index [1]. The depth reduction
 * is documented in §11.4: the measured peak is attention-workspace-dominated and depth-independent
 * (the score-matrix budget does not scale with depth), so three blocks bracket the per-block
 * transient that every real-depth block incurs. The weights are constant 0.001 via {@link MLX#full}
 * — the peak depends on geometry and dtype, not on values. One synthetic 1024×1024 image (grid (1,
 * 64, 64), 4096 patch rows, the default 4096-row raw budget's exact hard cap) is processed through
 * {@link Qwen3VlImageProcessor}.
 *
 * <p>The sustained per-image evaluated peak settles a baseline {@link
 * MLXMemoryTestSupport#settledActiveBytes()} after every static array (tower weights, the bracket
 * attention's weights, input and rotation) has been evaluated, runs the tower forward over the
 * image (its per-image child scope closes inside the forward), settles again, and takes the
 * difference — the hoisted merged and DeepStack rows that outlive the forward. The transient
 * per-image attention workspace is then bracketed by a direct single {@link TwoAxisRotaryAttention}
 * forward at 4096 rows with constant weights: {@link MLXMemory#resetPeak()} before, {@link
 * MLXMemory#peakBytes()} after the evaluated forward. Both measurements must stay below the plan §4
 * attention-workspace planning bound of 5 GiB (one float32 score matrix at 4096 rows / 16 heads is
 * 1 GiB).
 */
@EnabledIfNativeAvailable
class Qwen3VlVisionPeakTest {

  private static final int HIDDEN = 1024;
  private static final int NUM_HEADS = 16;
  private static final int HEAD_DIM = 64;
  private static final int INTERMEDIATE = 4096;
  private static final int OUT_HIDDEN = 2048;
  private static final int POSITION_EMBEDDINGS = 2304;
  private static final int DEPTH = 3;
  private static final int[] DEEPSTACK_INDEXES = {1};
  private static final int MERGE = 2;
  private static final int MERGE_BLOCK = HIDDEN * MERGE * MERGE;
  private static final int IMAGE = 1024;
  private static final int GRID = IMAGE / 16;
  private static final int PATCH_ROWS = GRID * GRID;
  private static final int MERGED_ROWS = (GRID / MERGE) * (GRID / MERGE);
  private static final long GIB = 1024L * 1024L * 1024L;
  private static final long PLANNING_BOUND = 5 * GIB;

  @Test
  void realGeometryForwardPeakStaysUnderPlanningBound() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      List<MLXArray> statics = new ArrayList<>();
      final Qwen3VlVisionTower tower = buildTower(scope, statics);
      // The transient bracket: one standalone attention of the same geometry with constant
      // weights, whose input rows and prepared rotation live in this scope too, so the baseline
      // below counts every array the bracket forward reads.
      final TwoAxisRotaryAttention attention =
          new TwoAxisRotaryAttention(
              scope,
              NUM_HEADS,
              HEAD_DIM,
              Qwen3VlVisionTower.ROPE_THETA,
              full(statics, scope, new int[] {3 * HIDDEN, HIDDEN}),
              full(statics, scope, new int[] {3 * HIDDEN}),
              full(statics, scope, new int[] {HIDDEN, HIDDEN}),
              full(statics, scope, new int[] {HIDDEN}));
      final MLXArray attentionInput = full(statics, scope, new int[] {PATCH_ROWS, HIDDEN});
      int[][] coordinates = Qwen3VlVisionTower.rotaryCoordinates(GRID, GRID, MERGE);
      final TwoAxisRotation rotation =
          new TwoAxisRope(Qwen3VlVisionTower.ROPE_THETA, HEAD_DIM / 2)
              .prepare(scope, PATCH_ROWS, coordinates[0], coordinates[1]);

      Qwen3VlImageProcessor processor = processor();
      Qwen3VlImageProcessorResult image = processor.process(syntheticImage());
      assertArrayEquals(new int[] {1, GRID, GRID}, image.gridThw(), "grid");
      assertEquals(PATCH_ROWS, image.rawPatchRows(), "patch rows");
      assertEquals(MERGED_ROWS, image.mergedRowCount(), "merged rows");

      // Everything the two measurements read is static: materialize it (and the rotation
      // tables) before the baseline so only the forwards' own work moves the counter.
      List<MLXArray> toMaterialize = new ArrayList<>(statics);
      toMaterialize.add(rotation.cos());
      toMaterialize.add(rotation.sin());
      MLX.eval(toMaterialize.toArray(MLXArray[]::new));
      MLXMemoryTestSupport.settledActiveBytes(); // warm settle before the baseline
      final long baseline = MLXMemoryTestSupport.settledActiveBytes();

      Qwen3VlVisionFeatures features = tower.forward(List.of(image));
      assertArrayEquals(new int[] {MERGED_ROWS, OUT_HIDDEN}, features.merged().shape(), "merged");
      assertEquals(DEEPSTACK_INDEXES.length, features.deepstackLevels().size(), "deepstack levels");
      assertArrayEquals(
          new int[] {MERGED_ROWS, OUT_HIDDEN},
          features.deepstackLevels().get(0).shape(),
          "deepstack level 0");
      long settledAfter = MLXMemoryTestSupport.settledActiveBytes();
      long sustained = settledAfter - baseline;
      long logicalRetained = 2L * MERGED_ROWS * OUT_HIDDEN * Float.BYTES;
      assertTrue(
          sustained >= logicalRetained,
          "sustained peak "
              + sustained
              + " B below the logical retained rows "
              + logicalRetained
              + " B");
      assertTrue(
          sustained < PLANNING_BOUND,
          "sustained peak "
              + sustained
              + " B exceeded the "
              + PLANNING_BOUND
              + " B planning bound");

      MLXMemory.resetPeak();
      MLX.eval(attention.forward(attentionInput, rotation));
      long transientDelta = MLXMemory.peakBytes() - baseline;
      assertTrue(
          transientDelta < PLANNING_BOUND,
          "attention transient "
              + transientDelta
              + " B exceeded the "
              + PLANNING_BOUND
              + " B planning bound");

      System.out.println(
          "WP5 vision peak: grid=(1,"
              + GRID
              + ","
              + GRID
              + "), sustained="
              + sustained
              + " B ("
              + sustained / (1024 * 1024)
              + " MiB), attention transient="
              + transientDelta
              + " B ("
              + String.format(Locale.ROOT, "%.2f", transientDelta / (double) GIB)
              + " GiB), bound="
              + PLANNING_BOUND / GIB
              + " GiB");
    }
  }

  /** The 2B artifact's slow-processor options, with maxPixels at the default 4096-row hard cap. */
  private static Qwen3VlImageProcessor processor() {
    return new Qwen3VlImageProcessor(
        Qwen3VlProcessorConfig.builder()
            .patchSize(16)
            .temporalPatchSize(2)
            .mergeSize(2)
            .minPixels(3136)
            .maxPixels(4096 * 16 * 16)
            .imageMean(new float[] {0.5f, 0.5f, 0.5f})
            .imageStd(new float[] {0.5f, 0.5f, 0.5f})
            .build());
  }

  /** One deterministic 1024×1024 image: per-pixel linear functions of the coordinates. */
  private static RgbImage syntheticImage() {
    byte[] pixels = new byte[IMAGE * IMAGE * 3];
    for (int y = 0; y < IMAGE; y++) {
      for (int x = 0; x < IMAGE; x++) {
        int i = (y * IMAGE + x) * 3;
        pixels[i] = (byte) ((x * 7 + y * 11) % 256);
        pixels[i + 1] = (byte) ((x * 13 + y * 5) % 256);
        pixels[i + 2] = (byte) ((x * 19 + y * 3) % 256);
      }
    }
    return new RgbImage(IMAGE, IMAGE, pixels);
  }

  /** The real-2B geometry at reduced depth 3 / DeepStack [1], constant 0.001 weights. */
  private static Qwen3VlVisionTower buildTower(MLXScope scope, List<MLXArray> statics) {
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    put(weights, statics, scope, "patchEmbedding.weight", HIDDEN, 2, 16, 16, 3);
    put(weights, statics, scope, "patchEmbedding.bias", HIDDEN);
    put(weights, statics, scope, "positionEmbedding.weight", POSITION_EMBEDDINGS, HIDDEN);
    for (int i = 0; i < DEPTH; i++) {
      String layer = "layer" + i + ".";
      put(weights, statics, scope, layer + "attention.qkv.weight", 3 * HIDDEN, HIDDEN);
      put(weights, statics, scope, layer + "attention.qkv.bias", 3 * HIDDEN);
      put(weights, statics, scope, layer + "attention.proj.weight", HIDDEN, HIDDEN);
      put(weights, statics, scope, layer + "attention.proj.bias", HIDDEN);
      for (String norm : new String[] {"norm1", "norm2"}) {
        put(weights, statics, scope, layer + norm + ".weight", HIDDEN);
        put(weights, statics, scope, layer + norm + ".bias", HIDDEN);
      }
      put(weights, statics, scope, layer + "mlp.0.weight", INTERMEDIATE, HIDDEN);
      put(weights, statics, scope, layer + "mlp.0.bias", INTERMEDIATE);
      put(weights, statics, scope, layer + "mlp.2.weight", HIDDEN, INTERMEDIATE);
      put(weights, statics, scope, layer + "mlp.2.bias", HIDDEN);
    }
    put(weights, statics, scope, "merger.norm.weight", HIDDEN);
    put(weights, statics, scope, "merger.norm.bias", HIDDEN);
    put(weights, statics, scope, "merger.linearFc1.weight", MERGE_BLOCK, MERGE_BLOCK);
    put(weights, statics, scope, "merger.linearFc1.bias", MERGE_BLOCK);
    put(weights, statics, scope, "merger.linearFc2.weight", OUT_HIDDEN, MERGE_BLOCK);
    put(weights, statics, scope, "merger.linearFc2.bias", OUT_HIDDEN);
    put(weights, statics, scope, "deepstackMerger0.norm.weight", MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.norm.bias", MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.linearFc1.weight", MERGE_BLOCK, MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.linearFc1.bias", MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.linearFc2.weight", OUT_HIDDEN, MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.linearFc2.bias", OUT_HIDDEN);
    return new Qwen3VlVisionTower(scope, NUM_HEADS, weights, DEEPSTACK_INDEXES);
  }

  private static void put(
      Map<String, MLXArray> weights,
      List<MLXArray> statics,
      MLXScope scope,
      String key,
      int... shape) {
    weights.put(key, full(statics, scope, shape));
  }

  private static MLXArray full(List<MLXArray> statics, MLXScope scope, int... shape) {
    MLXArray value = MLX.full(scope, shape, 0.001f, DType.FLOAT32);
    statics.add(value);
    return value;
  }
}
