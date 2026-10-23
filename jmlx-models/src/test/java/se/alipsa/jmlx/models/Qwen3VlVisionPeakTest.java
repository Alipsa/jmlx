package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.LocalDate;
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
 * patch/temporal 16/2, merge 2 — at three reduced depths, each with one DeepStack index (depth 1:
 * index 0; depths 3 and 4: index 1). The depth reduction is documented in §11.4: the tower's
 * per-block evaluate-and-release loop confines every block's intermediates to a private per-block
 * child scope that closes before the next block starts, so the per-image peak does not scale with
 * depth — the pre-fix tower leaked ≈1.09 GiB per block (the full depth-24 tower extrapolated ≈26.8
 * GiB). The depth-1-versus-depth-4 transient comparison below is the regression guard for that: the
 * delta must stay under 64 MiB, so the depth-3 transient is a fair proxy for the per-image
 * transient the plan §4 bound applies to at any depth. The weights are constant 0.001 via {@link
 * MLX#full} — the peak depends on geometry and dtype, not on values. One synthetic 1024×1024 image
 * (grid (1, 64, 64), 4096 patch rows, the default 4096-row raw budget's exact hard cap) is
 * processed through {@link Qwen3VlImageProcessor}.
 *
 * <p>Four tower measurements share one settled baseline, taken with every static array (all three
 * towers' weights, the bracket attention's weights, inputs and rotations) already evaluated:
 *
 * <ol>
 *   <li>The tower forward's own per-image transient peak at depth 1: the depth-independent floor —
 *       one block's intermediates plus the conv, the position-embedding gathers and the mergers,
 *       each evaluated and released inside the forward.
 *   <li>The same transient at the depth-3 proxy: must stay under the plan §4 planning bound of 5
 *       GiB and at least the 16 MiB of evaluated rows the forward hoists into the per-request
 *       scope.
 *   <li>The sustained per-image evaluated peak at depth 3: a settled active-bytes delta taken while
 *       the per-request scope is still open — the per-block and per-image scopes close inside the
 *       forward, and the hoisted merged and DeepStack rows live in the per-request scope until it
 *       closes, as in production — the rows the per-request scope retains for the request.
 *   <li>The same transient at depth 4; together with depth 1 it bounds per-block accumulation — the
 *       difference must stay under 64 MiB, so a deeper tower (the real depth 24) incurs no larger
 *       per-image transient than the depth-3 proxy.
 * </ol>
 *
 * <p>A fifth bracket, the transient per-image attention workspace of one standalone {@link
 * TwoAxisRotaryAttention} forward at 4096 rows with constant weights (run after every per-request
 * scope closes, so it carries no retained-row offset), must also stay under the plan §4 bound (one
 * float32 score matrix at 4096 rows / 16 heads is 1 GiB).
 *
 * <p>CI headroom: this suite is required by the {@code macos-26} hosted runner's native job in
 * {@code .github/workflows/ci.yml} and runs in the ordinary {@code test} task, where it is the
 * module's largest memory consumer. Its largest transient is the depth-1 peak, 1,757,020,216 B
 * (≈1.64 GiB, M5 Max, 2026-10-23), on top of the settled statics of the three towers and the
 * bracket (≈0.65 GiB), so the concurrent Metal peak is ≈2.3 GiB — the per-block
 * evaluate-and-release fix (this review round) dropped it from the pre-fix depth-3 peak of 3.81
 * GiB. Checked against the hosted runner's ≈7 GB RAM (a reviewer-recorded figure, not verified
 * against the runner image): ≈2.3 GiB Metal + ≤0.5 GiB test-JVM working set (no configured max
 * heap; the JVM's default ¼-RAM max is a ceiling, not a reservation) + ≈0.5 GiB Gradle daemon (512
 * MiB default) + ≈1.2 GiB OS ≈ 4.5 GiB, leaving ≈2.5 GiB of headroom, versus ≈1 GiB or less for the
 * pre-fix peak. See {@code req/plans/phase7-3b-reference-findings.md} §11.4 for the full
 * arithmetic.
 */
@EnabledIfNativeAvailable
class Qwen3VlVisionPeakTest {

  private static final int HIDDEN = 1024;
  private static final int NUM_HEADS = 16;
  private static final int HEAD_DIM = 64;
  private static final int INTERMEDIATE = 4096;
  private static final int OUT_HIDDEN = 2048;
  private static final int POSITION_EMBEDDINGS = 2304;
  private static final int DEPTH_MIN = 1;
  private static final int DEPTH_PROXY = 3;
  private static final int DEPTH_MAX = 4;

  /**
   * The depth-independence bound: the depth-4 transient minus the depth-1 transient must stay under
   * 64 MiB, so the per-image peak does not scale with depth.
   */
  private static final long DEPTH_DELTA_BOUND = 64L * 1024L * 1024L;

  private static final int MERGE = 2;
  private static final int MERGE_BLOCK = HIDDEN * MERGE * MERGE;
  private static final int IMAGE = 1024;
  private static final int GRID = IMAGE / 16;
  private static final int PATCH_ROWS = GRID * GRID;
  private static final int MERGED_ROWS = (GRID / MERGE) * (GRID / MERGE);
  private static final long GIB = 1024L * 1024L * 1024L;
  private static final long PLANNING_BOUND = 5 * GIB;

  @Test
  void realGeometryPeakIsDepthIndependentAndUnderPlanningBound() throws Exception {
    try (MLXScope scope = new MLXScope()) {
      List<MLXArray> statics = new ArrayList<>();
      final Qwen3VlVisionTower towerMin = buildTower(scope, statics, DEPTH_MIN, new int[] {0});
      final Qwen3VlVisionTower towerProxy = buildTower(scope, statics, DEPTH_PROXY, new int[] {1});
      final Qwen3VlVisionTower towerMax = buildTower(scope, statics, DEPTH_MAX, new int[] {1});
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

      // Everything the measurements read is static: materialize it (and the rotation tables)
      // before the baseline so only the forwards' own work moves the counter.
      List<MLXArray> toMaterialize = new ArrayList<>(statics);
      toMaterialize.add(rotation.cos());
      toMaterialize.add(rotation.sin());
      MLX.eval(toMaterialize.toArray(MLXArray[]::new));
      MLXMemoryTestSupport.settledActiveBytes(); // warm settle before the baseline
      final long baseline = MLXMemoryTestSupport.settledActiveBytes();

      // Each tower measurement is one forward in a fresh per-request child scope of the caller
      // scope (the SmolVlmModel pattern): the forward's per-image and per-block scopes close
      // inside the forward, and only the hoisted merged and DeepStack rows live in the request
      // scope until it closes, so a settled delta read while it is still open is the per-request
      // retained rows. Each transient is bracketed with the peak counter: reset right before the
      // forward (so no earlier work is counted), read right after it returns.
      MLXMemory.resetPeak();
      long peakMin;
      try (MLXScope request = scope.newChild()) {
        assertOutput(towerMin.forward(request, List.of(image)));
        peakMin = MLXMemory.peakBytes() - baseline;
      }
      MLXMemoryTestSupport.settledActiveBytes(); // settle the delayed frees before the next window
      MLXMemory.resetPeak();
      long peakProxy;
      long sustained;
      try (MLXScope request = scope.newChild()) {
        assertOutput(towerProxy.forward(request, List.of(image)));
        peakProxy = MLXMemory.peakBytes() - baseline;
        sustained = MLXMemoryTestSupport.settledActiveBytes() - baseline;
      }
      MLXMemoryTestSupport.settledActiveBytes(); // settle the delayed frees before the next window
      MLXMemory.resetPeak();
      long peakMax;
      try (MLXScope request = scope.newChild()) {
        assertOutput(towerMax.forward(request, List.of(image)));
        peakMax = MLXMemory.peakBytes() - baseline;
      }
      MLXMemoryTestSupport.settledActiveBytes(); // settle the delayed frees before the bracket

      final long logicalRetained = 2L * MERGED_ROWS * OUT_HIDDEN * Float.BYTES;
      for (int depth : new int[] {DEPTH_MIN, DEPTH_PROXY, DEPTH_MAX}) {
        long peak = depth == DEPTH_MIN ? peakMin : depth == DEPTH_PROXY ? peakProxy : peakMax;
        assertTrue(
            peak >= logicalRetained,
            "depth-"
                + depth
                + " tower peak "
                + peak
                + " B below the logical retained rows "
                + logicalRetained
                + " B");
      }
      assertTrue(
          peakProxy < PLANNING_BOUND,
          "depth-3 tower transient "
              + peakProxy
              + " B exceeded the "
              + PLANNING_BOUND
              + " B planning bound");
      assertTrue(
          sustained >= logicalRetained,
          "depth-3 sustained peak "
              + sustained
              + " B below the logical retained rows "
              + logicalRetained
              + " B");
      assertTrue(
          sustained < PLANNING_BOUND,
          "depth-3 sustained peak "
              + sustained
              + " B exceeded the "
              + PLANNING_BOUND
              + " B planning bound");
      final long depthDelta = peakMax - peakMin;
      assertTrue(
          depthDelta < DEPTH_DELTA_BOUND,
          "per-image transient delta "
              + depthDelta
              + " B between depth "
              + DEPTH_MIN
              + " and depth "
              + DEPTH_MAX
              + " exceeded the "
              + DEPTH_DELTA_BOUND
              + " B depth-independence bound");

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
          "WP5 vision peak (per-block scopes): grid=(1,"
              + GRID
              + ","
              + GRID
              + "), depth-1 transient="
              + peakMin
              + " B ("
              + gib(peakMin)
              + "), depth-3 transient="
              + peakProxy
              + " B ("
              + gib(peakProxy)
              + "), depth-3 sustained="
              + sustained
              + " B ("
              + sustained / (1024 * 1024)
              + " MiB), depth-4 transient="
              + peakMax
              + " B ("
              + gib(peakMax)
              + "), depth-4 minus depth-1="
              + depthDelta
              + " B ("
              + depthDelta / (1024 * 1024)
              + " MiB), attention bracket="
              + transientDelta
              + " B ("
              + gib(transientDelta)
              + "), bound="
              + PLANNING_BOUND / GIB
              + " GiB (plan §4); measured on M5 Max under the ordinary test task "
              + "(MLX_ENABLE_TF32=1 default, "
              + LocalDate.now()
              + ")");
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

  /**
   * Asserts one image's forward output shapes for any of the measured depths (every measured tower
   * has exactly one DeepStack level).
   */
  private static void assertOutput(Qwen3VlVisionFeatures features) {
    assertArrayEquals(new int[] {MERGED_ROWS, OUT_HIDDEN}, features.merged().shape(), "merged");
    assertEquals(1, features.deepstackLevels().size(), "deepstack levels");
    assertArrayEquals(
        new int[] {MERGED_ROWS, OUT_HIDDEN},
        features.deepstackLevels().get(0).shape(),
        "deepstack level 0");
  }

  /**
   * The real-2B geometry at the given reduced depth and DeepStack index, constant 0.001 weights.
   */
  private static Qwen3VlVisionTower buildTower(
      MLXScope scope, List<MLXArray> statics, int depth, int[] deepstackIndexes) {
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    put(weights, statics, scope, "patchEmbedding.weight", HIDDEN, 2, 16, 16, 3);
    put(weights, statics, scope, "patchEmbedding.bias", HIDDEN);
    put(weights, statics, scope, "positionEmbedding.weight", POSITION_EMBEDDINGS, HIDDEN);
    for (int i = 0; i < depth; i++) {
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
    return new Qwen3VlVisionTower(scope, NUM_HEADS, weights, deepstackIndexes);
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

  /** {@code bytes} as {@code "x.xx GiB (y MiB)"} for the evidence line. */
  private static String gib(long bytes) {
    return String.format(Locale.ROOT, "%.2f", bytes / (double) GIB)
        + " GiB ("
        + bytes / (1024 * 1024)
        + " MiB)";
  }
}
