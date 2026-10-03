package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

/**
 * Per-family proof that the lazy batched step API equals independent single-row runs: left-padded
 * prefill with unequal prompt lengths and a following batched decode, at {@code B > 1}, on every
 * supported family's committed checkpoint. Covers per-row masks and RoPE positions, Gemma embedding
 * scaling, Phi-3's fused projections, Mixtral's expert routing, and Mistral's sliding window.
 *
 * <p>Tolerance: MLX's default reduced-precision float32 matmul on M5 changes with batch shape. The
 * pinned runtime measured at most 1.10e-3 absolute logit error (Gemma decode); disabling TF32
 * reduced every family's error below 2.39e-7. Use a 2e-3 bound in the default mode and retain 1e-4
 * with {@code MLX_ENABLE_TF32=0}. See {@code req/phase6-batch-equivalence.md} for the hardware,
 * measurements and reproduction. Greedy token identity is asserted separately on these fixtures.
 */
@EnabledIfNativeAvailable
class BatchStepEquivalenceTest {
  static final float TOLERANCE = "0".equals(System.getenv("MLX_ENABLE_TF32")) ? 1e-4f : 2e-3f;
  private static final int[][] PROMPTS = {{1, 7, 42, 3, 19, 5}, {1, 9, 4}, {1, 2, 3, 4, 5}};
  private static final int[] NEXT = {95, 12, 77};

  @ParameterizedTest
  @ValueSource(strings = {"llama", "qwen2", "mistral", "gemma", "phi3", "mixtral"})
  void batchedPrefillAndDecodeMatchIndependentRows(String family) throws Exception {
    Path directory =
        Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens")
            .resolve("checkpoints")
            .resolve(family);
    try (MLXScope root = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(root, directory);
      int vocab = model.config().vocabSize();
      int layers = model.config().numHiddenLayers();

      // Reference: each row alone through the same lazy step API (B = 1, no padding).
      float[][] soloPrefill = new float[PROMPTS.length][];
      float[][] soloDecode = new float[PROMPTS.length][];
      for (int row = 0; row < PROMPTS.length; row++) {
        try (MLXScope run = root.newChild()) {
          List<KVCache> caches = caches(run, layers);
          soloPrefill[row] = step(model, run, caches, new int[][] {PROMPTS[row]}, vocab);
          soloDecode[row] = step(model, run, caches, new int[][] {{NEXT[row]}}, vocab);
        }
      }

      // Batched: left-pad to the longest prompt, one prefill, then one [B,1] decode.
      try (MLXScope run = root.newChild()) {
        List<KVCache> caches = caches(run, layers);
        int width = 0;
        for (int[] prompt : PROMPTS) {
          width = Math.max(width, prompt.length);
        }
        int[] valid = new int[PROMPTS.length];
        int[] padded = new int[PROMPTS.length * width];
        for (int row = 0; row < PROMPTS.length; row++) {
          valid[row] = PROMPTS[row].length;
          System.arraycopy(PROMPTS[row], 0, padded, row * width + (width - valid[row]), valid[row]);
        }
        float[] prefill = stepFlat(model, run, caches, padded, PROMPTS.length, width, valid);
        for (int row = 0; row < PROMPTS.length; row++) {
          assertEquals(argmax(soloPrefill[row]), argmax(slice(prefill, row, vocab)));
          assertArrayEquals(
              soloPrefill[row],
              slice(prefill, row, vocab),
              TOLERANCE,
              family + " prefill row " + row);
        }
        int[] ones = new int[PROMPTS.length];
        java.util.Arrays.fill(ones, 1);
        float[] decode = stepFlat(model, run, caches, NEXT, PROMPTS.length, 1, ones);
        for (int row = 0; row < PROMPTS.length; row++) {
          assertEquals(argmax(soloDecode[row]), argmax(slice(decode, row, vocab)));
          assertArrayEquals(
              soloDecode[row], slice(decode, row, vocab), TOLERANCE, family + " decode row " + row);
        }
      }
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"llama", "mistral"})
  void batchedStepEvaluatesNothingUntilTheCallerDoes(String family) throws Exception {
    // The lazy contract: stepLogits must not call the model's StepBoundaryEvaluator, so the
    // scheduler's joint selection evaluate is the step's only synchronization.
    Path directory =
        Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens")
            .resolve("checkpoints")
            .resolve(family);
    try (MLXScope root = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(root, directory);
      AtomicInteger evaluations = new AtomicInteger();
      model.setStepBoundaryEvaluatorForTest(
          arrays -> {
            evaluations.incrementAndGet();
            MLX.eval(arrays);
          });
      try (MLXScope run = root.newChild()) {
        List<KVCache> caches = caches(run, model.config().numHiddenLayers());
        int[] valid = {3, 3};
        MLXArray ids = MLX.array(run, new int[] {1, 2, 3, 4, 5, 6}, new int[] {2, 3});
        MLXArray logits = model.stepLogits(ids, caches, valid, null);
        assertEquals(0, evaluations.get(), "stepLogits is lazy");
        assertArrayEquals(
            new int[] {2, 1, model.config().vocabSize()}, logits.shape(), "last column only");
        model.stepBoundaryEvaluator().evaluate(logits);
        assertEquals(1, evaluations.get(), "the caller's evaluate is the one synchronization");
        assertTrue(caches.getFirst().nextPosition(0) > 0);
      }
    }
  }

  private static List<KVCache> caches(MLXScope scope, int layers) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < layers; i++) {
      caches.add(new KVCache(scope));
    }
    return caches;
  }

  private static float[] step(
      DecoderModel model, MLXScope run, List<KVCache> caches, int[][] ids, int vocab) {
    int width = ids[0].length;
    int[] flat = new int[ids.length * width];
    int[] valid = new int[ids.length];
    for (int row = 0; row < ids.length; row++) {
      System.arraycopy(ids[row], 0, flat, row * width, width);
      valid[row] = width;
    }
    return stepFlat(model, run, caches, flat, ids.length, width, valid);
  }

  private static float[] stepFlat(
      DecoderModel model,
      MLXScope run,
      List<KVCache> caches,
      int[] flat,
      int batch,
      int width,
      int[] valid) {
    try (MLXScope activation = run.newChild()) {
      MLXArray ids = MLX.array(activation, flat, new int[] {batch, width});
      MLXArray logits = model.stepLogits(ids, caches, valid, null);
      MLXArray[] arrays = new MLXArray[1 + 2 * caches.size()];
      arrays[0] = logits;
      System.arraycopy(DecoderModel.stepCacheArrays(caches), 0, arrays, 1, 2 * caches.size());
      model.stepBoundaryEvaluator().evaluate(arrays);
      return logits.toFloatArray();
    }
  }

  private static int argmax(float[] logits) {
    int best = 0;
    for (int i = 1; i < logits.length; i++) {
      if (logits[i] > logits[best]) {
        best = i;
      }
    }
    return best;
  }

  private static float[] slice(float[] flat, int row, int vocab) {
    float[] out = new float[vocab];
    System.arraycopy(flat, row * vocab, out, 0, vocab);
    return out;
  }
}
