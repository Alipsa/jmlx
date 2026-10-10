package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;

/**
 * Contract tests for the embedding-start entry of the batched decoder stack ({@link
 * DecoderModel#decoderStack}) and for its shape-keyed preflight {@link
 * DecoderModel#preflightBatch(int, int, List, int[])}: the checks added with the entry point must
 * reject a wrong rank, batch, width or hidden size and a wrong cache count before any layer
 * advances, a correctly shaped call must run the whole stack, and the preflight itself must keep
 * the poisoned-cache and width-vs-valid-length guarantees a WP6 caller relies on.
 */
@EnabledIfNativeAvailable
class DecoderStackContractTest {
  private static final int BATCH = 1;
  private static final int WIDTH = 3;
  private static MLXScope scope;
  private static DecoderModel model;

  @BeforeAll
  static void loadCheckpoint() throws Exception {
    scope = new MLXScope();
    model = (DecoderModel) TextGenerationModels.load(scope, SchedulerFixtures.checkpoint("llama"));
  }

  @AfterAll
  static void closeCheckpoint() {
    scope.close();
  }

  @Test
  void rankTwoEmbeddingsAreRejected() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      int hidden = model.config().hiddenSize();
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.decoderStack(
                      MLX.array(run, new float[BATCH * hidden], new int[] {BATCH, hidden}),
                      caches,
                      validLengths(),
                      WIDTH));
      assertTrue(failure.getMessage().startsWith("decoderStack requires embeddings shaped"));
      assertNoCacheAdvanced(caches);
    }
  }

  @Test
  void rankFourEmbeddingsAreRejected() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      int hidden = model.config().hiddenSize();
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.decoderStack(
                      embeddings(run, BATCH, WIDTH, hidden, 1), caches, validLengths(), WIDTH));
      assertTrue(failure.getMessage().startsWith("decoderStack requires embeddings shaped"));
      assertNoCacheAdvanced(caches);
    }
  }

  @Test
  void batchMismatchIsRejected() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      int hidden = model.config().hiddenSize();
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.decoderStack(
                      embeddings(run, BATCH + 1, WIDTH, hidden), caches, validLengths(), WIDTH));
      assertEquals(
          "decoderStack requires embeddings shaped ["
              + BATCH
              + ", "
              + WIDTH
              + ", "
              + hidden
              + "]: got ["
              + (BATCH + 1)
              + ", "
              + WIDTH
              + ", "
              + hidden
              + "]",
          failure.getMessage());
      assertNoCacheAdvanced(caches);
    }
  }

  @Test
  void widthMismatchIsRejected() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      int hidden = model.config().hiddenSize();
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.decoderStack(
                      embeddings(run, BATCH, WIDTH + 1, hidden), caches, validLengths(), WIDTH));
      assertEquals(
          "decoderStack requires embeddings shaped ["
              + BATCH
              + ", "
              + WIDTH
              + ", "
              + hidden
              + "]: got ["
              + BATCH
              + ", "
              + (WIDTH + 1)
              + ", "
              + hidden
              + "]",
          failure.getMessage());
      assertNoCacheAdvanced(caches);
    }
  }

  @Test
  void hiddenSizeMismatchIsRejected() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      int hidden = model.config().hiddenSize();
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.decoderStack(
                      embeddings(run, BATCH, WIDTH, hidden + 1), caches, validLengths(), WIDTH));
      assertEquals(
          "decoderStack requires embeddings shaped ["
              + BATCH
              + ", "
              + WIDTH
              + ", "
              + hidden
              + "]: got ["
              + BATCH
              + ", "
              + WIDTH
              + ", "
              + (hidden + 1)
              + "]",
          failure.getMessage());
      assertNoCacheAdvanced(caches);
    }
  }

  @Test
  void fewerCachesThanLayersAreRejected() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      int layers = model.config().numHiddenLayers();
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.decoderStack(
                      embeddings(run, BATCH, WIDTH, model.config().hiddenSize()),
                      caches.subList(0, layers - 1),
                      validLengths(),
                      WIDTH));
      assertEquals(
          "one KVCache is required per decoder layer: got "
              + (layers - 1)
              + " caches for "
              + layers
              + " layers",
          failure.getMessage());
      assertNoCacheAdvanced(caches);
    }
  }

  @Test
  void moreCachesThanLayersAreRejected() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      int layers = model.config().numHiddenLayers();
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.decoderStack(
                      embeddings(run, BATCH, WIDTH, model.config().hiddenSize()),
                      cachesWithExtra(run, caches),
                      validLengths(),
                      WIDTH));
      assertEquals(
          "one KVCache is required per decoder layer: got "
              + (layers + 1)
              + " caches for "
              + layers
              + " layers",
          failure.getMessage());
      assertNoCacheAdvanced(caches);
    }
  }

  @Test
  void correctShapeRunsTheWholeStack() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      int hidden = model.config().hiddenSize();
      MLXArray result =
          model.decoderStack(embeddings(run, BATCH, WIDTH, hidden), caches, validLengths(), WIDTH);
      int[] shape = result.shape();
      assertEquals(3, shape.length);
      assertEquals(BATCH, shape[0]);
      assertEquals(WIDTH, shape[1]);
      assertEquals(hidden, shape[2]);
      MLXArray[] arrays = new MLXArray[1 + 2 * caches.size()];
      arrays[0] = result;
      System.arraycopy(DecoderModel.stepCacheArrays(caches), 0, arrays, 1, 2 * caches.size());
      model.stepBoundaryEvaluator().evaluate(arrays);
      for (KVCache cache : caches) {
        assertEquals(WIDTH, cache.nextPosition(0));
      }
    }
  }

  @Test
  void shapePreflightRejectsPoisonedCache() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      caches.getFirst().poison();
      IllegalStateException failure =
          assertThrows(
              IllegalStateException.class,
              () -> model.preflightBatch(BATCH, WIDTH, caches, validLengths()));
      assertEquals("decoder batch cache is poisoned or has wrong batch size", failure.getMessage());
      assertNoCacheAdvanced(caches);
    }
  }

  @Test
  void shapePreflightRejectsWidthExceedingLongestValidRow() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      IllegalArgumentException failure =
          assertThrows(
              IllegalArgumentException.class,
              () -> model.preflightBatch(BATCH, WIDTH + 1, caches, validLengths()));
      assertEquals("left-padded width must equal longest valid row", failure.getMessage());
      assertNoCacheAdvanced(caches);
    }
  }

  private static int[] validLengths() {
    return new int[] {WIDTH};
  }

  private static MLXArray embeddings(
      MLXScope scope, int batch, int width, int hidden, int... trailing) {
    long count = (long) batch * width * hidden;
    for (int dimension : trailing) {
      count *= dimension;
    }
    int[] shape = new int[3 + trailing.length];
    shape[0] = batch;
    shape[1] = width;
    shape[2] = hidden;
    for (int i = 0; i < trailing.length; i++) {
      shape[3 + i] = trailing[i];
    }
    return MLX.array(scope, new float[(int) count], shape);
  }

  private static List<KVCache> caches(MLXScope scope) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.config().numHiddenLayers(); i++) {
      caches.add(new KVCache(scope));
    }
    return caches;
  }

  private static List<KVCache> cachesWithExtra(MLXScope scope, List<KVCache> caches) {
    List<KVCache> extended = new ArrayList<>(caches);
    extended.add(new KVCache(scope));
    return extended;
  }

  private static void assertNoCacheAdvanced(List<KVCache> caches) {
    for (KVCache cache : caches) {
      assertEquals(0, cache.nextPosition());
    }
  }
}
