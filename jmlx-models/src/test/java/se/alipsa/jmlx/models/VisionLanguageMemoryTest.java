package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXException;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXMemoryTestSupport;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/** Evaluated request ownership and allocator evidence, independent of allocator-cached bytes. */
@EnabledIfNativeAvailable
class VisionLanguageMemoryTest {
  // Warm measured retained-vs-logical overhead is at most 11,264 bytes in the tiny checkpoint.
  // A fixed 16 KiB envelope covers allocator rounding without scaling with requests/tokens.
  private static final long FIXED_MARGIN = 16384;
  // Cleanup retention bound, strictly below the fixture's featureBytes (5,120 B): any retained
  // request buffer -- the visual feature tensor in particular -- must fail the cleanup checks.
  // Both contexts (fresh-class JVM and full-suite JVM) settled at exactly the baseline once
  // every scope closed, so the bound only covers allocator rounding.
  private static final long CLEANUP_MARGIN = 512;

  @Test
  void repeatedRequestsAndEveryFailureReleaseState() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      var cases = SmolVlmGoldenTest.golden().required("cases");
      List<GenerationRequest> requests = new ArrayList<>();
      for (var c : cases) {
        requests.add(SmolVlmGoldenTest.request(c, 8));
      }
      for (var request : requests) {
        model.generate(request, ignored -> {});
      }
      long baseline = MLXMemoryTestSupport.settledActiveBytes();
      MLXMemory.resetPeak();
      long first = baseline;
      for (int repeat = 0; repeat < 12; repeat++) {
        for (var request : requests) {
          model.generate(request, ignored -> {});
          int capacity =
              model.plan(request.promptTokenIds(), request.images()).expandedLength() + 7;
          model.generate(
              request.withCachePolicy(GenerationCachePolicy.full(capacity)), ignored -> {});
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  model.generate(
                      request.withCachePolicy(GenerationCachePolicy.full(capacity - 1)),
                      ignored -> {}));
          AtomicBoolean cancelled = new AtomicBoolean(true);
          var cancellable =
              new GenerationRequest(request.promptTokenIds(), request.config(), cancelled::get)
                  .withImages(request.images());
          assertEquals(
              FinishReason.CANCELLED, model.generate(cancellable, ignored -> {}).finishReason());
          cancelled.set(false);
          assertEquals(
              FinishReason.CANCELLED,
              model.generate(cancellable, event -> cancelled.set(true)).finishReason());
          assertThrows(
              GenerationAbortedException.class,
              () ->
                  model.generate(
                      request,
                      event -> {
                        if (event.tokenId() != null) {
                          throw new IllegalStateException("listener");
                        }
                      }));
          for (String boundary : List.of("connector", "prefill-closed")) {
            model.setObserverForTest(
                stage -> {
                  if (stage.equals(boundary)) {
                    throw new MLXException("injected " + boundary);
                  }
                });
            if (!request.images().isEmpty() || boundary.equals("prefill-closed")) {
              assertThrows(MLXException.class, () -> model.generate(request, ignored -> {}));
            }
            model.setObserverForTest(ignored -> {});
          }
          model
              .decoderForTest()
              .setStepBoundaryEvaluatorForTest(
                  arrays -> {
                    try (MLXScope fault = arrays[0].scope().newChild()) {
                      MLXArray input = MLX.array(fault, new float[] {1, 2, 3, 4}, new int[] {2, 2});
                      MLXShape.transpose(input, new int[] {0, 0});
                    }
                  });
          assertThrows(MLXException.class, () -> model.generate(request, ignored -> {}));
          model.decoderForTest().setStepBoundaryEvaluatorForTest(StepBoundaryEvaluator.NATIVE);
          assertTrue(
              MLXMemoryTestSupport.settledActiveBytes() <= baseline + FIXED_MARGIN,
              "request leaked active storage");
        }
        if (repeat == 0) {
          first = MLXMemoryTestSupport.settledActiveBytes();
        }
      }
      long growth = MLXMemoryTestSupport.settledActiveBytes() - first;
      assertTrue(growth <= FIXED_MARGIN, "late request growth: " + growth);
      long peak = MLXMemory.peakBytes();
      System.out.println(
          "VLM requests: baseline=" + baseline + ", peak=" + peak + ", late_growth=" + growth);
    }
  }

  @Test
  void tokenizerFailureAndCancellationBetweenImagesReleaseState(@TempDir Path directory)
      throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      var cases = SmolVlmGoldenTest.golden().required("cases");
      var base = SmolVlmGoldenTest.request(cases.get(0), 5);
      int first = model.generate(base, ignored -> {}).generatedTokenIds().getFirst();
      ObjectMapper mapper = new ObjectMapper();
      var tree = mapper.readTree(SmolVlmGoldenTest.directory().resolve("tokenizer.json").toFile());
      ObjectNode vocab = (ObjectNode) tree.required("model").required("vocab");
      String missing =
          vocab.properties().stream()
              .filter(e -> e.getValue().asInt() == first)
              .map(Map.Entry::getKey)
              .findFirst()
              .orElseThrow();
      vocab.remove(missing);
      Path file = directory.resolve("tokenizer.json");
      Files.writeString(file, mapper.writeValueAsString(tree));
      var tokenizer = HfTokenizer.fromFile(file);
      var failing =
          GenerationRequest.text(
                  tokenizer,
                  cases.get(0).required("prompt").asString(),
                  PromptSpecialTokens.OMIT,
                  base.config(),
                  CancellationToken.NONE)
              .withImages(base.images());
      var multiple =
          java.util.stream.StreamSupport.stream(cases.spliterator(), false)
              .filter(c -> c.required("images").size() == 2)
              .findFirst()
              .orElseThrow();
      var twoImages = SmolVlmGoldenTest.request(multiple, 5);
      int firstImageTiles =
          model
              .plan(twoImages.promptTokenIds(), twoImages.images())
              .images()
              .getFirst()
              .tileCount();
      model.generate(twoImages, ignored -> {});
      long baseline = MLXMemoryTestSupport.settledActiveBytes();
      for (int repeat = 0; repeat < 12; repeat++) {
        var failure =
            assertThrows(
                GenerationAbortedException.class, () -> model.generate(failing, ignored -> {}));
        assertEquals("output decoder", failure.stage());
        AtomicBoolean cancelled = new AtomicBoolean();
        AtomicInteger tiles = new AtomicInteger();
        model.setObserverForTest(
            stage -> {
              if (stage.equals("connector") && tiles.incrementAndGet() == firstImageTiles) {
                cancelled.set(true);
              }
            });
        var cancellable =
            new GenerationRequest(twoImages.promptTokenIds(), twoImages.config(), cancelled::get)
                .withImages(twoImages.images());
        assertEquals(
            FinishReason.CANCELLED, model.generate(cancellable, ignored -> {}).finishReason());
        assertEquals(firstImageTiles, tiles.get(), "second image must never execute");
        model.setObserverForTest(ignored -> {});
        assertTrue(MLXMemoryTestSupport.settledActiveBytes() <= baseline + FIXED_MARGIN);
      }
      assertTrue(MLXMemoryTestSupport.settledActiveBytes() <= baseline + FIXED_MARGIN);
    }
  }

  /**
   * Visual release and per-step KV growth evidence. The four cleared endpoints -- right after
   * prefill closes and at decode steps 1, midLimit and 64 -- are asserted to retain exactly their
   * KV payload: {@link MLXMemory#clearCache()} empties the Metal allocator's buffer pool before
   * each endpoint, so that step's KV buffers are allocated fresh, at exactly the requested size
   * (every endpoint request here stays below the allocator's 16 KiB page-alignment threshold), and
   * the endpoint's retained bytes equal the logical payload in any pool state. Each endpoint is
   * measured while every request scope is still open, so any retained or overgrown request buffer
   * -- a visual feature tensor hoisted into the request or generation scope in particular -- fails
   * one of them before a scope closes. Every other step keeps the pool-independent lower bound
   * {@code retained >= logical}, the check that catches request storage that undergrew its logical
   * payload. The three decode endpoint values are additionally reported as {@code early}, {@code
   * mid} and {@code late} in the test's {@code VLM retention} evidence line, so {@code early} is
   * both asserted and reported. The three cleanup checks at the end then verify the release path
   * itself: that closing the request scope and running the production {@code generate} leave no
   * request buffer active.
   */
  @Test
  void visualReleaseAndEarlyLateKvSlopesHaveDerivedBudgets() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      var request =
          SmolVlmGoldenTest.request(SmolVlmGoldenTest.golden().required("cases").get(0), 1);
      model.generate(request, ignored -> {});
      long baseline = MLXMemoryTestSupport.settledActiveBytes();
      var plan = model.plan(request.promptTokenIds(), request.images());
      var config = model.decoderForTest().config();
      var descriptor =
          SmolVlmDescriptor.parse(
              JsonFiles.read(SmolVlmGoldenTest.directory().resolve("config.json")));
      long headDim = descriptor.text().headDim();
      // Two key/value tensors per layer, each kvHeads * headDim floats, per sequence
      // position.
      long kvPerPosition =
          2L * config.numHiddenLayers() * config.numKeyValueHeads() * headDim * Float.BYTES;
      long featureBytes = (long) plan.totalFeatureRows() * config.hiddenSize() * Float.BYTES;
      // One key or value tensor's bytes per position (kvHeads * headDim * 4);
      // kvPerPosition is 2 * numHiddenLayers times this. Keys and values are separate
      // buffers -- KVCache concatenates them independently -- so at step s each of the
      // 2 * numHiddenLayers buffers requests kvBytesPerBuffer * (expandedLength + s)
      // bytes. The boundary the mid endpoint must land on is the TEXT hidden tier
      // [1, expandedLength, hiddenSize] f32 (Q-projection and embedding outputs) that
      // prefill definitely allocates and frees; at midLimit each buffer's request equals
      // hiddenBytes exactly.
      long kvBytesPerBuffer = config.numKeyValueHeads() * headDim * Float.BYTES;
      long hiddenBytes = (long) plan.expandedLength() * config.hiddenSize() * Float.BYTES;
      assertTrue(
          hiddenBytes % kvBytesPerBuffer == 0,
          "hidden tier "
              + hiddenBytes
              + " B is not a multiple of the per-buffer KV size "
              + kvBytesPerBuffer
              + " B");
      long midLimit = hiddenBytes / kvBytesPerBuffer - plan.expandedLength();
      assertTrue(
          midLimit > 1 && midLimit < 64, "midLimit=" + midLimit + " outside the decode span");
      MLXArray deadFeature;
      MLXArray deadKeys;
      try (MLXScope generation = root.newChild()) {
        // Owns every buffer of this test-shaped request (caches, prefill, activations). It
        // is closed while `generation` is still open, so the test-shape cleanup check below
        // sees a feature tensor hoisted into `generation` as retained; retention in this scope
        // itself is caught by the exact endpoints above, measured before the close.
        MLXScope requestScope = generation.newChild();
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < config.numHiddenLayers(); i++) {
          caches.add(new KVCache(requestScope));
        }
        long withVisual;
        try (MLXScope prefill = requestScope.newChild()) {
          deadFeature = model.features(request, prefill);
          assertEquals(featureBytes / Float.BYTES, deadFeature.toFloatArray().length);
          long featuresOnly = MLXMemoryTestSupport.settledActiveBytes() - baseline;
          assertTrue(featuresOnly >= featureBytes, "features did not survive vision child closure");
          // Empty the allocator's buffer pool before the prefill runs: with an empty pool every
          // allocation up to the KV concat is a fresh, exact-size buffer, so once the prefill
          // scope closes the only live bytes are the four KV buffers, exactly sized, in any pool
          // state.
          MLXMemory.clearCache();
          var logits =
              model
                  .decoderForTest()
                  .stepLogits(
                      MLX.array(prefill, plan.expandedIds(), new int[] {1, plan.expandedLength()}),
                      caches,
                      new int[] {plan.expandedLength()},
                      embedded -> SmolVlmModel.merge(embedded, deadFeature, plan));
          MLX.eval(logits);
          MLX.eval(DecoderModel.stepCacheArrays(caches));
          withVisual = MLXMemoryTestSupport.settledActiveBytes();
        }
        assertThrows(IllegalStateException.class, deadFeature::toFloatArray);
        long afterPrefill = MLXMemoryTestSupport.settledActiveBytes();
        // Sanity only: the release is dominated by the other prefill buffers, so a retained
        // feature tensor alone would not move it; the cleanup checks below detect retention.
        assertTrue(
            withVisual - afterPrefill >= featureBytes, "prefill visual storage was retained");
        // The pool was cleared before the KV buffers were materialized and the only other
        // prefill buffers live in the just-closed prefill scope, so at this point the only live
        // buffers are the four exact-size KV buffers: retained bytes equal the logical KV
        // payload exactly, in any pool state.
        long logicalPrefill = kvPerPosition * plan.expandedLength();
        assertEquals(
            logicalPrefill,
            afterPrefill - baseline,
            "prefill retained=" + (afterPrefill - baseline) + ", logical=" + logicalPrefill);
        long early = 0;
        long mid = 0;
        long late = 0;
        for (int step = 1; step <= 64; step++) {
          // Cleared endpoint: the pool is empty for this whole step, and the only allocations
          // before its KV concat are single-token intermediates far smaller than the KV
          // request, so the step's KV buffers are allocated fresh, exactly sized, and the
          // endpoint retains exactly the logical payload, in any pool state (including whatever
          // prior test classes left pooled in this JVM).
          if (step == 1 || step == midLimit || step == 64) {
            MLXMemory.clearCache();
          }
          try (MLXScope activation = requestScope.newChild()) {
            var logits =
                model
                    .decoderForTest()
                    .stepLogits(
                        MLX.array(activation, new int[] {32}, new int[] {1, 1}),
                        caches,
                        new int[] {1},
                        null);
            MLX.eval(logits);
            MLX.eval(DecoderModel.stepCacheArrays(caches));
          }
          long retained = MLXMemoryTestSupport.settledActiveBytes() - baseline;
          long logical = kvPerPosition * (plan.expandedLength() + step);
          // Every retained buffer holds at least its logical payload, in any pool state;
          // the pooled-buffer slack on top is context-dependent and is not bounded here.
          assertTrue(
              retained >= logical,
              "step=" + step + ", retained=" + retained + ", logical=" + logical);
          if (step == 1 || step == midLimit || step == 64) {
            assertEquals(
                logical,
                retained,
                "step=" + step + ", retained=" + retained + ", logical=" + logical);
          }
          if (step == 1) {
            early = retained;
          } else if (step == midLimit) {
            mid = retained;
          } else if (step == 64) {
            late = retained;
          }
        }
        // Consistency check subsumed by the exact endpoint equalities above: the growth
        // span midLimit..64 must carry exactly the logical increment of 64 - midLimit
        // positions, (64 - midLimit) * kvPerPosition.
        long logicalGrowth = (64L - midLimit) * kvPerPosition;
        assertEquals(
            logicalGrowth, late - mid, "growth=" + (late - mid) + ", logical=" + logicalGrowth);
        deadKeys = caches.getFirst().keys();
        System.out.println(
            "VLM retention: KV_bytes_per_position="
                + kvPerPosition
                + ", features_bytes="
                + featureBytes
                + ", visual_release="
                + (withVisual - afterPrefill)
                + ", prefill_retained="
                + (afterPrefill - baseline)
                + ", early="
                + early
                + ", mid="
                + mid
                + ", late="
                + late);
        requestScope.close();
        assertThrows(IllegalStateException.class, deadKeys::toFloatArray);
        // Test-shape cleanup: verifies the scope-close release path itself -- that
        // requestScope.close() released every request buffer to the pool, leaving a residual
        // <= CLEANUP_MARGIN in any pool state. A feature tensor hoisted into `generation`
        // (which outlives the request scope) would still be active here and exceed the margin.
        // A feature hoisted into requestScope itself, the cache scope, is not visible to this
        // check -- the scope is already closed -- but is caught by the step-64 exact endpoint,
        // whose measurement precedes the close.
        long afterRequestClose = MLXMemoryTestSupport.settledActiveBytes();
        assertTrue(
            afterRequestClose - baseline <= CLEANUP_MARGIN,
            "test-shape request retained visual storage: " + (afterRequestClose - baseline));
      }
      // Every scope of the work is closed here, so every request buffer is freed and pooled,
      // and pooled bytes are excluded from the active counter: the counter is back at the
      // baseline in any pool state. A retained request buffer -- the 5,120-byte visual
      // feature tensor in this fixture -- exceeds CLEANUP_MARGIN and fails here.
      long afterCleanup = MLXMemoryTestSupport.settledActiveBytes();
      assertTrue(
          afterCleanup - baseline <= CLEANUP_MARGIN,
          "closed request retained visual storage: " + (afterCleanup - baseline));
      // The production path manages its own visual scopes internally (SmolVlmModel's
      // prefill scope closes after the prefill logits and KV are evaluated); it must
      // leave no visual storage behind either.
      model.generate(request, ignored -> {});
      long afterProduction = MLXMemoryTestSupport.settledActiveBytes();
      assertTrue(
          afterProduction - baseline <= CLEANUP_MARGIN,
          "production generate retained visual storage: " + (afterProduction - baseline));
    }
  }
}
