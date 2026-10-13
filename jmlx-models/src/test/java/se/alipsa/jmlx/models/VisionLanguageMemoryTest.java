package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXException;
import se.alipsa.jmlx.core.MLXMemory;
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
  private static final long SETTLE_INTERVAL_MS = 2;
  private static final long SETTLE_TAIL_MS = 16;
  private static final long SETTLE_TIMEOUT_MS = 1000;

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
      long baseline = settledActiveBytes();
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
              settledActiveBytes() <= baseline + FIXED_MARGIN, "request leaked active storage");
        }
        if (repeat == 0) {
          first = settledActiveBytes();
        }
      }
      long growth = settledActiveBytes() - first;
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
      long baseline = settledActiveBytes();
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
        assertTrue(settledActiveBytes() <= baseline + FIXED_MARGIN);
      }
      assertTrue(settledActiveBytes() <= baseline + FIXED_MARGIN);
    }
  }

  @Test
  void visualReleaseAndEarlyLateKvSlopesHaveDerivedBudgets() throws Exception {
    try (MLXScope root = new MLXScope()) {
      SmolVlmModel model = SmolVlmModel.load(root, SmolVlmGoldenTest.directory());
      var request =
          SmolVlmGoldenTest.request(SmolVlmGoldenTest.golden().required("cases").get(0), 1);
      model.generate(request, ignored -> {});
      long baseline = settledActiveBytes();
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
      // kvPerPosition is 2 * numHiddenLayers times this.
      long kvBytesPerBuffer = config.numKeyValueHeads() * headDim * Float.BYTES;
      // The MLP intermediate-tier buffer size, [1, expandedLength, intermediateSize] f32 (the
      // gate/up/down projection outputs prefill creates). Each layer's KV buffer packs key
      // and value, so at step s its request is 2 * kvBytesPerBuffer * (expandedLength + s)
      // bytes; at midLimit that request equals intermediateBytes exactly. The text hidden
      // tier is smaller than the request, so this is the smallest prefill tier that fits
      // and the buffer draws on it exact-fit, with no pooled slack in any pool state.
      long intermediateBytes =
          (long) plan.expandedLength() * config.intermediateSize() * Float.BYTES;
      long midLimit = intermediateBytes / (2L * kvBytesPerBuffer) - plan.expandedLength();
      MLXArray deadFeature;
      MLXArray deadKeys;
      try (MLXScope generation = root.newChild()) {
        // Owns every buffer of this test-shaped request (caches, prefill, activations). It
        // is closed while `generation` is still open, so the test-shape cleanup check below
        // sees a feature tensor hoisted out of the prefill scope as retained.
        MLXScope requestScope = generation.newChild();
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < config.numHiddenLayers(); i++) {
          caches.add(new KVCache(requestScope));
        }
        long withVisual;
        try (MLXScope prefill = requestScope.newChild()) {
          deadFeature = model.features(request, prefill);
          assertEquals(featureBytes / Float.BYTES, deadFeature.toFloatArray().length);
          long featuresOnly = settledActiveBytes() - baseline;
          assertTrue(featuresOnly >= featureBytes, "features did not survive vision child closure");
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
          withVisual = settledActiveBytes();
        }
        assertThrows(IllegalStateException.class, deadFeature::toFloatArray);
        long afterPrefill = settledActiveBytes();
        // Sanity only: the release is dominated by the other prefill buffers, so a retained
        // feature tensor alone would not move it; the cleanup checks below detect retention.
        assertTrue(
            withVisual - afterPrefill >= featureBytes, "prefill visual storage was retained");
        // Every retained buffer holds at least its logical payload, in any pool state. No
        // upper bound is asserted: the pooled-buffer slack (recycled-buffer tiers) depends
        // on prior allocations in the JVM; retained visual storage is detected by the
        // cleanup checks below.
        long logicalPrefill = kvPerPosition * plan.expandedLength();
        assertTrue(
            afterPrefill - baseline >= logicalPrefill,
            "prefill retained=" + (afterPrefill - baseline) + ", logical=" + logicalPrefill);
        long early = 0;
        long mid = 0;
        long late = 0;
        for (int step = 1; step <= 64; step++) {
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
          long retained = settledActiveBytes() - baseline;
          long logical = kvPerPosition * (plan.expandedLength() + step);
          // Every retained buffer holds at least its logical payload, in any pool state;
          // the pooled-buffer slack on top is context-dependent and is not bounded here.
          assertTrue(
              retained >= logical,
              "step=" + step + ", retained=" + retained + ", logical=" + logical);
          if (step == 1) {
            early = retained;
          } else if (step == midLimit) {
            mid = retained;
          } else if (step == 64) {
            late = retained;
          }
        }
        // The growth span midLimit..64 must carry the logical increment of 64 - midLimit
        // positions, (64 - midLimit) * kvPerPosition. midLimit sits on the intermediate-tier
        // boundary, where the per-buffer request equals the tier exactly and its endpoint
        // therefore carries no pooled slack in any pool state; the late endpoint draws on
        // fresh or small-recycled tiers beyond it, with a measured excess over logical of
        // 768 bytes in a fresh-class JVM and 2,048 bytes in the full-suite JVM. The
        // 8-KV-position margin is below half the span's logical growth, so halved growth
        // still fails.
        long logicalGrowth = (64L - midLimit) * kvPerPosition;
        assertTrue(
            Math.abs((late - mid) - logicalGrowth) <= 8 * kvPerPosition,
            "growth=" + (late - mid) + ", logical=" + logicalGrowth);
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
        // Test-shape cleanup: requestScope is closed while `generation` is still open, so a
        // feature tensor lifted out of the prefill scope by a longer-lived reference (a hoist
        // into `generation` in particular) is still active here. Every legitimately released
        // buffer is pooled, and pooled bytes are excluded from the active counter in any pool
        // state, so the residual is back at the baseline; the 5,120-byte fixture feature
        // tensor exceeds CLEANUP_MARGIN and fails here if retained.
        long afterRequestClose = settledActiveBytes();
        assertTrue(
            afterRequestClose - baseline <= CLEANUP_MARGIN,
            "test-shape request retained visual storage: " + (afterRequestClose - baseline));
      }
      // Every scope of the work is closed here, so every request buffer is freed and pooled,
      // and pooled bytes are excluded from the active counter: the counter is back at the
      // baseline in any pool state. A retained request buffer -- the 5,120-byte visual
      // feature tensor in this fixture -- exceeds CLEANUP_MARGIN and fails here.
      long afterCleanup = settledActiveBytes();
      assertTrue(
          afterCleanup - baseline <= CLEANUP_MARGIN,
          "closed request retained visual storage: " + (afterCleanup - baseline));
      // The production path manages its own visual scopes internally (SmolVlmModel's
      // prefill scope closes after the prefill logits and KV are evaluated); it must
      // leave no visual storage behind either.
      model.generate(request, ignored -> {});
      long afterProduction = settledActiveBytes();
      assertTrue(
          afterProduction - baseline <= CLEANUP_MARGIN,
          "production generate retained visual storage: " + (afterProduction - baseline));
    }
  }

  /**
   * {@link MLXMemory#activeBytes()} once its in-flight Metal completion-handler frees (see {@link
   * MLXMemory}'s class javadoc) have drained: polls until the value has been unchanged for {@value
   * #SETTLE_TAIL_MS} milliseconds. Throws {@link IllegalStateException} if the counter is still
   * moving {@value #SETTLE_TIMEOUT_MS} milliseconds after the first read, instead of silently
   * returning an unsettled value.
   */
  private static long settledActiveBytes() throws InterruptedException {
    long previous = MLXMemory.activeBytes();
    long lastChange = System.nanoTime();
    long deadline = lastChange + TimeUnit.MILLISECONDS.toNanos(SETTLE_TIMEOUT_MS);
    while (System.nanoTime() < deadline) {
      Thread.sleep(SETTLE_INTERVAL_MS);
      long current = MLXMemory.activeBytes();
      if (current != previous) {
        previous = current;
        lastChange = System.nanoTime();
      } else if (System.nanoTime() - lastChange >= TimeUnit.MILLISECONDS.toNanos(SETTLE_TAIL_MS)) {
        return current;
      }
    }
    throw new IllegalStateException(
        "MLXMemory.activeBytes() did not settle within "
            + SETTLE_TIMEOUT_MS
            + " ms; last value: "
            + previous);
  }
}
