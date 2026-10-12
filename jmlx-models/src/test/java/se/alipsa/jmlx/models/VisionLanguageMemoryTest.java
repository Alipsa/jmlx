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
      long headDim =
          SmolVlmDescriptor.parse(
                  JsonFiles.read(SmolVlmGoldenTest.directory().resolve("config.json")))
              .text()
              .headDim();
      long kvPerPosition =
          2L * config.numHiddenLayers() * config.numKeyValueHeads() * headDim * Float.BYTES;
      long featureBytes = (long) plan.totalFeatureRows() * config.hiddenSize() * Float.BYTES;
      MLXArray deadFeature;
      MLXArray deadKeys;
      try (MLXScope generation = root.newChild()) {
        List<KVCache> caches = new ArrayList<>();
        for (int i = 0; i < config.numHiddenLayers(); i++) {
          caches.add(new KVCache(generation));
        }
        long withVisual;
        try (MLXScope prefill = generation.newChild()) {
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
        assertTrue(
            withVisual - afterPrefill >= featureBytes, "prefill visual storage was retained");
        assertTrue(
            afterPrefill - baseline <= kvPerPosition * plan.expandedLength() + FIXED_MARGIN,
            "prefill retained="
                + (afterPrefill - baseline)
                + ", logical="
                + kvPerPosition * plan.expandedLength());
        long early = 0;
        long late = 0;
        for (int step = 1; step <= 64; step++) {
          try (MLXScope activation = generation.newChild()) {
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
          long expected = kvPerPosition * (plan.expandedLength() + step);
          assertTrue(
              Math.abs(retained - expected) <= FIXED_MARGIN,
              "retained=" + retained + ", expected KV=" + expected);
          if (step == 8) {
            early = retained;
          } else if (step == 64) {
            late = retained;
          }
        }
        assertTrue(Math.abs((late - early) - 56 * kvPerPosition) <= FIXED_MARGIN);
        deadKeys = caches.getFirst().keys();
        System.out.println(
            "VLM retention: KV_bytes_per_position="
                + kvPerPosition
                + ", projected_bytes="
                + featureBytes
                + ", visual_release="
                + (withVisual - afterPrefill)
                + ", prefill_retained="
                + (afterPrefill - baseline)
                + ", early="
                + early
                + ", late="
                + late);
      }
      assertThrows(IllegalStateException.class, deadKeys::toFloatArray);
      assertTrue(settledActiveBytes() <= baseline + FIXED_MARGIN);
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
