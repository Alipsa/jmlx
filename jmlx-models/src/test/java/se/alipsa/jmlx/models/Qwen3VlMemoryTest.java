package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
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
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import tools.jackson.databind.node.ObjectNode;

/** Evaluated visual release, KV growth and cleanup through the production request boundaries. */
@EnabledIfNativeAvailable
class Qwen3VlMemoryTest {
  // Measured endpoints retain exactly logical KV. Cleanup permits only sub-feature rounding;
  // this is far below the smallest main/DeepStack tensor (4096 bytes in this checkpoint).
  private static final long CLEANUP_MARGIN = 512;

  @Test
  void visualStorageDiesAtPrefillAndOnlyKvGrowsDuringDecode() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Qwen3VlModel.load(root, Qwen3VlModelFloatGoldenTest.directory());
      var request = Qwen3VlTestSupport.requests(1).getFirst();
      model.generate(request, ignored -> {});
      long baseline = MLXMemoryTestSupport.settledActiveBytes();
      var plan = model.plan(request.promptTokenIds(), request.images());
      var config = model.decoderForTest().config();
      long kvPerPosition =
          2L
              * config.numHiddenLayers()
              * config.numKeyValueHeads()
              * model.descriptorForTest().text().headDim()
              * Float.BYTES;
      // Exact allocator tiers: mid ends with 16 KiB per K/V buffer, late with 32 KiB.
      // This avoids counting buffer-pool page rounding as retained visual storage.
      int tierPositions = (int) (16384 / (kvPerPosition / (2 * config.numHiddenLayers())));
      int midStep = tierPositions - plan.expandedLength();
      int lateStep = 2 * tierPositions - plan.expandedLength();
      long featureBytes =
          (long) plan.featureDestinations().length
              * config.hiddenSize()
              * (1 + model.towerForTest().deepstackIndexes().length)
              * Float.BYTES;
      long[] beforeClose = {0};
      long[] afterClose = {0};
      model.setObserverForTest(
          stage -> {
            if (stage.equals("before-prefill")) {
              MLXMemory.clearCache();
            } else if (stage.equals("prefill-evaluated")) {
              beforeClose[0] = settled();
            } else if (stage.equals("prefill-closed")) {
              afterClose[0] = settled();
            }
          });
      MLXArray deadKeys;
      long early = 0;
      long mid = 0;
      long late = 0;
      try (MLXScope generation = root.newChild()) {
        var caches = Qwen3VlTestSupport.caches(model, generation);
        try (MLXScope activation = generation.newChild()) {
          model.prefill(CancellationToken.NONE, request.images(), plan, activation, caches);
        }
        long retained = settled() - baseline;
        assertEquals(
            kvPerPosition * plan.expandedLength() + 4L * model.metadata().vocabSize(),
            afterClose[0] - baseline,
            "immediately after prefill closes only KV and the returned logits may survive");
        assertEquals(kvPerPosition * plan.expandedLength(), retained, "post-prefill retained KV");
        assertTrue(
            beforeClose[0] - baseline - retained >= featureBytes,
            "prefill must release every final/DeepStack level");
        // Every retained cache and request-local position remains valid after visual closure.
        for (var cache : caches) {
          assertTrue(cache.keys().toFloatArray().length > 0);
          assertTrue(cache.values().toFloatArray().length > 0);
          assertEquals(plan.expandedLength(), cache.nextPosition());
        }
        for (int step = 1; step <= lateStep; step++) {
          boolean endpoint = step == 1 || step == midStep || step == lateStep;
          if (endpoint) {
            MLXMemory.clearCache();
          }
          try (MLXScope activation = generation.newChild()) {
            MLX.eval(Qwen3VlTestSupport.decode(model, activation, plan, caches, 7));
            MLX.eval(DecoderModel.stepCacheArrays(caches));
          }
          retained = settled() - baseline;
          long logical = kvPerPosition * (plan.expandedLength() + step);
          assertTrue(retained >= logical, "retained cache undergrew its payload");
          if (endpoint) {
            // Clearing the free-buffer pool before each endpoint removes reuse slack. Native
            // metadata lives on the host; activeBytes counts only the evaluated tensor buffers.
            assertEquals(logical, retained, "decode step " + step);
          }
          if (step == 1) {
            early = retained;
          }
          if (step == midStep) {
            mid = retained;
          }
          if (step == lateStep) {
            late = retained;
          }
        }
        assertEquals((lateStep - 1L) * kvPerPosition, late - early);
        assertEquals((lateStep - (long) midStep) * kvPerPosition, late - mid);
        deadKeys = caches.getFirst().keys();
      }
      assertThrows(IllegalStateException.class, deadKeys::toFloatArray);
      assertTrue(settled() <= baseline + CLEANUP_MARGIN);
      model.setObserverForTest(ignored -> {});
      // Direct feature probe verifies hoisting out of each image child and death with prefill.
      Qwen3VlVisionFeatures deadFeatures;
      try (MLXScope prefill = root.newChild()) {
        deadFeatures = model.features(CancellationToken.NONE, request.images(), prefill);
        assertEquals(
            featureBytes / 4,
            deadFeatures.merged().toFloatArray().length
                + deadFeatures.deepstackLevels().stream()
                    .mapToInt(a -> a.toFloatArray().length)
                    .sum());
      }
      assertThrows(IllegalStateException.class, deadFeatures.merged()::toFloatArray);
      for (var level : deadFeatures.deepstackLevels()) {
        assertThrows(IllegalStateException.class, level::toFloatArray);
      }
      assertTrue(settled() <= baseline + CLEANUP_MARGIN);
      System.out.println(
          "Qwen3-VL retention: baseline="
              + baseline
              + ", KV_bytes_per_position="
              + kvPerPosition
              + ", feature_bytes="
              + featureBytes
              + ", released="
              + (beforeClose[0] - baseline - kvPerPosition * plan.expandedLength())
              + ", early="
              + early
              + ", mid="
              + mid
              + ", late="
              + late);
    }
  }

  @Test
  void repeatedMixedRequestsAndFailuresReturnToBaseline(@TempDir Path directory) throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model = Qwen3VlModel.load(root, Qwen3VlModelFloatGoldenTest.directory());
      var requests = Qwen3VlTestSupport.requests(5);
      for (var request : requests) {
        model.generate(request, ignored -> {});
      }
      var base = requests.getFirst();
      var intactTokenizer = HfTokenizer.fromDirectory(Qwen3VlModelFloatGoldenTest.directory());
      var failurePolicy =
          new GenerationConfig(
              5,
              java.util.OptionalLong.empty(),
              0,
              0,
              1,
              0,
              1,
              0,
              100,
              java.util.Set.of(),
              java.util.Set.of(),
              false);
      var intact =
          GenerationRequest.text(
                  intactTokenizer,
                  "<|vision_start|><|image_pad|><|vision_end|>text",
                  PromptSpecialTokens.OMIT,
                  failurePolicy,
                  CancellationToken.NONE)
              .withImages(base.images());
      int first = model.generate(intact, ignored -> {}).generatedTokenIds().getFirst();
      // Remove the first generated vocabulary entry to cause a real output tokenizer failure.
      var tree =
          (ObjectNode)
              JsonFiles.read(Qwen3VlModelFloatGoldenTest.directory().resolve("tokenizer.json"));
      ObjectNode vocab = (ObjectNode) tree.required("model").required("vocab");
      String missing =
          vocab.properties().stream()
              .filter(e -> e.getValue().asInt() == first)
              .map(Map.Entry::getKey)
              .findFirst()
              .orElseThrow();
      assertTrue(
          java.util.Arrays.stream(intact.promptTokenIds()).noneMatch(id -> id == first),
          "failure token must not alter input encoding");
      assertTrue(first < 10 || first > 13, "failure token must be non-special");
      vocab.remove(missing);
      Path file = directory.resolve("tokenizer.json");
      Files.writeString(file, tree.toString());
      var tokenizer = HfTokenizer.fromFile(file);
      var failing =
          GenerationRequest.text(
                  tokenizer,
                  "<|vision_start|><|image_pad|><|vision_end|>text",
                  PromptSpecialTokens.OMIT,
                  failurePolicy,
                  CancellationToken.NONE)
              .withImages(base.images());
      long baseline = settled();
      MLXMemory.resetPeak();
      for (int repeat = 0; repeat < 8; repeat++) {
        for (var request : requests) {
          int firstToken = model.generate(request, ignored -> {}).generatedTokenIds().getFirst();
          assertClean(baseline);
          for (boolean eos : List.of(true, false)) {
            var policy =
                GenerationConfig.greedyDefaults(
                    5,
                    eos ? java.util.Set.of(firstToken) : java.util.Set.of(),
                    java.util.Set.of(firstToken));
            assertEquals(
                eos ? FinishReason.EOS : FinishReason.STOP_TOKEN,
                model
                    .generate(
                        new GenerationRequest(
                                request.promptTokenIds(), policy, CancellationToken.NONE)
                            .withImages(request.images()),
                        ignored -> {})
                    .finishReason());
            assertClean(baseline);
          }
          AtomicBoolean stop = new AtomicBoolean(true);
          var stopped =
              new GenerationRequest(request.promptTokenIds(), request.config(), stop::get)
                  .withImages(request.images());
          assertEquals(
              FinishReason.CANCELLED, model.generate(stopped, ignored -> {}).finishReason());
          assertClean(baseline);
          stop.set(false);
          assertEquals(
              FinishReason.CANCELLED,
              model.generate(stopped, event -> stop.set(true)).finishReason());
          assertClean(baseline);
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
          assertClean(baseline);
          for (String boundary :
              List.of(
                  "vision",
                  "image-closed",
                  "before-prefill",
                  "prefill-evaluated",
                  "prefill-closed",
                  "decode")) {
            if (request.images().isEmpty()
                && (boundary.equals("vision") || boundary.equals("image-closed"))) {
              continue;
            }
            model.setObserverForTest(
                stage -> {
                  if (stage.equals(boundary)) {
                    throw new MLXException("injected " + boundary);
                  }
                });
            assertThrows(MLXException.class, () -> model.generate(request, ignored -> {}));
            model.setObserverForTest(ignored -> {});
            assertClean(baseline);
          }
          AtomicBoolean cancelled = new AtomicBoolean();
          model.setObserverForTest(
              stage -> {
                if (stage.equals("image-closed") || stage.equals("before-prefill")) {
                  cancelled.set(true);
                }
              });
          var cancellable =
              new GenerationRequest(request.promptTokenIds(), request.config(), cancelled::get)
                  .withImages(request.images());
          assertEquals(
              FinishReason.CANCELLED, model.generate(cancellable, ignored -> {}).finishReason());
          model.setObserverForTest(ignored -> {});
          assertClean(baseline);
          // Fail both first selection and a cached selection through an actual native error.
          for (int failingStep : new int[] {0, 1}) {
            var evaluations = new java.util.concurrent.atomic.AtomicInteger();
            model
                .decoderForTest()
                .setStepBoundaryEvaluatorForTest(
                    arrays -> {
                      if (evaluations.getAndIncrement() == failingStep) {
                        try (MLXScope fault = arrays[0].scope().newChild()) {
                          MLXShape.transpose(
                              MLX.array(fault, new float[] {1, 2, 3, 4}, new int[] {2, 2}),
                              new int[] {0, 0});
                        }
                      }
                      MLX.eval(arrays);
                    });
            assertThrows(MLXException.class, () -> model.generate(request, ignored -> {}));
            model.decoderForTest().setStepBoundaryEvaluatorForTest(StepBoundaryEvaluator.NATIVE);
            assertClean(baseline);
          }
        }
        assertEquals(
            "output decoder",
            assertThrows(
                    GenerationAbortedException.class, () -> model.generate(failing, ignored -> {}))
                .stage());
        assertClean(baseline);
      }
      System.out.println(
          "Qwen3-VL cleanup: baseline="
              + baseline
              + ", final="
              + settled()
              + ", peak="
              + MLXMemory.peakBytes()
              + ", cleanup_margin="
              + CLEANUP_MARGIN);
    }
  }

  private static long settled() {
    try {
      return MLXMemoryTestSupport.settledActiveBytes();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(interrupted);
    }
  }

  private static void assertClean(long baseline) {
    long active = settled();
    assertTrue(
        active <= baseline + CLEANUP_MARGIN, "request retained " + (active - baseline) + " bytes");
  }
}
