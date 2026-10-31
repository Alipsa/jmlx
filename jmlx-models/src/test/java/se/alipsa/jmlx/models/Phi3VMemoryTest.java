package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.Phi3VGeometryPlan;
import se.alipsa.jmlx.vision.Phi3VImageProcessor;
import se.alipsa.jmlx.vision.Phi3VImageProcessorResult;
import se.alipsa.jmlx.vision.Phi3VProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.node.ObjectNode;

/**
 * Evaluated visual release, the retained projection rows' lifetime, KV growth and cleanup through
 * the production request boundaries, plus the per-image peak transient. The tiny WP6 checkpoint's
 * geometry (two GQA layers, one KV head, head dim 64, decoder hidden 128, FLOAT32) makes the
 * per-token KV budget {@code 2 x 2 x 1 x 64 x 4 = 1024 B} and the per-image retained rows {@code
 * numImgTokens x 128 x 4 B} exact.
 */
@EnabledIfNativeAvailable
class Phi3VMemoryTest {

  // Cleanup permits only sub-buffer rounding; this is far below the smallest retained tensor (the
  // 84-byte logits vector and the 233,984-byte wide-image projection rows in this checkpoint).
  private static final long CLEANUP_MARGIN = 512;

  // The allocator is exact below 16 KiB and rounds each larger live buffer up to its 16/32/64 KiB
  // doubling tier (the 7.3b WP7 findings). At the wide prompt's retained sizes, each of the four
  // K/V buffers (119,040 B after prefill) sits in the 128 KiB tier and the single projected-rows
  // buffer (233,984 B) in the 256 KiB tier.
  private static final long KV_TIER_SLACK = 4L * 16384;
  private static final long PROJECTED_TIER_SLACK = 32768;
  private static final long TIER_SLACK = KV_TIER_SLACK + PROJECTED_TIER_SLACK;

  @Test
  void visualWorkDiesAtPrefillRetainedRowsOutliveItAndOnlyKvGrowsDuringDecode() throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model =
          Phi3VModel.load(root, Phi3VTestSupport.directory(), TinyCheckpoints.PHI3V_CLIP_LAYERS);
      int hidden = model.decoderForTest().config().hiddenSize();
      var request =
          Phi3VTestSupport.request(
              Phi3VTestSupport.tokenizer(),
              "c d <|image_1|> e s",
              List.of(Phi3VTestSupport.wide()),
              8);
      var plan = model.plan(request.promptText(), request.images());
      int prompt = plan.expandedLength();
      assertEquals(465, prompt, "the wide-cross-boundary prompt must stay below the boundary");
      long kvPerPosition =
          2L
              * model.metadata().numHiddenLayers()
              * model.decoderForTest().config().numKeyValueHeads()
              * model.descriptorForTest().text().headDim()
              * Float.BYTES;
      long projected = Phi3VTestSupport.projectedRowBytes(plan, hidden);
      // The per-image transient bound from the geometry: the pixel input, the CLIP
      // activation arrays alive inside one block at a time (the entry row, the broadcast
      // pin, the q/k/v projections, the attention output, the residual adds and the FFN
      // output -- at most twelve [S, 577, 1024]-sized arrays), the attention's five
      // [S, 16, 577, 577]-sized score buffers (the matmul result, the scaled scores, the
      // mask-filled scores, the softmax output and its fused exp temporary) and a 64 MiB
      // slack for eval temporaries.
      Phi3VImageProcessor processor =
          new Phi3VImageProcessor(
              Phi3VProcessorConfig.fromJson(
                  Files.readAllBytes(
                      Phi3VTestSupport.directory().resolve("preprocessor_config.json"))));
      RgbImage wide = request.images().getFirst();
      Phi3VGeometryPlan geometry = processor.geometry(wide.width(), wide.height());
      int realSlots = geometry.cropBlocks() + 1;
      long seq = (long) realSlots * 577 * 1024 * Float.BYTES;
      long score = (long) realSlots * 16L * 577 * 577 * Float.BYTES;
      long pixelsBytes = (long) realSlots * 336L * 336 * 3 * Float.BYTES;
      long featureBytes = (long) realSlots * 576L * 1024 * Float.BYTES;
      long perImageBound = pixelsBytes + 12L * seq + 5L * score + 64L * 1024 * 1024;

      long baseline = settled();
      long[] beforeClose = {0};
      long[] afterClose = {0};
      long[] perImagePeak = {0};
      long[] early = {0};
      long[] late = {0};
      model.setObserverForTest(
          stage -> {
            if (stage.equals("preprocess")) {
              MLXMemory.resetPeak();
            } else if (stage.equals("image-closed")) {
              perImagePeak[0] = MLXMemory.peakBytes() - baseline;
            } else if (stage.equals("before-prefill")) {
              MLXMemory.clearCache();
            } else if (stage.equals("prefill-evaluated")) {
              beforeClose[0] = settled();
            } else if (stage.equals("prefill-closed")) {
              afterClose[0] = settled();
            }
          });
      try (MLXScope generation = root.newChild()) {
        var caches = Phi3VTestSupport.caches(model, generation);
        Phi3VModel.BoundaryState boundary = new Phi3VModel.BoundaryState(true);
        int[] next = {0};
        try (MLXScope activation = generation.newChild()) {
          next[0] =
              argmax(
                  model
                      .prefill(
                          request.cancellationToken(),
                          request.images(),
                          plan,
                          activation,
                          caches,
                          boundary)
                      .logits()
                      .toFloatArray());
        }
        assertTrue(
            perImagePeak[0] >= pixelsBytes + featureBytes,
            "the per-image peak "
                + perImagePeak[0]
                + " B is below the "
                + (pixelsBytes + featureBytes)
                + " B of the pixel input plus the evaluated features alive during the vision work");
        assertTrue(
            perImagePeak[0] <= perImageBound,
            "the per-image peak "
                + perImagePeak[0]
                + " B exceeds the "
                + perImageBound
                + " B geometry bound");
        // After the prefill activation scope closes, only the request's retained KV and the
        // projected rows (held for the re-prefill) may survive: no undergrowth, and the excess
        // is bounded by the four K/V buffers' tier padding.
        long retained = settled() - baseline;
        long logicalRetained = kvPerPosition * prompt + projected;
        assertTrue(
            retained >= logicalRetained,
            "retained "
                + retained
                + " B undergrew the "
                + logicalRetained
                + " B of KV plus retained projection rows");
        assertTrue(
            retained - logicalRetained <= TIER_SLACK + CLEANUP_MARGIN,
            "retained "
                + retained
                + " B exceeds the logical "
                + logicalRetained
                + " B by more than the tier slack");
        // Sampled while the activation scope is still open: the returned logits vector (vocab
        // f32) additionally lives in it.
        long logicalAtClose = logicalRetained + (long) model.metadata().vocabSize() * Float.BYTES;
        assertTrue(afterClose[0] - baseline >= logicalAtClose);
        assertTrue(afterClose[0] - baseline - logicalAtClose <= TIER_SLACK + CLEANUP_MARGIN);
        // And before the prefill scope closed, the evaluated per-image features were additionally
        // alive (the pixel input and the encoder's intermediates are already released by the
        // per-image eval and vision-scope closure): the prefill closure releases them.
        assertTrue(
            beforeClose[0] - baseline - retained >= featureBytes,
            "prefill did not release the per-image feature buffers");
        // Every retained cache remains valid after the visual closure.
        for (var cache : caches) {
          assertTrue(cache.keys().toFloatArray().length > 0);
          assertTrue(cache.values().toFloatArray().length > 0);
          assertEquals(prompt, cache.nextPosition());
        }
        // Decode steps 1..5 stay at or below the boundary (entry positions 465..469), so the
        // retained rows are never consumed and only KV grows.
        int[] history = new int[plan.expandedLength() + 5];
        int historyLength = 0;
        for (int id : plan.expandedIds()) {
          history[historyLength++] = Math.max(id, 0);
        }
        boundary.history = history;
        boundary.historyLength = historyLength;
        for (int step = 1; step <= 5; step++) {
          if (step == 1 || step == 5) {
            MLXMemory.clearCache();
          }
          try (MLXScope activation = generation.newChild()) {
            MLXArray ids = MLX.array(activation, new int[] {next[0]}, new int[] {1, 1});
            MLXArray logits = model.decodeStep(ids, caches, boundary, next[0]);
            MLX.eval(DecoderModel.stepCacheArrays(caches));
            next[0] = argmax(logits.toFloatArray());
          }
          long now = settled() - baseline;
          long logical = kvPerPosition * (prompt + step) + projected;
          assertTrue(
              now >= logical,
              "decode step "
                  + step
                  + ": retained "
                  + now
                  + " B undergrew the "
                  + logical
                  + " B of KV plus retained rows");
          if (step == 1 || step == 5) {
            // The pool was cleared before this step, so the excess over the logical payload is
            // at most the four K/V buffers' tier padding.
            assertTrue(
                now <= logical + TIER_SLACK + CLEANUP_MARGIN,
                "decode step "
                    + step
                    + ": retained "
                    + now
                    + " B exceeds the logical "
                    + logical
                    + " B by more than the tier slack");
          }
          if (step == 1) {
            early[0] = now;
          }
          if (step == 5) {
            late[0] = now;
          }
        }
        // The slope between the two cleared endpoints is the per-token KV budget for the four
        // steps, plus the tier padding difference between the endpoints.
        assertTrue(
            late[0] - early[0] <= 4L * kvPerPosition + TIER_SLACK,
            "decode slope "
                + (late[0] - early[0])
                + " B over 4 steps exceeds the "
                + (4L * kvPerPosition + TIER_SLACK)
                + " B per-token budget bound");
      }
      // The generation scope's closure releases the KV and the retained projection rows together.
      assertTrue(settled() <= baseline + CLEANUP_MARGIN);
      model.setObserverForTest(ignored -> {});
      // Direct tower probe: one image's per-slot features live in the caller's prefill scope and
      // die with it.
      Phi3VImageProcessorResult prepared = processor.process(wide);
      MLXArray deadFeatures;
      try (MLXScope prefill = root.newChild()) {
        float[] channelMajor = Arrays.copyOf(prepared.values(), realSlots * 3 * 336 * 336);
        MLXArray pixels =
            MLXShape.transpose(
                MLX.array(prefill, channelMajor, new int[] {realSlots, 3, 336, 336}),
                new int[] {0, 2, 3, 1});
        deadFeatures = model.towerForTest().forward(pixels);
        MLX.eval(deadFeatures);
      }
      assertThrows(IllegalStateException.class, deadFeatures::toFloatArray);
      assertTrue(settled() <= baseline + CLEANUP_MARGIN);
      System.out.println(
          "Phi-3.5-Vision retention: baseline="
              + baseline
              + ", KV_bytes_per_position="
              + kvPerPosition
              + ", projected_row_bytes="
              + projected
              + ", per_image_peak="
              + perImagePeak[0]
              + ", decode_slope_over_4_steps="
              + (late[0] - early[0]));
    }
  }

  @Test
  void repeatedMixedRequestsAndFailuresReturnToBaseline(@TempDir Path directory) throws Exception {
    try (MLXScope root = new MLXScope()) {
      var model =
          Phi3VModel.load(root, Phi3VTestSupport.directory(), TinyCheckpoints.PHI3V_CLIP_LAYERS);
      var requests = Phi3VTestSupport.requests(5);
      for (var request : requests) {
        model.generate(request, ignored -> {});
      }
      // A request whose output tokenizer cannot decode its own first generated token.
      var base = requests.get(1); // wide, one image
      int first = model.generate(base, ignored -> {}).generatedTokenIds().getFirst();
      var tree =
          (ObjectNode) JsonFiles.read(Phi3VTestSupport.directory().resolve("tokenizer.json"));
      ObjectNode vocab = (ObjectNode) tree.required("model").required("vocab");
      String missing =
          vocab.properties().stream()
              .filter(e -> e.getValue().asInt() == first)
              .map(Map.Entry::getKey)
              .findFirst()
              .orElseThrow();
      assertTrue(
          Arrays.stream(base.promptTokenIds()).noneMatch(id -> id == first),
          "failure token must not alter input encoding");
      vocab.remove(missing);
      Path file = directory.resolve("tokenizer.json");
      Files.writeString(file, tree.toString());
      var failingTokenizer = HfTokenizer.fromFile(file);
      var failing =
          GenerationRequest.text(
                  failingTokenizer,
                  base.promptText(),
                  base.promptSpecialTokens(),
                  base.config(),
                  CancellationToken.NONE)
              .withImages(base.images());
      long baseline = settled();
      MLXMemory.resetPeak();
      for (int repeat = 0; repeat < 3; repeat++) {
        for (var request : requests) {
          int firstToken = model.generate(request, ignored -> {}).generatedTokenIds().getFirst();
          assertClean(baseline);
          for (boolean eos : List.of(true, false)) {
            var policy =
                GenerationConfig.greedyDefaults(
                    5, eos ? Set.of(firstToken) : Set.of(), Set.of(firstToken));
            var variant =
                GenerationRequest.text(
                        Phi3VTestSupport.tokenizer(),
                        request.promptText(),
                        request.promptSpecialTokens(),
                        policy,
                        CancellationToken.NONE)
                    .withImages(request.images());
            assertEquals(
                eos ? FinishReason.EOS : FinishReason.STOP_TOKEN,
                model.generate(variant, ignored -> {}).finishReason());
            assertClean(baseline);
          }
          AtomicBoolean stop = new AtomicBoolean(true);
          var stopped =
              GenerationRequest.text(
                      Phi3VTestSupport.tokenizer(),
                      request.promptText(),
                      request.promptSpecialTokens(),
                      request.config(),
                      stop::get)
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
                  "preprocess",
                  "vision",
                  "image-closed",
                  "project",
                  "before-prefill",
                  "prefill-evaluated",
                  "prefill-closed",
                  "decode")) {
            if (request.images().isEmpty()
                && (boundary.equals("preprocess")
                    || boundary.equals("vision")
                    || boundary.equals("image-closed")
                    || boundary.equals("project"))) {
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
          // Fail both the first evaluation and a cached evaluation through an actual native
          // error.
          for (int failingStep : new int[] {0, 1}) {
            var evaluations = new AtomicInteger();
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
        // The crossing request: a failure before the re-prefill reset, and one through the
        // re-prefill's own evaluation (after the reset), both must clean up.
        var crossing =
            Phi3VTestSupport.request(
                Phi3VTestSupport.tokenizer(),
                "c d <|image_1|> e s",
                List.of(Phi3VTestSupport.wide()),
                8);
        model.setObserverForTest(
            stage -> {
              if (stage.equals("re-prefill")) {
                throw new MLXException("injected re-prefill");
              }
            });
        assertThrows(MLXException.class, () -> model.generate(crossing, ignored -> {}));
        model.setObserverForTest(ignored -> {});
        assertClean(baseline);
        var rePrefillEvaluations = new AtomicInteger();
        model
            .decoderForTest()
            .setStepBoundaryEvaluatorForTest(
                arrays -> {
                  if (rePrefillEvaluations.getAndIncrement() == 6) {
                    try (MLXScope fault = arrays[0].scope().newChild()) {
                      MLXShape.transpose(
                          MLX.array(fault, new float[] {1, 2, 3, 4}, new int[] {2, 2}),
                          new int[] {0, 0});
                    }
                  }
                  MLX.eval(arrays);
                });
        assertThrows(MLXException.class, () -> model.generate(crossing, ignored -> {}));
        model.decoderForTest().setStepBoundaryEvaluatorForTest(StepBoundaryEvaluator.NATIVE);
        assertClean(baseline);
        assertEquals(
            "output decoder",
            assertThrows(
                    GenerationAbortedException.class, () -> model.generate(failing, ignored -> {}))
                .stage());
        assertClean(baseline);
      }
      System.out.println(
          "Phi-3.5-Vision cleanup: baseline="
              + baseline
              + ", final="
              + settled()
              + ", peak="
              + MLXMemory.peakBytes()
              + ", cleanup_margin="
              + CLEANUP_MARGIN);
    }
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
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
