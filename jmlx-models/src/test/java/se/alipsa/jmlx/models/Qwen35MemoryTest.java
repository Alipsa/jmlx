package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXMemoryTestSupport;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCachePolicy;
import se.alipsa.jmlx.vision.ImageDecoder;

/**
 * Exact retained-state growth and evaluated request cleanup through the production hybrid stack.
 */
@EnabledIfNativeAvailable
class Qwen35MemoryTest {
  private static final Path ROOT = Path.of("..", "tools", "qwen35-reference");

  @Test
  void recurrentStateStaysFixedOnlyKvGrowsAndAllBorrowedStateDiesWithSession() throws Exception {
    try (MLXScope modelScope = new MLXScope()) {
      Qwen35Model model = Qwen35Model.load(modelScope, ROOT.resolve("fixtures/hf"));
      model.generate(request(), ignored -> {});
      long baseline = MLXMemoryTestSupport.settledActiveBytes();
      MLXMemory.clearCache();
      MLXArray borrowed;
      try (MLXScope request = modelScope.newChild();
          Qwen35Decoder.Session session =
              model.decoderForTest().newSession(request, KVCachePolicy.full())) {
        try (MLXScope step = request.newChild()) {
          MLXArray logits =
              session.build(MLX.array(step, new int[] {1, 3, 4, 5}, new int[] {1, 4}), null, true);
          session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
        }
        long fixed = 3L * (6 * 32 * 64 + 3 * 512) * Float.BYTES;
        assertEquals(fixed, session.recurrent(0).logicalStorageBytes() * 3);
        for (int end : new int[] {64, 128, 256}) {
          while (session.position() < end) {
            MLXMemoryTestSupport.settledActiveBytes();
            MLXMemory.clearCache();
            try (MLXScope step = request.newChild()) {
              MLXArray logits =
                  session.build(MLX.array(step, new int[] {37}, new int[] {1, 1}), null, true);
              session.evaluateAndCommit(logits, StepBoundaryEvaluator.NATIVE);
            }
          }
          // K/V allocator tiers are exact at these lengths; no visual or layer graph may survive.
          long observed = MLXMemoryTestSupport.settledActiveBytes() - baseline;
          assertEquals(fixed + 2L * end * 64 * Float.BYTES, observed, "position " + end);
        }
        borrowed = session.recurrent(0).state();
      }
      assertThrows(IllegalStateException.class, borrowed::shape);
      assertEquals(baseline, MLXMemoryTestSupport.settledActiveBytes());
    }
  }

  @Test
  void repeatedTextImageFailureAndListenerAbortRequestsReturnToExactModelBaseline()
      throws Exception {
    try (MLXScope modelScope = new MLXScope()) {
      Qwen35Model model = Qwen35Model.load(modelScope, ROOT.resolve("fixtures/hf"));
      var image = ImageDecoder.decode(ROOT.resolve("fixtures/images/rgb-73x45.png"));
      // Use two distinct wrappers, with two processed images released sequentially.
      var visual =
          new GenerationRequest(
                  new int[] {1, 118, 120, 119, 37, 118, 120, 119, 38},
                  GenerationConfig.greedyDefaults(3, Set.of()),
                  CancellationToken.NONE)
              .withImages(List.of(image, image));
      model.generate(request(), ignored -> {});
      model.generate(visual, ignored -> {});
      long baseline = MLXMemoryTestSupport.settledActiveBytes();
      for (int iteration = 0; iteration < 6; iteration++) {
        model.generate(request(), ignored -> {});
        model.generate(visual, ignored -> {});
        model.evaluatorForTest(
            roots -> {
              throw new IllegalStateException("injected execution failure");
            });
        assertThrows(IllegalStateException.class, () -> model.generate(request(), ignored -> {}));
        model.evaluatorForTest(StepBoundaryEvaluator.NATIVE);
        assertThrows(
            GenerationAbortedException.class,
            () ->
                model.generate(
                    request(),
                    event -> {
                      if (event.tokenId() != null) {
                        throw new IllegalStateException("injected listener failure");
                      }
                    }));
        assertEquals(
            baseline, MLXMemoryTestSupport.settledActiveBytes(), "request iteration " + iteration);
      }
    }
  }

  private static GenerationRequest request() {
    return new GenerationRequest(
        new int[] {1, 3, 4, 5},
        GenerationConfig.greedyDefaults(3, Set.of()),
        CancellationToken.NONE);
  }
}
