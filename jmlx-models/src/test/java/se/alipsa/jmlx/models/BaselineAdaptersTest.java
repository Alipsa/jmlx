package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

/**
 * Phase 8.2 WP1 gate: the benchmark-only adapters (req/plans/phase8-2-plan.md section 3.2/3.3)
 * prove their boundaries, exact step/token counts and public-generation equivalence on the tiny
 * all-zero llama fixture. The fixture makes every greedy selection token 0 (first-maximum tie-break
 * over equal logits), so token-identity assertions are the binding shape checks; no real-model
 * performance claim is made here.
 */
@EnabledIfNativeAvailable
class BaselineAdaptersTest {
  private static final int EMITTED = 128;

  @Test
  void controlledEmitsExactlyRequestedTokensWithFreshCaches(@TempDir Path dir) throws Exception {
    BaselineWorkerTestSupport.writeTinyArtifact(dir);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model =
          assertInstanceOf(DecoderModel.class, TextGenerationModels.load(scope, dir));
      int[] prompt = BaselineWorkerTestSupport.promptIds(32);

      BaselineAdapters.ControlledResult first =
          BaselineAdapters.runControlled(model, scope, prompt, 8);

      assertEquals(8, first.tokens().size());
      assertEquals(8, first.eventS().length);
      assertEquals(7, first.steps().size(), "exactly emitted - 1 cached steps");
      for (int k = 0; k < first.steps().size(); k++) {
        BaselineAdapters.Step step = first.steps().get(k);
        assertEquals(33 + k, step.position(), "1-based cache length after the step");
        assertTrue(step.s() > 0.0, "each step timer measures a positive interval");
        assertNull(step.argmaxId(), "headline steps record no argmax");
        assertNull(step.top2Margin(), "headline steps record no margin");
      }
      assertTrue(first.prefillS() > 0.0);
      assertTrue(first.ttftS() > 0.0);
      assertTrue(first.ttftS() >= first.prefillS(), "token 1 is visible after its argmax");
      for (int i = 1; i < first.eventS().length; i++) {
        assertTrue(first.eventS()[i] >= first.eventS()[i - 1]);
      }
      // All-zero weights: every selection is the first maximum, token 0.
      assertEquals(List.of(0, 0, 0, 0, 0, 0, 0, 0), first.tokens());
      // A second request on fresh caches produces the identical sequence (no state leak).
      assertEquals(
          first.tokens(), BaselineAdapters.runControlled(model, scope, prompt, 8).tokens());
    }
  }

  @Test
  void controlledEmitsFullProductionCountOnTinyModel(@TempDir Path dir) throws Exception {
    BaselineWorkerTestSupport.writeTinyArtifact(dir);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model =
          assertInstanceOf(DecoderModel.class, TextGenerationModels.load(scope, dir));
      BaselineAdapters.ControlledResult result =
          BaselineAdapters.runControlled(
              model, scope, BaselineWorkerTestSupport.promptIds(32), EMITTED);
      assertEquals(EMITTED, result.tokens().size());
      assertEquals(EMITTED, result.eventS().length);
      assertEquals(EMITTED - 1, result.steps().size());
      assertEquals(33, result.steps().get(0).position());
      assertEquals(32 + EMITTED - 1, result.steps().get(EMITTED - 2).position());
    }
  }

  @Test
  void fixedContinuationFeedsManifestShapeAndRecordsStepChecks(@TempDir Path dir) throws Exception {
    BaselineWorkerTestSupport.writeTinyArtifact(dir);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model =
          assertInstanceOf(DecoderModel.class, TextGenerationModels.load(scope, dir));
      int[] prompt = BaselineWorkerTestSupport.promptIds(512);
      int[] fed = Arrays.copyOfRange(BaselineWorkerTestSupport.promptIds(64), 32, 64);

      BaselineAdapters.FixedContinuationResult result =
          BaselineAdapters.runFixedContinuation(model, scope, prompt, fed);

      assertEquals(32, result.steps().size());
      for (int k = 0; k < result.steps().size(); k++) {
        BaselineAdapters.Step step = result.steps().get(k);
        assertEquals(513 + k, step.position(), "the fed token's 1-based position");
        assertTrue(step.s() > 0.0);
        assertEquals(0, step.argmaxId(), "all-zero logits: first maximum is token 0");
        assertEquals(0.0, step.top2Margin(), "equal logits: zero top-2 margin");
      }
      assertTrue(result.prefillS() > 0.0);
      assertTrue(result.stepRegionS() > 0.0, "step region excludes prefill but exceeds zero");
    }
  }

  @Test
  void stockMatchesControlledTokenIdsAndBoundaries(@TempDir Path dir) throws Exception {
    BaselineWorkerTestSupport.writeTinyArtifact(dir);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model =
          assertInstanceOf(DecoderModel.class, TextGenerationModels.load(scope, dir));
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      int[] prompt = BaselineWorkerTestSupport.promptIds(32);

      // Token 0 is not in the stop set, so the stock loop runs the full budget (finish length).
      BaselineAdapters.StockResult stock =
          BaselineAdapters.runStock(model, tokenizer, prompt, 8, Set.of(3));
      BaselineAdapters.ControlledResult controlled =
          BaselineAdapters.runControlled(model, scope, prompt, 8);

      assertEquals(FinishReason.MAX_TOKENS, stock.finishReason());
      assertEquals(controlled.tokens(), stock.tokens(), "same model, same greedy policy");
      assertEquals(8, stock.eventS().length);
      for (int i = 1; i < stock.eventS().length; i++) {
        assertTrue(stock.eventS()[i] >= stock.eventS()[i - 1]);
      }
      assertEquals(stock.eventS()[0], stock.ttftS(), 0.0);
      assertTrue(stock.fullGenerationS() >= stock.eventS()[stock.eventS().length - 1]);
    }
  }

  @Test
  void stockEosTerminationIncludesTheEosTokenEvent(@TempDir Path dir) throws Exception {
    BaselineWorkerTestSupport.writeTinyArtifact(dir);
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model =
          assertInstanceOf(DecoderModel.class, TextGenerationModels.load(scope, dir));
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(dir);
      int[] prompt = BaselineWorkerTestSupport.promptIds(32);

      // Every selection is token 0; the public loop emits the EOS token event, then stops.
      BaselineAdapters.StockResult eos =
          BaselineAdapters.runStock(model, tokenizer, prompt, 8, Set.of(0));
      assertEquals(FinishReason.EOS, eos.finishReason());
      assertEquals(List.of(0), eos.tokens());
      assertEquals(1, eos.eventS().length);
      assertEquals(eos.eventS()[0], eos.ttftS(), 0.0);
      assertTrue(eos.fullGenerationS() >= eos.eventS()[0]);
    }
  }

  // Review round 3, 6: the release-window failure paths (review round 2, D/E) are unit-tested
  // directly — a throwing PhaseObserver around a real scope close — rather than only by the
  // end-to-end adapter runs, which never fail their windows.

  @Test
  void releaseWindowClosesScopeWhenTheObserverDiesBeforeTheClose() {
    try (MLXScope scope = new MLXScope()) {
      RuntimeException original = new RuntimeException("original failure");
      RuntimeException observerFailure = new RuntimeException("observer failure");

      // With an exception in flight, the window's failure is attached to it, never thrown from
      // releaseWindow itself (the caller rethrows the original).
      BaselineAdapters.releaseWindow(
          (phase, position, work) -> {
            throw observerFailure;
          },
          "release",
          1,
          scope,
          original);

      assertArrayEquals(new Throwable[] {observerFailure}, original.getSuppressed());
      assertThrows(
          IllegalStateException.class,
          scope::newChild,
          "the window died before the close ran; the fallback close must have run");
    }
  }

  @Test
  void releaseWindowSuppressesObserverFailureAroundACleanClose() {
    try (MLXScope scope = new MLXScope()) {
      RuntimeException original = new RuntimeException("original failure");
      RuntimeException observerFailure = new RuntimeException("observer failure");

      BaselineAdapters.releaseWindow(
          (phase, position, work) -> {
            work.run();
            throw observerFailure;
          },
          "release",
          1,
          scope,
          original);

      // The window ran the close and then died: the observer's failure is attached to the
      // in-flight one, and the fallback close() on the already-closed scope is the idempotent
      // no-op it must be.
      assertArrayEquals(new Throwable[] {observerFailure}, original.getSuppressed());
      assertThrows(IllegalStateException.class, scope::newChild);
    }
  }

  @Test
  void releaseWindowKeepsTheOriginalWhenTheObserverDiesWithAnError() {
    try (MLXScope scope = new MLXScope()) {
      RuntimeException original = new RuntimeException("original failure");
      OutOfMemoryError observerFailure = new OutOfMemoryError("observer error");

      // An Error from the window is caught like a RuntimeException: attached to the in-flight
      // exception, never propagated out of releaseWindow in its place (review round 3, 2).
      BaselineAdapters.releaseWindow(
          (phase, position, work) -> {
            throw observerFailure;
          },
          "release",
          1,
          scope,
          original);

      assertArrayEquals(new Throwable[] {observerFailure}, original.getSuppressed());
      assertThrows(IllegalStateException.class, scope::newChild);
    }
  }

  @Test
  void releaseWindowPropagatesTheObserverFailureWithNoOriginal() {
    try (MLXScope scope = new MLXScope()) {
      RuntimeException observerFailure = new RuntimeException("observer failure");

      RuntimeException thrown =
          assertThrows(
              RuntimeException.class,
              () ->
                  BaselineAdapters.releaseWindow(
                      (phase, position, work) -> {
                        throw observerFailure;
                      },
                      "release",
                      1,
                      scope,
                      null));

      assertSame(observerFailure, thrown);
      assertThrows(IllegalStateException.class, scope::newChild);
    }
  }
}
