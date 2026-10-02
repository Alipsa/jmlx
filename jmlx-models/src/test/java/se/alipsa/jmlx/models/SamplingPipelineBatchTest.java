package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

/** The build/evaluate/readBack split and the stacked readback the batch scheduler relies on. */
@EnabledIfNativeAvailable
class SamplingPipelineBatchTest {
  private static final PenaltyInputs NO_HISTORY = new PenaltyInputs(new int[0], new float[0]);
  private static final int V = 8;
  private static final float[][] ROWS = {
    {1, 3, 2, 0, 5, 4, 1, 2}, {0, 0, 9, 1, 1, 1, 1, 1}, {4, 4, 4, 4, 4, 4, 4, 3}
  };

  private static GenerationConfig greedy() {
    return GenerationConfig.greedyDefaults(1, Set.of());
  }

  private static GenerationConfig sampled(int topK, float topP, boolean logProbabilities) {
    return new GenerationConfig(
        1, OptionalLong.of(7), 0.8f, topK, topP, 0, 1, 0, 0, Set.of(), Set.of(), logProbabilities);
  }

  private static MLXArray row(MLXScope scope, float[] logits) {
    return MLX.array(scope, logits, new int[] {1, 1, V});
  }

  @Test
  void splitPhasesEqualSelectForGreedyAndEverySampledShape() {
    List<GenerationConfig> policies =
        List.of(
            greedy(),
            sampled(0, 1, false),
            sampled(0, 1, true),
            sampled(3, 1, true),
            sampled(0, 0.7f, true),
            sampled(0, 0f, true),
            sampled(4, 0.9f, false));
    for (GenerationConfig policy : policies) {
      for (float[] logits : ROWS) {
        try (MLXScope direct = new MLXScope();
            SamplingPipeline whole = new SamplingPipeline(direct, policy, V);
            MLXScope directStep = direct.newChild();
            MLXScope split = new MLXScope();
            SamplingPipeline phased = new SamplingPipeline(split, policy, V);
            MLXScope splitStep = split.newChild()) {
          SamplingPipeline.Selection expected =
              whole.select(row(directStep, logits), NO_HISTORY, 0);
          SamplingPipeline.Built built = phased.build(row(splitStep, logits), NO_HISTORY);
          phased.evaluate(built);
          SamplingPipeline.Selection actual = phased.readBack(built, 0);
          assertEquals(expected.tokenId(), actual.tokenId(), policy.toString());
          assertEquals(expected.logProbability(), actual.logProbability(), policy.toString());
        }
      }
    }
  }

  @Test
  void sharedRankIndicesGiveTheSameSelectionAsPrivateOnes() {
    GenerationConfig policy = sampled(3, 0.9f, true);
    assertTrue(SamplingPipeline.needsRankIndices(policy));
    assertFalse(SamplingPipeline.needsRankIndices(sampled(0, 1, true)));
    assertFalse(SamplingPipeline.needsRankIndices(greedy()));
    try (MLXScope cohort = new MLXScope()) {
      MLXArray shared = SamplingPipeline.newRankIndices(cohort, V);
      for (float[] logits : ROWS) {
        try (MLXScope own = new MLXScope();
            SamplingPipeline privateRank = new SamplingPipeline(own, policy, V);
            MLXScope ownStep = own.newChild();
            SamplingPipeline sharing =
                new SamplingPipeline(cohort, policy, V, StepBoundaryEvaluator.NATIVE, shared);
            MLXScope step = cohort.newChild()) {
          SamplingPipeline.Selection a = privateRank.select(row(ownStep, logits), NO_HISTORY, 0);
          SamplingPipeline.Selection b = sharing.select(row(step, logits), NO_HISTORY, 0);
          assertEquals(a.tokenId(), b.tokenId());
          assertEquals(a.logProbability(), b.logProbability());
        }
      }
      // Closing a row's pipeline must not free the cohort-owned array.
      assertEquals(V, shared.shape()[2]);
    }
  }

  @Test
  void rankBasedPolicyWithoutAnArrayIsRejected() {
    try (MLXScope scope = new MLXScope()) {
      assertThrows(
          IllegalArgumentException.class,
          () ->
              new SamplingPipeline(
                  scope, sampled(3, 1, false), V, StepBoundaryEvaluator.NATIVE, null));
    }
  }

  @Test
  void stackedReadbackMatchesPerRowSelectionWithOneEvaluateCall() {
    // Mixed cohort: greedy, sampled, and a sampled row that asks for log probabilities.
    List<GenerationConfig> policies = List.of(greedy(), sampled(0, 1, false), sampled(3, 1, true));
    AtomicInteger evaluateCalls = new AtomicInteger();
    StepBoundaryEvaluator counting =
        arrays -> {
          evaluateCalls.incrementAndGet();
          MLX.eval(arrays);
        };
    try (MLXScope cohort = new MLXScope()) {
      MLXArray shared = SamplingPipeline.newRankIndices(cohort, V);
      List<SamplingPipeline> pipelines = new ArrayList<>();
      List<SamplingPipeline> reference = new ArrayList<>();
      List<MLXScope> referenceScopes = new ArrayList<>();
      for (GenerationConfig policy : policies) {
        pipelines.add(
            new SamplingPipeline(
                cohort,
                policy,
                V,
                counting,
                SamplingPipeline.needsRankIndices(policy) ? shared : null));
        MLXScope own = new MLXScope();
        referenceScopes.add(own);
        reference.add(new SamplingPipeline(own, policy, V));
      }
      try (MLXScope step = cohort.newChild()) {
        List<SamplingPipeline.Built> built = new ArrayList<>();
        for (int i = 0; i < policies.size(); i++) {
          built.add(pipelines.get(i).build(row(step, ROWS[i]), NO_HISTORY));
        }
        SamplingPipeline.Batched batched = SamplingPipeline.stack(step, built);
        assertEquals(List.of(policies.size(), 3), toList(batched.ints().shape()));
        assertEquals(List.of(policies.size()), toList(batched.logProbabilities().shape()));
        SamplingPipeline.evaluate(counting, batched);
        assertEquals(1, evaluateCalls.get(), "the whole cohort syncs once");

        boolean[] reports = new boolean[policies.size()];
        for (int i = 0; i < reports.length; i++) {
          reports[i] = policies.get(i).logProbabilities();
        }
        SamplingPipeline.RowReadBack[] rows = SamplingPipeline.readBack(batched, reports);
        assertEquals(1, evaluateCalls.get(), "reading back does not synchronize again");
        for (int i = 0; i < policies.size(); i++) {
          try (MLXScope refStep = referenceScopes.get(i).newChild()) {
            SamplingPipeline.Selection expected =
                reference.get(i).select(row(refStep, ROWS[i]), NO_HISTORY, 0);
            assertEquals(expected.tokenId(), rows[i].tokenId(), "row " + i);
            assertTrue(rows[i].finite());
            assertTrue(rows[i].temperedFinite());
            assertEquals(expected.logProbability(), rows[i].logProbability(), "row " + i);
          }
        }
        assertNull(rows[1].logProbability(), "a row that did not ask has no log probability");
      } finally {
        pipelines.forEach(SamplingPipeline::close);
        reference.forEach(SamplingPipeline::close);
        referenceScopes.forEach(MLXScope::close);
      }
    }
  }

  @Test
  void nonFiniteRowIsFlaggedWithoutDisturbingItsNeighbours() {
    GenerationConfig policy = greedy();
    try (MLXScope cohort = new MLXScope();
        SamplingPipeline a = new SamplingPipeline(cohort, policy, V);
        SamplingPipeline b = new SamplingPipeline(cohort, policy, V);
        MLXScope step = cohort.newChild()) {
      float[] poisoned = ROWS[0].clone();
      poisoned[2] = Float.NaN;
      List<SamplingPipeline.Built> built =
          List.of(
              a.build(row(step, poisoned), NO_HISTORY), b.build(row(step, ROWS[1]), NO_HISTORY));
      SamplingPipeline.Batched batched = SamplingPipeline.stack(step, built);
      SamplingPipeline.evaluate(StepBoundaryEvaluator.NATIVE, batched);
      SamplingPipeline.RowReadBack[] rows =
          SamplingPipeline.readBack(batched, new boolean[] {false, false});
      assertFalse(rows[0].finite(), "the NaN row is flagged non-finite");
      assertTrue(rows[1].finite());
      assertEquals(2, rows[1].tokenId(), "the clean row still selects its own argmax");
      assertEquals(
          "sampling stage finite-logit validation failed at decode step 4: logits must be finite",
          SamplingPipeline.nonFiniteLogits(4).getMessage());
    }
  }

  private static List<Integer> toList(int[] values) {
    List<Integer> out = new ArrayList<>();
    for (int value : values) {
      out.add(value);
    }
    return out;
  }
}
