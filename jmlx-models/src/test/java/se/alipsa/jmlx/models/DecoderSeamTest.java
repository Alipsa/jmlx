package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.AttentionMask;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.MultiAxisRope;
import se.alipsa.jmlx.nn.MultiAxisRotation;
import se.alipsa.jmlx.nn.RopeSpec;

/**
 * Seam tests for the frozen WP4 decoder integration (req/plans/phase7-3b-reference-findings.md
 * §6.9) on the pinned llama checkpoint: the request-supplied {@link MultiAxisRotation} threaded
 * through the model's single-row {@code decode} step and the batched {@code stepLogits} seam with a
 * post-block hook, and the explicit {@code DecodeStep} form of {@code runStepLoop}. Every float32
 * case compares the seam against an independent manual composition over a forked cache set, built
 * from a parallel {@link DecoderAssembler} component set with identical weight values and the same
 * operation order and scopes, so those comparisons run at zero tolerance. The cases are tagged
 * {@code full-float32} so they select into {@code :jmlx-models:float32GoldenTest} with {@code
 * MLX_ENABLE_TF32=0}; the single exception is the fast-kernel-versus-explicit-rotation comparison
 * in {@link #equalAxisRotationMatchesRopePathWithinKernelTolerance}, which uses the documented 1e-5
 * tolerance. The rejection cases run in the ordinary {@code :jmlx-models:test} task.
 */
@EnabledIfNativeAvailable
class DecoderSeamTest {

  private static final int[] SECTIONS = {4, 2, 2}; // rotaryDims 16 == llama headDim

  private static MLXScope scope;
  private static DecoderModel model;
  private static DecoderAssembler.Assembled assembled;
  private static MultiAxisRope rope;

  @BeforeAll
  static void loadCheckpoint() throws Exception {
    scope = new MLXScope();
    model = (DecoderModel) TextGenerationModels.load(scope, SchedulerFixtures.checkpoint("llama"));
    Path directory = SchedulerFixtures.checkpoint("llama");
    ArchitectureDescriptor descriptor =
        ArchitectureMappings.parse(JsonFiles.read(directory.resolve("config.json")));
    assembled =
        DecoderAssembler.assemble(
            scope,
            descriptor,
            CheckpointLoader.load(scope, directory, ArchitectureMappings.tensorPlan(descriptor)));
    rope = new MultiAxisRope(10_000f, SECTIONS);
    assertEquals(descriptor.headDim(), rope.rotaryDims());
    assertEquals(descriptor.rotaryDims(), rope.rotaryDims());
  }

  @AfterAll
  static void closeCheckpoint() {
    scope.close();
  }

  // ---------------------------------------------------------------------
  // float32-comparison cases (full-float32 tag, MLX_ENABLE_TF32=0)
  // ---------------------------------------------------------------------

  /**
   * A rotation prepared at cache position 82 with the pinned negative delta (82 - 56 = 26 on all
   * three axes) must decode the next token to logits bit-identical to the manual composition over a
   * forked cache set: the embedding, every block's rotation-threaded forward, the final
   * normalization and the head, in the same scope.
   */
  @Test
  @Tag("full-float32")
  void decodeWithNegativeDeltaRotationMatchesForkedReference() {
    int[] prompt = tokens(82);
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      model.forward(MLX.array(run, prompt, new int[] {1, prompt.length}), caches);
      try (MLXScope gen = run.newChild()) {
        List<KVCache> forks = forks(caches, gen);
        MLXArray ids = MLX.array(gen, new int[] {token(82)}, new int[] {1, 1});
        MultiAxisRotation rotation = rope.prepare(gen, 1, 1, new int[][] {{26}, {26}, {26}});
        MLXArray mask =
            AttentionMask.batched(gen, new int[] {82}, new int[] {0}, new int[] {1}, 1, 83, 0);
        MLXArray x = assembled.embedding().forward(ids);
        for (int i = 0; i < model.config().numHiddenLayers(); i++) {
          x = assembled.layers().get(i).forward(x, forks.get(i), mask, new int[] {1}, rotation);
        }
        x = assembled.finalNorm().forward(x);
        MLXArray expected = assembled.lmHead().forward(x);
        MLXArray actual = model.decode(ids, caches, rotation, null);
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        assertEquals(83, caches.getFirst().nextPosition());
        assertEquals(83, forks.getFirst().nextPosition());
      }
    }
  }

  /**
   * The seam overload with {@code (null, null)} must run the old single-row decode graph byte for
   * byte: the manual reference is the same embedding, the same step-frequency derivation (null for
   * the llama {@code Base} rope), the same four-argument block forwards with the implicit causal
   * mask, the same final normalization and head, over a forked cache set.
   */
  @Test
  @Tag("full-float32")
  void decodeWithNullSeamsMatchesOldSingleRowGraph() {
    int[] prompt = tokens(40);
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      model.forward(MLX.array(run, prompt, new int[] {1, prompt.length}), caches);
      try (MLXScope gen = run.newChild()) {
        List<KVCache> forks = forks(caches, gen);
        MLXArray ids = MLX.array(gen, new int[] {token(40)}, new int[] {1, 1});
        MLXArray x = assembled.embedding().forward(ids);
        MLXArray stepFrequencies = new RopeSpec.Base(10_000f).stepFrequencies(gen, 16, 41);
        for (int i = 0; i < model.config().numHiddenLayers(); i++) {
          x = assembled.layers().get(i).forward(x, forks.get(i), null, stepFrequencies);
        }
        x = assembled.finalNorm().forward(x);
        MLXArray expected = assembled.lmHead().forward(x);
        MLXArray actual = model.decode(ids, caches, null, null);
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        assertEquals(41, caches.getFirst().nextPosition());
      }
    }
  }

  /**
   * The seam-carrying batched {@code stepLogits} must apply the rotation in every block and fire
   * the post-block hook after every block's residual output and before the final normalization: the
   * manual reference mirrors both, on a forked cache set, and the last-row logits match at zero
   * tolerance.
   */
  @Test
  @Tag("full-float32")
  void stepLogitsWithRotationAndPostBlockHookMatchesForkedReference() {
    int[] prompt = tokens(6);
    int layers = model.config().numHiddenLayers();
    int hidden = model.config().hiddenSize();
    float[][] pert = new float[layers][];
    for (int layer = 0; layer < layers; layer++) {
      pert[layer] = new float[hidden];
      for (int j = 0; j < hidden; j++) {
        pert[layer][j] = (j % 5 - 2) * 1e-3f * (layer + 1);
      }
    }
    try (MLXScope run = scope.newChild()) {
      List<KVCache> caches = caches(run);
      try (MLXScope gen = run.newChild()) {
        List<KVCache> forks = forks(caches, gen);
        MLXArray ids = MLX.array(gen, prompt, new int[] {1, prompt.length});
        MultiAxisRotation rotation =
            rope.prepare(
                gen,
                1,
                6,
                new int[][] {{0, 1, 2, 3, 4, 5}, {0, 2, 4, 2, 4, 6}, {1, 1, 3, 3, 5, 5}});
        MLXArray mask =
            AttentionMask.batched(gen, new int[] {0}, new int[] {0}, new int[] {6}, 6, 6, 0);
        MLXArray x = assembled.embedding().forward(ids);
        for (int i = 0; i < layers; i++) {
          x = assembled.layers().get(i).forward(x, forks.get(i), mask, new int[] {6}, rotation);
          x = MLXOps.add(x, MLX.array(x.scope(), pert[i], new int[] {1, 1, hidden}));
        }
        x = assembled.finalNorm().forward(x);
        MLXArray last = MLXShape.slice(x, new int[] {0, 5, 0}, new int[] {1, 6, hidden});
        MLXArray expected = assembled.lmHead().forward(last);
        MLXArray actual =
            model.stepLogits(
                ids,
                caches,
                new int[] {6},
                null,
                rotation,
                (layerIndex, hiddenStates) ->
                    MLXOps.add(
                        hiddenStates,
                        MLX.array(
                            hiddenStates.scope(), pert[layerIndex], new int[] {1, 1, hidden})));
        assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        assertEquals(6, caches.getFirst().nextPosition());
      }
    }
  }

  /**
   * The explicit {@code DecodeStep} form of {@code runStepLoop} -- a first step running the plain
   * batched prefill and every later step calling the seam's {@code decode} with {@code (null,
   * null)} -- must generate exactly the tokens {@code generate} produces for the same request: the
   * greedy selection is deterministic, and the loop body, sampler and cache policy are the
   * identical loop.
   */
  @Test
  @Tag("full-float32")
  void runStepLoopWithExplicitDecodeStepMatchesGenerate() {
    int[] prompt = tokens(10);
    GenerationRequest request = SchedulerFixtures.greedy(prompt, 8);
    GenerationResult baseline = model.generate(request, ignored -> {});
    DecoderModel.FirstStep firstStep =
        (activation, stepCaches) -> {
          MLXArray ids = MLX.array(activation, prompt, new int[] {1, prompt.length});
          int[][] before =
              model.preflightBatch(1, prompt.length, stepCaches, new int[] {prompt.length});
          MLXArray logits = model.stepLogits(ids, stepCaches, new int[] {prompt.length}, null);
          return new DecoderModel.FirstStepResult(
              logits, cs -> DecoderModel.poisonBatchIfMutated(cs, before, before[0].length));
        };
    GenerationResult explicit =
        model.runStepLoop(
            request,
            request.config(),
            prompt,
            prompt,
            prompt.length,
            model.resolveCachePolicy(request.cachePolicy()),
            ignored -> {},
            firstStep,
            (ids, caches) -> model.decode(ids, caches, null, null));
    assertEquals(baseline.finishReason(), explicit.finishReason());
    assertEquals(baseline.generatedTokenIds(), explicit.generatedTokenIds());
  }

  /**
   * With equal axes on all three rows the interleaved multi-axis rotation is mathematically the
   * plain 1-D RoPE of the llama rope path, so the two {@code stepLogits} entries must agree within
   * the documented 1e-5 tolerance. They are not expected to be bit-equal: the rope path runs the
   * {@code mlx_fast_rope} kernel while the rotation path composes the explicit cos/sin tables, and
   * the two do not share a bitwise-exact last-ulp contract (Unit 2 findings, §6.9).
   */
  @Test
  @Tag("full-float32")
  void equalAxisRotationMatchesRopePathWithinKernelTolerance() {
    int[] prompt = tokens(7);
    int[] axis = {0, 1, 2, 3, 4, 5, 6};
    try (MLXScope run = scope.newChild()) {
      List<KVCache> ropeCaches = caches(run);
      List<KVCache> rotationCaches = caches(run);
      MLXArray ids = MLX.array(run, prompt, new int[] {1, prompt.length});
      MLXArray ropeLogits = model.stepLogits(ids, ropeCaches, new int[] {7}, null);
      MultiAxisRotation rotation =
          rope.prepare(run, 1, 7, new int[][] {axis, axis.clone(), axis.clone()});
      MLXArray rotationLogits =
          model.stepLogits(ids, rotationCaches, new int[] {7}, null, rotation, null);
      assertArrayEquals(ropeLogits.toFloatArray(), rotationLogits.toFloatArray(), 1e-5f);
    }
  }

  // ---------------------------------------------------------------------
  // rejection cases (ordinary test task, TF32=1)
  // ---------------------------------------------------------------------

  /**
   * The seam's {@code decode} must keep the original one-token and populated-cache validation: an
   * empty cache is rejected before any array is written, and a two-token step is rejected with the
   * populated caches left unadvanced.
   */
  @Test
  void decodeRotationOverloadRejectsInvalidShapesBeforeCacheMutation() {
    try (MLXScope run = scope.newChild()) {
      List<KVCache> empty = caches(run);
      try (MLXScope gen = run.newChild()) {
        MLXArray ids = MLX.array(gen, new int[] {5}, new int[] {1, 1});
        MultiAxisRotation rotation = rope.prepare(gen, 1, 1, new int[][] {{0}, {0}, {0}});
        IllegalArgumentException failure =
            assertThrows(
                IllegalArgumentException.class, () -> model.decode(ids, empty, rotation, null));
        assertEquals("decode requires one token and a populated cache", failure.getMessage());
      }
      for (KVCache cache : empty) {
        assertEquals(0, cache.nextPosition());
      }

      List<KVCache> populated = caches(run);
      model.forward(MLX.array(run, tokens(4), new int[] {1, 4}), populated);
      try (MLXScope gen = run.newChild()) {
        MLXArray twoTokens = MLX.array(gen, new int[] {1, 2}, new int[] {1, 2});
        MultiAxisRotation rotation = rope.prepare(gen, 1, 1, new int[][] {{4}, {4}, {4}});
        assertThrows(
            IllegalArgumentException.class,
            () -> model.decode(twoTokens, populated, rotation, null));
        assertEquals(4, populated.getFirst().nextPosition());
      }
    }
  }

  // ---------------------------------------------------------------------
  // helpers
  // ---------------------------------------------------------------------

  private static List<KVCache> caches(MLXScope target) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.config().numHiddenLayers(); i++) {
      caches.add(new KVCache(target));
    }
    return caches;
  }

  /** One forked (evaluated) copy per cache, in the destination scope. */
  private static List<KVCache> forks(List<KVCache> caches, MLXScope destination) {
    List<KVCache> forks = new ArrayList<>();
    for (KVCache cache : caches) {
      forks.add(cache.fork(destination));
    }
    return forks;
  }

  private static int[] tokens(int n) {
    int[] ids = new int[n];
    for (int i = 0; i < n; i++) {
      ids[i] = token(i);
    }
    return ids;
  }

  /** Deterministic in-vocabulary token at position {@code i}. */
  private static int token(int i) {
    return (i * 7 + 3) % 128;
  }
}
