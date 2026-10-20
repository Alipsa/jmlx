package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.AttentionMask;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

@EnabledIfNativeAvailable
@Tag("full-float32")
class DecoderSlidingWindowTest {
  @Test
  void matchesIndependentWindowBoundaryAndLongDecodeGoldens() throws Exception {
    Path root =
        Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens");
    JsonNode reference = new ObjectMapper().readTree(root.resolve("mistral-window.json").toFile());
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = MistralModel.load(modelScope, root.resolve("checkpoints/mistral"));
      for (JsonNode item : reference.path("cases")) {
        try (MLXScope inference = modelScope.newChild()) {
          List<KVCache> caches = slidingCaches(inference, model);
          int[] prompt = ints(item.path("prompt_ids"));
          int offset = 0;
          MLXArray logits = null;
          for (JsonNode chunk : item.path("chunk_lengths")) {
            int length = chunk.asInt();
            int[] ids = java.util.Arrays.copyOfRange(prompt, offset, offset + length);
            logits = model.forward(MLX.array(inference, ids, new int[] {1, length}), caches);
            offset += length;
          }
          int[] shape = logits.shape();
          MLXArray last =
              se.alipsa.jmlx.core.MLXShape.slice(
                  logits, new int[] {0, shape[1] - 1, 0}, new int[] {1, shape[1], shape[2]});
          assertArrayEquals(
              floats(item.path("last_logits")),
              last.toFloatArray(),
              1e-4f,
              item.path("name").asText());
        }
      }
      try (MLXScope inference = modelScope.newChild()) {
        List<KVCache> caches = slidingCaches(inference, model);
        int[] prompt = {1, 7, 42, 3, 19, 5};
        model.forward(MLX.array(inference, prompt, new int[] {1, prompt.length}), caches);
        for (JsonNode step : reference.path("long_decode")) {
          int token = step.path("token_id").asInt();
          MLXArray logits =
              model.forward(MLX.array(inference, new int[] {token}, new int[] {1, 1}), caches);
          assertArrayEquals(floats(step.path("logits")), logits.toFloatArray(), 1e-4f);
        }
        assertEquals(3, caches.getFirst().length());
      }
    }
  }

  @Test
  void explicitSlidingRetentionMatchesFullMaskedAttention(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        Files.readString(config)
            .replace(
                "\"model_type\":\"llama\"", "\"model_type\":\"mistral\",\"sliding_window\":4"));
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = MistralModel.load(scope, dir);
      List<KVCache> full = new ArrayList<>();
      List<KVCache> sliding = new ArrayList<>();
      for (int i = 0; i < model.config().numHiddenLayers(); i++) {
        full.add(new KVCache(scope));
        sliding.add(new KVCache(scope, KVCachePolicy.slidingWindow(4)));
      }
      int[][] inputs = {{1, 2, 3, 4, 5, 6}, {7}, {8}, {9}};
      for (int[] input : inputs) {
        MLXArray ids = MLX.array(scope, input, new int[] {1, input.length});
        assertArrayEquals(
            model.forward(ids, full).toFloatArray(),
            model.forward(ids, sliding).toFloatArray(),
            1e-4f);
      }
      assertEquals(9, sliding.getFirst().nextPosition());
      assertEquals(3, sliding.getFirst().length());
      assertEquals(0, full.getFirst().startPosition());
    }
  }

  /**
   * The hook-only single-row route -- rotation {@code null}, a no-op post-block hook -- is
   * bit-identical at the logits to the old single-row path on a sliding-window model whose caches
   * have already evicted keys (startPosition &gt; 0): the old path's 2-D sliding-window mask and
   * the batched route's {@code [1,1,T,L]} mask match element for element, so both routes select the
   * same SDPA result. Four one-token {@code decode} steps pin the evicted-cache decode against the
   * old path itself (every retained key within the window), and a final two-token step pins a step
   * whose key length exceeds the window so the masks really mask keys: its old-path reference is
   * composed manually over a forked cache set and sliced to the last row before the head, so both
   * sides project the same shape (the {@code forward} and {@code stepLogits} entries differ in that
   * projection shape, which shifts a few ulps in the head matmul alone).
   */
  @Test
  void noOpHookDecodeMatchesOldSingleRowPathOnEvictedSlidingCaches() throws Exception {
    Path root =
        Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens");
    Path directory = root.resolve("checkpoints/mistral");
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = MistralModel.load(modelScope, directory);
      ArchitectureDescriptor descriptor =
          ArchitectureMappings.parse(JsonFiles.read(directory.resolve("config.json")));
      DecoderAssembler.Assembled assembled =
          DecoderAssembler.assemble(
              modelScope,
              descriptor,
              CheckpointLoader.load(
                  modelScope, directory, ArchitectureMappings.tensorPlan(descriptor)));
      int[] prompt = new int[8];
      for (int i = 0; i < prompt.length; i++) {
        prompt[i] = token(i);
      }
      try (MLXScope inference = modelScope.newChild()) {
        List<KVCache> plain = slidingCaches(inference, model);
        List<KVCache> hooked = slidingCaches(inference, model);
        model.forward(MLX.array(inference, prompt, new int[] {1, prompt.length}), plain);
        model.forward(MLX.array(inference, prompt, new int[] {1, prompt.length}), hooked);
        for (int i = 0; i < plain.size(); i++) {
          assertArrayEquals(
              plain.get(i).keys().toFloatArray(), hooked.get(i).keys().toFloatArray());
          assertArrayEquals(
              plain.get(i).values().toFloatArray(), hooked.get(i).values().toFloatArray());
        }
        // The 8-token prompt (window 4) evicted: three retained keys per layer, first at 5.
        assertEquals(5, plain.getFirst().startPosition());
        assertEquals(8, plain.getFirst().nextPosition());
        try (MLXScope gen = inference.newChild()) {
          int input = token(8);
          for (int step = 0; step < 4; step++) {
            MLXArray ids = MLX.array(gen, new int[] {input}, new int[] {1, 1});
            MLXArray expected = model.decode(ids, plain, null, null);
            MLXArray actual = model.decode(ids, hooked, null, (layer, hidden) -> hidden);
            float[] oldLogits = expected.toFloatArray();
            float[] hookLogits = actual.toFloatArray();
            assertArrayEquals(
                rawBits(oldLogits),
                rawBits(hookLogits),
                "step " + step + ": the no-op-hook decode must be bit-identical to the old path");
            int oldGreedy = argMax(oldLogits);
            assertEquals(oldGreedy, argMax(hookLogits), "step " + step + ": greedy lockstep");
            input = oldGreedy;
          }
          // A two-token step: keyLength 5 exceeds the window 4, so the masks really mask keys.
          int[] twoTokens = {input, token(12)};
          MLXArray ids = MLX.array(gen, twoTokens, new int[] {1, twoTokens.length});
          List<KVCache> forks = new ArrayList<>();
          for (KVCache cache : plain) {
            forks.add(cache.fork(gen));
          }
          int next = plain.getFirst().nextPosition();
          int keyStart = plain.getFirst().startPosition();
          int keyLength = plain.getFirst().length() + twoTokens.length;
          Integer window = descriptor.attention().slidingWindow();
          MLXArray x = assembled.embedding().forward(ids);
          MLXArray mask =
              AttentionMask.slidingWindow(gen, next, keyStart, twoTokens.length, keyLength, window);
          MLXArray stepFrequencies =
              descriptor
                  .rope()
                  .stepFrequencies(gen, descriptor.rotaryDims(), next + twoTokens.length);
          int layers = model.config().numHiddenLayers();
          for (int i = 0; i < layers; i++) {
            x = assembled.layers().get(i).forward(x, forks.get(i), mask, stepFrequencies);
          }
          x = assembled.finalNorm().forward(x);
          MLXArray last =
              MLXShape.slice(x, new int[] {0, 1, 0}, new int[] {1, 2, model.config().hiddenSize()});
          MLXArray expected = assembled.lmHead().forward(last);
          MLXArray actual =
              model.stepLogits(
                  ids, hooked, new int[] {twoTokens.length}, null, null, (layer, hidden) -> hidden);
          assertArrayEquals(
              rawBits(expected.toFloatArray()),
              rawBits(actual.toFloatArray()),
              "the two-token masked step must be bit-identical to the old graph");
          assertEquals(14, hooked.getFirst().nextPosition());
          assertEquals(11, hooked.getFirst().startPosition());
          assertEquals(3, hooked.getFirst().length());
        }
      }
    }
  }

  @Test
  void slidingGenerationCancellationStopsBetweenSteps(@TempDir Path dir) throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        Files.readString(config)
            .replace(
                "\"model_type\":\"llama\"", "\"model_type\":\"mistral\",\"sliding_window\":4"));
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = MistralModel.load(scope, dir);
      AtomicBoolean cancelled = new AtomicBoolean();
      GenerationRequest request =
          new GenerationRequest(
                  new int[] {1, 2, 3, 4, 5},
                  GenerationConfig.greedyDefaults(8, Set.of()),
                  cancelled::get)
              .withCachePolicy(GenerationCachePolicy.slidingWindowFromModel());
      GenerationResult result =
          model.generate(
              request,
              event -> {
                if (event.tokenId() != null) {
                  cancelled.set(true);
                }
              });
      assertEquals(FinishReason.CANCELLED, result.finishReason());
      assertEquals(1, result.generatedTokenIds().size());
    }
  }

  private static List<KVCache> slidingCaches(MLXScope scope, DecoderModel model) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.config().numHiddenLayers(); i++) {
      caches.add(new KVCache(scope, KVCachePolicy.slidingWindow(4)));
    }
    return caches;
  }

  /** The raw IEEE bits of an evaluated row, so {@code -0.0} cannot pass for {@code +0.0}. */
  private static int[] rawBits(float[] values) {
    int[] bits = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      bits[i] = Float.floatToIntBits(values[i]);
    }
    return bits;
  }

  private static int argMax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }

  /** Deterministic in-vocabulary token at position {@code i}. */
  private static int token(int i) {
    return (i * 7 + 3) % 128;
  }

  private static int[] ints(JsonNode node) {
    int[] result = new int[node.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = node.get(i).asInt();
    }
    return result;
  }

  private static float[] floats(JsonNode node) {
    float[] result = new float[node.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = (float) node.get(i).asDouble();
    }
    return result;
  }
}
