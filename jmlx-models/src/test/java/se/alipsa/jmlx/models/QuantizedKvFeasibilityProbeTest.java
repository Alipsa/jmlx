package se.alipsa.jmlx.models;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Float-only feasibility probe for the 6.4.1 accuracy gates: records each fixture's top-1/top-2
 * margin distribution over the teacher-forced positions and the float chunked-versus-unchunked
 * noise. It uses no quantized result.
 */
@EnabledIfNativeAvailable
class QuantizedKvFeasibilityProbeTest {
  private static final int POSITIONS = 32;
  private static final int NOISE_CHUNK = 4;

  @ParameterizedTest
  @ValueSource(strings = {"llama", "llama31", "qwen2", "mistral", "gemma", "phi3", "mixtral"})
  void marginDistributionAndChunkNoise(String family) throws Exception {
    Path root =
        Path.of(System.getProperty("jmlx.repository.root"), "tools", "hf-reference", "goldens");
    JsonNode reference = new ObjectMapper().readTree(root.resolve(family + ".json").toFile());
    JsonNode promptNode = reference.path("prompt_ids");
    int[] prompt = new int[promptNode.size()];
    for (int i = 0; i < prompt.length; i++) {
      prompt[i] = promptNode.get(i).asInt();
    }
    Path checkpoint = root.resolve("checkpoints").resolve(family);
    try (MLXScope modelScope = new MLXScope();
        MLXScope inference = modelScope.newChild()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(modelScope, checkpoint);
      GenerationCachePolicy request =
          !"mistral".equals(family)
              ? GenerationCachePolicy.full()
              : GenerationCachePolicy.slidingWindowFromModel();
      List<float[]> positions = new ArrayList<>();
      List<Integer> tokens = new ArrayList<>();
      for (int token : prompt) {
        tokens.add(token);
      }
      List<KVCache> caches = caches(model, inference, request);
      try (MLXScope step = inference.newChild()) {
        MLXArray logits =
            model.forward(MLX.array(step, prompt, new int[] {1, prompt.length}), caches);
        positions.add(lastRow(logits));
      }
      while (positions.size() < POSITIONS) {
        int next = argmax(positions.getLast());
        tokens.add(next);
        try (MLXScope step = inference.newChild()) {
          MLXArray logits =
              model.forward(MLX.array(step, new int[] {next}, new int[] {1, 1}), caches);
          positions.add(lastRow(logits));
        }
      }
      StringBuilder margins = new StringBuilder();
      int[] excluded = new int[2];
      double[] caps = {0.01, 0.05};
      double minRange = Double.MAX_VALUE;
      for (float[] row : positions) {
        float max = -Float.MAX_VALUE;
        float second = -Float.MAX_VALUE;
        float min = Float.MAX_VALUE;
        for (float v : row) {
          min = Math.min(min, v);
          if (v > max) {
            second = max;
            max = v;
          } else if (v > second) {
            second = v;
          }
        }
        double range = max - min;
        minRange = Math.min(minRange, range);
        double margin = max - second;
        margins.append(String.format("%.4f/%.4f ", margin, range));
        for (int c = 0; c < caps.length; c++) {
          if (margin <= 2 * caps[c] * range) {
            excluded[c]++;
          }
        }
      }
      // Float chunked-versus-unchunked noise: the whole token stream through one forward versus
      // sequential forwards of NOISE_CHUNK tokens, last-position logits of each.
      int total = tokens.size() - 1;
      int[] stream = tokens.subList(0, total).stream().mapToInt(Integer::intValue).toArray();
      float[] whole;
      try (MLXScope chunkRoot = modelScope.newChild();
          MLXScope step = chunkRoot.newChild()) {
        List<KVCache> single = caches(model, chunkRoot, request);
        whole = lastRow(model.forward(MLX.array(step, stream, new int[] {1, total}), single));
      }
      float[] chunked = null;
      try (MLXScope chunkRoot = modelScope.newChild()) {
        List<KVCache> sequential = caches(model, chunkRoot, request);
        for (int start = 0; start < total; start += NOISE_CHUNK) {
          int[] piece =
              java.util.Arrays.copyOfRange(stream, start, Math.min(total, start + NOISE_CHUNK));
          try (MLXScope step = chunkRoot.newChild()) {
            chunked =
                lastRow(
                    model.forward(MLX.array(step, piece, new int[] {1, piece.length}), sequential));
          }
        }
      }
      float noise = 0;
      float wholeMax = -Float.MAX_VALUE;
      float wholeMin = Float.MAX_VALUE;
      for (int i = 0; i < whole.length; i++) {
        noise = Math.max(noise, Math.abs(whole[i] - chunked[i]));
        wholeMax = Math.max(wholeMax, whole[i]);
        wholeMin = Math.min(wholeMin, whole[i]);
      }
      System.out.printf(
          "KV_FEASIBILITY family=%s positions=%d excluded1pct=%d excluded5pct=%d"
              + " floatChunkNoise=%g floatChunkNoiseOfRangePct=%g chunk=%d margin/range=%s%n",
          family,
          positions.size(),
          excluded[0],
          excluded[1],
          noise,
          100.0 * noise / (wholeMax - wholeMin),
          NOISE_CHUNK,
          margins.toString().trim());
    }
  }

  private static List<KVCache> caches(DecoderModel model, MLXScope scope, GenerationCachePolicy p) {
    List<KVCache> caches = new ArrayList<>();
    for (int i = 0; i < model.config().numHiddenLayers(); i++) {
      caches.add(new KVCache(scope, model.resolveCachePolicy(p)));
    }
    return caches;
  }

  private static float[] lastRow(MLXArray logits) {
    int[] shape = logits.shape();
    int vocab = shape[2];
    float[] all = logits.toFloatArray();
    float[] row = new float[vocab];
    System.arraycopy(all, (shape[1] - 1) * vocab, row, 0, vocab);
    return row;
  }

  private static int argmax(float[] row) {
    int best = 0;
    for (int i = 1; i < row.length; i++) {
      if (row[i] > row[best]) {
        best = i;
      }
    }
    return best;
  }
}
