package se.alipsa.jmlx.benchmarks;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXOps;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.models.DecoderModel;
import se.alipsa.jmlx.models.TextGenerationModels;
import se.alipsa.jmlx.nn.KVCache;
import tools.jackson.databind.ObjectMapper;

/** Internal fresh-JVM worker; all MLX operations stay on its single platform thread. */
public final class Tf32Worker {
  private Tf32Worker() {}

  /**
   * Executes the coordinator's fixed workload under the inherited native precision setting.
   *
   * @param args checkpoint, output file, prompt length, decode steps, samples, warmups, batch and
   *     matmul size
   * @throws Exception if model loading, native evaluation or report writing fails
   */
  public static void main(String[] args) throws Exception {
    Path checkpoint = Path.of(args[0]);
    int length = Integer.parseInt(args[2]);
    int steps = Integer.parseInt(args[3]);
    int samples = Integer.parseInt(args[4]);
    int warmups = Integer.parseInt(args[5]);
    int batch = Integer.parseInt(args[6]);
    int size = Integer.parseInt(args[7]);
    List<Long> matmul = new ArrayList<>();
    List<Long> prefill = new ArrayList<>();
    List<Long> decode = new ArrayList<>();
    List<Double> checksums = new ArrayList<>();
    String dtype;
    String family;
    try (MLXScope root = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(root, checkpoint);
      family = model.config().modelType();
      int vocab = model.config().vocabSize();
      int[] prompt = new int[Math.multiplyExact(batch, length)];
      for (int row = 0; row < batch; row++) {
        for (int t = 0; t < length; t++) {
          prompt[row * length + t] = (int) ((7L * t + 11L * row + 1) % vocab);
        }
      }
      MLXArray promptIds = MLX.array(root, prompt, new int[] {batch, length});
      List<MLXArray> decodeIds = new ArrayList<>();
      for (int t = 0; t < steps; t++) {
        int[] ids = new int[batch];
        for (int row = 0; row < batch; row++) {
          ids[row] = (int) ((13L * t + 17L * row + 3) % vocab);
        }
        decodeIds.add(MLX.array(root, ids, new int[] {batch, 1}));
      }
      MLX.eval(promptIds);
      MLX.eval(decodeIds.toArray(MLXArray[]::new));
      int[] promptLengths = new int[batch];
      int[] ones = new int[batch];
      Arrays.fill(promptLengths, length);
      Arrays.fill(ones, 1);
      dtype = "unknown";
      for (int run = -warmups; run < samples; run++) {
        try (MLXScope inference = root.newChild()) {
          List<KVCache> caches = new ArrayList<>();
          for (int layer = 0; layer < model.config().numHiddenLayers(); layer++) {
            caches.add(new KVCache(inference));
          }
          long prefillNs;
          double checksum;
          // Same-shape views put inputs in the activation scope so results die with each step.
          try (MLXScope activation = inference.newChild()) {
            MLXArray input = MLXShape.broadcastTo(promptIds, activation, promptIds.shape());
            MLX.eval(input);
            long start = System.nanoTime();
            MLXArray logits = forward(model, input, caches, promptLengths, batch);
            prefillNs = System.nanoTime() - start;
            dtype = logits.dtype().name();
            checksum = checksum(logits);
          }
          long decodeNs = 0;
          for (MLXArray ids : decodeIds) {
            try (MLXScope activation = inference.newChild()) {
              MLXArray input = MLXShape.broadcastTo(ids, activation, ids.shape());
              MLX.eval(input);
              long start = System.nanoTime();
              MLXArray logits = forward(model, input, caches, ones, batch);
              decodeNs += System.nanoTime() - start;
              checksum += checksum(logits);
            }
          }
          if (run >= 0) {
            prefill.add(prefillNs);
            decode.add(decodeNs);
            checksums.add(checksum);
          }
        }
      }
    }
    try (MLXScope root = new MLXScope()) {
      float[] left = new float[Math.multiplyExact(size, size)];
      float[] right = new float[left.length];
      for (int i = 0; i < left.length; i++) {
        left[i] = (float) Math.sin(i * 0.013);
        right[i] = (float) Math.cos(i * 0.017);
      }
      MLXArray a = MLX.array(root, left, new int[] {size, size});
      MLXArray b = MLX.array(root, right, new int[] {size, size});
      MLX.eval(a, b);
      for (int run = -warmups; run < samples; run++) {
        try (MLXScope step = root.newChild()) {
          MLXArray input = MLXShape.broadcastTo(a, step, a.shape());
          MLX.eval(input);
          long start = System.nanoTime();
          MLXArray result = MLXOps.matmul(input, b);
          MLX.eval(result);
          long elapsed = System.nanoTime() - start;
          float first =
              MLXShape.slice(result, new int[] {0, 0}, new int[] {1, 1}).toFloatArray()[0];
          if (!Float.isFinite(first)) {
            throw new IllegalStateException("non-finite matmul result");
          }
          if (run >= 0) {
            matmul.add(elapsed);
          }
        }
      }
    }
    new ObjectMapper()
        .writerWithDefaultPrettyPrinter()
        .writeValue(
            Path.of(args[1]).toFile(),
            Map.of(
                "tf32",
                System.getenv("MLX_ENABLE_TF32"),
                "family",
                family,
                "logits_dtype",
                dtype,
                "matmul_ns",
                matmul,
                "prefill_ns",
                prefill,
                "decode_ns",
                decode,
                "logit_checksums",
                checksums));
  }

  private static MLXArray forward(
      DecoderModel model, MLXArray ids, List<KVCache> caches, int[] valid, int batch) {
    return batch == 1 ? model.forward(ids, caches) : model.forward(ids, caches, valid);
  }

  private static double checksum(MLXArray logits) {
    double total = 0;
    for (float value : logits.toFloatArray()) {
      if (!Float.isFinite(value)) {
        throw new IllegalStateException("non-finite logits");
      }
      total += value;
    }
    return total;
  }
}
