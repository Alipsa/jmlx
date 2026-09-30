package se.alipsa.jmlx.examples;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.models.CancellationToken;
import se.alipsa.jmlx.models.DecoderModel;
import se.alipsa.jmlx.models.GenerationCachePolicy;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationRequest;
import se.alipsa.jmlx.models.TextGenerationModels;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.nn.KVCachePolicy;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Opt-in local-checkpoint decode and native-memory benchmark. */
public final class DecodeBenchmark {
  private DecodeBenchmark() {}

  /**
   * Runs with {@code checkpoint-dir output-prefix [prompt-csv] [tokens] [samples] [warmups]
   * [policy]}.
   *
   * @param args local checkpoint, output prefix, and optional benchmark parameters
   * @throws Exception if loading, native evaluation, hashing, or report writing fails
   */
  public static void main(String[] args) throws Exception {
    if (args.length < 2 || args.length > 7) {
      throw new IllegalArgumentException(
          "usage: DecodeBenchmark checkpoint-dir output-prefix [prompt-csv] [tokens] [samples]"
              + " [warmups] [full|sliding-from-model]");
    }
    Path checkpoint = Path.of(args[0]);
    Path output = Path.of(args[1]);
    int[] prompt = args.length > 2 ? parsePrompt(args[2]) : new int[] {1, 2};
    int tokens = args.length > 3 ? Integer.parseInt(args[3]) : 8;
    int samples = args.length > 4 ? Integer.parseInt(args[4]) : 3;
    int warmups = args.length > 5 ? Integer.parseInt(args[5]) : 1;
    GenerationCachePolicy policy =
        args.length > 6 && args[6].equals("sliding-from-model")
            ? GenerationCachePolicy.slidingWindowFromModel()
            : GenerationCachePolicy.full();
    if (tokens < 1 || samples < 1 || warmups < 0) {
      throw new IllegalArgumentException("tokens/samples must be positive and warmups nonnegative");
    }
    long beforeLoad = MLXMemory.activeBytes();
    MLXMemory.resetPeak();
    long loadStart = System.nanoTime();
    try (MLXScope modelScope = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(modelScope, checkpoint);
      final long coldLoadNanos = System.nanoTime() - loadStart;
      final long afterLoad = MLXMemory.activeBytes();
      final long coldPeak = MLXMemory.peakBytes();
      // Hash only after the cold-load timer: hashing beforehand would warm the OS page cache.
      final Map<String, String> checkpointHashes = checkpointHashes(checkpoint);
      JsonNode config = new ObjectMapper().readTree(checkpoint.resolve("config.json").toFile());
      int configuredWindow = config.path("sliding_window").asInt(0);
      KVCachePolicy resolvedPolicy =
          policy.mode() == GenerationCachePolicy.Mode.FULL
              ? KVCachePolicy.full()
              : KVCachePolicy.slidingWindow(configuredWindow);
      List<Long> prefillNanos = new ArrayList<>();
      List<Long> decodeNanos = new ArrayList<>();
      List<Long> generationNanos = new ArrayList<>();
      List<Long> prefillPeak = new ArrayList<>();
      List<Long> decodePeak = new ArrayList<>();
      List<Long> activeBeforePrefill = new ArrayList<>();
      List<Long> activeAfterPrefill = new ArrayList<>();
      List<Long> activeBeforeDecode = new ArrayList<>();
      List<Long> activeAfterDecode = new ArrayList<>();
      List<Long> activeBeforeGeneration = new ArrayList<>();
      List<Long> activeAfterGeneration = new ArrayList<>();
      List<Long> peakGeneration = new ArrayList<>();
      List<List<Long>> activePerGeneratedToken = new ArrayList<>();
      for (int run = -warmups; run < samples; run++) {
        try (MLXScope inference = modelScope.newChild()) {
          List<KVCache> caches = new ArrayList<>();
          for (int layer = 0; layer < model.config().numHiddenLayers(); layer++) {
            caches.add(new KVCache(inference, resolvedPolicy));
          }
          try (MLXScope step = inference.newChild()) {
            MLXArray ids = MLX.array(step, prompt, new int[] {1, prompt.length});
            long before = MLXMemory.activeBytes();
            MLXMemory.resetPeak();
            long start = System.nanoTime();
            model.forward(ids, caches);
            long elapsed = System.nanoTime() - start;
            if (run >= 0) {
              prefillNanos.add(elapsed);
              activeBeforePrefill.add(before);
              activeAfterPrefill.add(MLXMemory.activeBytes());
              prefillPeak.add(MLXMemory.peakBytes());
            }
          }
          try (MLXScope step = inference.newChild()) {
            MLXArray ids = MLX.array(step, new int[] {prompt[prompt.length - 1]}, new int[] {1, 1});
            long before = MLXMemory.activeBytes();
            MLXMemory.resetPeak();
            long start = System.nanoTime();
            model.forward(ids, caches);
            long elapsed = System.nanoTime() - start;
            if (run >= 0) {
              decodeNanos.add(elapsed);
              activeBeforeDecode.add(before);
              activeAfterDecode.add(MLXMemory.activeBytes());
              decodePeak.add(MLXMemory.peakBytes());
            }
          }
        }
        GenerationRequest request =
            new GenerationRequest(
                    prompt,
                    GenerationConfig.greedyDefaults(tokens, Set.of()),
                    CancellationToken.NONE)
                .withCachePolicy(policy);
        MLXMemory.resetPeak();
        long before = MLXMemory.activeBytes();
        long start = System.nanoTime();
        model.generate(request, ignored -> {});
        long elapsed = System.nanoTime() - start;
        if (run >= 0) {
          generationNanos.add(elapsed);
          activeBeforeGeneration.add(before);
          activeAfterGeneration.add(MLXMemory.activeBytes());
          peakGeneration.add(MLXMemory.peakBytes());
          // Sample a separate repeat so memory-counter calls do not alter timed throughput.
          List<Long> intervalSamples = new ArrayList<>();
          model.generate(
              request,
              event -> {
                if (event.tokenId() != null) {
                  intervalSamples.add(MLXMemory.activeBytes());
                }
              });
          activePerGeneratedToken.add(intervalSamples);
        }
      }
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("schema", "jmlx-decode-benchmark-1");
      report.put("checkpoint", checkpoint.toAbsolutePath().toString());
      report.put("checkpoint_sha256", checkpointHashes);
      report.put("commit", commit());
      report.put("java", System.getProperty("java.version"));
      report.put("os", System.getProperty("os.name"));
      report.put("architecture", System.getProperty("os.arch"));
      report.put("device", System.getenv().getOrDefault("JMLX_BENCH_DEVICE", "unspecified"));
      report.put("mlx_c_pin", "fba4470");
      report.put("mlx_metal_pin", "0.31.2");
      report.put("model_type", model.config().modelType());
      report.put("model_revision", "local-checkpoint-sha256");
      report.put(
          "dtype", config.path("dtype").asText(config.path("torch_dtype").asText("unspecified")));
      report.put("checkpoint_sliding_window", config.path("sliding_window").asText("null"));
      report.put("cache_policy", policy.mode().name());
      report.put("batch", 1);
      report.put("prompt_length", prompt.length);
      report.put("tokens", tokens);
      report.put("warmups", warmups);
      report.put("samples", samples);
      report.put("cold_load_definition", "first load in fresh JVM; observed OS page-cache state");
      report.put("checkpoint_hash_timing", "after timed cold load");
      report.put("cold_load_ns", coldLoadNanos);
      report.put("active_before_load_bytes", beforeLoad);
      report.put("active_after_load_bytes", afterLoad);
      report.put("cold_load_peak_active_bytes", coldPeak);
      report.put("prefill_ns", stats(prefillNanos));
      report.put("one_token_decode_ns", stats(decodeNanos));
      report.put("sustained_generation_ns", stats(generationNanos));
      report.put("active_before_prefill_bytes", activeBeforePrefill);
      report.put("active_after_prefill_bytes", activeAfterPrefill);
      report.put("prefill_peak_active_bytes", prefillPeak);
      report.put("active_before_decode_bytes", activeBeforeDecode);
      report.put("active_after_decode_bytes", activeAfterDecode);
      report.put("decode_peak_active_bytes", decodePeak);
      report.put("active_before_generation_bytes", activeBeforeGeneration);
      report.put("tokens_per_second_median", tokens * 1_000_000_000.0 / median(generationNanos));
      report.put("active_after_generation_bytes", activeAfterGeneration);
      report.put("peak_generation_active_bytes", peakGeneration);
      report.put("active_per_generated_token_bytes", activePerGeneratedToken);
      report.put("allocator_cached_bytes", MLXMemory.cachedBytes());
      Path parent = output.toAbsolutePath().getParent();
      Files.createDirectories(parent);
      new ObjectMapper()
          .writeValue(output.resolveSibling(output.getFileName() + ".json").toFile(), report);
      String markdown =
          "# Decode benchmark\n\nCheckpoint: `"
              + checkpoint
              + "`\n\n"
              + "Cold load: "
              + coldLoadNanos / 1_000_000.0
              + " ms\n\n"
              + "Median prefill: "
              + median(prefillNanos) / 1_000_000.0
              + " ms\n\n"
              + "Median one-token decode: "
              + median(decodeNanos) / 1_000_000.0
              + " ms\n\n"
              + "Median sustained throughput: "
              + (tokens * 1_000_000_000.0 / median(generationNanos))
              + " tokens/s\n\n"
              + "Peak active native bytes: "
              + peakGeneration.stream().mapToLong(Long::longValue).max().orElse(0)
              + "\n";
      Files.writeString(output.resolveSibling(output.getFileName() + ".md"), markdown);
    }
  }

  private static int[] parsePrompt(String value) {
    int[] prompt = Arrays.stream(value.split(",")).mapToInt(Integer::parseInt).toArray();
    if (prompt.length == 0) {
      throw new IllegalArgumentException("prompt must contain token IDs");
    }
    return prompt;
  }

  private static String commit() {
    String ci = System.getenv("GITHUB_SHA");
    if (ci != null && !ci.isBlank()) {
      return ci;
    }
    try {
      Process process = new ProcessBuilder("git", "rev-parse", "HEAD").start();
      String revision = new String(process.getInputStream().readAllBytes()).trim();
      return process.waitFor() == 0 ? revision : "unknown";
    } catch (Exception unavailable) {
      return "unknown";
    }
  }

  private static Map<String, String> checkpointHashes(Path root) throws Exception {
    Map<String, String> hashes = new LinkedHashMap<>();
    try (var files = Files.walk(root)) {
      for (Path path : files.filter(Files::isRegularFile).sorted().toList()) {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = Files.newInputStream(path)) {
          byte[] chunk = new byte[65536];
          int read;
          while ((read = input.read(chunk)) != -1) {
            digest.update(chunk, 0, read);
          }
        }
        hashes.put(root.relativize(path).toString(), HexFormat.of().formatHex(digest.digest()));
      }
    }
    return hashes;
  }

  private static Map<String, Long> stats(List<Long> data) {
    List<Long> sorted = data.stream().sorted().toList();
    return Map.of(
        "median",
        median(sorted),
        "p10",
        sorted.get((int) Math.floor(.1 * (sorted.size() - 1))),
        "p90",
        sorted.get((int) Math.ceil(.9 * (sorted.size() - 1))));
  }

  private static long median(List<Long> data) {
    List<Long> sorted = data.stream().sorted().toList();
    return sorted.get(sorted.size() / 2);
  }
}
