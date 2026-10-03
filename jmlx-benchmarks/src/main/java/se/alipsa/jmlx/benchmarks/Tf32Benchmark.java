package se.alipsa.jmlx.benchmarks;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Opt-in comparison of decoder performance with and without native reduced precision. */
public final class Tf32Benchmark {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private Tf32Benchmark() {}

  /**
   * Runs {@code checkpoint output-directory [prompt-length decode-steps samples warmups forks
   * batch]}.
   *
   * @param args checkpoint, report directory and optional workload dimensions
   * @throws Exception if a worker, native runtime or report operation fails
   */
  public static void main(String[] args) throws Exception {
    Options options = Options.parse(args);
    Files.createDirectories(options.output());
    final Map<String, String> hashes = checkpointHashes(options.checkpoint());
    List<JsonNode> forks = new ArrayList<>();
    for (int pair = 0; pair < options.forks(); pair++) {
      for (String mode : pair % 2 == 0 ? List.of("1", "0") : List.of("0", "1")) {
        Path result = options.output().resolve("pair-" + pair + "-tf32-" + mode + ".json");
        System.out.println(
            "Pair " + (pair + 1) + "/" + options.forks() + ": MLX_ENABLE_TF32=" + mode);
        runWorker(options, mode, result);
        forks.add(MAPPER.readTree(result.toFile()));
      }
    }
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("schema", "jmlx-tf32-benchmark-1");
    report.put("created_at", Instant.now().toString());
    report.put("checkpoint", options.checkpoint().toString());
    report.put("checkpoint_sha256", hashes);
    report.put("java_version", System.getProperty("java.version"));
    report.put("os_name", System.getProperty("os.name"));
    report.put("os_version", System.getProperty("os.version"));
    report.put("os_arch", System.getProperty("os.arch"));
    report.put("cpu", cpu());
    report.put("native_pin", nativePin());
    report.put("workload", options);
    report.put("forks", forks);
    Map<String, Object> prefill = comparison(forks, "prefill_ns", options.forks());
    Map<String, Object> decode = comparison(forks, "decode_ns", options.forks());
    Map<String, Object> matmul = comparison(forks, "matmul_ns", options.forks());
    report.put("matmul", matmul);
    report.put("prefill", prefill);
    report.put("decode", decode);
    MAPPER
        .writerWithDefaultPrettyPrinter()
        .writeValue(options.output().resolve("comparison.json").toFile(), report);
    String summary =
        "# TF32 benchmark\n\n"
            + "Checkpoint: "
            + options.checkpoint()
            + "\n\n"
            + "Batch: "
            + options.batch()
            + "; prompt: "
            + options.promptLength()
            + "; decode steps: "
            + options.decodeSteps()
            + "; samples per JVM: "
            + options.samples()
            + "; warmups: "
            + options.warmups()
            + "; fork pairs: "
            + options.forks()
            + "\n\n"
            + "| Phase | TF32 enabled median (ms) | Disabled median (ms) | Disabled/enabled paired"
            + " ratio |\n"
            + "| --- | ---: | ---: | ---: |\n"
            + row("Float32 square matmul (" + options.matmulSize() + ")", matmul)
            + row("Prefill", prefill)
            + row("Cached decode (all steps)", decode)
            + "\n"
            + "A ratio above 1 means disabling TF32 was slower. Ratios use each pair's JVM medians;"
            + " displayed times pool all samples. Raw samples and per-pair ratios are in"
            + " comparison.json.\n\n"
            + "Matmul measures graph construction plus MLX.eval on fixed float32 square inputs.\n\n"
            + "Decoder timers include graph construction and synchronized GPU execution through"
            + " DecoderModel.forward. Prefill projects every position, unlike generation's"
            + " last-position-only projection. Loading, input preparation, warmup, host logit reads"
            + " and report writing are excluded. Decode uses fixed token IDs and a fresh cache per"
            + " sample. No sampling or tokenization.\n\n"
            + "The environment switch is explicit in each fresh JVM. A timing difference does not"
            + " prove hardware used TF32; unsupported hardware or non-float32 checkpoints may show"
            + " no effect. Small synthetic checkpoints measure overhead and are not representative"
            + " of large LLMs.\n";
    Files.writeString(options.output().resolve("comparison.md"), summary);
    System.out.println(summary);
  }

  private static String row(String label, Map<String, Object> data) {
    return String.format(
        Locale.ROOT,
        "| %s | %.3f | %.3f | %.3fx |%n",
        label,
        ((Number) data.get("enabled_median_ns")).doubleValue() / 1e6,
        ((Number) data.get("disabled_median_ns")).doubleValue() / 1e6,
        ((Number) data.get("disabled_over_enabled_median_ratio")).doubleValue());
  }

  private static void runWorker(Options options, String mode, Path output) throws Exception {
    List<String> command =
        new ArrayList<>(
            List.of(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--enable-native-access=ALL-UNNAMED"));
    String nativePath = System.getProperty("jmlx.library.path");
    if (nativePath != null) {
      command.add("-Djmlx.library.path=" + nativePath);
    }
    command.addAll(
        List.of(
            "-cp",
            System.getProperty("java.class.path"),
            Tf32Worker.class.getName(),
            options.checkpoint().toString(),
            output.toString(),
            Integer.toString(options.promptLength()),
            Integer.toString(options.decodeSteps()),
            Integer.toString(options.samples()),
            Integer.toString(options.warmups()),
            Integer.toString(options.batch()),
            Integer.toString(options.matmulSize())));
    ProcessBuilder builder = new ProcessBuilder(command).inheritIO();
    builder.environment().put("MLX_ENABLE_TF32", mode);
    Process worker = builder.start();
    try {
      int status = worker.waitFor();
      if (status != 0) {
        throw new IllegalStateException("TF32=" + mode + " worker exited with " + status);
      }
    } finally {
      if (worker.isAlive()) {
        worker.destroyForcibly();
      }
    }
  }

  static Map<String, Object> comparison(List<JsonNode> forks, String field, int pairs) {
    List<Double> enabled = new ArrayList<>();
    List<Double> disabled = new ArrayList<>();
    List<Double> ratios = new ArrayList<>();
    for (int pair = 0; pair < pairs; pair++) {
      List<Double> on = samples(forks.get(2 * pair), field);
      List<Double> off = samples(forks.get(2 * pair + 1), field);
      if (forks.get(2 * pair).path("tf32").asText().equals("0")) {
        List<Double> swap = on;
        on = off;
        off = swap;
      }
      enabled.addAll(on);
      disabled.addAll(off);
      ratios.add(median(off) / median(on));
    }
    return Map.of(
        "enabled_median_ns",
        median(enabled),
        "disabled_median_ns",
        median(disabled),
        "disabled_over_enabled_median_ratio",
        median(ratios),
        "paired_ratios",
        ratios);
  }

  private static List<Double> samples(JsonNode fork, String field) {
    List<Double> values = new ArrayList<>();
    for (JsonNode value : fork.path(field)) {
      values.add(value.asDouble());
    }
    return values;
  }

  static double median(List<Double> values) {
    if (values.isEmpty()) {
      throw new IllegalArgumentException("no samples");
    }
    double[] sorted = values.stream().mapToDouble(Double::doubleValue).sorted().toArray();
    int mid = sorted.length / 2;
    return sorted.length % 2 == 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  private static Map<String, String> checkpointHashes(Path directory) throws Exception {
    Map<String, String> hashes = new LinkedHashMap<>();
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.filter(Files::isRegularFile).sorted().toList()) {
        String name = path.getFileName().toString();
        if (!(name.equals("config.json")
            || name.endsWith(".safetensors")
            || name.endsWith(".safetensors.index.json"))) {
          continue;
        }
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = Files.newInputStream(path)) {
          byte[] buffer = new byte[65536];
          int count;
          while ((count = input.read(buffer)) != -1) {
            digest.update(buffer, 0, count);
          }
        }
        hashes.put(
            directory.relativize(path).toString(), HexFormat.of().formatHex(digest.digest()));
      }
    }
    return hashes;
  }

  private static Properties nativePin() throws Exception {
    Properties pin = new Properties();
    String directory = System.getProperty("jmlx.library.path", System.getenv("JMLX_LIBRARY_PATH"));
    if (directory != null && !directory.isBlank()) {
      Path file = Path.of(directory, "native-pin.properties");
      if (Files.isRegularFile(file)) {
        try (InputStream input = Files.newInputStream(file)) {
          pin.load(input);
        }
      }
    }
    return pin;
  }

  private static String cpu() throws Exception {
    if (!System.getProperty("os.name").equals("Mac OS X")) {
      return "unknown";
    }
    Process process =
        new ProcessBuilder("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string").start();
    String value =
        new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8)
            .trim();
    return process.waitFor() == 0 ? value : "unknown";
  }

  record Options(
      Path checkpoint,
      Path output,
      int promptLength,
      int decodeSteps,
      int samples,
      int warmups,
      int forks,
      int batch,
      int matmulSize) {
    static Options parse(String[] args) {
      if (args.length < 2 || args.length > 9) {
        throw new IllegalArgumentException(
            "usage: Tf32Benchmark checkpoint output-directory"
                + " [prompt-length decode-steps samples warmups forks batch matmul-size]");
      }
      int[] dimensions = {32, 16, 100, 20, 5, 1, 2048};
      for (int i = 2; i < args.length; i++) {
        dimensions[i - 2] = Integer.parseInt(args[i]);
      }
      for (int i = 0; i < dimensions.length; i++) {
        if (dimensions[i] < (i == 3 ? 0 : 1)) {
          throw new IllegalArgumentException("dimensions must be positive; warmups may be zero");
        }
      }
      Math.multiplyExact(dimensions[0], dimensions[5]);
      Math.multiplyExact(dimensions[6], dimensions[6]);
      return new Options(
          Path.of(args[0]).toAbsolutePath().normalize(),
          Path.of(args[1]).toAbsolutePath().normalize(),
          dimensions[0],
          dimensions[1],
          dimensions[2],
          dimensions[3],
          dimensions[4],
          dimensions[5],
          dimensions[6]);
    }
  }
}
