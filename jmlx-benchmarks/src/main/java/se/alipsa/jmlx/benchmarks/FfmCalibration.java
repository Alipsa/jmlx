package se.alipsa.jmlx.benchmarks;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Opt-in FFM calibration (phase 8.2 WP2, plan section 5.1): builds the no-op calibration dylib
 * under the output directory with {@code cc -shared -O2}, records compiler/flags and source/binary
 * SHA-256s in {@code build-info.json}, forks fresh JVMs running {@link FfmCalWorker}, and pools the
 * raw samples into per-shape p10/median/p90 summaries (retaining every raw sample) in {@code
 * ffm-calibration.json}.
 *
 * <p>The summary is the input of {@link FfmTransitionEstimate}; it is a measurement baseline of the
 * FFM transition per representative downcall shape, not a causal partition of model work.
 *
 * <p>Opt-in local tooling, not part of {@code check} or the publication surface.
 */
public final class FfmCalibration {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private FfmCalibration() {}

  /**
   * Runs {@code output-directory [iterations calls forks warmup]} (defaults 200 2000 5 2000).
   *
   * @param args the output directory and optional workload dimensions
   * @throws Exception if the compiler, a worker fork or a report operation fails
   */
  public static void main(String[] args) throws Exception {
    if (args.length < 1 || args.length > 5) {
      throw new IllegalArgumentException(
          "usage: FfmCalibration output-directory [iterations calls forks warmup]");
    }
    int[] workload = {200, 2000, 5, 2000};
    for (int i = 1; i < args.length; i++) {
      workload[i - 1] = Integer.parseInt(args[i]);
    }
    for (int value : workload) {
      if (value < 1) {
        throw new IllegalArgumentException("workload dimensions must be positive");
      }
    }
    Path output = Path.of(args[0]).toAbsolutePath().normalize();
    Files.createDirectories(output);

    Path source = extractSource(output);
    Path dylib = output.resolve("libffm_cal.dylib");
    build(dylib, source);
    ObjectNode buildInfo = buildInfo(output, source, dylib);

    List<JsonNode> raw = new ArrayList<>();
    for (int f = 0; f < workload[2]; f++) {
      Path forkFile = output.resolve("raw-fork-" + f + ".json");
      System.out.println(
          "Fork "
              + (f + 1)
              + "/"
              + workload[2]
              + " (iterations="
              + workload[0]
              + ", calls="
              + workload[1]
              + ")");
      runWorker(dylib, forkFile, workload[0], workload[1], workload[3]);
      raw.add(MAPPER.readTree(forkFile.toFile()));
    }
    ObjectNode summary =
        aggregate(raw, buildInfo, workload[0], workload[1], workload[2], workload[3]);
    Path summaryFile = output.resolve("ffm-calibration.json");
    MAPPER.writerWithDefaultPrettyPrinter().writeValue(summaryFile.toFile(), summary);
    System.out.println("Wrote " + summaryFile);
  }

  /**
   * Copies the calibration C source (a classpath resource) into the output directory; the dylib and
   * the extracted source are the only build artifacts, and they stay under the output directory.
   */
  static Path extractSource(Path output) throws Exception {
    try (InputStream in = FfmCalibration.class.getResourceAsStream("/ffm_cal.c")) {
      if (in == null) {
        throw new IllegalStateException("ffm_cal.c is not on the classpath");
      }
      Path source = output.resolve("ffm_cal.c");
      Files.copy(in, source, StandardCopyOption.REPLACE_EXISTING);
      return source;
    }
  }

  /** Compiles the dylib with {@code cc -shared -O2}. */
  static void build(Path dylib, Path source) throws Exception {
    List<String> command =
        List.of("cc", "-shared", "-O2", "-o", dylib.toString(), source.toString());
    Process process;
    try {
      process = new ProcessBuilder(command).inheritIO().start();
    } catch (IOException e) {
      throw new IllegalStateException(
          "cc is not available; the FFM calibration needs a C compiler", e);
    }
    int status = process.waitFor();
    if (status != 0) {
      throw new IllegalStateException("cc exited with " + status);
    }
  }

  /** Records the compiler, flags and source/binary SHA-256s as build-info.json. */
  static ObjectNode buildInfo(Path output, Path source, Path dylib) throws Exception {
    ObjectNode info = MAPPER.createObjectNode();
    info.put("schema", "jmlx-ffm-cal-build-1");
    info.put("created_at", Instant.now().toString());
    info.put("compiler", compilerVersion());
    ArrayNode command = info.putArray("command");
    command.add("cc");
    command.add("-shared");
    command.add("-O2");
    command.add("-o");
    command.add(dylib.getFileName().toString());
    command.add(source.getFileName().toString());
    info.put("source_sha256", BaselineBenchmark.sha256File(source));
    info.put("binary_sha256", BaselineBenchmark.sha256File(dylib));
    MAPPER
        .writerWithDefaultPrettyPrinter()
        .writeValue(output.resolve("build-info.json").toFile(), info);
    return info;
  }

  /** Forks one fresh JVM running the worker (the Tf32Benchmark fresh-JVM pattern). */
  static void runWorker(Path dylib, Path output, int iterations, int calls, int warmup)
      throws Exception {
    List<String> command =
        new ArrayList<>(
            List.of(
                Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--enable-native-access=ALL-UNNAMED",
                "-cp",
                System.getProperty("java.class.path"),
                FfmCalWorker.class.getName(),
                dylib.toString(),
                output.toString(),
                Integer.toString(iterations),
                Integer.toString(calls),
                Integer.toString(warmup)));
    Process worker = new ProcessBuilder(command).inheritIO().start();
    try {
      int status = worker.waitFor();
      if (status != 0) {
        throw new IllegalStateException("calibration worker exited with " + status);
      }
    } finally {
      if (worker.isAlive()) {
        worker.destroyForcibly();
      }
    }
  }

  /**
   * Pools the per-fork raw samples into the per-shape summary (pure; the forks' outputs are data).
   */
  static ObjectNode aggregate(
      List<JsonNode> rawReports,
      ObjectNode buildInfo,
      int iterations,
      int calls,
      int forks,
      int warmup) {
    ObjectNode summary = MAPPER.createObjectNode();
    summary.put("schema", "jmlx-ffm-cal-1");
    summary.put("created_at", Instant.now().toString());
    summary.put("java_version", System.getProperty("java.version"));
    summary.put("os_name", System.getProperty("os.name"));
    summary.put("os_version", System.getProperty("os.version"));
    summary.put("os_arch", System.getProperty("os.arch"));
    summary.set("build_info", buildInfo);
    ObjectNode workload = summary.putObject("workload");
    workload.put("iterations", iterations);
    workload.put("calls", calls);
    workload.put("forks", forks);
    workload.put("warmup", warmup);
    ObjectNode shapes = summary.putObject("shapes");
    for (int i = 0; i < FfmCalWorker.KEYS.length; i++) {
      List<Double> pooled = new ArrayList<>();
      for (JsonNode report : rawReports) {
        for (JsonNode value : report.path("shapes").path(FfmCalWorker.KEYS[i]).path("nanos")) {
          pooled.add(value.asDouble());
        }
      }
      baseline(shapes.putObject(FfmCalWorker.KEYS[i]), FfmCalWorker.SYMBOLS[i], pooled);
    }
    List<Double> loop = new ArrayList<>();
    for (JsonNode report : rawReports) {
      for (JsonNode value : report.path("loop").path("nanos")) {
        loop.add(value.asDouble());
      }
    }
    baseline(summary.putObject("loop"), "empty-loop", loop);
    return summary;
  }

  private static void baseline(ObjectNode node, String symbol, List<Double> values) {
    BaselineStatistics.Quantiles quantiles = BaselineStatistics.quantiles(values);
    double min = values.stream().mapToDouble(Double::doubleValue).min().orElse(Double.NaN);
    double max = values.stream().mapToDouble(Double::doubleValue).max().orElse(Double.NaN);
    node.put("symbol", symbol);
    node.put("n", values.size());
    node.put("p10_ns", quantiles.p10());
    node.put("median_ns", quantiles.median());
    node.put("p90_ns", quantiles.p90());
    node.put("min_ns", min);
    node.put("max_ns", max);
    ArrayNode raw = node.putArray("raw");
    for (double value : values) {
      raw.add(value);
    }
  }

  private static String compilerVersion() {
    try {
      Process process = new ProcessBuilder("cc", "--version").start();
      String firstLine =
          new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8)
              .lines()
              .findFirst()
              .orElse("cc");
      return process.waitFor() == 0 ? firstLine : "cc";
    } catch (Exception e) {
      return "cc";
    }
  }
}
