package se.alipsa.jmlx.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.core.MLXShape;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import tools.jackson.databind.ObjectMapper;

/** Opt-in evaluated Java prototypes; timing and binding-count evidence use separate JVM runs. */
@EnabledIfNativeAvailable
@Tag("qwen35-benchmark")
class GatedDeltaBenchmarkTest {
  @Test
  void recordEvaluatedPrototypeSamples() throws Exception {
    Path root = Path.of(System.getProperty("jmlx.repository.root"));
    final Path output =
        Path.of(
            System.getenv()
                .getOrDefault(
                    "JMLX_DELTA_BENCHMARK_OUTPUT",
                    root.resolve("build/benchmarks/qwen35-delta-java.json").toString()));
    int layers = Integer.parseInt(System.getenv().getOrDefault("JMLX_DELTA_BENCHMARK_LAYERS", "1"));
    final int samples =
        Integer.parseInt(System.getenv().getOrDefault("JMLX_DELTA_BENCHMARK_SAMPLES", "3"));
    final List<String> shapes =
        Arrays.asList(
            System.getenv().getOrDefault("JMLX_DELTA_BENCHMARK_SHAPES", "tiny,08b,27b").split(","));
    final int[] lengths =
        Arrays.stream(
                System.getenv()
                    .getOrDefault("JMLX_DELTA_BENCHMARK_LENGTHS", "64,256,1024,1280")
                    .split(","))
            .mapToInt(Integer::parseInt)
            .toArray();
    boolean counting = BindingCounterAgent.isInstalled();
    Map<String, Object> record = new LinkedHashMap<>();
    record.put(
        "kind", counting ? "binding counts; timings invalid" : "evaluated Java prototype timings");
    record.put("java", System.getProperty("java.runtime.version"));
    record.put("tf32", System.getenv("MLX_ENABLE_TF32"));
    record.put("git_sha", command("git", "rev-parse", "HEAD").strip());
    record.put("git_status", command("git", "status", "--porcelain"));
    record.put("layers_per_sample", layers);
    record.put(
        "gpu_dispatch_count",
        "not observed; binding counts are API downcalls, not Metal dispatches");
    List<Map<String, Object>> cases = new ArrayList<>();
    record.put("cases", cases);
    ObjectMapper mapper = new ObjectMapper();
    Files.createDirectories(output.toAbsolutePath().getParent());
    for (String shape : shapes) {
      int[] dimensions =
          switch (shape) {
            case "tiny" -> new int[] {2, 6, 32, 64};
            case "08b" -> new int[] {16, 16, 128, 128};
            case "27b" -> new int[] {16, 48, 128, 128};
            default -> throw new IllegalArgumentException("unknown shape " + shape);
          };
      for (int length : lengths) {
        try (MLXScope prepared = new MLXScope()) {
          Path inputPath =
              Path.of(
                  System.getenv("JMLX_DELTA_BENCHMARK_INPUT_DIR"),
                  shape + "-" + length + ".safetensors");
          final MLXArray[] inputs = prepare(prepared, inputPath);
          List<Map<String, Object>> measurements = new ArrayList<>();
          Map<String, Object> oneCase = new LinkedHashMap<>();
          oneCase.put("shape", shape);
          oneCase.put("dimensions", dimensions);
          oneCase.put("length", length);
          oneCase.put(
              "inputs_sha256",
              java.util.HexFormat.of()
                  .formatHex(
                      java.security.MessageDigest.getInstance("SHA-256")
                          .digest(Files.readAllBytes(inputPath))));
          oneCase.put("measurements", measurements);
          float[][] expected;
          try (MLXScope validation = prepared.newChild()) {
            GatedDeltaRule.Result reference = execute(validation, inputs, false);
            expected =
                new float[][] {reference.output().toFloatArray(), reference.state().toFloatArray()};
          }
          // Qualify the chunked implementation before collecting its performance evidence.
          try (MLXScope validation = prepared.newChild()) {
            GatedDeltaRule.Result chunked = execute(validation, inputs, true);
            assertArrayEquals(expected[0], chunked.output().toFloatArray(), 1e-4f);
            assertArrayEquals(expected[1], chunked.state().toFloatArray(), 1e-4f);
          }
          for (int sample = 0; sample < (counting ? 1 : samples); sample++) {
            for (int order = 0; order < 2; order++) {
              boolean chunked = (sample + order) % 2 == 0;
              Map<String, Object> measurement = new LinkedHashMap<>();
              measurement.put("path", chunked ? "chunked_ops" : "token_recurrence");
              measurement.put("sample", sample);
              measurement.put("host_state", hostState());
              MLXMemory.resetPeak();
              final long active = MLXMemory.activeBytes();
              if (counting) {
                BindingCounterAgent.start();
              }
              long start = System.nanoTime();
              for (int layer = 0; layer < layers; layer++) {
                try (MLXScope invocation = prepared.newChild()) {
                  execute(invocation, inputs, chunked);
                }
              }
              long elapsed = System.nanoTime() - start;
              if (counting) {
                Map<String, Long> counts = BindingCounterAgent.stop();
                assertFalse(counts.isEmpty(), "agent did not observe generated native downcalls");
                measurement.put("native_binding_calls", counts);
              } else {
                measurement.put("evaluated_and_cleanup_ns", elapsed);
              }
              measurement.put("peak_above_active", MLXMemory.peakBytes() - active);
              measurements.add(measurement);
            }
          }
          cases.add(oneCase);
          mapper.writerWithDefaultPrettyPrinter().writeValue(output.toFile(), record);
        }
      }
    }
  }

  private static GatedDeltaRule.Result execute(MLXScope scope, MLXArray[] input, boolean chunked) {
    MLXArray q = MLXShape.reshape(MLXShape.expandDims(input[0], scope, 0), input[0].shape());
    return chunked
        ? GatedDeltaRule.chunked(q, input[1], input[2], input[3], input[4], input[5])
        : GatedDeltaRule.recurrent(q, input[1], input[2], input[3], input[4], input[5]);
  }

  private static MLXArray[] prepare(MLXScope scope, Path path) {
    Map<String, MLXArray> saved = MLXIO.loadSafetensors(scope, path.toString()).tensors();
    MLXArray[] result = {
      saved.get("q"),
      saved.get("k"),
      saved.get("v"),
      saved.get("log_decay"),
      saved.get("beta"),
      MLXShape.transpose(saved.get("initial"), new int[] {0, 1, 3, 2})
    };
    MLX.eval(result);
    return result;
  }

  private static Map<String, Object> hostState() throws Exception {
    return Map.of(
        "utc",
        Instant.now().toString(),
        "system_load_average",
        ManagementFactory.getOperatingSystemMXBean().getSystemLoadAverage(),
        "vm_stat",
        command("vm_stat"),
        "top_processes",
        command("ps", "-Ao", "pid,pcpu,rss,comm", "-r").lines().limit(16).toList());
  }

  private static String command(String... arguments) throws Exception {
    Process process = new ProcessBuilder(arguments).redirectErrorStream(true).start();
    String text =
        new String(
            process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
    if (process.waitFor() != 0) {
      throw new IllegalStateException("benchmark observation command failed: " + arguments[0]);
    }
    return text;
  }
}
