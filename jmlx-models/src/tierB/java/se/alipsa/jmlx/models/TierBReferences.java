package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import se.alipsa.jmlx.core.MLXMemory;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Shared opt-in reference checks; task and numerical eligibility never depend on the host tuple.
 */
final class TierBReferences {
  final Path directory;
  final JsonNode manifest;
  final JsonNode reference;
  final double epsilon;
  final double margin;
  private double maximumError;

  TierBReferences(String task) throws Exception {
    String directory = System.getProperty("jmlx.tier.b.model.dir", "");
    assumeTrue(!directory.isBlank(), "Set JMLX_TIER_B_MODEL_DIR to run Tier B");
    this.directory = Path.of(directory);
    manifest =
        new ObjectMapper().readTree(Path.of(System.getProperty("jmlx.tier.b.manifest")).toFile());
    assertEquals(task, manifest.path("task").asString());
    String assertedTask = System.getProperty("jmlx.tier.b.expected.task", "");
    if (!assertedTask.isEmpty()) {
      assertEquals(task, assertedTask);
    }
    reference = manifest.required("reference");
    epsilon = reference.required("epsilon").asDouble();
    margin = reference.required("margin").asDouble();
    assertTrue(epsilon > 0 && Double.isFinite(epsilon) && margin > 0 && Double.isFinite(margin));
    for (var file : manifest.required("files").properties()) {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      try (var stream = Files.newInputStream(this.directory.resolve(file.getKey()))) {
        byte[] buffer = new byte[1024 * 1024];
        int count;
        while ((count = stream.read(buffer)) != -1) {
          digest.update(buffer, 0, count);
        }
      }
      assertEquals(
          file.getValue().required("sha256").asString(),
          HexFormat.of().formatHex(digest.digest()),
          file.getKey());
    }
    MLXMemory.resetPeak();
  }

  void model(ModelMetadata metadata) {
    String type = System.getProperty("jmlx.tier.b.expected.model.type", "");
    if (!type.isEmpty()) {
      assertEquals(type, metadata.modelType());
    }
  }

  void compare(JsonNode expected, float[] actual) {
    assertEquals(expected.size(), actual.length);
    for (int i = 0; i < actual.length; i++) {
      double error = Math.abs(expected.get(i).asDouble() - actual[i]);
      maximumError = Math.max(maximumError, error);
      assertTrue(
          Double.isFinite(error) && error <= epsilon,
          "logit/vector error at " + i + ": " + error + " > " + epsilon);
    }
  }

  void stable(double gap) {
    assertTrue(gap > 2 * epsilon + margin, "reference fails stable-margin eligibility: " + gap);
  }

  void report() throws Exception {
    Process process =
        new ProcessBuilder("ps", "-o", "rss=", "-p", Long.toString(ProcessHandle.current().pid()))
            .start();
    long rssKiB =
        Long.parseLong(
            new String(
                    process.getInputStream().readAllBytes(),
                    java.nio.charset.StandardCharsets.UTF_8)
                .trim());
    assertEquals(0, process.waitFor());
    System.out.println(
        "Tier B task="
            + manifest.path("task").asString()
            + ", max_abs_error="
            + maximumError
            + ", allowed_epsilon="
            + epsilon
            + ", margin="
            + margin
            + ", jvm_rss_kib="
            + rssKiB
            + ", native_peak_bytes="
            + MLXMemory.peakBytes()
            + ", numerical comparison=passed; margin eligibility=passed"
            + ", reference_status="
            + reference.path("status").asString()
            + ", host="
            + System.getProperty("os.version")
            + "/"
            + System.getProperty("os.arch")
            + ", recorded_pin="
            + manifest.path("recorded_pin").asString("unrecorded"));
  }

  static List<Integer> ids(JsonNode node) {
    List<Integer> result = new ArrayList<>();
    node.forEach(value -> result.add(value.intValue()));
    return result;
  }
}
