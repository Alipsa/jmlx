package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Phase 8.2 WP1 gate: launches the Java worker as a real subprocess (the shape the launcher uses)
 * and checks the negative protocol paths — validation failures exit 1 with a failed report in the
 * claimed directory, and the claim/missing-directory refusals write nothing. Every job here fails
 * validation before any model work, so no checkpoint artifact is required (the 0.6b artifact is not
 * git-committed) and the subprocess never touches native initialization.
 */
@EnabledIfNativeAvailable
class BaselineWorkerProcessTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private record ProcessResult(int exitCode, String stderr) {}

  private static Path repoRoot() {
    Path dir = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    while (dir != null && !Files.isRegularFile(dir.resolve("settings.gradle"))) {
      dir = dir.getParent();
    }
    Objects.requireNonNull(
        dir, "cannot find the repository root from " + System.getProperty("user.dir"));
    return dir;
  }

  /** Runs one worker job in a fresh JVM, the way the launcher forks it. */
  private static ProcessResult launch(Path jobFile) throws Exception {
    String libraryPath =
        Objects.requireNonNull(
            System.getProperty("jmlx.library.path"),
            "jmlx.library.path must be set by the root build.gradle");
    List<String> command = new ArrayList<>();
    command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
    command.add("--enable-native-access=ALL-UNNAMED");
    command.add("-Djmlx.library.path=" + libraryPath);
    command.add("-cp");
    command.add(System.getProperty("java.class.path"));
    command.add(BaselineWorker.class.getName());
    command.add(jobFile.toString());
    Process process = new ProcessBuilder(command).redirectErrorStream(false).start();
    String stderr = new String(process.getErrorStream().readAllBytes(), StandardCharsets.UTF_8);
    String stdout = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    int exitCode = process.waitFor();
    assertEquals("", stdout.trim(), "the worker prints nothing on stdout: " + stdout);
    return new ProcessResult(exitCode, stderr);
  }

  /** A valid job file against the committed frozen manifest, with the named defect applied. */
  private static Path writeJob(Path temp, String name, Map<String, Object> overrides)
      throws Exception {
    ObjectNode job = MAPPER.createObjectNode();
    job.put("schema", BaselineWorker.JOB_SCHEMA);
    job.put("adapter", "controlled");
    job.put("case", "qwen3-0.6b");
    job.put("prompt_length", 32);
    job.put("mode", "headline");
    job.put("timeout_s", 1800);
    job.put("precision_env", "1");
    ObjectNode requests = job.putObject("requests");
    requests.put("first", 1);
    requests.put("warmups", 5);
    requests.put("measured", 20);
    job.put("artifact_dir", temp.resolve("artifact").toString());
    Path manifest =
        repoRoot().resolve("tools").resolve("decode-baseline").resolve("workloads.json");
    job.put("manifest_path", manifest.toString());
    job.put("manifest_sha256", BaselineWorker.sha256File(manifest));
    job.put("repo_dir", repoRoot().toString());
    job.put("out_dir", temp.resolve(name).toString());
    for (Map.Entry<String, Object> entry : overrides.entrySet()) {
      switch (entry.getKey()) {
        case "schema" -> job.put("schema", (String) entry.getValue());
        case "manifest_path" -> job.put("manifest_path", (String) entry.getValue());
        case "manifest_sha256" -> job.put("manifest_sha256", (String) entry.getValue());
        case "artifact_dir" -> job.put("artifact_dir", (String) entry.getValue());
        case "timeout_s" -> job.put("timeout_s", (int) entry.getValue());
        case "requests_first" -> requests.put("first", (int) entry.getValue());
        default -> throw new IllegalArgumentException("unknown override " + entry.getKey());
      }
    }
    Path file = temp.resolve(name + ".job.json");
    Files.writeString(file, MAPPER.writeValueAsString(job), StandardCharsets.UTF_8);
    return file;
  }

  private static JsonNode workerBlock(Path outDir) throws Exception {
    return MAPPER.readTree(outDir.resolve("report.json").toFile()).path("worker");
  }

  private static void assertFailed(Path outDir, ProcessResult result, String reasonFragment)
      throws Exception {
    assertEquals(BaselineWorker.EXIT_FAILED, result.exitCode(), result.stderr());
    JsonNode worker = workerBlock(outDir);
    assertEquals("failed", worker.path("status").asString(null));
    String reason = worker.path("failure_reason").asString(null);
    assertTrue(reason != null && reason.contains(reasonFragment), String.valueOf(reason));
  }

  @Test
  void wrongSchemaExitsOneWithFailedReport(@TempDir Path temp) throws Exception {
    Path outDir = temp.resolve("schema");
    Files.createDirectories(outDir);
    Path job = writeJob(temp, "schema", Map.of("schema", "jmlx-decode-baseline-job-9"));
    assertFailed(outDir, launch(job), "job schema must be");
  }

  @Test
  void missingOutDirExitsOneAndCreatesNothing(@TempDir Path temp) throws Exception {
    Path job = writeJob(temp, "absent", Map.of());
    ProcessResult result = launch(job);
    assertEquals(BaselineWorker.EXIT_FAILED, result.exitCode(), result.stderr());
    assertTrue(
        result.stderr().contains("out_dir must be created by the launcher"), result.stderr());
    assertFalse(Files.exists(temp.resolve("absent")));
  }

  @Test
  void fileInWayOutDirExitsOne(@TempDir Path temp) throws Exception {
    Files.writeString(temp.resolve("blocker"), "not a directory", StandardCharsets.UTF_8);
    Path job = writeJob(temp, "blocker", Map.of());
    ProcessResult result = launch(job);
    assertEquals(BaselineWorker.EXIT_FAILED, result.exitCode(), result.stderr());
    assertTrue(
        result.stderr().contains("out_dir must be created by the launcher"), result.stderr());
  }

  @Test
  void preSeededClaimRefusesWithoutWriting(@TempDir Path temp) throws Exception {
    Path outDir = temp.resolve("claimed");
    Files.createDirectories(outDir);
    Files.writeString(
        outDir.resolve("worker-claim"),
        """
        {"utc":"2026-11-07T00:00:00Z","pid":99999}
        """,
        StandardCharsets.UTF_8);
    Path job = writeJob(temp, "claimed", Map.of());
    ProcessResult result = launch(job);
    assertEquals(BaselineWorker.EXIT_FAILED, result.exitCode(), result.stderr());
    assertTrue(result.stderr().contains("already claimed"), result.stderr());
    assertFalse(Files.exists(outDir.resolve("report.json")));
    assertEquals(
        List.of("worker-claim"),
        Files.list(outDir).map(p -> p.getFileName().toString()).sorted().toList());
  }

  @Test
  void nonCanonicalManifestCopyIsRejected(@TempDir Path temp) throws Exception {
    Path manifest =
        repoRoot().resolve("tools").resolve("decode-baseline").resolve("workloads.json");
    Path copy = temp.resolve("workloads-copy.json");
    Files.copy(manifest, copy);
    Path outDir = temp.resolve("copy");
    Files.createDirectories(outDir);
    Path job = writeJob(temp, "copy", Map.of("manifest_path", copy.toString()));
    assertFailed(outDir, launch(job), "canonical frozen manifest");
  }

  @Test
  void manifestShaMismatchIsRejected(@TempDir Path temp) throws Exception {
    Path outDir = temp.resolve("sha");
    Files.createDirectories(outDir);
    Path job = writeJob(temp, "sha", Map.of("manifest_sha256", "0".repeat(64)));
    assertFailed(outDir, launch(job), "manifest SHA-256 mismatch");
  }

  @Test
  void requestsDifferingFromRunPolicyAreRejected(@TempDir Path temp) throws Exception {
    Path outDir = temp.resolve("requests");
    Files.createDirectories(outDir);
    Path job = writeJob(temp, "requests", Map.of("requests_first", 2));
    assertFailed(outDir, launch(job), "requests.first differs");
  }

  @Test
  void timeoutDifferingFromRunPolicyIsRejected(@TempDir Path temp) throws Exception {
    Path outDir = temp.resolve("timeout");
    Files.createDirectories(outDir);
    Path job = writeJob(temp, "timeout", Map.of("timeout_s", 1801));
    assertFailed(outDir, launch(job), "timeout_s differs");
  }

  @Test
  void missingArtifactDirIsRejected(@TempDir Path temp) throws Exception {
    Path outDir = temp.resolve("artifact");
    Files.createDirectories(outDir);
    Path job =
        writeJob(
            temp, "artifact", Map.of("artifact_dir", temp.resolve("no-such-artifact").toString()));
    assertFailed(outDir, launch(job), "artifact_dir does not exist");
  }
}
