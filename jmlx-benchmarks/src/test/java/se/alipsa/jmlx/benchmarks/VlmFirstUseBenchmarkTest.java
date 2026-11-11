package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * End-to-end launcher tests over a hermetic mini repository (no real model, no native runtime, no
 * network): a temp-dir repo with a tiny Tier-B-shaped manifest and digest-verified images, two
 * temp-dir git trees (each with a staged-pin file and a byte-identical worker source), and a fake
 * worker starter that simulates every protocol outcome. The tests are pure Java and run on any
 * platform; the real fork path is covered by the smoke and measurement sessions.
 */
class VlmFirstUseBenchmarkTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private static final String WORKER_SOURCE_PATH =
      "jmlx-benchmarks/src/main/java/se/alipsa/jmlx/benchmarks/VlmFirstUseWorker.java";
  private static final String WORKER_SOURCE =
      "package se.alipsa.jmlx.benchmarks;\n// fake worker\n";
  private static final String WORKER_SOURCE_SHA =
      sha256Hex(WORKER_SOURCE.getBytes(StandardCharsets.UTF_8));

  @TempDir Path base;

  /** The simulated worker outcome for one cell (workload/tree/pair). */
  private enum Behavior {
    OK,
    FAILED,
    CORRECTNESS_FAILED,
    BACKSTOP_TIMEOUT,
    DEAD,
    CRASHED,
    JOB_VERIFICATION,
    INVALID_REPORT,
    KILLED,
    THROW
  }

  /** Records the jobs it started and leaves whatever protocol artifacts the behavior demands. */
  private static final class FakeStarter implements VlmFirstUseBenchmark.WorkerStarter {
    final List<VlmFirstUseBenchmark.JobData> jobs = new ArrayList<>();
    final Map<String, Behavior> behaviors = new LinkedHashMap<>();
    final Map<String, Long> waitLimits = new LinkedHashMap<>();

    /** Delays the first simulated fork; the session-budget test uses it to burn the budget. */
    long firstStartDelayMs = 0L;

    @Override
    public int start(
        VlmFirstUseBenchmark.JobData job, Path jobFile, Path stderrLog, long waitLimitSeconds)
        throws Exception {
      String cell = job.workload().name() + "/" + job.tree().name() + "/p" + job.pairIndex();
      Behavior behavior = behaviors.getOrDefault(cell, Behavior.OK);
      jobs.add(job);
      waitLimits.put(cell, waitLimitSeconds);
      if (behavior == Behavior.THROW) {
        throw new IllegalStateException("simulated fork failure");
      }
      if (!jobs.isEmpty() && jobs.size() == 1 && firstStartDelayMs > 0) {
        Thread.sleep(firstStartDelayMs);
      }
      Files.createDirectories(stderrLog.getParent());
      Files.writeString(stderrLog, "fake stderr");
      Path cellDir = job.cell();
      switch (behavior) {
        case OK ->
            writeReport(
                cellDir,
                job,
                "ok",
                null,
                List.of(0.5, 1.0, 1.5, 2.0),
                List.of(10.0, 20.0, 30.0, 40.0));
        case FAILED -> writeReport(cellDir, job, "failed", "simulated failure", null, null);
        case CORRECTNESS_FAILED ->
            writeReport(cellDir, job, "correctness_failed", "simulated mismatch", null, null);
        case BACKSTOP_TIMEOUT ->
            writeReport(cellDir, job, "timeout", "internal backstop elapsed 930 s", null, null);
        case DEAD -> Files.writeString(cellDir.resolve("claim.json"), "{\"status\":\"started\"}");
        case CRASHED -> {}
        case JOB_VERIFICATION -> {}
        case INVALID_REPORT ->
            Files.writeString(
                cellDir.resolve("report.json"), "{\"schema\":\"wrong-schema\",\"status\":\"ok\"}");
        case KILLED -> {}
        default -> {}
      }
      if (behavior == Behavior.JOB_VERIFICATION) {
        return VlmFirstUseWorker.EXIT_JOB_VERIFICATION_FAILED;
      }
      if (behavior == Behavior.CRASHED) {
        return 134;
      }
      if (behavior == Behavior.KILLED) {
        return VlmFirstUseBenchmark.WORKER_KILLED;
      }
      return 0;
    }

    private void writeReport(
        Path cellDir,
        VlmFirstUseBenchmark.JobData job,
        String status,
        String failure,
        List<Double> ttft,
        List<Double> full)
        throws IOException {
      ObjectNode report = MAPPER.createObjectNode();
      report.put("schema", VlmFirstUseWorker.REPORT_SCHEMA);
      report.put("status", status);
      if (failure != null) {
        report.put("failure_reason", failure);
      }
      report.put("workload", job.workload().name());
      report.put("tree", job.tree().name());
      report.put("pair_index", job.pairIndex());
      report.put("wall_s", 1.0);
      ObjectNode first = report.putObject("first_request");
      first.put("ttft_s", 1.0);
      first.put("full_generation_s", 2.0);
      ObjectNode load = report.putObject("load");
      load.put("wall_s", 0.5);
      if (ttft != null) {
        var array = report.putArray("measured_ttft_s");
        ttft.forEach(array::add);
      }
      if (full != null) {
        var array = report.putArray("measured_full_generation_s");
        full.forEach(array::add);
      }
      Files.writeString(cellDir.resolve("report.json"), report.toString());
    }
  }

  /** One hermetic mini repository plus two fake trees, ready for a session spec. */
  private record Setup(
      Path root,
      Path repoRoot,
      Path oldTree,
      Path newTree,
      Path modelDir,
      Path hostrec,
      String oldSha,
      String newSha,
      String manifestSha) {}

  /**
   * Builds the mini repo (manifest + two digest-verified images), the artifact dir, the hostrec
   * script and the two git trees (worker source committed, staged pin file present), all under
   * {@code base/tag} so every call starts from pristine trees.
   */
  private Setup setup(String tag) throws Exception {
    Path root = base.resolve(tag);
    Path repo = root.resolve("repo");
    Path images = repo.resolve("images");
    Files.createDirectories(images);
    byte[] imageA = "image-a-bytes".getBytes(StandardCharsets.UTF_8);
    byte[] imageB = "image-b-bytes".getBytes(StandardCharsets.UTF_8);
    Files.write(images.resolve("mini-a.png"), imageA);
    Files.write(images.resolve("mini-b.png"), imageB);
    String manifest =
        """
        {
          "images": [
            {"path": "images/mini-a.png", "sha256": "%s"},
            {"path": "images/mini-b.png", "sha256": "%s"}
          ],
          "reference": {
            "cases": [
              {
                "max_new_tokens": 8,
                "image_indices": [0, 1],
                "messages": [],
                "rendered_prompt": "p",
                "unexpanded_ids": [1, 2],
                "expanded_ids": [1, 2, 3, 4],
                "target_ids": [9, 9],
                "finish_reason": "MAX_TOKENS"
              }
            ]
          }
        }
        """
            .formatted(sha256Hex(imageA), sha256Hex(imageB));
    Path manifestPath = repo.resolve("tools/tier-b/mini.json");
    Files.createDirectories(manifestPath.getParent());
    Files.writeString(manifestPath, manifest);
    Path modelDir = root.resolve("models");
    Files.createDirectories(modelDir);
    Path hostrec = root.resolve("hostrec.sh");
    Files.writeString(hostrec, "#!/bin/sh\nexit 0\n");
    Files.setPosixFilePermissions(hostrec, PosixFilePermissions.fromString("rwxr-xr-x"));
    Path oldTree = root.resolve("tree-old");
    Path newTree = root.resolve("tree-new");
    String oldSha = makeTree(oldTree, "tree-old");
    String newSha = makeTree(newTree, "tree-new");
    // Distinct content also makes the two commits distinct: identical content committed in the
    // same second would hash to the same commit SHA and mask a tree mix-up.
    assertFalse(oldSha.equals(newSha));
    return new Setup(
        root,
        repo,
        oldTree,
        newTree,
        modelDir,
        hostrec,
        oldSha,
        newSha,
        sha256Hex(manifest.getBytes(StandardCharsets.UTF_8)));
  }

  /** Creates one fake git tree with the committed worker source and staged native dir. */
  private String makeTree(Path tree, String label) throws Exception {
    Files.createDirectories(tree);
    run(tree, "git", "init");
    run(tree, "git", "config", "user.email", "test@example.com");
    run(tree, "git", "config", "user.name", "test");
    Files.createDirectories(
        tree.resolve("jmlx-benchmarks/src/main/java/se/alipsa/jmlx/benchmarks"));
    Files.writeString(tree.resolve(WORKER_SOURCE_PATH), WORKER_SOURCE);
    Files.writeString(tree.resolve("README"), label + "\n");
    run(tree, "git", "add", ".");
    run(tree, "git", "commit", "-m", "init");
    Files.createDirectories(tree.resolve("install/lib"));
    Files.createDirectories(tree.resolve("native/lib"));
    Files.writeString(tree.resolve("native/lib/mlx.metallib"), "fake metallib");
    Files.writeString(
        tree.resolve("native/lib/native-pin.properties"),
        "mlxMetalVersion=0.31.2\nmlxcCommit=fba4470\n");
    return run(tree, "git", "rev-parse", "HEAD").trim();
  }

  /** The base session spec over the setup; tests mutate a copy. */
  private ObjectNode spec(Setup setup, String outName, int pairs) {
    ObjectNode spec = MAPPER.createObjectNode();
    spec.put("session", "test-" + outName);
    spec.put("out_root", setup.root().resolve("out-" + outName).toString());
    spec.put("repo_root", setup.repoRoot().toString());
    spec.put("hostrec_script", setup.hostrec().toString());
    ObjectNode trees = spec.putObject("trees");
    ObjectNode old = trees.putObject("old");
    old.put("path", setup.oldTree().toString());
    old.put("expected_tree_sha", setup.oldSha());
    old.put("class_dir", setup.oldTree().resolve("install/lib").toString());
    old.put("native_dir", setup.oldTree().resolve("native/lib").toString());
    ObjectNode newer = trees.putObject("new");
    newer.put("path", setup.newTree().toString());
    newer.put("expected_tree_sha", setup.newSha());
    newer.put("class_dir", setup.newTree().resolve("install/lib").toString());
    newer.put("native_dir", setup.newTree().resolve("native/lib").toString());
    ObjectNode workload = spec.putArray("workloads").addObject();
    workload.put("name", "mini");
    workload.put("manifest", "tools/tier-b/mini.json");
    workload.put("manifest_sha256", setup.manifestSha());
    workload.put("model_dir", setup.modelDir().toString());
    workload.putArray("stop_token_ids").add(151645);
    spec.put("pairs", pairs);
    spec.put("warmups", 1);
    spec.put("measured", 4);
    spec.put("worker_timeout_s", 60);
    spec.put("session_time_budget_s", 600);
    spec.put("precision_env", "MLX_ENABLE_TF32");
    spec.put("precision_value", "0");
    return spec;
  }

  @Test
  void allOkSession() throws Exception {
    Setup setup = setup("all-ok");
    FakeStarter starter = new FakeStarter();
    ObjectNode spec = spec(setup, "all-ok", 2);
    int exit = new VlmFirstUseBenchmark(starter).runSession(spec);
    assertEquals(VlmFirstUseBenchmark.EXIT_OK, exit);
    assertEquals(4, starter.jobs.size());

    Path outRoot = setup.root().resolve("out-all-ok");
    JsonNode report = MAPPER.readTree(outRoot.resolve("session-report.json").toFile());
    assertEquals("jmlx-vlm-first-use-session-1", report.path("schema").asString());
    assertEquals("test-all-ok", report.path("session").asString());
    assertEquals(2, report.path("pairs").asInt());
    assertEquals(1, report.path("warmups").asInt());
    assertEquals(4, report.path("measured").asInt());
    List<String> forkOrder = new ArrayList<>();
    for (JsonNode entry : report.path("fork_order")) {
      forkOrder.add(entry.asString());
    }
    assertEquals(List.of("mini/old/p0", "mini/new/p0", "mini/new/p1", "mini/old/p1"), forkOrder);

    // Trees: recorded SHA, tracked-clean, byte-identical worker source, staged pin from disk.
    JsonNode oldTree = report.path("trees").get(0);
    JsonNode newTree = report.path("trees").get(1);
    assertEquals("old", oldTree.path("name").asString());
    assertEquals("new", newTree.path("name").asString());
    assertEquals(setup.oldSha(), oldTree.path("tree_sha").asString());
    assertEquals(setup.newSha(), newTree.path("tree_sha").asString());
    assertFalse(oldTree.path("tracked_dirty").asBoolean());
    assertFalse(newTree.path("tracked_dirty").asBoolean());
    assertEquals(WORKER_SOURCE_SHA, oldTree.path("worker_source_sha256").asString());
    assertEquals(WORKER_SOURCE_SHA, newTree.path("worker_source_sha256").asString());
    assertEquals("0.31.2", oldTree.path("native_pin").path("mlxMetalVersion").asString());
    assertEquals("fba4470", oldTree.path("native_pin").path("mlxcCommit").asString());

    // Every worker ok, with its report embedded.
    assertEquals(4, report.path("workers").size());
    for (JsonNode worker : report.path("workers")) {
      assertEquals("ok", worker.path("classification").asString());
      assertEquals(
          VlmFirstUseWorker.REPORT_SCHEMA, worker.path("report").path("schema").asString());
    }

    // Per-pair summary: warm quantiles (type-7) and the paired median ratios.
    JsonNode pairs = report.path("summary").path("pairs");
    assertEquals(2, pairs.size());
    for (JsonNode pairNode : pairs) {
      JsonNode oldWarm = pairNode.path("old").path("warm");
      // measured_ttft_s [0.5, 1.0, 1.5, 2.0]: p10 0.65, median 1.25, p90 1.85
      assertEquals(0.65, oldWarm.path("ttft_p10").asDouble(), 1e-12);
      assertEquals(1.25, oldWarm.path("ttft_median").asDouble(), 1e-12);
      assertEquals(1.85, oldWarm.path("ttft_p90").asDouble(), 1e-12);
      // measured_full_generation_s [10, 20, 30, 40]: p10 13, median 25, p90 37
      assertEquals(13.0, oldWarm.path("full_p10").asDouble(), 1e-12);
      assertEquals(25.0, oldWarm.path("full_median").asDouble(), 1e-12);
      assertEquals(37.0, oldWarm.path("full_p90").asDouble(), 1e-12);
      assertEquals(4, oldWarm.path("n").asInt());
      // new worker's fake report carries the same arrays in this test.
      JsonNode newWarm = pairNode.path("new").path("warm");
      assertEquals(
          oldWarm.path("ttft_median").asDouble(), newWarm.path("ttft_median").asDouble(), 1e-12);
      assertEquals(1.0, pairNode.path("ratio_new_over_old").path("ttft").asDouble(), 1e-12);
      assertEquals(
          1.0, pairNode.path("ratio_new_over_old").path("full_generation").asDouble(), 1e-12);
      // First-request values stay raw per worker.
      assertEquals(
          1.0, pairNode.path("old").path("first_request").path("ttft_s").asDouble(), 1e-12);
    }

    // Pooled warm over the two ok workers per tree (8 samples each).
    JsonNode pooled = report.path("summary").path("pooled_warm").path("mini");
    assertEquals(8, pooled.path("old").path("n").asInt());
    assertEquals(8, pooled.path("new").path("n").asInt());
    assertEquals(1.25, pooled.path("old").path("ttft_median").asDouble(), 1e-12);

    // The job file carries the session's declared settings, not placeholders.
    JsonNode job = MAPPER.readTree(outRoot.resolve("mini/old-p0/job.json").toFile());
    assertEquals(VlmFirstUseWorker.JOB_SCHEMA, job.path("schema").asString());
    assertEquals(1, job.path("warmups").asInt());
    assertEquals(4, job.path("measured").asInt());
    assertEquals(60, job.path("worker_timeout_s").asLong());
    assertEquals("MLX_ENABLE_TF32", job.path("precision_env").asString());
    assertEquals("0", job.path("precision_value").asString());
    assertEquals(8, job.path("max_new_tokens").asInt());
    assertEquals(1, job.path("stop_token_ids").size());
    assertEquals(151645, job.path("stop_token_ids").get(0).asInt());
    assertEquals(2, job.path("image_paths").size());
    assertEquals(2, job.path("image_sha256s").size());

    // Host records bracketed every fork; the thermal record notes are present.
    String hostCmds = Files.readString(outRoot.resolve("hostrec/host.log.cmds"));
    assertEquals(8, hostCmds.lines().count());
    assertTrue(hostCmds.contains("pre:mini:old:p0"));
    assertTrue(hostCmds.contains("post:mini:old:p0"));
    assertEquals(2, report.path("thermal").size());
  }

  @Test
  void everyFailureModeIsClassified() throws Exception {
    Setup setup = setup("failures");
    FakeStarter starter = new FakeStarter();
    starter.behaviors.put("mini/old/p0", Behavior.FAILED);
    starter.behaviors.put("mini/new/p0", Behavior.CORRECTNESS_FAILED);
    starter.behaviors.put("mini/old/p1", Behavior.BACKSTOP_TIMEOUT);
    starter.behaviors.put("mini/new/p1", Behavior.DEAD);
    starter.behaviors.put("mini/old/p2", Behavior.CRASHED);
    starter.behaviors.put("mini/new/p2", Behavior.JOB_VERIFICATION);
    starter.behaviors.put("mini/old/p3", Behavior.INVALID_REPORT);
    starter.behaviors.put("mini/new/p3", Behavior.KILLED);
    int exit = new VlmFirstUseBenchmark(starter).runSession(spec(setup, "failures", 4));
    assertEquals(VlmFirstUseBenchmark.EXIT_WORKERS_FAILED, exit);

    JsonNode report =
        MAPPER.readTree(setup.root().resolve("out-failures/session-report.json").toFile());
    Map<String, String> classification = new LinkedHashMap<>();
    for (JsonNode worker : report.path("workers")) {
      classification.put(
          worker.path("workload").asString()
              + "/"
              + worker.path("tree").asString()
              + "/p"
              + worker.path("pair_index").asInt(),
          worker.path("classification").asString());
    }
    assertEquals("failed", classification.get("mini/old/p0"));
    assertEquals("correctness_failed", classification.get("mini/new/p0"));
    assertEquals("timeout", classification.get("mini/old/p1"));
    assertEquals("dead", classification.get("mini/new/p1"));
    assertEquals("crashed", classification.get("mini/old/p2"));
    assertEquals("job-verification-failed", classification.get("mini/new/p2"));
    assertEquals("invalid-report", classification.get("mini/old/p3"));
    assertEquals("timeout", classification.get("mini/new/p3"));
  }

  @Test
  void sessionBudgetStopsForking() throws Exception {
    Setup setup = setup("budget");
    FakeStarter starter = new FakeStarter();
    starter.firstStartDelayMs = 1500;
    ObjectNode spec = spec(setup, "budget", 2);
    spec.put("session_time_budget_s", 1);
    int exit = new VlmFirstUseBenchmark(starter).runSession(spec);
    assertEquals(VlmFirstUseBenchmark.EXIT_WORKERS_FAILED, exit);
    assertEquals(1, starter.jobs.size());

    JsonNode report =
        MAPPER.readTree(setup.root().resolve("out-budget/session-report.json").toFile());
    List<String> classifications = new ArrayList<>();
    for (JsonNode worker : report.path("workers")) {
      classifications.add(
          worker.path("classification").asString()
              + (worker.path("detail").isMissingNode()
                  ? ""
                  : ":" + worker.path("detail").asString()));
    }
    assertEquals(4, classifications.size());
    assertEquals("ok", classifications.get(0));
    for (int i = 1; i < classifications.size(); i++) {
      assertTrue(
          classifications.get(i).startsWith("not-started:session time budget exceeded"),
          classifications.get(i));
    }
  }

  @Test
  void starterFailureIsNotStarted() throws Exception {
    Setup setup = setup("starter-fail");
    FakeStarter starter = new FakeStarter();
    starter.behaviors.put("mini/old/p0", Behavior.THROW);
    int exit = new VlmFirstUseBenchmark(starter).runSession(spec(setup, "starter-fail", 1));
    assertEquals(VlmFirstUseBenchmark.EXIT_WORKERS_FAILED, exit);
    JsonNode report =
        MAPPER.readTree(setup.root().resolve("out-starter-fail/session-report.json").toFile());
    assertEquals("not-started", report.path("workers").get(0).path("classification").asString());
    assertTrue(
        report
            .path("workers")
            .get(0)
            .path("detail")
            .asString()
            .contains("launcher failed to start"));
    // The session still ran the remaining fork.
    assertEquals("ok", report.path("workers").get(1).path("classification").asString());
  }

  @Test
  void specFailuresRefuseBeforeAnyFork() throws Exception {
    // A fresh setup per scenario: each refusal must be attributable to that scenario's mutation
    // alone, not to a leaked one from an earlier subcase.
    refused(
        "out-root-exists",
        (s, out) -> {
          try {
            Files.createDirectories(Path.of(s.path("out_root").asString()));
          } catch (IOException e) {
            throw new IllegalStateException(e);
          }
        });
    refused(
        "out-root-inside-repo",
        (s, out) ->
            s.put("out_root", Path.of(s.path("repo_root").asString()).resolve("out").toString()));
    refused(
        "missing-tree",
        (s, out) ->
            pathObj(s, "trees", "old").put("path", base.resolve("no-such-tree").toString()));
    refused(
        "sha-mismatch",
        (s, out) -> pathObj(s, "trees", "new").put("expected_tree_sha", "0".repeat(40)));
    refused(
        "tracked-dirty",
        (s, out) -> {
          try {
            Files.writeString(
                base.resolve(pathObj(s, "trees", "new").path("path").asString())
                    .resolve(WORKER_SOURCE_PATH),
                "tampered\n");
          } catch (IOException e) {
            throw new IllegalStateException(e);
          }
        });
    refused(
        "manifest-digest-mismatch",
        (s, out) -> workload0(s).put("manifest_sha256", "0".repeat(64)));
    refused(
        "model-dir-missing",
        (s, out) -> workload0(s).put("model_dir", base.resolve("no-model").toString()));
    refused("budget-zero", (s, out) -> s.put("session_time_budget_s", 0));
    refused(
        "missing-hostrec",
        (s, out) -> s.put("hostrec_script", base.resolve("no-hostrec.sh").toString()));
    refused("empty-stops", (s, out) -> workload0(s).putArray("stop_token_ids"));
    refused(
        "same-trees",
        (s, out) -> {
          ObjectNode old = pathObj(s, "trees", "old");
          ObjectNode newer = pathObj(s, "trees", "new");
          newer.put("path", old.path("path").asString());
          newer.put("expected_tree_sha", old.path("expected_tree_sha").asString());
          newer.put("class_dir", old.path("class_dir").asString());
          newer.put("native_dir", old.path("native_dir").asString());
        });
  }

  /** Navigates object fields of the spec and returns the target as an ObjectNode. */
  private static ObjectNode pathObj(ObjectNode root, String... keys) {
    JsonNode node = root;
    for (String key : keys) {
      node = node.path(key);
    }
    return (ObjectNode) node;
  }

  /** The first workload object of the spec. */
  private static ObjectNode workload0(ObjectNode spec) {
    return (ObjectNode) spec.path("workloads").get(0);
  }

  /** Builds a fresh setup and spec, applies the mutation, and asserts a pre-fork refusal. */
  private void refused(String scenario, BiConsumer<ObjectNode, String> mutator) throws Exception {
    Setup setup = setup("refuse-" + scenario);
    ObjectNode spec = spec(setup, scenario, 1);
    mutator.accept(spec, scenario);
    FakeStarter starter = new FakeStarter();
    assertEquals(
        VlmFirstUseBenchmark.EXIT_SPEC_FAILED, new VlmFirstUseBenchmark(starter).runSession(spec));
    // The out-root-exists scenario pre-creates its out_root by construction; every other
    // refusal must leave the session directory uncreated.
    if (!"out-root-exists".equals(scenario)) {
      assertFalse(Files.exists(Path.of(spec.path("out_root").asString())));
    }
    assertEquals(0, starter.jobs.size(), "no fork may happen after a spec failure");
  }

  @Test
  void workerSourceDriftBetweenTreesIsRefused() throws Exception {
    refused(
        "worker-drift",
        (s, out) -> {
          try {
            Files.writeString(
                base.resolve(pathObj(s, "trees", "new").path("path").asString())
                    .resolve(WORKER_SOURCE_PATH),
                WORKER_SOURCE + "// drifted\n");
          } catch (IOException e) {
            throw new IllegalStateException(e);
          }
        });
  }

  private static String run(Path workdir, String... command) throws Exception {
    ProcessBuilder builder = new ProcessBuilder(command);
    builder.directory(workdir.toFile());
    builder.redirectErrorStream(true);
    Process process = builder.start();
    String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    int exit = process.waitFor();
    if (exit != 0) {
      throw new IOException(command[0] + " exited " + exit + ": " + output);
    }
    return output;
  }

  private static String sha256Hex(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }
}
