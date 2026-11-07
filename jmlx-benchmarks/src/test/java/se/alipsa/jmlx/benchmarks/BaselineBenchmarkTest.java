package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * End-to-end launcher tests over a hermetic mini repository (no real model, no network): a temp-dir
 * checkout with a tiny frozen manifest, one artifact directory, and a fake worker starter that
 * simulates every protocol outcome. The tests are pure Java and run on any platform; the real fork
 * path is covered by the worker process tests in jmlx-models.
 */
class BaselineBenchmarkTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final String CLASSPATH = "fake-models-test-classpath";
  private static final String CLASSPATH_PROPERTY = "jmlx.baseline.models.test.classpath";
  private static final String MANIFEST =
      """
      {
        "run_policy": {
          "cache_regime": "primed-per-request",
          "worker_timeout_s": 300,
          "reference_pair_worker_timeout_s": 60,
          "first_requests": 1,
          "warmups": 0,
          "measured_requests": 2
        },
        "text_cases": [{"name": "tiny"}],
        "controlled_generation": {"prompt_lengths": [32, 512]},
        "stock_generation": {"wired_limit": "none"}
      }
      """;

  @TempDir Path base;

  @AfterEach
  void clearClasspathProperty() {
    System.clearProperty(CLASSPATH_PROPERTY);
  }

  /** Simulated worker outcome for one worker directory. */
  private enum Behavior {
    OK,
    DEAD,
    CRASHED,
    INVALID,
    FAILED,
    TIMEOUT,
    THROW
  }

  /** Records the jobs it started and leaves whatever protocol artifacts the behavior demands. */
  private static final class FakeStarter implements BaselineBenchmark.WorkerStarter {
    final List<BaselineJob> jobs = new ArrayList<>();
    final Map<String, Behavior> behaviors = new LinkedHashMap<>();

    @Override
    public int start(BaselineJob job, Path jobFile, Path stderrLog) throws Exception {
      String dir =
          BaselineJob.dirName(
              job.runtime(), job.adapter(), job.caseName(), job.promptLength(), job.mode());
      Behavior behavior = behaviors.getOrDefault(dir, Behavior.OK);
      jobs.add(job);
      if (behavior == Behavior.THROW) {
        throw new IllegalStateException("simulated fork failure");
      }
      Files.createDirectories(stderrLog.getParent());
      Files.writeString(stderrLog, "fake stderr");
      Path outDir = Path.of(job.outDir());
      switch (behavior) {
        case OK -> writeReport(job, "ok", null, true);
        case FAILED -> writeReport(job, "failed", "simulated failure", true);
        case TIMEOUT -> writeReport(job, "timeout", "simulated timeout", true);
        case INVALID -> writeReport(job, "ok", null, false);
        case DEAD -> Files.writeString(outDir.resolve("worker-claim"), dir);
        case CRASHED -> {
          // no artifacts: the launcher must classify the cell as crashed
        }
        default -> {}
      }
      return 0;
    }

    private void writeReport(BaselineJob job, String status, String failure, boolean valid)
        throws IOException {
      List<ObjectNode> requests = new ArrayList<>();
      int index = 0;
      for (int i = 0; i < job.first(); i++) {
        requests.add(requestFor(job, index++, "first", 0.02));
      }
      for (int i = 0; i < job.warmups(); i++) {
        requests.add(requestFor(job, index++, "warmup", 0.025));
      }
      for (int i = 0; i < job.measured(); i++) {
        requests.add(requestFor(job, index++, "measured", 0.03 + 0.01 * i));
      }
      ObjectNode worker =
          BaselineReportFixtures.workerBlock(
              job.runtime(),
              job.adapter(),
              job.caseName(),
              job.promptLength(),
              job.mode(),
              status,
              failure,
              job.manifestSha256(),
              requests.toArray(new ObjectNode[0]));
      ObjectNode report = BaselineReportFixtures.report(worker);
      if (!valid) {
        ObjectNode first = (ObjectNode) ((ArrayNode) worker.get("requests")).get(0);
        first.putNull("streaming_decode_tps");
        ((ArrayNode) first.get("event_s")).remove(7);
      }
      Path outDir = Path.of(job.outDir());
      Files.writeString(outDir.resolve("worker-claim"), job.caseName());
      Files.writeString(outDir.resolve("report.json"), MAPPER.writeValueAsString(report));
    }

    private static ObjectNode requestFor(BaselineJob job, int index, String kind, double ttft) {
      if ("fixed-continuation".equals(job.mode())) {
        return BaselineReportFixtures.fixedContinuationRequest(index, kind);
      }
      return BaselineReportFixtures.request(index, kind, ttft);
    }
  }

  /** One temp-dir checkout plus its artifact directory. */
  private static final class Setup {
    Path repo;
    Path manifest;
    String manifestSha;
    Path artifact;
  }

  private Setup setup() throws IOException {
    Setup setup = new Setup();
    setup.repo = base.resolve("repo");
    Path tools = setup.repo.resolve("tools/decode-baseline");
    Files.createDirectories(tools);
    Files.writeString(setup.repo.resolve("settings.gradle"), "// mini repo");
    setup.manifest = tools.resolve("workloads.json");
    Files.writeString(setup.manifest, MANIFEST);
    setup.manifestSha = BaselineBenchmark.sha256File(setup.manifest);
    setup.artifact = base.resolve("artifact-tiny");
    Files.createDirectories(setup.artifact);
    Files.writeString(setup.artifact.resolve("model.safetensors"), "tiny");
    return setup;
  }

  private static ObjectNode worker(
      String runtime, String adapter, String caseName, int length, String mode) {
    ObjectNode node = MAPPER.createObjectNode();
    node.put("runtime", runtime);
    node.put("adapter", adapter);
    node.put("case", caseName);
    node.put("prompt_length", length);
    node.put("mode", mode);
    return node;
  }

  private ObjectNode buildSpec(Setup setup, Path outRoot, String classpath, ObjectNode... workers) {
    ObjectNode spec = MAPPER.createObjectNode();
    spec.put("out_root", outRoot.toString());
    spec.put("repo_dir", setup.repo.toString());
    spec.put("precision_env", "1");
    spec.put("java_worker_classpath", classpath);
    spec.putObject("artifact_dirs").put("tiny", setup.artifact.toString());
    ArrayNode array = spec.putArray("workers");
    for (ObjectNode entry : workers) {
      array.add(entry);
    }
    return spec;
  }

  private static JsonNode session(ObjectNode spec) throws IOException {
    Path file = Path.of(spec.path("out_root").asString("")).resolve("session.json");
    return Files.isRegularFile(file) ? MAPPER.readTree(file) : null;
  }

  @Test
  void allOkSession() throws IOException {
    Setup setup = setup();
    FakeStarter starter = new FakeStarter();
    Path outRoot = base.resolve("session");
    ObjectNode spec =
        buildSpec(setup, outRoot, CLASSPATH, worker("java", "controlled", "tiny", 32, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_OK, exit);

    final String dir = "java-controlled-tiny-32-headline";
    assertEquals(1, starter.jobs.size());
    BaselineJob job = starter.jobs.get(0);
    assertEquals("controlled", job.adapter());
    assertEquals("tiny", job.caseName());
    assertEquals(32, job.promptLength());
    assertEquals("headline", job.mode());
    assertEquals(1, job.first());
    assertEquals(0, job.warmups());
    assertEquals(2, job.measured());
    assertEquals(300, job.timeoutS());
    assertEquals("1", job.precisionEnv());
    assertEquals(setup.artifact.toString(), job.artifactDir());
    assertEquals(setup.manifest.toString(), job.manifestPath());
    assertEquals(setup.manifestSha, job.manifestSha256());
    assertEquals(setup.repo.toString(), job.repoDir());
    assertEquals(outRoot.resolve(dir).toString(), job.outDir());
    assertEquals(32, job.fcOffset());
    assertEquals(32, job.fcCount());
    assertEquals("java", job.runtime());
    assertEquals(CLASSPATH, job.modelsTestClasspath());

    assertTrue(Files.isRegularFile(outRoot.resolve("session.json")));
    assertTrue(Files.isRegularFile(outRoot.resolve("spec.json")));
    assertTrue(Files.isRegularFile(outRoot.resolve("jobs").resolve(dir + ".job.json")));
    assertTrue(Files.isRegularFile(outRoot.resolve("stderr").resolve(dir + ".log")));
    assertTrue(Files.isRegularFile(outRoot.resolve(dir).resolve("report.json")));
    assertTrue(Files.isRegularFile(outRoot.resolve(dir).resolve("worker-claim")));

    JsonNode jobFile = MAPPER.readTree(outRoot.resolve("jobs").resolve(dir + ".job.json"));
    assertEquals("jmlx-decode-baseline-job-1", jobFile.path("schema").asString(""));
    assertEquals("controlled", jobFile.path("adapter").asString(""));
    assertEquals("tiny", jobFile.path("case").asString(""));
    assertEquals(32, jobFile.path("prompt_length").asInt(0));
    assertEquals("headline", jobFile.path("mode").asString(""));
    assertEquals(1, jobFile.path("requests").path("first").asInt(0));
    assertEquals(0, jobFile.path("requests").path("warmups").asInt(0));
    assertEquals(2, jobFile.path("requests").path("measured").asInt(0));
    assertEquals(300, jobFile.path("timeout_s").asInt(0));
    assertEquals("1", jobFile.path("precision_env").asString(""));
    assertEquals(setup.artifact.toString(), jobFile.path("artifact_dir").asString(""));
    assertEquals(setup.manifest.toString(), jobFile.path("manifest_path").asString(""));
    assertEquals(setup.manifestSha, jobFile.path("manifest_sha256").asString(""));
    assertEquals(setup.repo.toString(), jobFile.path("repo_dir").asString(""));
    assertEquals(job.outDir(), jobFile.path("out_dir").asString(""));
    assertFalse(jobFile.has("fixed_continuation"));
    // The launcher-only fields never leak into the frozen job-file schema.
    assertFalse(jobFile.has("runtime"));
    assertFalse(jobFile.has("models_test_classpath"));

    JsonNode root = session(spec);
    assertEquals("jmlx-decode-baseline-1", root.path("schema").asString(""));
    JsonNode sessionBlock = root.path("session");
    assertFalse(sessionBlock.path("utc_start").asString("").isEmpty());
    assertFalse(sessionBlock.path("utc_end").asString("").isEmpty());
    assertEquals("MLX_ENABLE_TF32=1", sessionBlock.path("precision_env").asString(""));
    assertEquals("primed-per-request", sessionBlock.path("cache_regime").asString(""));
    assertEquals(setup.manifestSha, sessionBlock.path("workload").asString(""));
    assertEquals(
        System.getProperty("java.version"), sessionBlock.path("java").path("version").asString(""));
    assertTrue(sessionBlock.path("python").isNull());
    assertTrue(sessionBlock.path("host").isObject());
    // The temp dir is not a git checkout; the launcher records that explicitly.
    assertEquals("", sessionBlock.path("code").path("sha").asString("x"));
    assertTrue(sessionBlock.path("code").path("dirty").asBoolean(false));
    assertEquals("", sessionBlock.path("native_pins").path("java").path("metal").asString("x"));
    assertEquals(
        "", sessionBlock.path("native_pins").path("java").path("mlxc_commit").asString("x"));
    assertTrue(sessionBlock.path("native_pins").path("python").path("mlx").isNull());
    assertEquals(1, sessionBlock.path("fork_order").size());
    assertEquals("java", sessionBlock.path("fork_order").get(0).asString(""));

    JsonNode worker0 = root.path("workers").get(0);
    assertEquals(dir, worker0.path("dir").asString(""));
    assertEquals("java", worker0.path("runtime").asString(""));
    assertEquals("controlled", worker0.path("adapter").asString(""));
    assertEquals("tiny", worker0.path("case").asString(""));
    assertEquals(32, worker0.path("prompt_length").asInt(0));
    assertEquals("headline", worker0.path("mode").asString(""));
    assertEquals("ok", worker0.path("status").asString(""));
    assertEquals(setup.manifestSha, worker0.path("identity").path("manifest_sha256").asString(""));
    assertEquals(3, worker0.path("requests").size());

    JsonNode summary = root.path("summary");
    assertEquals("linear-interpolation-type-7", summary.path("quantile_convention").asString(""));
    assertEquals(0, summary.path("pairs").size());
    assertEquals(0, summary.path("exclusions").size());
    // Two measured ttft samples 0.03 and 0.04; all four headline metrics present
    // (step_region_s is null outside fixed-continuation). The cell key is mode-scoped:
    // headline and fixed-continuation never pool, and reference-pair samples (observed
    // regime) are excluded from the headline quantiles entirely.
    JsonNode ttft =
        summary.path("per_runtime").path("java").path("headline/tiny/32").path("ttft_s");
    assertEquals(2, ttft.path("n").asInt(0));
    assertEquals(0.031, ttft.path("p10").asDouble(0), 1e-12);
    assertEquals(0.035, ttft.path("median").asDouble(0), 1e-12);
    assertEquals(0.039, ttft.path("p90").asDouble(0), 1e-12);
    assertEquals(4, summary.path("per_runtime").path("java").path("headline/tiny/32").size());
    // The single first request is a separate distribution, mode-scoped.
    JsonNode first = summary.path("first_request").path("java").path("headline").path("ttft_s");
    assertEquals(1, first.path("n").asInt(0));
    assertEquals(0.02, first.path("median").asDouble(0), 1e-12);
    assertEquals(4, summary.path("first_request").path("java").path("headline").size());
  }

  private record ModeRun(JsonNode jobFile, JsonNode session) {}

  private ModeRun runSingleMode(Setup setup, String mode) throws IOException {
    FakeStarter starter = new FakeStarter();
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve("session-" + mode),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, mode));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_OK, exit);
    String dir = BaselineJob.dirName("java", "controlled", "tiny", 32, mode);
    JsonNode jobFile =
        MAPPER.readTree(base.resolve("session-" + mode).resolve("jobs").resolve(dir + ".job.json"));
    return new ModeRun(jobFile, session(spec));
  }

  @Test
  void fixedContinuationJob() throws IOException {
    ModeRun run = runSingleMode(setup(), "fixed-continuation");
    JsonNode jobFile = run.jobFile();
    assertEquals(300, jobFile.path("timeout_s").asInt(0));
    assertEquals(1, jobFile.path("requests").path("first").asInt(0));
    assertEquals(0, jobFile.path("requests").path("warmups").asInt(0));
    assertEquals(2, jobFile.path("requests").path("measured").asInt(0));
    assertEquals(32, jobFile.path("fixed_continuation").path("offset").asInt(0));
    assertEquals(32, jobFile.path("fixed_continuation").path("count").asInt(0));
    // The FC samples live in their own mode cell: only its two measured metrics, no ttft.
    JsonNode cell =
        run.session()
            .path("summary")
            .path("per_runtime")
            .path("java")
            .path("fixed-continuation/tiny/32");
    assertEquals(2, cell.path("prefill_s").path("n").asInt(0));
    assertEquals(0.02, cell.path("prefill_s").path("median").asDouble(0), 1e-12);
    assertEquals(2, cell.path("step_region_s").path("n").asInt(0));
    assertEquals(0.32, cell.path("step_region_s").path("median").asDouble(0), 1e-12);
    assertFalse(cell.has("ttft_s"));
    assertEquals(2, cell.size());
  }

  @Test
  void referencePairJob() throws IOException {
    ModeRun run = runSingleMode(setup(), "reference-pair");
    JsonNode jobFile = run.jobFile();
    assertEquals(60, jobFile.path("timeout_s").asInt(0));
    assertEquals(0, jobFile.path("requests").path("first").asInt(0));
    assertEquals(0, jobFile.path("requests").path("warmups").asInt(0));
    assertEquals(1, jobFile.path("requests").path("measured").asInt(0));
    assertFalse(jobFile.has("fixed_continuation"));
    // Observed-regime samples never enter the primed headline quantiles (WP3 pairs).
    assertEquals(0, run.session().path("summary").path("per_runtime").size());
    assertEquals(0, run.session().path("summary").path("first_request").size());
    assertEquals(0, run.session().path("summary").path("exclusions").size());
  }

  private void assertRefused(ObjectNode specNode, FakeStarter starter) {
    int exit = new BaselineBenchmark(starter).runSession(specNode);
    assertEquals(BaselineBenchmark.EXIT_SPEC_FAILED, exit);
    assertTrue(starter.jobs.isEmpty(), "spec validation must fork nothing");
    assertFalse(
        Files.exists(Path.of(specNode.path("out_root").asString("")).resolve("session.json")));
  }

  @Test
  void refusedMissingOutRoot() throws IOException {
    FakeStarter starter = new FakeStarter();
    ObjectNode spec =
        buildSpec(
            setup(),
            base.resolve("session"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    spec.put("out_root", "");
    assertRefused(spec, starter);
  }

  @Test
  void refusedOutRootInsideRepo() throws IOException {
    FakeStarter starter = new FakeStarter();
    Setup setup = setup();
    assertRefused(
        buildSpec(
            setup,
            setup.repo.resolve("run"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline")),
        starter);
  }

  @Test
  void refusedOutRootExists() throws IOException {
    FakeStarter starter = new FakeStarter();
    Setup setup = setup();
    Path outRoot = base.resolve("session");
    Files.createDirectories(outRoot);
    ObjectNode spec =
        buildSpec(setup, outRoot, CLASSPATH, worker("java", "controlled", "tiny", 32, "headline"));
    assertEquals(
        BaselineBenchmark.EXIT_SPEC_FAILED, new BaselineBenchmark(starter).runSession(spec));
    assertTrue(starter.jobs.isEmpty());
    assertFalse(Files.exists(outRoot.resolve("session.json")));
  }

  @Test
  void refusedRepoWithoutManifest() throws IOException {
    FakeStarter starter = new FakeStarter();
    Setup setup = setup();
    Files.delete(setup.manifest);
    assertRefused(
        buildSpec(
            setup,
            base.resolve("session"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline")),
        starter);
  }

  @Test
  void refusedUnknownCase() throws IOException {
    FakeStarter starter = new FakeStarter();
    assertRefused(
        buildSpec(
            setup(),
            base.resolve("session"),
            CLASSPATH,
            worker("java", "controlled", "nope", 32, "headline")),
        starter);
  }

  @Test
  void refusedNonFrozenLength() throws IOException {
    FakeStarter starter = new FakeStarter();
    assertRefused(
        buildSpec(
            setup(),
            base.resolve("session"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 64, "headline")),
        starter);
  }

  @Test
  void refusedDuplicateCell() throws IOException {
    FakeStarter starter = new FakeStarter();
    assertRefused(
        buildSpec(
            setup(),
            base.resolve("session"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"),
            worker("java", "controlled", "tiny", 32, "headline")),
        starter);
  }

  @Test
  void refusedMissingArtifactDir() throws IOException {
    FakeStarter starter = new FakeStarter();
    Setup setup = setup();
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve("session"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    ((ObjectNode) spec.get("artifact_dirs")).put("tiny", base.resolve("missing").toString());
    assertRefused(spec, starter);
  }

  @Test
  void refusedModelsTestAliasWithoutProperty() throws IOException {
    FakeStarter starter = new FakeStarter();
    assertRefused(
        buildSpec(
            setup(),
            base.resolve("session"),
            "@models-test",
            worker("java", "controlled", "tiny", 32, "headline")),
        starter);
  }

  @Test
  void refusedBadPrecision() throws IOException {
    FakeStarter starter = new FakeStarter();
    ObjectNode spec =
        buildSpec(
            setup(),
            base.resolve("session"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    spec.put("precision_env", "2");
    assertRefused(spec, starter);
  }

  @Test
  void refusedEmptyWorkers() throws IOException {
    FakeStarter starter = new FakeStarter();
    assertRefused(buildSpec(setup(), base.resolve("session"), CLASSPATH), starter);
  }

  @Test
  void refusedBadRuntime() throws IOException {
    FakeStarter starter = new FakeStarter();
    assertRefused(
        buildSpec(
            setup(),
            base.resolve("session"),
            CLASSPATH,
            worker("js", "controlled", "tiny", 32, "headline")),
        starter);
  }

  @Test
  void refusedNotJsonSpec() throws IOException {
    Path specFile = base.resolve("spec.json");
    Files.writeString(specFile, "not json");
    assertEquals(
        BaselineBenchmark.EXIT_SPEC_FAILED, new BaselineBenchmark(new FakeStarter()).run(specFile));
  }

  @Test
  void refusedMissingSpecFile() {
    assertEquals(
        BaselineBenchmark.EXIT_SPEC_FAILED,
        new BaselineBenchmark(new FakeStarter()).run(base.resolve("absent.json")));
    assertEquals(
        BaselineBenchmark.EXIT_SPEC_FAILED, new BaselineBenchmark(new FakeStarter()).run(null));
  }

  private JsonNode runOutcome(Setup setup, Behavior behavior) throws IOException {
    FakeStarter starter = new FakeStarter();
    String dir = "java-controlled-tiny-32-headline";
    starter.behaviors.put(dir, behavior);
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve("session-outcome"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_WORKERS_FAILED, exit);
    JsonNode root = session(spec);
    // The session report is written even when every worker fails.
    assertTrue(root != null);
    assertEquals(BaselineReportValidator.SCHEMA, root.path("schema").asString(""));
    return root;
  }

  @Test
  void deadWorker() throws IOException {
    JsonNode root = runOutcome(setup(), Behavior.DEAD);
    JsonNode worker0 = root.path("workers").get(0);
    assertEquals("dead", worker0.path("outcome").asString(""));
    assertFalse(worker0.has("report"));
    assertFalse(worker0.has("status"));
    assertEquals("java-controlled-tiny-32-headline", worker0.path("dir").asString(""));
    JsonNode exclusion = root.path("summary").path("exclusions").get(0);
    assertEquals("dead", exclusion.path("reason").asString(""));
    assertFalse(exclusion.has("violations"));
    assertEquals(0, root.path("summary").path("per_runtime").size());
  }

  @Test
  void crashedWorker() throws IOException {
    JsonNode root = runOutcome(setup(), Behavior.CRASHED);
    assertEquals("crashed", root.path("workers").get(0).path("outcome").asString(""));
  }

  @Test
  void invalidReportWorker() throws IOException {
    JsonNode root = runOutcome(setup(), Behavior.INVALID);
    JsonNode worker0 = root.path("workers").get(0);
    assertEquals("invalid-report", worker0.path("outcome").asString(""));
    assertTrue(worker0.has("report"));
    JsonNode exclusion = root.path("summary").path("exclusions").get(0);
    assertEquals("invalid-report", exclusion.path("reason").asString(""));
    assertTrue(
        exclusion.path("violations").toString().contains("event_s must have emitted entries"));
  }

  @Test
  void failedWorker() throws IOException {
    JsonNode root = runOutcome(setup(), Behavior.FAILED);
    JsonNode worker0 = root.path("workers").get(0);
    assertEquals("failed", worker0.path("status").asString(""));
    assertEquals("simulated failure", worker0.path("failure_reason").asString(""));
    assertEquals("java-controlled-tiny-32-headline", worker0.path("dir").asString(""));
    assertTrue(worker0.path("requests").isArray());
    JsonNode exclusion = root.path("summary").path("exclusions").get(0);
    assertEquals("failed", exclusion.path("reason").asString(""));
    assertFalse(exclusion.has("violations"));
    // Non-ok samples never enter the summary.
    assertEquals(0, root.path("summary").path("per_runtime").size());
    assertEquals(0, root.path("summary").path("first_request").size());
  }

  @Test
  void timeoutWorker() throws IOException {
    JsonNode root = runOutcome(setup(), Behavior.TIMEOUT);
    assertEquals("timeout", root.path("workers").get(0).path("status").asString(""));
    assertEquals(
        "simulated timeout", root.path("workers").get(0).path("failure_reason").asString(""));
    assertEquals(
        "timeout", root.path("summary").path("exclusions").get(0).path("reason").asString(""));
  }

  @Test
  void starterFailureStopsSession() throws IOException {
    FakeStarter starter = new FakeStarter();
    final String dir0 = "java-controlled-tiny-32-headline";
    final String dir1 = "java-controlled-tiny-512-headline";
    starter.behaviors.put(dir0, Behavior.THROW);
    ObjectNode spec =
        buildSpec(
            setup(),
            base.resolve("session-stop"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"),
            worker("java", "controlled", "tiny", 512, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_WORKERS_FAILED, exit);
    // The first worker recorded a job but no fork happened; the second was never set up.
    assertEquals(1, starter.jobs.size());
    JsonNode root = session(spec);
    assertEquals("not-started", root.path("workers").get(0).path("outcome").asString(""));
    assertEquals("not-started", root.path("workers").get(1).path("outcome").asString(""));
    assertEquals(dir1, root.path("workers").get(1).path("dir").asString(""));
    assertEquals(0, root.path("session").path("fork_order").size());
    assertEquals(2, root.path("summary").path("exclusions").size());
    assertTrue(
        root.path("summary")
            .path("exclusions")
            .get(0)
            .path("violations")
            .toString()
            .contains("launcher failed to start the worker"));
  }

  @Test
  void modelsTestAliasResolves() throws IOException {
    Setup setup = setup();
    FakeStarter starter = new FakeStarter();
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve("session-alias"),
            "@models-test",
            worker("java", "controlled", "tiny", 32, "headline"));
    System.setProperty(CLASSPATH_PROPERTY, CLASSPATH);
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_OK, exit);
    assertEquals(CLASSPATH, starter.jobs.get(0).modelsTestClasspath());
  }

  @Test
  void workerCommands() {
    BaselineJob javaJob =
        new BaselineJob(
            "controlled",
            "tiny",
            32,
            "headline",
            1,
            0,
            2,
            300,
            "1",
            "/artifact",
            "/repo/tools/decode-baseline/workloads.json",
            "sha",
            "/repo",
            "/out",
            32,
            32,
            "java",
            "cp1:cp2");
    Path jobFile = Path.of("/session/jobs/job.job.json");
    List<String> command = javaJob.command(jobFile);
    assertEquals(
        Path.of(System.getProperty("java.home"), "bin", "java").toString(), command.get(0));
    assertTrue(command.contains("--enable-native-access=ALL-UNNAMED"));
    assertTrue(command.contains("-Djmlx.library.path=/repo/native/install/lib"));
    assertEquals("cp1:cp2", command.get(command.indexOf("-cp") + 1));
    assertEquals("se.alipsa.jmlx.models.BaselineWorker", command.get(command.size() - 2));
    assertEquals(jobFile.toString(), command.get(command.size() - 1));

    BaselineJob pythonJob =
        new BaselineJob(
            "stock",
            "tiny",
            32,
            "headline",
            1,
            0,
            2,
            300,
            "0",
            "/artifact",
            "/repo/tools/decode-baseline/workloads.json",
            "sha",
            "/repo",
            "/out",
            32,
            32,
            "python",
            "");
    List<String> python = pythonJob.command(jobFile);
    assertEquals("/repo/tools/decode-baseline/.venv/bin/python", python.get(0));
    assertEquals("-I", python.get(1));
    assertEquals("/repo/tools/decode-baseline/worker.py", python.get(2));
    assertEquals(jobFile.toString(), python.get(3));
  }
}
