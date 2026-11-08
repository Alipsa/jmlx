package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
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
  private static final String AGENT_JAR_PROPERTY = "jmlx.baseline.binding.agent.jar";
  private static final String MANIFEST =
      """
      {
        "run_policy": {
          "cache_regime": "primed-per-request",
          "worker_timeout_s": 300,
          "reference_pair_worker_timeout_s": 60,
          "cell_time_budget_s": 3600,
          "session_time_budget_s": 7200,
          "first_requests": 1,
          "warmups": 0,
          "measured_requests": 2,
          "emitted_tokens": 8
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
    System.clearProperty(AGENT_JAR_PROPERTY);
    // The launcher's digest pin is process-wide state; every test that builds a mini repository
    // re-pins it in setup(), and each test must leave the production pin in place.
    BaselineBenchmark.setPinnedManifestSha256ForTest(BaselineBenchmark.PINNED_MANIFEST_SHA256);
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
    final Map<String, Long> waitLimits = new LinkedHashMap<>();

    /** Delays each simulated fork; the session-budget test uses it to burn the budget. */
    long startDelayMs = 0L;

    /** Tamper hook: run on the worker block before the report is written (null normally). */
    Consumer<ObjectNode> reportMutator = null;

    /** When set, the starter returns WORKER_KILLED and leaves no report for that cell. */
    String killedCell = null;

    /** When set, each simulated fork populates the session's isolated native-cache directory. */
    boolean createNativeCache = false;

    @Override
    public int start(BaselineJob job, Path jobFile, Path stderrLog, long waitLimitSeconds)
        throws Exception {
      String dir =
          BaselineJob.dirName(
              job.runtime(),
              job.adapter(),
              job.caseName(),
              job.promptLength(),
              job.mode(),
              job.instrumentation());
      Behavior behavior = behaviors.getOrDefault(dir, Behavior.OK);
      jobs.add(job);
      waitLimits.put(dir, waitLimitSeconds);
      if (behavior == Behavior.THROW) {
        throw new IllegalStateException("simulated fork failure");
      }
      if (startDelayMs > 0) {
        Thread.sleep(startDelayMs);
      }
      Files.createDirectories(stderrLog.getParent());
      Files.writeString(stderrLog, "fake stderr");
      Path outDir = Path.of(job.outDir());
      if (createNativeCache) {
        // Simulate the extraction fallback having populated the session's isolated cache.
        Path cache = outDir.toAbsolutePath().normalize().getParent().resolve("native-cache");
        Files.createDirectories(cache);
        Files.writeString(cache.resolve("staged.txt"), "cached native files");
      }
      if (dir.equals(killedCell)) {
        // A wedged worker the launcher killed after the wait limit: whatever artifacts it had
        // (none here, or a claim if the behavior says so) are all the launcher classifies from.
        if (behavior == Behavior.DEAD) {
          Files.writeString(outDir.resolve("worker-claim"), dir);
        }
        return BaselineBenchmark.WORKER_KILLED;
      }
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
      if (!"none".equals(job.instrumentation())) {
        BaselineReportFixtures.instrumentWorker(worker, job.instrumentation());
        for (ObjectNode request : requests) {
          BaselineReportFixtures.instrument(request, job.instrumentation());
        }
      }
      if (!valid) {
        ObjectNode first = (ObjectNode) ((ArrayNode) worker.get("requests")).get(0);
        first.putNull("streaming_decode_tps");
        ((ArrayNode) first.get("event_s")).remove(7);
      }
      if (reportMutator != null) {
        reportMutator.accept(worker);
      }
      Path outDir = Path.of(job.outDir());
      Files.writeString(outDir.resolve("worker-claim"), job.caseName());
      // The report wraps the (already tampered) worker node by reference.
      ObjectNode report = BaselineReportFixtures.report(worker);
      Files.writeString(outDir.resolve("report.json"), MAPPER.writeValueAsString(report));
    }

    private static ObjectNode requestFor(BaselineJob job, int index, String kind, double ttft) {
      if ("fixed-continuation".equals(job.mode())) {
        return BaselineReportFixtures.fixedContinuationRequest(index, kind);
      }
      if ("python".equals(job.runtime()) && "stock".equals(job.adapter())) {
        // Python stock drains the one queued lookahead forward (the frozen worker).
        return BaselineReportFixtures.stockRequest(index, kind, ttft, 1);
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
    return setup(MANIFEST);
  }

  /** The mini checkout with a manifest variant (the run_policy and digest-pin tests). */
  private Setup setup(String manifest) throws IOException {
    Setup setup = new Setup();
    setup.repo = base.resolve("repo");
    Path tools = setup.repo.resolve("tools/decode-baseline");
    Files.createDirectories(tools);
    Files.writeString(setup.repo.resolve("settings.gradle"), "// mini repo");
    setup.manifest = tools.resolve("workloads.json");
    Files.writeString(setup.manifest, manifest);
    setup.manifestSha = BaselineBenchmark.sha256File(setup.manifest);
    // The launcher's digest pin is process-wide; every mini repository re-pins it to its own
    // manifest, and the @AfterEach restores the production pin.
    BaselineBenchmark.setPinnedManifestSha256ForTest(setup.manifestSha);
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
    // The launcher-only fields never leak into the frozen job-file schema; the headline
    // (uninstrumented) job file stays byte-identical to the WP1 shape.
    assertFalse(jobFile.has("runtime"));
    assertFalse(jobFile.has("models_test_classpath"));
    assertFalse(jobFile.has("instrumentation"));

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
    // fork_order carries the full cell identifiers in fork order.
    assertEquals(
        "java-controlled-tiny-32-headline", sessionBlock.path("fork_order").get(0).asString(""));
    assertEquals("spec-order", sessionBlock.path("orchestration").asString(""));
    assertEquals(
        outRoot.resolve("native-cache").toString(), sessionBlock.path("native_cache").asString(""));

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
    // (step_region_s is null outside fixed-continuation). The cell key is adapter- and
    // mode-scoped: controlled and stock never pool, headline and fixed-continuation never
    // pool, and reference-pair samples (observed regime) are excluded from the headline
    // quantiles entirely.
    JsonNode ttft =
        summary.path("per_runtime").path("java").path("controlled/headline/tiny/32").path("ttft_s");
    assertEquals(2, ttft.path("n").asInt(0));
    assertEquals(0.031, ttft.path("p10").asDouble(0), 1e-12);
    assertEquals(0.035, ttft.path("median").asDouble(0), 1e-12);
    assertEquals(0.039, ttft.path("p90").asDouble(0), 1e-12);
    assertEquals(
        4, summary.path("per_runtime").path("java").path("controlled/headline/tiny/32").size());
    // The single first request is a separate distribution, adapter- and mode-scoped per cell.
    JsonNode first =
        summary
            .path("first_request")
            .path("java")
            .path("controlled/headline/tiny/32")
            .path("ttft_s");
    assertEquals(1, first.path("n").asInt(0));
    assertEquals(0.02, first.path("median").asDouble(0), 1e-12);
    assertEquals(
        4, summary.path("first_request").path("java").path("controlled/headline/tiny/32").size());
    // The launcher gave the fork min(timeout_s + 90 s backstop, cell_time_budget_s).
    assertEquals(390L, starter.waitLimits.get(dir));
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
    String dir = BaselineJob.dirName("java", "controlled", "tiny", 32, mode, "none");
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
    // The FC samples live in their own adapter-/mode-scoped cell: prefill and step-region over the
    // measured requests, plus step_s — the contract's decode comparison — pooled per step over
    // every measured request's 32 frozen positions (32 x 2 = 64 samples here), no ttft.
    JsonNode cell =
        run.session()
            .path("summary")
            .path("per_runtime")
            .path("java")
            .path("controlled/fixed-continuation/tiny/32");
    assertEquals(2, cell.path("prefill_s").path("n").asInt(0));
    assertEquals(0.02, cell.path("prefill_s").path("median").asDouble(0), 1e-12);
    assertEquals(2, cell.path("step_region_s").path("n").asInt(0));
    assertEquals(0.32, cell.path("step_region_s").path("median").asDouble(0), 1e-12);
    assertEquals(64, cell.path("step_s").path("n").asInt(0));
    assertEquals(0.01, cell.path("step_s").path("median").asDouble(0), 1e-12);
    assertFalse(cell.has("ttft_s"));
    assertEquals(3, cell.size());
    // The FC first request contributes only its request-level metrics to first_request
    // (prefill and step region); step_s is a measured-only metric — the decode comparison
    // pools the measured requests' steps, never the first request's.
    JsonNode firstCell =
        run.session()
            .path("summary")
            .path("first_request")
            .path("java")
            .path("controlled/fixed-continuation/tiny/32");
    assertEquals(1, firstCell.path("prefill_s").path("n").asInt(0));
    assertEquals(1, firstCell.path("step_region_s").path("n").asInt(0));
    assertFalse(firstCell.has("step_s"));
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

  /**
   * Runs one headline cell with a tampered ok report and asserts the launcher classified it
   * invalid-report with the job-mismatch violation, not silently counted it.
   */
  private JsonNode invalidReportSession(Setup setup, String outName, Consumer<ObjectNode> mutator)
      throws IOException {
    FakeStarter starter = new FakeStarter();
    starter.reportMutator = mutator;
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve(outName),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_WORKERS_FAILED, exit);
    return session(spec);
  }

  @Test
  void reportNotMatchingJobIsInvalid() throws IOException {
    // The report describes a different case than the job the launcher launched for it.
    JsonNode root =
        invalidReportSession(
            setup(), "session-mismatch", workerBlock -> workerBlock.put("case", "other"));
    assertEquals("invalid-report", root.path("workers").get(0).path("outcome").asString(""));
    assertTrue(
        root.path("summary")
            .path("exclusions")
            .get(0)
            .path("violations")
            .toString()
            .contains("differs from the launched job's"));
  }

  @Test
  void okReportWithoutRequestsIsInvalid() throws IOException {
    // An ok report with no requests is rejected: it cannot have done the work the job ordered.
    JsonNode root =
        invalidReportSession(
            setup(), "session-no-requests", workerBlock -> workerBlock.putArray("requests"));
    assertEquals("invalid-report", root.path("workers").get(0).path("outcome").asString(""));
    assertTrue(
        root.path("summary")
            .path("exclusions")
            .get(0)
            .path("violations")
            .toString()
            .contains("must carry exactly the job's 3 requests"));
  }

  @Test
  void failedReportShortPrefixIsAccepted() throws IOException {
    // A failed report may carry a shorter complete prefix of the job's request mix (the run
    // died mid-way); it is still classified failed, not invalid-report.
    FakeStarter starter = new FakeStarter();
    starter.behaviors.put("java-controlled-tiny-32-headline", Behavior.FAILED);
    starter.reportMutator =
        workerBlock -> {
          ArrayNode requests = (ArrayNode) workerBlock.get("requests");
          requests.remove(requests.size() - 1);
        };
    ObjectNode spec =
        buildSpec(
            setup(),
            base.resolve("session-short-prefix"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_WORKERS_FAILED, exit);
    JsonNode root = session(spec);
    assertEquals("failed", root.path("workers").get(0).path("status").asString(""));
    assertEquals(
        "failed", root.path("summary").path("exclusions").get(0).path("reason").asString(""));
    assertFalse(
        root.path("summary").path("exclusions").get(0).has("violations"),
        "a short prefix is not a report violation");
  }

  @Test
  void failedReportWithExtraRequestsIsInvalid() throws IOException {
    FakeStarter starter = new FakeStarter();
    starter.behaviors.put("java-controlled-tiny-32-headline", Behavior.FAILED);
    starter.reportMutator =
        workerBlock -> {
          ArrayNode requests = (ArrayNode) workerBlock.get("requests");
          requests.add(BaselineReportFixtures.request(requests.size(), "measured", 0.06));
        };
    ObjectNode spec =
        buildSpec(
            setup(),
            base.resolve("session-extra-requests"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_WORKERS_FAILED, exit);
    JsonNode root = session(spec);
    assertEquals("invalid-report", root.path("workers").get(0).path("outcome").asString(""));
    assertTrue(
        root.path("summary")
            .path("exclusions")
            .get(0)
            .path("violations")
            .toString()
            .contains("more than the job's 3"));
  }

  @Test
  void killedWorkerIsCrashedWithKillNote() throws IOException {
    // The worker wedged: the launcher killed it after the wait limit and classified the cell
    // from the artifacts it left (none here: crashed), recording the kill in the exclusion.
    FakeStarter starter = new FakeStarter();
    starter.killedCell = "java-controlled-tiny-32-headline";
    ObjectNode spec =
        buildSpec(
            setup(),
            base.resolve("session-killed"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_WORKERS_FAILED, exit);
    JsonNode root = session(spec);
    assertEquals("crashed", root.path("workers").get(0).path("outcome").asString(""));
    // The recorded wait limit is min(timeout_s + 90 s backstop, cell_time_budget_s) = 390.
    assertTrue(
        root.path("summary")
            .path("exclusions")
            .get(0)
            .path("violations")
            .toString()
            .contains("launcher killed the worker after the wait limit (390 s)"));
  }

  @Test
  void killedWorkerWithClaimIsDead() throws IOException {
    FakeStarter starter = new FakeStarter();
    starter.behaviors.put("java-controlled-tiny-32-headline", Behavior.DEAD);
    starter.killedCell = "java-controlled-tiny-32-headline";
    ObjectNode spec =
        buildSpec(
            setup(),
            base.resolve("session-killed-dead"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_WORKERS_FAILED, exit);
    JsonNode root = session(spec);
    assertEquals("dead", root.path("workers").get(0).path("outcome").asString(""));
    assertTrue(
        root.path("summary")
            .path("exclusions")
            .get(0)
            .path("violations")
            .toString()
            .contains("launcher killed the worker after the wait limit"),
        "the kill note is recorded even when the claim already marks the cell dead");
  }

  @Test
  void sessionBudgetStopsForking() throws IOException {
    // A one-second session budget with a fork that takes longer: the second worker is never
    // set up and is recorded not-started with the budget reason, the first ran to completion.
    Setup setup =
        setup(MANIFEST.replace("\"session_time_budget_s\": 7200", "\"session_time_budget_s\": 1"));
    FakeStarter starter = new FakeStarter();
    starter.startDelayMs = 1200L;
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve("session-budget"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"),
            worker("java", "controlled", "tiny", 512, "headline"));
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_WORKERS_FAILED, exit);
    JsonNode root = session(spec);
    assertEquals("ok", root.path("workers").get(0).path("status").asString(""));
    assertEquals("not-started", root.path("workers").get(1).path("outcome").asString(""));
    assertEquals(1, root.path("session").path("fork_order").size());
    assertTrue(
        root.path("summary")
            .path("exclusions")
            .get(0)
            .path("violations")
            .toString()
            .contains("session time budget exceeded"));
  }

  @Test
  void runPolicyMissingFieldIsRefused() throws IOException {
    Setup setup = setup(MANIFEST.replace("\"emitted_tokens\"", "\"emitted_tokens_absent\""));
    assertRefused(
        buildSpec(
            setup,
            base.resolve("session-policy-missing"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline")),
        new FakeStarter());
  }

  @Test
  void runPolicyNonpositiveTimeoutIsRefused() throws IOException {
    Setup setup = setup(MANIFEST.replace("\"worker_timeout_s\": 300", "\"worker_timeout_s\": 0"));
    assertRefused(
        buildSpec(
            setup,
            base.resolve("session-policy-zero"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline")),
        new FakeStarter());
  }

  @Test
  void runPolicyNonIntegerIsRefused() throws IOException {
    Setup setup = setup(MANIFEST.replace("\"warmups\": 0", "\"warmups\": \"0\""));
    assertRefused(
        buildSpec(
            setup,
            base.resolve("session-policy-string"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline")),
        new FakeStarter());
  }

  @Test
  void digestPinMismatchIsRefused() throws IOException {
    Setup setup = setup();
    // A hand-edited manifest (the seam now names a different digest) is a spec failure, even
    // though its self-consistent digest would satisfy the workers' re-hash.
    BaselineBenchmark.setPinnedManifestSha256ForTest("0".repeat(64));
    assertRefused(
        buildSpec(
            setup,
            base.resolve("session-digest-pin"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline")),
        new FakeStarter());
  }

  @Test
  void pinnedDigestMatchesFrozenManifest() throws IOException, URISyntaxException {
    // Backstop (parity with the root build's verifyDecodeBaselineManifest): the launcher's
    // pinned digest must be this checkout's frozen manifest digest, so a manifest re-pin that
    // forgets the launcher fails the build instead of silently accepting edited manifests.
    Path here =
        Path.of(
            Objects.requireNonNull(
                    BaselineBenchmarkTest.class.getProtectionDomain().getCodeSource().getLocation())
                .toURI());
    Path root = here;
    while (root != null
        && !Files.isRegularFile(root.resolve("tools/decode-baseline/workloads.json"))) {
      root = root.getParent();
    }
    assertNotNull(root, "could not find the repository root from the test class location");
    assertEquals(
        BaselineBenchmark.PINNED_MANIFEST_SHA256,
        BaselineBenchmark.sha256File(root.resolve("tools/decode-baseline/workloads.json")));
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
            "binding-count",
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
            "cp1:cp2",
            "/agent/binding-agent.jar",
            "/session/native-cache");
    Path jobFile = Path.of("/session/jobs/job.job.json");
    List<String> command = javaJob.command(jobFile);
    assertEquals(
        Path.of(System.getProperty("java.home"), "bin", "java").toString(), command.get(0));
    assertTrue(command.contains("--enable-native-access=ALL-UNNAMED"));
    assertTrue(command.contains("-Djmlx.library.path=/repo/native/install/lib"));
    // The session's isolated extraction cache and the binding-count agent are both on the
    // forked JVM's command line.
    assertTrue(command.contains("-Djmlx.native.cache.path=/session/native-cache"));
    assertTrue(command.contains("-javaagent:/agent/binding-agent.jar"));
    assertEquals("cp1:cp2", command.get(command.indexOf("-cp") + 1));
    assertEquals("se.alipsa.jmlx.models.BaselineWorker", command.get(command.size() - 2));
    assertEquals(jobFile.toString(), command.get(command.size() - 1));

    BaselineJob uninstrumentedJob =
        new BaselineJob(
            "controlled",
            "tiny",
            32,
            "headline",
            "none",
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
            "cp1:cp2",
            "",
            "/session/native-cache");
    List<String> plain = uninstrumentedJob.command(jobFile);
    assertTrue(plain.contains("-Djmlx.native.cache.path=/session/native-cache"));
    assertTrue(plain.stream().noneMatch(argument -> argument.startsWith("-javaagent:")));

    BaselineJob pythonJob =
        new BaselineJob(
            "stock",
            "tiny",
            32,
            "headline",
            "none",
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
            "",
            "",
            "/session/native-cache");
    List<String> python = pythonJob.command(jobFile);
    assertEquals("/repo/tools/decode-baseline/.venv/bin/python", python.get(0));
    assertEquals("-I", python.get(1));
    assertEquals("/repo/tools/decode-baseline/worker.py", python.get(2));
    assertEquals(jobFile.toString(), python.get(3));
  }

  @Test
  void alternatingOrchestrationRefusesConsecutiveSameRuntime() throws IOException {
    Setup setup = setup();
    FakeStarter starter = new FakeStarter();
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve("session-alt-bad"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"),
            worker("java", "controlled", "tiny", 512, "headline"));
    spec.put("orchestration", "alternating");
    assertRefused(spec, starter);
  }

  @Test
  void alternatingOrchestrationRefusesUnknownValue() throws IOException {
    Setup setup = setup();
    FakeStarter starter = new FakeStarter();
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve("session-alt-unknown"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    spec.put("orchestration", "round-robin");
    assertRefused(spec, starter);
  }

  @Test
  void alternatingOrchestrationSession() throws IOException {
    // A valid alternating session (one java, one python worker): the session block records the
    // orchestration and the full-cell fork order, and the python worker's identity fills the
    // session's python block.
    Setup setup = setup();
    Path venvPython = setup.repo.resolve("tools/decode-baseline/.venv/bin/python");
    Files.createDirectories(venvPython.getParent());
    Files.writeString(venvPython, "# fake pinned venv python");
    FakeStarter starter = new FakeStarter();
    ObjectNode spec =
        buildSpec(
            setup,
            base.resolve("session-alt"),
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"),
            worker("python", "stock", "tiny", 32, "headline"));
    spec.put("orchestration", "alternating");
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_OK, exit);
    JsonNode sessionBlock = session(spec).path("session");
    assertEquals("alternating", sessionBlock.path("orchestration").asString(""));
    JsonNode forkOrder = sessionBlock.path("fork_order");
    assertEquals(2, forkOrder.size());
    assertEquals("java-controlled-tiny-32-headline", forkOrder.get(0).asString(""));
    assertEquals("python-stock-tiny-32-headline", forkOrder.get(1).asString(""));
    assertTrue(sessionBlock.path("python").isObject());
  }

  @Test
  void pythonInstrumentationIsRefused() throws IOException {
    Setup setup = setup();
    FakeStarter starter = new FakeStarter();
    ObjectNode entry = worker("python", "stock", "tiny", 32, "headline");
    entry.put("instrumentation", "tracing");
    assertRefused(buildSpec(setup, base.resolve("session-py-instr"), CLASSPATH, entry), starter);
  }

  @Test
  void bindingCountWithoutAgentJarIsRefused() throws IOException {
    Setup setup = setup();
    FakeStarter starter = new FakeStarter();
    ObjectNode entry = worker("java", "controlled", "tiny", 32, "headline");
    entry.put("instrumentation", "binding-count");
    assertRefused(buildSpec(setup, base.resolve("session-no-agent"), CLASSPATH, entry), starter);
  }

  @Test
  void bindingCountJobRunsInstrumentedCell() throws IOException {
    Setup setup = setup();
    FakeStarter starter = new FakeStarter();
    ObjectNode entry = worker("java", "controlled", "tiny", 32, "headline");
    entry.put("instrumentation", "binding-count");
    System.setProperty(AGENT_JAR_PROPERTY, "/agent/binding-agent.jar");
    Path outRoot = base.resolve("session-agent");
    ObjectNode spec = buildSpec(setup, outRoot, CLASSPATH, entry);
    int exit = new BaselineBenchmark(starter).runSession(spec);
    assertEquals(BaselineBenchmark.EXIT_OK, exit);

    BaselineJob job = starter.jobs.get(0);
    assertEquals("binding-count", job.instrumentation());
    assertEquals("/agent/binding-agent.jar", job.bindingAgentJar());
    assertEquals(outRoot.resolve("native-cache").toString(), job.nativeCacheDir());
    String dir = "java-controlled-tiny-32-headline-binding-count";
    assertEquals(
        dir, BaselineJob.dirName("java", "controlled", "tiny", 32, "headline", "binding-count"));
    // The job file carries the instrumentation; the fork command carries the agent.
    JsonNode jobFile = MAPPER.readTree(outRoot.resolve("jobs").resolve(dir + ".job.json"));
    assertEquals("binding-count", jobFile.path("instrumentation").asString(""));
    List<String> command = job.command(outRoot.resolve("jobs").resolve(dir + ".job.json"));
    assertTrue(command.contains("-javaagent:/agent/binding-agent.jar"));
    // A valid instrumented ok report is excluded from the headline aggregation by
    // construction (frozen section 13.3), with the exclusion recorded.
    JsonNode root = session(spec);
    assertEquals("ok", root.path("workers").get(0).path("status").asString(""));
    assertEquals(0, root.path("summary").path("per_runtime").size());
    assertEquals(0, root.path("summary").path("first_request").size());
    JsonNode exclusion = root.path("summary").path("exclusions").get(0);
    assertEquals("instrumented", exclusion.path("reason").asString(""));
    assertEquals(dir, exclusion.path("worker").asString(""));
  }

  @Test
  void nativeCacheRetainedOnSuccessDeletedOnFailure() throws IOException {
    Setup setup = setup();
    FakeStarter okStarter = new FakeStarter();
    okStarter.createNativeCache = true;
    Path okRoot = base.resolve("session-cache-ok");
    ObjectNode okSpec =
        buildSpec(setup, okRoot, CLASSPATH, worker("java", "controlled", "tiny", 32, "headline"));
    assertEquals(BaselineBenchmark.EXIT_OK, new BaselineBenchmark(okStarter).runSession(okSpec));
    assertTrue(
        Files.isRegularFile(okRoot.resolve("native-cache").resolve("staged.txt")),
        "a successful session retains its isolated native-cache directory");

    Setup failedSetup = setup();
    FakeStarter failedStarter = new FakeStarter();
    failedStarter.createNativeCache = true;
    failedStarter.behaviors.put("java-controlled-tiny-32-headline", Behavior.FAILED);
    Path failedRoot = base.resolve("session-cache-failed");
    ObjectNode failedSpec =
        buildSpec(
            failedSetup,
            failedRoot,
            CLASSPATH,
            worker("java", "controlled", "tiny", 32, "headline"));
    assertEquals(
        BaselineBenchmark.EXIT_WORKERS_FAILED,
        new BaselineBenchmark(failedStarter).runSession(failedSpec));
    assertFalse(
        Files.exists(failedRoot.resolve("native-cache")),
        "a failed session deletes its task-owned isolated native-cache directory");
  }

  @Test
  void validatorAcceptsInstrumentedBindingCountReport() {
    ObjectNode request = BaselineReportFixtures.request(0, "first", 0.02);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    assertTrue(BaselineReportValidator.validate(BaselineReportFixtures.report(worker)).isEmpty());
  }

  @Test
  void validatorAcceptsInstrumentedTracingReport() {
    ObjectNode request = BaselineReportFixtures.request(0, "first", 0.02);
    BaselineReportFixtures.instrument(request, "tracing");
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "tracing");
    List<String> violations =
        BaselineReportValidator.validate(BaselineReportFixtures.report(worker));
    assertTrue(violations.isEmpty(), violations.toString());
  }

  @Test
  void validatorRejectsMemoryBlockOnUninstrumentedWorker() {
    ObjectNode request = BaselineReportFixtures.request(0, "first", 0.02);
    request.putObject("memory").put("active_bytes", 1L);
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    List<String> violations =
        BaselineReportValidator.validate(BaselineReportFixtures.report(worker));
    assertTrue(
        violations.stream()
            .anyMatch(v -> v.contains("memory is null when the worker is not instrumented")),
        violations.toString());
  }

  @Test
  void validatorRejectsInstrumentedReportMissingCountBlock() {
    // A binding-count worker whose request record carries the tracing block instead of
    // binding_counts: the block mismatch is a structural violation, not a silent correction.
    ObjectNode request = BaselineReportFixtures.request(0, "first", 0.02);
    BaselineReportFixtures.instrument(request, "tracing");
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    List<String> violations =
        BaselineReportValidator.validate(BaselineReportFixtures.report(worker));
    assertTrue(
        violations.stream().anyMatch(v -> v.contains("binding_counts block")),
        violations.toString());
  }

  @Test
  void validatorRejectsHostRecordOnUninstrumentedWorker() {
    ObjectNode request = BaselineReportFixtures.request(0, "first", 0.02);
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    ((ObjectNode) worker.get("identity")).putObject("host");
    List<String> violations =
        BaselineReportValidator.validate(BaselineReportFixtures.report(worker));
    assertTrue(
        violations.stream().anyMatch(v -> v.contains("no identity.host record")),
        violations.toString());
  }
}
