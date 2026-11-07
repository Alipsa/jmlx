package se.alipsa.jmlx.benchmarks;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Phase 8.2 decode-baseline launcher (req/plans/phase8-2-plan.md sections 3.2/3.3; frozen contract
 * in req/plans/phase8-2-reference-findings.md section 13). Consumes one spec JSON, validates it and
 * the frozen manifest against the repository, forks one worker process per spec entry
 * (sequentially, in the spec's fork order), and assembles the session report: the session block
 * (the section 10 block minus the identity fields that moved per-worker), the embedded worker
 * reports (one record per worker, including invalid and dead ones) and the summary with its
 * exclusion list and type-7 quantiles over the measured samples.
 *
 * <p>Rules: the session directory ({@code out_root}) must not pre-exist and must be outside the
 * repository — the launcher creates it and refuses to reuse or overwrite; every worker gets its own
 * launcher-created directory and job file; the precision mode is set in each child process
 * environment (the Java worker verifies it, the Python worker sets it from the job); a claimed
 * directory without a report is dead, a worker that left neither claim nor report crashed — both
 * are excluded and never presented as success. Exit codes: 0 all workers ok, 1 spec validation
 * failure (nothing forked), 2 at least one failed/timeout/dead/crashed/not-started/invalid-report
 * worker (the session report is still written).
 *
 * <p>Opt-in local tooling, not part of {@code check} or the publication surface.
 */
public final class BaselineBenchmark {
  public static final int EXIT_OK = 0;
  public static final int EXIT_SPEC_FAILED = 1;
  public static final int EXIT_WORKERS_FAILED = 2;

  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** The per-request metrics aggregated from ok worker reports into the summary. */
  private static final String[] SUMMARY_METRICS = {
    "ttft_s", "prefill_s", "full_generation_s", "streaming_decode_tps", "step_region_s"
  };

  /** One worker process launch; injectable for the pure-Java launcher tests. */
  public interface WorkerStarter {
    /** Forks the worker for {@code job} and returns its exit code. */
    int start(BaselineJob job, Path jobFile, Path stderrLog) throws Exception;
  }

  private static final WorkerStarter FORK =
      (job, jobFile, stderrLog) -> {
        ProcessBuilder builder = new ProcessBuilder(job.command(jobFile));
        builder.environment().put("MLX_ENABLE_TF32", job.precisionEnv());
        // redirectError(File) creates (or truncates) the log; the worker prints nothing on stdout.
        builder.redirectError(stderrLog.toFile());
        builder.redirectOutput(ProcessBuilder.Redirect.DISCARD);
        Process process = builder.start();
        return process.waitFor();
      };

  private final WorkerStarter starter;

  /** Creates the launcher with the real process-forking starter. */
  public BaselineBenchmark() {
    this(FORK);
  }

  BaselineBenchmark(WorkerStarter starter) {
    this.starter = Objects.requireNonNull(starter, "starter");
  }

  /**
   * Runs {@code BaselineBenchmark <spec.json>} and exits with the launcher exit code.
   *
   * @param args the single spec JSON path
   */
  public static void main(String[] args) {
    System.exit(new BaselineBenchmark().run(args.length == 1 ? Path.of(args[0]) : null));
  }

  /** Runs one session from the spec file and returns the launcher exit code. */
  public int run(Path specFile) {
    if (specFile == null || !Files.isRegularFile(specFile)) {
      System.err.println("usage: BaselineBenchmark <spec.json>");
      return EXIT_SPEC_FAILED;
    }
    JsonNode spec;
    try {
      spec = MAPPER.readTree(specFile.toFile());
    } catch (JacksonException e) {
      System.err.println("spec is not readable JSON: " + e.getMessage());
      return EXIT_SPEC_FAILED;
    }
    return runSession(spec);
  }

  int runSession(JsonNode spec) {
    List<String> violations = new ArrayList<>();
    String outRoot = spec.path("out_root").asString(null);
    String repoDir = spec.path("repo_dir").asString(null);
    String precisionEnv = spec.path("precision_env").asString(null);
    Path repo = repoDir == null ? null : Path.of(repoDir).toAbsolutePath().normalize();
    if (outRoot == null || outRoot.isEmpty()) {
      violations.add("out_root is required");
    }
    if (repo == null
        || !Files.isRegularFile(repo.resolve("settings.gradle"))
        || !Files.isRegularFile(repo.resolve("tools/decode-baseline/workloads.json"))) {
      violations.add("repo_dir must be a checkout containing tools/decode-baseline/workloads.json");
    }
    if (!"1".equals(precisionEnv) && !"0".equals(precisionEnv)) {
      violations.add("precision_env must be \"1\" or \"0\"");
    }
    String modelsClasspath = resolveClasspath(spec.path("java_worker_classpath").asString(null));
    if (modelsClasspath == null) {
      violations.add(
          "java_worker_classpath must be set (or the \"@models-test\" alias resolved via the"
              + " jmlx.baseline.models.test.classpath system property)");
    }
    List<BaselineJob.WorkerEntry> workers = new ArrayList<>();
    for (JsonNode entry : spec.path("workers")) {
      int index = workers.size();
      String runtime = entry.path("runtime").asString(null);
      String adapter = entry.path("adapter").asString(null);
      String mode = entry.path("mode").asString(null);
      if (!"java".equals(runtime) && !"python".equals(runtime)) {
        violations.add("workers[" + index + "].runtime must be java or python");
      }
      if (!"controlled".equals(adapter) && !"stock".equals(adapter)) {
        violations.add("workers[" + index + "].adapter must be controlled or stock");
      }
      if (!"headline".equals(mode)
          && !"fixed-continuation".equals(mode)
          && !"reference-pair".equals(mode)) {
        violations.add("workers[" + index + "].mode is not a frozen mode");
      }
      String caseName = entry.path("case").asString(null);
      if (caseName == null || caseName.isEmpty()) {
        violations.add("workers[" + index + "].case is required");
      }
      int promptLength = entry.path("prompt_length").asInt(0);
      if (promptLength <= 0) {
        violations.add("workers[" + index + "].prompt_length must be positive");
      }
      workers.add(new BaselineJob.WorkerEntry(runtime, adapter, caseName, promptLength, mode));
    }
    if (workers.isEmpty()) {
      violations.add("workers must be nonempty");
    }
    Path manifest = repo == null ? null : repo.resolve("tools/decode-baseline/workloads.json");
    JsonNode manifestNode = null;
    String manifestSha = null;
    if (manifest != null && Files.isRegularFile(manifest)) {
      manifestSha = sha256File(manifest);
      try {
        manifestNode = MAPPER.readTree(manifest.toFile());
      } catch (JacksonException e) {
        violations.add("the frozen manifest is not readable JSON: " + e.getMessage());
      }
    }
    Set<String> seenCells = new LinkedHashSet<>();
    for (int i = 0; i < workers.size(); i++) {
      validateEntry(
          workers.get(i),
          i,
          repo,
          manifestNode,
          spec.path("artifact_dirs"),
          modelsClasspath,
          seenCells,
          violations);
    }
    if (!violations.isEmpty()) {
      violations.forEach(v -> System.err.println("spec: " + v));
      return EXIT_SPEC_FAILED;
    }

    Path session = Path.of(outRoot).toAbsolutePath().normalize();
    try {
      if (Files.exists(session)) {
        System.err.println(
            "refusing out_root: " + session + " already exists (unique per session)");
        return EXIT_SPEC_FAILED;
      }
      if (session.startsWith(repo) || session.equals(repo)) {
        System.err.println("refusing out_root inside the repository: " + session);
        return EXIT_SPEC_FAILED;
      }
      Files.createDirectories(session);
    } catch (IOException e) {
      System.err.println("cannot create the session directory: " + e.getMessage());
      return EXIT_SPEC_FAILED;
    }
    writeSpecCopy(session.resolve("spec.json"), spec);
    // Session start timestamp for the report; captured before the fork loop, used after it.
    final String utcStart = Instant.now().toString();
    Path jobsDir = session.resolve("jobs");
    Path stderrs = session.resolve("stderr");
    try {
      Files.createDirectories(jobsDir);
      Files.createDirectories(stderrs);
    } catch (IOException e) {
      System.err.println("cannot create the session subdirectories: " + e.getMessage());
      return EXIT_WORKERS_FAILED;
    }

    // Fork sequentially in the spec's fork order. A worker that exits nonzero is a normal
    // outcome (its report carries the failure); a starter exception is launcher-side and stops
    // the session — the remaining workers are recorded not-started, never silently dropped.
    List<BaselineJob> builtJobs = new ArrayList<>();
    List<Boolean> started = new ArrayList<>();
    for (int i = 0; i < workers.size(); i++) {
      BaselineJob.WorkerEntry entry = workers.get(i);
      String dirName =
          BaselineJob.dirName(
              entry.runtime(),
              entry.adapter(),
              entry.caseName(),
              entry.promptLength(),
              entry.mode());
      Path outDir = session.resolve(dirName);
      Path jobFile = jobsDir.resolve(dirName + ".job.json");
      BaselineJob job = null;
      try {
        Files.createDirectories(outDir);
        job =
            buildJob(
                entry,
                outDir,
                repo,
                manifest,
                manifestNode,
                manifestSha,
                precisionEnv,
                spec.path("artifact_dirs"),
                modelsClasspath);
        job.write(jobFile);
      } catch (IOException e) {
        System.err.println("cannot set up the job for " + dirName + ": " + e.getMessage());
      }
      boolean forked = job != null;
      if (forked) {
        try {
          starter.start(job, jobFile, stderrs.resolve(dirName + ".log"));
        } catch (Exception e) {
          System.err.println("worker " + dirName + " could not be started: " + e);
          forked = false;
        }
      }
      builtJobs.add(job);
      started.add(forked);
      if (!forked) {
        break;
      }
    }
    for (int i = builtJobs.size(); i < workers.size(); i++) {
      builtJobs.add(null);
      started.add(false);
    }

    Assembled assembled =
        assembleSession(
            session,
            repo,
            manifestSha,
            manifestNode,
            precisionEnv,
            utcStart,
            workers,
            builtJobs,
            started);
    boolean allOk = assembled.outcomes().stream().allMatch("ok"::equals);
    try {
      Files.writeString(
          session.resolve("session.json"),
          MAPPER.writeValueAsString(assembled.report()),
          StandardCharsets.UTF_8);
    } catch (IOException | JacksonException e) {
      System.err.println("session report write failed: " + e.getMessage());
      return EXIT_WORKERS_FAILED;
    }
    return allOk ? EXIT_OK : EXIT_WORKERS_FAILED;
  }

  /** The assembled session report plus the per-worker outcomes that decide the exit code. */
  private record Assembled(ObjectNode report, List<String> outcomes) {}

  private Assembled assembleSession(
      Path session,
      Path repo,
      String manifestSha,
      JsonNode manifestNode,
      String precisionEnv,
      String utcStart,
      List<BaselineJob.WorkerEntry> workers,
      List<BaselineJob> builtJobs,
      List<Boolean> started) {
    ObjectNode root = MAPPER.createObjectNode();
    root.put("schema", BaselineReportValidator.SCHEMA);
    ObjectNode sessionBlock = root.putObject("session");
    sessionBlock.put("utc_start", utcStart);
    sessionBlock.put("utc_end", Instant.now().toString());
    sessionBlock.set("host", hostBlock(session));
    sessionBlock.set("code", repoIdentity(repo));
    sessionBlock.put("precision_env", "MLX_ENABLE_TF32=" + precisionEnv);
    sessionBlock.put(
        "cache_regime",
        manifestNode.path("run_policy").path("cache_regime").asString("primed-per-request"));
    sessionBlock.put("workload", manifestSha);
    ObjectNode java = sessionBlock.putObject("java");
    java.put("vendor", System.getProperty("java.vendor", ""));
    java.put("version", System.getProperty("java.version", ""));
    java.put("flags", System.getProperty("java.vm.command", ""));
    // Replaced below when a Python worker's identity names its venv (WP1 runs are Java-only).
    sessionBlock.putNull("python");
    sessionBlock.set("native_pins", nativePins(repo));
    ArrayNode forkOrder = sessionBlock.putArray("fork_order");

    ObjectNode summary = root.putObject("summary");
    summary.put("quantile_convention", "linear-interpolation-type-7");
    // Reference pairs are aggregated at WP3; WP1 sessions carry the empty list.
    summary.putArray("pairs");
    ArrayNode exclusions = summary.putArray("exclusions");
    ObjectNode perRuntime = summary.putObject("per_runtime");
    ObjectNode firstRequest = summary.putObject("first_request");
    ArrayNode workerRecords = root.putArray("workers");

    List<String> outcomes = new ArrayList<>();
    Map<String, Map<String, List<Double>>> measured = new LinkedHashMap<>();
    Map<String, Map<String, List<Double>>> firsts = new LinkedHashMap<>();
    for (int i = 0; i < workers.size(); i++) {
      BaselineJob.WorkerEntry entry = workers.get(i);
      BaselineJob job = builtJobs.get(i);
      String dirName =
          BaselineJob.dirName(
              entry.runtime(),
              entry.adapter(),
              entry.caseName(),
              entry.promptLength(),
              entry.mode());
      if (job == null || !started.get(i)) {
        outcomes.add("not-started");
        addExclusion(
            exclusions, dirName, "not-started", List.of("launcher failed to start the worker"));
        workerRecords.add(stubRecord(entry, dirName, "not-started", null));
        continue;
      }
      forkOrder.add(job.runtime());
      Path reportPath = session.resolve(dirName).resolve("report.json");
      JsonNode report = readReport(reportPath);
      String outcome;
      List<String> reportViolations = List.of();
      if (report != null) {
        reportViolations = BaselineReportValidator.validate(report);
        outcome =
            reportViolations.isEmpty()
                ? report.path("worker").path("status").asString("failed")
                : "invalid-report";
      } else {
        outcome =
            Files.exists(session.resolve(dirName).resolve("worker-claim")) ? "dead" : "crashed";
      }
      outcomes.add(outcome);
      if (!"ok".equals(outcome)) {
        addExclusion(exclusions, dirName, outcome, reportViolations);
      }
      if (report != null && reportViolations.isEmpty()) {
        ObjectNode workerBlock = (ObjectNode) report.get("worker");
        workerBlock.put("dir", dirName);
        workerRecords.add(workerBlock);
        if ("ok".equals(outcome)) {
          accumulate(job, report, measured, firsts);
        }
      } else {
        workerRecords.add(stubRecord(entry, dirName, outcome, report));
      }
      if ("python".equals(job.runtime()) && report != null) {
        JsonNode python = report.path("worker").path("identity").path("python");
        if (python.isObject()) {
          sessionBlock.set("python", python);
        }
      }
    }
    finishQuantiles(perRuntime, measured);
    finishFirstRequest(firstRequest, firsts);
    return new Assembled(root, outcomes);
  }

  /** Reads a worker report, or null when it is absent or not a JSON object. */
  private static JsonNode readReport(Path path) {
    if (!Files.isRegularFile(path)) {
      return null;
    }
    try {
      JsonNode node = MAPPER.readTree(path.toFile());
      return node == null || !node.isObject() ? null : node;
    } catch (JacksonException e) {
      return null;
    }
  }

  private static void addExclusion(
      ArrayNode exclusions, String dirName, String reason, List<String> reportViolations) {
    ObjectNode exclusion = exclusions.addObject();
    exclusion.put("worker", dirName);
    exclusion.put("reason", reason);
    if (!reportViolations.isEmpty()) {
      ArrayNode list = exclusion.putArray("violations");
      for (String violation : reportViolations) {
        list.add(violation);
      }
    }
  }

  /** The session-report record for a worker that has no valid report. */
  private static ObjectNode stubRecord(
      BaselineJob.WorkerEntry entry, String dirName, String outcome, JsonNode report) {
    ObjectNode node = MAPPER.createObjectNode();
    node.put("dir", dirName);
    node.put("runtime", entry.runtime());
    node.put("adapter", entry.adapter());
    node.put("case", entry.caseName());
    node.put("prompt_length", entry.promptLength());
    node.put("mode", entry.mode());
    node.put("outcome", outcome);
    if (report != null) {
      node.set("report", report);
    }
    return node;
  }

  /**
   * Collects the summary samples: measured requests per (runtime, mode, case, length) cell and
   * first requests per (runtime, mode); warmups are excluded from both. Modes are never pooled —
   * headline and fixed-continuation measure different quantities, and reference-pair requests run
   * the observed (unprimed) cache regime, whose samples belong to the WP3 {@code pairs}
   * aggregation, never to the primed headline quantiles.
   */
  private static void accumulate(
      BaselineJob job,
      JsonNode report,
      Map<String, Map<String, List<Double>>> measured,
      Map<String, Map<String, List<Double>>> firsts) {
    if ("reference-pair".equals(job.mode())) {
      return;
    }
    String cell =
        job.runtime() + "|" + job.mode() + "/" + job.caseName() + "/" + job.promptLength();
    String firstKey = job.runtime() + "|" + job.mode();
    for (JsonNode request : report.path("worker").path("requests")) {
      String kind = request.path("kind").asString(null);
      for (String metric : SUMMARY_METRICS) {
        JsonNode value = request.path(metric);
        if (!value.isDouble()) {
          continue;
        }
        if ("measured".equals(kind)) {
          measured
              .computeIfAbsent(cell, k -> new LinkedHashMap<>())
              .computeIfAbsent(metric, k -> new ArrayList<>())
              .add(value.asDouble());
        } else if ("first".equals(kind)) {
          firsts
              .computeIfAbsent(firstKey, k -> new LinkedHashMap<>())
              .computeIfAbsent(metric, k -> new ArrayList<>())
              .add(value.asDouble());
        }
      }
    }
  }

  /** Replaces the measured sample lists with their type-7 quantile summaries. */
  private static void finishQuantiles(
      ObjectNode perRuntime, Map<String, Map<String, List<Double>>> measured) {
    Map<String, Map<String, Map<String, List<Double>>>> byRuntime = new LinkedHashMap<>();
    for (Map.Entry<String, Map<String, List<Double>>> cell : measured.entrySet()) {
      int separator = cell.getKey().indexOf('|');
      String runtime = cell.getKey().substring(0, separator);
      byRuntime
          .computeIfAbsent(runtime, k -> new LinkedHashMap<>())
          .put(cell.getKey().substring(separator + 1), cell.getValue());
    }
    for (Map.Entry<String, Map<String, Map<String, List<Double>>>> runtime : byRuntime.entrySet()) {
      ObjectNode cells = perRuntime.putObject(runtime.getKey());
      for (Map.Entry<String, Map<String, List<Double>>> cell : runtime.getValue().entrySet()) {
        ObjectNode metrics = cells.putObject(cell.getKey());
        for (Map.Entry<String, List<Double>> metric : cell.getValue().entrySet()) {
          metrics.set(
              metric.getKey(), quantileNode(BaselineStatistics.quantiles(metric.getValue())));
        }
      }
    }
  }

  /** The first-request samples are a separate distribution, quantiled per (runtime, mode). */
  private static void finishFirstRequest(
      ObjectNode firstRequest, Map<String, Map<String, List<Double>>> firsts) {
    for (Map.Entry<String, Map<String, List<Double>>> entry : firsts.entrySet()) {
      int separator = entry.getKey().indexOf('|');
      ObjectNode metrics =
          firstRequest
              .putObject(entry.getKey().substring(0, separator))
              .putObject(entry.getKey().substring(separator + 1));
      for (Map.Entry<String, List<Double>> metric : entry.getValue().entrySet()) {
        metrics.set(metric.getKey(), quantileNode(BaselineStatistics.quantiles(metric.getValue())));
      }
    }
  }

  private static ObjectNode quantileNode(BaselineStatistics.Quantiles q) {
    ObjectNode node = MAPPER.createObjectNode();
    node.put("p10", q.p10());
    node.put("median", q.median());
    node.put("p90", q.p90());
    node.put("n", q.n());
    return node;
  }

  private static void writeSpecCopy(Path file, JsonNode spec) {
    // The spec copy is best-effort provenance; a failure here must not abort the session.
    try {
      Files.writeString(file, MAPPER.writeValueAsString(spec), StandardCharsets.UTF_8);
    } catch (IOException | JacksonException ignored) {
      // the session proceeds without the spec copy
    }
  }

  private static void validateEntry(
      BaselineJob.WorkerEntry entry,
      int index,
      Path repo,
      JsonNode manifestNode,
      JsonNode artifactDirs,
      String modelsClasspath,
      Set<String> seenCells,
      List<String> violations) {
    String prefix = "workers[" + index + "] ";
    if (entry.caseName() == null || entry.caseName().isEmpty()) {
      return;
    }
    if (manifestNode != null) {
      boolean found = false;
      boolean frozenLength = false;
      for (JsonNode c : manifestNode.path("text_cases")) {
        if (entry.caseName().equals(c.path("name").asString(null))) {
          found = true;
          for (JsonNode l : manifestNode.path("controlled_generation").path("prompt_lengths")) {
            if (l.asInt() == entry.promptLength()) {
              frozenLength = true;
            }
          }
        }
      }
      if (!found) {
        violations.add(prefix + "case '" + entry.caseName() + "' is not a manifest text case");
      } else if (!frozenLength) {
        violations.add(
            prefix
                + "prompt_length "
                + entry.promptLength()
                + " is not a frozen manifest length for "
                + entry.caseName());
      }
    }
    String artifactDir = artifactDirs.path(entry.caseName()).asString(null);
    if (artifactDir == null || artifactDir.isEmpty() || !Files.isDirectory(Path.of(artifactDir))) {
      violations.add(
          prefix + "artifact_dirs." + entry.caseName() + " must name an existing directory");
    }
    if ("java".equals(entry.runtime()) && (modelsClasspath == null || modelsClasspath.isEmpty())) {
      violations.add(prefix + "a java worker needs the models test classpath");
    }
    if ("python".equals(entry.runtime())
        && (repo == null
            || !Files.isRegularFile(repo.resolve("tools/decode-baseline/.venv/bin/python")))) {
      violations.add(
          prefix
              + "the pinned decode-baseline venv python is missing (run"
              + " tools/decode-baseline/install.sh)");
    }
    String cell =
        String.join(
            "/",
            entry.runtime(),
            entry.adapter(),
            entry.caseName(),
            Integer.toString(entry.promptLength()),
            entry.mode());
    if (!seenCells.add(cell)) {
      violations.add(prefix + "duplicate cell " + cell);
    }
  }

  private static BaselineJob buildJob(
      BaselineJob.WorkerEntry entry,
      Path outDir,
      Path repo,
      Path manifest,
      JsonNode manifestNode,
      String manifestSha,
      String precisionEnv,
      JsonNode artifactDirs,
      String modelsClasspath) {
    JsonNode policy = manifestNode.path("run_policy");
    boolean referencePair = "reference-pair".equals(entry.mode());
    int timeoutS =
        referencePair
            ? policy.path("reference_pair_worker_timeout_s").asInt(720)
            : policy.path("worker_timeout_s").asInt(1800);
    int first = referencePair ? 0 : policy.path("first_requests").asInt(1);
    int warmups = referencePair ? 0 : policy.path("warmups").asInt(0);
    int measured = referencePair ? 1 : policy.path("measured_requests").asInt(0);
    String artifactDir = artifactDirs.path(entry.caseName()).asString("");
    return new BaselineJob(
        entry.adapter(),
        entry.caseName(),
        entry.promptLength(),
        entry.mode(),
        first,
        warmups,
        measured,
        timeoutS,
        precisionEnv,
        artifactDir,
        manifest.toString(),
        manifestSha,
        repo.toString(),
        outDir.toString(),
        32,
        32,
        entry.runtime(),
        modelsClasspath);
  }

  private static ObjectNode hostBlock(Path session) {
    ObjectNode host = MAPPER.createObjectNode();
    host.put("chip", sysctl("hw.model", ""));
    host.put("ram_bytes", sysctlLong("hw.memsize", 0L));
    host.put("os", uname("-s"));
    host.put("os_build", uname("-r"));
    host.put("disk_free_bytes", diskFreeBytes(session));
    // The hostrec pre/post records are written by the WP3 hostrec.sh convention; WP1
    // sessions carry the missing-sample nulls.
    ObjectNode hostrec = host.putObject("hostrec");
    hostrec.putNull("pre");
    hostrec.putNull("post");
    return host;
  }

  private static ObjectNode repoIdentity(Path repo) {
    String sha = git(repo, "rev-parse", "HEAD");
    if (sha == null) {
      ObjectNode code = MAPPER.createObjectNode();
      code.put("sha", "");
      code.put("branch", "");
      code.put("dirty", true);
      return code;
    }
    ObjectNode code = MAPPER.createObjectNode();
    code.put("sha", sha);
    code.put("branch", git(repo, "rev-parse", "--abbrev-ref", "HEAD"));
    String status = git(repo, "status", "--porcelain");
    code.put("dirty", status != null && !status.isEmpty());
    return code;
  }

  private static ObjectNode nativePins(Path repo) {
    ObjectNode pins = MAPPER.createObjectNode();
    ObjectNode javaPins = pins.putObject("java");
    javaPins.put("metal", "");
    javaPins.put("mlxc_commit", "");
    Path pinFile = repo.resolve("native/install/lib/native-pin.properties");
    if (Files.isRegularFile(pinFile)) {
      for (String line : readLines(pinFile)) {
        int eq = line.indexOf('=');
        if (eq > 0) {
          String key = line.substring(0, eq).trim();
          String value = line.substring(eq + 1).trim();
          if ("mlxMetalVersion".equals(key)) {
            javaPins.put("metal", value);
          } else if ("mlxcCommit".equals(key)) {
            javaPins.put("mlxc_commit", value);
          }
        }
      }
    }
    // Python pins are read from the venv by the Python workers; WP1 carries the nulls.
    ObjectNode pythonPins = pins.putObject("python");
    pythonPins.putNull("mlx");
    pythonPins.putNull("mlx_metal");
    return pins;
  }

  private static String resolveClasspath(String raw) {
    if (raw == null || raw.isEmpty()) {
      return null;
    }
    if ("@models-test".equals(raw)) {
      String resolved = System.getProperty("jmlx.baseline.models.test.classpath");
      return resolved == null || resolved.isEmpty() ? null : resolved;
    }
    return raw;
  }

  private static String sysctl(String key, String fallback) {
    String out = commandOutput(new String[] {"sysctl", "-n", key});
    return out == null ? fallback : out;
  }

  private static long sysctlLong(String key, long fallback) {
    String out = sysctl(key, null);
    if (out == null) {
      return fallback;
    }
    try {
      return Long.parseLong(out);
    } catch (NumberFormatException e) {
      return fallback;
    }
  }

  private static String uname(String flag) {
    String out = commandOutput(new String[] {"uname", flag});
    return out == null ? "" : out;
  }

  private static long diskFreeBytes(Path session) {
    Path probe = Files.exists(session) ? session : Path.of(System.getProperty("user.dir", "/"));
    String out = commandOutput(new String[] {"df", "-Pk", probe.toString()});
    if (out == null) {
      return 0L;
    }
    String[] lines = out.strip().split("\n");
    if (lines.length < 2) {
      return 0L;
    }
    String[] parts = lines[1].split("\\s+");
    if (parts.length < 3) {
      return 0L;
    }
    try {
      return Long.parseLong(parts[3]) * 1024L;
    } catch (NumberFormatException e) {
      return 0L;
    }
  }

  private static String git(Path repo, String... args) {
    List<String> command = new ArrayList<>(List.of("git", "-C", repo.toString()));
    command.addAll(List.of(args));
    return commandOutput(command);
  }

  private static String commandOutput(List<String> command) {
    try {
      Process process = new ProcessBuilder(command).redirectErrorStream(true).start();
      String out = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
      return process.waitFor() == 0 ? out.strip() : null;
    } catch (Exception e) {
      return null;
    }
  }

  private static String commandOutput(String[] command) {
    return commandOutput(List.of(command));
  }

  private static List<String> readLines(Path path) {
    try {
      return Files.readAllLines(path, StandardCharsets.UTF_8);
    } catch (IOException e) {
      return List.of();
    }
  }

  static String sha256File(Path path) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      try (var in = Files.newInputStream(path)) {
        byte[] chunk = new byte[8 * 1024 * 1024];
        int read;
        while ((read = in.read(chunk)) >= 0) {
          digest.update(chunk, 0, read);
        }
      }
      return HexFormat.of().formatHex(digest.digest());
    } catch (IOException | NoSuchAlgorithmException e) {
      throw new IllegalStateException("sha256 failed for " + path, e);
    }
  }
}
