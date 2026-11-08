package se.alipsa.jmlx.models;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.management.ManagementFactory;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.IntConsumer;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Phase 8.2 decode-baseline Java worker (controlled and stock adapters).
 *
 * <p>Consumes a launcher-written job file (schema {@code jmlx-decode-baseline-job-1}) and writes a
 * per-worker report (schema {@code jmlx-decode-baseline-1}) plus a {@code worker-ready} marker into
 * the job's output directory. Implements the same frozen contract as the pinned Python worker
 * ({@code tools/decode-baseline/worker.py}): req/plans/phase8-2-reference-findings.md sections 13.1
 * (job file), 13.2 (worker protocol and output layout) and 13.3 (report record).
 *
 * <p>Rules enforced here mirror the Python worker:
 *
 * <ul>
 *   <li>the frozen workload manifest is re-hashed and every consumed file digest is re-verified
 *       before any model work; the {@code manifest_path} must be the repository's canonical {@code
 *       tools/decode-baseline/workloads.json} and {@code repo_dir} the repository this worker runs
 *       from, both anchored on the worker's own location, so a hand-written job cannot name a
 *       different checkout;
 *   <li>the artifact's {@code *.safetensors} set must equal the case's pinned set exactly — no
 *       unpinned weight file is primed or loaded;
 *   <li>the process environment {@code MLX_ENABLE_TF32} is verified before any native
 *       initialization: the launcher sets it on the worker process from the job (JDK 25 removed
 *       {@code System.setenv}), so a mis-launched worker or a shell-inherited precision mode fails
 *       the run instead of silently selecting it (findings section 14.1 deviation a);
 *   <li>one sequential full read of every checkpoint safetensors file precedes each request (cache
 *       regime primed-per-request); {@code prime_s} is recorded and excluded from every measured
 *       interval — reference-pair workers prime nothing;
 *   <li>before any validation, the worker claims the launcher-created output directory with an
 *       exclusive-create {@code worker-claim} file; an out_dir that is not an existing directory is
 *       rejected before the claim; a pre-existing claim refuses the run (the claim is sticky — a
 *       claimed directory without a report is dead for the launcher);
 *   <li>the report is written at most once, atomically, via a per-process temp file, and never
 *       overwrites an existing {@code report.json}; a success whose report write is refused is
 *       downgraded to failed, and the refusal message is printed with the final status;
 *   <li>a wall-clock timeout — enforced at request boundaries, re-checked after the final request,
 *       and backed by a grace-period backstop — produces a timeout report and a nonzero exit; a
 *       worker never presents a partial result as success.
 * </ul>
 *
 * <p>Exit codes: 0 ok, 1 failed (validation, contract or unexpected error), 3 timeout.
 *
 * <p>Benchmark-only test source: this class and the adapters it drives are never published as API.
 * The package-visible seams ({@code setAnchorsForTest}, {@code setExitHookForTest}, {@code
 * setWatchdogGraceMsForTest}, {@code setRequestDelayMsForTest}) exist only so the protocol and the
 * timeout path can be executed by tests; the launcher never sets them.
 */
public final class BaselineWorker {
  public static final String JOB_SCHEMA = "jmlx-decode-baseline-job-1";
  public static final String REPORT_SCHEMA = "jmlx-decode-baseline-1";
  public static final int EXIT_OK = 0;
  public static final int EXIT_FAILED = 1;
  public static final int EXIT_TIMEOUT = 3;

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final long PRIME_CHUNK = 1 << 20;
  private static final long DEFAULT_WATCHDOG_GRACE_MS = 30_000L;

  /** Anchors the manifest/repo identity checks on the worker's own location. */
  record Anchors(Path canonicalManifest, Path canonicalRepo) {}

  /** Parsed and validated job (section 13.1). */
  private record Validated(
      String adapter,
      String caseName,
      int promptLength,
      String mode,
      Map<String, Integer> requests,
      int timeoutS,
      String precisionEnv,
      Path artifactDir,
      Path manifestPath,
      String manifestSha,
      Path repoDir,
      Path outDir,
      List<Integer> prompt,
      List<Integer> fedIds,
      boolean ioAffected,
      Map<String, Object> runPolicy,
      String wiredLimitPolicy,
      List<Integer> stopTokenIds) {}

  private static final Object REPORT_LOCK = new Object();

  // Mutable run state; the watchdog backstop and the main thread both reach it.
  private static String utcStart;
  private static String utcEnd;
  private static String status = "failed";
  private static String failureReason;
  private static String adapter;
  private static String caseName;
  private static Integer promptLength;
  private static String mode;
  private static Map<String, Object> identity = Map.of();
  private static List<Integer> stopTokenIds = List.of();
  private static Double loadS;
  private static String loadState;
  private static final List<Map<String, Object>> requests = new ArrayList<>();
  private static final List<String> notes = new ArrayList<>();
  private static boolean reportWritten;
  private static boolean reportRefused;
  private static Path outDir;
  private static MLXScope modelScope;

  private static Anchors anchors;
  private static IntConsumer exitHook = System::exit;
  private static long watchdogGraceMs = DEFAULT_WATCHDOG_GRACE_MS;
  private static long requestDelayMs = 0L;

  private static final AtomicBoolean fired = new AtomicBoolean();
  private static final AtomicBoolean finished = new AtomicBoolean();

  /**
   * Monotonic per-run token: each {@link #run} increments it, and that run's watchdog captures it.
   * A watchdog whose token no longer matches the current one is stale (a later run owns the shared
   * run state now) and must drop every remaining action silently.
   */
  private static final AtomicLong runToken = new AtomicLong();

  private static Thread mainWorkerThread;

  private BaselineWorker() {}

  /** Worker entry point: one argument, the launcher-written job file. */
  public static void main(String[] args) {
    int code;
    try {
      if (args.length != 1 || !Files.isRegularFile(Path.of(args[0]))) {
        throw new WorkerError("usage: BaselineWorker <job.json>");
      }
      code = run(MAPPER.readTree(Path.of(args[0]).toFile()));
    } catch (WorkerError e) {
      code = handleValidationFailure(e);
    } catch (Exception e) {
      code = handleUnexpected(e);
    }
    mainWorkerThread = null;
    exitHook.accept(code);
  }

  /**
   * Runs one worker job to completion and returns the exit code without exiting the process, so the
   * protocol and timeout paths can be executed in-process by tests. Production reaches it through
   * {@link #main} only.
   */
  static int run(JsonNode job) {
    resetRunState();
    mainWorkerThread = Thread.currentThread();
    long token;
    // The token increment and the fired/finished reset are one critical section: a watchdog from
    // a previous run must see either the old token with the old flags, or the new token with the
    // new flags — never the new flags under the old token.
    synchronized (REPORT_LOCK) {
      token = runToken.incrementAndGet();
      fired.set(false);
      finished.set(false);
    }
    utcStart = utcNow();
    try {
      return execute(job, token);
    } catch (WorkerError e) {
      return handleValidationFailure(e);
    } catch (Exception e) {
      return handleUnexpected(e);
    } finally {
      synchronized (REPORT_LOCK) {
        if (runToken.get() == token) {
          finished.set(true);
        }
      }
    }
  }

  private static void resetRunState() {
    utcEnd = null;
    status = "failed";
    failureReason = null;
    adapter = null;
    caseName = null;
    promptLength = null;
    mode = null;
    identity = Map.of();
    stopTokenIds = List.of();
    loadS = null;
    loadState = null;
    requests.clear();
    notes.clear();
    reportWritten = false;
    reportRefused = false;
    outDir = null;
    modelScope = null;
  }

  private static int execute(JsonNode job, long token) throws Exception {
    // The launcher creates the directory; the claim below can only distinguish "already
    // claimed" from "cannot be written", so a missing (or file-in-the-way) out_dir is
    // rejected explicitly, not as an IO error from the claim create.
    String rawOutDir = job.path("out_dir").asString("");
    if (!rawOutDir.isEmpty()) {
      Path dir = Path.of(rawOutDir);
      outDir = dir;
      if (!Files.isDirectory(dir)) {
        throw new WorkerError("out_dir must be created by the launcher");
      }
      // Claim the launcher-created output directory before any validation: a second worker
      // (valid or not) pointed at a folder this worker owns must refuse the folder, not
      // write its report into it. The claim is sticky: a killed worker leaves it, so a
      // claimed folder without a report is dead for the launcher.
      Path claim = dir.resolve("worker-claim");
      try {
        Files.writeString(
            claim, claimJson().toString(), StandardCharsets.UTF_8, StandardOpenOption.CREATE_NEW);
      } catch (FileAlreadyExistsException e) {
        System.err.println(
            "failure_reason: output folder already claimed by another worker ("
                + claim
                + " exists); refusing to write into "
                + rawOutDir);
        return EXIT_FAILED;
      }
    }
    Validated v = validate(job);
    adapter = v.adapter();
    caseName = v.caseName();
    promptLength = v.promptLength();
    mode = v.mode();
    outDir = v.outDir();

    // Verified, not set, before any native initialization (plan section 3.2): JDK 25 removed
    // System.setenv (deprecated for removal), so the launcher owns the worker process
    // environment and sets MLX_ENABLE_TF32 on the child (findings section 13.5 Java
    // deviation). A mis-launched worker — or one inheriting a shell precision mode the
    // launcher did not override — fails the run instead of silently selecting it.
    String actualPrecisionEnv = System.getenv("MLX_ENABLE_TF32");
    require(
        v.precisionEnv().equals(actualPrecisionEnv),
        "MLX_ENABLE_TF32 process environment "
            + (actualPrecisionEnv == null ? "(unset)" : "'" + actualPrecisionEnv + "'")
            + " differs from the job precision_env '"
            + v.precisionEnv()
            + "'; the launcher must set the worker process environment");
    stopTokenIds = List.copyOf(v.stopTokenIds());
    identity =
        Map.of(
            "code",
            repoIdentity(v.repoDir()),
            "manifest_sha256",
            v.manifestSha(),
            "run_policy",
            v.runPolicy(),
            "precision_env",
            "MLX_ENABLE_TF32=" + v.precisionEnv(),
            "java",
            Map.of(
                "version",
                System.getProperty("java.version", ""),
                "vendor",
                System.getProperty("java.vendor", ""),
                "flags",
                String.join(" ", ManagementFactory.getRuntimeMXBean().getInputArguments())),
            "device",
            Map.of(
                "device_name", hostString("hw.model", ""),
                "memory_size_bytes", hostLong("hw.memsize", 0L),
                "max_recommended_working_set_bytes", 0L),
            "wired_limit_policy",
            v.wiredLimitPolicy(),
            "cache_regime",
            "reference-pair".equals(v.mode())
                ? "observed"
                : String.valueOf(v.runPolicy().get("cache_regime")),
            "io_affected",
            v.ioAffected(),
            "instrumented",
            false);
    notes.add(
        "java identity.device carries host facts (sysctl); the Java worker has no MLX"
            + " device-info facade, so max_recommended_working_set_bytes is 0");

    Path ready = v.outDir().resolve("worker-ready");
    try {
      Files.writeString(
          ready,
          readyJson(v.adapter(), v.caseName()).toString(),
          StandardCharsets.UTF_8,
          StandardOpenOption.CREATE_NEW);
    } catch (FileAlreadyExistsException e) {
      // The folder was claimed by this run, so the catch-all writes a failed report into it.
      throw new WorkerError("file already exists: " + ready);
    }

    // The deadline sits at worker-ready, before model load — the same anchor the pinned
    // Python worker uses (findings section 13.4 item 13/14 placement).
    long deadlineNanos = System.nanoTime() + v.timeoutS() * 1_000_000_000L;
    Thread watchdog = new Thread(() -> watchdog(deadlineNanos, token), "baseline-worker-watchdog");
    watchdog.setDaemon(true);
    watchdog.start();

    try (MLXScope scope = new MLXScope()) {
      modelScope = scope;
      long loadStart = System.nanoTime();
      TextGenerationModel loaded = TextGenerationModels.load(modelScope, v.artifactDir());
      loadS = (System.nanoTime() - loadStart) / 1_000_000_000d;
      loadState = "materialized";
      if (!(loaded instanceof DecoderModel model)) {
        throw new WorkerError(
            "case " + v.caseName() + " loaded a non-decoder model; text cases are decoders");
      }
      HfTokenizer tokenizer = null;
      if ("stock".equals(v.adapter())) {
        // The tokenizer loads first: its special-token eos participates in the resolved stop set.
        tokenizer = HfTokenizer.fromDirectory(v.artifactDir());
        assertStockStopSet(v.artifactDir(), tokenizer, v.caseName(), v.stopTokenIds());
        notes.add(
            "stock stop set resolved from the artifact (tokenizer special-token eos plus"
                + " generation_config.json and config.json eos_token_id) equals the manifest"
                + " frozen set for "
                + v.caseName());
        notes.add(
            "stock (public generate with the checkpoint's own HfTokenizer) retains the"
                + " generation loop's incremental detokenization and finalization inside the"
                + " stream; event times include it. unused_lookahead_forwards is 0: the"
                + " synchronous Java loop evaluates every step at its selection boundary and"
                + " queues no lookahead forward, so nothing is pending when generate returns");
      }
      if ("fixed-continuation".equals(v.mode())) {
        notes.add(
            "fixed-continuation steps the manifest prefix512[32:64] ids through the loaded model"
                + " (Java: the model's own seam step path, one id per step); no sampling, no"
                + " lookahead pipeline; qualification workload per plan section 3.3");
      }

      List<String> kinds = new ArrayList<>();
      for (int i = 0; i < v.requests().get("first"); i++) {
        kinds.add("first");
      }
      for (int i = 0; i < v.requests().get("warmups"); i++) {
        kinds.add("warmup");
      }
      for (int i = 0; i < v.requests().get("measured"); i++) {
        kinds.add("measured");
      }
      int emitted = (int) v.runPolicy().get("emitted_tokens");
      for (int index = 0; index < kinds.size(); index++) {
        if (fired.get()) {
          status = "timeout";
          failureReason = "worker wall-clock timeout exceeded";
          break;
        }
        // Test seam: simulate a wedged request so the backstop path can be executed (the
        // production worker never sets this).
        if (requestDelayMs > 0) {
          Thread.sleep(requestDelayMs);
        }
        double primeS = "reference-pair".equals(v.mode()) ? 0.0 : prime(v.artifactDir());
        requests.add(requestRecord(v, model, tokenizer, primeS, index, kinds.get(index), emitted));
      }
    }

    // Re-check the deadline after the final request: if it fired while the last request was
    // running, the run is a timeout even though that request finished within the watchdog
    // grace.
    if (fired.get() && !("timeout".equals(status))) {
      status = "timeout";
      failureReason = "worker wall-clock timeout exceeded";
    }
    if ("failed".equals(status) && failureReason == null) {
      status = "ok";
    }
    boolean written = writeReport(v.outDir());
    if ("ok".equals(status) && !written) {
      // The write can only be refused here when a concurrent process already reported this
      // directory; this worker's results are then not recorded, so it must not present the
      // run as a success.
      status = "failed";
      failureReason = "report write refused (directory already reported); results not recorded";
    }
    if (reportRefused) {
      // Printed after the downgrade so the message carries the final status.
      System.err.println(refusalMessage(v.outDir()));
    }
    return switch (status) {
      case "ok" -> EXIT_OK;
      case "timeout" -> EXIT_TIMEOUT;
      default -> EXIT_FAILED;
    };
  }

  /**
   * Hard backstop: if the main loop has not exited shortly after the deadline, write the timeout
   * report and exit 3. The watchdog carries its run's token: if a later run has started in the same
   * process, the shared run state (flags, status, out_dir) belongs to that run, so every remaining
   * action is dropped silently — a stale watchdog must never fire into, report for, or exit on
   * behalf of a healthy later run. When the report is already written the status overwrite is
   * skipped (the main thread's report stays authoritative); the exit is unconditional. When the run
   * finished before the deadline, or finished normally after a fire, no backstop exit is performed.
   */
  private static void watchdog(long deadlineNanos, long token) {
    try {
      while (!fired.get() && !finished.get()) {
        Thread.sleep(1000L);
        if (runToken.get() != token || finished.get()) {
          return;
        }
        if (System.nanoTime() < deadlineNanos) {
          continue;
        }
        boolean fire;
        synchronized (REPORT_LOCK) {
          fire = runToken.get() == token && !fired.get() && !finished.get();
          if (fire) {
            fired.set(true);
          }
        }
        if (!fire) {
          return;
        }
      }
      if (!fired.get()) {
        return;
      }
      Thread.sleep(watchdogGraceMs);
      boolean alreadyWritten;
      synchronized (REPORT_LOCK) {
        if (runToken.get() != token) {
          return;
        }
        alreadyWritten = reportWritten;
        if (!alreadyWritten) {
          status = "timeout";
          failureReason = "worker wall-clock timeout exceeded";
        }
      }
      if (alreadyWritten || finished.get() || runToken.get() != token) {
        return;
      }
      try {
        writeReport(outDir);
      } catch (Exception e) {
        System.err.println("failed to write timeout report: " + e);
      }
      if (reportRefused) {
        System.err.println(refusalMessage(outDir));
      }
      exitHook.accept(EXIT_TIMEOUT);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  private static Map<String, Object> requestRecord(
      Validated v,
      DecoderModel model,
      HfTokenizer tokenizer,
      double primeS,
      int index,
      String kind,
      int emitted) {
    Map<String, Object> record = new LinkedHashMap<>();
    record.put("index", index);
    record.put("kind", kind);
    record.put("prime_s", primeS);
    record.put("prime_excluded", true);
    if ("fixed-continuation".equals(v.mode())) {
      BaselineAdapters.FixedContinuationResult result =
          BaselineAdapters.runFixedContinuation(
              model, modelScope(), promptArray(v.prompt()), fedArray(v.fedIds()));
      List<Map<String, Object>> steps = new ArrayList<>();
      for (BaselineAdapters.Step step : result.steps()) {
        Map<String, Object> s = new LinkedHashMap<>();
        s.put("position", step.position());
        s.put("s", step.s());
        s.put("argmax_id", step.argmaxId());
        s.put("top2_margin", step.top2Margin());
        steps.add(s);
      }
      record.put("ttft_s", null);
      record.put("prefill_s", result.prefillS());
      record.put("steps", steps);
      record.put("event_s", List.of());
      record.put("tokens", List.of());
      record.put("streaming_decode_tps", null);
      // Step region only (prefill excluded): it includes the per-step argmax/sort host work,
      // so it is not decode time (findings section 13.3).
      record.put("step_region_s", result.stepRegionS());
      record.put("full_generation_s", null);
      record.put("emitted", 0);
      record.put("finish_reason", "fixed-continuation");
      record.put("unused_lookahead_forwards", 0);
      record.put("memory", null);
      record.put("io_affected", v.ioAffected());
      return record;
    }
    if ("controlled".equals(v.adapter())) {
      BaselineAdapters.ControlledResult result =
          BaselineAdapters.runControlled(model, modelScope(), promptArray(v.prompt()), emitted);
      List<Map<String, Object>> steps = new ArrayList<>();
      for (BaselineAdapters.Step step : result.steps()) {
        Map<String, Object> s = new LinkedHashMap<>();
        s.put("position", step.position());
        s.put("s", step.s());
        steps.add(s);
      }
      int n = result.tokens().size();
      Double streaming = n >= 2 ? (n - 1) / (result.eventS()[n - 1] - result.eventS()[0]) : null;
      record.put("ttft_s", result.ttftS());
      record.put("prefill_s", result.prefillS());
      record.put("steps", steps);
      record.put("event_s", result.eventS());
      record.put("tokens", result.tokens());
      record.put("streaming_decode_tps", streaming);
      record.put("step_region_s", null);
      record.put("full_generation_s", result.eventS()[n - 1]);
      record.put("emitted", n);
      record.put("finish_reason", "controlled-max-tokens");
      record.put("unused_lookahead_forwards", 0);
      record.put("memory", null);
      record.put("io_affected", v.ioAffected());
      return record;
    }
    BaselineAdapters.StockResult result =
        BaselineAdapters.runStock(
            model, tokenizer, promptArray(v.prompt()), emitted, new TreeSet<>(v.stopTokenIds()));
    int n = result.tokens().size();
    Double streaming = n >= 2 ? (n - 1) / (result.eventS()[n - 1] - result.eventS()[0]) : null;
    record.put("ttft_s", n > 0 ? result.ttftS() : null);
    record.put("prefill_s", null);
    record.put("steps", null);
    record.put("event_s", result.eventS());
    record.put("tokens", result.tokens());
    record.put("streaming_decode_tps", streaming);
    record.put("step_region_s", null);
    record.put("full_generation_s", result.fullGenerationS());
    record.put("emitted", n);
    record.put("finish_reason", finishReasonName(result.finishReason()));
    // The synchronous Java loop evaluates every step at its selection boundary; the one
    // queued lookahead forward is a property of the pinned Python stock loop, not this one.
    record.put("unused_lookahead_forwards", 0);
    record.put("memory", null);
    record.put("io_affected", v.ioAffected());
    return record;
  }

  private static int[] promptArray(List<Integer> ids) {
    int[] out = new int[ids.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = ids.get(i);
    }
    return out;
  }

  private static int[] fedArray(List<Integer> ids) {
    return ids == null ? new int[0] : promptArray(ids);
  }

  private static MLXScope modelScope() {
    return modelScope;
  }

  private static String finishReasonName(FinishReason reason) {
    return switch (reason) {
      case EOS, STOP_TOKEN -> "stop";
      case MAX_TOKENS -> "length";
      case CANCELLED -> "cancelled";
    };
  }

  /**
   * The stock stop-set assertion (findings section 13.4 item 4): the artifact-resolved set — the
   * tokenizer's special-token eos id (when it has one) plus the {@code generation_config.json} and
   * {@code config.json} {@code eos_token_id} values — must equal the manifest frozen set, so a
   * future drift fails the worker instead of changing behavior silently. The tokenizer's own eos is
   * part of the resolved set because {@code HfTokenizer} is the stop set the stock path actually
   * feeds the generation loop.
   */
  private static void assertStockStopSet(
      Path artifactDir, HfTokenizer tokenizer, String caseName, List<Integer> frozen) {
    Set<Integer> resolved = new TreeSet<>();
    JsonNode generationConfig = readJsonOrNull(artifactDir.resolve("generation_config.json"));
    JsonNode config = readJsonOrNull(artifactDir.resolve("config.json"));
    collectEos(generationConfig, resolved);
    collectEos(config, resolved);
    OptionalInt tokenizerEos = tokenizer.eosTokenId();
    if (tokenizerEos.isPresent()) {
      resolved.add(tokenizerEos.getAsInt());
    }
    if (!resolved.equals(new TreeSet<>(frozen))) {
      throw new WorkerError(
          "resolved stop set "
              + resolved
              + " differs from the manifest frozen set "
              + frozen
              + " for case "
              + caseName);
    }
  }

  private static void collectEos(JsonNode node, Set<Integer> into) {
    if (node == null) {
      return;
    }
    JsonNode eos = node.path("eos_token_id");
    if (eos.isInt()) {
      into.add(eos.asInt());
    } else if (eos.isArray()) {
      eos.forEach(e -> into.add(e.asInt()));
    }
  }

  private static JsonNode readJsonOrNull(Path path) {
    if (!Files.isRegularFile(path)) {
      return null;
    }
    try {
      return MAPPER.readTree(path.toFile());
    } catch (JacksonException e) {
      throw new WorkerError("unreadable checkpoint metadata " + path + ": " + e.getMessage());
    }
  }

  /**
   * Job validation (section 13.1), before any model work. The manifest anchors are derived from the
   * worker's own location (test-overridable), never from the job's own words.
   */
  private static Validated validate(JsonNode job) {
    require(
        JOB_SCHEMA.equals(job.path("schema").asString(null)), "job schema must be " + JOB_SCHEMA);
    String adapter = job.path("adapter").asString(null);
    require(
        "controlled".equals(adapter) || "stock".equals(adapter),
        "adapter must be controlled or stock");
    String mode = job.path("mode").asString(null);
    require(
        "headline".equals(mode)
            || "fixed-continuation".equals(mode)
            || "reference-pair".equals(mode),
        "mode must be headline, fixed-continuation or reference-pair");
    String caseName = job.path("case").asString(null);
    require(caseName != null && !caseName.isEmpty(), "case is required");
    Integer promptLength = intOrNull(job.path("prompt_length"));
    require(promptLength != null && promptLength > 0, "prompt_length is required");
    String precisionEnv = job.path("precision_env").asString(null);
    require(
        "1".equals(precisionEnv) || "0".equals(precisionEnv), "precision_env must be '1' or '0'");
    Integer timeoutS = intOrNull(job.path("timeout_s"));
    require(timeoutS != null && timeoutS > 0, "timeout_s must be a positive integer");
    Map<String, Path> paths = new LinkedHashMap<>();
    for (String key : new String[] {"artifact_dir", "manifest_path", "repo_dir", "out_dir"}) {
      String value = job.path(key).asString(null);
      require(value != null && !value.isEmpty(), key + " is required");
      paths.put(key, Path.of(value));
    }
    String manifestSha = job.path("manifest_sha256").asString(null);
    require(manifestSha != null && !manifestSha.isEmpty(), "manifest_sha256 is required");

    Anchors anchors = currentAnchors();
    Path manifestPath = paths.get("manifest_path");
    // Anchored on this worker's own location, not the job's repo_dir: a hand-written job
    // cannot point at a copy of the manifest, however self-consistent its digest.
    require(
        manifestPath
            .toAbsolutePath()
            .normalize()
            .equals(anchors.canonicalManifest().toAbsolutePath().normalize()),
        "manifest_path must be the canonical frozen manifest " + anchors.canonicalManifest());
    require(Files.isRegularFile(manifestPath), "manifest path does not exist");
    String manifestActual = sha256File(manifestPath);
    require(manifestActual.equals(manifestSha), "manifest SHA-256 mismatch (frozen contract)");
    JsonNode man = readJson(manifestPath);

    JsonNode textCases = man.path("text_cases");
    JsonNode vlmCases = man.path("vlm_cases");
    JsonNode caseNode = null;
    boolean isVlm = false;
    for (JsonNode c : textCases) {
      if (caseName.equals(c.path("name").asString(null))) {
        caseNode = c;
      }
    }
    for (JsonNode c : vlmCases) {
      if (caseName.equals(c.path("name").asString(null))) {
        isVlm = true;
      }
    }
    require(caseNode != null, "case '" + caseName + "' is not a text case (VLM cases are WP4)");
    require(!isVlm, "VLM cases are out of scope for WP1 text workers");
    JsonNode lengths = man.path("controlled_generation").path("prompt_lengths");
    boolean frozenLength = false;
    for (JsonNode l : lengths) {
      if (l.asInt() == promptLength) {
        frozenLength = true;
      }
    }
    require(frozenLength, "prompt_length " + promptLength + " is not a frozen manifest length");

    JsonNode policy = man.path("run_policy");
    require(policy.isObject(), "manifest run_policy block is missing");
    JsonNode requestsNode = job.path("requests");
    require(requestsNode.isObject(), "requests block is required");
    Map<String, Integer> requests = new LinkedHashMap<>();
    for (String key : new String[] {"first", "warmups", "measured"}) {
      Integer value = intOrNull(requestsNode.path(key));
      require(value != null && value >= 0, "requests." + key + " must be a nonnegative integer");
      requests.put(key, value);
    }
    if ("reference-pair".equals(mode)) {
      // Observed-cache reference pair (findings section 7.3): no priming, one measured
      // request, the separate shorter timeout.
      require(
          requests.get("first") == 0
              && requests.get("warmups") == 0
              && requests.get("measured") == 1,
          "reference-pair must be exactly one measured request with no priming");
    } else {
      require(
          requests.get("first").equals(intOrNull(policy.path("first_requests"))),
          "requests.first differs from run_policy.first_requests");
      require(
          requests.get("warmups").equals(intOrNull(policy.path("warmups"))),
          "requests.warmups differs from run_policy");
      require(
          requests.get("measured").equals(intOrNull(policy.path("measured_requests"))),
          "requests.measured differs from run_policy");
    }
    int emitted = intOrNull(policy.path("emitted_tokens"));
    require(
        emitted == man.path("controlled_generation").path("max_new_tokens").asInt(-1),
        "run_policy.emitted_tokens differs from controlled_generation.max_new_tokens");
    Integer expectedTimeout =
        intOrNull(
            "reference-pair".equals(mode)
                ? policy.path("reference_pair_worker_timeout_s")
                : policy.path("worker_timeout_s"));
    require(
        expectedTimeout != null && expectedTimeout > 0 && timeoutS.equals(expectedTimeout),
        "timeout_s differs from the manifest run_policy value");
    boolean ioAffected = false;
    JsonNode ioCells = policy.path("io_affected_cells");
    for (JsonNode cell : ioCells) {
      if (cell.asString("").equals(caseName + "/" + promptLength)) {
        ioAffected = true;
      }
    }

    Path artifactDir = paths.get("artifact_dir");
    require(Files.isDirectory(artifactDir), "artifact_dir does not exist");
    for (Map.Entry<String, JsonNode> entry : caseNode.path("files").properties()) {
      Path file = artifactDir.resolve(entry.getKey());
      require(Files.isRegularFile(file), "checkpoint file missing: " + entry.getKey());
      String actual = sha256File(file);
      require(
          actual.equals(entry.getValue().path("sha256").asString(null)),
          "checkpoint digest mismatch for " + entry.getKey());
    }
    // The safetensors set must equal the pinned set exactly: an unpinned weight file would
    // be primed and loaded without verification.
    Set<String> pinnedSt = new TreeSet<>();
    for (String name : caseNode.path("files").propertyNames()) {
      if (name.endsWith(".safetensors")) {
        pinnedSt.add(name);
      }
    }
    Set<String> presentSt = new TreeSet<>();
    try (var stream = Files.list(artifactDir)) {
      for (Path p :
          stream.filter(f -> f.getFileName().toString().endsWith(".safetensors")).toList()) {
        presentSt.add(p.getFileName().toString());
      }
    } catch (IOException e) {
      throw new WorkerError("unreadable artifact_dir: " + e.getMessage());
    }
    require(
        presentSt.equals(pinnedSt),
        "artifact_dir safetensors set differs from the pinned set (extra="
            + new TreeSet<>(setDiff(presentSt, pinnedSt))
            + ", missing="
            + new TreeSet<>(setDiff(pinnedSt, presentSt))
            + "); unpinned weight files would be primed and loaded unverified");

    Path repoDir = paths.get("repo_dir");
    // Anchored on this worker's own location, not the job's word: the report's code
    // identity must name the checkout this worker actually ran from.
    require(
        repoDir
            .toAbsolutePath()
            .normalize()
            .equals(anchors.canonicalRepo().toAbsolutePath().normalize()),
        "repo_dir must be the repository this worker runs from: " + anchors.canonicalRepo());
    Path outDir = paths.get("out_dir");
    require(
        !Files.exists(outDir.resolve("report.json")),
        "out_dir already contains a report (overwrite rejected)");

    JsonNode promptEntry =
        caseNode.path("prompts").path("per_length").path(String.valueOf(promptLength));
    JsonNode ids = promptEntry.path("ids");
    require(
        promptEntry.isObject() && ids.isArray(), "frozen prompt entry must carry ids and sha256");
    List<Integer> prompt = new ArrayList<>();
    ids.forEach(i -> prompt.add(i.asInt()));
    require(prompt.size() == promptLength, "frozen prompt length mismatch");
    StringBuilder canonical = new StringBuilder("[");
    for (int i = 0; i < prompt.size(); i++) {
      if (i > 0) {
        canonical.append(',');
      }
      canonical.append(prompt.get(i));
    }
    canonical.append(']');
    require(
        sha256Hex(canonical.toString().getBytes(StandardCharsets.UTF_8))
            .equals(promptEntry.path("sha256").asString(null)),
        "frozen prompt canonical SHA-256 mismatch");

    List<Integer> fedIds = null;
    if ("fixed-continuation".equals(mode)) {
      JsonNode cont = job.path("fixed_continuation");
      require(cont.isObject(), "fixed_continuation spec is required in this mode");
      List<Integer> base = new ArrayList<>();
      caseNode
          .path("prompts")
          .path("per_length")
          .path("512")
          .path("ids")
          .forEach(i -> base.add(i.asInt()));
      Integer offset = intOrNull(cont.path("offset"));
      Integer count = intOrNull(cont.path("count"));
      require(
          offset != null && count != null && offset == 32 && count == 32 && base.size() >= 64,
          "fixed_continuation must be prefix512[32:64] (manifest prompt_generator rule)");
      fedIds = List.copyOf(base.subList(offset, offset + count));
    }

    Map<String, Object> runPolicy = new LinkedHashMap<>();
    runPolicy.put("worker_timeout_s", intOrNull(policy.path("worker_timeout_s")));
    runPolicy.put(
        "reference_pair_worker_timeout_s",
        intOrNull(policy.path("reference_pair_worker_timeout_s")));
    runPolicy.put("cell_time_budget_s", intOrNull(policy.path("cell_time_budget_s")));
    runPolicy.put("session_time_budget_s", intOrNull(policy.path("session_time_budget_s")));
    runPolicy.put("first_requests", intOrNull(policy.path("first_requests")));
    runPolicy.put("warmups", intOrNull(policy.path("warmups")));
    runPolicy.put("measured_requests", intOrNull(policy.path("measured_requests")));
    runPolicy.put("emitted_tokens", intOrNull(policy.path("emitted_tokens")));
    runPolicy.put("cache_regime", policy.path("cache_regime").asString(null));
    List<Integer> stopTokenIds = new ArrayList<>();
    caseNode.path("stop_token_ids").forEach(i -> stopTokenIds.add(i.asInt()));
    stopTokenIds.sort(Comparator.naturalOrder());
    return new Validated(
        adapter,
        caseName,
        promptLength,
        mode,
        requests,
        timeoutS,
        precisionEnv,
        artifactDir,
        manifestPath,
        manifestActual,
        repoDir,
        outDir,
        prompt,
        fedIds,
        ioAffected,
        runPolicy,
        "controlled".equals(adapter)
            ? man.path("controlled_generation").path("wired_limit").asString("")
            : man.path("stock_generation").path("wired_limit").asString(""),
        stopTokenIds);
  }

  private static Anchors currentAnchors() {
    if (anchors != null) {
      return anchors;
    }
    Path classDir;
    try {
      classDir =
          Path.of(BaselineWorker.class.getResource("BaselineWorker.class").toURI())
              .toAbsolutePath()
              .normalize();
    } catch (URISyntaxException e) {
      throw new WorkerError("cannot anchor the worker from its class location: " + e);
    }
    Path repoRoot = classDir;
    while (repoRoot != null
        && !Files.isDirectory(repoRoot.resolve(".git"))
        && !Files.isRegularFile(repoRoot.resolve("settings.gradle"))) {
      repoRoot = repoRoot.getParent();
    }
    if (repoRoot == null) {
      throw new WorkerError("cannot anchor the worker to a repository root from " + classDir);
    }
    return new Anchors(repoRoot.resolve("tools/decode-baseline/workloads.json"), repoRoot);
  }

  private static Map<String, Object> repoIdentity(Path repoDir) {
    String sha = git(repoDir, "rev-parse", "HEAD");
    // A non-git anchor (the in-process protocol tests) degrades gracefully; a production
    // checkout always resolves.
    if (sha == null) {
      return Map.of("sha", "", "branch", "", "dirty", true);
    }
    String branch = git(repoDir, "rev-parse", "--abbrev-ref", "HEAD");
    String status = git(repoDir, "status", "--porcelain");
    boolean dirty = status != null && !status.isEmpty();
    return Map.of("sha", sha, "branch", branch, "dirty", dirty);
  }

  private static String git(Path dir, String... args) {
    try {
      List<String> command = new ArrayList<>(List.of("git", "-C", dir.toString()));
      command.addAll(Arrays.asList(args));
      Process process = new ProcessBuilder(command).redirectErrorStream(false).start();
      String out = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
      if (process.waitFor() != 0) {
        return null;
      }
      return out.strip();
    } catch (Exception e) {
      return null;
    }
  }

  /** One sequential full read of every checkpoint safetensors file (section 13.2 priming). */
  private static double prime(Path artifactDir) {
    long t0 = System.nanoTime();
    List<Path> files;
    try (var stream = Files.list(artifactDir)) {
      files =
          stream
              .filter(f -> f.getFileName().toString().endsWith(".safetensors"))
              .sorted(Comparator.comparing(p -> p.getFileName().toString()))
              .toList();
    } catch (IOException e) {
      throw new WorkerError("unreadable artifact_dir: " + e.getMessage());
    }
    for (Path path : files) {
      try (var in = Files.newInputStream(path)) {
        byte[] chunk = new byte[(int) PRIME_CHUNK];
        while (in.read(chunk) >= 0) {
          // sequential full read
        }
      } catch (IOException e) {
        throw new WorkerError("prime read failed for " + path + ": " + e.getMessage());
      }
    }
    return (System.nanoTime() - t0) / 1_000_000_000d;
  }

  /**
   * Writes the report atomically, at most once, and never over an existing report.json.
   *
   * <p>Returns true iff this call produced report.json; false when the report was already written
   * (by this process) or an existing report was detected. The per-process temp name keeps a
   * concurrent worker in the same directory from truncating the write mid-flight, and a write
   * failure removes the temp file before propagating.
   *
   * <p>Java has no hard-link primitive, and {@code ATOMIC_MOVE} is a POSIX {@code rename()} that
   * <em>replaces</em> an existing target without error — so the no-overwrite guarantee the pinned
   * Python worker gets for free from {@code os.link} (which fails on an existing path) must be
   * checked here: under {@link #REPORT_LOCK}, if report.json already exists the write is refused
   * and the existing bytes are left untouched. The exclusive {@code worker-claim} means only the
   * claimant writes into a directory, so the only pre-existing report this can meet is one seeded
   * by the launcher or a previous run — exactly the case the refusal exists for.
   */
  static boolean writeReport(Path dir) {
    synchronized (REPORT_LOCK) {
      reportRefused = false;
      if (reportWritten) {
        return false;
      }
      reportWritten = true;
      utcEnd = utcNow();
      Map<String, Object> worker = new LinkedHashMap<>();
      worker.put("runtime", "java");
      worker.put("pid", ProcessHandle.current().pid());
      worker.put("adapter", adapter);
      worker.put("case", caseName);
      worker.put("prompt_length", promptLength);
      worker.put("mode", mode);
      worker.put("status", status);
      worker.put("failure_reason", failureReason);
      worker.put("utc_start", utcStart);
      worker.put("utc_end", utcEnd);
      worker.put("identity", identity);
      worker.put("stop_token_ids", stopTokenIds);
      worker.put("load_s", loadS);
      worker.put("load_state", loadState);
      worker.put("requests", List.copyOf(requests));
      worker.put("notes", List.copyOf(notes));
      Map<String, Object> report = new LinkedHashMap<>();
      report.put("schema", REPORT_SCHEMA);
      report.put("worker", worker);
      ObjectNode node = MAPPER.valueToTree(report);
      Path finalPath = dir.resolve("report.json");
      Path tmp = dir.resolve("report.json.tmp." + ProcessHandle.current().pid());
      try {
        Files.write(tmp, MAPPER.writeValueAsBytes(node));
        if (Files.exists(finalPath)) {
          reportRefused = true;
          return false;
        }
        Files.move(tmp, finalPath, StandardCopyOption.ATOMIC_MOVE);
        return true;
      } catch (IOException e) {
        throw new UncheckedIOException("report write failed for " + finalPath, e);
      } finally {
        try {
          Files.deleteIfExists(tmp);
        } catch (IOException ignored) {
          // the move consumed the temp file, or it never existed
        }
      }
    }
  }

  /**
   * The stderr message for a refused report write, built from the final state after the success
   * downgrade, so it never reports an ok status the run was downgraded away from.
   */
  static String refusalMessage(Path dir) {
    String detail =
        failureReason == null
            ? "with no failure reason"
            : "with failure_reason '" + failureReason + "'";
    return "run ended as "
        + status
        + " "
        + detail
        + " but a report already exists at "
        + dir.resolve("report.json")
        + "; leaving it untouched";
  }

  private static int handleValidationFailure(WorkerError e) {
    if (utcStart == null) {
      utcStart = utcNow();
    }
    status = "failed";
    failureReason = e.getMessage();
    if (outDir != null && Files.isDirectory(outDir)) {
      writeReport(outDir);
      if (reportRefused) {
        System.err.println(refusalMessage(outDir));
      }
    } else {
      System.err.println("worker validation failed: " + e.getMessage());
    }
    return EXIT_FAILED;
  }

  private static int handleUnexpected(Exception e) {
    if (utcStart == null) {
      utcStart = utcNow();
    }
    status = "failed";
    failureReason = "unexpected: " + e.getClass().getSimpleName() + ": " + e.getMessage();
    if (outDir != null) {
      try {
        writeReport(outDir);
        if (reportRefused) {
          System.err.println(refusalMessage(outDir));
        }
      } catch (RuntimeException ignored) {
        // leave whatever the directory already holds untouched
      }
    }
    System.err.println("worker failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
    return EXIT_FAILED;
  }

  private static ObjectNode claimJson() {
    ObjectNode node = MAPPER.createObjectNode();
    node.put("utc", utcNow());
    node.put("pid", ProcessHandle.current().pid());
    return node;
  }

  private static ObjectNode readyJson(String adapter, String caseName) {
    ObjectNode node = MAPPER.createObjectNode();
    node.put("utc", utcNow());
    node.put("pid", ProcessHandle.current().pid());
    node.put("adapter", adapter);
    node.put("case", caseName);
    return node;
  }

  private static String utcNow() {
    return Instant.now().toString();
  }

  private static void require(boolean condition, String reason) {
    if (!condition) {
      throw new WorkerError(reason);
    }
  }

  private static Integer intOrNull(JsonNode node) {
    return node.isInt() ? node.asInt() : null;
  }

  private static Set<String> setDiff(Set<String> a, Set<String> b) {
    Set<String> diff = new LinkedHashSet<>(a);
    diff.removeAll(b);
    return diff;
  }

  private static String hostString(String key, String fallback) {
    try {
      Process process = new ProcessBuilder("sysctl", "-n", key).start();
      String out =
          new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
      return process.waitFor() == 0 ? out : fallback;
    } catch (Exception e) {
      return fallback;
    }
  }

  private static long hostLong(String key, long fallback) {
    String value = hostString(key, "");
    try {
      return value.isEmpty() ? fallback : Long.parseLong(value);
    } catch (NumberFormatException e) {
      return fallback;
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
      throw new WorkerError("sha256 failed for " + path + ": " + e.getMessage());
    }
  }

  private static String sha256Hex(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException e) {
      throw new WorkerError("sha256 failed: " + e.getMessage());
    }
  }

  private static JsonNode readJson(Path path) {
    try {
      JsonNode node = MAPPER.readTree(path.toFile());
      if (node == null || !node.isObject()) {
        throw new WorkerError("manifest is not a JSON object: " + path);
      }
      return node;
    } catch (JacksonException e) {
      throw new WorkerError("unreadable manifest " + path + ": " + e.getMessage());
    }
  }

  /** Validation and contract failure. */
  static final class WorkerError extends RuntimeException {
    WorkerError(String message) {
      super(message);
    }
  }

  // --- Test-only seams. The launcher never sets any of these. ---

  static void setAnchorsForTest(Path canonicalManifest, Path canonicalRepo) {
    anchors = new Anchors(canonicalManifest, canonicalRepo);
  }

  static void clearAnchorsForTest() {
    anchors = null;
  }

  static void setExitHookForTest(IntConsumer hook) {
    exitHook = hook;
  }

  static void setWatchdogGraceMsForTest(long graceMs) {
    watchdogGraceMs = graceMs;
  }

  static void setRequestDelayMsForTest(long delayMs) {
    requestDelayMs = delayMs;
  }

  static void resetTestSeams() {
    anchors = null;
    exitHook = System::exit;
    watchdogGraceMs = DEFAULT_WATCHDOG_GRACE_MS;
    requestDelayMs = 0L;
  }
}
