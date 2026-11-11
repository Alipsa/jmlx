package se.alipsa.jmlx.benchmarks;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.models.CancellationToken;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationRequest;
import se.alipsa.jmlx.models.GenerationResult;
import se.alipsa.jmlx.models.TextGenerationModel;
import se.alipsa.jmlx.models.TextGenerationModels;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Phase 8.2 WP4 VLM first-inference worker (req/plans/phase8-2-plan.md section 5.2; frozen contract
 * in req/plans/phase8-2-reference-findings.md section 17). Runs in a fresh JVM forked by {@link
 * VlmFirstUseBenchmark} from exactly one tree (the current checkout or the isolated old-pin
 * compatibility worktree), one workload case per process.
 *
 * <p>Sequence: handshake claim, staged-native-pin and environment verification, request
 * construction from the digest-verified Tier-B manifest (same image/prompt/config bytes and
 * generation policy on both pins), model load, the first image+text generation (the first-use
 * measurement, captured before any warmup or probe), correctness verification of the first
 * request's greedy output against the manifest's recorded reference, then the declared discarded
 * warmups and measured warm repetitions of the same request. All intervals use the monotonic clock;
 * UTC stamps bracket the run for host-record correlation.
 *
 * <p>The source file is byte-identical between the two trees (it references only the stable public
 * API shared by both); the measured-code record is the tree SHA plus this file's digest, not the
 * file's location. Exit codes: 0 ok, 1 job verification failure, 2 runtime failure, 3 correctness
 * failure, 4 internal backstop timeout.
 *
 * <p>Opt-in local tooling, not part of {@code check} or the publication surface.
 */
public final class VlmFirstUseWorker {
  /** Report/claim protocol schema. */
  public static final String REPORT_SCHEMA = "jmlx-vlm-first-use-1";

  public static final String JOB_SCHEMA = "jmlx-vlm-first-use-job-1";

  public static final int EXIT_OK = 0;
  public static final int EXIT_JOB_VERIFICATION_FAILED = 1;
  public static final int EXIT_RUNTIME_FAILED = 2;
  public static final int EXIT_CORRECTNESS_FAILED = 3;
  public static final int EXIT_TIMEOUT = 4;

  /** Backstop grace past the declared worker timeout before the internal timer fires. */
  private static final long BACKSTOP_GRACE_S = 30L;

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private VlmFirstUseWorker() {}

  /** Runs {@code VlmFirstUseWorker <job.json>} and exits with the worker exit code. */
  public static void main(String[] args) {
    int code;
    try {
      code = run(args);
    } catch (Exception e) {
      System.err.println("vlm-first-use worker failed: " + e);
      code = EXIT_RUNTIME_FAILED;
    }
    System.exit(code);
  }

  private static int run(String[] args) throws Exception {
    if (args.length != 1) {
      System.err.println("usage: VlmFirstUseWorker <job.json>");
      return EXIT_JOB_VERIFICATION_FAILED;
    }
    Path jobPath = Path.of(args[0]).toAbsolutePath();
    if (!Files.isRegularFile(jobPath)) {
      System.err.println("job file not found: " + jobPath);
      return EXIT_JOB_VERIFICATION_FAILED;
    }
    Job job;
    try {
      job = Job.parse(MAPPER.readTree(jobPath));
    } catch (RuntimeException e) {
      System.err.println("job verification failed: " + e.getMessage());
      return EXIT_JOB_VERIFICATION_FAILED;
    }
    return new Run(job).execute();
  }

  /** The validated job record (launcher -> worker). */
  record Job(
      String workload,
      String tree,
      Path treePath,
      int pairIndex,
      Path modelDir,
      Path manifestPath,
      String manifestSha256,
      List<Path> imagePaths,
      List<String> imageSha256s,
      int maxNewTokens,
      List<Integer> stopTokenIds,
      int warmups,
      int measured,
      long workerTimeoutS,
      String precisionEnv,
      String precisionValue,
      String forkUtc,
      Path outDir) {

    static Job parse(JsonNode root) {
      int pairIndex = root.required("pair_index").asInt();
      int maxNewTokens = root.required("max_new_tokens").asInt();
      if (pairIndex < 0 || maxNewTokens <= 0) {
        throw new IllegalArgumentException(
            "pair_index must be nonnegative and max_new_tokens positive: " + root);
      }
      int warmups = root.required("warmups").asInt();
      int measured = root.required("measured").asInt();
      if (warmups < 0 || measured <= 0) {
        throw new IllegalArgumentException(
            "warmups must be nonnegative and measured positive: " + root);
      }
      long workerTimeoutS = root.required("worker_timeout_s").asLong();
      if (workerTimeoutS <= 0) {
        throw new IllegalArgumentException("worker_timeout_s must be positive: " + root);
      }
      List<Path> imagePaths = new ArrayList<>();
      for (JsonNode p : root.required("image_paths")) {
        imagePaths.add(Path.of(p.asString()).toAbsolutePath());
      }
      List<String> imageSha256s = new ArrayList<>();
      for (JsonNode h : root.required("image_sha256s")) {
        imageSha256s.add(h.asString());
      }
      if (imagePaths.size() != imageSha256s.size() || imagePaths.isEmpty()) {
        throw new IllegalArgumentException("image paths and digests must pair up and be nonempty");
      }
      List<Integer> stopTokenIds = new ArrayList<>();
      for (JsonNode t : root.required("stop_token_ids")) {
        stopTokenIds.add(t.asInt());
      }
      if (stopTokenIds.isEmpty()) {
        throw new IllegalArgumentException("stop_token_ids must be nonempty (explicitly recorded)");
      }
      return new Job(
          root.required("workload").asString(),
          root.required("tree").asString(),
          Path.of(root.required("tree_path").asString()),
          pairIndex,
          Path.of(root.required("model_dir").asString()).toAbsolutePath(),
          Path.of(root.required("manifest_path").asString()).toAbsolutePath(),
          root.required("manifest_sha256").asString(),
          List.copyOf(imagePaths),
          List.copyOf(imageSha256s),
          maxNewTokens,
          List.copyOf(stopTokenIds),
          warmups,
          measured,
          workerTimeoutS,
          root.required("precision_env").asString(),
          root.required("precision_value").asString(),
          root.required("fork_utc").asString(),
          Path.of(root.required("out_dir").asString()).toAbsolutePath());
    }
  }

  /** One worker execution; owns the claim/report protocol and the backstop timer. */
  private static final class Run {
    private final Job job;
    private final long baseNanos = System.nanoTime();
    private final Instant utcStart = Instant.now();
    private final ScheduledExecutorService backstop = Executors.newSingleThreadScheduledExecutor();
    private final List<Double> tokenEventTimesS = new ArrayList<>();
    private long firstTokenTimeNanos;
    private volatile boolean backstopFired;
    private Map<String, String> stagedPin;
    private ManifestCase reference;

    private Run(Job job) {
      this.job = job;
    }

    int execute() {
      try {
        return executeInner();
      } catch (Exception e) {
        // A failure after the claim still leaves a report (the launcher classifies dead/crashed
        // only for workers that left neither artifact).
        finish("failed", String.valueOf(e), EXIT_RUNTIME_FAILED);
        return EXIT_RUNTIME_FAILED;
      }
    }

    private int executeInner() throws Exception {
      // Environment and staged-pin verification before any measurement.
      String actualEnv = System.getenv(job.precisionEnv);
      if (!job.precisionValue.equals(actualEnv)) {
        throw new IllegalStateException(
            job.precisionEnv
                + " must be "
                + job.precisionValue
                + " (declared precision), got '"
                + actualEnv
                + "'");
      }
      stagedPin = readStagedPinProperties();
      for (int i = 0; i < job.imagePaths.size(); i++) {
        String sha = sha256Hex(job.imagePaths.get(i));
        if (!sha.equals(job.imageSha256s.get(i))) {
          throw new IllegalStateException(
              "image digest mismatch ("
                  + job.imagePaths.get(i)
                  + "): "
                  + sha
                  + " != "
                  + job.imageSha256s.get(i));
        }
      }
      reference = loadManifest();

      // Handshake claim, then the internal backstop.
      ObjectNode claimNode = MAPPER.createObjectNode();
      claimNode.put("schema", REPORT_SCHEMA);
      claimNode.put("status", "started");
      claimNode.put("utc", utcStart.toString());
      claimNode.put("pair_index", job.pairIndex);
      Path claim = job.outDir.resolve("claim.json");
      Files.write(claim, claimNode.toString().getBytes(StandardCharsets.UTF_8));
      backstop.schedule(
          () -> {
            backstopFired = true;
            System.err.println("vlm-first-use worker backstop fired");
            finish(
                "timeout",
                "internal backstop elapsed " + (job.workerTimeoutS + BACKSTOP_GRACE_S) + " s",
                EXIT_TIMEOUT);
            System.exit(EXIT_TIMEOUT);
          },
          job.workerTimeoutS + BACKSTOP_GRACE_S,
          TimeUnit.SECONDS);

      List<RgbImage> images = new ArrayList<>();
      for (Path image : job.imagePaths) {
        images.add(ImageDecoder.decode(image));
      }
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(job.modelDir);
      GenerationRequest request = buildRequest(tokenizer, images);
      verifyPromptAgainstManifest(tokenizer);

      FirstCapture first = new FirstCapture();
      List<Double> measuredTtft = new ArrayList<>();
      List<Double> measuredFull = new ArrayList<>();
      List<Boolean> measuredIdsMatch = new ArrayList<>();
      long loadNanos;
      Map<String, Map<String, Long>> memory = new LinkedHashMap<>();
      try (MLXScope root = new MLXScope()) {
        long loadStart = System.nanoTime();
        TextGenerationModel model = TextGenerationModels.load(root, job.modelDir);
        loadNanos = System.nanoTime() - loadStart;
        memory.put("after_load", memorySample());

        generate(
            model,
            request,
            true,
            (ids, finish, ttft, full, positions) -> {
              first.ids = ids;
              first.finish = finish;
              first.ttftNanos = ttft;
              first.fullNanos = full;
              first.positions = positions;
              first.captured = true;
            });
        if (!first.captured) {
          throw new IllegalStateException("first generation emitted no completion");
        }
        memory.put("after_first", memorySample());

        List<Integer> expected = first.ids;
        for (int i = 0; i < job.warmups; i++) {
          final int warmupIndex = i;
          generate(
              model,
              request,
              false,
              (ids, finish, ttft, full, positions) ->
                  verifyWarmIds(ids, expected, "warmup " + warmupIndex));
        }
        for (int i = 0; i < job.measured; i++) {
          final int measuredIndex = i;
          generate(
              model,
              request,
              false,
              (ids, finish, ttft, full, positions) -> {
                verifyWarmIds(ids, expected, "measured " + measuredIndex);
                measuredTtft.add(seconds(ttft));
                measuredFull.add(seconds(full));
                measuredIdsMatch.add(true);
              });
        }
        memory.put("after_final", memorySample());
      }

      // Correctness after capture: the first request's greedy output must equal the
      // manifest's recorded reference.
      boolean idsMatch = first.ids.equals(reference.targetIds);
      boolean finishMatch = reference.finishReason.equals(first.finish);
      boolean positionsMatch = reference.expandedIdsSize == first.positions;
      if (backstopFired) {
        return EXIT_TIMEOUT;
      }
      if (!(idsMatch && finishMatch && positionsMatch)) {
        finish(
            "correctness_failed",
            "first-request correctness mismatch: ids_match="
                + idsMatch
                + " finish_match="
                + finishMatch
                + " positions_match="
                + positionsMatch,
            EXIT_CORRECTNESS_FAILED);
      }
      finish(
          "ok",
          null,
          EXIT_OK,
          new ReportData(
              stagedPin,
              loadNanos,
              first.ttftNanos,
              first.fullNanos,
              first.ids,
              first.finish,
              first.positions,
              idsMatch,
              finishMatch,
              positionsMatch,
              measuredTtft,
              measuredFull,
              measuredIdsMatch,
              memory));
      return EXIT_OK;
    }

    /** Runs one generation and returns the TTFT/full-generation intervals in nanos. */
    private long[] generate(
        TextGenerationModel model, GenerationRequest request, boolean recordEvents, OnDone onDone) {
      firstTokenTimeNanos = -1;
      long entry = System.nanoTime();
      GenerationResult result =
          model.generate(
              request,
              event -> {
                if (event.tokenId() == null || firstTokenTimeNanos >= 0) {
                  return;
                }
                long now = System.nanoTime();
                firstTokenTimeNanos = now;
                // Only the first request records the full token event list; every generation
                // (warmups included) still stamps the first-token time, which is the TTFT.
                if (recordEvents) {
                  tokenEventTimesS.add(seconds(now - entry));
                }
              });
      long terminal = System.nanoTime();
      long ttft = firstTokenTimeNanos >= 0 ? firstTokenTimeNanos - entry : -1;
      onDone.accept(
          result.generatedTokenIds(),
          String.valueOf(result.finishReason()),
          ttft,
          terminal - entry,
          result.promptPositions());
      return new long[] {ttft, terminal - entry};
    }

    private void verifyWarmIds(List<Integer> ids, List<Integer> expected, String label) {
      if (!ids.equals(expected)) {
        throw new IllegalStateException(
            label
                + " generated IDs diverged from the first request (greedy must be deterministic)");
      }
    }

    private GenerationRequest buildRequest(HfTokenizer tokenizer, List<RgbImage> images) {
      GenerationRequest request =
          GenerationRequest.chat(
                  tokenizer,
                  reference.messages,
                  ChatTemplateOptions.defaults(true),
                  GenerationConfig.greedyDefaults(job.maxNewTokens, Set.copyOf(job.stopTokenIds)),
                  CancellationToken.NONE)
              .withImages(images);
      return request;
    }

    private void verifyPromptAgainstManifest(HfTokenizer tokenizer) {
      String rendered =
          tokenizer.renderChat(reference.messages, ChatTemplateOptions.defaults(true));
      if (!rendered.equals(reference.renderedPrompt)) {
        throw new IllegalStateException(
            "rendered prompt differs from the manifest's recorded rendered_prompt");
      }
      List<Integer> encoded = tokenizer.encode(rendered, false);
      if (!encoded.equals(reference.unexpandedIds)) {
        throw new IllegalStateException(
            "encoded prompt IDs differ from the manifest's unexpanded_ids");
      }
    }

    private ManifestCase loadManifest() throws Exception {
      String sha = sha256Hex(job.manifestPath);
      if (!sha.equals(job.manifestSha256)) {
        throw new IllegalStateException(
            "manifest digest mismatch: " + sha + " != " + job.manifestSha256);
      }
      JsonNode manifest = MAPPER.readTree(job.manifestPath);
      JsonNode reference = manifest.required("reference").required("cases").get(0);
      List<Map<String, Object>> messages = new ArrayList<>();
      for (JsonNode message : reference.required("messages")) {
        Map<String, Object> converted = new LinkedHashMap<>();
        converted.put("role", message.required("role").asString());
        JsonNode content = message.required("content");
        if (content.isTextual()) {
          converted.put("content", content.asString());
        } else {
          List<Map<String, Object>> parts = new ArrayList<>();
          for (JsonNode part : content) {
            Map<String, Object> partMap = new HashMap<>();
            for (Map.Entry<String, JsonNode> field : part.properties()) {
              JsonNode value = field.getValue();
              partMap.put(
                  field.getKey(),
                  value.isIntegralNumber()
                      ? value.asLong()
                      : value.isTextual() ? value.asString() : value);
            }
            parts.add(partMap);
          }
          converted.put("content", parts);
        }
        messages.add(converted);
      }
      List<Integer> targetIds = new ArrayList<>();
      for (JsonNode id : reference.required("target_ids")) {
        targetIds.add(id.asInt());
      }
      List<Integer> unexpandedIds = new ArrayList<>();
      for (JsonNode id : reference.required("unexpanded_ids")) {
        unexpandedIds.add(id.asInt());
      }
      return new ManifestCase(
          List.copyOf(messages),
          reference.required("rendered_prompt").asString(),
          List.copyOf(targetIds),
          List.copyOf(unexpandedIds),
          reference.required("expanded_ids").size(),
          reference.required("finish_reason").asString());
    }

    /** The manifest's recorded reference case (case 0: the frozen WP0 case per workload). */
    private record ManifestCase(
        List<Map<String, Object>> messages,
        String renderedPrompt,
        List<Integer> targetIds,
        List<Integer> unexpandedIds,
        int expandedIdsSize,
        String finishReason) {}

    /** The staged runtime's pin identity read from the worker's own library path. */
    private Map<String, String> readStagedPinProperties() throws IOException {
      String libraryPath = System.getProperty("jmlx.library.path");
      if (libraryPath == null) {
        throw new IllegalStateException("jmlx.library.path system property is not set");
      }
      Path pinFile = Path.of(libraryPath).toAbsolutePath().resolve("native-pin.properties");
      if (!Files.isRegularFile(pinFile)) {
        throw new IllegalStateException("staged pin properties missing: " + pinFile);
      }
      Map<String, String> properties = new LinkedHashMap<>();
      for (String line : Files.readAllLines(pinFile, StandardCharsets.UTF_8)) {
        int eq = line.indexOf('=');
        if (eq > 0) {
          properties.put(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
        }
      }
      if (!properties.containsKey("mlxMetalVersion") || !properties.containsKey("mlxcCommit")) {
        throw new IllegalStateException("staged pin properties incomplete: " + pinFile);
      }
      properties.put("library_path", Path.of(libraryPath).toAbsolutePath().toString());
      return properties;
    }

    private Map<String, Long> memorySample() {
      Map<String, Long> sample = new LinkedHashMap<>();
      sample.put("active_bytes", MLXMemory.activeBytes());
      sample.put("peak_bytes", MLXMemory.peakBytes());
      sample.put("cached_bytes", MLXMemory.cachedBytes());
      return sample;
    }

    private static double seconds(long nanos) {
      return nanos / 1e9;
    }

    private static String sha256Hex(Path path) throws IOException {
      try {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (var in = Files.newInputStream(path)) {
          byte[] buffer = new byte[1 << 16];
          int read;
          while ((read = in.read(buffer)) >= 0) {
            digest.update(buffer, 0, read);
          }
        }
        return HexFormat.of().formatHex(digest.digest());
      } catch (java.security.NoSuchAlgorithmException e) {
        throw new IllegalStateException("SHA-256 unavailable", e);
      }
    }

    /** Per-generation completion callback (IDs, finish reason, TTFT/full in nanos, positions). */
    @FunctionalInterface
    private interface OnDone {
      void accept(List<Integer> ids, String finish, long ttftNanos, long fullNanos, int positions);
    }

    /** Mutable capture of the first request's completion (assigned from the event lambda). */
    private static final class FirstCapture {
      List<Integer> ids;
      String finish;
      long ttftNanos;
      long fullNanos;
      int positions;
      boolean captured;
    }

    /** The report's measurement payload, populated by a successful run. */
    private record ReportData(
        Map<String, String> pin,
        long loadNanos,
        long firstTtftNanos,
        long firstFullNanos,
        List<Integer> generatedIds,
        String firstFinishReason,
        int firstPromptPositions,
        boolean idsMatch,
        boolean finishMatch,
        boolean positionsMatch,
        List<Double> measuredTtft,
        List<Double> measuredFull,
        List<Boolean> measuredIdsMatch,
        Map<String, Map<String, Long>> memory) {}

    private void finish(
        String status, String failureReason, int code, ReportData data, long wallNanos) {
      ObjectNode report = MAPPER.createObjectNode();
      report.put("schema", REPORT_SCHEMA);
      report.put("status", status);
      if (failureReason != null) {
        report.put("failure_reason", failureReason);
      }
      report.put("workload", job.workload);
      report.put("tree", job.tree);
      report.put("tree_path", job.treePath.toString());
      report.put("pair_index", job.pairIndex);
      report.put("utc_start", utcStart.toString());
      report.put("utc_end", Instant.now().toString());
      report.put("wall_s", wallNanos / 1e9);
      report.put("fork_utc", job.forkUtc);
      report.put("java_version", System.getProperty("java.version"));
      ObjectNode precision = report.putObject("precision");
      precision.put("env", job.precisionEnv);
      precision.put("value", job.precisionValue);
      if (data != null) {
        ObjectNode pin = report.putObject("native_pin");
        for (Map.Entry<String, String> entry : data.pin().entrySet()) {
          pin.put(entry.getKey(), entry.getValue());
        }
        ObjectNode workload = report.putObject("workload");
        workload.put("model_dir", job.modelDir.toString());
        workload.put("manifest_path", job.manifestPath.toString());
        workload.put("manifest_sha256", job.manifestSha256);
        ArrayNode images = workload.putArray("images");
        for (int i = 0; i < job.imagePaths.size(); i++) {
          ObjectNode image = images.addObject();
          image.put("path", job.imagePaths.get(i).toString());
          image.put("sha256", job.imageSha256s.get(i));
        }
        workload.put("max_new_tokens", job.maxNewTokens);
        ArrayNode stops = workload.putArray("stop_token_ids");
        for (int id : job.stopTokenIds) {
          stops.add(id);
        }
        ObjectNode load = report.putObject("load");
        load.put("wall_s", seconds(data.loadNanos()));
        load.put(
            "lazy",
            "load is lazy (no eval during load); weight materialization lands inside the first "
                + "forward");
        ObjectNode first = report.putObject("first_request");
        first.put("ttft_s", seconds(data.firstTtftNanos()));
        first.put("full_generation_s", seconds(data.firstFullNanos()));
        ArrayNode eventTimes = first.putArray("token_event_times_s");
        for (double t : tokenEventTimesS) {
          eventTimes.add(t);
        }
        ArrayNode ids = first.putArray("generated_ids");
        for (int id : data.generatedIds()) {
          ids.add(id);
        }
        first.put("finish_reason", data.firstFinishReason());
        first.put("prompt_positions", data.firstPromptPositions());
        ObjectNode correct = first.putObject("correct");
        correct.put("ids_match", data.idsMatch());
        correct.put("finish_match", data.finishMatch());
        correct.put("prompt_positions_match", data.positionsMatch());
        ObjectNode memory = report.putObject("memory");
        for (Map.Entry<String, Map<String, Long>> sample : data.memory().entrySet()) {
          ObjectNode sampleNode = memory.putObject(sample.getKey());
          for (Map.Entry<String, Long> field : sample.getValue().entrySet()) {
            sampleNode.put(field.getKey(), field.getValue());
          }
        }
        memory.put(
            "note",
            "process-wide MLX counters sampled after each phase's eval; Metal "
                + "completion-handler frees can lag the eval return, so post-eval values may be "
                + "momentarily high");
        // The warm (post-warmup) measurements of the same request, one entry per measured
        // repetition; the launcher aggregates these into per-pair and pooled quantiles.
        ArrayNode ttft = report.putArray("measured_ttft_s");
        for (double t : data.measuredTtft()) {
          ttft.add(t);
        }
        ArrayNode full = report.putArray("measured_full_generation_s");
        for (double t : data.measuredFull()) {
          full.add(t);
        }
        ArrayNode matches = report.putArray("measured_ids_match");
        for (boolean m : data.measuredIdsMatch()) {
          matches.add(m);
        }
      }
      try {
        Path reportPath = job.outDir.resolve("report.json");
        Files.write(reportPath, report.toString().getBytes(StandardCharsets.UTF_8));
        Files.deleteIfExists(job.outDir.resolve("claim.json"));
      } catch (IOException e) {
        System.err.println("vlm-first-use worker could not write the report: " + e);
      }
    }

    private void finish(String status, String failureReason, int code) {
      finish(status, failureReason, code, null, System.nanoTime() - baseNanos);
    }

    private void finish(String status, String failureReason, int code, ReportData data) {
      finish(status, failureReason, code, data, System.nanoTime() - baseNanos);
    }
  }
}
