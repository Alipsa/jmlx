package se.alipsa.jmlx.benchmarks;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.FileTime;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import tools.jackson.core.JacksonException;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Phase 8.2 WP4 launcher: the both-pin, both-VLM first-inference session
 * (req/plans/phase8-2-plan.md section 5.2; frozen contract in
 * req/plans/phase8-2-reference-findings.md section 17). Consumes one spec JSON and forks one {@link
 * VlmFirstUseWorker} per workload, pair and tree: the two trees are the current checkout (new
 * native pin) and the isolated old-pin compatibility worktree, and for each frozen pair the leading
 * tree alternates (pair 0 leads old, pair 1 leads new, ...), so every pin leads equally and
 * consecutive forks always run on different trees.
 *
 * <p>Before any fork, the launcher verifies each tree's HEAD against the spec's expected SHA, its
 * tracked-clean state (untracked source files other than the shared worker are rejected), the
 * staged native binaries and pin properties (read from disk, never trusted from the spec), the
 * byte-identical worker source in both trees, and every installed jar's digest. Each JMLX jar must
 * be at least as recent as its own tracked main sources and build file; the benchmarks jar must
 * also be as recent as the worker source, which may be untracked in the old tree. The launcher
 * records the verified pin and native binary digests. Each workload's Tier-B manifest is
 * digest-verified and its frozen case 0 parsed (max new tokens and the resolved image
 * paths/digests); the session directory must not pre-exist and must sit outside the repository.
 * Every fork is bracketed by {@code hostrec.sh} pre/post records, and full-output thermal records
 * bracket the session.
 *
 * <p>The session stops forking once {@code session_time_budget_s} has elapsed (the remaining
 * workers are recorded not-started, never silently dropped); the budget gates new forks, not
 * in-flight workers, so the session wall can exceed the budget by at most one worker wait limit.
 * Every forked worker leaves a classification in the session report: a valid report's status (ok /
 * failed / correctness_failed / timeout / job_verification_failed), or, with no report, timeout
 * (launcher wait limit killed it), dead (claim without report), job_verification_failed (exit 1) or
 * crashed (anything else).
 *
 * <p>Exit codes: 0 all workers ok, 1 spec validation failure (nothing forked), 2 at least one
 * non-ok worker (the session report is still written). Opt-in local tooling, not part of {@code
 * check} or the publication surface.
 */
public final class VlmFirstUseBenchmark {
  public static final int EXIT_OK = 0;
  public static final int EXIT_SPEC_FAILED = 1;
  public static final int EXIT_WORKERS_FAILED = 2;

  /**
   * The value a {@link WorkerStarter} returns when it had to kill a worker that did not exit within
   * the launcher's per-worker wait limit. Not a process exit code; the worker left whatever
   * protocol artifacts it had (its cell is then classified dead/crashed from them).
   */
  public static final int WORKER_KILLED = -1;

  /**
   * Headroom the launcher gives a worker past its own wall-clock timeout before the wait limit is
   * reached: the worker's 30 s backstop grace plus 60 s of process-level margin.
   */
  private static final long WAIT_LIMIT_BACKSTOP_S = 90L;

  /** The deadline the launcher gives a killed worker to die before it gives up waiting for it. */
  private static final long KILL_SETTLE_S = 10L;

  /**
   * The deadline for one hostrec.sh append; a hung script is killed and recorded, not waited on.
   */
  private static final long HOSTREC_TIMEOUT_S = 30L;

  /** The exclusion reason for a worker the launcher never reached (starter failure). */
  private static final String REASON_LAUNCHER = "launcher failed to start the worker";

  /** The exclusion reason for a worker skipped because the session's time budget had elapsed. */
  private static final String REASON_BUDGET = "session time budget exceeded; worker not started";

  /** The session's fork order, recorded in the session report. */
  private static final String ORCHESTRATION =
      "workload-major, pair-major, tree-minor; the leading tree alternates per pair "
          + "(even pairs lead old, odd pairs lead new), so consecutive forks always run on "
          + "different trees and each native pin leads equally";

  private static final List<String> JMLX_MODULES =
      List.of(
          "jmlx-native-macos-arm64",
          "jmlx-benchmarks",
          "jmlx-tokenizer",
          "jmlx-vision",
          "jmlx-models",
          "jmlx-jinja",
          "jmlx-core",
          "jmlx-ffi");

  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** One worker process launch; injectable for the pure-Java launcher tests. */
  public interface WorkerStarter {
    /**
     * Forks the worker for {@code job}, waits at most {@code waitLimitSeconds} for it to exit and
     * returns its exit code; a worker that has not exited by the limit is killed and {@link
     * #WORKER_KILLED} is returned instead.
     */
    int start(JobData job, Path jobFile, Path stderrLog, long waitLimitSeconds) throws Exception;
  }

  private static final WorkerStarter FORK =
      (job, jobFile, stderrLog, waitLimitSeconds) -> {
        ProcessBuilder builder = new ProcessBuilder(job.command(jobFile));
        builder.environment().put(job.precisionEnv(), job.precisionValue());
        // redirectError(File) creates (or truncates) the log; the worker prints nothing on stdout.
        builder.redirectError(stderrLog.toFile());
        builder.redirectOutput(ProcessBuilder.Redirect.DISCARD);
        Process process = builder.start();
        // The wait limit sits past the worker's own timeout plus its backstop grace: a healthy
        // worker always reports (or fails) on its own, so the kill only fires for a wedged process.
        if (!process.waitFor(waitLimitSeconds, TimeUnit.SECONDS)) {
          process.destroyForcibly();
          process.waitFor(KILL_SETTLE_S, TimeUnit.SECONDS);
          return WORKER_KILLED;
        }
        return process.exitValue();
      };

  private final WorkerStarter starter;

  /** Creates the launcher with the real process-forking starter. */
  public VlmFirstUseBenchmark() {
    this(FORK);
  }

  VlmFirstUseBenchmark(WorkerStarter starter) {
    this.starter = Objects.requireNonNull(starter, "starter");
  }

  /**
   * Runs {@code VlmFirstUseBenchmark <spec.json>} and exits with the launcher exit code.
   *
   * @param args the single spec JSON path
   */
  public static void main(String[] args) {
    System.exit(new VlmFirstUseBenchmark().run(args.length == 1 ? Path.of(args[0]) : null));
  }

  /** Runs one session from the spec file and returns the launcher exit code. */
  public int run(Path specFile) {
    if (specFile == null || !Files.isRegularFile(specFile)) {
      System.err.println("usage: VlmFirstUseBenchmark <spec.json>");
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

  /** Parses the spec and runs the session; a validation failure exits before any fork. */
  int runSession(JsonNode specNode) {
    Spec spec;
    try {
      spec = Spec.parse(specNode);
    } catch (RuntimeException e) {
      System.err.println("spec validation failed: " + e.getMessage());
      return EXIT_SPEC_FAILED;
    }
    try {
      return new Session(spec, starter).execute();
    } catch (SpecFailure e) {
      System.err.println("spec validation failed: " + e.getMessage());
      return EXIT_SPEC_FAILED;
    } catch (Exception e) {
      System.err.println("vlm-first-use session failed: " + e);
      return EXIT_WORKERS_FAILED;
    }
  }

  /** A validation failure detected after spec parsing (tree/manifest checks); fails pre-fork. */
  static final class SpecFailure extends RuntimeException {
    SpecFailure(String message) {
      super(message);
    }
  }

  /** The validated spec (launcher input; schema {@code jmlx-vlm-first-use-spec-1}). */
  record Spec(
      String session,
      Path outRoot,
      Path repoRoot,
      Path hostrecScript,
      Map<String, Tree> trees,
      List<Workload> workloads,
      int pairs,
      int warmups,
      int measured,
      long workerTimeoutS,
      long sessionTimeBudgetS,
      String precisionEnv,
      String precisionValue) {

    static Spec parse(JsonNode root) {
      String session = root.required("session").asString();
      if (session.isBlank()) {
        throw new IllegalArgumentException("session is required");
      }
      // Normalized: a lexical containment check on an un-normalized path would let "./" and
      // ".." segments slip through. (Symlink resolution is not possible: out_root must not
      // pre-exist, so there is nothing to resolve yet.)
      Path outRoot = Path.of(root.required("out_root").asString()).toAbsolutePath().normalize();
      Path repoRoot = Path.of(root.required("repo_root").asString()).toAbsolutePath().normalize();
      Map<String, Tree> trees = new LinkedHashMap<>();
      for (String name : List.of("old", "new")) {
        JsonNode tree = root.required("trees").required(name);
        trees.put(
            name,
            new Tree(
                name,
                Path.of(tree.required("path").asString()).toAbsolutePath().normalize(),
                tree.required("expected_tree_sha").asString(),
                Path.of(tree.required("class_dir").asString()).toAbsolutePath().normalize(),
                Path.of(tree.required("native_dir").asString()).toAbsolutePath().normalize()));
      }
      // The session directory must sit outside the repository and outside both measured trees:
      // session artifacts inside a measured tree would make it tracked-dirty for the next run.
      if (outRoot.startsWith(repoRoot)
          || outRoot.startsWith(trees.get("old").path())
          || outRoot.startsWith(trees.get("new").path())) {
        throw new IllegalArgumentException(
            "out_root must be outside the repository and both trees: " + outRoot);
      }
      List<Workload> workloads = new ArrayList<>();
      for (JsonNode workload : root.required("workloads")) {
        List<Integer> stops = new ArrayList<>();
        for (JsonNode stop : workload.required("stop_token_ids")) {
          stops.add(stop.asInt());
        }
        workloads.add(
            new Workload(
                workload.required("name").asString(),
                Path.of(workload.required("manifest").asString()),
                workload.required("manifest_sha256").asString(),
                Path.of(workload.required("model_dir").asString()).toAbsolutePath(),
                List.copyOf(stops)));
      }
      int pairs = root.required("pairs").asInt();
      int warmups = root.required("warmups").asInt();
      if (pairs <= 0 || warmups < 0) {
        throw new IllegalArgumentException(
            "pairs must be positive and warmups nonnegative: " + root);
      }
      int measured = root.required("measured").asInt();
      if (measured <= 0) {
        throw new IllegalArgumentException("measured must be positive: " + root);
      }
      long workerTimeoutS = root.required("worker_timeout_s").asLong();
      long sessionTimeBudgetS = root.required("session_time_budget_s").asLong();
      if (workerTimeoutS <= 0 || sessionTimeBudgetS <= 0) {
        throw new IllegalArgumentException(
            "worker_timeout_s and session_time_budget_s must be positive: " + root);
      }
      String precisionEnv = root.required("precision_env").asString();
      String precisionValue = root.required("precision_value").asString();
      if (precisionEnv.isBlank() || precisionValue.isBlank()) {
        throw new IllegalArgumentException("precision_env and precision_value are required");
      }
      if (workloads.isEmpty()) {
        throw new IllegalArgumentException("spec needs at least one workload");
      }
      if (workloads.size() != workloads.stream().map(Workload::name).distinct().count()) {
        throw new IllegalArgumentException("workload names must be unique");
      }
      if (trees.get("old").path().equals(trees.get("new").path())) {
        throw new IllegalArgumentException("the old and new trees must differ");
      }
      if (workloads.stream().anyMatch(workload -> workload.stopTokenIds().isEmpty())) {
        throw new IllegalArgumentException(
            "each workload needs a nonempty, explicitly recorded stop_token_ids set");
      }
      // LinkedHashMap, not Map.copyOf: the session report's tree order (old first, new second)
      // is part of the evidence layout, and Map.copyOf does not preserve iteration order.
      return new Spec(
          session,
          outRoot,
          repoRoot,
          Path.of(root.required("hostrec_script").asString()).toAbsolutePath(),
          new LinkedHashMap<>(trees),
          List.copyOf(workloads),
          pairs,
          warmups,
          measured,
          workerTimeoutS,
          sessionTimeBudgetS,
          precisionEnv,
          precisionValue);
    }
  }

  /** One native tree (worktree root, recorded SHA, installDist lib dir, staged native dir). */
  record Tree(String name, Path path, String expectedTreeSha, Path classDir, Path nativeDir) {}

  /** One VLM workload (Tier-B manifest + model directory + the recorded stop set). */
  record Workload(
      String name,
      Path manifest,
      String manifestSha256,
      Path modelDir,
      List<Integer> stopTokenIds) {

    /** The manifest path, resolved against the repo root when relative. */
    Path manifestAt(Path repoRoot) {
      return manifest.isAbsolute() ? manifest : repoRoot.resolve(manifest);
    }
  }

  /** One resolved, digest-verified image of the frozen case. */
  record ImageRef(Path path, String sha) {}

  /** The frozen case's resolved inputs (case 0 of the manifest, per workload). */
  record WorkloadPrep(int maxNewTokens, List<ImageRef> images) {}

  /** One forked worker's classification and (when present) parsed report. */
  record WorkerOutcome(
      String workload,
      String tree,
      int pairIndex,
      String classification,
      JsonNode report,
      String detail) {}

  /** One tree's verified identity (HEAD, tracked state, worker source digest, installed jars). */
  record TreeInfo(
      Tree tree,
      String treeSha,
      boolean trackedDirty,
      List<String> untracked,
      String workerSha,
      Map<String, String> jarSha256,
      Map<String, String> nativePin,
      Map<String, String> nativeSha256) {}

  /** The per-fork job record (launcher -> worker; schema {@code jmlx-vlm-first-use-job-1}). */
  record JobData(
      Workload workload,
      Tree tree,
      int pairIndex,
      Path manifestPath,
      WorkloadPrep prep,
      int warmups,
      int measured,
      long workerTimeoutS,
      String precisionEnv,
      String precisionValue,
      Instant forkUtc,
      Path cell) {

    String toJson() {
      ObjectNode job = MAPPER.createObjectNode();
      job.put("schema", VlmFirstUseWorker.JOB_SCHEMA);
      job.put("workload", workload.name());
      job.put("tree", tree.name());
      job.put("tree_path", tree.path().toString());
      job.put("pair_index", pairIndex);
      job.put("model_dir", workload.modelDir().toString());
      // The manifest path the launcher verified from disk (resolved against the repo root when
      // the spec gave a relative path), not the spec's raw string: the worker must never
      // re-resolve it against its own working directory.
      job.put("manifest_path", manifestPath.toString());
      job.put("manifest_sha256", workload.manifestSha256());
      ArrayNode paths = job.putArray("image_paths");
      ArrayNode digests = job.putArray("image_sha256s");
      for (ImageRef image : prep.images()) {
        paths.add(image.path().toString());
        digests.add(image.sha());
      }
      job.put("max_new_tokens", prep.maxNewTokens());
      ArrayNode stops = job.putArray("stop_token_ids");
      for (int stop : workload.stopTokenIds()) {
        stops.add(stop);
      }
      job.put("warmups", warmups);
      job.put("measured", measured);
      job.put("worker_timeout_s", workerTimeoutS);
      job.put("precision_env", precisionEnv);
      job.put("precision_value", precisionValue);
      job.put("fork_utc", forkUtc.toString());
      job.put("out_dir", cell.toString());
      return job.toString();
    }

    /** The exact fork command: the tree's own installDist, its own staged native lib. */
    List<String> command(Path jobFile) {
      String java = Path.of(System.getProperty("java.home")).resolve("bin/java").toString();
      return List.of(
          java,
          "--enable-native-access=ALL-UNNAMED",
          "-Djmlx.library.path=" + tree.nativeDir().toString(),
          "-cp",
          tree.classDir().resolve("*").toString(),
          "se.alipsa.jmlx.benchmarks.VlmFirstUseWorker",
          jobFile.toString());
    }
  }

  /** One session's execution: verification, forking, host records and the session report. */
  static final class Session {
    private final Spec spec;
    private final WorkerStarter starter;
    private final long sessionStartNanos = System.nanoTime();
    private final Instant utcStart = Instant.now();
    private final List<WorkerOutcome> outcomes = new ArrayList<>();
    private final List<String> forkOrder = new ArrayList<>();
    private final Map<String, TreeInfo> treeInfos = new LinkedHashMap<>();
    private final Map<String, WorkloadPrep> preps = new LinkedHashMap<>();
    private final List<String> thermalNotes = new ArrayList<>();
    private final List<String> hostrecFailures = new ArrayList<>();
    private boolean budgetExhausted;

    private Session(Spec spec, WorkerStarter starter) {
      this.spec = spec;
      this.starter = starter;
    }

    int execute() throws Exception {
      verifyEnvironment();
      Files.createDirectories(spec.outRoot);
      runThermal("start", spec.outRoot.resolve("thermal-start.txt"));
      for (Workload workload : spec.workloads) {
        runWorkload(workload, preps.get(workload.name()));
      }
      runThermal("end", spec.outRoot.resolve("thermal-end.txt"));
      ObjectNode report = sessionReport();
      Files.write(
          spec.outRoot.resolve("session-report.json"),
          report.toString().getBytes(StandardCharsets.UTF_8));
      boolean allOk = outcomes.stream().allMatch(outcome -> "ok".equals(outcome.classification()));
      return allOk ? EXIT_OK : EXIT_WORKERS_FAILED;
    }

    /**
     * The pre-fork verification: trees (HEAD, tracked-clean, staged pin, byte-identical worker
     * source), manifests (digest, frozen case, images) and the hostrec script. Any failure is a
     * {@link SpecFailure} raised before the first fork.
     */
    private void verifyEnvironment() throws Exception {
      if (Files.exists(spec.outRoot)) {
        throw new SpecFailure("out_root must not pre-exist: " + spec.outRoot);
      }
      if (!Files.isRegularFile(spec.hostrecScript)) {
        throw new SpecFailure("hostrec script missing: " + spec.hostrecScript);
      }
      String firstWorkerSha = null;
      for (Tree tree : spec.trees.values()) {
        if (!Files.isDirectory(tree.path)) {
          throw new SpecFailure("tree path missing: " + tree.path);
        }
        if (!Files.isDirectory(tree.classDir)) {
          throw new SpecFailure("tree class dir missing: " + tree.classDir);
        }
        if (!Files.isRegularFile(tree.nativeDir.resolve("mlx.metallib"))) {
          throw new SpecFailure("tree has no staged metallib: " + tree.nativeDir);
        }
        Map<String, String> pin = readPinProperties(tree.nativeDir);
        if (pin.containsKey("error")
            || !pin.containsKey("mlxMetalVersion")
            || !pin.containsKey("mlxcCommit")) {
          throw new SpecFailure("tree staged pin properties incomplete: " + tree.nativeDir);
        }
        String head = runCommand(tree.path, "git", "rev-parse", "HEAD")[0].trim();
        if (!head.equals(tree.expectedTreeSha)) {
          throw new SpecFailure(
              "tree " + tree.name + " HEAD " + head + " != expected " + tree.expectedTreeSha);
        }
        List<String> status =
            runCommandLines(tree.path, "git", "status", "--porcelain", "--untracked-files=all");
        List<String> dirty =
            status.stream().filter(line -> !line.isBlank() && !line.startsWith("??")).toList();
        if (!dirty.isEmpty()) {
          // An accepted run must be attributable to the recorded tree SHA; tracked changes
          // (never untracked files) mean the measured code is not the committed code.
          throw new SpecFailure(
              "tree "
                  + tree.name
                  + " is tracked-dirty (measured code must equal the recorded "
                  + "tree SHA): "
                  + String.join(" ", dirty));
        }
        Path workerSource =
            tree.path.resolve(
                "jmlx-benchmarks/src/main/java/se/alipsa/jmlx/benchmarks/VlmFirstUseWorker.java");
        if (!Files.isRegularFile(workerSource)) {
          throw new SpecFailure("worker source missing in tree " + tree.name + ": " + workerSource);
        }
        String workerSha = sha256Hex(workerSource);
        if (firstWorkerSha == null) {
          firstWorkerSha = workerSha;
        } else if (!firstWorkerSha.equals(workerSha)) {
          throw new SpecFailure(
              "worker source differs between trees (must be byte-identical): "
                  + firstWorkerSha
                  + " vs "
                  + workerSha);
        }
        // The executed jars are digested and checked against the source/build inputs of their
        // own module. This detects an installDist left behind after switching commits.
        Map<String, String> jarSha = new LinkedHashMap<>();
        boolean benchmarksJarFound = false;
        try (var listing = Files.list(tree.classDir)) {
          List<Path> jars =
              listing
                  .filter(path -> path.getFileName().toString().endsWith(".jar"))
                  .sorted(
                      Comparator.comparing(
                          path -> path.getFileName().toString(), String.CASE_INSENSITIVE_ORDER))
                  .toList();
          for (Path jar : jars) {
            String jarName = jar.getFileName().toString();
            for (String module : JMLX_MODULES) {
              if (jarName.startsWith(module + "-") || jarName.equals(module + ".jar")) {
                if (module.equals("jmlx-benchmarks")) {
                  benchmarksJarFound = true;
                }
                FileTime newestInput = newestModuleInput(tree.path, module);
                if (module.equals("jmlx-benchmarks")) {
                  FileTime workerTime = Files.getLastModifiedTime(workerSource);
                  if (workerTime.compareTo(newestInput) > 0) {
                    newestInput = workerTime;
                  }
                }
                if (Files.getLastModifiedTime(jar).compareTo(newestInput) < 0) {
                  throw new SpecFailure(module + " jar predates its source or build file: " + jar);
                }
                break;
              }
            }
            jarSha.put(jar.getFileName().toString(), sha256Hex(jar));
          }
        }
        if (!benchmarksJarFound) {
          throw new SpecFailure("tree has no jmlx-benchmarks jar in class dir: " + tree.classDir);
        }
        // Untracked files are recorded, not rejected (a staged native dir is untracked on purpose).
        List<String> untracked =
            status.stream()
                .filter(line -> line.startsWith("??"))
                .map(line -> line.substring(3))
                .toList();
        for (String path : untracked) {
          if ((path.startsWith("src/") || path.contains("/src/"))
              && !path.equals(tree.path.relativize(workerSource).toString())) {
            throw new SpecFailure("untracked source path in tree " + tree.name + ": " + path);
          }
        }
        Map<String, String> nativeSha = new LinkedHashMap<>();
        for (String name : List.of("mlx.metallib", "libmlxc.dylib", "libmlx.dylib")) {
          Path binary = tree.nativeDir.resolve(name);
          if (!Files.isRegularFile(binary)) {
            throw new SpecFailure("tree native binary missing: " + binary);
          }
          nativeSha.put(name, sha256Hex(binary));
        }
        // Tracked-dirty trees are refused above, so every recorded tree is tracked-clean.
        treeInfos.put(
            tree.name,
            new TreeInfo(tree, head, false, untracked, workerSha, jarSha, pin, nativeSha));
      }
      for (Workload workload : spec.workloads) {
        Path manifest = workload.manifestAt(spec.repoRoot);
        if (!Files.isRegularFile(manifest)) {
          throw new SpecFailure("manifest missing: " + manifest);
        }
        String sha = sha256Hex(manifest);
        if (!sha.equals(workload.manifestSha256)) {
          throw new SpecFailure(
              "manifest digest mismatch for "
                  + workload.name
                  + ": "
                  + sha
                  + " != "
                  + workload.manifestSha256);
        }
        if (!Files.isDirectory(workload.modelDir)) {
          throw new SpecFailure(
              "model dir missing for " + workload.name + ": " + workload.modelDir);
        }
        preps.put(workload.name, verifyWorkloadCase(workload, manifest));
      }
    }

    private static FileTime newestModuleInput(Path tree, String module) throws Exception {
      FileTime newest = FileTime.fromMillis(0);
      for (String relative :
          runCommandLines(
              tree, "git", "ls-files", "--", module + "/src/main", module + "/build.gradle")) {
        if (!relative.isBlank()) {
          FileTime time = Files.getLastModifiedTime(tree.resolve(relative));
          if (time.compareTo(newest) > 0) {
            newest = time;
          }
        }
      }
      return newest;
    }

    /** Parses the manifest's frozen case 0 and verifies its images against the recorded digests. */
    private WorkloadPrep verifyWorkloadCase(Workload workload, Path manifest) {
      JsonNode manifestNode = MAPPER.readTree(manifest.toFile());
      JsonNode caseNode = manifestNode.required("reference").required("cases").get(0);
      int maxNewTokens = caseNode.required("max_new_tokens").asInt();
      if (maxNewTokens <= 0) {
        throw new SpecFailure(
            "manifest " + workload.name + " case 0 max_new_tokens must be positive");
      }
      JsonNode imagesNode = manifestNode.required("images");
      List<Integer> indices = new ArrayList<>();
      JsonNode indicesNode = caseNode.path("image_indices");
      if (indicesNode.isMissingNode() || !indicesNode.isArray() || indicesNode.isEmpty()) {
        for (int i = 0; i < imagesNode.size(); i++) {
          indices.add(i);
        }
      } else {
        for (JsonNode index : indicesNode) {
          int i = index.asInt();
          if (i < 0 || i >= imagesNode.size()) {
            throw new SpecFailure(
                "manifest " + workload.name + " image index " + i + " is out of range");
          }
          indices.add(i);
        }
      }
      List<ImageRef> images = new ArrayList<>();
      for (int i : indices) {
        JsonNode image = imagesNode.get(i);
        Path path = spec.repoRoot.resolve(image.required("path").asString()).toAbsolutePath();
        if (!Files.isRegularFile(path)) {
          throw new SpecFailure("manifest " + workload.name + " image missing: " + path);
        }
        String sha = sha256Hex(path);
        if (!sha.equals(image.required("sha256").asString())) {
          throw new SpecFailure("manifest " + workload.name + " image digest mismatch: " + path);
        }
        images.add(new ImageRef(path, sha));
      }
      return new WorkloadPrep(maxNewTokens, List.copyOf(images));
    }

    /** Forks every worker of one workload (alternating leader per pair) and classifies each. */
    private void runWorkload(Workload workload, WorkloadPrep prep) throws Exception {
      for (int pair = 0; pair < spec.pairs; pair++) {
        String leader = pair % 2 == 0 ? "old" : "new";
        for (String treeName : List.of(leader, otherTree(leader))) {
          boolean budgetHit =
              !budgetExhausted
                  && System.nanoTime() - sessionStartNanos
                      >= spec.sessionTimeBudgetS * 1_000_000_000L;
          if (budgetExhausted || budgetHit) {
            budgetExhausted = true;
            outcomes.add(
                new WorkerOutcome(
                    workload.name, treeName, pair, "not-started", null, REASON_BUDGET));
            continue;
          }
          Tree tree = spec.trees.get(treeName);
          Path cell = spec.outRoot.resolve(workload.name).resolve(treeName + "-p" + pair);
          Files.createDirectories(cell);
          Instant forkUtc = Instant.now();
          JobData job =
              new JobData(
                  workload,
                  tree,
                  pair,
                  workload.manifestAt(spec.repoRoot),
                  prep,
                  spec.warmups,
                  spec.measured,
                  spec.workerTimeoutS,
                  spec.precisionEnv,
                  spec.precisionValue,
                  forkUtc,
                  cell);
          Path jobFile = cell.resolve("job.json");
          Files.write(jobFile, job.toJson().getBytes(StandardCharsets.UTF_8));
          forkOrder.add(workload.name + "/" + treeName + "/p" + pair);
          hostrec("pre:" + workload.name + ":" + treeName + ":p" + pair);
          int code;
          try {
            code =
                starter.start(
                    job,
                    jobFile,
                    cell.resolve("worker-stderr.log"),
                    spec.workerTimeoutS + WAIT_LIMIT_BACKSTOP_S);
          } catch (Exception e) {
            System.err.println(
                "launcher failed to start "
                    + workload.name
                    + "/"
                    + treeName
                    + "/p"
                    + pair
                    + ": "
                    + e);
            outcomes.add(
                new WorkerOutcome(
                    workload.name, treeName, pair, "not-started", null, REASON_LAUNCHER));
            continue;
          }
          hostrec("post:" + workload.name + ":" + treeName + ":p" + pair);
          outcomes.add(classify(workload.name, treeName, pair, cell, code));
        }
      }
    }

    private String otherTree(String tree) {
      return "old".equals(tree) ? "new" : "old";
    }

    /** Reads the worker's protocol artifacts and classifies the fork (no report is ever faked). */
    WorkerOutcome classify(String workload, String tree, int pair, Path cell, int starterResult) {
      Path reportPath = cell.resolve("report.json");
      if (Files.isRegularFile(reportPath)) {
        try {
          JsonNode parsed = MAPPER.readTree(reportPath.toFile());
          if (!VlmFirstUseWorker.REPORT_SCHEMA.equals(parsed.path("schema").asString())) {
            return new WorkerOutcome(
                workload,
                tree,
                pair,
                "invalid-report",
                parsed,
                "report schema differs from " + VlmFirstUseWorker.REPORT_SCHEMA);
          }
          String status = parsed.path("status").asString();
          if (!List.of("ok", "failed", "correctness_failed", "timeout", "job_verification_failed")
              .contains(status)) {
            return new WorkerOutcome(
                workload, tree, pair, "invalid-report", parsed, "unknown report status: " + status);
          }
          if (!workload.equals(parsed.path("workload").asString())
              || !tree.equals(parsed.path("tree").asString())
              || pair != parsed.path("pair_index").asInt(-1)) {
            return new WorkerOutcome(
                workload, tree, pair, "invalid-report", parsed, "report cell identity differs");
          }
          if (!parsed.path("exit_code").isInt()
              || (parsed.path("exit_code").asInt(-1) != starterResult
                  && !("timeout".equals(status)
                      && starterResult == WORKER_KILLED
                      && parsed.path("exit_code").asInt(-1) == VlmFirstUseWorker.EXIT_TIMEOUT))) {
            return new WorkerOutcome(
                workload, tree, pair, "invalid-report", parsed, "report exit_code differs");
          }
          JsonNode pin = parsed.path("native_pin");
          Map<String, String> expectedPin = treeInfos.get(tree).nativePin();
          if ("ok".equals(status) || pin.isObject()) {
            for (String key : List.of("mlxMetalVersion", "mlxcCommit", "library_path")) {
              if (!expectedPin.get(key).equals(pin.path(key).asString())) {
                return new WorkerOutcome(
                    workload,
                    tree,
                    pair,
                    "invalid-report",
                    parsed,
                    "report native pin differs: " + key);
              }
            }
          }
          if ("ok".equals(status)) {
            // Defense in depth: a worker that reports ok must also have exited 0 and have every
            // first_request.correct.* field true. A mismatch means the worker's report is
            // internally inconsistent with its own protocol; it is not an ok cell and its warm
            // values must not enter the pooled or paired statistics (plan section 7:
            // "missing/failed/instrumented samples cannot enter successful headline summaries").
            if (starterResult == VlmFirstUseWorker.EXIT_OK && firstRequestCorrect(parsed)) {
              if (!measuredValuesValid(parsed.path("measured_ttft_s"), spec.measured)
                  || !measuredValuesValid(
                      parsed.path("measured_full_generation_s"), spec.measured)) {
                return new WorkerOutcome(
                    workload,
                    tree,
                    pair,
                    "invalid-report",
                    parsed,
                    "measured sample count or value differs");
              }
              return new WorkerOutcome(workload, tree, pair, "ok", parsed, null);
            }
            return new WorkerOutcome(
                workload,
                tree,
                pair,
                "invalid-report",
                parsed,
                "report status is ok but the exit code ("
                    + starterResult
                    + ") or first_request.correct.* fields disagree with it");
          }
          int expectedExit =
              switch (status) {
                case "failed" -> VlmFirstUseWorker.EXIT_RUNTIME_FAILED;
                case "correctness_failed" -> VlmFirstUseWorker.EXIT_CORRECTNESS_FAILED;
                case "job_verification_failed" -> VlmFirstUseWorker.EXIT_JOB_VERIFICATION_FAILED;
                default -> VlmFirstUseWorker.EXIT_TIMEOUT;
              };
          if (starterResult != expectedExit
              && !("timeout".equals(status) && starterResult == WORKER_KILLED)) {
            return new WorkerOutcome(
                workload,
                tree,
                pair,
                "invalid-report",
                parsed,
                "report status and exit code differ");
          }
          return new WorkerOutcome(workload, tree, pair, status, parsed, null);
        } catch (Exception e) {
          return new WorkerOutcome(
              workload, tree, pair, "invalid-report", null, "report unreadable: " + e.getMessage());
        }
      }
      if (starterResult == WORKER_KILLED) {
        return new WorkerOutcome(
            workload, tree, pair, "timeout", null, "launcher wait limit reached; worker killed");
      }
      if (Files.isRegularFile(cell.resolve("claim.json"))) {
        return new WorkerOutcome(workload, tree, pair, "dead", null, "claim without report");
      }
      if (starterResult == VlmFirstUseWorker.EXIT_JOB_VERIFICATION_FAILED) {
        return new WorkerOutcome(
            workload,
            tree,
            pair,
            "job_verification_failed",
            null,
            "worker rejected the job (exit 1)");
      }
      return new WorkerOutcome(
          workload,
          tree,
          pair,
          "crashed",
          null,
          "worker exited " + starterResult + " before the handshake");
    }

    /** True when the report's first_request.correct object is present and every field is true. */
    private static boolean firstRequestCorrect(JsonNode report) {
      JsonNode correct = report.path("first_request").path("correct");
      if (!correct.isObject() || correct.isEmpty()) {
        return false;
      }
      for (Map.Entry<String, JsonNode> field : correct.properties()) {
        if (!field.getValue().isBoolean() || !field.getValue().asBoolean()) {
          return false;
        }
      }
      return true;
    }

    private static boolean measuredValuesValid(JsonNode values, int count) {
      if (!values.isArray() || values.size() != count) {
        return false;
      }
      for (JsonNode value : values) {
        if (!value.isNumber() || !Double.isFinite(value.asDouble()) || value.asDouble() <= 0) {
          return false;
        }
      }
      return true;
    }

    /**
     * One hostrec.sh append (the 60 s polling loop is the session wrapper's job, not this one's).
     * Failures never abort the session but are recorded in the session report: a missing host
     * record is a plan section 4 trigger for a further session.
     */
    void hostrec(String label) {
      Path dir = spec.outRoot.resolve("hostrec");
      try {
        Files.createDirectories(dir);
        ProcessBuilder builder = new ProcessBuilder(spec.hostrecScript.toString(), label);
        builder.environment().put("HOSTREC_DIR", dir.toString());
        builder.redirectErrorStream(true);
        Process process = builder.start();
        // Bounded wait: a hung script must not wedge the session. hostrec emits at most a few
        // lines, so reading after the wait cannot lose output to a full pipe buffer.
        if (!process.waitFor(HOSTREC_TIMEOUT_S, TimeUnit.SECONDS)) {
          process.destroyForcibly();
          hostrecFailures.add(
              label + ": hostrec.sh did not exit within " + HOSTREC_TIMEOUT_S + " s");
          return;
        }
        int exit = process.exitValue();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        if (exit != 0) {
          hostrecFailures.add(label + ": hostrec.sh exited " + exit);
        }
        Files.writeString(
            dir.resolve("host.log.cmds"),
            label + "\n",
            StandardCharsets.UTF_8,
            StandardOpenOption.CREATE,
            StandardOpenOption.APPEND);
        if (!output.isBlank()) {
          System.err.println("hostrec " + label + ": " + output.trim());
        }
      } catch (Exception e) {
        hostrecFailures.add(label + ": " + e.getMessage());
      }
    }

    /** Full-output thermal record (session start/end), kept per plan section 4. */
    void runThermal(String when, Path file) {
      try {
        String output = String.join("\n", runCommand(Path.of("/"), "pmset", "-g", "therm"));
        Files.writeString(file, output, StandardCharsets.UTF_8);
        thermalNotes.add(when + ":" + file);
      } catch (Exception e) {
        thermalNotes.add(when + ":failed(" + e.getMessage() + ")");
      }
    }

    /** The assembled session report (raw worker records + per-pair and pooled warm statistics). */
    ObjectNode sessionReport() {
      ObjectNode report = MAPPER.createObjectNode();
      // Version 3 adds verified native_sha256 and a consistent job_verification_failed category.
      report.put("schema", "jmlx-vlm-first-use-session-3");
      report.put("session", spec.session);
      report.put("utc_start", utcStart.toString());
      report.put("utc_end", Instant.now().toString());
      report.put("out_root", spec.outRoot.toString());
      report.put("orchestration", ORCHESTRATION);
      report.put("pairs", spec.pairs);
      report.put("warmups", spec.warmups);
      report.put("measured", spec.measured);
      report.put("worker_timeout_s", spec.workerTimeoutS);
      report.put("session_time_budget_s", spec.sessionTimeBudgetS);
      ObjectNode precision = report.putObject("precision");
      precision.put("env", spec.precisionEnv);
      precision.put("value", spec.precisionValue);
      ArrayNode trees = report.putArray("trees");
      for (TreeInfo info : treeInfos.values()) {
        ObjectNode tree = trees.addObject();
        tree.put("name", info.tree().name());
        tree.put("path", info.tree().path().toString());
        tree.put("tree_sha", info.treeSha());
        tree.put("expected_tree_sha", info.tree().expectedTreeSha());
        tree.put("tracked_dirty", info.trackedDirty());
        ArrayNode untracked = tree.putArray("untracked");
        for (String file : info.untracked()) {
          untracked.add(file);
        }
        tree.put("worker_source_sha256", info.workerSha());
        ObjectNode jars = tree.putObject("jar_sha256");
        for (Map.Entry<String, String> entry : info.jarSha256().entrySet()) {
          jars.put(entry.getKey(), entry.getValue());
        }
        ObjectNode pin = tree.putObject("native_pin");
        for (Map.Entry<String, String> entry : info.nativePin().entrySet()) {
          pin.put(entry.getKey(), entry.getValue());
        }
        ObjectNode nativeDigests = tree.putObject("native_sha256");
        for (Map.Entry<String, String> entry : info.nativeSha256().entrySet()) {
          nativeDigests.put(entry.getKey(), entry.getValue());
        }
        tree.put("class_dir", info.tree().classDir().toString());
      }
      ArrayNode workloads = report.putArray("workloads");
      for (Workload workload : spec.workloads) {
        ObjectNode node = workloads.addObject();
        node.put("name", workload.name());
        node.put("manifest", workload.manifestAt(spec.repoRoot).toString());
        node.put("manifest_sha256", workload.manifestSha256());
        node.put("model_dir", workload.modelDir().toString());
        ArrayNode stops = node.putArray("stop_token_ids");
        for (int stop : workload.stopTokenIds()) {
          stops.add(stop);
        }
      }
      ArrayNode order = report.putArray("fork_order");
      for (String entry : forkOrder) {
        order.add(entry);
      }
      ArrayNode thermal = report.putArray("thermal");
      for (String note : thermalNotes) {
        thermal.add(note);
      }
      report.put(
          "hostrec",
          "hostrec.sh pre/post appends per fork in "
              + spec.outRoot.resolve("hostrec").resolve("host.log")
              + " (one line per record); the 60 s polling loop is the session wrapper's job, and "
              + "the wrapper copies its host log into the session directory as hostrec.log; "
              + "failed or timed-out hostrec calls are listed in hostrec_failures");
      ArrayNode hostrecFailuresNode = report.putArray("hostrec_failures");
      for (String failure : hostrecFailures) {
        hostrecFailuresNode.add(failure);
      }
      ArrayNode workers = report.putArray("workers");
      for (WorkerOutcome outcome : outcomes) {
        ObjectNode node = workers.addObject();
        node.put("workload", outcome.workload());
        node.put("tree", outcome.tree());
        node.put("pair_index", outcome.pairIndex());
        node.put("classification", outcome.classification());
        if (outcome.detail() != null) {
          node.put("detail", outcome.detail());
        }
        if (outcome.report() != null) {
          node.set("report", outcome.report());
        }
      }
      report.set("summary", summary());
      return report;
    }

    /**
     * Warm quantiles per tree per pair, the paired per-pair warm-median ratios, and the pooled warm
     * statistics per workload per tree. First-request values stay raw per worker: they come from
     * fresh processes and are never pooled into the warm statistics.
     */
    ObjectNode summary() {
      ObjectNode summary = MAPPER.createObjectNode();
      summary.put("quantile_convention", "linear-interpolation-type-7");
      ArrayNode pairs = summary.putArray("pairs");
      for (Workload workload : spec.workloads) {
        for (int pair = 0; pair < spec.pairs; pair++) {
          final int pairIndex = pair;
          ObjectNode pairNode = pairs.addObject();
          pairNode.put("workload", workload.name());
          pairNode.put("pair_index", pairIndex);
          Map<String, Double> warmMedians = new LinkedHashMap<>();
          for (String treeName : List.of("old", "new")) {
            WorkerOutcome outcome =
                outcomes.stream()
                    .filter(
                        o ->
                            o.workload().equals(workload.name())
                                && o.tree().equals(treeName)
                                && o.pairIndex() == pairIndex)
                    .findFirst()
                    .orElse(null);
            ObjectNode member = pairNode.putObject(treeName);
            if (outcome == null || outcome.report() == null) {
              member.put(
                  "classification", outcome == null ? "not-started" : outcome.classification());
              continue;
            }
            member.put("classification", outcome.classification());
            member.set("first_request", outcome.report().path("first_request"));
            member.set("load", outcome.report().path("load"));
            if (!"ok".equals(outcome.classification())) {
              continue;
            }
            ObjectNode warm = member.putObject("warm");
            BaselineStatistics.Quantiles ttft =
                quantilesOf(outcome.report().path("measured_ttft_s"));
            BaselineStatistics.Quantiles full =
                quantilesOf(outcome.report().path("measured_full_generation_s"));
            warm.put("ttft_p10", ttft.p10());
            warm.put("ttft_median", ttft.median());
            warm.put("ttft_p90", ttft.p90());
            warm.put("full_p10", full.p10());
            warm.put("full_median", full.median());
            warm.put("full_p90", full.p90());
            warm.put("n", ttft.n());
            warmMedians.put(treeName + ">ttft", ttft.median());
            warmMedians.put(treeName + ">full", full.median());
          }
          if (warmMedians.size() == 4 && warmMedians.values().stream().allMatch(v -> v > 0)) {
            ObjectNode ratio = pairNode.putObject("ratio_new_over_old");
            ratio.put("ttft", warmMedians.get("new>ttft") / warmMedians.get("old>ttft"));
            ratio.put("full_generation", warmMedians.get("new>full") / warmMedians.get("old>full"));
            ratio.put(
                "note", "paired per-pair warm medians (type-7); pairs are not independent samples");
          }
        }
      }
      ObjectNode pooled = summary.putObject("pooled_warm");
      for (Workload workload : spec.workloads) {
        ObjectNode workloadNode = pooled.putObject(workload.name());
        for (String treeName : List.of("old", "new")) {
          List<Double> ttft = new ArrayList<>();
          List<Double> full = new ArrayList<>();
          for (WorkerOutcome outcome : outcomes) {
            if (!outcome.workload().equals(workload.name())
                || !outcome.tree().equals(treeName)
                || !"ok".equals(outcome.classification())
                || outcome.report() == null) {
              continue;
            }
            for (JsonNode value : outcome.report().path("measured_ttft_s")) {
              ttft.add(value.asDouble());
            }
            for (JsonNode value : outcome.report().path("measured_full_generation_s")) {
              full.add(value.asDouble());
            }
          }
          ObjectNode treeNode = workloadNode.putObject(treeName);
          if (ttft.isEmpty() || full.isEmpty()) {
            treeNode.put("n", 0);
            treeNode.put("note", "no ok workers; warm quantiles unavailable");
            continue;
          }
          BaselineStatistics.Quantiles ttftQ = BaselineStatistics.quantiles(ttft);
          BaselineStatistics.Quantiles fullQ = BaselineStatistics.quantiles(full);
          treeNode.put("ttft_p10", ttftQ.p10());
          treeNode.put("ttft_median", ttftQ.median());
          treeNode.put("ttft_p90", ttftQ.p90());
          treeNode.put("full_p10", fullQ.p10());
          treeNode.put("full_median", fullQ.median());
          treeNode.put("full_p90", fullQ.p90());
          treeNode.put("n", ttftQ.n());
          treeNode.put(
              "note",
              "pooled over the ok workers' measured requests of this tree; process pairs, not "
                  + "independent samples");
        }
      }
      return summary;
    }

    private BaselineStatistics.Quantiles quantilesOf(JsonNode values) {
      List<Double> list = new ArrayList<>();
      for (JsonNode value : values) {
        list.add(value.asDouble());
      }
      return BaselineStatistics.quantiles(list);
    }

    /** The staged pin properties of one tree (launcher-side read; the worker re-reads its own). */
    private static Map<String, String> readPinProperties(Path nativeDir) {
      Map<String, String> properties = new LinkedHashMap<>();
      Path pinFile = nativeDir.resolve("native-pin.properties");
      try {
        for (String line : Files.readAllLines(pinFile, StandardCharsets.UTF_8)) {
          int eq = line.indexOf('=');
          if (eq > 0) {
            properties.put(line.substring(0, eq).trim(), line.substring(eq + 1).trim());
          }
        }
      } catch (IOException e) {
        properties.put("error", String.valueOf(e.getMessage()));
      }
      properties.put("library_path", nativeDir.toAbsolutePath().toString());
      return properties;
    }

    private static String[] runCommand(Path workdir, String... command) throws Exception {
      ProcessBuilder builder = new ProcessBuilder(command);
      builder.directory(workdir.toFile());
      builder.redirectErrorStream(true);
      Process process = builder.start();
      String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
      int exit = process.waitFor();
      if (exit != 0) {
        throw new IOException(command[0] + " exited " + exit + ": " + output);
      }
      return output.split("\n");
    }

    private static List<String> runCommandLines(Path workdir, String... command) throws Exception {
      List<String> lines = new ArrayList<>();
      for (String line : runCommand(workdir, command)) {
        lines.add(line);
      }
      return lines;
    }

    private static String sha256Hex(Path path) {
      try {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (var in = Files.newInputStream(path)) {
          byte[] buffer = new byte[1 << 16];
          int read;
          while ((read = in.read(buffer)) >= 0) {
            digest.update(buffer, 0, read);
          }
        }
        return java.util.HexFormat.of().formatHex(digest.digest());
      } catch (IOException e) {
        throw new UncheckedIOException("sha256 failed for " + path, e);
      } catch (NoSuchAlgorithmException e) {
        throw new IllegalStateException("SHA-256 unavailable", e);
      }
    }
  }
}
