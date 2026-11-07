package se.alipsa.jmlx.benchmarks;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * One launcher-written worker job file (schema {@code jmlx-decode-baseline-job-1}; frozen contract
 * in req/plans/phase8-2-reference-findings.md section 13.1). The launcher is the only writer; the
 * worker treats every field as data to be re-verified, never as a source of truth over the manifest
 * anchors. The last two components are launcher-side metadata (the fork command needs them); the
 * job file itself carries no classpath.
 *
 * @param adapter controlled or stock
 * @param caseName manifest text-case name
 * @param promptLength frozen manifest prompt length
 * @param mode headline, fixed-continuation or reference-pair
 * @param first run_policy first_requests (0 for reference-pair)
 * @param warmups run_policy warmups (0 for reference-pair)
 * @param measured run_policy measured_requests (1 for reference-pair)
 * @param timeoutS the exact run_policy worker timeout for this mode
 * @param precisionEnv the session precision mode ("1" or "0"); the worker verifies it
 * @param artifactDir the spec's artifact_dirs entry for the case
 * @param manifestPath the canonical frozen manifest path
 * @param manifestSha256 the frozen manifest digest
 * @param repoDir the checkout the workers run from
 * @param outDir the launcher-created worker output directory
 * @param fcOffset fixed-continuation offset (32; written only in that mode)
 * @param fcCount fixed-continuation count (32; written only in that mode)
 * @param runtime java or python (launcher-side: selects the fork command)
 * @param modelsTestClasspath the Java worker's classpath (launcher-side only)
 */
record BaselineJob(
    String adapter,
    String caseName,
    int promptLength,
    String mode,
    int first,
    int warmups,
    int measured,
    int timeoutS,
    String precisionEnv,
    String artifactDir,
    String manifestPath,
    String manifestSha256,
    String repoDir,
    String outDir,
    int fcOffset,
    int fcCount,
    String runtime,
    String modelsTestClasspath) {

  static final String SCHEMA = "jmlx-decode-baseline-job-1";

  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** Writes the job file; the directory must already exist (launcher-created). */
  void write(Path file) throws IOException {
    ObjectNode job = MAPPER.createObjectNode();
    job.put("schema", SCHEMA);
    job.put("adapter", adapter);
    job.put("case", caseName);
    job.put("prompt_length", promptLength);
    job.put("mode", mode);
    ObjectNode requests = job.putObject("requests");
    requests.put("first", first);
    requests.put("warmups", warmups);
    requests.put("measured", measured);
    job.put("timeout_s", timeoutS);
    job.put("precision_env", precisionEnv);
    job.put("artifact_dir", artifactDir);
    job.put("manifest_path", manifestPath);
    job.put("manifest_sha256", manifestSha256);
    job.put("repo_dir", repoDir);
    job.put("out_dir", outDir);
    if ("fixed-continuation".equals(mode)) {
      ObjectNode cont = job.putObject("fixed_continuation");
      cont.put("offset", fcOffset);
      cont.put("count", fcCount);
    }
    Files.writeString(file, MAPPER.writeValueAsString(job), StandardCharsets.UTF_8);
  }

  /** The worker directory name: unique per (runtime, adapter, case, length, mode) cell. */
  static String dirName(
      String runtime, String adapter, String caseName, int promptLength, String mode) {
    return String.join("-", runtime, adapter, caseName, Integer.toString(promptLength), mode);
  }

  /**
   * The per-runtime command the launcher forks for this job. The Java worker is the models module's
   * test-class {@code BaselineWorker} on the models test runtime classpath; the Python worker is
   * the pinned venv interpreter in isolated mode on the frozen worker source. The precision mode is
   * a worker-process environment variable the caller sets, not a command-line argument (JDK 25 has
   * no {@code System.setenv}; the Java worker verifies the environment).
   */
  List<String> command(Path jobFile) {
    List<String> command = new ArrayList<>();
    if ("java".equals(runtime)) {
      command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
      command.add("--enable-native-access=ALL-UNNAMED");
      command.add("-Djmlx.library.path=" + repoDir + "/native/install/lib");
      command.add("-cp");
      command.add(modelsTestClasspath);
      command.add("se.alipsa.jmlx.models.BaselineWorker");
    } else {
      command.add(repoDir + "/tools/decode-baseline/.venv/bin/python");
      command.add("-I");
      command.add(repoDir + "/tools/decode-baseline/worker.py");
    }
    command.add(jobFile.toString());
    return command;
  }

  /** A parsed launcher spec's worker entry (spec section: {@code workers}). */
  record WorkerEntry(
      String runtime, String adapter, String caseName, int promptLength, String mode) {}
}
