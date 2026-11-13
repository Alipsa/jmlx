package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Pure-Java tests of the worker's post-capture verdict and job parsing (plan section 6:
 * "worker-failure tests"). No native runtime, no real model: the verdict is a static pure function
 * and the job is the launcher -> worker handshake record.
 */
class VlmFirstUseWorkerTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void allCorrectIsOk() {
    VlmFirstUseWorker.Verdict verdict =
        VlmFirstUseWorker.postCaptureVerdict(true, true, true, List.of());
    assertEquals("ok", verdict.status());
    assertEquals(VlmFirstUseWorker.EXIT_OK, verdict.exitCode());
    assertNull(verdict.failureReason());
  }

  @Test
  void eachFirstRequestMismatchIsACorrectnessFailure() {
    VlmFirstUseWorker.Verdict ids =
        VlmFirstUseWorker.postCaptureVerdict(false, true, true, List.of());
    assertCorrectnessFailure(ids, "target_ids");

    VlmFirstUseWorker.Verdict finish =
        VlmFirstUseWorker.postCaptureVerdict(true, false, true, List.of());
    assertCorrectnessFailure(finish, "finish_reason");

    VlmFirstUseWorker.Verdict positions =
        VlmFirstUseWorker.postCaptureVerdict(true, true, false, List.of());
    assertCorrectnessFailure(positions, "expanded_ids");
  }

  @Test
  void warmAndMeasuredDivergencesAreLabeled() {
    VlmFirstUseWorker.Verdict warm =
        VlmFirstUseWorker.postCaptureVerdict(
            true,
            true,
            true,
            List.of(new VlmFirstUseWorker.WarmDivergence("warmup 0", List.of(1, 2))));
    assertCorrectnessFailure(warm, "warmup 0");

    VlmFirstUseWorker.Verdict measured =
        VlmFirstUseWorker.postCaptureVerdict(
            true,
            true,
            true,
            List.of(new VlmFirstUseWorker.WarmDivergence("measured 1", List.of(3, 4))));
    assertCorrectnessFailure(measured, "measured 1");
  }

  @Test
  void combinedProblemsAreAllPresent() {
    VlmFirstUseWorker.Verdict verdict =
        VlmFirstUseWorker.postCaptureVerdict(
            false,
            false,
            false,
            List.of(
                new VlmFirstUseWorker.WarmDivergence("warmup 2", List.of(1)),
                new VlmFirstUseWorker.WarmDivergence("measured 0", List.of(2))));
    assertEquals("correctness_failed", verdict.status());
    assertEquals(VlmFirstUseWorker.EXIT_CORRECTNESS_FAILED, verdict.exitCode());
    String reason = verdict.failureReason();
    assertNotNull(reason);
    assertTrue(reason.contains("target_ids"), reason);
    assertTrue(reason.contains("finish_reason"), reason);
    assertTrue(reason.contains("expanded_ids"), reason);
    assertTrue(reason.contains("warmup 2"), reason);
    assertTrue(reason.contains("measured 0"), reason);
  }

  /** Asserts a correctness failure (exit 3) whose reason names the given problem. */
  private static void assertCorrectnessFailure(VlmFirstUseWorker.Verdict verdict, String problem) {
    assertEquals("correctness_failed", verdict.status());
    assertEquals(VlmFirstUseWorker.EXIT_CORRECTNESS_FAILED, verdict.exitCode());
    assertNotNull(verdict.failureReason());
    assertTrue(verdict.failureReason().contains(problem), verdict.failureReason());
  }

  /** A full valid job in the launcher's handshake shape. */
  private static ObjectNode validJob() {
    ObjectNode job = MAPPER.createObjectNode();
    job.put("schema", VlmFirstUseWorker.JOB_SCHEMA);
    job.put("workload", "qwen3vl-2b");
    job.put("tree", "new");
    job.put("tree_path", "/trees/new");
    job.put("pair_index", 2);
    job.put("model_dir", "/models/qwen3vl");
    job.put("manifest_path", "/repo/tools/tier-b/qwen3vl.json");
    job.put("manifest_sha256", "a".repeat(64));
    job.putArray("image_paths").add("/repo/images/a.png");
    job.putArray("image_sha256s").add("b".repeat(64));
    job.put("max_new_tokens", 8);
    job.putArray("stop_token_ids").add(151645);
    job.put("warmups", 3);
    job.put("measured", 5);
    job.put("worker_timeout_s", 900);
    job.put("precision_env", "MLX_ENABLE_TF32");
    job.put("precision_value", "0");
    job.put("fork_utc", "2026-11-11T00:00:00Z");
    job.put("out_dir", "/out/qwen3vl-2b/new-p2");
    return job;
  }

  @Test
  void validJobParses() {
    VlmFirstUseWorker.Job job = VlmFirstUseWorker.Job.parse(validJob());
    assertEquals("qwen3vl-2b", job.workload());
    assertEquals("new", job.tree());
    assertEquals(2, job.pairIndex());
    assertEquals(Path.of("/repo/tools/tier-b/qwen3vl.json"), job.manifestPath());
    assertEquals("a".repeat(64), job.manifestSha256());
    assertEquals(List.of("b".repeat(64)), job.imageSha256s());
    assertEquals(List.of(151645), job.stopTokenIds());
    assertEquals(3, job.warmups());
    assertEquals(5, job.measured());
    assertEquals(900, job.workerTimeoutS());
  }

  @Test
  void missingSchemaIsRejected() {
    ObjectNode job = validJob();
    job.remove("schema");
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> VlmFirstUseWorker.Job.parse(job));
    assertTrue(e.getMessage().contains(VlmFirstUseWorker.JOB_SCHEMA), e.getMessage());
  }

  @Test
  void wrongSchemaIsRejected() {
    ObjectNode job = validJob();
    job.put("schema", "jmlx-vlm-first-use-job-0");
    assertThrows(IllegalArgumentException.class, () -> VlmFirstUseWorker.Job.parse(job));
  }
}
