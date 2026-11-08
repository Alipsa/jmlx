package se.alipsa.jmlx.benchmarks;

import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Shared structural report fixtures for the launcher's pure-Java tests: one complete controlled
 * request record (8 tokens, 7 cached steps) whose timings are deterministic, so the validator and
 * the session summary can be asserted against hand-computed values.
 */
final class BaselineReportFixtures {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private BaselineReportFixtures() {}

  /**
   * One complete controlled request record: 8 tokens, 7 cached steps starting at position 33, event
   * times {@code ttft + 0.01 * k}.
   */
  static ObjectNode request(int index, String kind, double ttft) {
    ObjectNode request = MAPPER.createObjectNode();
    request.put("index", index);
    request.put("kind", kind);
    request.put("prime_s", 0.01);
    request.put("prime_excluded", true);
    request.put("prefill_s", 0.01);
    request.put("ttft_s", ttft);
    ArrayNode steps = request.putArray("steps");
    for (int k = 0; k < 7; k++) {
      steps.addObject().put("position", 33 + k).put("s", 0.01);
    }
    double[] eventS = new double[8];
    for (int k = 0; k < 8; k++) {
      eventS[k] = ttft + 0.01 * k;
    }
    ArrayNode events = request.putArray("event_s");
    for (double v : eventS) {
      events.add(v);
    }
    ArrayNode tokens = request.putArray("tokens");
    for (int k = 0; k < 8; k++) {
      tokens.add(0);
    }
    request.put("streaming_decode_tps", 7.0 / (eventS[7] - eventS[0]));
    request.put("full_generation_s", eventS[7]);
    request.put("emitted", 8);
    request.put("finish_reason", "controlled-max-tokens");
    request.put("unused_lookahead_forwards", 0);
    request.putNull("memory");
    request.put("io_affected", false);
    request.putNull("step_region_s");
    return request;
  }

  /**
   * One complete fixed-continuation request record: 32 measured steps at positions 32..63 (offset
   * 32, count 32 per the manifest rule), no emission, no ttft/full generation.
   */
  static ObjectNode fixedContinuationRequest(int index, String kind) {
    ObjectNode request = MAPPER.createObjectNode();
    request.put("index", index);
    request.put("kind", kind);
    request.put("prime_s", 0.01);
    request.put("prime_excluded", true);
    request.put("prefill_s", 0.02);
    request.putNull("ttft_s");
    ArrayNode steps = request.putArray("steps");
    for (int k = 0; k < 32; k++) {
      steps
          .addObject()
          .put("position", 32 + k)
          .put("s", 0.01)
          .put("argmax_id", 1 + (k % 3))
          .put("top2_margin", 0.1);
    }
    request.putArray("event_s");
    request.putArray("tokens");
    request.putNull("full_generation_s");
    request.put("emitted", 0);
    request.put("finish_reason", "fixed-continuation");
    request.put("unused_lookahead_forwards", 0);
    request.putNull("memory");
    request.put("io_affected", false);
    request.put("step_region_s", 0.32);
    return request;
  }

  /**
   * One complete stock request record: 8 tokens, host-visible event times only; prefill_s and steps
   * are null (stock runs the shipped loop and does not expose the controlled prefill/decode
   * contract).
   */
  static ObjectNode stockRequest(int index, String kind, double ttft, int unused) {
    ObjectNode request = MAPPER.createObjectNode();
    request.put("index", index);
    request.put("kind", kind);
    request.put("prime_s", 0.01);
    request.put("prime_excluded", true);
    request.putNull("prefill_s");
    request.put("ttft_s", ttft);
    request.putNull("steps");
    double[] eventS = new double[8];
    for (int k = 0; k < 8; k++) {
      eventS[k] = ttft + 0.01 * k;
    }
    ArrayNode events = request.putArray("event_s");
    for (double v : eventS) {
      events.add(v);
    }
    ArrayNode tokens = request.putArray("tokens");
    for (int k = 0; k < 8; k++) {
      tokens.add(0);
    }
    request.put("streaming_decode_tps", 7.0 / (eventS[7] - eventS[0]));
    request.put("full_generation_s", eventS[7]);
    request.put("emitted", 8);
    request.put("finish_reason", "stop");
    request.put("unused_lookahead_forwards", unused);
    request.putNull("memory");
    request.put("io_affected", false);
    request.putNull("step_region_s");
    return request;
  }

  /**
   * A structurally complete worker report block (the {@code worker} object of schema {@code
   * jmlx-decode-baseline-1}) with the given status and request records.
   */
  static ObjectNode workerBlock(
      String runtime,
      String adapter,
      String caseName,
      int promptLength,
      String mode,
      String status,
      String failureReason,
      String manifestSha,
      ObjectNode... requests) {
    ObjectNode worker = MAPPER.createObjectNode();
    worker.put("runtime", runtime);
    worker.put("pid", 4242);
    worker.put("adapter", adapter);
    worker.put("case", caseName);
    worker.put("prompt_length", promptLength);
    worker.put("mode", mode);
    worker.put("status", status);
    worker.putNull("failure_reason");
    if (failureReason != null) {
      worker.put("failure_reason", failureReason);
    }
    worker.put("utc_start", "2026-11-07T00:00:00Z");
    worker.put("utc_end", "2026-11-07T00:00:10Z");
    ObjectNode identity = worker.putObject("identity");
    ObjectNode code = identity.putObject("code");
    code.put("sha", "s".repeat(40));
    code.put("branch", "main");
    code.put("dirty", false);
    identity.put("manifest_sha256", manifestSha);
    identity.putObject("run_policy").put("worker_timeout_s", 300);
    identity.put("precision_env", "MLX_ENABLE_TF32=1");
    if ("java".equals(runtime)) {
      identity.putObject("java").put("version", "25");
    } else {
      identity.putObject("python").put("version", "3.12.16");
    }
    identity.putObject("device").put("memory_size_bytes", 137438953472L);
    identity.put("wired_limit_policy", "none");
    identity.put("cache_regime", "primed-per-request");
    identity.put("io_affected", false);
    identity.put("instrumented", false);
    worker.putArray("stop_token_ids").add(3);
    worker.put("load_s", 0.3);
    worker.put("load_state", "materialized");
    ArrayNode requestsNode = worker.putArray("requests");
    for (ObjectNode request : requests) {
      requestsNode.add(request);
    }
    worker.putArray("notes");
    return worker;
  }

  /** The full report object ({@code schema} + {@code worker}). */
  static ObjectNode report(ObjectNode worker) {
    ObjectNode report = MAPPER.createObjectNode();
    report.put("schema", BaselineReportValidator.SCHEMA);
    report.set("worker", worker);
    return report;
  }
}
