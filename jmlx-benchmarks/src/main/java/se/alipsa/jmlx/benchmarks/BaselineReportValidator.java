package se.alipsa.jmlx.benchmarks;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import tools.jackson.databind.JsonNode;

/**
 * Schema validation of a worker report (schema {@code jmlx-decode-baseline-1}; frozen contract in
 * req/plans/phase8-2-reference-findings.md section 13.3). The launcher validates every report
 * before embedding it in the session report; a structural violation makes the worker's cell
 * ineligible (excluded and recorded), never silently corrected. Returns the list of violations; an
 * empty list is a valid report.
 */
final class BaselineReportValidator {
  static final String SCHEMA = "jmlx-decode-baseline-1";
  private static final Set<String> RUNTIMES = Set.of("java", "python");
  private static final Set<String> ADAPTERS = Set.of("controlled", "stock");
  private static final Set<String> MODES =
      Set.of("headline", "fixed-continuation", "reference-pair");
  private static final Set<String> STATUSES = Set.of("ok", "failed", "timeout");
  private static final Set<String> KINDS = Set.of("first", "warmup", "measured");
  private static final Set<String> FINISH_REASONS =
      Set.of("controlled-max-tokens", "stop", "length", "fixed-continuation", "cancelled");
  private static final double TOLERANCE = 1e-9;

  private BaselineReportValidator() {}

  static List<String> validate(JsonNode report) {
    List<String> violations = new ArrayList<>();
    if (report == null || !report.isObject()) {
      return List.of("report is not a JSON object");
    }
    if (!SCHEMA.equals(report.path("schema").asString(null))) {
      violations.add("schema must be " + SCHEMA);
    }
    JsonNode worker = report.path("worker");
    if (!worker.isObject()) {
      violations.add("worker block is missing");
      return violations;
    }
    if (!RUNTIMES.contains(worker.path("runtime").asString(null))) {
      violations.add("runtime must be java or python");
    }
    if (worker.path("pid").asLong(0) <= 0) {
      violations.add("pid must be a positive integer");
    }
    if (!ADAPTERS.contains(worker.path("adapter").asString(null))) {
      violations.add("adapter must be controlled or stock");
    }
    if (worker.path("prompt_length").asInt(0) <= 0) {
      violations.add("prompt_length must be a positive integer");
    }
    String mode = worker.path("mode").asString(null);
    if (!MODES.contains(mode)) {
      violations.add("mode must be headline, fixed-continuation or reference-pair");
    }
    String status = worker.path("status").asString(null);
    if (!STATUSES.contains(status)) {
      violations.add("status must be ok, failed or timeout");
    }
    if ("ok".equals(status) && !worker.path("failure_reason").isNull()) {
      violations.add("an ok report carries no failure_reason");
    }
    if (!"ok".equals(status) && !worker.path("failure_reason").isString()) {
      violations.add("a non-ok report carries a textual failure_reason");
    }
    if (worker.path("utc_start").asString("").isEmpty()
        || worker.path("utc_end").asString("").isEmpty()) {
      violations.add("utc_start and utc_end are required");
    }
    JsonNode identity = worker.path("identity");
    if (!identity.path("manifest_sha256").isString()
        || identity.path("manifest_sha256").asString("").isEmpty()) {
      violations.add("identity.manifest_sha256 is required");
    }
    if (!identity.path("run_policy").isObject()) {
      violations.add("identity.run_policy is required");
    }
    if (!identity.path("device").isObject()) {
      violations.add("identity.device is required");
    }
    if (!identity.path("code").isObject()) {
      violations.add("identity.code is required");
    }
    if (!worker.path("stop_token_ids").isArray()) {
      violations.add("stop_token_ids must be an array");
    }
    if (!worker.path("load_s").isDouble() && !worker.path("load_s").isNull()) {
      violations.add("load_s must be a number or null");
    }
    if (!worker.path("load_state").isString() && !worker.path("load_state").isNull()) {
      violations.add("load_state must be text or null");
    }
    if (!worker.path("notes").isArray()) {
      violations.add("notes must be an array");
    }
    JsonNode requests = worker.path("requests");
    if (!requests.isArray()) {
      violations.add("requests must be an array");
      return violations;
    }
    if ("reference-pair".equals(mode) && requests.size() != 1) {
      violations.add("reference-pair reports exactly one measured request");
    }
    for (int i = 0; i < requests.size(); i++) {
      validateRequest(
          requests.get(i),
          i,
          mode,
          worker.path("adapter").asString(null),
          worker.path("runtime").asString(null),
          violations);
    }
    return violations;
  }

  private static void validateRequest(
      JsonNode request,
      int index,
      String mode,
      String adapter,
      String runtime,
      List<String> violations) {
    String prefix = "requests[" + index + "].";
    if (!request.isObject()) {
      violations.add(prefix + "is not an object");
      return;
    }
    if (request.path("index").asInt(-1) != index) {
      violations.add(prefix + "index must be " + index);
    }
    if (!KINDS.contains(request.path("kind").asString(null))) {
      violations.add(prefix + "kind must be first, warmup or measured");
    }
    if (!request.path("prime_s").isDouble()) {
      violations.add(prefix + "prime_s must be a number");
    }
    if (!request.path("prime_excluded").asBoolean(false)) {
      violations.add(prefix + "prime_excluded must be true");
    }
    if (!request.path("memory").isNull()) {
      violations.add(prefix + "memory is null in WP1");
    }
    if (!request.path("io_affected").isBoolean()) {
      violations.add(prefix + "io_affected must be a boolean");
    }
    if ("fixed-continuation".equals(mode)) {
      validateFixedContinuation(request, prefix, violations);
      return;
    }
    int emitted = request.path("emitted").asInt(-1);
    if (emitted < 0) {
      violations.add(prefix + "emitted must be a nonnegative integer");
      return;
    }
    boolean controlled = "controlled".equals(adapter);
    JsonNode steps = request.path("steps");
    JsonNode eventS = request.path("event_s");
    JsonNode tokens = request.path("tokens");
    if (controlled) {
      // The controlled adapter exposes the prefill/decode contract of plan section 4.
      if (!request.path("prefill_s").isDouble()) {
        violations.add(prefix + "controlled prefill_s must be a number");
      }
      if (!steps.isArray()) {
        violations.add(prefix + "controlled steps must be an array");
      } else if (emitted > 0 && steps.size() != emitted - 1) {
        violations.add(prefix + "steps must have emitted - 1 entries (controlled loop)");
      }
      for (int k = 0; steps.isArray() && k < steps.size(); k++) {
        if (!steps.get(k).path("position").isIntegralNumber()) {
          violations.add(prefix + "steps[" + k + "].position must be an integer");
        }
        if (!steps.get(k).path("s").isDouble()) {
          violations.add(prefix + "steps[" + k + "].s must be a number");
        }
      }
    } else {
      // Stock runs the shipped loop; the controlled prefill/decode fields stay null.
      if (!steps.isNull()) {
        violations.add(prefix + "stock steps is null");
      }
      if (!request.path("prefill_s").isNull()) {
        violations.add(prefix + "stock prefill_s is null");
      }
    }
    if (!eventS.isArray() || !tokens.isArray()) {
      violations.add(prefix + "event_s and tokens must be arrays");
      return;
    }
    if (emitted == 0) {
      boolean emptySteps = !controlled || steps.size() == 0;
      if (!emptySteps || eventS.size() != 0 || tokens.size() != 0) {
        violations.add(prefix + "a zero-emission request carries empty steps/event_s/tokens");
      }
      if (!request.path("ttft_s").isNull()) {
        violations.add(prefix + "ttft_s is null when nothing was emitted");
      }
    } else {
      if (eventS.size() != emitted) {
        violations.add(prefix + "event_s must have emitted entries");
      }
      if (tokens.size() != emitted) {
        violations.add(prefix + "tokens must have emitted entries");
      }
      double ttft =
          request.path("ttft_s").isDouble() ? request.path("ttft_s").asDouble() : Double.NaN;
      if (!request.path("ttft_s").isDouble()) {
        violations.add(prefix + "ttft_s must be a number when tokens were emitted");
      } else if (eventS.size() == emitted
          && Math.abs(ttft - eventS.get(0).asDouble()) > TOLERANCE) {
        violations.add(prefix + "ttft_s must equal event_s[0]");
      }
      JsonNode fullGeneration = request.path("full_generation_s");
      if (!fullGeneration.isDouble()) {
        violations.add(prefix + "full_generation_s must be a number when tokens were emitted");
      } else if (eventS.size() == emitted
          && Math.abs(fullGeneration.asDouble() - eventS.get(emitted - 1).asDouble()) > 0.05) {
        violations.add(prefix + "full_generation_s must be the last host-visible event time");
      }
    }
    if (!FINISH_REASONS.contains(request.path("finish_reason").asString(null))) {
      violations.add(prefix + "finish_reason is not a frozen label");
    }
    // Uniform key set: exactly one of full_generation_s / step_region_s is valued per mode;
    // outside fixed-continuation the step-region wall is null.
    if (!request.path("step_region_s").isNull()) {
      violations.add(prefix + "step_region_s is null outside fixed-continuation");
    }
    JsonNode unused = request.path("unused_lookahead_forwards");
    if (!unused.isIntegralNumber()) {
      violations.add(prefix + "unused_lookahead_forwards must be an integer");
    } else if (controlled) {
      if (unused.asInt() != 0) {
        violations.add(prefix + "controlled requests report zero unused lookahead forwards");
      }
    } else if ("java".equals(runtime)) {
      if (unused.asInt() != 0) {
        violations.add(prefix + "java stock reports zero unused lookahead forwards");
      }
    } else {
      // Python stock drains the one queued lookahead iff the run is non-empty (worker.py);
      // this holds for zero-emission runs too, so it is checked before any shape shortcut.
      boolean nonEmpty = emitted > 0;
      boolean one = unused.asInt() == 1;
      if (nonEmpty != one) {
        violations.add(
            prefix + "python stock drains one queued lookahead iff the run is non-empty");
      }
    }
    if (request.path("streaming_decode_tps").isDouble()
        && emitted >= 2
        && eventS.size() == emitted) {
      double expected =
          (emitted - 1) / (eventS.get(emitted - 1).asDouble() - eventS.get(0).asDouble());
      if (Math.abs(request.path("streaming_decode_tps").asDouble() - expected) > 1e-9) {
        violations.add(
            prefix + "streaming_decode_tps must be (emitted - 1) / (last - first) event time");
      }
    }
  }

  private static void validateFixedContinuation(
      JsonNode request, String prefix, List<String> violations) {
    if (request.path("emitted").asInt(1) != 0) {
      violations.add(prefix + "fixed-continuation requests emit nothing");
    }
    if (!request.path("event_s").isArray() || request.path("event_s").size() != 0) {
      violations.add(prefix + "fixed-continuation carries an empty event_s");
    }
    if (!request.path("tokens").isArray() || request.path("tokens").size() != 0) {
      violations.add(prefix + "fixed-continuation carries empty tokens");
    }
    if (!request.path("ttft_s").isNull()) {
      violations.add(prefix + "fixed-continuation ttft_s is null");
    }
    if (!request.path("full_generation_s").isNull()) {
      violations.add(prefix + "fixed-continuation full_generation_s is null");
    }
    if (!request.path("step_region_s").isDouble()
        || request.path("step_region_s").asDouble() <= 0.0) {
      violations.add(prefix + "fixed-continuation step_region_s must be a positive number");
    }
    if (!request.path("prefill_s").isDouble()) {
      violations.add(prefix + "fixed-continuation prefill_s must be a number");
    }
    JsonNode steps = request.path("steps");
    if (!steps.isArray() || steps.size() != 32) {
      violations.add(prefix + "fixed-continuation steps exactly 32 entries");
      return;
    }
    for (int k = 0; k < steps.size(); k++) {
      JsonNode step = steps.get(k);
      if (!step.path("position").isIntegralNumber()
          || !step.path("s").isDouble()
          || !step.path("argmax_id").isIntegralNumber()
          || !step.path("top2_margin").isDouble()) {
        violations.add(prefix + "steps[" + k + "] must carry position/s/argmax_id/top2_margin");
      }
    }
    if (!"fixed-continuation".equals(request.path("finish_reason").asString(null))) {
      violations.add(prefix + "fixed-continuation finish_reason label");
    }
  }
}
