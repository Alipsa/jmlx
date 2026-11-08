package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Schema validation of worker reports against the frozen jmlx-decode-baseline-1 contract
 * (req/plans/phase8-2-reference-findings.md section 13.3): every valid shape returns no violations,
 * and every structural defect is reported, never silently corrected.
 */
class BaselineReportValidatorTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final String SHA = "a".repeat(64);

  private static ObjectNode baseControlledJava() {
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java",
            "controlled",
            "tiny",
            32,
            "headline",
            "ok",
            null,
            SHA,
            BaselineReportFixtures.request(0, "first", 0.02),
            BaselineReportFixtures.request(1, "measured", 0.03));
    return BaselineReportFixtures.report(worker);
  }

  private static ObjectNode worker(ObjectNode report) {
    return (ObjectNode) report.get("worker");
  }

  private static ObjectNode request(ObjectNode report, int index) {
    return (ObjectNode) ((ArrayNode) worker(report).get("requests")).get(index);
  }

  private static List<String> violations(ObjectNode report) {
    return BaselineReportValidator.validate(report);
  }

  private static void assertViolation(ObjectNode report, String expected) {
    List<String> violations = violations(report);
    assertTrue(
        violations.stream().anyMatch(v -> v.contains(expected)),
        "expected a violation containing '" + expected + "' in " + violations);
  }

  @Test
  void validControlledJava() {
    assertTrue(violations(baseControlledJava()).isEmpty());
  }

  @Test
  void validStockPython() {
    ObjectNode request = BaselineReportFixtures.stockRequest(0, "measured", 0.05, 1);
    // A non-empty python stock run drains exactly one queued lookahead forward.
    request.put("finish_reason", "length");
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "python", "stock", "tiny", 32, "headline", "ok", null, SHA, request);
    assertTrue(violations(BaselineReportFixtures.report(worker)).isEmpty());
  }

  @Test
  void validStockJava() {
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java",
            "stock",
            "tiny",
            32,
            "headline",
            "ok",
            null,
            SHA,
            BaselineReportFixtures.stockRequest(0, "measured", 0.05, 0));
    assertTrue(violations(BaselineReportFixtures.report(worker)).isEmpty());
  }

  @Test
  void stockShapesStayNull() {
    // A controlled-shaped request inside a stock worker violates the stock null contract.
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "python",
            "stock",
            "tiny",
            32,
            "headline",
            "ok",
            null,
            SHA,
            BaselineReportFixtures.request(0, "measured", 0.05));
    ObjectNode report = BaselineReportFixtures.report(worker);
    assertViolation(report, "stock steps is null");
    assertViolation(report, "stock prefill_s is null");
  }

  @Test
  void controlledPrefillMustBeNumber() {
    ObjectNode report = baseControlledJava();
    request(report, 0).putNull("prefill_s");
    assertViolation(report, "controlled prefill_s must be a number");
  }

  @Test
  void validFixedContinuation() {
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java",
            "controlled",
            "tiny",
            32,
            "fixed-continuation",
            "ok",
            null,
            SHA,
            BaselineReportFixtures.fixedContinuationRequest(0, "measured"));
    assertTrue(violations(BaselineReportFixtures.report(worker)).isEmpty());
  }

  @Test
  void validReferencePair() {
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java",
            "controlled",
            "tiny",
            32,
            "reference-pair",
            "ok",
            null,
            SHA,
            BaselineReportFixtures.request(0, "measured", 0.05));
    assertTrue(violations(BaselineReportFixtures.report(worker)).isEmpty());
  }

  @Test
  void nonObjectReport() {
    List<String> violations = BaselineReportValidator.validate(MAPPER.createArrayNode());
    assertTrue(violations.stream().anyMatch(v -> v.contains("not a JSON object")));
    assertTrue(
        BaselineReportValidator.validate(null).stream()
            .anyMatch(v -> v.contains("not a JSON object")));
  }

  @Test
  void missingWorkerBlock() {
    ObjectNode report = MAPPER.createObjectNode();
    report.put("schema", BaselineReportValidator.SCHEMA);
    assertViolation(report, "worker block is missing");
  }

  @Test
  void wrongSchema() {
    ObjectNode report = baseControlledJava();
    report.put("schema", "other-schema");
    assertViolation(report, "schema must be " + BaselineReportValidator.SCHEMA);
  }

  @Test
  void okWithFailureReason() {
    ObjectNode report = baseControlledJava();
    worker(report).put("failure_reason", "boom");
    assertViolation(report, "an ok report carries no failure_reason");
  }

  @Test
  void nonOkWithoutFailureReason() {
    ObjectNode report = baseControlledJava();
    worker(report).put("status", "failed");
    assertViolation(report, "a non-ok report carries a textual failure_reason");
  }

  @Test
  void stepCountMismatch() {
    ObjectNode report = baseControlledJava();
    request(report, 0).put("emitted", 9);
    assertViolation(report, "steps must have emitted - 1 entries");
  }

  @Test
  void eventCountMismatch() {
    ObjectNode report = baseControlledJava();
    ObjectNode request = request(report, 0);
    request.putNull("streaming_decode_tps");
    ((ArrayNode) request.get("event_s")).remove(7);
    assertViolation(report, "event_s must have emitted entries");
  }

  @Test
  void ttftMismatch() {
    ObjectNode report = baseControlledJava();
    request(report, 0).put("ttft_s", 0.5);
    assertViolation(report, "ttft_s must equal event_s[0]");
  }

  @Test
  void fullGenerationMismatch() {
    ObjectNode report = baseControlledJava();
    request(report, 0).put("full_generation_s", 1.0);
    assertViolation(report, "full_generation_s must be the last host-visible event time");
  }

  @Test
  void streamingTpsMismatch() {
    ObjectNode report = baseControlledJava();
    request(report, 0).put("streaming_decode_tps", 999.0);
    assertViolation(report, "streaming_decode_tps must be (emitted - 1) / (last - first)");
  }

  @Test
  void controlledUnusedLookahead() {
    ObjectNode report = baseControlledJava();
    request(report, 0).put("unused_lookahead_forwards", 1);
    assertViolation(report, "controlled requests report zero unused lookahead forwards");
  }

  @Test
  void javaStockUnusedLookahead() {
    ObjectNode request = BaselineReportFixtures.stockRequest(0, "measured", 0.05, 1);
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "stock", "tiny", 32, "headline", "ok", null, SHA, request);
    assertViolation(BaselineReportFixtures.report(worker), "java stock reports zero unused");
  }

  @Test
  void pythonStockDrainWhenNonEmpty() {
    // unused 0 on a non-empty run violates the drain-one rule.
    ObjectNode request = BaselineReportFixtures.stockRequest(0, "measured", 0.05, 0);
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "python", "stock", "tiny", 32, "headline", "ok", null, SHA, request);
    assertViolation(
        BaselineReportFixtures.report(worker),
        "python stock drains one queued lookahead iff the run is non-empty");
  }

  @Test
  void pythonStockDrainOnEmptyRun() {
    ObjectNode request = BaselineReportFixtures.stockRequest(0, "measured", 0.0, 1);
    request.putNull("ttft_s");
    request.putNull("streaming_decode_tps");
    ((ArrayNode) request.get("event_s")).removeAll();
    ((ArrayNode) request.get("tokens")).removeAll();
    request.putNull("full_generation_s");
    request.put("emitted", 0);
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "python", "stock", "tiny", 32, "headline", "ok", null, SHA, request);
    assertViolation(
        BaselineReportFixtures.report(worker),
        "python stock drains one queued lookahead iff the run is non-empty");
  }

  @Test
  void memoryMustBeNull() {
    ObjectNode report = baseControlledJava();
    request(report, 0).put("memory", 1.5);
    assertViolation(report, "memory is null when the worker is not instrumented");
  }

  @Test
  void primeExcludedMustBeTrue() {
    ObjectNode report = baseControlledJava();
    request(report, 0).put("prime_excluded", false);
    assertViolation(report, "prime_excluded must be true");
  }

  @Test
  void requestIndexMismatch() {
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java",
            "controlled",
            "tiny",
            32,
            "headline",
            "ok",
            null,
            SHA,
            BaselineReportFixtures.request(0, "first", 0.02),
            BaselineReportFixtures.request(0, "measured", 0.03));
    assertViolation(BaselineReportFixtures.report(worker), "requests[1].index must be 1");
  }

  @Test
  void stepRegionNullOutsideFixedContinuation() {
    ObjectNode report = baseControlledJava();
    request(report, 0).put("step_region_s", 0.07);
    assertViolation(report, "step_region_s is null outside fixed-continuation");
  }

  @Test
  void fixedContinuationStepCount() {
    ObjectNode fc = BaselineReportFixtures.fixedContinuationRequest(0, "measured");
    ((ArrayNode) fc.get("steps")).remove(31);
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "fixed-continuation", "ok", null, SHA, fc);
    assertViolation(BaselineReportFixtures.report(worker), "fixed-continuation steps exactly 32");
  }

  @Test
  void referencePairTwoRequests() {
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java",
            "controlled",
            "tiny",
            32,
            "reference-pair",
            "ok",
            null,
            SHA,
            BaselineReportFixtures.request(0, "measured", 0.05),
            BaselineReportFixtures.request(1, "measured", 0.06));
    assertViolation(
        BaselineReportFixtures.report(worker),
        "reference-pair reports exactly one measured request");
  }

  @Test
  void nonObjectWorker() {
    ObjectNode report = baseControlledJava();
    report.set("worker", MAPPER.createArrayNode());
    assertViolation(report, "worker block is missing");
  }
}
