package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Pure-Java tests of the FFM transition estimate's descriptor normalization and its budget/coverage
 * accounting against hand-computed values on synthetic binding-count reports (no native runtime and
 * no compiler needed).
 */
class FfmTransitionEstimateTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final String MATMUL_DESCRIPTOR =
      "(Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;"
          + "Ljava/lang/foreign/MemorySegment;Ljava/lang/foreign/MemorySegment;)I";
  private static final String EVAL_DESCRIPTOR = "(J)V";

  @Test
  void normalizeMapsWrapperDescriptorsToShapeKeys() {
    assertEquals("(MMMM)I", FfmTransitionEstimate.normalize(MATMUL_DESCRIPTOR));
    assertEquals("(J)V", FfmTransitionEstimate.normalize(EVAL_DESCRIPTOR));
    assertEquals(
        "(AM)M",
        FfmTransitionEstimate.normalize(
            "(Ljava/lang/foreign/SegmentAllocator;Ljava/lang/foreign/MemorySegment;)"
                + "Ljava/lang/foreign/MemorySegment;"));
    assertEquals("(I)M", FfmTransitionEstimate.normalize("(I)Ljava/lang/foreign/MemorySegment;"));
    assertNull(FfmTransitionEstimate.normalize("(Ljava/lang/String;I)I"));
    assertNull(FfmTransitionEstimate.normalize("int(int)"));
    assertNull(FfmTransitionEstimate.normalize(null));
  }

  @Test
  void estimatesPerStepBudgetWithCoverage() {
    ObjectNode request = BaselineReportFixtures.request(0, "measured", 0.02);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode counts = MAPPER.createObjectNode();
    ArrayNode phases = counts.putArray("phases");
    // The controlled adapter's full window set, including the release windows (findings 15.6):
    // prefill phases, one step-phase window per phase per position, and the request release.
    phase(phases, "prefill-sync", 0, 4, 1, 0);
    phase(phases, "prefill-release", 0, 0, 2, 0);
    phase(phases, "construction", 33, 1, 0, 0);
    phase(phases, "sync", 33, 0, 1, 0);
    phase(phases, "selection", 33, 0, 1, 0);
    phase(phases, "release", 33, 0, 1, 0);
    phase(phases, "sync", 34, 2, 0, 0);
    phase(phases, "release", 34, 1, 0, 0);
    phase(phases, "request-release", 0, 0, 3, 0);
    counts.putObject("total").put("mlx_matmul", 8L).put("mlx_eval", 8L);
    request.set("binding_counts", counts);

    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ((ObjectNode) worker.get("binding_signatures")).put("mlx_matmul", MATMUL_DESCRIPTOR);

    ObjectNode estimate =
        FfmTransitionEstimate.estimate(BaselineReportFixtures.report(worker), calibrationSummary());
    assertEquals("jmlx-ffm-transition-estimate-1", estimate.path("schema").asString());
    assertEquals(1, estimate.path("measured_requests").asInt());
    assertEquals(17L, estimate.path("totals").path("binding_calls").asLong());
    assertEquals(17L, estimate.path("totals").path("classified_calls").asLong());
    assertEquals(1.0, estimate.path("totals").path("coverage").asDouble());
    assertEquals(0, estimate.path("uncovered").size());

    // The single measured request's step-phase sample per position, including its release window:
    // step 33 = 1 matmul @ 90 ns + 3 evals @ 40 ns; step 34 = 3 matmuls @ 90 ns.
    assertEquals(2, estimate.path("per_decode_step").path("n").asInt());
    assertEquals(
        33, estimate.path("per_decode_step").path("positions").get(0).path("position").asInt());
    assertEquals(
        210, estimate.path("per_decode_step").path("positions").get(0).path("median_ns").asLong());
    assertEquals(
        270, estimate.path("per_decode_step").path("positions").get(1).path("median_ns").asLong());
    // Pooled over the (request, position) samples [210, 270]: type-7 p10/median/p90.
    assertEquals(216.0, estimate.path("per_decode_step").path("p10_ns").asDouble());
    assertEquals(240.0, estimate.path("per_decode_step").path("median_ns").asDouble());
    assertEquals(264.0, estimate.path("per_decode_step").path("p90_ns").asDouble());

    // Prefill = the prefill-* windows only (the request-release at position 0 is in neither
    // prefill nor any step).
    assertEquals(1, estimate.path("prefill").path("n").asInt());
    assertEquals(480, estimate.path("prefill").path("median_ns").asLong());
    assertEquals(1, estimate.path("request_total").path("n").asInt());
    assertEquals(1080, estimate.path("request_total").path("median_ns").asLong());

    assertEquals(8L, estimate.path("per_shape").path("(MMMM)I").path("calls").asLong());
    assertEquals(720, estimate.path("per_shape").path("(MMMM)I").path("budget_ns").asLong());
    assertEquals(9L, estimate.path("per_shape").path("(J)V").path("calls").asLong());
    assertEquals(360, estimate.path("per_shape").path("(J)V").path("budget_ns").asLong());
    assertTrue(estimate.path("assumptions").size() >= 4);
  }

  @Test
  void poolsStepBudgetsPerRequestNotAcrossRequests() {
    // Two measured requests with different per-step costs at the same position: the per-step
    // value is the quantile over the requests' own samples [40, 90], never their sum (the
    // N-times-too-large pooling bug of findings section 15.6). A warmup request with large
    // counts is excluded from every aggregate.
    ObjectNode first = BaselineReportFixtures.request(0, "measured", 0.02);
    BaselineReportFixtures.instrument(first, "binding-count");
    ObjectNode countsA = MAPPER.createObjectNode();
    ArrayNode phasesA = countsA.putArray("phases");
    phase(phasesA, "prefill-sync", 0, 0, 1, 0);
    phase(phasesA, "construction", 33, 0, 1, 0);
    first.set("binding_counts", countsA);

    ObjectNode second = BaselineReportFixtures.request(1, "measured", 0.03);
    BaselineReportFixtures.instrument(second, "binding-count");
    ObjectNode countsB = MAPPER.createObjectNode();
    ArrayNode phasesB = countsB.putArray("phases");
    phase(phasesB, "prefill-sync", 0, 1, 0, 0);
    phase(phasesB, "construction", 33, 1, 0, 0);
    second.set("binding_counts", countsB);

    ObjectNode warmup = BaselineReportFixtures.request(2, "warmup", 0.04);
    BaselineReportFixtures.instrument(warmup, "binding-count");
    ObjectNode countsW = MAPPER.createObjectNode();
    ArrayNode phasesW = countsW.putArray("phases");
    phase(phasesW, "prefill-sync", 0, 100, 0, 0);
    phase(phasesW, "construction", 33, 100, 0, 0);
    warmup.set("binding_counts", countsW);

    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", first, second, warmup);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ((ObjectNode) worker.get("binding_signatures")).put("mlx_matmul", MATMUL_DESCRIPTOR);

    ObjectNode estimate =
        FfmTransitionEstimate.estimate(BaselineReportFixtures.report(worker), calibrationSummary());
    assertEquals(2, estimate.path("measured_requests").asInt());
    // The warmup's 200 matmul calls are nowhere in the aggregates.
    assertEquals(4L, estimate.path("totals").path("binding_calls").asLong());
    assertEquals(4L, estimate.path("totals").path("classified_calls").asLong());
    assertEquals(1.0, estimate.path("totals").path("coverage").asDouble());

    assertEquals(1, estimate.path("per_decode_step").path("positions").size());
    assertEquals(
        65.0,
        estimate.path("per_decode_step").path("positions").get(0).path("median_ns").asDouble());
    assertEquals(45.0, estimate.path("per_decode_step").path("p10_ns").asDouble());
    assertEquals(65.0, estimate.path("per_decode_step").path("median_ns").asDouble());
    assertEquals(85.0, estimate.path("per_decode_step").path("p90_ns").asDouble());

    assertEquals(2, estimate.path("prefill").path("n").asInt());
    assertEquals(65.0, estimate.path("prefill").path("median_ns").asDouble());
    assertEquals(2, estimate.path("request_total").path("n").asInt());
    assertEquals(130.0, estimate.path("request_total").path("median_ns").asDouble());

    assertEquals(2L, estimate.path("per_shape").path("(J)V").path("calls").asLong());
    assertEquals(80, estimate.path("per_shape").path("(J)V").path("budget_ns").asLong());
    assertEquals(2L, estimate.path("per_shape").path("(MMMM)I").path("calls").asLong());
    assertEquals(180, estimate.path("per_shape").path("(MMMM)I").path("budget_ns").asLong());
  }

  @Test
  void refusesNonControlledWorkerReports() {
    ObjectNode request = BaselineReportFixtures.stockRequest(0, "measured", 0.02, 0);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "stock", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                FfmTransitionEstimate.estimate(
                    BaselineReportFixtures.report(worker), calibrationSummary()));
    assertTrue(error.getMessage().contains("controlled"));
  }

  @Test
  void refusesReportsWithoutMeasuredRequests() {
    ObjectNode warmup = BaselineReportFixtures.request(0, "warmup", 0.02);
    BaselineReportFixtures.instrument(warmup, "binding-count");
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", warmup);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                FfmTransitionEstimate.estimate(
                    BaselineReportFixtures.report(worker), calibrationSummary()));
    assertTrue(error.getMessage().contains("measured"));
  }

  @Test
  void reportsUncoveredSignatures() {
    ObjectNode request = BaselineReportFixtures.request(0, "measured", 0.02);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode counts = MAPPER.createObjectNode();
    ArrayNode phases = counts.putArray("phases");
    phase(phases, "prefill-sync", 0, 1, 3, 3);
    request.set("binding_counts", counts);

    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ObjectNode signatures = (ObjectNode) worker.get("binding_signatures");
    signatures.put("unparsed", "(Ljava/lang/String;I)I");

    // The synthetic calibration only knows the (J)V shape.
    ObjectNode estimate =
        FfmTransitionEstimate.estimate(BaselineReportFixtures.report(worker), calibrationSummary());
    assertEquals(7L, estimate.path("totals").path("binding_calls").asLong());
    assertEquals(3L, estimate.path("totals").path("classified_calls").asLong());
    assertEquals(4L, estimate.path("totals").path("unclassified_calls").asLong());
    assertEquals(3.0 / 7.0, estimate.path("totals").path("coverage").asDouble(), 1e-12);
    assertEquals(3L, estimate.path("uncovered").path("unparsed").path("calls").asLong());
    assertEquals(
        "(Ljava/lang/String;I)I",
        estimate.path("uncovered").path("unparsed").path("descriptor").asString(""));
    // The missing-descriptor mlx_matmul and the unparsable one are both uncovered.
    assertEquals(1L, estimate.path("uncovered").path("mlx_matmul").path("calls").asLong());
    assertEquals(1, estimate.path("per_shape").size());
  }

  private static void phase(
      ArrayNode phases, String name, long position, long matmul, long eval, long unparsed) {
    phases
        .addObject()
        .put("phase", name)
        .put("position", position)
        .putObject("counts")
        .put("mlx_matmul", matmul)
        .put("mlx_eval", eval)
        .put("unparsed", unparsed);
  }

  private static ObjectNode calibrationSummary() {
    ObjectNode calibration = MAPPER.createObjectNode();
    calibration.put("schema", "jmlx-ffm-cal-1");
    calibration.putObject("workload").put("iterations", 200).put("calls", 2000);
    ObjectNode shapes = calibration.putObject("shapes");
    shapes.putObject("(MMMM)I").put("median_ns", 90).put("p10_ns", 80).put("p90_ns", 110);
    shapes.putObject("(J)V").put("median_ns", 40).put("p10_ns", 30).put("p90_ns", 50);
    return calibration;
  }
}
