package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Pure-Java tests of the FFM transition estimate's descriptor normalization and its budget/coverage
 * accounting against hand-computed values on a synthetic binding-count report (no native runtime
 * and no compiler needed).
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
    ObjectNode request = BaselineReportFixtures.request(0, "first", 0.02);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode counts = MAPPER.createObjectNode();
    ArrayNode phases = counts.putArray("phases");
    phase(phases, "prefill-sync", 0, 4, 1, 0);
    phase(phases, "construction", 33, 1, 0, 0);
    phase(phases, "sync", 33, 0, 1, 0);
    phase(phases, "selection", 33, 0, 1, 0);
    phase(phases, "sync", 34, 2, 0, 0);
    counts.putObject("total").put("mlx_matmul", 7L).put("mlx_eval", 3L);
    request.set("binding_counts", counts);

    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ((ObjectNode) worker.get("binding_signatures")).put("mlx_matmul", MATMUL_DESCRIPTOR);

    ObjectNode estimate =
        FfmTransitionEstimate.estimate(BaselineReportFixtures.report(worker), calibrationSummary());
    assertEquals("jmlx-ffm-transition-estimate-1", estimate.path("schema").asString());
    assertEquals(10L, estimate.path("totals").path("binding_calls").asLong());
    assertEquals(10L, estimate.path("totals").path("classified_calls").asLong());
    assertEquals(1.0, estimate.path("totals").path("coverage").asDouble());
    assertEquals(0, estimate.path("uncovered").size());

    // Step 33 = 1 matmul @ 90 ns + 2 evals @ 40 ns; step 34 = 2 matmuls @ 90 ns.
    assertEquals(2, estimate.path("per_decode_step").path("n").asInt());
    assertEquals(
        33, estimate.path("per_decode_step").path("positions").get(0).path("position").asInt());
    assertEquals(
        170, estimate.path("per_decode_step").path("positions").get(0).path("median_ns").asLong());
    assertEquals(
        140, estimate.path("per_decode_step").path("positions").get(0).path("p10_ns").asLong());
    assertEquals(
        210, estimate.path("per_decode_step").path("positions").get(0).path("p90_ns").asLong());
    assertEquals(
        180, estimate.path("per_decode_step").path("positions").get(1).path("median_ns").asLong());
    assertEquals(175, estimate.path("per_decode_step").path("median_ns").asLong());

    assertEquals(1, estimate.path("prefill").path("n").asInt());
    assertEquals(400, estimate.path("prefill").path("median_ns").asLong());
    assertEquals(1, estimate.path("request_total").path("n").asInt());
    assertEquals(750, estimate.path("request_total").path("median_ns").asLong());

    assertEquals(7L, estimate.path("per_shape").path("(MMMM)I").path("calls").asLong());
    assertEquals(630, estimate.path("per_shape").path("(MMMM)I").path("budget_ns").asLong());
    assertEquals(3L, estimate.path("per_shape").path("(J)V").path("calls").asLong());
    assertEquals(120, estimate.path("per_shape").path("(J)V").path("budget_ns").asLong());
    assertTrue(estimate.path("assumptions").size() >= 4);
  }

  @Test
  void reportsUncoveredSignatures() {
    ObjectNode request = BaselineReportFixtures.request(0, "first", 0.02);
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
