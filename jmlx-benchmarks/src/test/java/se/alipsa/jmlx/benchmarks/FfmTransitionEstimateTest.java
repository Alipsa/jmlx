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
 * Pure-Java tests of the FFM transition estimate's native-descriptor classification and its
 * budget/coverage accounting (including the two reported ranges) against hand-computed values on
 * synthetic binding-count reports (no native runtime and no compiler needed).
 */
class FfmTransitionEstimateTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  // Real native descriptors from the pinned jextract bindings (findings section 15.4, review
  // round 2, A), keyed to the calibration shapes they classify into.
  private static final String MATMUL_DESCRIPTOR =
      "(a8:[*:b1][a8(ctx):[*:b1]](mlx_array_)[a8(ctx):[*:b1]](mlx_array_)"
          + "[a8(ctx):[*:b1]](mlx_stream_))i4";
  private static final String EVAL_DESCRIPTOR = "([a8(ctx):[*:b1]](mlx_vector_array_))i4";
  private static final String FAST_ROPE_DESCRIPTOR =
      "(a8:[*:b1][a8(ctx):[*:b1]](mlx_array_)i4z1"
          + "[f4(value)z1(has_value)x3](mlx_optional_float_)f4i4"
          + "[a8(ctx):[*:b1]](mlx_array_)[a8(ctx):[*:b1]](mlx_stream_))i4";
  private static final String QUANTIZED_MATMUL_DESCRIPTOR =
      "(a8:[*:b1][a8(ctx):[*:b1]](mlx_array_)[a8(ctx):[*:b1]](mlx_array_)"
          + "[a8(ctx):[*:b1]](mlx_array_)[a8(ctx):[*:b1]](mlx_array_)z1"
          + "[i4(value)z1(has_value)x3](mlx_optional_int_)"
          + "[i4(value)z1(has_value)x3](mlx_optional_int_)a8:[*:b1]"
          + "[a8(ctx):[*:b1]](mlx_stream_))i4";
  private static final String DEQUANTIZE_DESCRIPTOR =
      "(a8:[*:b1][a8(ctx):[*:b1]](mlx_array_)[a8(ctx):[*:b1]](mlx_array_)"
          + "[a8(ctx):[*:b1]](mlx_array_)"
          + "[i4(value)z1(has_value)x3](mlx_optional_int_)"
          + "[i4(value)z1(has_value)x3](mlx_optional_int_)a8:[*:b1]"
          + "[a8(ctx):[*:b1]](mlx_array_)"
          + "[i4(value)z1(has_value)x3](mlx_optional_dtype_)"
          + "[a8(ctx):[*:b1]](mlx_stream_))i4";
  private static final String ARRAY_NEW_DESCRIPTOR = "()[a8(ctx):[*:b1]](mlx_array_)";

  @Test
  void shapeKeyMapsNativeDescriptorsToShapeKeys() {
    assertEquals("(PMMM)I", FfmTransitionEstimate.shapeKey(MATMUL_DESCRIPTOR));
    assertEquals("(M)I", FfmTransitionEstimate.shapeKey(EVAL_DESCRIPTOR));
    assertEquals("(PMIZMFIMM)I", FfmTransitionEstimate.shapeKey(FAST_ROPE_DESCRIPTOR));
    assertEquals("(PMMMMZMMPM)I", FfmTransitionEstimate.shapeKey(QUANTIZED_MATMUL_DESCRIPTOR));
    assertEquals("(PMMMMMPMMM)I", FfmTransitionEstimate.shapeKey(DEQUANTIZE_DESCRIPTOR));
    assertEquals("()M", FfmTransitionEstimate.shapeKey(ARRAY_NEW_DESCRIPTOR));
    assertEquals("(M)P", FfmTransitionEstimate.shapeKey("([a8(ctx):[*:b1]](mlx_array_))a8:[*:b1]"));
    assertEquals("(F)M", FfmTransitionEstimate.shapeKey("(f4)[a8(ctx):[*:b1]](mlx_array_)"));
    // Parsed, but a shape the calibration does not model: classification fails at the shapes
    // lookup, not here.
    assertEquals("(II)I", FfmTransitionEstimate.shapeKey("(i4i4)i4"));
    // A by-value struct that is not 8 bytes is unmodeled (the pinned header has no such class).
    assertNull(FfmTransitionEstimate.shapeKey("([j8(length)a8(ctx):[*:b1]](mlx_vector_array_))i4"));
    assertNull(FfmTransitionEstimate.shapeKey("(L)I"));
    assertNull(FfmTransitionEstimate.shapeKey("(i4"));
    assertNull(FfmTransitionEstimate.shapeKey("int(int)"));
    assertNull(FfmTransitionEstimate.shapeKey(null));
  }

  @Test
  void estimatesPerStepBudgetWithCoverage() {
    ObjectNode request = BaselineReportFixtures.request(0, "measured", 0.02);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode counts = MAPPER.createObjectNode();
    ArrayNode phases = counts.putArray("phases");
    // The controlled adapter's full window set, including the release windows (findings 15.6):
    // prefill phases, one step-phase window per phase per position, and the request release.
    phase(phases, "prefill-sync", 0, 4, 1, 0, 0);
    phase(phases, "prefill-release", 0, 0, 2, 0, 0);
    phase(phases, "construction", 33, 1, 0, 0, 0);
    phase(phases, "sync", 33, 0, 1, 0, 0);
    phase(phases, "selection", 33, 0, 1, 0, 0);
    phase(phases, "release", 33, 0, 1, 0, 0);
    phase(phases, "sync", 34, 2, 0, 0, 0);
    phase(phases, "release", 34, 1, 0, 0, 0);
    phase(phases, "request-release", 0, 0, 3, 0, 0);
    counts.putObject("total").put("mlx_matmul", 8L).put("mlx_eval", 9L);
    request.set("binding_counts", counts);

    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ((ObjectNode) worker.get("binding_descriptors")).put("mlx_matmul", MATMUL_DESCRIPTOR);

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
    // The calibration's own sampling range for the same pool: the p10-based samples are
    // [170, 240] (type-7 p10 177), the p90-based samples are [260, 330] (type-7 p90 323).
    assertEquals(177.0, estimate.path("per_decode_step").path("calibration_p10_ns").asDouble());
    assertEquals(323.0, estimate.path("per_decode_step").path("calibration_p90_ns").asDouble());
    // Per position, one sample per pool: the range endpoints are the single-sample quantiles.
    assertEquals(
        170.0,
        estimate
            .path("per_decode_step")
            .path("positions")
            .get(0)
            .path("calibration_p10_ns")
            .asDouble());
    assertEquals(
        260.0,
        estimate
            .path("per_decode_step")
            .path("positions")
            .get(0)
            .path("calibration_p90_ns")
            .asDouble());
    assertEquals(
        240.0,
        estimate
            .path("per_decode_step")
            .path("positions")
            .get(1)
            .path("calibration_p10_ns")
            .asDouble());
    assertEquals(
        330.0,
        estimate
            .path("per_decode_step")
            .path("positions")
            .get(1)
            .path("calibration_p90_ns")
            .asDouble());

    // Prefill = the prefill-* windows only (the request-release at position 0 is in neither
    // prefill nor any step).
    assertEquals(1, estimate.path("prefill").path("n").asInt());
    assertEquals(480, estimate.path("prefill").path("median_ns").asLong());
    assertEquals(410.0, estimate.path("prefill").path("calibration_p10_ns").asDouble());
    assertEquals(590.0, estimate.path("prefill").path("calibration_p90_ns").asDouble());
    assertEquals(1, estimate.path("request_total").path("n").asInt());
    assertEquals(1080, estimate.path("request_total").path("median_ns").asLong());
    assertEquals(910.0, estimate.path("request_total").path("calibration_p10_ns").asDouble());
    assertEquals(1330.0, estimate.path("request_total").path("calibration_p90_ns").asDouble());

    assertEquals(8L, estimate.path("per_shape").path("(PMMM)I").path("calls").asLong());
    // The per-call costs are the per-sample quantiles divided by the workload's call count (10).
    assertEquals(
        90.0, estimate.path("per_shape").path("(PMMM)I").path("median_ns_per_call").asDouble());
    assertEquals(
        80.0, estimate.path("per_shape").path("(PMMM)I").path("p10_ns_per_call").asDouble());
    assertEquals(
        110.0, estimate.path("per_shape").path("(PMMM)I").path("p90_ns_per_call").asDouble());
    assertEquals(720, estimate.path("per_shape").path("(PMMM)I").path("budget_ns").asLong());
    assertEquals(640L, estimate.path("per_shape").path("(PMMM)I").path("budget_p10_ns").asLong());
    assertEquals(880L, estimate.path("per_shape").path("(PMMM)I").path("budget_p90_ns").asLong());
    assertEquals(9L, estimate.path("per_shape").path("(M)I").path("calls").asLong());
    assertEquals(
        40.0, estimate.path("per_shape").path("(M)I").path("median_ns_per_call").asDouble());
    assertEquals(360, estimate.path("per_shape").path("(M)I").path("budget_ns").asLong());
    assertEquals(270L, estimate.path("per_shape").path("(M)I").path("budget_p10_ns").asLong());
    assertEquals(450L, estimate.path("per_shape").path("(M)I").path("budget_p90_ns").asLong());
    assertTrue(estimate.path("assumptions").size() >= 4);
  }

  @Test
  void refusesCalibrationsWithoutACallCount() {
    // The per-call costs divide the per-sample quantiles by workload.calls; without a positive
    // count the division is undefined and the run must fail clearly, not emit infinities.
    ObjectNode request = BaselineReportFixtures.request(0, "measured", 0.02);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ObjectNode calibration = calibrationSummary();
    ((ObjectNode) calibration.get("workload")).remove("calls");
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                FfmTransitionEstimate.estimate(BaselineReportFixtures.report(worker), calibration));
    assertTrue(error.getMessage().contains("workload.calls"));
  }

  @Test
  void poolsStepBudgetsPerRequestNotAcrossRequests() {
    // Two measured requests with different per-step costs at the same position: the per-step
    // value is the quantile over the requests' own samples [90, 90], never their sum (the
    // N-times-too-large pooling bug of findings section 15.6). A warmup request with large
    // counts is excluded from every aggregate, including the calibration-range fields.
    ObjectNode first = BaselineReportFixtures.request(0, "measured", 0.02);
    BaselineReportFixtures.instrument(first, "binding-count");
    ObjectNode countsA = MAPPER.createObjectNode();
    ArrayNode phasesA = countsA.putArray("phases");
    phase(phasesA, "prefill-sync", 0, 0, 1, 0, 0);
    phase(phasesA, "construction", 33, 1, 0, 0, 0);
    first.set("binding_counts", countsA);

    ObjectNode second = BaselineReportFixtures.request(1, "measured", 0.03);
    BaselineReportFixtures.instrument(second, "binding-count");
    ObjectNode countsB = MAPPER.createObjectNode();
    ArrayNode phasesB = countsB.putArray("phases");
    phase(phasesB, "prefill-sync", 0, 1, 0, 0, 0);
    phase(phasesB, "construction", 33, 1, 0, 0, 0);
    second.set("binding_counts", countsB);

    ObjectNode warmup = BaselineReportFixtures.request(2, "warmup", 0.04);
    BaselineReportFixtures.instrument(warmup, "binding-count");
    ObjectNode countsW = MAPPER.createObjectNode();
    ArrayNode phasesW = countsW.putArray("phases");
    phase(phasesW, "prefill-sync", 0, 100, 0, 0, 0);
    phase(phasesW, "construction", 33, 100, 0, 0, 0);
    warmup.set("binding_counts", countsW);

    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", first, second, warmup);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ((ObjectNode) worker.get("binding_descriptors")).put("mlx_matmul", MATMUL_DESCRIPTOR);

    ObjectNode estimate =
        FfmTransitionEstimate.estimate(BaselineReportFixtures.report(worker), calibrationSummary());
    assertEquals(2, estimate.path("measured_requests").asInt());
    // The warmup's 200 matmul calls are nowhere in the aggregates.
    assertEquals(4L, estimate.path("totals").path("binding_calls").asLong());
    assertEquals(4L, estimate.path("totals").path("classified_calls").asLong());
    assertEquals(1.0, estimate.path("totals").path("coverage").asDouble());

    assertEquals(1, estimate.path("per_decode_step").path("positions").size());
    // Both requests pay one matmul at position 33; the request-to-request spread collapses.
    assertEquals(
        90.0,
        estimate.path("per_decode_step").path("positions").get(0).path("median_ns").asDouble());
    // The calibration range does not collapse: 80/110 ns per matmul.
    assertEquals(80.0, estimate.path("per_decode_step").path("calibration_p10_ns").asDouble());
    assertEquals(110.0, estimate.path("per_decode_step").path("calibration_p90_ns").asDouble());

    assertEquals(2, estimate.path("prefill").path("n").asInt());
    // Request A's prefill is one eval (40 ns), request B's one matmul (90 ns): type-7
    // p10/median/p90 of [40, 90]; the calibration range pools [30, 80] and [50, 110].
    assertEquals(45.0, estimate.path("prefill").path("p10_ns").asDouble());
    assertEquals(65.0, estimate.path("prefill").path("median_ns").asDouble());
    assertEquals(85.0, estimate.path("prefill").path("p90_ns").asDouble());
    assertEquals(35.0, estimate.path("prefill").path("calibration_p10_ns").asDouble());
    assertEquals(104.0, estimate.path("prefill").path("calibration_p90_ns").asDouble());
    assertEquals(2, estimate.path("request_total").path("n").asInt());
    assertEquals(155.0, estimate.path("request_total").path("median_ns").asDouble());
    assertEquals(115.0, estimate.path("request_total").path("calibration_p10_ns").asDouble());
    assertEquals(214.0, estimate.path("request_total").path("calibration_p90_ns").asDouble());

    assertEquals(1L, estimate.path("per_shape").path("(M)I").path("calls").asLong());
    assertEquals(40, estimate.path("per_shape").path("(M)I").path("budget_ns").asLong());
    assertEquals(3L, estimate.path("per_shape").path("(PMMM)I").path("calls").asLong());
    assertEquals(270, estimate.path("per_shape").path("(PMMM)I").path("budget_ns").asLong());
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
  void refusesReportsWithoutNativeDescriptors() {
    // A binding-count report predating the native-descriptor recording (review round 2, A) is
    // rejected with a clear error pointing at the re-run, not silently misclassified.
    ObjectNode request = BaselineReportFixtures.request(0, "measured", 0.02);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ((ObjectNode) worker.get("binding_descriptors")).remove("mlx_eval");
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                FfmTransitionEstimate.estimate(
                    BaselineReportFixtures.report(worker), calibrationSummary()));
    assertTrue(error.getMessage().contains("binding_descriptors"));
  }

  @Test
  void reportsUncoveredSignatures() {
    ObjectNode request = BaselineReportFixtures.request(0, "measured", 0.02);
    BaselineReportFixtures.instrument(request, "binding-count");
    ObjectNode counts = MAPPER.createObjectNode();
    ArrayNode phases = counts.putArray("phases");
    phase(phases, "prefill-sync", 0, 0, 3, 3, 1);
    request.set("binding_counts", counts);

    ObjectNode worker =
        BaselineReportFixtures.workerBlock(
            "java", "controlled", "tiny", 32, "headline", "ok", null, "sha", request);
    BaselineReportFixtures.instrumentWorker(worker, "binding-count");
    ((ObjectNode) worker.get("binding_descriptors")).put("unparsed", "(L)I");
    // "missing" carries no binding_descriptors entry at all.

    ObjectNode estimate =
        FfmTransitionEstimate.estimate(BaselineReportFixtures.report(worker), calibrationSummary());
    assertEquals(7L, estimate.path("totals").path("binding_calls").asLong());
    assertEquals(3L, estimate.path("totals").path("classified_calls").asLong());
    assertEquals(4L, estimate.path("totals").path("unclassified_calls").asLong());
    assertEquals(3.0 / 7.0, estimate.path("totals").path("coverage").asDouble(), 1e-12);
    assertEquals(3L, estimate.path("uncovered").path("unparsed").path("calls").asLong());
    assertEquals(
        "(L)I", estimate.path("uncovered").path("unparsed").path("descriptor").asString(""));
    // The missing-descriptor binding is uncovered with the empty recorded descriptor.
    assertEquals(1L, estimate.path("uncovered").path("missing").path("calls").asLong());
    assertEquals("", estimate.path("uncovered").path("missing").path("descriptor").asString(null));
    assertEquals(1, estimate.path("per_shape").size());
  }

  private static void phase(
      ArrayNode phases,
      String name,
      long position,
      long matmul,
      long eval,
      long unparsed,
      long missing) {
    phases
        .addObject()
        .put("phase", name)
        .put("position", position)
        .putObject("counts")
        .put("mlx_matmul", matmul)
        .put("mlx_eval", eval)
        .put("unparsed", unparsed)
        .put("missing", missing);
  }

  /**
   * The synthetic calibration, in the real schema's per-sample units (one sample is {@code calls}
   * consecutive downcalls): the per-call costs are 90/80/110 ns for (PMMM)I and 40/30/50 ns for
   * (M)I, so the hand-computed budgets below use those per-call values directly.
   */
  private static ObjectNode calibrationSummary() {
    ObjectNode calibration = MAPPER.createObjectNode();
    calibration.put("schema", "jmlx-ffm-cal-1");
    calibration.putObject("workload").put("iterations", 200).put("calls", 10);
    ObjectNode shapes = calibration.putObject("shapes");
    shapes.putObject("(PMMM)I").put("median_ns", 900).put("p10_ns", 800).put("p90_ns", 1100);
    shapes.putObject("(M)I").put("median_ns", 400).put("p10_ns", 300).put("p90_ns", 500);
    return calibration;
  }
}
