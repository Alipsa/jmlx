package se.alipsa.jmlx.benchmarks;

import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Raw-to-summary FFM transition estimate (phase 8.2 WP2, plan section 5.1). Consumes a
 * binding-count worker report (the worker's {@code binding_signatures} plus the {@code
 * requests[].binding_counts} phase windows) and an {@link FfmCalibration} summary, and produces the
 * per-decode-step budget {@code Σ count × calibrated nanos per shape} with its p10/p90 range, the
 * coverage of classified calls, the uncovered signatures and the explicit assumptions of the
 * estimate.
 *
 * <p>It is a calibrated estimate, not a causal partition of the step's time (plan section 5.1): the
 * assumptions are repeated in every output. It accepts only controlled worker reports (the stock
 * adapter's single whole-request window cannot be split into prefill and per-step phases), and it
 * aggregates only {@code kind == "measured"} requests: one step-budget sample per (measured
 * request, position) — the request's own step-phase window budgets at that position — so per-step
 * and per-position values never scale with the number of requests, and warmups/firsts are excluded
 * from every aggregate (totals, per_shape, uncovered, prefill, per-step, request_total).
 *
 * <p>Opt-in local tooling, not part of {@code check} or the publication surface.
 */
public final class FfmTransitionEstimate {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final List<String> ASSUMPTIONS =
      List.of(
          "The calibration measures no-op C functions with the same parameter/return register"
              + " shapes as the generated mlx-c downcalls, through the generated wrappers' own"
              + " invocation form (invokeExact on a static final MethodHandle bound in a private"
              + " holder class's static initializer): it is the FFM transition cost (method-handle"
              + " dispatch, argument marshalling, return handling, plus the per-call"
              + " result-segment allocation through the caller's SegmentAllocator for"
              + " struct-return shapes) plus a small fixed Java prologue/epilogue per call, and it"
              + " does not include the real mlx-c's C-side work (Metal dispatch, kernel launches,"
              + " synchronization, allocator work). For the struct-return shapes the calibration's"
              + " result allocator is a plain confined arena, not production MLXScope's"
              + " synchronized shared-arena allocator, so the (AM)M figure excludes that"
              + " allocator's extra cost.",
          "Carrier fidelity: the pinned mlx-c header passes every opaque struct (a single 8-byte"
              + " pointer member) by value, and on AArch64 an 8-byte struct parameter is carried"
              + " in one 64-bit register — the same C ABI as a void* — while 8-byte struct returns"
              + " by value use the same register convention, so the no-op carriers are"
              + " register-equivalent to the real ones; the Java side matches through the same"
              + " struct-by-value declarations the generated wrappers use (every M parameter slot"
              + " is the struct's 8-byte layout). Caveat: a hypothetical pointer-kind M parameter"
              + " (no 8-byte struct read) is indistinguishable in the recorded JVM descriptor"
              + " (both are MemorySegment) and would be slightly over-measured by the"
              + " struct-by-value calibration.",
          "Counts are attempted mlx-c downcalls through the generated wrappers (the"
              + " BindingCounterAgent), one measurement per downcall; the estimate does not cover"
              + " Metal dispatches, kernels or GPU synchronization (plan section 5.1).",
          "A calibrated estimate, not a causal partition: the budget assumes the per-shape"
              + " transition cost measured in bare fresh JVMs stays stable across the instrumented"
              + " session and workload, and the p10/p90 range reflects the calibration's sampling"
              + " spread across its forks.");

  private FfmTransitionEstimate() {}

  /**
   * Runs {@code worker-report.json ffm-calibration.json output-json}.
   *
   * @param args the binding-count worker report, the calibration summary and the output path
   * @throws Exception if a report read or the output write fails
   */
  public static void main(String[] args) throws Exception {
    if (args.length != 3) {
      throw new IllegalArgumentException(
          "usage: FfmTransitionEstimate worker-report.json ffm-calibration.json output-json");
    }
    JsonNode report = MAPPER.readTree(Path.of(args[0]).toFile());
    JsonNode calibration = MAPPER.readTree(Path.of(args[1]).toFile());
    ObjectNode estimate = estimate(report, calibration);
    MAPPER.writerWithDefaultPrettyPrinter().writeValue(Path.of(args[2]).toFile(), estimate);
  }

  /**
   * Computes the estimate (pure; the report and calibration JSON are data). Refuses non-controlled
   * worker reports and reports with no measured request.
   */
  static ObjectNode estimate(JsonNode report, JsonNode calibration) {
    JsonNode worker = report.path("worker");
    String adapter = worker.path("adapter").asString("");
    if (!"controlled".equals(adapter)) {
      throw new IllegalArgumentException(
          "the FFM transition estimate requires a controlled worker report (got adapter '"
              + adapter
              + "'): the stock adapter's single whole-request window cannot be split into"
              + " prefill and per-step phases");
    }
    JsonNode signatures = worker.path("binding_signatures");
    JsonNode shapes = calibration.path("shapes");

    long bindingCalls = 0;
    long classifiedCalls = 0;
    // key -> [calls, budget median ns, budget p10 ns, budget p90 ns]
    Map<String, long[]> perShape = new LinkedHashMap<>();
    // binding name -> [calls]
    Map<String, long[]> uncovered = new LinkedHashMap<>();
    Map<String, String> uncoveredDescriptors = new LinkedHashMap<>();
    // position -> one sample per measured request: that request's step-phase budget at the
    // position (pooling the requests' own sums, never summing across requests)
    Map<Long, List<Long>> stepSamples = new LinkedHashMap<>();
    List<Long> allStepSamples = new ArrayList<>();
    List<Long> prefillBudgets = new ArrayList<>();
    List<Long> requestTotals = new ArrayList<>();
    int measuredRequests = 0;

    for (JsonNode request : worker.path("requests")) {
      // Measured requests only: warmups and firsts measured the same code, but pooling them into
      // every aggregate would mix cold-start behavior into the budget (findings section 15.6).
      if (!"measured".equals(request.path("kind").asString(""))) {
        continue;
      }
      measuredRequests++;
      long requestTotal = 0;
      long requestPrefill = 0;
      // this request's step-phase budget per position (the per-position sample)
      Map<Long, Long> requestStepBudgets = new LinkedHashMap<>();
      for (JsonNode window : request.path("binding_counts").path("phases")) {
        String phase = window.path("phase").asString("");
        long position = window.path("position").asLong(0);
        long windowMedian = 0;
        long windowP10 = 0;
        long windowP90 = 0;
        for (String name : window.path("counts").propertyNames()) {
          long count = window.path("counts").path(name).asLong(0);
          if (count <= 0) {
            continue;
          }
          bindingCalls += count;
          String descriptor = signatures.path(name).asString("");
          String key = normalize(descriptor);
          if (key == null || !shapes.has(key)) {
            uncovered.computeIfAbsent(name, n -> new long[1])[0] += count;
            uncoveredDescriptors.put(name, descriptor);
            continue;
          }
          classifiedCalls += count;
          long[] shape = perShape.computeIfAbsent(key, k -> new long[4]);
          shape[0] += count;
          double median = shapes.path(key).path("median_ns").asDouble(0);
          shape[1] += Math.round(count * median);
          double p10 = shapes.path(key).path("p10_ns").asDouble(0);
          shape[2] += Math.round(count * p10);
          double p90 = shapes.path(key).path("p90_ns").asDouble(0);
          shape[3] += Math.round(count * p90);
          windowMedian += Math.round(count * median);
          windowP10 += Math.round(count * p10);
          windowP90 += Math.round(count * p90);
        }
        requestTotal += windowMedian;
        if (phase.startsWith("prefill-")) {
          requestPrefill += windowMedian;
        } else if ("construction".equals(phase)
            || "sync".equals(phase)
            || "selection".equals(phase)
            || "release".equals(phase)) {
          requestStepBudgets.merge(position, windowMedian, Long::sum);
        }
      }
      if (requestPrefill > 0) {
        prefillBudgets.add(requestPrefill);
      }
      requestTotals.add(requestTotal);
      for (Map.Entry<Long, Long> entry : requestStepBudgets.entrySet()) {
        stepSamples.computeIfAbsent(entry.getKey(), p -> new ArrayList<>()).add(entry.getValue());
        allStepSamples.add(entry.getValue());
      }
    }
    if (measuredRequests == 0) {
      throw new IllegalArgumentException(
          "the worker report carries no measured requests; the FFM transition estimate"
              + " aggregates measured requests only");
    }

    ObjectNode estimate = MAPPER.createObjectNode();
    estimate.put("schema", "jmlx-ffm-transition-estimate-1");
    estimate.put("created_at", Instant.now().toString());
    estimate.put("calibration_schema", calibration.path("schema").asString(""));
    estimate.set("calibration_workload", calibration.path("workload"));
    estimate.put("measured_requests", measuredRequests);

    ObjectNode totals = estimate.putObject("totals");
    totals.put("binding_calls", bindingCalls);
    totals.put("classified_calls", classifiedCalls);
    totals.put("unclassified_calls", bindingCalls - classifiedCalls);
    totals.put("coverage", bindingCalls == 0 ? 0.0 : (double) classifiedCalls / bindingCalls);

    ObjectNode perStep = estimate.putObject("per_decode_step");
    ArrayNode positions = perStep.putArray("positions");
    for (Map.Entry<Long, List<Long>> entry : stepSamples.entrySet()) {
      BaselineStatistics.Quantiles quantiles =
          BaselineStatistics.quantiles(doubleList(entry.getValue()));
      positions
          .addObject()
          .put("position", entry.getKey())
          .put("median_ns", quantiles.median())
          .put("p10_ns", quantiles.p10())
          .put("p90_ns", quantiles.p90());
    }
    stepSummary(perStep, allStepSamples);

    stepSummary(estimate.putObject("prefill"), prefillBudgets);
    stepSummary(estimate.putObject("request_total"), requestTotals);

    ObjectNode perShapeOut = estimate.putObject("per_shape");
    for (Map.Entry<String, long[]> entry : perShape.entrySet()) {
      perShapeOut
          .putObject(entry.getKey())
          .put("calls", entry.getValue()[0])
          .put("median_ns_per_call", shapes.path(entry.getKey()).path("median_ns").asDouble(0))
          .put("budget_ns", entry.getValue()[1]);
    }

    ObjectNode uncoveredOut = estimate.putObject("uncovered");
    for (Map.Entry<String, long[]> entry : uncovered.entrySet()) {
      uncoveredOut
          .putObject(entry.getKey())
          .put("descriptor", uncoveredDescriptors.get(entry.getKey()))
          .put("calls", entry.getValue()[0]);
    }

    ArrayNode assumptions = estimate.putArray("assumptions");
    for (String assumption : ASSUMPTIONS) {
      assumptions.add(assumption);
    }
    return estimate;
  }

  private static List<Double> doubleList(List<Long> values) {
    List<Double> out = new ArrayList<>(values.size());
    for (long v : values) {
      out.add((double) v);
    }
    return out;
  }

  /**
   * Type-7 quantiles (p10/median/p90) of a budget pool with its sample count; the pool is one
   * sample per measured request (or per (measured request, position) for the per-step summary),
   * never a sum across requests.
   */
  private static void stepSummary(ObjectNode node, List<Long> budgets) {
    BaselineStatistics.Quantiles quantiles = BaselineStatistics.quantiles(doubleList(budgets));
    node.put("n", budgets.size());
    node.put("p10_ns", quantiles.p10());
    node.put("median_ns", quantiles.median());
    node.put("p90_ns", quantiles.p90());
  }

  /**
   * Maps a generated wrapper's JVM descriptor (as recorded by the binding-count agent) to a
   * calibration shape key, or null when the descriptor names a type the calibration does not model.
   */
  static String normalize(String descriptor) {
    if (descriptor == null || !descriptor.startsWith("(")) {
      return null;
    }
    int close = descriptor.indexOf(')');
    if (close < 0) {
      return null;
    }
    StringBuilder params = new StringBuilder();
    for (int i = 1; i < close; i++) {
      char c = descriptor.charAt(i);
      if (c == 'L') {
        int semi = descriptor.indexOf(';', i);
        if (semi == -1 || semi > close) {
          return null;
        }
        String type = descriptor.substring(i, semi + 1);
        if (type.equals("Ljava/lang/foreign/MemorySegment;")) {
          params.append('M');
        } else if (type.equals("Ljava/lang/foreign/SegmentAllocator;")) {
          params.append('A');
        } else {
          return null;
        }
        i = semi;
      } else if (c == 'I' || c == 'J' || c == 'F' || c == 'D' || c == 'Z') {
        params.append(c);
      } else {
        return null;
      }
    }
    String returnPart = descriptor.substring(close + 1);
    String returnKey;
    if (returnPart.equals("Ljava/lang/foreign/MemorySegment;")) {
      returnKey = "M";
    } else if (returnPart.length() == 1 && "I J F D Z V".indexOf(returnPart.charAt(0)) >= 0) {
      returnKey = returnPart;
    } else {
      return null;
    }
    return "(" + params + ")" + returnKey;
  }
}
