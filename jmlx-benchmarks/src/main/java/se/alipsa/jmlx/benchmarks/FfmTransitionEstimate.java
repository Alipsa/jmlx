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
 * assumptions are repeated in every output.
 *
 * <p>Opt-in local tooling, not part of {@code check} or the publication surface.
 */
public final class FfmTransitionEstimate {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final List<String> ASSUMPTIONS =
      List.of(
          "The calibration measures no-op C functions with the same parameter/return register"
              + " shapes as the generated mlx-c downcalls, through Project Panama downcall handles"
              + " in the generated wrappers' own invokeExact form: it is the FFM transition cost"
              + " (method-handle dispatch, argument marshalling, return handling, plus the"
              + " per-call result-segment allocation through the caller's SegmentAllocator for"
              + " struct-return shapes) plus a small fixed Java prologue/epilogue per call, and it"
              + " does not include the real mlx-c's C-side work (Metal dispatch, kernel launches,"
              + " synchronization, allocator work).",
          "Carrier fidelity: every opaque mlx-c struct is a single 8-byte pointer, passed in a"
              + " 64-bit register exactly like a void* on AArch64, and 8-byte struct returns by"
              + " value use the same register convention, so the no-op carriers are"
              + " register-equivalent to the real ones.",
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

  /** Computes the estimate (pure; the report and calibration JSON are data). */
  static ObjectNode estimate(JsonNode report, JsonNode calibration) {
    JsonNode worker = report.path("worker");
    JsonNode signatures = worker.path("binding_signatures");
    JsonNode shapes = calibration.path("shapes");

    long bindingCalls = 0;
    long classifiedCalls = 0;
    // key -> [calls, budget median ns, budget p10 ns, budget p90 ns]
    Map<String, long[]> perShape = new LinkedHashMap<>();
    // binding name -> [calls]
    Map<String, long[]> uncovered = new LinkedHashMap<>();
    Map<String, String> uncoveredDescriptors = new LinkedHashMap<>();
    // position -> [median ns, p10 ns, p90 ns] over the step's phase windows
    Map<Long, long[]> steps = new LinkedHashMap<>();
    List<Double> prefillBudgets = new ArrayList<>();
    List<Double> requestTotals = new ArrayList<>();

    for (JsonNode request : worker.path("requests")) {
      long requestTotal = 0;
      long requestPrefill = 0;
      for (JsonNode window : request.path("binding_counts").path("phases")) {
        String phase = window.path("phase").asString("");
        long position = window.path("position").asLong(0);
        long windowMedian = 0;
        long windowP10 = 0;
        long windowP90 = 0;
        for (String name : window.path("counts").propertyNames()) {
          int count = window.path("counts").path(name).asInt(0);
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
        if (position == 0) {
          requestPrefill += windowMedian;
        } else if ("construction".equals(phase)
            || "sync".equals(phase)
            || "selection".equals(phase)) {
          long[] step = steps.computeIfAbsent(position, p -> new long[3]);
          step[0] += windowMedian;
          step[1] += windowP10;
          step[2] += windowP90;
        }
      }
      if (requestPrefill > 0) {
        prefillBudgets.add((double) requestPrefill);
      }
      requestTotals.add((double) requestTotal);
    }

    ObjectNode estimate = MAPPER.createObjectNode();
    estimate.put("schema", "jmlx-ffm-transition-estimate-1");
    estimate.put("created_at", Instant.now().toString());
    estimate.put("calibration_schema", calibration.path("schema").asString(""));
    estimate.set("calibration_workload", calibration.path("workload"));

    ObjectNode totals = estimate.putObject("totals");
    totals.put("binding_calls", bindingCalls);
    totals.put("classified_calls", classifiedCalls);
    totals.put("unclassified_calls", bindingCalls - classifiedCalls);
    totals.put("coverage", bindingCalls == 0 ? 0.0 : (double) classifiedCalls / bindingCalls);

    ObjectNode perStep = estimate.putObject("per_decode_step");
    ArrayNode positions = perStep.putArray("positions");
    List<Double> stepMedians = new ArrayList<>();
    for (Map.Entry<Long, long[]> entry : steps.entrySet()) {
      stepMedians.add((double) entry.getValue()[0]);
      positions
          .addObject()
          .put("position", entry.getKey())
          .put("median_ns", (double) entry.getValue()[0])
          .put("p10_ns", (double) entry.getValue()[1])
          .put("p90_ns", (double) entry.getValue()[2]);
    }
    stepSummary(perStep, stepMedians);

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

  private static void stepSummary(ObjectNode node, List<Double> budgets) {
    BaselineStatistics.Quantiles quantiles = BaselineStatistics.quantiles(budgets);
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
