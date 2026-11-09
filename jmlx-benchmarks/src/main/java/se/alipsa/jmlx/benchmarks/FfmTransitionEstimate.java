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
 * binding-count worker report (the worker's {@code binding_descriptors} — the native function
 * descriptors of the generated downcalls, by exact binding name — plus the {@code
 * requests[].binding_counts} phase windows) and an {@link FfmCalibration} summary, and produces the
 * per-decode-step budget {@code Σ count × calibrated nanos per call per shape} with its
 * request-to-request range, the calibration's own sampling range (the {@code calibration_*}
 * fields), the coverage of classified calls, the uncovered signatures and the explicit assumptions
 * of the estimate.
 *
 * <p>The calibration's quantiles are per sample (one sample is the workload's fixed number of
 * consecutive downcalls, {@code workload.calls}), so every per-call cost divides the per-sample
 * quantile by that count (the per-sample {@code p10_ns}/{@code median_ns}/{@code p90_ns} never
 * enters a budget as-is). Type-7 quantiles commute with a positive scale factor, so the divided
 * quantiles are exactly the quantiles of the per-call samples. Every budget accumulates its {@code
 * count x per-call cost} terms in double, with no intermediate whole-nanosecond rounding (review
 * round 3, 3) — rounding each (binding, window) term would bias a ~50-binding step by up to ±25 ns
 * at per-call costs of tens of nanoseconds — and the per-shape budgets round once, at their output
 * values.
 *
 * <p>It is a calibrated estimate, not a causal partition of the step's time (plan section 5.1): the
 * assumptions are repeated in every output. It accepts only controlled worker reports (the stock
 * adapter's single whole-request window cannot be split into prefill and per-step phases), and it
 * aggregates only {@code kind == "measured"} requests: one step-budget sample per (measured
 * request, position) — the request's own step-phase window budgets at that position — so per-step
 * and per-position values never scale with the number of requests, and warmups/firsts are excluded
 * from every aggregate (totals, per_shape, uncovered, prefill, per-step, request_total).
 *
 * <p>The binding-name → shape classification is native-descriptor-only (review round 2, A): the
 * counts are classified by the worker-recorded {@code binding_descriptors} (the jextract {@code
 * FunctionDescriptor} of each downcall), never by the JVM-form {@code binding_signatures} — which
 * stays in the report as provenance but no longer drives the estimate. Reports without a nonempty
 * {@code binding_descriptors} object (predating the recording) are refused with a clear error, and
 * a binding name missing from it is uncovered.
 *
 * <p>Opt-in local tooling, not part of {@code check} or the publication surface.
 */
public final class FfmTransitionEstimate {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final List<String> ASSUMPTIONS =
      List.of(
          "The calibration measures no-op C functions with the same parameter/return shapes as"
              + " the generated mlx-c downcalls, through the generated wrappers' own invocation"
              + " form (invokeExact on a static final MethodHandle bound in a private holder"
              + " class's static initializer): it is the FFM transition cost (method-handle"
              + " dispatch, argument marshalling, return handling, plus the per-call"
              + " result-segment allocation through the caller's SegmentAllocator for the"
              + " struct-return shapes) plus a small fixed Java prologue/epilogue per call, and"
              + " it does not include the real mlx-c's C-side work (Metal dispatch, kernel"
              + " launches, synchronization, allocator work). The P/M argument carriers live on"
              + " a process-lifetime shared arena — the same arena kind production MLXScope.Holder"
              + " uses for every native handle — and the struct-return shapes allocate their"
              + " results through a per-sample allocator that mirrors a production step scope's"
              + " lifecycle (fresh synchronized shared-arena allocation plus per-scope tracking,"
              + " closed after every sample outside the timed region), so the measured per-call"
              + " allocation cost is production's, not a bare confined arena's, and no result"
              + " segment, set entry or arena reservation outlives its sample's batch.",
          "Carrier fidelity: the pinned mlx-c header passes every opaque struct (a single 8-byte"
              + " pointer member) and every mlx_optional_* scalar wrapper (an 8-byte"
              + " value/has-value struct) by value, and on AArch64 an 8-byte struct parameter is"
              + " carried in one 64-bit register — the same C ABI as a void* — while 8-byte"
              + " struct returns by value use the same register convention, so the no-op"
              + " carriers (void* for the P slots, an 8-byte struct for the M slots) are"
              + " register-equivalent to the real ones; the Java side matches through the same"
              + " struct-by-value declarations the generated wrappers use (every M slot is the"
              + " struct's 8-byte layout). The shape keys are the native descriptor classes of"
              + " the bindings the controlled workloads actually call (review round 2, A), and"
              + " the estimate classifies the counts by those recorded descriptors.",
          "Counts are attempted mlx-c downcalls through the generated wrappers (the"
              + " BindingCounterAgent), one measurement per downcall; the estimate does not cover"
              + " Metal dispatches, kernels or GPU synchronization (plan section 5.1).",
          "A calibrated estimate, not a causal partition: the budget assumes the per-shape"
              + " transition cost measured in bare fresh JVMs stays stable across the instrumented"
              + " session and workload. Two ranges are reported: p10_ns/median_ns/p90_ns are the"
              + " request-to-request spread of the per-request budgets computed from each shape's"
              + " calibration median (one sample per measured request, and per (measured request,"
              + " position)), while calibration_p10_ns/calibration_p90_ns are the calibration's"
              + " own sampling range — the same per-request and per-position budgets computed from"
              + " each shape's calibration p10 and p90 samples.");

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
   * worker reports, reports without a nonempty {@code binding_descriptors} object, reports with no
   * measured request and calibration summaries without a positive {@code workload.calls}.
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
    JsonNode descriptors = worker.path("binding_descriptors");
    if (!descriptors.isObject() || descriptors.size() == 0) {
      throw new IllegalArgumentException(
          "the worker report carries no nonempty binding_descriptors: binding-count reports"
              + " predating the native-descriptor recording (findings section 15.7, review"
              + " round 2, A) cannot be classified by the downcall's native descriptor shape —"
              + " re-run the binding-count worker with the current code");
    }
    JsonNode shapes = calibration.path("shapes");
    // The calibration's quantiles are per SAMPLE (one sample is `calls` consecutive downcalls,
    // jmlx-ffm-cal-1), so the per-call costs divide by the workload's call count. Type-7
    // quantiles commute with a positive scale factor, so this is exactly the quantile over the
    // per-call samples.
    long callsPerSample = calibration.path("workload").path("calls").asLong(0);
    if (callsPerSample < 1) {
      throw new IllegalArgumentException(
          "the calibration summary carries no positive workload.calls: the jmlx-ffm-cal-1"
              + " quantiles are per-sample (one sample is `calls` consecutive downcalls), and"
              + " the per-call costs cannot be derived without the call count");
    }
    double perCall = 1.0 / (double) callsPerSample;

    long bindingCalls = 0;
    long classifiedCalls = 0;
    // key -> the shape's counted calls
    Map<String, Long> perShapeCalls = new LinkedHashMap<>();
    // key -> [budget median ns, budget p10 ns, budget p90 ns], accumulated in double and rounded
    // once per output value (review round 3, 3): rounding each (binding, window) term to whole
    // nanoseconds biases a ~50-binding step by up to ±25 ns at per-call costs of tens of ns.
    Map<String, double[]> perShapeBudgets = new LinkedHashMap<>();
    // binding name -> [calls]
    Map<String, long[]> uncovered = new LinkedHashMap<>();
    Map<String, String> uncoveredDescriptors = new LinkedHashMap<>();
    // position -> one sample per measured request: that request's step-phase budget at the
    // position (pooling the requests' own sums, never summing across requests), computed from
    // the calibration's median/p10/p90 samples respectively
    Map<Long, List<Double>> stepSamples = new LinkedHashMap<>();
    Map<Long, List<Double>> stepSamplesP10 = new LinkedHashMap<>();
    Map<Long, List<Double>> stepSamplesP90 = new LinkedHashMap<>();
    List<Double> allStepSamples = new ArrayList<>();
    List<Double> allStepP10 = new ArrayList<>();
    List<Double> allStepP90 = new ArrayList<>();
    List<Double> prefillBudgets = new ArrayList<>();
    List<Double> prefillP10Budgets = new ArrayList<>();
    List<Double> prefillP90Budgets = new ArrayList<>();
    List<Double> requestTotals = new ArrayList<>();
    List<Double> requestTotalsP10 = new ArrayList<>();
    List<Double> requestTotalsP90 = new ArrayList<>();
    int measuredRequests = 0;

    for (JsonNode request : worker.path("requests")) {
      // Measured requests only: warmups and firsts measured the same code, but pooling them into
      // every aggregate would mix cold-start behavior into the budget (findings section 15.6).
      if (!"measured".equals(request.path("kind").asString(""))) {
        continue;
      }
      measuredRequests++;
      double requestTotal = 0.0;
      double requestTotalP10 = 0.0;
      double requestTotalP90 = 0.0;
      double requestPrefill = 0.0;
      double requestPrefillP10 = 0.0;
      double requestPrefillP90 = 0.0;
      // this request's step-phase budget per position (the per-position samples)
      Map<Long, Double> requestStepBudgets = new LinkedHashMap<>();
      Map<Long, Double> requestStepP10 = new LinkedHashMap<>();
      Map<Long, Double> requestStepP90 = new LinkedHashMap<>();
      for (JsonNode window : request.path("binding_counts").path("phases")) {
        double windowMedian = 0.0;
        double windowP10 = 0.0;
        double windowP90 = 0.0;
        for (String name : window.path("counts").propertyNames()) {
          long count = window.path("counts").path(name).asLong(0);
          if (count <= 0) {
            continue;
          }
          bindingCalls += count;
          // Native descriptor only: the JVM-form binding_signatures stay in the report as
          // provenance but never classify a count (review round 2, A).
          String descriptor = descriptors.path(name).asString("");
          String key = shapeKey(descriptor);
          if (key == null || !shapes.has(key)) {
            uncovered.computeIfAbsent(name, n -> new long[1])[0] += count;
            uncoveredDescriptors.put(name, descriptor);
            continue;
          }
          classifiedCalls += count;
          perShapeCalls.merge(key, count, Long::sum);
          double[] shape = perShapeBudgets.computeIfAbsent(key, k -> new double[3]);
          // Per-call costs (the per-sample quantiles divided by the workload's call count);
          // the terms accumulate unrounded (review round 3, 3).
          double median = shapes.path(key).path("median_ns").asDouble(0) * perCall;
          shape[0] += count * median;
          double p10 = shapes.path(key).path("p10_ns").asDouble(0) * perCall;
          shape[1] += count * p10;
          double p90 = shapes.path(key).path("p90_ns").asDouble(0) * perCall;
          shape[2] += count * p90;
          windowMedian += count * median;
          windowP10 += count * p10;
          windowP90 += count * p90;
        }
        // The window's classification, read after its counts: prefill-* windows charge the
        // request's prefill budget, the step phases at position > 0 the per-position step
        // budget, everything else (request-release) the request total only.
        String phase = window.path("phase").asString("");
        long position = window.path("position").asLong(0);
        if (phase.startsWith("prefill-")) {
          requestPrefill += windowMedian;
          requestPrefillP10 += windowP10;
          requestPrefillP90 += windowP90;
        } else if (position > 0
            && ("construction".equals(phase)
                || "sync".equals(phase)
                || "selection".equals(phase)
                || "release".equals(phase))) {
          requestStepBudgets.merge(position, windowMedian, Double::sum);
          requestStepP10.merge(position, windowP10, Double::sum);
          requestStepP90.merge(position, windowP90, Double::sum);
        }
        requestTotal += windowMedian;
        requestTotalP10 += windowP10;
        requestTotalP90 += windowP90;
      }
      if (requestPrefill > 0.0) {
        prefillBudgets.add(requestPrefill);
        prefillP10Budgets.add(requestPrefillP10);
        prefillP90Budgets.add(requestPrefillP90);
      }
      requestTotals.add(requestTotal);
      requestTotalsP10.add(requestTotalP10);
      requestTotalsP90.add(requestTotalP90);
      for (Map.Entry<Long, Double> entry : requestStepBudgets.entrySet()) {
        stepSamples.computeIfAbsent(entry.getKey(), p -> new ArrayList<>()).add(entry.getValue());
        allStepSamples.add(entry.getValue());
        stepSamplesP10
            .computeIfAbsent(entry.getKey(), p -> new ArrayList<>())
            .add(requestStepP10.get(entry.getKey()));
        allStepP10.add(requestStepP10.get(entry.getKey()));
        stepSamplesP90
            .computeIfAbsent(entry.getKey(), p -> new ArrayList<>())
            .add(requestStepP90.get(entry.getKey()));
        allStepP90.add(requestStepP90.get(entry.getKey()));
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
    for (Map.Entry<Long, List<Double>> entry : stepSamples.entrySet()) {
      BaselineStatistics.Quantiles quantiles = BaselineStatistics.quantiles(entry.getValue());
      positions
          .addObject()
          .put("position", entry.getKey())
          .put("median_ns", quantiles.median())
          .put("p10_ns", quantiles.p10())
          .put("p90_ns", quantiles.p90())
          .put("calibration_p10_ns", calibrationP10(stepSamplesP10.get(entry.getKey())))
          .put("calibration_p90_ns", calibrationP90(stepSamplesP90.get(entry.getKey())));
    }
    stepSummary(perStep, allStepSamples, allStepP10, allStepP90);

    stepSummary(
        estimate.putObject("prefill"), prefillBudgets, prefillP10Budgets, prefillP90Budgets);
    stepSummary(
        estimate.putObject("request_total"), requestTotals, requestTotalsP10, requestTotalsP90);

    ObjectNode perShapeOut = estimate.putObject("per_shape");
    for (Map.Entry<String, double[]> entry : perShapeBudgets.entrySet()) {
      String key = entry.getKey();
      perShapeOut
          .putObject(key)
          .put("calls", perShapeCalls.get(key))
          .put("median_ns_per_call", shapes.path(key).path("median_ns").asDouble(0) * perCall)
          .put("p10_ns_per_call", shapes.path(key).path("p10_ns").asDouble(0) * perCall)
          .put("p90_ns_per_call", shapes.path(key).path("p90_ns").asDouble(0) * perCall)
          .put("budget_ns", Math.round(entry.getValue()[0]))
          .put("budget_p10_ns", Math.round(entry.getValue()[1]))
          .put("budget_p90_ns", Math.round(entry.getValue()[2]));
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

  /** The type-7 p10 of a calibration-p10 budget pool (its lower range endpoint). */
  private static double calibrationP10(List<Double> budgets) {
    return BaselineStatistics.quantiles(budgets).p10();
  }

  /** The type-7 p90 of a calibration-p90 budget pool (its upper range endpoint). */
  private static double calibrationP90(List<Double> budgets) {
    return BaselineStatistics.quantiles(budgets).p90();
  }

  /**
   * Type-7 quantiles (p10/median/p90) of a budget pool with its sample count, plus the
   * calibration's own sampling range for the same pool (review round 2, B): the type-7 p10 of the
   * budgets computed from each shape's calibration p10 samples, and the type-7 p90 of the budgets
   * computed from its p90 samples. The pool is one sample per measured request (or per (measured
   * request, position) for the per-step summary), never a sum across requests.
   */
  private static void stepSummary(
      ObjectNode node, List<Double> budgets, List<Double> p10Budgets, List<Double> p90Budgets) {
    BaselineStatistics.Quantiles quantiles = BaselineStatistics.quantiles(budgets);
    node.put("n", budgets.size());
    node.put("p10_ns", quantiles.p10());
    node.put("median_ns", quantiles.median());
    node.put("p90_ns", quantiles.p90());
    node.put("calibration_p10_ns", calibrationP10(p10Budgets));
    node.put("calibration_p90_ns", calibrationP90(p90Budgets));
  }

  /**
   * Maps a generated downcall's native function descriptor (the {@code FunctionDescriptor} toString
   * recorded by the worker's {@code binding_descriptors}) to a calibration shape key, or null when
   * the descriptor names a type the calibration does not model (review round 2, A).
   *
   * <p>Grammar: the descriptor is {@code (params)return}. Each top-level token is one of:
   *
   * <ul>
   *   <li>{@code a8} (optionally named, optionally carrying an {@code a8:[*:b1]} element
   *       annotation) — a pointer, P;
   *   <li>{@code [body](name)} — a struct passed by value; M when its byte size is 8 (every opaque
   *       mlx-c struct and every {@code mlx_optional_*} wrapper in the pinned header), unmodeled
   *       otherwise;
   *   <li>{@code i4}/{@code j8}/{@code f4}/{@code d8}/{@code z1}/{@code v} — the scalars I, J, F,
   *       D, Z, V.
   * </ul>
   *
   * <p>Struct bodies may nest pointers and structs (and carry padding as {@code xN}), so both the
   * token walk and the size walk match brackets instead of searching for the first close.
   */
  static String shapeKey(String descriptor) {
    if (descriptor == null || descriptor.length() < 3 || descriptor.charAt(0) != '(') {
      return null;
    }
    List<String> tokens = new ArrayList<>();
    int i = 1;
    boolean params = true;
    try {
      while (i < descriptor.length()) {
        if (params && descriptor.charAt(i) == ')') {
          params = false;
          i++;
          continue;
        }
        i = token(descriptor, i, descriptor.length(), tokens);
      }
    } catch (RuntimeException e) {
      return null;
    }
    if (params || tokens.isEmpty()) {
      return null;
    }
    StringBuilder key = new StringBuilder("(");
    for (int t = 0; t < tokens.size() - 1; t++) {
      key.append(tokens.get(t));
    }
    return key.append(')').append(tokens.get(tokens.size() - 1)).toString();
  }

  /**
   * Classifies the top-level token at {@code i}; appends its key character and returns the index
   * past it.
   */
  private static int token(String s, int i, int limit, List<String> out) {
    char c = s.charAt(i);
    if (c == '[') {
      int j = matchBracket(s, i, limit);
      int nameEnd = structNameEnd(s, j, limit);
      // The body runs from i + 1 to j - 2: j is one past the struct's own closing bracket.
      long size = structSize(s, i + 1, j - 1);
      if (size != 8) {
        throw new IllegalArgumentException("unmodeled by-value struct size " + size);
      }
      out.add("M");
      return nameEnd;
    }
    if (s.startsWith("a8", i)) {
      int j = skipName(s, i + 2, limit);
      if (s.startsWith(":[*:b1]", j)) {
        j += 7;
      }
      out.add("P");
      return j;
    }
    if (s.startsWith("i4", i)) {
      out.add("I");
      return skipName(s, i + 2, limit);
    }
    if (s.startsWith("j8", i)) {
      out.add("J");
      return skipName(s, i + 2, limit);
    }
    if (s.startsWith("f4", i)) {
      out.add("F");
      return skipName(s, i + 2, limit);
    }
    if (s.startsWith("d8", i)) {
      out.add("D");
      return skipName(s, i + 2, limit);
    }
    if (s.startsWith("z1", i)) {
      out.add("Z");
      return skipName(s, i + 2, limit);
    }
    if (c == 'v') {
      out.add("V");
      return i + 1;
    }
    throw new IllegalArgumentException("unmodeled token at " + i);
  }

  /** The index just past the ']' matching the '[' at {@code i}, or a fail. */
  private static int matchBracket(String s, int i, int limit) {
    int depth = 0;
    for (int j = i; j < limit; j++) {
      char ch = s.charAt(j);
      if (ch == '[') {
        depth++;
      } else if (ch == ']') {
        depth--;
        if (depth == 0) {
          return j + 1;
        }
      }
    }
    throw new IllegalArgumentException("unterminated struct at " + i);
  }

  /**
   * The index just past the {@code (name)} following a struct body; the name never nests brackets,
   * so it cannot contain a struct's own close paren.
   */
  private static int structNameEnd(String s, int bodyEnd, int limit) {
    if (bodyEnd >= limit || s.charAt(bodyEnd) != '(') {
      throw new IllegalArgumentException("struct without a name at " + bodyEnd);
    }
    int nameEnd = s.indexOf(')', bodyEnd + 1);
    if (nameEnd < 0 || nameEnd > limit) {
      throw new IllegalArgumentException("unterminated struct name at " + bodyEnd);
    }
    return nameEnd + 1;
  }

  /** The byte size of a struct body (the sum of its members' sizes), or a fail. */
  private static long structSize(String s, int from, int to) {
    long size = 0;
    int i = from;
    while (i < to) {
      char c = s.charAt(i);
      if (c == '[') {
        int j = matchBracket(s, i, to);
        int nameEnd = structNameEnd(s, j, to);
        size += structSize(s, i + 1, j - 1);
        i = nameEnd;
      } else if (c == 'x') {
        int j = i + 1;
        while (j < to && Character.isDigit(s.charAt(j))) {
          j++;
        }
        if (j == i + 1) {
          throw new IllegalArgumentException("unterminated padding at " + i);
        }
        size += Long.parseLong(s.substring(i + 1, j));
        i = j;
      } else if (s.startsWith("a8", i)) {
        int j = skipName(s, i + 2, to);
        if (s.startsWith(":[*:b1]", j)) {
          j += 7;
        }
        size += 8;
        i = j;
      } else if (s.startsWith("i4", i) || s.startsWith("f4", i)) {
        size += 4;
        i = skipName(s, i + 2, to);
      } else if (s.startsWith("j8", i) || s.startsWith("d8", i)) {
        size += 8;
        i = skipName(s, i + 2, to);
      } else if (s.startsWith("z1", i)) {
        size += 1;
        i = skipName(s, i + 2, to);
      } else {
        throw new IllegalArgumentException("unmodeled member at " + i);
      }
    }
    return size;
  }

  /** The index just past an optional {@code (name)} at {@code i} (a token suffix, not a param). */
  private static int skipName(String s, int i, int limit) {
    if (i < limit && s.charAt(i) == '(') {
      int close = s.indexOf(')', i);
      if (close < 0 || close > limit) {
        throw new IllegalArgumentException("unterminated name at " + i);
      }
      return close + 1;
    }
    return i;
  }
}
