package se.alipsa.jmlx.benchmarks;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Raw-to-summary verifier for the published Phase 8.2 evidence (WP5, plan section 7): re-derives
 * every aggregate in {@code req/benchmarks/phase8-2/summary.json} from the committed raw trees
 * {@code req/benchmarks/phase8-2/raw/wp3} and {@code raw/wp4} (639 files, SHA-256-pinned in {@code
 * evidence-index.json}), and pins the measured code, the workload manifest and the Tier-B input
 * bytes against what the workers recorded. A reviewer can run this one test to confirm that every
 * published number is reproducible from the committed raw bytes, that the exact measured code and
 * input bytes are identified, and that the FFM budget medians hold within the calibration-recording
 * bound stated in {@code wp3_ffm.note}. Pure Java (no native runtime, no network); part of {@code
 * :jmlx-benchmarks:test}, so it runs on the cross-platform Linux CI job as well.
 *
 * <p>Conventions re-implemented here (identical to the Python generator that produced {@code
 * summary.json}): type-7 quantiles via {@link BaselineStatistics} (bit-identical to the generator's
 * linear interpolation), per-pair warm medians over the 20 measured requests, cell quantiles over
 * the pair medians of the cell's pair set, ratios as the type-7 median of the per-pair java/python
 * median ratios, and the FFM estimate from {@link FfmTransitionEstimate} over the committed
 * binding-count reports with the calibration reconstructed from the committed record (per-call
 * medians times the workload's 2000 calls, p10 = p90 = median, because the measurement-time
 * per-shape ranges are not archived).
 */
final class Phase82SummaryVerifierTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final String EVIDENCE = "req/benchmarks/phase8-2";
  private static final String RAW = EVIDENCE + "/raw";
  private static final String WP3_DIAG_SESSION = "diag-5-1-20261109";
  private static final String[] CASES = {"qwen3-0.6b", "qwen3-8b-4bit", "tinymistral-6x248m"};
  private static final int[] LENGTHS = {32, 512, 2048};
  private static final String[] METRICS = {
    "full_generation_s", "streaming_decode_tps", "ttft_s", "prefill_s"
  };
  // WP3 measured code (findings 16.1; every worker's identity block pins it).
  private static final String WP3_CODE_SHA = "cc0c603455caca69d434963990843aca98e6e606";
  private static final String WP3_BRANCH = "phase8-2-wp3";
  private static final String JAVA_VERSION = "25.0.4.1";
  private static final String JAVA_VENDOR = "Eclipse Adoptium";
  // WP4 both-pin trees (findings 17.1/17.3/17.6; the session-report tree blocks pin them).
  private static final String WP4_OLD_TREE = "46cb325bec9b4878534cc5c6bda1a569dfe4aa4c";
  private static final Map<String, String> WP4_NEW_TREES =
      Map.of(
          "wp4-observed-cache-1", "e4cb1f0f43ddd90242237f7e155a399ca5c84cfd",
          "wp4-observed-cache-2", "ecc69a50e7a7263e955ac20c716b71344ce86c34");
  private static final int WP3_WORKER_COUNT = 344;
  private static final int WP4_WORKER_COUNT = 52;
  // The FFM calibration workload (findings 15.4, the index's ffm_calibration_record.workload
  // string): 5 forks x 200 iterations x 2000 calls.
  private static final long CALIBRATION_CALLS = 2000;
  // The four struct-return shapes recorded to 1 decimal in findings 15.4 (the rest to 2);
  // their per-call recording bound is therefore 0.05 ns/call instead of 0.005 ns/call.
  private static final Set<String> ONE_DECIMAL_SHAPES = Set.of("()M", "(F)M", "(PJ)M", "(PPII)M");
  private static final Pattern WP3_SESSION_DIR =
      Pattern.compile("session-([a-z]+\\d*)-(controlled|stock)-pair(\\d+)-\\d{8}");
  // Published FFM count semantics (findings 16.6): phase name and the uniformity coefficient
  // (2540 = the per-position step window, uniform across 20 x 127 windows; 20 = the per-request
  // sum, uniform across the 20 measured requests).
  private static final Map<String, String> PHASE_NAMES =
      Map.of(
          "prefill_construction", "prefill-construction",
          "prefill_sync", "prefill-sync",
          "prefill_selection", "prefill-selection",
          "prefill_release", "prefill-release",
          "step_construction", "construction",
          "step_sync", "sync",
          "step_selection", "selection",
          "step_release", "release",
          "request_release", "request-release");
  private static final Set<String> STEP_PHASES =
      Set.of("construction", "sync", "selection", "release");

  private Phase82SummaryVerifierTest() {}

  /**
   * The repository root, located from the test class's code source (the committed evidence lives in
   * it).
   */
  private static Path repoRoot() throws URISyntaxException {
    Path here =
        Path.of(
            Objects.requireNonNull(
                    Phase82SummaryVerifierTest.class
                        .getProtectionDomain()
                        .getCodeSource()
                        .getLocation())
                .toURI());
    Path root = here;
    while (root != null && !Files.isRegularFile(root.resolve(EVIDENCE + "/evidence-index.json"))) {
      root = root.getParent();
    }
    assertNotNull(root, "could not locate the repository root from the test class location");
    return root;
  }

  private static JsonNode read(Path path) {
    return MAPPER.readTree(path.toFile());
  }

  private static JsonNode summary() throws URISyntaxException {
    return read(repoRoot().resolve(EVIDENCE + "/summary.json"));
  }

  private static JsonNode index() throws URISyntaxException {
    return read(repoRoot().resolve(EVIDENCE + "/evidence-index.json"));
  }

  private static void assertSame(double expected, double actual, String context) {
    assertEquals(
        expected, actual, 0.0, context + " (expected " + expected + ", got " + actual + ")");
  }

  private static double median(List<Double> values) {
    return BaselineStatistics.quantiles(values).median();
  }

  /**
   * CPython 3.12's built-in {@code sum} over floats ({@code bltin_sum} in {@code bltinmodule.c}):
   * Neumaier-compensated left-to-right addition, starting from 0.0, with the {@code t != total}
   * guard, returning {@code total + c} at the end. The summary generator sums the tracing windows
   * with this exact algorithm, so a naive Java accumulation diverges by ULPs on the residual
   * medians; bit-identical parsing (verified against the raw JSON's shortest-repr doubles) makes
   * this loop the only summation that reproduces the published values.
   */
  private static double pySum(List<Double> values) {
    double total = 0.0;
    double c = 0.0;
    for (double v : values) {
      if (Math.abs(total) >= Math.abs(v)) {
        double t = total + v;
        if (t != total) {
          c += (total - t) + v;
        }
        total = t;
      } else {
        double t = v + total;
        if (t != v) {
          c += (v - t) + total;
        }
        total = t;
      }
    }
    return total + c;
  }

  private static void assertMinMedMax(List<Double> values, JsonNode published, String context) {
    assertEquals(
        values.stream().min(Comparator.naturalOrder()).get(),
        published.path("min").asDouble(),
        0.0,
        context + ".min");
    assertEquals(
        values.stream().max(Comparator.naturalOrder()).get(),
        published.path("max").asDouble(),
        0.0,
        context + ".max");
    assertSame(median(values), published.path("median").asDouble(), context + ".median");
  }

  // ---------------------------------------------------------------- raw tree integrity

  @Test
  void rawTreeIsPinnedAndClean() throws URISyntaxException, IOException {
    Path root = repoRoot();
    JsonNode idx = index();
    assertEquals("jmlx-phase8-2-evidence-index-1", idx.path("schema").asString());
    byte[] marker = "/Users/per".getBytes(StandardCharsets.UTF_8);
    int checked = 0;
    for (String part : new String[] {"wp3", "wp4"}) {
      JsonNode committed = idx.path("committed_files").path(part);
      assertTrue(committed.isObject(), part + " committed_files missing");
      Set<String> onDisk = new HashSet<>();
      try (var stream = Files.walk(root.resolve(RAW + "/" + part))) {
        for (Path p : stream.filter(Files::isRegularFile).toList()) {
          onDisk.add(root.resolve(RAW + "/" + part).relativize(p).toString().replace('\\', '/'));
        }
      }
      assertEquals(onDisk.size(), committed.size(), part + ": index/disk file-count mismatch");
      for (Map.Entry<String, JsonNode> e : committed.properties()) {
        Path file = root.resolve(RAW + "/" + part + "/" + e.getKey());
        assertTrue(
            onDisk.contains(e.getKey()), part + ": index lists a missing file " + e.getKey());
        byte[] bytes = Files.readAllBytes(file);
        assertEquals(
            e.getValue().asString(),
            BaselineBenchmark.sha256File(file),
            part + " digest mismatch: " + e.getKey());
        for (int i = 0; i + marker.length <= bytes.length; i++) {
          boolean hit = true;
          for (int j = 0; j < marker.length; j++) {
            if (bytes[i + j] != marker[j]) {
              hit = false;
              break;
            }
          }
          if (hit) {
            fail(part + " file " + e.getKey() + " contains the local username path prefix");
          }
        }
        checked++;
      }
    }
    assertEquals(458 + 181, checked, "total committed raw files");
    // The workload manifest digest is pinned three ways: the index, the launcher constant and disk.
    String manifestSha =
        idx.path("sources").path("wp3").path("workload_manifest").path("sha256").asString();
    assertEquals(
        BaselineBenchmark.PINNED_MANIFEST_SHA256,
        manifestSha,
        "index manifest sha vs launcher pin");
    assertEquals(
        BaselineBenchmark.PINNED_MANIFEST_SHA256,
        BaselineBenchmark.sha256File(root.resolve("tools/decode-baseline/workloads.json")),
        "workloads.json on disk vs pin");
    // The pre-normalization digest manifests (committed under req/plans/) pin the pre-copy tree.
    for (String part : new String[] {"wp3", "wp4"}) {
      String manifest = idx.path("sources").path(part).path("manifest").asString();
      assertEquals(
          idx.path("sources").path(part).path("manifest_sha256").asString(),
          BaselineBenchmark.sha256File(root.resolve(manifest)),
          part + " raw-digest manifest on disk vs index");
    }
  }

  // ---------------------------------------------------------------- summary head

  @Test
  void summaryHeadIsPinned() throws URISyntaxException {
    JsonNode s = summary();
    assertEquals("jmlx-phase8-2-summary-1", s.path("schema").asString());
    assertEquals("linear-interpolation-type-7", s.path("quantile_convention").asString());
    JsonNode code = s.path("measured_code").path("wp3");
    assertEquals(WP3_CODE_SHA, code.path("sha").asString());
    assertEquals(WP3_BRANCH, code.path("branch").asString());
    assertFalse(code.path("dirty").asBoolean());
    JsonNode wp4 = s.path("measured_code").path("wp4");
    assertEquals(WP4_OLD_TREE, wp4.path("old_tree").asString());
    for (Map.Entry<String, String> e : WP4_NEW_TREES.entrySet()) {
      assertEquals(e.getValue(), wp4.path("new_tree").path(e.getKey()).asString());
    }
    JsonNode host = s.path("measured_code").path("host");
    assertEquals("Mac17,6", host.path("chip").asString());
    assertEquals(137438953472L, host.path("ram_bytes").asLong());
    assertEquals("Darwin", host.path("os").asString());
    assertEquals("27.2.0", host.path("os_build").asString());
    assertEquals(JAVA_VENDOR, s.path("measured_code").path("java").path("vendor").asString());
    assertEquals(JAVA_VERSION, s.path("measured_code").path("java").path("version").asString());
    assertEquals(
        "MLX_ENABLE_TF32=1", s.path("measured_code").path("precision").path("wp3").asString());
    assertEquals(
        "MLX_ENABLE_TF32=0", s.path("measured_code").path("precision").path("wp4").asString());
    assertEquals(
        BaselineBenchmark.PINNED_MANIFEST_SHA256,
        s.path("measured_code").path("workload_manifest").path("sha256").asString());
    JsonNode head = s.path("wp3_text_baseline");
    assertEquals("headline", head.path("mode").asString());
    assertEquals("MLX_ENABLE_TF32=1", head.path("precision_env").asString());
    assertEquals("primed-per-request", head.path("cache_regime").asString());
    assertEquals(128, head.path("emitted_tokens").asInt());
    assertEquals(1, head.path("first_requests").asInt());
    assertEquals(5, head.path("warmups").asInt());
    assertEquals(20, head.path("measured_requests").asInt());
  }

  // ---------------------------------------------------------------- wp3 headline cells

  private record SessionRef(String dirName, String adapter) {}

  /** (block, pair) -> the session directory and its adapter, from the committed tree layout. */
  private static Map<String, SessionRef> wp3SessionDirs() throws URISyntaxException, IOException {
    Map<String, SessionRef> out = new TreeMap<>();
    Path sessions = repoRoot().resolve(RAW + "/wp3/sessions");
    try (var stream = Files.list(sessions)) {
      for (Path dir : stream.filter(Files::isDirectory).toList()) {
        var m = WP3_SESSION_DIR.matcher(dir.getFileName().toString());
        if (m.matches()) {
          String key = m.group(1) + "|" + m.group(3);
          assertFalse(out.containsKey(key), "duplicate session directory for " + key);
          out.put(key, new SessionRef(dir.getFileName().toString(), m.group(2)));
        }
      }
    }
    return out;
  }

  /**
   * One parsed WP3 headline worker: status, the four metric lists over its measured requests (null
   * entries where the report records JSON null, as the stock workers do for prefill), the measured
   * token lists, the load time and the single first request's full generation time.
   */
  private record Wp3Metrics(
      String status,
      List<Double> fullGeneration,
      List<Double> tps,
      List<Double> ttft,
      List<Double> prefill,
      List<List<Integer>> tokens,
      double loadS,
      double firstFullGeneration) {
    static Wp3Metrics parse(JsonNode worker) {
      List<Double> fg = new ArrayList<>();
      List<Double> tps = new ArrayList<>();
      List<Double> ttft = new ArrayList<>();
      List<Double> prefill = new ArrayList<>();
      List<List<Integer>> tokens = new ArrayList<>();
      double first = Double.NaN;
      for (JsonNode q : worker.path("requests")) {
        String kind = q.path("kind").asString("");
        if ("measured".equals(kind)) {
          for (int i = 0; i < METRICS.length; i++) {
            JsonNode v = q.path(METRICS[i]);
            List<Double> target = i == 0 ? fg : (i == 1 ? tps : (i == 2 ? ttft : prefill));
            target.add(v.isNull() ? null : v.asDouble());
          }
          List<Integer> toks = new ArrayList<>();
          for (JsonNode t : q.path("tokens")) {
            toks.add(t.asInt());
          }
          tokens.add(toks);
        } else if ("first".equals(kind)) {
          first = q.path("full_generation_s").asDouble();
        }
      }
      return new Wp3Metrics(
          worker.path("status").asString(""),
          fg,
          tps,
          ttft,
          prefill,
          tokens,
          worker.path("load_s").asDouble(),
          first);
    }
  }

  private static Wp3Metrics wp3Worker(
      Map<String, SessionRef> dirs,
      String block,
      int pair,
      String runtime,
      String adapter,
      String caseName,
      int length)
      throws URISyntaxException {
    SessionRef ref = dirs.get(block + "|" + pair);
    assertNotNull(ref, "no committed session directory for block " + block + " pair " + pair);
    assertEquals(adapter, ref.adapter(), "adapter mismatch for " + block + "|" + pair);
    String workerDir = runtime + "-" + adapter + "-" + caseName + "-" + length + "-headline";
    Path report =
        repoRoot()
            .resolve(RAW + "/wp3/sessions/" + ref.dirName() + "/" + workerDir + "/report.json");
    assertTrue(Files.isRegularFile(report), "missing worker report " + report);
    return Wp3Metrics.parse(read(report).path("worker"));
  }

  private static Wp3Metrics tryWp3Worker(
      Map<String, SessionRef> dirs,
      String block,
      int pair,
      String runtime,
      String adapter,
      String caseName,
      int length)
      throws URISyntaxException {
    SessionRef ref = dirs.get(block + "|" + pair);
    if (ref == null || !adapter.equals(ref.adapter())) {
      return null;
    }
    String workerDir = runtime + "-" + adapter + "-" + caseName + "-" + length + "-headline";
    Path report =
        repoRoot()
            .resolve(RAW + "/wp3/sessions/" + ref.dirName() + "/" + workerDir + "/report.json");
    if (!Files.isRegularFile(report)) {
      return null;
    }
    return Wp3Metrics.parse(read(report).path("worker"));
  }

  private static List<JsonNode> pairEntries(JsonNode pairSet) {
    assertTrue(pairSet.isArray() && pairSet.size() > 0, "pair_set must be a nonempty array");
    List<JsonNode> out = new ArrayList<>();
    for (JsonNode p : pairSet) {
      out.add(p);
    }
    return out;
  }

  private static void verifySettledCell(JsonNode cell, String adapter, Map<String, SessionRef> dirs)
      throws URISyntaxException {
    String caseName = cell.path("case").asString();
    int length = cell.path("prompt_length").asInt();
    assertEquals(
        length == 2048 && "qwen3-8b-4bit".equals(caseName),
        cell.path("io_affected").asBoolean(),
        caseName + "/" + length + " io_affected");
    List<JsonNode> pairs = pairEntries(cell.path("pair_set"));
    assertEquals(pairs.size(), cell.path("n_pairs").asInt(), caseName + "/" + length);
    // per metric, per runtime: the per-pair value lists (possibly with nulls)
    Map<String, List<List<Double>>> values = new LinkedHashMap<>();
    for (String rt : new String[] {"java", "python"}) {
      for (String m : METRICS) {
        values.put(rt + "/" + m, new ArrayList<>());
      }
    }
    for (JsonNode p : pairs) {
      String block = p.path("session").asString();
      int pair = p.path("pair").asInt();
      for (String rt : new String[] {"java", "python"}) {
        Wp3Metrics w = wp3Worker(dirs, block, pair, rt, adapter, caseName, length);
        assertEquals(
            "ok", w.status(), rt + " " + caseName + "/" + length + " " + block + " pair " + pair);
        assertEquals(20, w.fullGeneration().size(), "measured request count");
        for (int i = 0; i < METRICS.length; i++) {
          List<Double> list =
              i == 0 ? w.fullGeneration() : (i == 1 ? w.tps() : (i == 2 ? w.ttft() : w.prefill()));
          values.get(rt + "/" + METRICS[i]).add(list);
        }
      }
    }
    for (String rt : new String[] {"java", "python"}) {
      for (String m : METRICS) {
        List<List<Double>> perPair = values.get(rt + "/" + m);
        boolean allNull = perPair.stream().flatMap(List::stream).allMatch(Objects::isNull);
        JsonNode published = cell.path(rt).path(m);
        if (allNull) {
          assertTrue(published.isMissingNode(), rt + " " + m + " must be absent (all null)");
          continue;
        }
        List<Double> pairMeds = new ArrayList<>();
        for (List<Double> v : perPair) {
          pairMeds.add(median(v.stream().filter(Objects::nonNull).toList()));
        }
        BaselineStatistics.Quantiles q = BaselineStatistics.quantiles(pairMeds);
        // The settled-cell quantiles are the [p10, median, p90] array (the summary's convention).
        assertSame(q.p10(), published.get(0).asDouble(), rt + " " + m + " p10");
        assertSame(q.median(), published.get(1).asDouble(), rt + " " + m + " median");
        assertSame(q.p90(), published.get(2).asDouble(), rt + " " + m + " p90");
      }
    }
    for (String m : METRICS) {
      boolean javaAllNull =
          values.get("java/" + m).stream().flatMap(List::stream).allMatch(Objects::isNull);
      JsonNode published = cell.path("ratio").path(m);
      if (javaAllNull) {
        assertTrue(published.isMissingNode(), "ratio " + m + " must be absent (all null)");
        continue;
      }
      List<Double> ratios = new ArrayList<>();
      for (int i = 0; i < pairs.size(); i++) {
        List<Double> javaPair =
            values.get("java/" + m).get(i).stream().filter(Objects::nonNull).toList();
        List<Double> pythonPair =
            values.get("python/" + m).get(i).stream().filter(Objects::nonNull).toList();
        ratios.add(median(javaPair) / median(pythonPair));
      }
      assertSame(median(ratios), published.asDouble(), "ratio " + m);
    }
  }

  @Test
  void wp3HeadlineCellsRecomputeFromRaw() throws URISyntaxException, IOException {
    JsonNode wp3 = summary().path("wp3_text_baseline");
    Map<String, SessionRef> dirs = wp3SessionDirs();
    JsonNode controlled = wp3.path("controlled");
    for (JsonNode cell : controlled.path("settled")) {
      verifySettledCell(cell, "controlled", dirs);
    }
    for (JsonNode cell : controlled.path("unstable")) {
      String caseName = cell.path("case").asString();
      int length = cell.path("prompt_length").asInt();
      assertTrue(cell.path("io_affected").asBoolean());
      List<JsonNode> pairs = pairEntries(cell.path("pair_set"));
      assertEquals(pairs.size(), cell.path("n_pairs").asInt());
      for (String rt : new String[] {"java", "python"}) {
        for (String m : new String[] {"full_generation_s", "streaming_decode_tps"}) {
          List<Double> pairMeds = new ArrayList<>();
          for (JsonNode p : pairs) {
            Wp3Metrics w =
                wp3Worker(
                    dirs,
                    p.path("session").asString(),
                    p.path("pair").asInt(),
                    rt,
                    "controlled",
                    caseName,
                    length);
            assertEquals("ok", w.status());
            assertEquals(20, w.fullGeneration().size());
            List<Double> vals = new ArrayList<>();
            for (double x : (m.equals("full_generation_s") ? w.fullGeneration() : w.tps())) {
              vals.add(x);
            }
            pairMeds.add(median(vals));
          }
          BaselineStatistics.Quantiles q = BaselineStatistics.quantiles(pairMeds);
          assertSame(q.p10(), cell.path(rt).path(m).path("p10").asDouble(), rt + " " + m + " p10");
          assertSame(q.p90(), cell.path(rt).path(m).path("p90").asDouble(), rt + " " + m + " p90");
        }
      }
      assertTrue(cell.path("ratio").isMissingNode(), "unstable cells carry no ratio");
    }
    for (JsonNode cell : wp3.path("stock").path("settled")) {
      verifySettledCell(cell, "stock", dirs);
    }
    JsonNode blocked = wp3.path("stock").path("blocked");
    assertEquals(1, blocked.size());
    JsonNode cell = blocked.get(0);
    verifySettledCell(cell, "stock", dirs);
    String caseName = cell.path("case").asString();
    int length = cell.path("prompt_length").asInt();
    JsonNode excluded = cell.path("excluded_members");
    assertEquals(2, excluded.size());
    List<JsonNode> allPairs = new ArrayList<>(pairEntries(cell.path("pair_set")));
    for (JsonNode e : excluded) {
      assertEquals("java", e.path("runtime").asString());
      Wp3Metrics w =
          wp3Worker(
              dirs,
              e.path("session").asString(),
              e.path("pair").asInt(),
              "java",
              "stock",
              caseName,
              length);
      assertEquals(e.path("n_measured").asInt(), w.fullGeneration().size());
      assertEquals(20, w.fullGeneration().size());
      assertSame(
          median(w.fullGeneration()),
          e.path("validated_full_generation_s_median").asDouble(),
          "excluded "
              + e.path("session").asString()
              + " pair "
              + e.path("pair").asInt()
              + " fg median");
      assertSame(
          median(w.tps()),
          e.path("validated_streaming_decode_tps_median").asDouble(),
          "excluded tps median");
      allPairs.add(e);
    }
    assertEquals(12, allPairs.size(), "with-excluded pair count");
    for (String m : new String[] {"full_generation_s", "streaming_decode_tps"}) {
      List<Double> ratios = new ArrayList<>();
      for (JsonNode p : allPairs) {
        Wp3Metrics j =
            wp3Worker(
                dirs,
                p.path("session").asString(),
                p.path("pair").asInt(),
                "java",
                "stock",
                caseName,
                length);
        Wp3Metrics py =
            wp3Worker(
                dirs,
                p.path("session").asString(),
                p.path("pair").asInt(),
                "python",
                "stock",
                caseName,
                length);
        List<Double> jv = m.equals("full_generation_s") ? j.fullGeneration() : j.tps();
        List<Double> pv = m.equals("full_generation_s") ? py.fullGeneration() : py.tps();
        ratios.add(median(jv) / median(pv));
      }
      assertSame(
          median(ratios),
          cell.path("with_excluded_ratio").path(m).asDouble(),
          "with_excluded_ratio " + m);
    }
    assertEquals(12, cell.path("with_excluded_ratio").path("n_pairs").asInt());
    // The protocol constants of the summary head are pinned by every worker's own identity.
    JsonNode identity =
        read(repoRoot()
                .resolve(RAW + "/wp3/sessions/")
                .resolve(
                    dirs.get("a|0").dirName()
                        + "/java-controlled-qwen3-0.6b-32-headline/report.json"))
            .path("worker")
            .path("identity")
            .path("run_policy");
    assertEquals(128, identity.path("emitted_tokens").asInt());
    assertEquals(1, identity.path("first_requests").asInt());
    assertEquals(5, identity.path("warmups").asInt());
    assertEquals(20, identity.path("measured_requests").asInt());
    assertEquals("primed-per-request", identity.path("cache_regime").asString());
  }

  // ---------------------------------------------------------------- wp3 greedy identity

  @Test
  void wp3GreedyIdentityRecomputesFromRaw() throws URISyntaxException, IOException {
    JsonNode wp3 = summary().path("wp3_text_baseline");
    Map<String, SessionRef> dirs = wp3SessionDirs();
    verifyIdentityBlock(wp3.path("controlled").path("greedy_identity"), "controlled", dirs);
    verifyIdentityBlock(wp3.path("stock").path("greedy_identity"), "stock", dirs);
  }

  private static void verifyIdentityBlock(
      JsonNode blockNode, String adapter, Map<String, SessionRef> dirs)
      throws URISyntaxException, IOException {
    assertTrue(blockNode.isObject() && !blockNode.isEmpty());
    for (Map.Entry<String, JsonNode> blockEntry : blockNode.properties()) {
      String name = blockEntry.getKey();
      assertTrue(name.startsWith("session_"), "unexpected identity block " + name);
      String blockId = name.substring("session_".length());
      List<Map.Entry<String, SessionRef>> members = new ArrayList<>();
      for (Map.Entry<String, SessionRef> e : dirs.entrySet()) {
        String[] bp = e.getKey().split("\\|");
        if (bp[0].equals(blockId)) {
          assertEquals(adapter, e.getValue().adapter(), name + " adapter consistency");
          members.add(e);
        }
      }
      members.sort(Comparator.comparingInt(e -> Integer.parseInt(e.getKey().split("\\|")[1])));
      assertTrue(!members.isEmpty(), "no committed session directories for " + name);
      Map<String, int[]> recomputed = new TreeMap<>();
      for (String caseName : CASES) {
        for (int length : LENGTHS) {
          int total = 0;
          int same = 0;
          for (Map.Entry<String, SessionRef> member : members) {
            int pair = Integer.parseInt(member.getKey().split("\\|")[1]);
            Wp3Metrics j = tryWp3Worker(dirs, blockId, pair, "java", adapter, caseName, length);
            Wp3Metrics py = tryWp3Worker(dirs, blockId, pair, "python", adapter, caseName, length);
            if (j == null || py == null || !"ok".equals(j.status()) || !"ok".equals(py.status())) {
              continue;
            }
            if (j.tokens().size() != 20 || py.tokens().size() != 20) {
              continue;
            }
            for (int i = 0; i < 20; i++) {
              total++;
              if (j.tokens().get(i).equals(py.tokens().get(i))) {
                same++;
              }
            }
          }
          if (total > 0) {
            recomputed.put(caseName + "/" + length, new int[] {same, total});
          }
        }
      }
      JsonNode published = blockEntry.getValue();
      Set<String> publishedKeys = new HashSet<>();
      for (Map.Entry<String, JsonNode> e : published.properties()) {
        publishedKeys.add(e.getKey());
      }
      assertEquals(recomputed.keySet(), publishedKeys, name + " compared-cell set");
      for (Map.Entry<String, int[]> e : recomputed.entrySet()) {
        assertSame(
            (double) e.getValue()[0],
            published.path(e.getKey()).path("identical").asDouble(),
            name + " " + e.getKey() + " identical");
        assertSame(
            (double) e.getValue()[1],
            published.path(e.getKey()).path("total").asDouble(),
            name + " " + e.getKey() + " total");
      }
    }
  }

  // ---------------------------------------------------------------- wp3 first requests

  @Test
  void wp3FirstRequestsSessionARecomputeFromRaw() throws URISyntaxException, IOException {
    JsonNode first =
        summary().path("wp3_text_baseline").path("controlled").path("first_requests_session_a");
    assertEquals(9, first.size());
    Map<String, SessionRef> dirs = wp3SessionDirs();
    for (JsonNode e : first) {
      String caseName = e.path("case").asString();
      int length = e.path("prompt_length").asInt();
      for (String rt : new String[] {"java", "python"}) {
        List<Double> fr = new ArrayList<>();
        List<Double> load = new ArrayList<>();
        for (int pair = 0; pair < 6; pair++) {
          Wp3Metrics w = wp3Worker(dirs, "a", pair, rt, "controlled", caseName, length);
          assertEquals("ok", w.status());
          fr.add(w.firstFullGeneration());
          load.add(w.loadS());
        }
        assertMinMedMax(fr, e.path(rt).path("full_generation_s"), rt + " first full_generation_s");
        assertMinMedMax(load, e.path(rt).path("load_s"), rt + " first load_s");
      }
    }
  }

  // ---------------------------------------------------------------- wp3 diagnostics

  @Test
  void wp3DiagnosticsRecomputeFromRaw() throws URISyntaxException {
    JsonNode cells = summary().path("wp3_diagnostics").path("cells");
    assertEquals(9, cells.size());
    for (JsonNode cell : cells) {
      String caseName = cell.path("case").asString();
      int length = cell.path("prompt_length").asInt();
      Path base = repoRoot().resolve(RAW + "/wp3/sessions/" + WP3_DIAG_SESSION);
      JsonNode tr =
          read(base.resolve(
                  "java-controlled-" + caseName + "-" + length + "-headline-tracing/report.json"))
              .path("worker");
      List<JsonNode> tb = new ArrayList<>();
      for (JsonNode q : tr.path("requests")) {
        if ("measured".equals(q.path("kind").asString())) {
          tb.add(q);
        }
      }
      assertEquals(20, tb.size());
      List<Double> wall = new ArrayList<>();
      List<Double> cpu = new ArrayList<>();
      List<Double> residualWall = new ArrayList<>();
      List<Double> residualCpu = new ArrayList<>();
      for (JsonNode q : tb) {
        JsonNode t = q.path("tracing");
        wall.add(t.path("request_wall_s").asDouble() * 1000);
        cpu.add(t.path("request_cpu_s").asDouble() * 1000);
        // The generator's sum() over the tracing windows (see pySum).
        List<Double> windowWall = new ArrayList<>();
        List<Double> windowCpu = new ArrayList<>();
        for (JsonNode w : t.path("windows")) {
          windowWall.add(w.path("wall_s").asDouble());
          windowCpu.add(w.path("cpu_s").asDouble());
        }
        residualWall.add((t.path("request_wall_s").asDouble() - pySum(windowWall)) * 1000);
        residualCpu.add((t.path("request_cpu_s").asDouble() - pySum(windowCpu)) * 1000);
      }
      JsonNode pub = cell.path("tracing");
      assertSame(
          median(wall), pub.path("wall_ms").asDouble(), caseName + "/" + length + " wall_ms");
      assertSame(median(cpu), pub.path("cpu_ms").asDouble(), caseName + "/" + length + " cpu_ms");
      assertSame(
          median(residualWall),
          pub.path("residual_wall_ms").asDouble(),
          caseName + "/" + length + " residual_wall_ms");
      assertSame(
          median(residualCpu),
          pub.path("residual_cpu_ms").asDouble(),
          caseName + "/" + length + " residual_cpu_ms");
      for (Map.Entry<String, String> phase :
          Map.of(
                  "prefill_sync_us", "prefill-sync",
                  "request_release_us", "request-release")
              .entrySet()) {
        List<Double> vals = new ArrayList<>();
        for (JsonNode q : tb) {
          for (JsonNode w : q.path("tracing").path("windows")) {
            if (phase.getValue().equals(w.path("phase").asString())) {
              vals.add(w.path("wall_s").asDouble() * 1e6);
            }
          }
        }
        assertSame(
            median(vals),
            pub.path(phase.getKey()).asDouble(),
            caseName + "/" + length + " " + phase.getKey());
      }
      for (String phase : new String[] {"construction", "sync", "selection", "release"}) {
        List<Double> vals = new ArrayList<>();
        for (JsonNode q : tb) {
          for (JsonNode w : q.path("tracing").path("windows")) {
            if (phase.equals(w.path("phase").asString()) && w.path("position").asLong() > 0) {
              vals.add(w.path("wall_s").asDouble() * 1e6);
            }
          }
        }
        assertSame(
            median(vals),
            pub.path("step_" + phase + "_us").asDouble(),
            caseName + "/" + length + " step_" + phase + "_us");
      }
      // Memory: medians over the 20 measured requests of the binding-count worker (the findings
      // 16.6 table values), rss shown for both workers.
      JsonNode memPub = cell.path("memory");
      assertEquals(
          "qwen3-8b-4bit".equals(caseName) && length == 2048,
          memPub.path("rss_peak_bytes_binding_count_suspect").asBoolean());
      JsonNode bc =
          read(base.resolve(
                  "java-controlled-"
                      + caseName
                      + "-"
                      + length
                      + "-headline-binding-count/report.json"))
              .path("worker");
      for (Map.Entry<String, String> field :
          Map.of(
                  "active_bytes", "active_bytes",
                  "peak_bytes", "peak_active_bytes",
                  "allocator_cached_bytes", "cached_bytes",
                  "rss_peak_bytes_binding_count", "rss_peak_bytes")
              .entrySet()) {
        List<Double> vals = new ArrayList<>();
        for (JsonNode q : bc.path("requests")) {
          if ("measured".equals(q.path("kind").asString())) {
            vals.add(q.path("memory").path(field.getValue()).asDouble());
          }
        }
        assertEquals(
            (long) median(vals),
            memPub.path(field.getKey()).asLong(),
            caseName + "/" + length + " " + field.getKey());
      }
      List<Double> rssTracing = new ArrayList<>();
      for (JsonNode q : tr.path("requests")) {
        if ("measured".equals(q.path("kind").asString())) {
          rssTracing.add(q.path("memory").path("rss_peak_bytes").asDouble());
        }
      }
      assertEquals(
          (long) median(rssTracing),
          memPub.path("rss_peak_bytes_tracing").asLong(),
          caseName + "/" + length + " rss_peak_bytes_tracing");
    }
  }

  // ---------------------------------------------------------------- wp3 FFM

  /**
   * The calibration reconstructed from the committed record: the per-call medians times the
   * workload call count (per-call p10 = p90 = median, the measurement-time ranges not being
   * archived), plus the recorded floor. This is exactly the calibration the {@code wp3_ffm.note}
   * bound is stated against.
   */
  private static ObjectNode reconstructedCalibration(JsonNode record) {
    ObjectNode cal = MAPPER.createObjectNode();
    cal.put("schema", "jmlx-ffm-cal-1-reconstructed-median-only");
    ObjectNode workload = cal.putObject("workload");
    workload.put("calls", CALIBRATION_CALLS);
    workload.put(
        "note",
        "reconstructed from evidence-index.json sources.wp3.ffm_calibration_record: per-call"
            + " medians x calls; p10 = p90 = median (the per-shape p10/p90 are not archived)");
    ObjectNode shapes = cal.putObject("shapes");
    for (Map.Entry<String, JsonNode> e : record.path("shape_median_ns_per_call").properties()) {
      double perSample = e.getValue().asDouble() * (double) CALIBRATION_CALLS;
      shapes
          .putObject(e.getKey())
          .put("median_ns", perSample)
          .put("p10_ns", perSample)
          .put("p90_ns", perSample);
    }
    cal.set("floor", record.path("floor"));
    return cal;
  }

  /**
   * The per-shape call counts a budget's bound needs: the per-position (single step window) maximum
   * for the step budget, the per-request maxima for the prefill and request-total budgets. The step
   * windows are uniform across positions and requests (the count-semantics check asserts it), so
   * the maximum is the recorded per-position count, taken over every window rather than divided.
   */
  private static final class ShapeCounts {
    final Map<String, Long> stepMaxPerPosition = new HashMap<>();
    final Map<String, Long> prefillMaxPerRequest = new HashMap<>();
    final Map<String, Long> totalMaxPerRequest = new HashMap<>();
  }

  private static ShapeCounts perShapeCounts(JsonNode worker) {
    ShapeCounts out = new ShapeCounts();
    JsonNode descriptors = worker.path("binding_descriptors");
    for (JsonNode req : worker.path("requests")) {
      if (!"measured".equals(req.path("kind").asString())) {
        continue;
      }
      Map<String, Long> reqPrefill = new HashMap<>();
      Map<String, Long> reqTotal = new HashMap<>();
      for (JsonNode window : req.path("binding_counts").path("phases")) {
        String phase = window.path("phase").asString();
        long position = window.path("position").asLong();
        for (Map.Entry<String, JsonNode> c : window.path("counts").properties()) {
          if (c.getValue().asLong() <= 0) {
            continue;
          }
          String key = FfmTransitionEstimate.shapeKey(descriptors.path(c.getKey()).asString(""));
          assertNotNull(key, "unclassifiable binding " + c.getKey());
          if (phase.startsWith("prefill-")) {
            reqPrefill.merge(key, c.getValue().asLong(), Long::sum);
          } else if (position > 0 && STEP_PHASES.contains(phase)) {
            out.stepMaxPerPosition.merge(key, c.getValue().asLong(), Math::max);
          }
          reqTotal.merge(key, c.getValue().asLong(), Long::sum);
        }
      }
      for (Map.Entry<String, Long> e : reqPrefill.entrySet()) {
        out.prefillMaxPerRequest.merge(e.getKey(), e.getValue(), Long::max);
      }
      for (Map.Entry<String, Long> e : reqTotal.entrySet()) {
        out.totalMaxPerRequest.merge(e.getKey(), e.getValue(), Long::max);
      }
    }
    return out;
  }

  /**
   * The per-call recording bound of a budget: 0.05 ns/call for the 1-decimal shapes, 0.005 for the
   * rest.
   */
  private static double shapeBound(Map<String, Long> perShapeCalls) {
    double bound = 0.0;
    for (Map.Entry<String, Long> e : perShapeCalls.entrySet()) {
      bound += (ONE_DECIMAL_SHAPES.contains(e.getKey()) ? 0.05 : 0.005) * e.getValue();
    }
    return bound;
  }

  @Test
  void wp3FfmEstimatesRecomputeFromRaw() throws URISyntaxException {
    JsonNode s = summary();
    JsonNode record = index().path("sources").path("wp3").path("ffm_calibration_record");
    ObjectNode cal = reconstructedCalibration(record);
    JsonNode ffm = s.path("wp3_ffm");
    JsonNode perCase = ffm.path("per_case");
    assertEquals(3, perCase.size());
    for (JsonNode pc : perCase) {
      String caseName = pc.path("case").asString();
      Set<String> countKeys = new HashSet<>();
      for (Map.Entry<String, JsonNode> e : pc.path("downcall_counts_per_request").properties()) {
        countKeys.add(e.getKey());
      }
      assertEquals(PHASE_NAMES.keySet(), countKeys, caseName + " downcall_counts_per_request keys");
      double stepMedian = Double.NaN;
      double prefillMedian = Double.NaN;
      double totalMedian = Double.NaN;
      for (int length : LENGTHS) {
        String rel =
            "raw/wp3/sessions/"
                + WP3_DIAG_SESSION
                + "/java-controlled-"
                + caseName
                + "-"
                + length
                + "-headline-binding-count/report.json";
        assertEquals(
            rel, ffm.path("reports").path(caseName).path(String.valueOf(length)).asString());
        JsonNode report = read(repoRoot().resolve(EVIDENCE).resolve(rel));
        ObjectNode est = FfmTransitionEstimate.estimate(report, cal);
        assertEquals(20, est.path("measured_requests").asInt());
        JsonNode totals = est.path("totals");
        assertSame(1.0, totals.path("coverage").asDouble(), caseName + "/" + length + " coverage");
        assertEquals(0, totals.path("unclassified_calls").asLong());
        assertEquals(0, est.path("uncovered").size(), caseName + "/" + length + " uncovered");
        double step = est.path("per_decode_step").path("median_ns").asDouble();
        double prefill = est.path("prefill").path("median_ns").asDouble();
        double total = est.path("request_total").path("median_ns").asDouble();
        // The per-request budget is length-independent: bit-identical medians at 32/512/2048.
        if (Double.isNaN(stepMedian)) {
          stepMedian = step;
          prefillMedian = prefill;
          totalMedian = total;
        } else {
          assertSame(stepMedian, step, caseName + " per_decode_step cross-length identity");
          assertSame(prefillMedian, prefill, caseName + " prefill cross-length identity");
          assertSame(totalMedian, total, caseName + " request_total cross-length identity");
        }
        // The published budget medians were computed from the unrounded calibration; the committed
        // record stores per-call medians, so the recomputation must land within the per-call
        // recording bound plus the published value's own display rounding.
        ShapeCounts counts = perShapeCounts(report.path("worker"));
        assertTrue(
            Math.abs(step - pc.path("per_decode_step_us").asDouble() * 1e3)
                <= shapeBound(counts.stepMaxPerPosition) + 5.0,
            caseName + "/" + length + " per_decode_step beyond the recording bound");
        assertTrue(
            Math.abs(prefill - pc.path("prefill_us").asDouble() * 1e3)
                <= shapeBound(counts.prefillMaxPerRequest) + 5.0,
            caseName + "/" + length + " prefill beyond the recording bound");
        assertTrue(
            Math.abs(total - pc.path("request_total_ms").asDouble() * 1e6)
                <= shapeBound(counts.totalMaxPerRequest) + 500.0,
            caseName + "/" + length + " request_total beyond the recording bound");
        // Count semantics: step_* is every individual position-step window's downcall total
        // (uniform across all 20 x 127 windows); prefill_* and request_release are the per-request
        // sums (uniform across the 20 measured requests); the total ties them to the raw windows.
        JsonNode worker = report.path("worker");
        Map<String, Set<Long>> stepWindowTotals = new HashMap<>();
        Map<String, List<Long>> reqPhaseTotals = new HashMap<>();
        long totalCalls = 0;
        for (JsonNode req : worker.path("requests")) {
          if (!"measured".equals(req.path("kind").asString())) {
            continue;
          }
          Map<String, Long> reqSums = new HashMap<>();
          for (JsonNode window : req.path("binding_counts").path("phases")) {
            long wsum = 0;
            for (JsonNode c : window.path("counts").values()) {
              wsum += c.asLong();
            }
            totalCalls += wsum;
            String phase = window.path("phase").asString();
            if (window.path("position").asLong() > 0 && STEP_PHASES.contains(phase)) {
              stepWindowTotals.computeIfAbsent(phase, k -> new HashSet<>()).add(wsum);
            } else {
              reqSums.merge(phase, wsum, Long::sum);
            }
          }
          for (Map.Entry<String, Long> e : reqSums.entrySet()) {
            reqPhaseTotals.computeIfAbsent(e.getKey(), k -> new ArrayList<>()).add(e.getValue());
          }
        }
        for (Map.Entry<String, String> pubKey : PHASE_NAMES.entrySet()) {
          long expected = pc.path("downcall_counts_per_request").path(pubKey.getKey()).asLong();
          if (pubKey.getKey().startsWith("step_")) {
            assertEquals(
                Set.of(expected),
                stepWindowTotals.get(pubKey.getValue()),
                caseName + "/" + length + " " + pubKey.getKey() + " per-window uniformity");
          } else {
            List<Long> sums = reqPhaseTotals.get(pubKey.getValue());
            assertNotNull(
                sums, caseName + "/" + length + " " + pubKey.getKey() + " per-request sums");
            assertEquals(20, sums.size());
            assertTrue(
                sums.stream().allMatch(v -> v == expected),
                caseName + "/" + length + " " + pubKey.getKey() + " per-request uniformity");
          }
        }
        assertEquals(
            pc.path("total_calls_20_measured").asLong(),
            totalCalls,
            caseName + "/" + length + " total_calls_20_measured");
      }
    }
  }

  // ---------------------------------------------------------------- wp4 VLM sessions

  @Test
  void wp4VlmSessionsRecomputeFromRaw() throws URISyntaxException {
    JsonNode s = summary();
    JsonNode wp4 = s.path("wp4_vlm_first_use");
    JsonNode sessions = wp4.path("sessions");
    assertEquals(2, sessions.size());
    for (JsonNode sess : sessions) {
      String name = sess.path("session").asString();
      assertTrue(WP4_NEW_TREES.containsKey(name), "unexpected wp4 session " + name);
      Path base = repoRoot().resolve(RAW + "/wp4/sessions/" + name);
      JsonNode d = read(base.resolve("session-report.json"));
      assertEquals(d.path("schema").asString(), sess.path("schema").asString());
      assertEquals(d.path("utc_start").asString(), sess.path("utc_start").asString());
      assertEquals(d.path("utc_end").asString(), sess.path("utc_end").asString());
      assertEquals(d.path("warmups").asInt(), sess.path("warmups").asInt());
      assertEquals(d.path("measured").asInt(), sess.path("measured_per_worker").asInt());
      assertEquals(
          d.path("precision").path("env").asString(),
          sess.path("precision").path("env").asString());
      assertEquals(
          d.path("precision").path("value").asString(),
          sess.path("precision").path("value").asString());
      assertEquals("0", sess.path("precision").path("value").asString(), "TF32 must be off");
      // Trees: the measured tree shas, cleanliness and the summary's transcription of them.
      assertEquals(2, d.path("trees").size());
      String oldWorkerSha = null;
      for (JsonNode t : d.path("trees")) {
        String treeName = t.path("name").asString();
        assertEquals(t.path("tree_sha").asString(), t.path("expected_tree_sha").asString());
        assertFalse(t.path("tracked_dirty").asBoolean(), treeName + " tracked_dirty");
        JsonNode pub = sess.path("trees").path(treeName);
        assertEquals(t.path("tree_sha").asString(), pub.path("tree_sha").asString());
        assertEquals(
            t.path("worker_source_sha256").asString(), pub.path("worker_source_sha256").asString());
        JsonNode pin = t.path("native_pin");
        JsonNode pubPin = pub.path("native_pin");
        for (Map.Entry<String, JsonNode> f : pin.properties()) {
          assertEquals(
              f.getValue().asString(),
              pubPin.path(f.getKey()).asString(),
              treeName + " native_pin." + f.getKey());
        }
        if ("old".equals(treeName)) {
          assertEquals(WP4_OLD_TREE, t.path("tree_sha").asString());
          oldWorkerSha = t.path("worker_source_sha256").asString();
        } else if ("new".equals(treeName)) {
          assertEquals(WP4_NEW_TREES.get(name), t.path("tree_sha").asString());
          assertEquals(
              oldWorkerSha,
              t.path("worker_source_sha256").asString(),
              "the worker source is identical on both pins");
        } else {
          fail("unexpected tree " + treeName);
        }
      }
      for (JsonNode wl : sess.path("workloads")) {
        String workload = wl.path("workload").asString();
        verifyWp4Workload(base, name, d, wl);
      }
    }
  }

  private static void verifyWp4Workload(
      Path sessionBase, String session, JsonNode report, JsonNode wl) throws URISyntaxException {
    String workload = wl.path("workload").asString();
    List<JsonNode> oldWorkers = new ArrayList<>();
    List<JsonNode> newWorkers = new ArrayList<>();
    for (int p = 0; p < 6; p++) {
      for (int ti = 0; ti < 2; ti++) {
        String tree = ti == 0 ? "old" : "new";
        List<JsonNode> target = ti == 0 ? oldWorkers : newWorkers;
        Path f = sessionBase.resolve(workload + "/" + tree + "-p" + p + "/report.json");
        assertTrue(Files.isRegularFile(f), "missing wp4 worker " + f);
        target.add(read(f));
      }
    }
    JsonNode oldPub = wl.path("per_tree").path("old");
    JsonNode newPub = wl.path("per_tree").path("new");
    verifyWp4Tree(oldWorkers, "old", oldPub, report);
    verifyWp4Tree(newWorkers, "new", newPub, report);
    // Paired warm medians and ratios, per pair.
    List<Double> oldFull = new ArrayList<>();
    List<Double> newFull = new ArrayList<>();
    List<Double> oldTtft = new ArrayList<>();
    List<Double> newTtft = new ArrayList<>();
    for (int p = 0; p < 6; p++) {
      oldFull.add(median(doubles(oldWorkers.get(p).path("measured_full_generation_s"))));
      newFull.add(median(doubles(newWorkers.get(p).path("measured_full_generation_s"))));
      oldTtft.add(median(doubles(oldWorkers.get(p).path("measured_ttft_s"))));
      newTtft.add(median(doubles(newWorkers.get(p).path("measured_ttft_s"))));
    }
    JsonNode paired = wl.path("paired");
    for (int p = 0; p < 6; p++) {
      assertSame(
          oldFull.get(p),
          paired.path("warm_full_generation_medians").path("old").get(p).asDouble(),
          "paired old median p" + p);
      assertSame(
          newFull.get(p),
          paired.path("warm_full_generation_medians").path("new").get(p).asDouble(),
          "paired new median p" + p);
    }
    List<Double> fullRatios = new ArrayList<>();
    List<Double> ttftRatios = new ArrayList<>();
    for (int p = 0; p < 6; p++) {
      fullRatios.add(newFull.get(p) / oldFull.get(p));
      ttftRatios.add(newTtft.get(p) / oldTtft.get(p));
    }
    for (int p = 0; p < 6; p++) {
      assertSame(
          fullRatios.get(p),
          paired.path("ratio_new_over_old").path("full_generation_s").get(p).asDouble(),
          "paired full ratio p" + p);
      assertSame(
          ttftRatios.get(p),
          paired.path("ratio_new_over_old").path("ttft_s").get(p).asDouble(),
          "paired ttft ratio p" + p);
    }
    assertSame(
        median(fullRatios),
        paired.path("ratio_new_over_old").path("full_generation_s_median").asDouble(),
        "paired full ratio median");
    assertSame(
        median(ttftRatios),
        paired.path("ratio_new_over_old").path("ttft_s_median").asDouble(),
        "paired ttft ratio median");
  }

  private static List<Double> doubles(JsonNode array) {
    List<Double> out = new ArrayList<>();
    for (JsonNode v : array) {
      out.add(v.asDouble());
    }
    return out;
  }

  private static void verifyWp4Tree(
      List<JsonNode> wks, String tree, JsonNode published, JsonNode sessionReport)
      throws URISyntaxException {
    for (JsonNode w : wks) {
      assertEquals(
          "ok", w.path("status").asString(), tree + " worker " + w.path("pair_index").asInt());
    }
    int measuredPerWorker = sessionReport.path("measured").asInt();
    List<Double> frTtft = new ArrayList<>();
    List<Double> frFull = new ArrayList<>();
    List<Double> load = new ArrayList<>();
    List<Double> warmTtft = new ArrayList<>();
    List<Double> warmFull = new ArrayList<>();
    for (JsonNode w : wks) {
      frTtft.add(w.path("first_request").path("ttft_s").asDouble());
      frFull.add(w.path("first_request").path("full_generation_s").asDouble());
      load.add(w.path("load").path("wall_s").asDouble());
      JsonNode mt = w.path("measured_ttft_s");
      JsonNode mf = w.path("measured_full_generation_s");
      assertEquals(measuredPerWorker, mt.size(), tree + " measured_ttft_s length");
      assertEquals(measuredPerWorker, mf.size(), tree + " measured_full_generation_s length");
      warmTtft.addAll(doubles(mt));
      warmFull.addAll(doubles(mf));
      // Correctness: the pinned reference behavior held for every measured request.
      for (Map.Entry<String, JsonNode> c : w.path("first_request").path("correct").properties()) {
        assertTrue(c.getValue().asBoolean(), tree + " first_request.correct." + c.getKey());
      }
      for (JsonNode m : w.path("measured_ids_match")) {
        assertTrue(m.asBoolean(), tree + " measured_ids_match");
      }
      JsonNode div = w.path("divergences");
      assertTrue(
          div.isMissingNode() || div.isNull() || (div.isArray() && div.size() == 0),
          tree + " must carry no divergences");
    }
    JsonNode fr = published.path("first_request");
    assertMinMedMax(frTtft, fr.path("ttft_s"), tree + " first ttft");
    assertMinMedMax(frFull, fr.path("full_generation_s"), tree + " first full_generation");
    assertMinMedMax(load, fr.path("load_wall_s"), tree + " first load wall");
    assertEquals(6 * measuredPerWorker, warmTtft.size());
    assertEquals(6 * measuredPerWorker, warmFull.size());
    JsonNode warm = published.path("warm");
    assertEquals(6 * measuredPerWorker, warm.path("n").asInt());
    BaselineStatistics.Quantiles qt = BaselineStatistics.quantiles(warmTtft);
    assertSame(qt.p10(), warm.path("ttft_s").get(0).asDouble(), tree + " warm ttft p10");
    assertSame(qt.median(), warm.path("ttft_s").get(1).asDouble(), tree + " warm ttft median");
    assertSame(qt.p90(), warm.path("ttft_s").get(2).asDouble(), tree + " warm ttft p90");
    BaselineStatistics.Quantiles qf = BaselineStatistics.quantiles(warmFull);
    assertSame(qf.p10(), warm.path("full_generation_s").get(0).asDouble(), tree + " warm full p10");
    assertSame(
        qf.median(), warm.path("full_generation_s").get(1).asDouble(), tree + " warm full median");
    assertSame(qf.p90(), warm.path("full_generation_s").get(2).asDouble(), tree + " warm full p90");
    for (String phase : new String[] {"after_load", "after_first", "after_final"}) {
      long minPeak = Long.MAX_VALUE;
      long maxPeak = Long.MIN_VALUE;
      long minActive = Long.MAX_VALUE;
      long maxActive = Long.MIN_VALUE;
      for (JsonNode w : wks) {
        JsonNode m = w.path("memory").path(phase);
        minPeak = Math.min(minPeak, m.path("peak_bytes").asLong());
        maxPeak = Math.max(maxPeak, m.path("peak_bytes").asLong());
        minActive = Math.min(minActive, m.path("active_bytes").asLong());
        maxActive = Math.max(maxActive, m.path("active_bytes").asLong());
      }
      JsonNode pm = published.path("memory").path(phase).path("peak_bytes");
      assertEquals(minPeak, pm.get(0).asLong(), phase + " peak min");
      assertEquals(maxPeak, pm.get(1).asLong(), phase + " peak max");
      JsonNode am = published.path("memory").path(phase).path("active_bytes");
      assertEquals(minActive, am.get(0).asLong(), phase + " active min");
      assertEquals(maxActive, am.get(1).asLong(), phase + " active max");
    }
  }

  // ---------------------------------------------------------------- identity pinning

  @Test
  void workerIdentitiesAndInputsArePinned() throws URISyntaxException, IOException {
    Path root = repoRoot();
    JsonNode idx = index();
    int wp3Workers = 0;
    for (Map.Entry<String, JsonNode> e : idx.path("committed_files").path("wp3").properties()) {
      String p = e.getKey();
      if (!p.endsWith("/report.json")) {
        continue;
      }
      wp3Workers++;
      JsonNode w = read(root.resolve(RAW + "/wp3/" + p)).path("worker");
      JsonNode id = w.path("identity");
      assertEquals(
          BaselineBenchmark.PINNED_MANIFEST_SHA256, id.path("manifest_sha256").asString(), p);
      assertEquals(WP3_CODE_SHA, id.path("code").path("sha").asString(), p);
      assertFalse(id.path("code").path("dirty").asBoolean(), p);
      assertEquals(WP3_BRANCH, id.path("code").path("branch").asString(), p);
      String[] parts = p.split("/");
      String expectedPrecision =
          parts[1].startsWith("qual-fc-tf320") ? "MLX_ENABLE_TF32=0" : "MLX_ENABLE_TF32=1";
      assertEquals(expectedPrecision, id.path("precision_env").asString(), p);
      if (parts[2].startsWith("java-")) {
        assertEquals(JAVA_VERSION, id.path("java").path("version").asString(), p);
        assertEquals(JAVA_VENDOR, id.path("java").path("vendor").asString(), p);
      }
    }
    assertEquals(WP3_WORKER_COUNT, wp3Workers);
    int wp4Workers = 0;
    for (Map.Entry<String, JsonNode> e : idx.path("committed_files").path("wp4").properties()) {
      String p = e.getKey();
      if (!p.endsWith("/report.json")) {
        continue;
      }
      wp4Workers++;
      JsonNode w = read(root.resolve(RAW + "/wp4/" + p));
      // The input block: schema -1 records it as "workload", schema -2 as "workload_inputs".
      JsonNode inputs =
          w.path("workload").isObject() ? w.path("workload") : w.path("workload_inputs");
      assertTrue(inputs.isObject() && !inputs.isEmpty(), "no workload input block in " + p);
      String manifestPath = inputs.path("manifest_path").asString();
      assertEquals(
          inputs.path("manifest_sha256").asString(),
          BaselineBenchmark.sha256File(root.resolve(manifestPath)),
          "manifest on disk vs " + p);
      for (JsonNode img : inputs.path("images")) {
        assertEquals(
            img.path("sha256").asString(),
            BaselineBenchmark.sha256File(root.resolve(img.path("path").asString())),
            "image on disk vs " + p);
      }
      assertEquals(JAVA_VERSION, w.path("java_version").asString(), p);
      assertEquals("0", w.path("precision").path("value").asString(), p);
    }
    assertEquals(WP4_WORKER_COUNT, wp4Workers);
  }
}
