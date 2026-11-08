package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Phase 8.2 WP1 gate: executes the Java worker's protocol and timeout paths in-process against a
 * test-built mini manifest pinning the tiny artifact (the 0.6b artifact is not git-committed, so no
 * real-artifact CI test exists — plan WP1 gate). Covers the ok/failed/timeout reports, the
 * claim/ready markers, the report refusal and success-downgrade contract, and the wall-clock
 * timeout at a request boundary, after the final request and through the grace backstop.
 */
@EnabledIfNativeAvailable
class BaselineWorkerProtocolTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @AfterEach
  void resetSeams() {
    BaselineWorker.resetTestSeams();
  }

  private record RunOutcome(int exitCode, String stderr) {}

  private static RunOutcome runCapturing(JsonNode job) {
    PrintStream old = System.err;
    ByteArrayOutputStream buffer = new ByteArrayOutputStream();
    int code;
    try {
      System.setErr(new PrintStream(buffer, true, StandardCharsets.UTF_8));
      code = BaselineWorker.run(job);
    } finally {
      System.setErr(old);
    }
    return new RunOutcome(code, buffer.toString(StandardCharsets.UTF_8));
  }

  private static JsonNode readJob(Path file) throws Exception {
    return MAPPER.readTree(file.toFile());
  }

  private static JsonNode readReport(Path outDir) throws Exception {
    return MAPPER.readTree(outDir.resolve("report.json").toFile());
  }

  private static BaselineWorkerTestSupport.Workspace setup(
      Path temp, int workerTimeoutS, int referencePairTimeoutS, int emitted, int measured)
      throws Exception {
    BaselineWorkerTestSupport.Workspace ws =
        BaselineWorkerTestSupport.build(
            temp, workerTimeoutS, referencePairTimeoutS, emitted, measured);
    BaselineWorker.setAnchorsForTest(ws.manifestPath(), ws.repoDir());
    return ws;
  }

  /** Writes a valid headline controlled job into a launcher-created {@code name} directory. */
  private static Path writeJob(
      Path temp,
      BaselineWorkerTestSupport.Workspace ws,
      String name,
      Consumer<ObjectNode> customize)
      throws Exception {
    ObjectNode job = MAPPER.createObjectNode();
    job.put("schema", BaselineWorker.JOB_SCHEMA);
    job.put("adapter", "controlled");
    job.put("case", BaselineWorkerTestSupport.CASE_NAME);
    job.put("prompt_length", 32);
    job.put("mode", "headline");
    job.put("timeout_s", ws.workerTimeoutS());
    job.put("precision_env", "1");
    ObjectNode requests = job.putObject("requests");
    requests.put("first", 1);
    requests.put("warmups", 0);
    requests.put("measured", ws.measuredRequests());
    job.put("artifact_dir", ws.artifactDir().toString());
    job.put("manifest_path", ws.manifestPath().toString());
    job.put("manifest_sha256", ws.manifestSha256());
    job.put("repo_dir", ws.repoDir().toString());
    job.put("out_dir", temp.resolve(name).toString());
    customize.accept(job);
    Path file = temp.resolve(name + ".job.json");
    Files.writeString(file, MAPPER.writeValueAsString(job), StandardCharsets.UTF_8);
    return file;
  }

  private static void createOutDir(Path temp, String name) throws Exception {
    Files.createDirectories(temp.resolve(name));
  }

  /**
   * Polls {@code condition} every 50 ms until it holds or 20 s elapse, asserting the message
   * otherwise — the bounded-poll replacement for fixed sleeps in ordering-sensitive tests.
   */
  private static void awaitTrue(BooleanSupplier condition, String message) throws Exception {
    long by = System.nanoTime() + 20_000_000_000L;
    while (!condition.getAsBoolean() && System.nanoTime() < by) {
      Thread.sleep(50);
    }
    assertTrue(condition.getAsBoolean(), message);
  }

  private static void assertControlledRequestRecord(JsonNode request) {
    assertEquals(8, request.path("emitted").asInt());
    assertEquals("controlled-max-tokens", request.path("finish_reason").asString(null));
    assertEquals(0, request.path("unused_lookahead_forwards").asInt());
    assertTrue(request.path("prime_excluded").asBoolean());
    assertTrue(request.path("prime_s").asDouble() >= 0.0);
    assertTrue(request.path("ttft_s").asDouble() > 0.0);
    assertTrue(request.path("prefill_s").asDouble() > 0.0);
    assertTrue(request.path("step_region_s").isNull());
    assertTrue(request.path("memory").isNull());
    assertFalse(request.path("io_affected").asBoolean());
    List<Integer> tokens = new ArrayList<>();
    request.path("tokens").forEach(t -> tokens.add(t.asInt()));
    assertEquals(List.of(0, 0, 0, 0, 0, 0, 0, 0), tokens, "all-zero weights select token 0");
    JsonNode steps = request.path("steps");
    assertEquals(7, steps.size(), "exactly emitted - 1 cached steps");
    for (int k = 0; k < steps.size(); k++) {
      assertEquals(33 + k, steps.get(k).path("position").asInt());
      assertTrue(steps.get(k).path("s").asDouble() > 0.0);
    }
    JsonNode eventS = request.path("event_s");
    assertEquals(8, eventS.size());
    for (int i = 1; i < eventS.size(); i++) {
      assertTrue(eventS.get(i).asDouble() >= eventS.get(i - 1).asDouble());
    }
    assertTrue(eventS.get(0).asDouble() > 0.0);
    assertEquals(eventS.get(7).asDouble(), request.path("full_generation_s").asDouble(), 0.0);
    assertTrue(request.path("streaming_decode_tps").asDouble() > 0.0);
  }

  @Test
  void controlledHeadlineWritesCompleteOkReport(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "out1");
    Path job = writeJob(temp, ws, "out1", ignored -> {});

    RunOutcome outcome = runCapturing(readJob(job));

    assertEquals(BaselineWorker.EXIT_OK, outcome.exitCode());
    Path outDir = temp.resolve("out1");
    JsonNode report = readReport(outDir);
    assertEquals(BaselineWorker.REPORT_SCHEMA, report.path("schema").asString(null));
    JsonNode worker = report.path("worker");
    assertEquals("java", worker.path("runtime").asString(null));
    assertEquals(ProcessHandle.current().pid(), worker.path("pid").asLong());
    assertEquals("controlled", worker.path("adapter").asString(null));
    assertEquals(BaselineWorkerTestSupport.CASE_NAME, worker.path("case").asString(null));
    assertEquals(32, worker.path("prompt_length").asInt());
    assertEquals("headline", worker.path("mode").asString(null));
    assertEquals("ok", worker.path("status").asString(null));
    assertTrue(worker.path("failure_reason").isNull());
    assertFalse(worker.path("utc_start").asString("").isEmpty());
    assertFalse(worker.path("utc_end").asString("").isEmpty());
    JsonNode identity = worker.path("identity");
    assertTrue(identity.path("code").isObject());
    assertTrue(identity.path("code").path("sha").isTextual());
    assertEquals(ws.manifestSha256(), identity.path("manifest_sha256").asString(null));
    assertEquals(ws.workerTimeoutS(), identity.path("run_policy").path("worker_timeout_s").asInt());
    assertEquals("MLX_ENABLE_TF32=1", identity.path("precision_env").asString(null));
    assertTrue(identity.path("java").path("version").isTextual());
    assertTrue(identity.path("device").isObject());
    assertTrue(identity.path("device").path("memory_size_bytes").isIntegralNumber());
    assertEquals("none", identity.path("wired_limit_policy").asString(null));
    assertEquals("primed-per-request", identity.path("cache_regime").asString(null));
    assertTrue(identity.path("io_affected").asBoolean() == false);
    assertTrue(identity.path("instrumented").asBoolean() == false);
    List<Integer> stopIds = new ArrayList<>();
    worker.path("stop_token_ids").forEach(t -> stopIds.add(t.asInt()));
    assertEquals(List.of(3), stopIds);
    assertTrue(worker.path("load_s").asDouble() > 0.0);
    assertEquals("materialized", worker.path("load_state").asString(null));
    JsonNode requests = worker.path("requests");
    assertEquals(2, requests.size());
    assertEquals(0, requests.get(0).path("index").asInt());
    assertEquals("first", requests.get(0).path("kind").asString(null));
    assertEquals(1, requests.get(1).path("index").asInt());
    assertEquals("measured", requests.get(1).path("kind").asString(null));
    assertControlledRequestRecord(requests.get(0));
    assertControlledRequestRecord(requests.get(1));
    // Protocol markers.
    JsonNode claim = MAPPER.readTree(outDir.resolve("worker-claim").toFile());
    assertFalse(claim.path("utc").asString("").isEmpty());
    assertEquals(ProcessHandle.current().pid(), claim.path("pid").asLong());
    JsonNode ready = MAPPER.readTree(outDir.resolve("worker-ready").toFile());
    assertEquals("controlled", ready.path("adapter").asString(null));
    assertEquals(BaselineWorkerTestSupport.CASE_NAME, ready.path("case").asString(null));
  }

  @Test
  void stockHeadlineEmitsSameIdsAndCarriesStockNulls(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "stock");
    createOutDir(temp, "controlled");
    Path stockJob = writeJob(temp, ws, "stock", job -> job.put("adapter", "stock"));
    Path controlledJob = writeJob(temp, ws, "controlled", ignored -> {});

    assertEquals(BaselineWorker.EXIT_OK, runCapturing(readJob(controlledJob)).exitCode());
    assertEquals(BaselineWorker.EXIT_OK, runCapturing(readJob(stockJob)).exitCode());

    JsonNode stock = readReport(temp.resolve("stock")).path("worker").path("requests").get(0);
    JsonNode controlled =
        readReport(temp.resolve("controlled")).path("worker").path("requests").get(0);
    List<Integer> stockTokens = new ArrayList<>();
    stock.path("tokens").forEach(t -> stockTokens.add(t.asInt()));
    List<Integer> controlledTokens = new ArrayList<>();
    controlled.path("tokens").forEach(t -> controlledTokens.add(t.asInt()));
    assertEquals(controlledTokens, stockTokens, "same model, same greedy policy");
    assertEquals(8, stockTokens.size());
    assertEquals("length", stock.path("finish_reason").asString(null));
    assertTrue(stock.path("prefill_s").isNull());
    assertTrue(stock.path("steps").isNull());
    assertEquals(0, stock.path("unused_lookahead_forwards").asInt());
    assertTrue(stock.path("step_region_s").isNull());
    assertTrue(
        stock.path("full_generation_s").asDouble() >= stock.path("event_s").get(7).asDouble(),
        "the full-generation boundary covers the finalization after the last token event");
    assertTrue(readReport(temp.resolve("stock")).path("worker").path("notes").size() > 0);
  }

  @Test
  void stockFailsWhenTokenizerEosExtendsTheFrozenStopSet(@TempDir Path temp) throws Exception {
    // The fixture gains a special-token eos the config files do not carry: the resolved stop
    // set becomes {3, 4} and must fail the assertion, so a drift between the artifact's stop
    // sources and the manifest frozen set never reaches the generation loop. The eos content
    // is a family control token, spelled with unicode escapes here: the literal control string
    // must never appear in the test sources.
    // CHECKSTYLE.SUPPRESS: AvoidEscapedUnicodeCharacters for +1 lines
    String eosContent = "\u003c|im_end|\u003e";
    String tokenizerJson =
        BaselineWorkerTestSupport.DEFAULT_TOKENIZER_JSON.replace(
            "\"added_tokens\":[]",
            "\"added_tokens\":[{\"id\":4,\"content\":\""
                + eosContent
                + "\",\"special\":true,\"lstrip\":false,\"rstrip\":false,"
                + "\"normalized\":false}]");
    BaselineWorkerTestSupport.Workspace ws =
        BaselineWorkerTestSupport.buildWithTokenizer(temp, 300, 60, 8, 1, tokenizerJson);
    BaselineWorker.setAnchorsForTest(ws.manifestPath(), ws.repoDir());
    createOutDir(temp, "stock");
    Path job = writeJob(temp, ws, "stock", j -> j.put("adapter", "stock"));

    RunOutcome outcome = runCapturing(readJob(job));

    assertFailedReportWithReason(temp.resolve("stock"), outcome.exitCode(), "resolved stop set");
  }

  @Test
  void fixedContinuationReportsPerStepChecks(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "fc");
    Path job =
        writeJob(
            temp,
            ws,
            "fc",
            j -> {
              j.put("mode", "fixed-continuation");
              j.put("prompt_length", 512);
              ObjectNode cont = j.putObject("fixed_continuation");
              cont.put("offset", 32);
              cont.put("count", 32);
            });

    assertEquals(BaselineWorker.EXIT_OK, runCapturing(readJob(job)).exitCode());

    JsonNode request = readReport(temp.resolve("fc")).path("worker").path("requests").get(0);
    assertEquals("fixed-continuation", request.path("finish_reason").asString(null));
    assertEquals(0, request.path("emitted").asInt());
    assertTrue(request.path("event_s").isArray());
    assertEquals(0, request.path("event_s").size());
    assertEquals(0, request.path("tokens").size());
    assertTrue(request.path("ttft_s").isNull());
    assertTrue(request.path("streaming_decode_tps").isNull());
    assertTrue(request.path("full_generation_s").isNull());
    assertTrue(request.path("step_region_s").asDouble() > 0.0);
    assertTrue(request.path("prefill_s").asDouble() > 0.0);
    JsonNode steps = request.path("steps");
    assertEquals(32, steps.size());
    for (int k = 0; k < steps.size(); k++) {
      assertEquals(513 + k, steps.get(k).path("position").asInt());
      assertEquals(0, steps.get(k).path("argmax_id").asInt(), "all-zero logits");
      assertEquals(0.0, steps.get(k).path("top2_margin").asDouble(), 0.0);
      assertTrue(steps.get(k).path("s").asDouble() > 0.0);
    }
  }

  @Test
  void deadlineExceededAtRequestBoundaryTimesOutWithCompleteRecords(@TempDir Path temp)
      throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 2, 60, 8, 4);
    createOutDir(temp, "slow");
    Path job = writeJob(temp, ws, "slow", ignored -> {});
    BaselineWorker.setRequestDelayMsForTest(500);

    RunOutcome outcome = runCapturing(readJob(job));

    assertEquals(BaselineWorker.EXIT_TIMEOUT, outcome.exitCode());
    JsonNode worker = readReport(temp.resolve("slow")).path("worker");
    assertEquals("timeout", worker.path("status").asString(null));
    assertEquals(
        "worker wall-clock timeout exceeded", worker.path("failure_reason").asString(null));
    JsonNode requests = worker.path("requests");
    // The tiny in-process load is well under the 2 s deadline, so at least the first request
    // and at least one measured request completed before the boundary break; every recorded
    // request is a complete record.
    assertTrue(requests.size() >= 2, "at least two complete request records: " + requests.size());
    for (JsonNode request : requests) {
      assertControlledRequestRecord(request);
    }
  }

  @Test
  void deadlineExceededAfterFinalRequestTimesOutReferencePair(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 1, 8, 1);
    createOutDir(temp, "rp");
    Path job =
        writeJob(
            temp,
            ws,
            "rp",
            j -> {
              j.put("mode", "reference-pair");
              j.put("timeout_s", ws.referencePairTimeoutS());
              j.withObject("requests").put("first", 0);
            });
    BaselineWorker.setRequestDelayMsForTest(1500);

    RunOutcome outcome = runCapturing(readJob(job));

    assertEquals(BaselineWorker.EXIT_TIMEOUT, outcome.exitCode());
    JsonNode worker = readReport(temp.resolve("rp")).path("worker");
    assertEquals("timeout", worker.path("status").asString(null));
    assertEquals(
        "worker wall-clock timeout exceeded", worker.path("failure_reason").asString(null));
    JsonNode requests = worker.path("requests");
    assertEquals(1, requests.size(), "the single reference-pair request finished and is recorded");
    assertEquals("measured", requests.get(0).path("kind").asString(null));
    assertEquals(
        0.0,
        requests.get(0).path("prime_s").asDouble(),
        0.0,
        "reference-pair workers prime nothing");
    assertControlledRequestRecord(requests.get(0));
  }

  @Test
  void backstopWritesTimeoutReportAndExitsWhenRequestIsWedged(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 1, 60, 8, 1);
    createOutDir(temp, "wedged");
    Path job = writeJob(temp, ws, "wedged", ignored -> {});
    BaselineWorker.setRequestDelayMsForTest(30_000);
    BaselineWorker.setWatchdogGraceMsForTest(500);
    CountDownLatch exitHooked = new CountDownLatch(1);
    AtomicInteger exitCode = new AtomicInteger(-1);
    BaselineWorker.setExitHookForTest(
        code -> {
          exitCode.set(code);
          exitHooked.countDown();
        });
    JsonNode jobNode = readJob(job);
    Thread worker =
        new Thread(
            () -> {
              try {
                BaselineWorker.run(jobNode);
              } catch (Exception e) {
                // The interrupt after the backstop exits the run through the unexpected handler.
              }
            },
            "protocol-test-worker");
    worker.start();
    assertTrue(exitHooked.await(20, TimeUnit.SECONDS), "the backstop exit hook fired");
    assertEquals(BaselineWorker.EXIT_TIMEOUT, exitCode.get());
    JsonNode workerReport = readReport(temp.resolve("wedged")).path("worker");
    assertEquals("timeout", workerReport.path("status").asString(null));
    assertEquals(
        "worker wall-clock timeout exceeded", workerReport.path("failure_reason").asString(null));
    worker.interrupt();
    worker.join(10_000);
    assertFalse(worker.isAlive());
  }

  @Test
  void staleWatchdogCannotActOnALaterRun(@TempDir Path temp) throws Exception {
    // Run A (tiny workspace a) is held on a one-shot latch in its request loop and its watchdog
    // fires; run B (workspace b) starts in the same process while A's fired watchdog is in its
    // grace sleep. Without the run-token disarm, A's grace action wakes inside B's window,
    // overwrites B's status, writes a timeout report into B's directory (so B's own report
    // write is refused) and fires the exit hook; with it, the grace action sees B's token and
    // drops, and B completes normally with its own report. A's hold is released only after B
    // has finished, so none of A's post-loop failure handling can race B's report write — every
    // A-vs-B ordering step is held by the latch or observed by a bounded poll, not encoded in
    // a sleep margin.
    BaselineWorkerTestSupport.Workspace wsA =
        BaselineWorkerTestSupport.build(temp.resolve("a"), 2, 60, 8, 0);
    createOutDir(temp.resolve("a"), "outA");
    Path jobA = writeJob(temp.resolve("a"), wsA, "outA", ignored -> {});

    BaselineWorker.setWatchdogGraceMsForTest(4000);
    CountDownLatch holdA = new CountDownLatch(1);
    BaselineWorker.setRequestHoldForTest(holdA);
    AtomicInteger hookFires = new AtomicInteger();
    BaselineWorker.setExitHookForTest(hookFires::addAndGet);
    // Anchors for run A; switched to B's workspace once A is held.
    BaselineWorker.setAnchorsForTest(wsA.manifestPath(), wsA.repoDir());

    AtomicReference<Integer> codeA = new AtomicReference<>();
    Thread runA =
        new Thread(
            () -> {
              try {
                codeA.set(BaselineWorker.run(readJob(jobA)));
              } catch (Exception e) {
                // The run completes on its own; any exception here is a test error.
                codeA.set(-1);
              }
            },
            "protocol-test-stale-watchdog-a");
    runA.start();

    // Hold A deterministically, in order: worker-ready exists (the deadline is anchored there),
    // the latch has been consumed (A is in its request loop, so run B cannot consume it) and
    // the watchdog has fired (the deadline is ready + 2 s). The fire must precede B's start,
    // or the test would pass without exercising the fired-watchdog disarm.
    awaitTrue(
        () -> Files.exists(temp.resolve("a/outA/worker-ready")), "run A reached worker-ready");
    awaitTrue(BaselineWorker::requestHoldConsumedForTest, "run A consumed the hold latch");
    awaitTrue(BaselineWorker::hasFiredForTest, "run A's watchdog fired");

    // Start run B in the same process, while A's fired watchdog is in its 4 s grace sleep.
    BaselineWorkerTestSupport.Workspace wsB =
        BaselineWorkerTestSupport.build(temp.resolve("b"), 6, 60, 8, 1);
    createOutDir(temp.resolve("b"), "outB");
    BaselineWorker.setAnchorsForTest(wsB.manifestPath(), wsB.repoDir());
    Path jobB = writeJob(temp.resolve("b"), wsB, "outB", ignored -> {});
    AtomicReference<Integer> codeB = new AtomicReference<>();
    Thread runB =
        new Thread(
            () -> {
              try {
                codeB.set(BaselineWorker.run(readJob(jobB)));
              } catch (Exception e) {
                codeB.set(-1);
              }
            },
            "protocol-test-stale-watchdog-b");
    runB.start();

    runB.join(30_000);
    assertFalse(runB.isAlive(), "run B finished");
    assertEquals(
        BaselineWorker.EXIT_OK,
        codeB.get(),
        "the later run must not be timed out, report-refused or exited by a stale watchdog");
    JsonNode workerB = readReport(temp.resolve("b/outB")).path("worker");
    assertEquals("ok", workerB.path("status").asString(null));
    assertTrue(workerB.path("failure_reason").isNull());
    assertEquals(
        wsB.manifestSha256(),
        workerB.path("identity").path("manifest_sha256").asString(null),
        "B's report is B's own, not a stale timeout report written for run A");
    assertEquals(0, hookFires.get(), "no backstop exit may fire for a healthy later run");

    // Release A only now: its resumed request is guaranteed to run after B's report is on disk.
    holdA.countDown();

    // Run A finishes on its own once released; it must not hang. Run A's own outcome is an
    // in-process artifact, not a protocol result: when its held request resumes, run B has
    // already replaced and closed the process-static model scope (B's execute completed), so
    // A's MLX access fails and A's report write goes to the (now static) B directory, where it
    // is refused because B's report already exists. The worker's design is one run per process
    // (production forks a fresh JVM per cell), so A is expected to leave no report of its own;
    // the test's gate is that the later run B stays clean.
    runA.join(30_000);
    assertFalse(runA.isAlive(), "run A finished");
    assertFalse(
        Files.exists(temp.resolve("a/outA/report.json")),
        "run A's resumed request fails on the closed scope and leaves no report");
  }

  @Test
  void claimRefusalWritesNothingAndFails(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    Path outDir = temp.resolve("claimed");
    Files.createDirectories(outDir);
    Files.writeString(
        outDir.resolve("worker-claim"),
        """
        {"utc":"2026-11-07T00:00:00Z","pid":99999}
        """,
        StandardCharsets.UTF_8);
    Path job = writeJob(temp, ws, "claimed", ignored -> {});

    RunOutcome outcome = runCapturing(readJob(job));

    assertEquals(BaselineWorker.EXIT_FAILED, outcome.exitCode());
    assertTrue(outcome.stderr().contains("already claimed"), outcome.stderr());
    assertFalse(Files.exists(outDir.resolve("report.json")), "a refused run writes no report");
    assertEquals(
        List.of("worker-claim"),
        Files.list(outDir).map(p -> p.getFileName().toString()).sorted().toList());
  }

  @Test
  void missingOutDirIsRefusedBeforeAnyClaim(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    Path job = writeJob(temp, ws, "absent", ignored -> {});

    RunOutcome outcome = runCapturing(readJob(job));

    assertEquals(BaselineWorker.EXIT_FAILED, outcome.exitCode());
    assertTrue(
        outcome.stderr().contains("out_dir must be created by the launcher"), outcome.stderr());
    assertFalse(Files.exists(temp.resolve("absent")), "the worker never creates out_dir");
  }

  @Test
  void fileInWayOutDirIsRefused(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    Path blocker = temp.resolve("blocker");
    Files.writeString(blocker, "not a directory", StandardCharsets.UTF_8);
    Path job = writeJob(temp, ws, "blocker", ignored -> {});

    RunOutcome outcome = runCapturing(readJob(job));

    assertEquals(BaselineWorker.EXIT_FAILED, outcome.exitCode());
    assertTrue(
        outcome.stderr().contains("out_dir must be created by the launcher"), outcome.stderr());
    assertEquals("not a directory", Files.readString(blocker, StandardCharsets.UTF_8));
  }

  @Test
  void existingReportIsPreservedAndTheRunFails(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    Path outDir = temp.resolve("reported");
    Files.createDirectories(outDir);
    String existing = "{\"schema\":\"jmlx-decode-baseline-1\",\"sentinel\":true}";
    Files.writeString(outDir.resolve("report.json"), existing, StandardCharsets.UTF_8);
    Path job = writeJob(temp, ws, "reported", ignored -> {});

    RunOutcome outcome = runCapturing(readJob(job));

    assertEquals(BaselineWorker.EXIT_FAILED, outcome.exitCode());
    assertTrue(outcome.stderr().contains("already exists at"), outcome.stderr());
    assertEquals(existing, Files.readString(outDir.resolve("report.json"), StandardCharsets.UTF_8));
  }

  private static void assertFailedReportWithReason(
      Path outDir, int exitCode, String expectedReasonFragment) throws Exception {
    assertEquals(BaselineWorker.EXIT_FAILED, exitCode);
    assertTrue(
        Files.exists(outDir.resolve("report.json")),
        "a worker that claimed the directory writes its failed report");
    JsonNode worker = readReport(outDir).path("worker");
    assertEquals("failed", worker.path("status").asString(null));
    String reason = worker.path("failure_reason").asString(null);
    assertNotNull(reason);
    assertTrue(reason.contains(expectedReasonFragment), reason);
  }

  @Test
  void rejectsWrongJobSchema(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "schema");
    Path job = writeJob(temp, ws, "schema", j -> j.put("schema", "jmlx-decode-baseline-job-9"));
    RunOutcome outcome = runCapturing(readJob(job));
    assertFailedReportWithReason(temp.resolve("schema"), outcome.exitCode(), "job schema must be");
  }

  @Test
  void rejectsManifestShaMismatch(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "sha");
    Path job = writeJob(temp, ws, "sha", j -> j.put("manifest_sha256", "0".repeat(64)));
    RunOutcome outcome = runCapturing(readJob(job));
    assertFailedReportWithReason(
        temp.resolve("sha"), outcome.exitCode(), "manifest SHA-256 mismatch");
  }

  @Test
  void rejectsNonCanonicalManifestCopy(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "copy");
    Path copy = temp.resolve("copy-of-manifest.json");
    Files.copy(ws.manifestPath(), copy);
    Path job = writeJob(temp, ws, "copy", j -> j.put("manifest_path", copy.toString()));
    RunOutcome outcome = runCapturing(readJob(job));
    assertFailedReportWithReason(
        temp.resolve("copy"), outcome.exitCode(), "canonical frozen manifest");
  }

  @Test
  void rejectsRequestsThatDifferFromRunPolicy(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "requests");
    Path job = writeJob(temp, ws, "requests", j -> j.withObject("requests").put("first", 2));
    RunOutcome outcome = runCapturing(readJob(job));
    assertFailedReportWithReason(
        temp.resolve("requests"), outcome.exitCode(), "requests.first differs");
  }

  @Test
  void rejectsTimeoutThatDiffersFromRunPolicy(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "timeout");
    Path job = writeJob(temp, ws, "timeout", j -> j.put("timeout_s", ws.workerTimeoutS() + 1));
    RunOutcome outcome = runCapturing(readJob(job));
    assertFailedReportWithReason(temp.resolve("timeout"), outcome.exitCode(), "timeout_s differs");
  }

  @Test
  void rejectsExtraUnpinnedSafetensorsFile(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "extra");
    Files.writeString(
        ws.artifactDir().resolve("extra.safetensors"), "unpinned", StandardCharsets.UTF_8);
    Path job = writeJob(temp, ws, "extra", ignored -> {});
    RunOutcome outcome = runCapturing(readJob(job));
    assertFailedReportWithReason(
        temp.resolve("extra"), outcome.exitCode(), "safetensors set differs");
  }

  @Test
  void rejectsFixedContinuationOutsideManifestRule(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "fc");
    Path job =
        writeJob(
            temp,
            ws,
            "fc",
            j -> {
              j.put("mode", "fixed-continuation");
              j.put("prompt_length", 512);
              ObjectNode cont = j.putObject("fixed_continuation");
              cont.put("offset", 33);
              cont.put("count", 32);
            });
    RunOutcome outcome = runCapturing(readJob(job));
    assertFailedReportWithReason(
        temp.resolve("fc"), outcome.exitCode(), "fixed_continuation must be prefix512[32:64]");
  }

  @Test
  void rejectsPrecisionEnvMismatchWithProcessEnvironment(@TempDir Path temp) throws Exception {
    BaselineWorkerTestSupport.Workspace ws = setup(temp, 300, 60, 8, 1);
    createOutDir(temp, "precision");
    Path job = writeJob(temp, ws, "precision", j -> j.put("precision_env", "0"));
    RunOutcome outcome = runCapturing(readJob(job));
    assertFailedReportWithReason(temp.resolve("precision"), outcome.exitCode(), "MLX_ENABLE_TF32");
  }
}
