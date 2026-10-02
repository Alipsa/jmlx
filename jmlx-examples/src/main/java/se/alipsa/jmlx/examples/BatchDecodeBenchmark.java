package se.alipsa.jmlx.examples;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.models.BatchGenerationScheduler;
import se.alipsa.jmlx.models.BatchRequestHandle;
import se.alipsa.jmlx.models.BatchSchedulerConfig;
import se.alipsa.jmlx.models.CancellationToken;
import se.alipsa.jmlx.models.DecoderModel;
import se.alipsa.jmlx.models.GenerationConfig;
import se.alipsa.jmlx.models.GenerationRequest;
import se.alipsa.jmlx.models.TextGenerationModels;
import tools.jackson.databind.ObjectMapper;

/**
 * Opt-in direct-versus-batched throughput and latency benchmark on a local checkpoint.
 *
 * <p>The same heterogeneous requests run once sequentially through the direct path and once
 * together through a {@link BatchGenerationScheduler}. MLX is used by one thread at a time: the
 * direct run finishes and closes its scope before the scheduler starts, and memory counters are
 * read only while no other thread is using MLX. Nothing is downloaded.
 */
public final class BatchDecodeBenchmark {
  private BatchDecodeBenchmark() {}

  /**
   * Runs with {@code checkpoint-dir output-prefix [prompts] [new-tokens-csv] [samples] [warmups]
   * [max-batch]}, where {@code prompts} is semicolon-separated token-ID lists.
   *
   * @param args local checkpoint, output prefix, and optional benchmark parameters
   * @throws Exception if loading, generation, or report writing fails
   */
  public static void main(String[] args) throws Exception {
    if (args.length < 2 || args.length > 7) {
      throw new IllegalArgumentException(
          "usage: BatchDecodeBenchmark checkpoint-dir output-prefix [prompts 1,2;1,2,3]"
              + " [new-tokens-csv 8,16] [samples] [warmups] [max-batch]");
    }
    Path checkpoint = Path.of(args[0]);
    final Path output = Path.of(args[1]);
    int[][] prompts =
        Arrays.stream((args.length > 2 ? args[2] : "1,7,42,3,19,5;1,9,4;1,2,3,4,5").split(";"))
            .map(BatchDecodeBenchmark::ids)
            .toArray(int[][]::new);
    int[] newTokens = ids(args.length > 3 ? args[3] : "24,8,16");
    int samples = args.length > 4 ? Integer.parseInt(args[4]) : 5;
    int warmups = args.length > 5 ? Integer.parseInt(args[5]) : 2;
    int maxBatch = args.length > 6 ? Integer.parseInt(args[6]) : prompts.length;
    if (prompts.length != newTokens.length || samples < 1 || warmups < 0 || maxBatch < 1) {
      throw new IllegalArgumentException("one new-token count per prompt; samples >= 1");
    }
    List<GenerationRequest> requests = new ArrayList<>();
    for (int i = 0; i < prompts.length; i++) {
      requests.add(
          new GenerationRequest(
              prompts[i],
              GenerationConfig.greedyDefaults(newTokens[i], Set.of()),
              CancellationToken.NONE));
    }
    final long totalTokens = Arrays.stream(newTokens).asLongStream().sum();

    List<Run> direct = new ArrayList<>();
    List<BatchedRun> batched = new ArrayList<>();
    for (int run = -warmups; run < samples; run++) {
      // Alternate which side goes first so warm-up, thermal and allocator-order effects are not
      // always charged to the same one.
      boolean directFirst = (run & 1) == 0;
      Run d = directFirst ? runDirect(checkpoint, requests) : null;
      BatchedRun b = runBatched(checkpoint, requests, maxBatch);
      if (d == null) {
        d = runDirect(checkpoint, requests);
      }
      if (run >= 0) {
        direct.add(d);
        batched.add(b);
      }
    }

    Map<String, Object> report = new LinkedHashMap<>();
    report.put("schema", "jmlx-batch-decode-benchmark-1");
    report.put("checkpoint", checkpoint.toAbsolutePath().toString());
    report.put("device", System.getenv().getOrDefault("JMLX_BENCH_DEVICE", "unspecified"));
    report.put("java", System.getProperty("java.version"));
    report.put("requests", prompts.length);
    report.put("max_batch_size", maxBatch);
    report.put("queue_depth_at_submit", prompts.length);
    report.put("prompt_lengths", Arrays.stream(prompts).mapToInt(p -> p.length).toArray());
    report.put("new_tokens", newTokens);
    report.put("total_generated_tokens", totalTokens);
    report.put("warmups", warmups);
    report.put("samples", samples);
    report.put("direct", summary(direct, totalTokens));
    report.put("batched", summary(batched.stream().map(BatchedRun::metrics).toList(), totalTokens));
    // The cohorts that actually ran, per sample: the worker may form a cohort of one before the
    // rest of the queue arrives, so without this a speedup cannot be attributed to a batch shape.
    report.put("batched_cohort_sizes", batched.stream().map(BatchedRun::cohortSizes).toList());
    report.put(
        "speedup_median",
        median(direct.stream().map(Run::wallNanos).toList())
            / (double) median(batched.stream().map(r -> r.metrics().wallNanos()).toList()));
    Files.createDirectories(output.toAbsolutePath().getParent());
    new ObjectMapper()
        .writeValue(output.resolveSibling(output.getFileName() + ".json").toFile(), report);
    Files.writeString(
        output.resolveSibling(output.getFileName() + ".md"),
        "# Batch decode benchmark\n\nrequests="
            + prompts.length
            + " tokens="
            + totalTokens
            + "\n\ndirect median tokens/s: "
            + ((Map<?, ?>) report.get("direct")).get("tokens_per_second_median")
            + "\n\nbatched median tokens/s: "
            + ((Map<?, ?>) report.get("batched")).get("tokens_per_second_median")
            + "\n\nspeedup (median wall): "
            + report.get("speedup_median")
            + "\n");
  }

  /** One timed run: wall time, per-request latencies since the run began, and native memory. */
  private record Run(
      long wallNanos,
      long[] firstTokenNanos,
      long[] totalNanos,
      long peakBytes,
      long activeAfterBytes) {}

  /** A batched run plus the cohort sizes it actually ran, for attribution in the report. */
  private record BatchedRun(Run metrics, List<Integer> cohortSizes) {}

  private static Run runDirect(Path checkpoint, List<GenerationRequest> requests) throws Exception {
    long before = MLXMemory.activeBytes();
    MLXMemory.resetPeak();
    long[] first = new long[requests.size()];
    long[] total = new long[requests.size()];
    long begin = System.nanoTime();
    try (MLXScope scope = new MLXScope()) {
      DecoderModel model = (DecoderModel) TextGenerationModels.load(scope, checkpoint);
      begin = System.nanoTime(); // exclude load: only generation is compared
      for (int i = 0; i < requests.size(); i++) {
        final int index = i;
        final long start = begin;
        model.generate(
            requests.get(i),
            event -> {
              if (event.tokenId() != null && first[index] == 0) {
                first[index] = System.nanoTime() - start;
              }
            });
        total[i] = System.nanoTime() - begin;
      }
    }
    long wall = total[total.length - 1];
    return new Run(wall, first, total, MLXMemory.peakBytes(), MLXMemory.activeBytes() - before);
  }

  private static BatchedRun runBatched(
      Path checkpoint, List<GenerationRequest> requests, int maxBatch) throws Exception {
    long before = MLXMemory.activeBytes();
    MLXMemory.resetPeak();
    int n = requests.size();
    long[] first = new long[n];
    long[] total = new long[n];
    AtomicLong begin = new AtomicLong();
    BatchSchedulerConfig config = new BatchSchedulerConfig(maxBatch, Math.max(n, 1), 8192, 4096);
    // start() returns once the worker has loaded the model, so load time is outside the clock.
    // Requests are submitted back to back; how they group into cohorts is the scheduler's call,
    // and the first one may start alone before the rest are queued. The cohorts that actually ran
    // are returned with the run (recorded as batched_cohort_sizes in the report) so a speedup can
    // be attributed to a batch shape.
    BatchGenerationScheduler.ModelFactory factory =
        scope -> TextGenerationModels.load(scope, checkpoint);
    List<Integer> cohortSizes;
    try (BatchGenerationScheduler scheduler = BatchGenerationScheduler.start(config, factory)) {
      List<CompletableFuture<?>> done = new ArrayList<>();
      begin.set(System.nanoTime());
      for (int i = 0; i < n; i++) {
        final int index = i;
        BatchRequestHandle handle =
            scheduler.submit(
                requests.get(i),
                event -> {
                  if (event.tokenId() != null && first[index] == 0) {
                    first[index] = System.nanoTime() - begin.get();
                  }
                });
        done.add(
            handle
                .stage()
                .thenAccept(result -> total[index] = System.nanoTime() - begin.get())
                .toCompletableFuture());
      }
      CompletableFuture.allOf(done.toArray(CompletableFuture[]::new)).join();
      cohortSizes = scheduler.cohortSizes();
    }
    // The scheduler is closed: its worker has exited, so reading the counters is single-threaded.
    long wall = Arrays.stream(total).max().orElse(0);
    return new BatchedRun(
        new Run(wall, first, total, MLXMemory.peakBytes(), MLXMemory.activeBytes() - before),
        cohortSizes);
  }

  private static Map<String, Object> summary(List<Run> runs, long totalTokens) {
    Map<String, Object> out = new LinkedHashMap<>();
    List<Long> wall = runs.stream().map(Run::wallNanos).toList();
    out.put("wall_ns", List.copyOf(wall));
    out.put("wall_ns_median", median(wall));
    out.put("tokens_per_second_median", totalTokens * 1_000_000_000.0 / median(wall));
    out.put("first_token_ns_per_request", perRequest(runs, Run::firstTokenNanos));
    out.put("total_ns_per_request", perRequest(runs, Run::totalNanos));
    out.put("peak_active_bytes", runs.stream().map(Run::peakBytes).toList());
    out.put("active_bytes_leaked_after_run", runs.stream().map(Run::activeAfterBytes).toList());
    return out;
  }

  /** Median across samples of each request's latency. */
  private static List<Long> perRequest(List<Run> runs, java.util.function.Function<Run, long[]> f) {
    int n = f.apply(runs.getFirst()).length;
    List<Long> medians = new ArrayList<>();
    for (int i = 0; i < n; i++) {
      final int index = i;
      medians.add(median(runs.stream().map(r -> f.apply(r)[index]).toList()));
    }
    return medians;
  }

  private static long median(List<Long> data) {
    List<Long> sorted = data.stream().sorted().toList();
    return sorted.get(sorted.size() / 2);
  }

  private static int[] ids(String csv) {
    return Arrays.stream(csv.split(",")).mapToInt(Integer::parseInt).toArray();
  }
}
