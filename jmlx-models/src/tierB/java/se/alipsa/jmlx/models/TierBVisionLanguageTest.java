package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXMemory;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import se.alipsa.jmlx.tokenizer.ChatTemplateOptions;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import se.alipsa.jmlx.vision.ImageDecoder;
import se.alipsa.jmlx.vision.ImageTensor;
import se.alipsa.jmlx.vision.RgbImage;
import se.alipsa.jmlx.vision.SmolVlmImageProcessor;
import se.alipsa.jmlx.vision.SmolVlmProcessorConfig;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Opt-in pinned real image/text references, stable greedy IDs and gated measured memory.
 *
 * <p>Each case's recorded {@code pixel_values_sha256} is re-computed from the pinned image through
 * {@code jmlx-vision}'s processor and asserted (see {@link #pixelValuesSha256}), so preprocessing
 * on the real artifact is verified byte-for-byte, not only indirectly through the logits.
 *
 * <p>The memory gates close a review finding: the previously recorded Tier-B evidence (RSS 487,312
 * KiB) sampled below this checkpoint's own 513 MB weight file because {@code ps -o rss} excludes
 * the MLX/Metal allocations the process holds, so it could never bound real usage and a memory
 * regression would have surfaced only as a runner OOM. Both bounds below are upper-bound gates on
 * the two measured figures, sized from the manifest's recorded local_evidence
 * (tools/tier-b/smolvlm-256m-instruct.json: native peak 5,737,416,464 bytes on macOS 27.0.1, TF32
 * off, pinned mlx-metal 0.31.2 / mlx-c fba4470) so a regression that still fits in the
 * runner's RAM fails this suite explicitly.
 */
@EnabledIfNativeAvailable
class TierBVisionLanguageTest {
  // Recorded in the manifest's local_evidence (macOS 27.0.1, TF32 off, pinned mlx-metal 0.31.2
  // / mlx-c fba4470): the suite's MLX allocator peak is 5,737,416,464 bytes. The bound adds 500
  // MiB, well above cross-host allocator variance (the peak is mlx's own byte accounting, not OS
  // memory) yet below one extra float32 copy of this checkpoint's weights (about 1 GB). It is
  // a regression detector, not a guarantee the macos-26-xlarge tier survives a large
  // regression: the figure is read only after generation finishes, so a regression big enough
  // to push the process past the tier's RAM fails the job as an OOM, not as this assertion.
  private static final long NATIVE_PEAK_BOUND_BYTES = 5_737_416_464L + 500L * 1024 * 1024;
  // The same local run's sampled process-footprint peak is 7,129,368,904 bytes (the 5.7 GB MLX
  // pool plus the JVM). The bound is 8 GiB: about 20% headroom for JVM/OS variance on the hosted
  // runner, still far below the 14 GB of the macos-26-xlarge tier the suite runs on. The ~1.36
  // GiB of headroom is deliberately not tight enough to catch one extra float32 weight copy
  // (about 1 GB plus rounding); catching that regression is the job of the tighter
  // NATIVE_PEAK_BOUND_BYTES gate above.
  private static final long FOOTPRINT_BOUND_BYTES = 8L * 1024 * 1024 * 1024;

  @Test
  void expandedPromptCachedLogitsAndStableGreedyIds() throws Exception {
    TierBReferences reference = new TierBReferences("image-text-generation");
    final HfTokenizer tokenizer = HfTokenizer.fromDirectory(reference.directory);
    Path repository = Path.of(System.getProperty("jmlx.repository.root"));
    var logitsSpec = reference.reference.required("logits_file");
    Path logitsFile = repository.resolve(logitsSpec.required("path").asString());
    assertEquals(
        logitsSpec.required("sha256").asString(),
        HexFormat.of()
            .formatHex(
                MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(logitsFile))));
    JsonNode logitsGolden = JsonFiles.read(logitsFile).required("cases");
    assertEquals(reference.reference.required("cases").size(), logitsGolden.size());
    List<RgbImage> images = new ArrayList<>();
    for (JsonNode spec : reference.manifest.required("images")) {
      Path file = repository.resolve(spec.required("path").asString());
      assertEquals(
          spec.required("sha256").asString(),
          HexFormat.of()
              .formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file))));
      assertTrue(!spec.required("license").asString().isBlank());
      images.add(ImageDecoder.decode(file));
    }
    // The checkpoint's preprocessor settings (hash-verified by TierBReferences above) drive the
    // byte-for-byte pixel check; the model under test parses the same file at load time.
    SmolVlmImageProcessor processor =
        new SmolVlmImageProcessor(
            SmolVlmProcessorConfig.fromJson(
                Files.readAllBytes(reference.directory.resolve("preprocessor_config.json"))));
    try (FootprintSampler footprint = new FootprintSampler();
        MLXScope root = new MLXScope()) {
      SmolVlmModel model = (SmolVlmModel) TextGenerationModels.load(root, reference.directory);
      reference.model(model.metadata());
      int caseIndex = 0;
      for (JsonNode input : reference.reference.required("cases")) {
        JsonNode expectedLogits = logitsGolden.get(caseIndex++);
        assertEquals(input.required("target_ids").size(), expectedLogits.size());
        List<Map<String, Object>> messages = new ArrayList<>();
        ObjectMapper mapper = new ObjectMapper();
        for (JsonNode message : input.required("messages")) {
          @SuppressWarnings("unchecked")
          Map<String, Object> converted = mapper.convertValue(message, Map.class);
          messages.add(converted);
        }
        List<RgbImage> selected = new ArrayList<>();
        for (JsonNode index : input.required("image_indices")) {
          selected.add(images.get(index.asInt()));
        }
        assertEquals(
            input.required("pixel_values_sha256").asString(),
            pixelValuesSha256(processor, selected),
            "processed pixel values drifted from the reference bytes");
        assertEquals(
            input.required("rendered_prompt").asString(),
            tokenizer.renderChat(messages, ChatTemplateOptions.defaults(true)));
        GenerationRequest request =
            GenerationRequest.chat(
                    tokenizer,
                    messages,
                    ChatTemplateOptions.defaults(true),
                    GenerationConfig.greedyDefaults(
                        input.required("max_new_tokens").asInt(),
                        Set.of(tokenizer.eosTokenId().orElseThrow())),
                    CancellationToken.NONE)
                .withImages(selected);
        assertEquals(
            TierBReferences.ids(input.required("unexpanded_ids")),
            Arrays.stream(request.promptTokenIds()).boxed().toList());
        SmolVlmPromptPlan plan = model.plan(request.promptTokenIds(), request.images());
        assertEquals(
            TierBReferences.ids(input.required("expanded_ids")),
            Arrays.stream(plan.expandedIds()).boxed().toList());
        try (MLXScope state = root.newChild()) {
          List<KVCache> caches = new ArrayList<>();
          for (int i = 0; i < model.decoderForTest().config().numHiddenLayers(); i++) {
            caches.add(new KVCache(state));
          }
          for (int step = 0; step < expectedLogits.size(); step++) {
            try (MLXScope activation = state.newChild()) {
              MLXArray logits;
              if (step == 0) {
                MLXArray features = model.features(request, activation);
                logits =
                    model
                        .decoderForTest()
                        .stepLogits(
                            MLX.array(
                                activation,
                                plan.expandedIds(),
                                new int[] {1, plan.expandedLength()}),
                            caches,
                            new int[] {plan.expandedLength()},
                            embedded -> SmolVlmModel.merge(embedded, features, plan));
              } else {
                int token = input.required("target_ids").get(step - 1).asInt();
                logits =
                    model
                        .decoderForTest()
                        .stepLogits(
                            MLX.array(activation, new int[] {token}, new int[] {1, 1}),
                            caches,
                            new int[] {1},
                            null);
              }
              float[] actual = logits.toFloatArray();
              reference.compare(expectedLogits.get(step), actual);
              JsonNode expected = expectedLogits.get(step);
              int best = 0;
              int runnerUp = 1;
              if (expected.get(runnerUp).asDouble() > expected.get(best).asDouble()) {
                best = 1;
                runnerUp = 0;
              }
              for (int i = 2; i < expected.size(); i++) {
                if (expected.get(i).asDouble() > expected.get(best).asDouble()) {
                  runnerUp = best;
                  best = i;
                } else if (expected.get(i).asDouble() > expected.get(runnerUp).asDouble()) {
                  runnerUp = i;
                }
              }
              double gap = expected.get(best).asDouble() - expected.get(runnerUp).asDouble();
              assertEquals(input.required("gaps").get(step).asDouble(), gap, 1e-6);
              assertEquals(
                  List.of(best, runnerUp),
                  TierBReferences.ids(input.required("top_two_ids").get(step)));
              reference.stable(gap);
              int winner = 0;
              for (int i = 1; i < actual.length; i++) {
                if (actual[i] > actual[winner]) {
                  winner = i;
                }
              }
              assertEquals(input.required("target_ids").get(step).asInt(), winner);
              MLX.eval(DecoderModel.stepCacheArrays(caches));
            }
          }
        }
        GenerationResult result = model.generate(request, ignored -> {});
        assertEquals(TierBReferences.ids(input.required("target_ids")), result.generatedTokenIds());
        assertEquals(
            TierBReferences.ids(input.required("unexpanded_ids")), result.promptTokenIds());
        assertEquals(plan.expandedLength(), result.promptPositions());
        assertEquals(
            FinishReason.valueOf(input.required("finish_reason").asString()),
            result.finishReason());
      }
      reference.report();
      long nativePeak = MLXMemory.peakBytes();
      long peakFootprint = footprint.maximum();
      assertTrue(
          nativePeak <= NATIVE_PEAK_BOUND_BYTES,
          "MLX native peak "
              + nativePeak
              + " bytes exceeds the "
              + NATIVE_PEAK_BOUND_BYTES
              + " bound");
      assertTrue(
          peakFootprint <= FOOTPRINT_BOUND_BYTES,
          "process footprint "
              + peakFootprint
              + " bytes exceeds the "
              + FOOTPRINT_BOUND_BYTES
              + " bound");
      System.out.println("Tier B sampled_peak_phys_footprint_bytes=" + peakFootprint);
    }
  }

  /**
   * Re-computes a reference case's {@code pixel_values_sha256} for its selected images.
   *
   * <p>{@code vision_tier_b.py} records {@code
   * hashlib.sha256(inputs['pixel_values'].numpy().tobytes())}, where the pinned slow {@code
   * Idefics3ImageProcessor} returns a C-contiguous little-endian float32 tensor, channel-first
   * ({@code [tiles, 3, height, width]}), with one tile per split crop (row-major) plus the global
   * thumbnail, per selected image in selection order. jmlx-vision emits the same normalized float32
   * values per tile in NHWC order, so each tile is transposed channel-first here before
   * serialization. The processor pads each image's tiles to that image's own largest tile, which
   * equals the reference's batch-wide padding for the pinned manifest (one image per request, every
   * tile the same size).
   */
  private static String pixelValuesSha256(SmolVlmImageProcessor processor, List<RgbImage> images)
      throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    for (RgbImage image : images) {
      for (ImageTensor tile : processor.process(image).tiles()) {
        int[] shape = tile.shape(); // NHWC [1, height, width, 3]
        int height = shape[1];
        int width = shape[2];
        ByteBuffer bytes =
            ByteBuffer.allocate(3 * height * width * 4).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = tile.values();
        for (int channel = 0; channel < 3; channel++) {
          for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
              bytes.putFloat(values[(y * width + x) * 3 + channel]);
            }
          }
        }
        digest.update(bytes.array());
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  /**
   * Samples the test JVM's macOS physical footprint every 20 ms while inference runs; this thread
   * never calls MLX.
   *
   * <p>{@code ps -o rss} counts only the process's private resident pages and misses memory the MLX
   * allocator and Metal hold for the process (the recorded Tier-B RSS sampled below the
   * checkpoint's own weight size), so it is not useful memory evidence. {@code ri_phys_footprint}
   * from libproc's {@code proc_pid_rusage(RUSAGE_INFO_V2)} is the figure Activity Monitor and
   * {@code top} report as the process's memory: it covers the JVM's private and shared pages plus
   * file-backed mappings, so it is what the hosted runner's RAM must actually hold.
   *
   * <p>Binding: {@code /usr/lib/libproc.dylib} is loaded with {@link System#load} and resolved
   * through {@link SymbolLookup#loaderLookup()}, the same mechanism the committed jextract bindings
   * use for libmlxc. In the macOS SDK's {@code struct rusage_info_v2} (sys/resource.h) {@code
   * ri_phys_footprint} sits at byte 72 -- the 16-byte {@code ri_uuid} plus the seven 8-byte fields
   * {@code ri_user_time} through {@code ri_resident_size} precede it -- verified against the SDK
   * header's layout on this machine. The buffer over-allocates the whole struct, and each sample is
   * rejected below a plausibility floor so a layout shift that moves or zeroes the field fails the
   * gate instead of passing it with a tiny reading.
   */
  private static final class FootprintSampler implements AutoCloseable {
    // proc_pid_rusage(int pid, int flavor, void *buffer); RUSAGE_INFO_V2 = 2.
    private static final int FLAVOR_RUSAGE_INFO_V2 = 2;
    // struct rusage_info_v2 in sys/resource.h: ri_phys_footprint at byte 72.
    private static final int PHYS_FOOTPRINT_OFFSET = 72;
    // A running test JVM always holds tens of MiB of footprint; a smaller reading means the
    // field was misparsed, not that the process shrank.
    private static final long MINIMUM_PLAUSIBLE_BYTES = 64L * 1024 * 1024;

    // libproc must be loaded before its symbol is resolved; both run once at class
    // initialization, before any sample.
    private static final MethodHandle PROC_PID_RUSAGE = bindProcPidRusage();

    private static MethodHandle bindProcPidRusage() {
      System.load("/usr/lib/libproc.dylib");
      return Linker.nativeLinker()
          .downcallHandle(
              SymbolLookup.loaderLookup()
                  .or(Linker.nativeLinker().defaultLookup())
                  .findOrThrow("proc_pid_rusage"),
              FunctionDescriptor.of(
                  ValueLayout.JAVA_INT,
                  ValueLayout.JAVA_INT,
                  ValueLayout.JAVA_INT,
                  ValueLayout.ADDRESS));
    }

    private volatile boolean running = true;
    private volatile long maximum;
    private volatile Exception failure;
    private final Thread worker;

    private FootprintSampler() {
      worker =
          new Thread(
              () -> {
                try {
                  while (running) {
                    sample();
                    Thread.sleep(20);
                  }
                } catch (Exception exception) {
                  failure = exception;
                }
              },
              "tier-b-footprint-sampler");
      worker.start();
    }

    long maximum() throws InterruptedException {
      sample();
      return maximum;
    }

    private synchronized void sample() throws InterruptedException {
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment buffer = arena.allocate(256, 8);
        int status;
        try {
          status =
              (int)
                  PROC_PID_RUSAGE.invoke(
                      (int) ProcessHandle.current().pid(), FLAVOR_RUSAGE_INFO_V2, buffer);
        } catch (InterruptedException exception) {
          throw exception;
        } catch (Throwable exception) {
          throw new IllegalStateException("proc_pid_rusage downcall failed", exception);
        }
        if (status != 0) {
          throw new IllegalStateException("proc_pid_rusage failed with status " + status);
        }
        long footprint = buffer.get(ValueLayout.JAVA_LONG, PHYS_FOOTPRINT_OFFSET);
        if (footprint < MINIMUM_PLAUSIBLE_BYTES) {
          throw new IllegalStateException(
              "implausible phys_footprint "
                  + footprint
                  + " bytes at offset "
                  + PHYS_FOOTPRINT_OFFSET);
        }
        maximum = Math.max(maximum, footprint);
      }
    }

    @Override
    public void close() throws InterruptedException {
      running = false;
      worker.join();
      assertEquals(null, failure, "footprint sampler failure");
      assertTrue(maximum >= MINIMUM_PLAUSIBLE_BYTES);
    }
  }
}
