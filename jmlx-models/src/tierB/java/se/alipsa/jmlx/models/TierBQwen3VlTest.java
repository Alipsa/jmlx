package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
import se.alipsa.jmlx.vision.Qwen3VlImageProcessor;
import se.alipsa.jmlx.vision.Qwen3VlProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/**
 * Opt-in pinned real Qwen3-VL 2B image/text references: byte-for-byte preprocessing on the pinned
 * artifact, exact expansion and RoPE-delta facts against the independent CPU float32 reference,
 * cached-versus-reference logits at every greedy step, and stable greedy IDs under measured
 * top-logit margins.
 *
 * <p>Each case's recorded {@code pixel_values_sha256} list is re-computed from the pinned images
 * through {@code jmlx-vision}'s Qwen3-VL processor (the same call the model's prefill runs) and
 * asserted (see {@link #sha256LeFloat32}), so preprocessing on the real artifact is verified
 * byte-for-byte, not only indirectly through the logits. The {@code reference.gaps} and {@code
 * reference.top_two_ids} recorded by the reference generator are re-derived from the reference
 * vectors themselves (a pin of the reference's own consistency), while the stability gate {@code
 * TierBReferences#stable} is applied to both the reference vectors and the Java run's own measured
 * margins.
 *
 * <p>The memory gates mirror {@link TierBVisionLanguageTest}: upper-bound gates on the two measured
 * figures, sized from the manifest's recorded local_evidence
 * (tools/tier-b/qwen3vl-2b-instruct.json) so a footprint or allocator regression fails this suite
 * explicitly. The footprint bound is deliberately close to the measured peak: it is a regression
 * detector, not a guarantee that the CI runner survives a large regression (the OS and the Gradle
 * daemon share the runner's RAM, and macOS may swap or kill before the suite observes it; see the
 * workflow comment in .github/workflows/tier-b.yml).
 */
@EnabledIfNativeAvailable
class TierBQwen3VlTest {
  // Recorded in the manifest's local_evidence (macOS 27.0.1, TF32 off, pinned mlx-metal 0.31.2
  // / mlx-c fba4470): the suite's MLX allocator peak is 9,828,890,788 bytes. The bound adds 500
  // MiB, well above cross-host allocator variance (the peak is mlx's own byte accounting, not OS
  // memory). It is a regression detector, not a guarantee the CI runner's RAM survives a
  // large regression: peakBytes() is read only after generation finishes, and a regression
  // retaining one extra float32 copy of this checkpoint's weights (about 8.5 GB) would push the
  // process past the runner's RAM before the assertion runs.
  private static final long NATIVE_PEAK_BOUND_BYTES = 9_828_890_788L + 500L * 1024 * 1024;
  // The same local run's sampled process-footprint peak is 11,979,116,480 bytes (the ~9.2 GB MLX
  // pool plus the JVM). The bound adds ~520 MB of headroom above it for JVM/OS variance on the
  // CI runner: it fails the suite explicitly when the footprint regresses, but it is NOT a
  // guarantee the self-hosted runner's RAM survives a large regression (the OS and the
  // Gradle daemon share that RAM, and macOS may swap or kill before the suite observes it).
  private static final long FOOTPRINT_BOUND_BYTES = 12_500_000_000L;

  @Test
  void multiImagePromptCachedLogitsAndStableGreedyIds() throws Exception {
    TierBReferences reference = new TierBReferences("qwen3-vl-generation");
    final HfTokenizer tokenizer = HfTokenizer.fromDirectory(reference.directory);
    Path repository = Path.of(System.getProperty("jmlx.repository.root"));
    var logitsSpec = reference.reference.required("logits_file");
    Path logitsFile = repository.resolve(logitsSpec.required("path").asString());
    assertEquals(
        logitsSpec.required("sha256").asString(),
        HexFormat.of()
            .formatHex(
                MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(logitsFile))));
    // Little-endian float32: per case, per greedy step, one full vocab-size vector per step.
    byte[] logitsBytes = Files.readAllBytes(logitsFile);
    float[] logitsBlob = new float[logitsBytes.length / 4];
    ByteBuffer.wrap(logitsBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(logitsBlob);
    Qwen3VlImageProcessor processor =
        new Qwen3VlImageProcessor(
            Qwen3VlProcessorConfig.fromJson(
                Files.readAllBytes(reference.directory.resolve("preprocessor_config.json"))));
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
    int caseIndex = 0;
    try (TierBFootprintSampler footprint = new TierBFootprintSampler();
        MLXScope root = new MLXScope()) {
      long loadStart = System.nanoTime();
      Qwen3VlModel model = (Qwen3VlModel) TextGenerationModels.load(root, reference.directory);
      long loadWallMs = (System.nanoTime() - loadStart) / 1_000_000;
      System.out.println("Tier B qwen3-vl load_wall_ms=" + loadWallMs);
      reference.model(model.metadata());
      int vocab = model.metadata().vocabSize();
      int blobOffset = 0;
      for (JsonNode input : reference.reference.required("cases")) {
        caseIndex++;
        final int steps = input.required("target_ids").size();
        List<RgbImage> selected = new ArrayList<>();
        for (JsonNode index : input.required("image_indices")) {
          selected.add(images.get(index.asInt()));
        }
        for (int i = 0; i < selected.size(); i++) {
          var processed = processor.process(selected.get(i));
          assertEquals(
              input.required("pixel_values_sha256").get(i).asString(),
              sha256LeFloat32(processed.values()),
              "processed pixel values drifted from the reference bytes (image " + i + ")");
          int[] grid =
              new int[] {
                input.required("grids").get(i).get(0).asInt(),
                input.required("grids").get(i).get(1).asInt(),
                input.required("grids").get(i).get(2).asInt()
              };
          assertEquals(
              grid[1] * grid[2] / 4, processed.mergedRowCount(), "merged rows for image " + i);
        }
        List<Map<String, Object>> messages = new ArrayList<>();
        ObjectMapper mapper = new ObjectMapper();
        for (JsonNode message : input.required("messages")) {
          @SuppressWarnings("unchecked")
          Map<String, Object> converted = mapper.convertValue(message, Map.class);
          messages.add(converted);
        }
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
        Qwen3VlPromptPlan plan = model.plan(request.promptTokenIds(), request.images());
        assertEquals(
            TierBReferences.ids(input.required("expanded_ids")),
            Arrays.stream(plan.expandedIds()).boxed().toList());
        assertEquals(input.required("delta").asInt(), plan.positionDelta());
        for (int i = 0; i < selected.size(); i++) {
          int[] grid =
              new int[] {
                input.required("grids").get(i).get(0).asInt(),
                input.required("grids").get(i).get(1).asInt(),
                input.required("grids").get(i).get(2).asInt()
              };
          assertEquals(
              grid[1] * grid[2] / 4,
              plan.images().get(i).featureCount(),
              "expanded image-token run for image " + i);
        }
        try (MLXScope generation = root.newChild()) {
          List<KVCache> caches = Qwen3VlTestSupport.caches(model, generation);
          for (int step = 0; step < steps; step++) {
            try (MLXScope activation = generation.newChild()) {
              MLXArray logits;
              if (step == 0) {
                logits =
                    model
                        .prefill(CancellationToken.NONE, selected, plan, activation, caches)
                        .logits();
              } else {
                int token = input.required("target_ids").get(step - 1).asInt();
                logits = Qwen3VlTestSupport.decode(model, activation, plan, caches, token);
                MLX.eval(DecoderModel.stepCacheArrays(caches));
              }
              float[] actual = logits.toFloatArray();
              float[] expected =
                  Arrays.copyOfRange(
                      logitsBlob, blobOffset + step * vocab, blobOffset + (step + 1) * vocab);
              reference.compare(expected, actual);
              // The reference's own recorded consistency: re-derive gap and top-two from the
              // reference vector and pin them against the manifest.
              double gap = gapOf(expected);
              assertEquals(input.required("gaps").get(step).asDouble(), gap, 1e-6);
              assertEquals(
                  TierBReferences.ids(input.required("top_two_ids").get(step)),
                  topTwoIds(expected));
              reference.stable(gap);
              // The Java run's own measured margin must also be stable, so the exact greedy ID
              // the real runtime picks is credible, not just the reference's.
              double javaGap = gapOf(actual);
              assertTrue(
                  javaGap > 2 * reference.epsilon + reference.margin,
                  "Java top-logit margin " + javaGap + " at step " + step + " is not stable");
              assertEquals(input.required("target_ids").get(step).asInt(), argmax(actual));
            }
          }
          // The blob advance is exact: one full vector per consumed step, per case.
          blobOffset += vocab * steps;
        }
        long generateStart = System.nanoTime();
        GenerationResult result = model.generate(request, ignored -> {});
        final long generateWallMs = (System.nanoTime() - generateStart) / 1_000_000;
        assertEquals(TierBReferences.ids(input.required("target_ids")), result.generatedTokenIds());
        // The plan's exact decoded-output assertion: the end-to-end generate's decoded text must
        // equal the reference's tokenizer.decode(target_ids, skip_special_tokens=True).
        assertEquals(input.required("expected_text").asString(), result.generatedText());
        assertEquals(
            TierBReferences.ids(input.required("unexpanded_ids")), result.promptTokenIds());
        assertEquals(plan.expandedLength(), result.promptPositions());
        assertEquals(
            FinishReason.valueOf(input.required("finish_reason").asString()),
            result.finishReason());
        System.out.println(
            "Tier B qwen3-vl case="
                + caseIndex
                + " expanded_length="
                + plan.expandedLength()
                + " generate_wall_ms="
                + generateWallMs);
      }
      assertEquals(logitsBlob.length, blobOffset, "logits blob accounting");
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

  /** The reference generator's per-image pixel hash: little-endian float32 bytes, C order. */
  private static String sha256LeFloat32(float[] values) throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      bytes.putFloat(value);
    }
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes.array()));
  }

  private static long argmax(float[] values) {
    int winner = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[winner]) {
        winner = i;
      }
    }
    return winner;
  }

  private static double gapOf(float[] values) {
    int best = 0;
    int runnerUp = 1;
    if (values[runnerUp] > values[best]) {
      best = 1;
      runnerUp = 0;
    }
    for (int i = 2; i < values.length; i++) {
      if (values[i] > values[best]) {
        runnerUp = best;
        best = i;
      } else if (values[i] > values[runnerUp]) {
        runnerUp = i;
      }
    }
    return values[best] - values[runnerUp];
  }

  private static List<Integer> topTwoIds(float[] values) {
    int best = 0;
    int runnerUp = 1;
    if (values[runnerUp] > values[best]) {
      best = 1;
      runnerUp = 0;
    }
    for (int i = 2; i < values.length; i++) {
      if (values[i] > values[best]) {
        runnerUp = best;
        best = i;
      } else if (values[i] > values[runnerUp]) {
        runnerUp = i;
      }
    }
    return List.of(best, runnerUp);
  }
}
