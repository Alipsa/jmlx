package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashSet;
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
 * Opt-in manual 8B 4-bit run (Phase 7.3b WP8, plan §9: "Pin and run a compatible public 8B 4-bit
 * artifact manually"). Never part of ordinary check: it runs only when {@code JMLX_8B_PROBE_DIR}
 * points at a local copy of the pinned {@code mlx-community/Qwen3-VL-8B-Instruct-4bit} snapshot,
 * and is evidence-only (no parity claim against the independent mlx-vlm probe; see {@code
 * req/phase6-4-benchmark.md}).
 *
 * <p>The pinned artifact's {@code model.safetensors.index.json} is a stale converter remnant: it is
 * the original HF checkpoint's 4-shard index ({@code model-0000X-of-00004.safetensors}, HF tensor
 * names, total_size of the pre-conversion model), while the two committed shards carry the MLX
 * conversion's layout (1,258 tensors under {@code language_model}/vision prefixes, {@code format:
 * mlx}). jmlx's {@code CheckpointLoader} preflight correctly refuses an index that references
 * shards the artifact does not contain, so this probe verifies every one of the 16 pinned file
 * digests and then loads the 15-file run directory built as hardlinks of the verified files with
 * the stale index excluded. Successful load is the tensor-plan identity evidence: the full Qwen3-VL
 * plan resolved all 1,258 shard tensors (254 U32-packed language leaves with their BF16 scale/bias
 * sidecars, the entirely-BF16 vision tower).
 *
 * <p>Sequence and preprocessing facts are asserted against the independent mlx-vlm probe golden
 * (tools/qwen-vl-reference/goldens/qwen3-vl-8b-wp1-mlx-probe.json), which consumed the same chat
 * template, tokenizer, fixtures and slow-processor pixels: expansion length, image-token span, RoPE
 * delta and the per-image processed-pixel SHA must agree on both runtimes. Greedy output IDs are
 * compared to the probe's but not asserted equal -- different runtime (jmlx float32 dequantized
 * 4-bit on Metal) and different checkpoint (8B vs the probe's own 8B 4-bit run) makes that a parity
 * question the plan deliberately does not claim.
 */
@EnabledIfNativeAvailable
class Qwen3Vl8bManualProbeTest {
  // How many greedy steps to run per image (the mlx-vlm probe recorded five).
  private static final int TOKENS = 5;
  private static final List<String> PROBE_IMAGES =
      List.of("model-rgb-256x256.png", "model-rgb-64x32.png");
  private static final String PROBE_PROMPT_TEXT = "Describe the image in one short sentence.";

  @Test
  void manualProbe() throws Exception {
    String artifactArg = System.getenv("JMLX_8B_PROBE_DIR");
    assumeTrue(
        artifactArg != null && !artifactArg.isBlank(),
        "Set JMLX_8B_PROBE_DIR to a local copy of the pinned 8B 4-bit snapshot to run the probe");
    Path repository = Path.of(System.getProperty("jmlx.repository.root"));
    Path artifact = Path.of(artifactArg);
    if (!artifact.isAbsolute()) {
      // The test JVM's working directory is the module directory; relative paths are
      // repository-root-relative, like every other Tier-B/JMLX path environment variable.
      artifact = repository.resolve(artifact);
    }
    artifact = artifact.toAbsolutePath().normalize();
    ObjectMapper mapper = new ObjectMapper();
    JsonNode golden =
        mapper.readTree(
            repository
                .resolve("tools/qwen-vl-reference/goldens/qwen3-vl-8b-wp1-mlx-probe.json")
                .toFile());

    // Every one of the 16 pinned files must still carry its WP1 digest before anything else.
    var files = golden.required("artifact").required("files");
    Set<String> verified = new LinkedHashSet<>();
    for (JsonNode file : files) {
      String name = file.required("path").asString();
      Path path = artifact.resolve(name);
      String sha = sha256(path);
      assertEquals(file.required("sha256").asString(), sha, "artifact file drifted: " + name);
      verified.add(name);
    }
    assertEquals(16, verified.size(), "the pinned 8B artifact has 16 files");

    // The stale-index evidence, recorded rather than assumed: the index's weight_map references
    // shards absent from the artifact and names absent from the shards' MLX-layout headers.
    JsonNode index = mapper.readTree(artifact.resolve("model.safetensors.index.json").toFile());
    JsonNode weightMap = index.required("weight_map");
    Set<String> indexedShards = new LinkedHashSet<>();
    weightMap.properties().forEach(entry -> indexedShards.add(entry.getValue().asString()));
    Path artifactRoot = artifact;
    assertTrue(
        indexedShards.stream().anyMatch(s -> !Files.isRegularFile(artifactRoot.resolve(s))),
        "expected at least one weight_map shard reference the artifact does not contain");

    // Run directory: hardlinks of the 15 verified non-index files (the index is excluded;
    // everything else the loader reads -- shards, configs, tokenizer, preprocessor -- is the
    // byte-identical pinned file, so the load below runs the pinned artifact, minus the
    // converter-remnant index).
    Path runDir = artifact.resolveSibling(artifact.getFileName() + "-jmlx-run");
    if (Files.isDirectory(runDir)) {
      Files.walk(runDir).sorted(java.util.Comparator.reverseOrder()).forEach(this::deleteFile);
    }
    Files.createDirectories(runDir);
    for (String name : verified) {
      if ("model.safetensors.index.json".equals(name)) {
        continue;
      }
      var link =
          new ProcessBuilder(
                  "ln", artifact.resolve(name).toString(), runDir.resolve(name).toString())
              .start();
      assertEquals(0, link.waitFor(), "hardlink failed: " + name);
    }
    long totalTensors = 0;
    long scalesTensors = 0;
    for (Map.Entry<Path, Set<String>> entry :
        SafetensorsHeaders.tensorNames(
                List.of(
                    runDir.resolve("model-00001-of-00002.safetensors"),
                    runDir.resolve("model-00002-of-00002.safetensors")))
            .entrySet()) {
      totalTensors += entry.getValue().size();
      for (String key : entry.getValue()) {
        if (key.endsWith(".scales")) {
          scalesTensors++;
          assertTrue(
              !key.startsWith("vision_tower."), "vision tensors must be unquantized: " + key);
        }
      }
    }
    // Findings §(8B tensor categories): 1,258 tensors, 254 U32-packed language leaves, each with
    // one BF16 scale sidecar; the vision tower is entirely unquantized.
    assertEquals(1258, totalTensors, "8B shard tensor count");
    assertEquals(254, scalesTensors, "8B packed-leaf scale sidecar count");

    MLXMemory.resetPeak();
    long loadStart = System.nanoTime();
    try (MLXScope root = new MLXScope()) {
      Qwen3VlModel model = (Qwen3VlModel) TextGenerationModels.load(root, runDir);
      long loadWallMs = (System.nanoTime() - loadStart) / 1_000_000;
      HfTokenizer tokenizer = HfTokenizer.fromDirectory(runDir);
      Qwen3VlImageProcessor processor =
          new Qwen3VlImageProcessor(
              Qwen3VlProcessorConfig.fromJson(
                  Files.readAllBytes(runDir.resolve("preprocessor_config.json"))));
      for (String imageName : PROBE_IMAGES) {
        runCase(mapper, golden, model, tokenizer, processor, repository, root, imageName);
      }
      System.out.println(
          "8B probe load_wall_ms="
              + loadWallMs
              + " native_peak_bytes="
              + MLXMemory.peakBytes()
              + " active_bytes="
              + MLXMemory.activeBytes());
    }
  }

  private void runCase(
      ObjectMapper mapper,
      JsonNode golden,
      Qwen3VlModel model,
      HfTokenizer tokenizer,
      Qwen3VlImageProcessor processor,
      Path repository,
      MLXScope root,
      String imageName)
      throws Exception {
    JsonNode c = golden.required("model").required("cases").required(imageName);
    RgbImage image =
        ImageDecoder.decode(
            repository.resolve("tools/image-oracle/fixtures/qwen3vl").resolve(imageName));
    // The mlx-vlm probe injected the fast-comparison runner's slow_compensated pixel buffer for
    // this exact fixture; the jmlx port must reproduce those bytes before the model sees them.
    assertEquals(
        c.required("export").required("pixel_values_sha256").asString(),
        sha256LeFloat32(processor.process(image).values()),
        imageName + " processed pixels drifted from the probe bytes");
    List<Map<String, Object>> messages =
        List.of(
            Map.of(
                "role",
                "user",
                "content",
                List.of(
                    Map.of("type", "image"), Map.of("type", "text", "text", PROBE_PROMPT_TEXT))));
    GenerationRequest request =
        GenerationRequest.chat(
                tokenizer,
                messages,
                ChatTemplateOptions.defaults(true),
                GenerationConfig.greedyDefaults(
                    TOKENS, Set.of(tokenizer.eosTokenId().orElseThrow())),
                CancellationToken.NONE)
            .withImages(List.of(image));
    Qwen3VlPromptPlan plan = model.plan(request.promptTokenIds(), request.images());
    JsonNode sequence = c.required("sequence");
    assertEquals(
        sequence.required("length").asInt(), plan.expandedLength(), imageName + " expanded length");
    Qwen3VlImagePlacement placement = plan.images().get(0);
    int spanStart = sequence.required("image_token_span").get(0).asInt();
    int spanEnd = sequence.required("image_token_span").get(1).asInt();
    assertEquals(spanStart, placement.featureStart(), imageName + " image-token span start");
    // The probe's span is inclusive on both ends ([start, start + count - 1]).
    assertEquals(
        spanEnd - spanStart + 1, placement.featureCount(), imageName + " image-token span count");
    assertEquals(
        c.required("rope").required("delta_mlx").asInt(),
        plan.positionDelta(),
        imageName + " rope delta");

    List<Integer> ids = new ArrayList<>();
    double minimumGap = Double.MAX_VALUE;
    long prefillNanos = 0;
    long decodeNanos = 0;
    try (MLXScope generation = root.newChild()) {
      List<KVCache> caches = Qwen3VlTestSupport.caches(model, generation);
      for (int step = 0; step < TOKENS; step++) {
        long stepStart = System.nanoTime();
        float[] logits;
        try (MLXScope activation = generation.newChild()) {
          MLXArray values;
          if (step == 0) {
            values =
                model
                    .prefill(CancellationToken.NONE, List.of(image), plan, activation, caches)
                    .logits();
          } else {
            values =
                Qwen3VlTestSupport.decode(model, activation, plan, caches, ids.get(ids.size() - 1));
            MLX.eval(DecoderModel.stepCacheArrays(caches));
          }
          logits = values.toFloatArray();
        }
        long stepWallNanos = System.nanoTime() - stepStart;
        int token = argmax(logits);
        ids.add(token);
        minimumGap = Math.min(minimumGap, topGap(logits));
        if (step == 0) {
          prefillNanos = stepWallNanos;
        } else {
          decodeNanos += stepWallNanos;
        }
      }
    }
    List<Integer> probeIds = new ArrayList<>();
    for (JsonNode id : c.required("greedy_ids")) {
      probeIds.add(id.asInt());
    }
    System.out.println(
        "8B probe case="
            + imageName
            + " image_px="
            + image.width()
            + "x"
            + image.height()
            + " grid="
            + c.required("grid").toString()
            + " expanded_length="
            + plan.expandedLength()
            + " prefill_ms="
            + (prefillNanos / 1_000_000)
            + " prefill_tokens_per_s="
            + (plan.expandedLength() * 1_000_000_000.0 / prefillNanos)
            + " decode_ms_per_token="
            + ((decodeNanos / (TOKENS - 1)) / 1_000_000)
            + " decode_tokens_per_s="
            + ((TOKENS - 1) * 1_000_000_000.0 / decodeNanos)
            + " greedy_ids="
            + ids
            + " min_top_gap="
            + minimumGap
            + " mlx_vlm_probe_ids="
            + probeIds
            + " ids_match_probe="
            + ids.equals(probeIds)
            + " decoded_text="
            + tokenizer.decode(ids, false));
  }

  /** Streaming SHA-256: the shard files are multi-GB and must never be read all at once. */
  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var in = Files.newInputStream(path)) {
      byte[] buffer = new byte[8 * 1024 * 1024];
      int read;
      while ((read = in.read(buffer)) >= 0) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private void deleteFile(Path path) {
    try {
      Files.delete(path);
    } catch (Exception e) {
      throw new RuntimeException("failed to delete prepared run directory entry " + path, e);
    }
  }

  /** The probe generator's pixel hash: little-endian float32 bytes, row-major. */
  private static String sha256LeFloat32(float[] values) throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      bytes.putFloat(value);
    }
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes.array()));
  }

  private static int argmax(float[] values) {
    int winner = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[winner]) {
        winner = i;
      }
    }
    return winner;
  }

  private static double topGap(float[] values) {
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
}
