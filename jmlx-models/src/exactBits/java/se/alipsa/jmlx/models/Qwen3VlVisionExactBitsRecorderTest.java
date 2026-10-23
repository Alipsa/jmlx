package se.alipsa.jmlx.models;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.mlx_h;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessor;
import se.alipsa.jmlx.vision.Qwen3VlImageProcessorResult;
import se.alipsa.jmlx.vision.Qwen3VlProcessorConfig;
import se.alipsa.jmlx.vision.RgbImage;
import tools.jackson.databind.JsonNode;

/**
 * Opt-in exact-bit recorder/comparator for the WP5 Qwen3-VL vision tower's per-block
 * evaluate-and-release change (PR #58 review, findings §11.4). It mirrors the 7.3a decoder
 * recorder's mechanism — driven by {@code jmlx.exact.bits.mode} ({@code record} | {@code verify})
 * and the {@code MLX_ENABLE_TF32} environment (1 = default reduced precision, 0 = full float32),
 * with the four {@code exactBits{Record,Verify}VisionTf32{On,Off}} tasks setting both and forking
 * their own JVMs — but records one vision-specific probe: the {@code Qwen3VlVisionPeakTest} depth-3
 * synthetic probe, i.e. the real-2B tower geometry (hidden 1024, 16 heads, head_dim 64,
 * intermediate 4096, out_hidden 2048, 2304 position embeddings, patch/temporal 16/2, merge 2) at
 * depth 3 with DeepStack index [1], constant 0.001 weights via {@link MLX#full} (the probe is fully
 * deterministic — no seed is involved) and the peak test's synthetic 1024×1024 image (grid (1, 64,
 * 64), 4096 patch rows, fixed per-pixel function of the coordinates) processed through {@link
 * Qwen3VlImageProcessor}.
 *
 * <p>The probe runs through {@code Qwen3VlVisionTower.forwardForTest}, and six captures are
 * recorded: the final features (the merged rows and DeepStack level 0) plus the per-image golden
 * captures (the interpolated position embedding, the patch-embedding output, and the block 0 and
 * block 1 outputs) — so a bit change anywhere in the per-block loop shows up in a capture.
 * Recordings are PR evidence under {@code build/exact-bits/wp5-vision/tf32-<0|1>} — a sibling of,
 * and never colliding with, the 7.3a decoder recorder's {@code build/exact-bits/tf32-<0|1>}:
 *
 * <ul>
 *   <li>{@code <capture>.json} — one file per capture, carrying {@code values}, the Base64 of the
 *       big-endian IEEE-754 bits of every float32 in row-major order ({@code
 *       Float.floatToIntBits}).
 *   <li>{@code manifest.json} — host metadata (git commit, the two native pins, device, macOS,
 *       precision, the probe-spec hash, and the provenance-only never-compared {@code gitDirty}
 *       tree-dirtiness flag and the SHA-256 of the guarded {@code Qwen3VlVisionTower} source) plus
 *       the capture list.
 * </ul>
 *
 * <p>Record mode refuses to overwrite an existing {@code tf32-<0|1>} recording: a baseline is
 * evidence, and silently replacing it with a recording from a later commit would invalidate the
 * very comparison it feeds. Verify mode checks the recorded manifest cheaply first — existence,
 * format version, metadata in every compared field, capture set — and then requires bit-exact
 * agreement of every float; the only field allowed to differ is the git commit (the comparison
 * crosses the review's own commits). Deliberately NOT {@code @EnabledIfNativeAvailable}: as an
 * opt-in baseline tool it fails loudly when the native runtime is absent instead of silently
 * skipping.
 */
class Qwen3VlVisionExactBitsRecorderTest {

  static final int FORMAT_VERSION = 1;

  // The Qwen3VlVisionPeakTest depth-3 probe constants (real-2B geometry at reduced depth).
  private static final int HIDDEN = 1024;
  private static final int NUM_HEADS = 16;
  private static final int HEAD_DIM = 64;
  private static final int INTERMEDIATE = 4096;
  private static final int OUT_HIDDEN = 2048;
  private static final int POSITION_EMBEDDINGS = 2304;
  private static final int DEPTH = 3;
  private static final int[] DEEPSTACK_INDEXES = {1};
  private static final int MERGE = 2;
  private static final int MERGE_BLOCK = HIDDEN * MERGE * MERGE;
  private static final int IMAGE = 1024;
  private static final int GRID = IMAGE / 16;
  private static final int PATCH_ROWS = GRID * GRID;
  private static final int MERGED_ROWS = (GRID / MERGE) * (GRID / MERGE);
  private static final float WEIGHT = 0.001f;

  /** The capture names in the recorded order (final features first, then the golden captures). */
  private static final List<String> CAPTURES =
      List.of("merged", "deepstack0", "posEmbedInterpolated", "patchEmbed", "block0", "block1");

  @Test
  void recordOrVerify() throws Exception {
    String mode = System.getProperty("jmlx.exact.bits.mode", "");
    if (!mode.equals("record") && !mode.equals("verify")) {
      throw new AssertionError(
          "jmlx.exact.bits.mode must be 'record' or 'verify'; run one of the "
              + "exactBits{Record,Verify}Vision Gradle tasks, not this test directly");
    }
    String tf32 = System.getenv("MLX_ENABLE_TF32");
    if (!"0".equals(tf32) && !"1".equals(tf32)) {
      throw new AssertionError("MLX_ENABLE_TF32 must be set to 0 or 1 by the Gradle task");
    }
    Path repoRoot = Path.of(System.getProperty("jmlx.repository.root", "."));
    requireNativeRuntime(repoRoot);
    Path modeDir =
        Path.of(System.getProperty("jmlx.exact.bits.dir", "build/exact-bits"))
            .resolve("wp5-vision")
            .resolve("tf32-" + tf32);
    ExactBitsMetadata global = metadata(repoRoot, tf32);
    // Record mode never silently replaces an existing baseline: fail before the GPU capture and
    // before any file is written, naming the existing recording's commit.
    if (mode.equals("record")) {
      refuseOverwrite(modeDir);
    }
    JsonNode manifest = mode.equals("verify") ? verifyPreCapture(global, modeDir) : null;
    Map<String, float[]> captures = captureProbe();
    if (mode.equals("record")) {
      record(global, captures, modeDir);
      System.out.println("exact-bits record vision tf32=" + tf32 + ": " + summary(captures));
    } else {
      verifyPostCapture(global, manifest, captures, modeDir);
      // The recorded commit at a glance: whether the on-disk baseline is the pre-change one or a
      // later re-record shows here immediately.
      System.out.println(
          "exact-bits verify vision tf32="
              + tf32
              + " (recorded at "
              + manifest.path("metadata").path("gitCommit").asString()
              + "): "
              + summary(captures));
    }
  }

  /**
   * Runs the depth-3 probe and returns its captures in the recorded order. Every shape is checked
   * before any capture is written or compared: a divergent shape would otherwise surface as a
   * confusing length mismatch deep in the bit comparison instead of here, naming the capture.
   */
  private static Map<String, float[]> captureProbe() {
    try (MLXScope scope = new MLXScope()) {
      List<MLXArray> statics = new ArrayList<>();
      final Qwen3VlVisionTower tower = buildTower(scope, statics);
      MLX.eval(statics.toArray(MLXArray[]::new));
      Qwen3VlImageProcessorResult image = processor().process(syntheticImage());
      checkShape(image.gridThw(), "grid", 1, GRID, GRID);
      if (image.rawPatchRows() != PATCH_ROWS) {
        throw new AssertionError(
            "patch rows: expected " + PATCH_ROWS + ", got " + image.rawPatchRows());
      }
      try (MLXScope request = scope.newChild()) {
        Qwen3VlVisionTower.ForwardCaptures forward = tower.forwardForTest(request, List.of(image));
        Qwen3VlVisionFeatures features = forward.features();
        checkShape(features.merged().shape(), "merged", MERGED_ROWS, OUT_HIDDEN);
        if (features.deepstackLevels().size() != DEEPSTACK_INDEXES.length) {
          throw new AssertionError(
              "deepstack levels: expected "
                  + DEEPSTACK_INDEXES.length
                  + ", got "
                  + features.deepstackLevels().size());
        }
        Qwen3VlVisionTower.ImageCaptures imageCaptures = forward.images().get(0);
        checkShape(
            features.deepstackLevels().get(0).shape(), "deepstack0", MERGED_ROWS, OUT_HIDDEN);
        checkShape(
            imageCaptures.posEmbedInterpolated().shape(),
            "posEmbedInterpolated",
            PATCH_ROWS,
            HIDDEN);
        checkShape(imageCaptures.patchEmbed().shape(), "patchEmbed", PATCH_ROWS, HIDDEN);
        checkShape(imageCaptures.blockOutputs()[0].shape(), "block0", PATCH_ROWS, HIDDEN);
        checkShape(imageCaptures.blockOutputs()[1].shape(), "block1", PATCH_ROWS, HIDDEN);
        Map<String, float[]> out = new LinkedHashMap<>();
        out.put("merged", features.merged().toFloatArray());
        out.put("deepstack0", features.deepstackLevels().get(0).toFloatArray());
        out.put("posEmbedInterpolated", imageCaptures.posEmbedInterpolated().toFloatArray());
        out.put("patchEmbed", imageCaptures.patchEmbed().toFloatArray());
        out.put("block0", imageCaptures.blockOutputs()[0].toFloatArray());
        out.put("block1", imageCaptures.blockOutputs()[1].toFloatArray());
        return out;
      }
    }
  }

  /** The host/runtime metadata for one recording; the guarded source is the vision tower. */
  static ExactBitsMetadata metadata(Path repoRoot, String tf32) throws Exception {
    String commit = gitRevParse(repoRoot);
    boolean dirty = gitStatusDirty(repoRoot);
    String towerSha256 =
        ExactBitsSpec.sha256(
            Files.readAllBytes(
                repoRoot.resolve(
                    "jmlx-models/src/main/java/se/alipsa/jmlx/models/Qwen3VlVisionTower.java")));
    Path pinFile = repoRoot.resolve("native/install/lib/native-pin.properties");
    Properties pins = new Properties();
    try (InputStream in = Files.newInputStream(pinFile)) {
      pins.load(in);
    }
    return new ExactBitsMetadata(
        commit,
        requirePin(pins, "mlxMetalVersion", pinFile),
        requirePin(pins, "mlxcCommit", pinFile),
        deviceName(),
        System.getProperty("os.name"),
        System.getProperty("os.version"),
        tf32,
        probeSpecHash(),
        Map.of(),
        Map.of(),
        dirty,
        towerSha256);
  }

  /**
   * SHA-256 of the canonical encoding of the probe specification, so a probe change can never
   * silently compare against a recording made from a different one.
   */
  private static String probeSpecHash() {
    StringBuilder canonical = new StringBuilder();
    canonical.append('{');
    canonical.append("\"schema\":\"jmlx-phase7-3b-wp5-vision-probe-v1\"");
    canonical.append(",\"hidden\":").append(HIDDEN);
    canonical.append(",\"numHeads\":").append(NUM_HEADS);
    canonical.append(",\"headDim\":").append(HEAD_DIM);
    canonical.append(",\"intermediate\":").append(INTERMEDIATE);
    canonical.append(",\"outHidden\":").append(OUT_HIDDEN);
    canonical.append(",\"positionEmbeddings\":").append(POSITION_EMBEDDINGS);
    canonical.append(",\"depth\":").append(DEPTH);
    canonical.append(",\"deepstackIndexes\":[").append(DEEPSTACK_INDEXES[0]).append(']');
    canonical.append(",\"merge\":").append(MERGE);
    canonical.append(",\"image\":").append(IMAGE);
    canonical.append(",\"weightBits\":").append(Float.floatToIntBits(WEIGHT));
    canonical.append('}');
    return ExactBitsSpec.sha256(canonical.toString());
  }

  /** The MLX default device as a stable string: the same query the 7.3a recorder asserts. */
  private static String deviceName() {
    try (Arena tmp = Arena.ofConfined()) {
      MemorySegment typeOut = tmp.allocate(ValueLayout.JAVA_INT);
      int status = mlx_h.mlx_device_get_type(typeOut, MLX.defaultDevice());
      if (status != 0) {
        throw new AssertionError("mlx_device_get_type failed with status " + status);
      }
      int type = typeOut.get(ValueLayout.JAVA_INT, 0);
      return type == 1 ? "gpu" : type == 0 ? "cpu" : "unknown(" + type + ")";
    }
  }

  /** Fails (rather than skips) with the actionable bootstrap step when the runtime is absent. */
  private static void requireNativeRuntime(Path repoRoot) {
    String libDir = System.getProperty("jmlx.library.path", "");
    Path metallib =
        libDir.isBlank()
            ? repoRoot.resolve("native/install/lib/mlx.metallib")
            : Path.of(libDir).resolve("mlx.metallib");
    if (!Files.isRegularFile(metallib)) {
      throw new AssertionError(
          "exact-bits recording needs the staged native runtime, but "
              + metallib
              + " is missing -- run ./scripts/bootstrap-native.sh first");
    }
  }

  /** Reads a required native pin property, naming the missing key and file. */
  private static String requirePin(Properties pins, String key, Path pinFile) {
    String value = pins.getProperty(key);
    if (value == null) {
      throw new AssertionError(
          "native pin property '"
              + key
              + "' is missing from "
              + pinFile
              + " -- run ./scripts/bootstrap-native.sh");
    }
    return value;
  }

  private static String gitRevParse(Path repoRoot) throws Exception {
    Process process =
        new ProcessBuilder("git", "rev-parse", "HEAD")
            .directory(repoRoot.toFile())
            .redirectErrorStream(true)
            .start();
    String out = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8).trim();
    if (process.waitFor() != 0) {
      throw new AssertionError("git rev-parse HEAD failed in " + repoRoot + ": " + out);
    }
    return out;
  }

  /** Whether the tree is dirty: {@code git status --porcelain} prints anything at all. */
  private static boolean gitStatusDirty(Path repoRoot) throws Exception {
    Process process =
        new ProcessBuilder("git", "status", "--porcelain")
            .directory(repoRoot.toFile())
            .redirectErrorStream(true)
            .start();
    String out = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
    if (process.waitFor() != 0) {
      throw new AssertionError("git status --porcelain failed in " + repoRoot + ": " + out);
    }
    return !out.isEmpty();
  }

  /** Refuses to replace an existing recording (its {@code manifest.json} already present). */
  private static void refuseOverwrite(Path modeDir) throws IOException {
    Path manifestPath = modeDir.resolve("manifest.json");
    if (!Files.isRegularFile(manifestPath)) {
      return;
    }
    JsonNode manifest = JsonFiles.read(manifestPath);
    throw new AssertionError(
        "refusing to overwrite the existing exact-bit recording under "
            + modeDir
            + " (recorded at git commit "
            + manifest.path("metadata").path("gitCommit").asString()
            + ") -- a baseline is never silently replaced; to re-record, move or delete "
            + modeDir
            + " first");
  }

  /**
   * The cheap half of verification, before the GPU capture is spent: the recording must exist, use
   * the supported format version, carry comparable metadata (every field except the git commit and
   * the provenance-only gitDirty/tower hash) and name exactly the probe captures. Returns the
   * parsed manifest for {@link #verifyPostCapture}.
   */
  static JsonNode verifyPreCapture(ExactBitsMetadata global, Path modeDir) throws IOException {
    Path manifestPath = modeDir.resolve("manifest.json");
    if (!Files.isRegularFile(manifestPath)) {
      throw new AssertionError(
          "no exact-bit recordings under "
              + modeDir
              + " -- record first: ./gradlew :jmlx-models:exactBitsRecordVision");
    }
    JsonNode manifest = JsonFiles.read(manifestPath);
    if (manifest.path("formatVersion").asInt() != FORMAT_VERSION) {
      throw new AssertionError(
          "unsupported recording format version " + manifest.path("formatVersion").asInt());
    }
    List<String> mismatches =
        global.mismatches(ExactBitsMetadata.fromJson(manifest.path("metadata")), false);
    if (!mismatches.isEmpty()) {
      throw new AssertionError(
          "recordings are not comparable on this host; re-record: metadata mismatch in "
              + mismatches);
    }
    List<String> recordedNames = new ArrayList<>();
    for (JsonNode node : manifest.path("captures")) {
      recordedNames.add(node.asString());
    }
    if (!recordedNames.equals(CAPTURES)) {
      throw new AssertionError(
          "capture set differs from the recording: recorded "
              + recordedNames
              + ", candidate "
              + CAPTURES
              + " -- re-record after recording the probe at the base commit");
    }
    return manifest;
  }

  /** The expensive half of verification, after the probe has been re-run: bit-exact everywhere. */
  private static void verifyPostCapture(
      ExactBitsMetadata global, JsonNode manifest, Map<String, float[]> captures, Path modeDir)
      throws IOException {
    for (String capture : CAPTURES) {
      Path file = modeDir.resolve(capture + ".json");
      if (!Files.isRegularFile(file)) {
        throw new AssertionError("missing recording " + file + " -- record first");
      }
      JsonNode node = JsonFiles.read(file);
      List<String> mismatches =
          global.mismatches(ExactBitsMetadata.fromJson(node.path("metadata")), true);
      if (!mismatches.isEmpty()) {
        throw new AssertionError(
            "recordings are not comparable on this host; re-record: metadata mismatch in "
                + mismatches
                + " ["
                + capture
                + "]");
      }
      verifyCapture(capture, node, captures.get(capture), modeDir);
    }
    System.out.println("verified the vision probe against " + modeDir.resolve("manifest.json"));
  }

  /** One capture's bit-exact check: equal length, then equal bits in row-major order. */
  private static void verifyCapture(String capture, JsonNode node, float[] actual, Path modeDir) {
    int[] expected = floatBits(node.path("values").asString());
    int[] actualBits = new int[actual.length];
    for (int i = 0; i < actual.length; i++) {
      actualBits[i] = Float.floatToIntBits(actual[i]);
    }
    if (expected.length != actualBits.length) {
      throw new AssertionError(
          "exact-bit mismatch ["
              + capture
              + "]: lengths differ: recorded "
              + expected.length
              + " floats, candidate "
              + actualBits.length);
    }
    for (int i = 0; i < expected.length; i++) {
      if (expected[i] != actualBits[i]) {
        throw new AssertionError(
            "exact-bit mismatch ["
                + capture
                + "]: first divergent element "
                + i
                + "; expected bits 0x"
                + Integer.toHexString(expected[i])
                + " ("
                + Float.intBitsToFloat(expected[i])
                + "), actual bits 0x"
                + Integer.toHexString(actualBits[i])
                + " ("
                + Float.intBitsToFloat(actualBits[i])
                + ")");
      }
    }
  }

  private static void record(ExactBitsMetadata global, Map<String, float[]> captures, Path modeDir)
      throws IOException {
    Files.createDirectories(modeDir);
    int floats = 0;
    for (Map.Entry<String, float[]> capture : captures.entrySet()) {
      floatBits(global, capture.getKey(), capture.getValue(), modeDir);
      floats += capture.getValue().length;
    }
    writeManifest(global, captures, modeDir);
  }

  /** One capture file: the Base64 of the big-endian IEEE-754 bits, row-major. */
  private static void floatBits(ExactBitsMetadata global, String capture, float[] values, Path dir)
      throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4 * values.length);
    for (float value : values) {
      buffer.putFloat(value);
    }
    Files.writeString(
        dir.resolve(capture + ".json"),
        captureJson(
            capture,
            global.toJsonString(),
            "\"values\":\"" + Base64.getEncoder().encodeToString(buffer.array()) + '"'));
  }

  /** Decodes a recorded {@code values} string into the raw bits of each float, in order. */
  private static int[] floatBits(String encoded) {
    byte[] bytes = Base64.getDecoder().decode(encoded);
    if (bytes.length % 4 != 0) {
      throw new AssertionError("recorded values are not a multiple of 4 bytes: " + bytes.length);
    }
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    int[] bits = new int[bytes.length / 4];
    for (int i = 0; i < bits.length; i++) {
      bits[i] = buffer.getInt();
    }
    return bits;
  }

  private static String captureJson(String capture, String metadataJson, String payload) {
    return "{\"formatVersion\":"
        + FORMAT_VERSION
        + ",\"capture\":\""
        + ExactBitsMetadata.escapeJson(capture)
        + "\",\"metadata\":"
        + metadataJson
        + ","
        + payload
        + "}\n";
  }

  private static void writeManifest(
      ExactBitsMetadata global, Map<String, float[]> captures, Path modeDir) throws IOException {
    StringBuilder out = new StringBuilder();
    out.append("{\"formatVersion\":").append(FORMAT_VERSION).append(',');
    out.append("\"metadata\":").append(global.toJsonString()).append(',');
    out.append("\"captures\":[");
    int i = 0;
    for (String capture : captures.keySet()) {
      if (i++ > 0) {
        out.append(',');
      }
      out.append('"').append(ExactBitsMetadata.escapeJson(capture)).append('"');
    }
    out.append("]}\n");
    Files.writeString(modeDir.resolve("manifest.json"), out.toString());
  }

  /** The run summary: capture count and total float count. */
  private static String summary(Map<String, float[]> captures) {
    int floats = 0;
    for (float[] values : captures.values()) {
      floats += values.length;
    }
    return captures.size() + " captures, " + floats + " floats";
  }

  private static void checkShape(int[] shape, String capture, int... expected) {
    if (!Arrays.equals(shape, expected)) {
      throw new AssertionError(
          capture
              + " shape: expected "
              + Arrays.toString(expected)
              + ", got "
              + Arrays.toString(shape));
    }
  }

  /** The real-2B geometry at reduced depth 3 / DeepStack [1], constant 0.001 weights. */
  private static Qwen3VlVisionTower buildTower(MLXScope scope, List<MLXArray> statics) {
    Map<String, MLXArray> weights = new LinkedHashMap<>();
    put(weights, statics, scope, "patchEmbedding.weight", HIDDEN, 2, 16, 16, 3);
    put(weights, statics, scope, "patchEmbedding.bias", HIDDEN);
    put(weights, statics, scope, "positionEmbedding.weight", POSITION_EMBEDDINGS, HIDDEN);
    for (int i = 0; i < DEPTH; i++) {
      String layer = "layer" + i + ".";
      put(weights, statics, scope, layer + "attention.qkv.weight", 3 * HIDDEN, HIDDEN);
      put(weights, statics, scope, layer + "attention.qkv.bias", 3 * HIDDEN);
      put(weights, statics, scope, layer + "attention.proj.weight", HIDDEN, HIDDEN);
      put(weights, statics, scope, layer + "attention.proj.bias", HIDDEN);
      for (String norm : new String[] {"norm1", "norm2"}) {
        put(weights, statics, scope, layer + norm + ".weight", HIDDEN);
        put(weights, statics, scope, layer + norm + ".bias", HIDDEN);
      }
      put(weights, statics, scope, layer + "mlp.0.weight", INTERMEDIATE, HIDDEN);
      put(weights, statics, scope, layer + "mlp.0.bias", INTERMEDIATE);
      put(weights, statics, scope, layer + "mlp.2.weight", HIDDEN, INTERMEDIATE);
      put(weights, statics, scope, layer + "mlp.2.bias", HIDDEN);
    }
    put(weights, statics, scope, "merger.norm.weight", HIDDEN);
    put(weights, statics, scope, "merger.norm.bias", HIDDEN);
    put(weights, statics, scope, "merger.linearFc1.weight", MERGE_BLOCK, MERGE_BLOCK);
    put(weights, statics, scope, "merger.linearFc1.bias", MERGE_BLOCK);
    put(weights, statics, scope, "merger.linearFc2.weight", OUT_HIDDEN, MERGE_BLOCK);
    put(weights, statics, scope, "merger.linearFc2.bias", OUT_HIDDEN);
    put(weights, statics, scope, "deepstackMerger0.norm.weight", MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.norm.bias", MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.linearFc1.weight", MERGE_BLOCK, MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.linearFc1.bias", MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.linearFc2.weight", OUT_HIDDEN, MERGE_BLOCK);
    put(weights, statics, scope, "deepstackMerger0.linearFc2.bias", OUT_HIDDEN);
    return new Qwen3VlVisionTower(scope, NUM_HEADS, weights, DEEPSTACK_INDEXES);
  }

  private static void put(
      Map<String, MLXArray> weights,
      List<MLXArray> statics,
      MLXScope scope,
      String key,
      int... shape) {
    weights.put(key, full(statics, scope, shape));
  }

  private static MLXArray full(List<MLXArray> statics, MLXScope scope, int... shape) {
    MLXArray value = MLX.full(scope, shape, WEIGHT, DType.FLOAT32);
    statics.add(value);
    return value;
  }

  /** The 2B artifact's slow-processor options, with maxPixels at the default 4096-row hard cap. */
  private static Qwen3VlImageProcessor processor() {
    return new Qwen3VlImageProcessor(
        Qwen3VlProcessorConfig.builder()
            .patchSize(16)
            .temporalPatchSize(2)
            .mergeSize(2)
            .minPixels(3136)
            .maxPixels(4096 * 16 * 16)
            .imageMean(new float[] {0.5f, 0.5f, 0.5f})
            .imageStd(new float[] {0.5f, 0.5f, 0.5f})
            .build());
  }

  /** One deterministic 1024×1024 image: per-pixel linear functions of the coordinates. */
  private static RgbImage syntheticImage() {
    byte[] pixels = new byte[IMAGE * IMAGE * 3];
    for (int y = 0; y < IMAGE; y++) {
      for (int x = 0; x < IMAGE; x++) {
        int i = (y * IMAGE + x) * 3;
        pixels[i] = (byte) ((x * 7 + y * 11) % 256);
        pixels[i + 1] = (byte) ((x * 13 + y * 5) % 256);
        pixels[i + 2] = (byte) ((x * 19 + y * 3) % 256);
      }
    }
    return new RgbImage(IMAGE, IMAGE, pixels);
  }
}
