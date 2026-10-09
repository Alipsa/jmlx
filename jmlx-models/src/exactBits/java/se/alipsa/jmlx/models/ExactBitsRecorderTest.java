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
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.ffi.mlx_h;
import tools.jackson.databind.JsonNode;

/**
 * Opt-in exact-bit recorder/comparator for the WP5 decoder-embedding refactor.
 *
 * <p>Driven by {@code jmlx.exact.bits.mode} ({@code record} | {@code verify}) and the {@code
 * MLX_ENABLE_TF32} environment (1 = default reduced precision, 0 = full float32); the four {@code
 * exactBits{Record,Verify}Tf32{On,Off}} tasks set both and fork their own JVMs. Deliberately NOT
 * {@code @EnabledIfNativeAvailable}: as an opt-in baseline tool it fails loudly when the native
 * runtime is absent instead of silently skipping. Recordings are PR evidence under {@code
 * build/exact-bits}, never committed goldens:
 *
 * <ul>
 *   <li>{@code tf32-<0|1>/<variant>/<capture>.json} -- one file per capture. Float captures carry
 *       {@code values}, the Base64 of the big-endian IEEE-754 bits of every float32 in row-major
 *       order ({@code Float.floatToIntBits}); the greedy capture carries {@code generatedTokenIds};
 *       the scheduler capture carries per-row {@code tokenIds} and {@code finishReason}.
 *   <li>{@code tf32-<0|1>/manifest.json} -- host metadata plus every variant's applicability, input
 *       SHA-256 hashes, derived quantized checkpoint SHA-256 hashes and capture list.
 * </ul>
 *
 * <p>Verify mode recomputes everything and requires bit-exact agreement of every float and ID, and
 * metadata equality in every field except the git commit (the comparison crosses the refactor's own
 * commits); anything else that differs means the recordings are not comparable on this host.
 */
class ExactBitsRecorderTest {

  static final int FORMAT_VERSION = 1;

  @Test
  void recordOrVerify() throws Exception {
    String mode = System.getProperty("jmlx.exact.bits.mode", "");
    if (!mode.equals("record") && !mode.equals("verify")) {
      throw new AssertionError(
          "jmlx.exact.bits.mode must be 'record' or 'verify'; run one of the "
              + "exactBitsRecord/exactBitsVerify Gradle tasks, not this test directly");
    }
    String tf32 = System.getenv("MLX_ENABLE_TF32");
    if (!"0".equals(tf32) && !"1".equals(tf32)) {
      throw new AssertionError("MLX_ENABLE_TF32 must be set to 0 or 1 by the Gradle task");
    }
    Path repoRoot = Path.of(System.getProperty("jmlx.repository.root", "."));
    requireNativeRuntime(repoRoot);
    Path modeDir =
        Path.of(System.getProperty("jmlx.exact.bits.dir", "build/exact-bits"))
            .resolve("tf32-" + tf32);
    Path goldens = repoRoot.resolve("tools/hf-reference/goldens");

    List<ExactBitsCapture.Variant> variants = captureAll(goldens, modeDir);
    ExactBitsMetadata global = metadata(repoRoot);
    if (mode.equals("record")) {
      record(global, variants, modeDir);
    } else {
      verify(global, variants, modeDir);
    }
    System.out.println("exact-bits " + mode + " tf32=" + tf32 + ": " + summary(variants));
  }

  /** Every family, then its quantized variant, in the spec's deterministic order. */
  private static List<ExactBitsCapture.Variant> captureAll(Path goldens, Path modeDir)
      throws Exception {
    List<ExactBitsCapture.Variant> variants = new ArrayList<>();
    for (String family : ExactBitsSpec.FAMILIES) {
      Path checkpoint = goldens.resolve("checkpoints").resolve(family);
      Path goldenFile = goldens.resolve(family + ".json");
      JsonNode golden = JsonFiles.read(goldenFile);
      variants.add(ExactBitsCapture.capture(family, family, checkpoint, goldenFile, golden));
      if (ExactBitsSpec.QUANTIZED_FAMILIES.contains(family)) {
        Map<String, String> derived = new TreeMap<>();
        Path derivedDir =
            ExactBitsCapture.deriveQuantized(
                checkpoint, modeDir.resolve("derived").resolve(family + "-q4"), derived);
        ExactBitsCapture.Variant quantized =
            ExactBitsCapture.capture(family + "-q4", family, derivedDir, goldenFile, golden);
        quantized.derivedHashes.putAll(derived);
        variants.add(quantized);
      }
    }
    return variants;
  }

  private static ExactBitsMetadata metadata(Path repoRoot) throws Exception {
    String commit = gitRevParse(repoRoot);
    Properties pins = new Properties();
    try (InputStream in =
        Files.newInputStream(repoRoot.resolve("native/install/lib/native-pin.properties"))) {
      pins.load(in);
    }
    return new ExactBitsMetadata(
        commit,
        pins.getProperty("mlxMetalVersion"),
        pins.getProperty("mlxcCommit"),
        deviceName(),
        System.getProperty("os.name"),
        System.getProperty("os.version"),
        System.getenv("MLX_ENABLE_TF32"),
        ExactBitsSpec.sha256(),
        Map.of(),
        Map.of());
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

  /** The MLX default device as a stable string: the same query MLXGpuVerificationTest asserts. */
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

  private static void record(
      ExactBitsMetadata global, List<ExactBitsCapture.Variant> variants, Path modeDir)
      throws IOException {
    Files.createDirectories(modeDir);
    for (ExactBitsCapture.Variant variant : variants) {
      if (!variant.applicable) {
        System.out.println("not applicable: " + variant.name + " (" + variant.reason + ")");
        continue;
      }
      Path dir = Files.createDirectories(modeDir.resolve(variant.name));
      int floats = 0;
      for (Map.Entry<String, float[]> capture : variant.floatCaptures.entrySet()) {
        floatBits(global, dir, variant, capture.getKey(), capture.getValue());
        floats += capture.getValue().length;
      }
      writeGreedy(global, dir, variant);
      writeScheduler(global, dir, variant);
      System.out.println(
          "recorded "
              + variant.name
              + ": "
              + captureCount(variant)
              + " captures, "
              + floats
              + " floats");
    }
    writeManifest(global, variants, modeDir);
  }

  private static void verify(
      ExactBitsMetadata global, List<ExactBitsCapture.Variant> variants, Path modeDir)
      throws IOException {
    Path manifestPath = modeDir.resolve("manifest.json");
    if (!Files.isRegularFile(manifestPath)) {
      throw new AssertionError(
          "no exact-bit recordings under "
              + modeDir
              + " -- record first: ./gradlew :jmlx-models:exactBitsRecord");
    }
    JsonNode manifest = JsonFiles.read(manifestPath);
    if (manifest.path("formatVersion").asInt() != FORMAT_VERSION) {
      throw new AssertionError(
          "unsupported recording format version " + manifest.path("formatVersion").asInt());
    }
    List<String> mismatches =
        global.mismatches(ExactBitsMetadata.fromJson(manifest.path("metadata")), false);
    if (!mismatches.isEmpty()) {
      throw new AssertionError(notComparable(mismatches));
    }
    Map<String, JsonNode> recordedVariants = new LinkedHashMap<>();
    for (JsonNode node : manifest.path("variants")) {
      recordedVariants.put(node.path("name").asString(), node);
    }
    List<String> names = variants.stream().map(v -> v.name).toList();
    if (!names.equals(new ArrayList<>(recordedVariants.keySet()))) {
      throw new AssertionError(
          "variant set differs from the recording: recorded "
              + recordedVariants.keySet()
              + ", candidate "
              + names
              + " -- re-record after recording the spec at the base commit");
    }
    for (ExactBitsCapture.Variant variant : variants) {
      JsonNode recordedVariant = recordedVariants.get(variant.name);
      if (variant.applicable != recordedVariant.path("applicable").asBoolean(true)) {
        throw new AssertionError(
            variant.name
                + " applicability differs: recorded "
                + recordedVariant.path("applicable").asBoolean(true)
                + " (reason: "
                + recordedVariant.path("reason").asString()
                + "), candidate "
                + variant.applicable
                + " (reason: "
                + variant.reason
                + ")");
      }
      if (!variant.applicable) {
        continue;
      }
      if (!variant.derivedHashes.equals(mapOf(recordedVariant.path("derivedHashes")))) {
        throw new AssertionError(
            "quantized checkpoint derivation for "
                + variant.name
                + " is not reproducible: recorded "
                + variant.derivedHashes
                + ", candidate "
                + mapOf(recordedVariant.path("derivedHashes")));
      }
      List<String> recordedCaptures = stringList(recordedVariant.path("captures"));
      List<String> candidateCaptures = captureNames(variant);
      if (!candidateCaptures.equals(recordedCaptures)) {
        throw new AssertionError(
            variant.name
                + " capture set differs: recorded "
                + recordedCaptures
                + ", candidate "
                + candidateCaptures);
      }
      for (String capture : recordedCaptures) {
        verifyCapture(global, variant, capture, modeDir);
      }
    }
    System.out.println("verified " + variants.size() + " variants against " + manifestPath);
  }

  private static void verifyCapture(
      ExactBitsMetadata global, ExactBitsCapture.Variant variant, String capture, Path modeDir)
      throws IOException {
    Path file = modeDir.resolve(variant.name).resolve(capture + ".json");
    if (!Files.isRegularFile(file)) {
      throw new AssertionError("missing recording " + file + " -- record first");
    }
    JsonNode node = JsonFiles.read(file);
    List<String> mismatches =
        metadataFor(global, variant)
            .mismatches(ExactBitsMetadata.fromJson(node.path("metadata")), true);
    if (!mismatches.isEmpty()) {
      throw new AssertionError(
          notComparable(mismatches) + " [" + variant.name + " / " + capture + "]");
    }
    if (variant.floatCaptures.containsKey(capture)) {
      float[] actual = variant.floatCaptures.get(capture);
      int[] expected = floatBits(node.path("values").asString());
      int[] actualBits = new int[actual.length];
      for (int i = 0; i < actual.length; i++) {
        actualBits[i] = Float.floatToIntBits(actual[i]);
      }
      if (expected.length != actualBits.length) {
        throw new AssertionError(
            mismatch(
                variant.name,
                capture,
                0,
                "lengths differ: recorded "
                    + expected.length
                    + " floats, candidate "
                    + actual.length));
      }
      for (int i = 0; i < expected.length; i++) {
        if (expected[i] != actualBits[i]) {
          throw new AssertionError(
              mismatch(
                  variant.name,
                  capture,
                  i,
                  "expected bits 0x"
                      + Integer.toHexString(expected[i])
                      + " ("
                      + Float.intBitsToFloat(expected[i])
                      + "), actual bits 0x"
                      + Integer.toHexString(actualBits[i])
                      + " ("
                      + Float.intBitsToFloat(actualBits[i])
                      + ")"));
        }
      }
      return;
    }
    if ("greedy".equals(capture)) {
      List<Integer> expected = ints(node.path("generatedTokenIds"));
      compareIds(variant.name, capture, expected, variant.greedyIds);
      return;
    }
    if ("scheduler".equals(capture)) {
      List<JsonNode> recordedRows = new ArrayList<>();
      for (JsonNode row : node.path("rows")) {
        recordedRows.add(row);
      }
      List<ExactBitsCapture.SchedulerRow> actual = variant.schedulerRows;
      if (recordedRows.size() != actual.size()) {
        throw new AssertionError(
            mismatch(
                variant.name,
                capture,
                0,
                "row count differs: recorded "
                    + recordedRows.size()
                    + ", candidate "
                    + actual.size()));
      }
      for (int row = 0; row < recordedRows.size(); row++) {
        compareIds(
            variant.name,
            capture + " row " + row,
            ints(recordedRows.get(row).path("tokenIds")),
            actual.get(row).tokenIds());
        String reason = recordedRows.get(row).path("finishReason").asString();
        if (!reason.equals(actual.get(row).finishReason())) {
          throw new AssertionError(
              mismatch(
                  variant.name,
                  capture + " row " + row,
                  0,
                  "finishReason recorded "
                      + reason
                      + ", candidate "
                      + actual.get(row).finishReason()));
        }
      }
      return;
    }
    throw new AssertionError("unknown capture " + capture + " for " + variant.name);
  }

  private static String notComparable(List<String> fields) {
    return "recordings are not comparable on this host; re-record: metadata mismatch in " + fields;
  }

  private static String mismatch(String variant, String capture, int index, String detail) {
    return "exact-bit mismatch ["
        + variant
        + " / "
        + capture
        + "]: first divergent element "
        + index
        + "; "
        + detail;
  }

  private static void compareIds(
      String variant, String capture, List<Integer> expected, List<Integer> actual) {
    if (expected.size() != actual.size()) {
      throw new AssertionError(
          mismatch(
              variant,
              capture,
              0,
              "ID list length recorded " + expected.size() + ", candidate " + actual.size()));
    }
    for (int i = 0; i < expected.size(); i++) {
      if (!expected.get(i).equals(actual.get(i))) {
        throw new AssertionError(
            mismatch(
                variant,
                capture,
                i,
                "expected ID " + expected.get(i) + ", actual ID " + actual.get(i)));
      }
    }
  }

  private static void writeGreedy(
      ExactBitsMetadata global, Path dir, ExactBitsCapture.Variant variant) throws IOException {
    StringBuilder ids = new StringBuilder("[");
    List<Integer> generated = variant.greedyIds;
    for (int i = 0; i < generated.size(); i++) {
      if (i > 0) {
        ids.append(',');
      }
      ids.append(generated.get(i));
    }
    Files.writeString(
        dir.resolve("greedy.json"),
        captureJson(
            variant.name,
            "greedy",
            metadataFor(global, variant),
            "\"generatedTokenIds\":" + ids + ']'));
  }

  private static void writeScheduler(
      ExactBitsMetadata global, Path dir, ExactBitsCapture.Variant variant) throws IOException {
    StringBuilder rows = new StringBuilder("[");
    List<ExactBitsCapture.SchedulerRow> recorded = variant.schedulerRows;
    for (int i = 0; i < recorded.size(); i++) {
      if (i > 0) {
        rows.append(',');
      }
      rows.append("{\"tokenIds\":[")
          .append(
              recorded.get(i).tokenIds().stream()
                  .map(Object::toString)
                  .reduce((a, b) -> a + "," + b)
                  .orElse(""))
          .append("],\"finishReason\":\"")
          .append(recorded.get(i).finishReason())
          .append("\"}");
    }
    Files.writeString(
        dir.resolve("scheduler.json"),
        captureJson(
            variant.name, "scheduler", metadataFor(global, variant), "\"rows\":" + rows + ']'));
  }

  private static void floatBits(
      ExactBitsMetadata global,
      Path dir,
      ExactBitsCapture.Variant variant,
      String capture,
      float[] values)
      throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4 * values.length);
    for (float value : values) {
      buffer.putFloat(value);
    }
    Files.writeString(
        dir.resolve(capture + ".json"),
        captureJson(
            variant.name,
            capture,
            metadataFor(global, variant),
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

  private static String captureJson(
      String variant, String capture, ExactBitsMetadata meta, String payload) {
    return "{\"formatVersion\":"
        + FORMAT_VERSION
        + ",\"variant\":\""
        + variant
        + "\",\"capture\":\""
        + capture
        + "\",\"metadata\":"
        + meta.toJsonString()
        + ","
        + payload
        + "}\n";
  }

  private static void writeManifest(
      ExactBitsMetadata global, List<ExactBitsCapture.Variant> variants, Path modeDir)
      throws IOException {
    StringBuilder out = new StringBuilder();
    out.append("{\"formatVersion\":").append(FORMAT_VERSION).append(',');
    out.append("\"metadata\":").append(global.toJsonString()).append(',');
    out.append("\"variants\":[");
    for (int i = 0; i < variants.size(); i++) {
      ExactBitsCapture.Variant variant = variants.get(i);
      if (i > 0) {
        out.append(',');
      }
      out.append("{\"name\":\"").append(variant.name).append('"');
      out.append(",\"applicable\":").append(variant.applicable);
      out.append(",\"reason\":\"").append(variant.reason).append('"');
      out.append(",\"inputHashes\":").append(mapToJson(variant.inputHashes));
      out.append(",\"derivedHashes\":").append(mapToJson(variant.derivedHashes));
      out.append(",\"captures\":[");
      List<String> captures = captureNames(variant);
      for (int j = 0; j < captures.size(); j++) {
        if (j > 0) {
          out.append(',');
        }
        out.append('"').append(captures.get(j)).append('"');
      }
      out.append("]}");
    }
    out.append("]}\n");
    Files.writeString(modeDir.resolve("manifest.json"), out.toString());
  }

  private static ExactBitsMetadata metadataFor(
      ExactBitsMetadata global, ExactBitsCapture.Variant variant) {
    return new ExactBitsMetadata(
        global.gitCommit,
        global.mlxMetalVersion,
        global.mlxcCommit,
        global.device,
        global.osName,
        global.osVersion,
        global.mlxEnableTf32,
        global.specHash,
        variant.inputHashes,
        variant.derivedHashes);
  }

  /** Capture names in the recorded order: float captures, then greedy, then scheduler. */
  private static List<String> captureNames(ExactBitsCapture.Variant variant) {
    List<String> out = new ArrayList<>(variant.floatCaptures.keySet());
    if (variant.applicable) {
      out.add("greedy");
      out.add("scheduler");
    }
    return out;
  }

  private static int captureCount(ExactBitsCapture.Variant variant) {
    return variant.floatCaptures.size() + 2;
  }

  private static String summary(List<ExactBitsCapture.Variant> variants) {
    int applicable = 0;
    int captures = 0;
    List<String> notApplicable = new ArrayList<>();
    for (ExactBitsCapture.Variant variant : variants) {
      if (variant.applicable) {
        applicable++;
        captures += captureCount(variant);
      } else {
        notApplicable.add(variant.name);
      }
    }
    return applicable
        + " applicable variants, "
        + captures
        + " captures"
        + (notApplicable.isEmpty() ? "" : ", not applicable: " + notApplicable);
  }

  private static String mapToJson(Map<String, String> map) {
    StringBuilder out = new StringBuilder("{");
    int i = 0;
    for (Map.Entry<String, String> entry : map.entrySet()) {
      if (i++ > 0) {
        out.append(',');
      }
      out.append('"').append(entry.getKey()).append("\":\"").append(entry.getValue()).append('"');
    }
    return out.append('}').toString();
  }

  private static Map<String, String> mapOf(JsonNode node) {
    Map<String, String> out = new TreeMap<>();
    if (node.isObject()) {
      node.properties().forEach(e -> out.put(e.getKey(), e.getValue().asString()));
    }
    return out;
  }

  private static List<String> stringList(JsonNode node) {
    List<String> out = new ArrayList<>();
    for (JsonNode child : node) {
      out.add(child.asString());
    }
    return out;
  }

  private static List<Integer> ints(JsonNode node) {
    List<Integer> out = new ArrayList<>();
    for (JsonNode child : node) {
      out.add(child.asInt());
    }
    return out;
  }
}
