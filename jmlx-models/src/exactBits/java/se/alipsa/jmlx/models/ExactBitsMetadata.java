package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import tools.jackson.databind.JsonNode;

/**
 * Host and runtime metadata of one exact-bit recording.
 *
 * <p>Every field except {@code gitCommit} must match exactly between a recording and a candidate
 * run, or the recordings are not comparable on that host: the commit is the only field allowed to
 * differ, because the point of the comparison is to cross the refactor's own commits. {@code
 * gitDirty} and {@code decoderModelSha256} are provenance only, like the commit: whether the tree
 * was dirty when the recording was made, and the SHA-256 of the DecoderModel source it was recorded
 * against. They are recorded but never compared, and recordings made before they existed omit them
 * (absent reads as {@code false} / {@code ""}).
 */
final class ExactBitsMetadata {

  final String gitCommit;
  final String mlxMetalVersion;
  final String mlxcCommit;
  final String device;
  final String osName;
  final String osVersion;
  final String mlxEnableTf32;
  final String specHash;
  final Map<String, String> inputHashes;
  final Map<String, String> derivedHashes;
  final boolean gitDirty;
  final String decoderModelSha256;

  ExactBitsMetadata(
      String gitCommit,
      String mlxMetalVersion,
      String mlxcCommit,
      String device,
      String osName,
      String osVersion,
      String mlxEnableTf32,
      String specHash,
      Map<String, String> inputHashes,
      Map<String, String> derivedHashes,
      boolean gitDirty,
      String decoderModelSha256) {
    this.gitCommit = gitCommit;
    this.mlxMetalVersion = mlxMetalVersion;
    this.mlxcCommit = mlxcCommit;
    this.device = device;
    this.osName = osName;
    this.osVersion = osVersion;
    this.mlxEnableTf32 = mlxEnableTf32;
    this.specHash = specHash;
    this.inputHashes = Map.copyOf(new TreeMap<>(inputHashes));
    this.derivedHashes = Map.copyOf(new TreeMap<>(derivedHashes));
    this.gitDirty = gitDirty;
    this.decoderModelSha256 = decoderModelSha256;
  }

  /** Renders the metadata object in fixed key order, escaping every string value for JSON. */
  String toJsonString() {
    StringBuilder out = new StringBuilder("{");
    out.append("\"gitCommit\":\"").append(escapeJson(gitCommit)).append("\",");
    out.append("\"mlxMetalVersion\":\"").append(escapeJson(mlxMetalVersion)).append("\",");
    out.append("\"mlxcCommit\":\"").append(escapeJson(mlxcCommit)).append("\",");
    out.append("\"device\":\"").append(escapeJson(device)).append("\",");
    out.append("\"osName\":\"").append(escapeJson(osName)).append("\",");
    out.append("\"osVersion\":\"").append(escapeJson(osVersion)).append("\",");
    out.append("\"mlxEnableTf32\":\"").append(escapeJson(mlxEnableTf32)).append("\",");
    out.append("\"specHash\":\"").append(escapeJson(specHash)).append("\",");
    out.append("\"inputHashes\":").append(mapToJson(inputHashes)).append(',');
    out.append("\"derivedHashes\":").append(mapToJson(derivedHashes)).append(',');
    out.append("\"gitDirty\":").append(gitDirty).append(',');
    out.append("\"decoderModelSha256\":\"").append(escapeJson(decoderModelSha256)).append('"');
    out.append('}');
    return out.toString();
  }

  static ExactBitsMetadata fromJson(JsonNode node) {
    Map<String, String> inputs = strings(node.path("inputHashes"));
    Map<String, String> derived = strings(node.path("derivedHashes"));
    // Provenance-only fields: recordings made before they existed omit them, so an absent value
    // reads as the documented default (false / ""). They are never compared, so the default can
    // never flip a comparison result.
    JsonNode gitDirty = node.path("gitDirty");
    JsonNode decoderModelSha256 = node.path("decoderModelSha256");
    return new ExactBitsMetadata(
        node.path("gitCommit").asString(),
        node.path("mlxMetalVersion").asString(),
        node.path("mlxcCommit").asString(),
        node.path("device").asString(),
        node.path("osName").asString(),
        node.path("osVersion").asString(),
        node.path("mlxEnableTf32").asString(),
        node.path("specHash").asString(),
        inputs,
        derived,
        gitDirty.isBoolean() && gitDirty.asBoolean(false),
        decoderModelSha256.isTextual() ? decoderModelSha256.asString() : "");
  }

  /**
   * Names every compared field on which this metadata disagrees with {@code recorded}. The git
   * commit is never compared, and neither are {@code gitDirty} and {@code decoderModelSha256}
   * (provenance only; recordings made before they existed omit them); the input/derived hash maps
   * are compared only when {@code includeHashMaps} is set, because the per-mode manifest stores
   * them per variant instead.
   */
  List<String> mismatches(ExactBitsMetadata recorded, boolean includeHashMaps) {
    List<String> out = new ArrayList<>();
    addIfDifferent(out, "mlxMetalVersion", mlxMetalVersion, recorded.mlxMetalVersion);
    addIfDifferent(out, "mlxcCommit", mlxcCommit, recorded.mlxcCommit);
    addIfDifferent(out, "device", device, recorded.device);
    addIfDifferent(out, "osName", osName, recorded.osName);
    addIfDifferent(out, "osVersion", osVersion, recorded.osVersion);
    addIfDifferent(out, "mlxEnableTf32", mlxEnableTf32, recorded.mlxEnableTf32);
    addIfDifferent(out, "specHash", specHash, recorded.specHash);
    if (includeHashMaps) {
      if (!inputHashes.equals(recorded.inputHashes)) {
        out.add("inputHashes");
      }
      if (!derivedHashes.equals(recorded.derivedHashes)) {
        out.add("derivedHashes");
      }
    }
    return out;
  }

  private static void addIfDifferent(
      List<String> out, String field, String actual, String expected) {
    if (actual == null ? expected != null : !actual.equals(expected)) {
      out.add(field);
    }
  }

  private static String mapToJson(Map<String, String> map) {
    StringBuilder out = new StringBuilder("{");
    int i = 0;
    for (Map.Entry<String, String> entry : map.entrySet()) {
      if (i++ > 0) {
        out.append(',');
      }
      out.append('"')
          .append(escapeJson(entry.getKey()))
          .append("\":\"")
          .append(escapeJson(entry.getValue()))
          .append('"');
    }
    return out.append('}').toString();
  }

  /**
   * Escapes a value for interpolation into a hand-built JSON string literal: backslash and double
   * quote get backslash escapes, tab/newline/carriage return/backspace/form feed use their standard
   * short escapes, and every other C0 control character U+0000-U+001F is written as a
   * backslash-{@code u} escape followed by four lowercase hex digits. A no-op for values without
   * such characters, so re-serialization of the recorded safe values stays byte-identical. A {@code
   * null} input is not accepted (it fails with a {@link NullPointerException}); a call site with a
   * nullable source (e.g. a properties read) must check for null where the nullness is meaningful
   * instead of relying on the escaper.
   */
  static String escapeJson(String value) {
    StringBuilder out = new StringBuilder(value.length());
    for (int i = 0; i < value.length(); i++) {
      char c = value.charAt(i);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\b' -> out.append("\\b");
        case '\f' -> out.append("\\f");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        default -> {
          if (c < 0x20) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    return out.toString();
  }

  private static Map<String, String> strings(JsonNode node) {
    Map<String, String> out = new TreeMap<>();
    if (node.isObject()) {
      node.properties().forEach(e -> out.put(e.getKey(), e.getValue().asString()));
    }
    return Collections.unmodifiableMap(out);
  }
}
