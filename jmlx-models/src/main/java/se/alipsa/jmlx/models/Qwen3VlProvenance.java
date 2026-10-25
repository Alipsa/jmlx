package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import tools.jackson.databind.JsonNode;

/**
 * Qwen3-VL checkpoint provenance, selected from the shard headers' {@code __metadata__} blocks and
 * the config's quantization declarations before any tensor is loaded (findings §8).
 *
 * <p>The two layouts this loader family distinguishes: HF float safetensors (no {@code format}
 * metadata and no quantization declaration) and the MLX-affine-quantized layout the pinned {@code
 * mlx-community/Qwen3-VL-8B-Instruct-4bit} artifact carries (shard metadata {@code format: mlx}
 * plus a config quantization block of exactly {@code {bits: 4, group_size: positive integer, mode:
 * "affine"}} under {@code quantization} or {@code quantization_config}; when both keys are present
 * they must be JSON-equal). Every other combination is rejected by name, so a checkpoint can never
 * silently select the wrong tensor plan.
 *
 * <p>Both plans load: an MLX-affine checkpoint is recognized (the shard metadata plus a conforming
 * config quantization block) and its group size and bit width are returned on the {@link Resolved}
 * result so the loader can select the packed tensor plan; an HF float checkpoint resolves to {@link
 * Kind#HF_FLOAT} with no group or bits. Every other combination is rejected by name, so a
 * checkpoint can never silently select the wrong tensor plan.
 */
final class Qwen3VlProvenance {

  /** The checkpoint layouts the selector recognizes. */
  enum Kind {
    HF_FLOAT,
    MLX_AFFINE
  }

  private Qwen3VlProvenance() {}

  /**
   * Selects the checkpoint's tensor layout.
   *
   * @param directory the checkpoint directory (its shards are discovered, not loaded)
   * @param configTree the already-read {@code config.json} tree (the quantization declarations are
   *     read from it; the strict config parse must not see the quantization keys, which sit outside
   *     the pinned ten-key outer set)
   * @return the recognized layout: {@link Kind#HF_FLOAT} with null group and bits, or {@link
   *     Kind#MLX_AFFINE} with the config's group size and bit width
   * @throws IllegalArgumentException for any other format/quantization combination, naming the
   *     exact mismatch
   * @throws IOException if a shard header cannot be read
   */
  static Resolved resolve(Path directory, JsonNode configTree) throws IOException {
    Objects.requireNonNull(directory, "directory");
    Objects.requireNonNull(configTree, "configTree");
    Map<String, String> metadata = SafetensorsHeaders.metadata(CheckpointLoader.shards(directory));
    String format = metadata.get("format");
    Declaration quantization = quantization(configTree);
    if (format == null) {
      if (quantization == null) {
        return new Resolved(Kind.HF_FLOAT, null, null);
      }
      throw new IllegalArgumentException(
          "config.json declares a quantization block but the checkpoint shards carry no format"
              + " metadata: unknown Qwen3-VL checkpoint provenance");
    }
    if (!"mlx".equals(format)) {
      throw new IllegalArgumentException(
          "unsupported Qwen3-VL checkpoint format '" + format + "' in the shard metadata");
    }
    if (quantization == null) {
      throw new IllegalArgumentException(
          "shard metadata format=mlx requires a config quantization block of exactly"
              + " {bits: 4, group_size: positive integer, mode: \"affine\"}, but none is declared");
    }
    requireAffine(quantization.key(), quantization.block());
    return new Resolved(
        Kind.MLX_AFFINE,
        quantization.block().get("group_size").intValue(),
        quantization.block().get("bits").intValue());
  }

  /**
   * The resolved checkpoint layout: the recognized {@link Kind} plus, for {@link Kind#MLX_AFFINE}
   * only, the config's quantization {@code group_size} and {@code bits} (both non-null and already
   * validated by {@link #resolve}). For {@link Kind#HF_FLOAT} the group size and bits are null,
   * since the float layout declares no quantization.
   */
  record Resolved(Kind kind, Integer groupSize, Integer bits) {}

  /**
   * The config's quantization declaration, if any: {@code quantization} wins when present, else
   * {@code quantization_config}; both present must be JSON-equal. The record carries the winning
   * key so rejection messages name the right field. Null when neither is declared.
   */
  private record Declaration(String key, JsonNode block) {}

  private static Declaration quantization(JsonNode configTree) {
    JsonNode block = configTree.hasNonNull("quantization") ? configTree.get("quantization") : null;
    JsonNode alias =
        configTree.hasNonNull("quantization_config") ? configTree.get("quantization_config") : null;
    if (block == null && alias == null) {
      return null;
    }
    if (block != null && alias != null && !block.equals(alias)) {
      throw new IllegalArgumentException(
          "config.json quantization and quantization_config disagree");
    }
    return block != null
        ? new Declaration("quantization", block)
        : new Declaration("quantization_config", alias);
  }

  /**
   * The MLX-affine block contract: exactly {@code bits: 4}, a positive integer group size, affine.
   */
  private static void requireAffine(String key, JsonNode quant) {
    if (!quant.isObject()) {
      throw new IllegalArgumentException("config.json " + key + " must be an object");
    }
    if (quant.size() != 3 || !quant.has("bits") || !quant.has("group_size") || !quant.has("mode")) {
      throw new IllegalArgumentException(
          "config.json "
              + key
              + " must hold exactly bits, group_size and mode, got "
              + java.util.List.copyOf(quant.propertyNames()));
    }
    JsonNode bits = quant.get("bits");
    if (!bits.isIntegralNumber() || !bits.canConvertToInt() || bits.intValue() != 4) {
      throw new IllegalArgumentException("config.json " + key + ".bits must be 4");
    }
    JsonNode groupSize = quant.get("group_size");
    if (!groupSize.isIntegralNumber()
        || !groupSize.canConvertToInt()
        || groupSize.intValue() <= 0) {
      throw new IllegalArgumentException(
          "config.json " + key + ".group_size must be a positive integer");
    }
    if (!"affine".equals(quant.path("mode").asString())) {
      throw new IllegalArgumentException(
          "config.json " + key + ".mode must be affine, got " + quant.path("mode"));
    }
  }
}
