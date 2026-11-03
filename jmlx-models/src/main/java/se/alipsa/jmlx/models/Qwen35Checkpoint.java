package se.alipsa.jmlx.models;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import tools.jackson.databind.JsonNode;

/** Complete native-free checkpoint classification and tensor geometry preflight. */
final class Qwen35Checkpoint {
  private static final Pattern MTP = Pattern.compile("(^|\\.)mtp(\\.|$)");
  private static final Set<String> FLOATS = Set.of("F32", "F16", "BF16");

  /** Classification is structural; it never claims to identify which converter wrote values. */
  record Checked(
      Qwen35Descriptor descriptor,
      boolean converted,
      boolean imageCapable,
      List<Path> shards,
      Map<String, String> names,
      Set<String> shiftedNorms,
      Set<String> transposedConvolutions,
      boolean transposePatch,
      Map<String, SafetensorsHeaders.TensorInfo> inventory) {
    Checked {
      shards = List.copyOf(shards);
      names = Map.copyOf(names);
      shiftedNorms = Set.copyOf(shiftedNorms);
      transposedConvolutions = Set.copyOf(transposedConvolutions);
      inventory = Map.copyOf(inventory);
    }
  }

  private record Expected(List<Integer> shape, boolean packable, boolean shifted) {
    Expected {
      shape = List.copyOf(shape);
    }
  }

  private Qwen35Checkpoint() {}

  static Checked inspect(Path directory, JsonNode config) throws IOException {
    List<Path> shards = CheckpointLoader.shards(directory);
    Map<String, SafetensorsHeaders.TensorInfo> inventory = SafetensorsHeaders.inventory(shards);
    Checked checked = classify(config, inventory);
    // Reuse index-to-shard ownership checks after complete type/shape/name validation.
    CheckpointLoader.preflight(
        directory, new TensorPlan(inventory.keySet(), Set.of(), Set.of(), Set.of()));
    return checked;
  }

  /**
   * Classifies an already byte-validated complete inventory; also checks committed header records.
   */
  static Checked classify(JsonNode config, Map<String, SafetensorsHeaders.TensorInfo> inventory) {
    final List<Path> shards =
        inventory.values().stream()
            .map(SafetensorsHeaders.TensorInfo::shard)
            .distinct()
            .sorted()
            .toList();
    boolean hf = inventory.keySet().stream().anyMatch(k -> k.startsWith("model.language_model."));
    boolean converted =
        inventory.keySet().stream().anyMatch(k -> k.startsWith("language_model.model."));
    if (hf == converted) {
      throw invalid("mixed or absent text prefixes");
    }
    if (converted && inventory.keySet().stream().anyMatch(k -> MTP.matcher(k).find())) {
      throw invalid("converted text prefix with MTP subtree");
    }
    Qwen35Descriptor descriptor = Qwen35Descriptor.parse(config, converted);
    Map<String, String> names = new LinkedHashMap<>();
    Set<String> skipped = new HashSet<>();
    for (String raw : inventory.keySet()) {
      String canonical;
      if (raw.startsWith(hf ? "model.language_model." : "language_model.model.")) {
        canonical =
            "text."
                + raw.substring((hf ? "model.language_model." : "language_model.model.").length());
      } else if (raw.startsWith(hf ? "model.visual." : "vision_tower.")) {
        canonical = "vision." + raw.substring(13);
      } else if (raw.startsWith(hf ? "lm_head." : "language_model.lm_head.")) {
        canonical =
            "lm_head." + raw.substring((hf ? "lm_head." : "language_model.lm_head.").length());
      } else if (hf && raw.startsWith("mtp.")) {
        skipped.add(raw);
        continue;
      } else {
        throw invalid("unexpected tensor '" + raw + "'");
      }
      if (names.putIfAbsent(canonical, raw) != null) {
        throw invalid("colliding remap to '" + canonical + "'");
      }
    }
    boolean vision = names.keySet().stream().anyMatch(k -> k.startsWith("vision."));
    if (vision && descriptor.vision() == null) {
      throw invalid("vision tensors without vision_config");
    }
    Map<String, Expected> expected = expected(descriptor, vision);
    Set<String> optional = new HashSet<>();
    if (descriptor.text().tied()) {
      optional.add("lm_head.weight");
      if (descriptor.affine()) {
        optional.add("lm_head.scales");
        optional.add("lm_head.biases");
      }
    }
    Set<String> allowed = new HashSet<>(expected.keySet());
    allowed.addAll(optional);
    // Sidecars are required only for matrices whose actual header carries packed U32 weights.
    for (var entry : expected.entrySet()) {
      if (entry.getValue().packable()) {
        String base = entry.getKey().substring(0, entry.getKey().length() - 6);
        allowed.add(base + "scales");
        allowed.add(base + "biases");
      }
    }
    Set<String> required = new HashSet<>(expected.keySet());
    required.removeAll(optional);
    new TensorPlan(required, difference(allowed, required), Set.of(), Set.of())
        .validate(names.keySet());
    if (!names.containsKey("lm_head.weight")
        && (names.containsKey("lm_head.scales") || names.containsKey("lm_head.biases"))) {
      throw invalid("lm_head sidecars without weight");
    }
    Set<String> shiftedNorms = new HashSet<>();
    Set<String> transposedConvolutions = new HashSet<>();
    boolean transposePatch = false;
    for (var entry : expected.entrySet()) {
      String canonical = entry.getKey();
      String raw = names.get(canonical);
      if (raw == null && optional.contains(canonical)) {
        continue;
      }
      SafetensorsHeaders.TensorInfo actual = inventory.get(raw);
      Expected spec = entry.getValue();
      if (canonical.endsWith("linear_attn.conv1d.weight")) {
        List<Integer> shape = spec.shape();
        List<Integer> stored = hf ? List.of(shape.get(0), 1, shape.get(1)) : shape;
        require(actual, raw, stored, FLOATS);
        if (hf) {
          transposedConvolutions.add(canonical);
        }
      } else if (canonical.equals("vision.patch_embed.proj.weight")) {
        var v = descriptor.vision();
        List<Integer> channelsFirst =
            List.of(v.hidden(), 3, v.temporalPatch(), v.patch(), v.patch());
        if (channelsFirst.equals(spec.shape())) {
          throw invalid("ambiguous patch axes for '" + raw + "'");
        }
        if (actual.shape().equals(channelsFirst) && !actual.shape().equals(spec.shape())) {
          transposePatch = true;
          require(actual, raw, channelsFirst, FLOATS);
        } else {
          require(actual, raw, spec.shape(), FLOATS);
        }
      } else if (spec.packable()) {
        matrix(descriptor.affine(), canonical, raw, spec.shape(), names, inventory);
      } else {
        require(actual, raw, spec.shape(), FLOATS);
      }
      if (hf && spec.shifted()) {
        shiftedNorms.add(canonical);
      }
    }
    validateMtp(descriptor.text(), skipped, inventory);
    return new Checked(
        descriptor,
        converted,
        vision,
        shards,
        names,
        shiftedNorms,
        transposedConvolutions,
        transposePatch,
        inventory);
  }

  private static Set<String> difference(Set<String> a, Set<String> b) {
    Set<String> result = new HashSet<>(a);
    result.removeAll(b);
    return result;
  }

  private static void matrix(
      boolean affine,
      String key,
      String raw,
      List<Integer> shape,
      Map<String, String> names,
      Map<String, SafetensorsHeaders.TensorInfo> inventory) {
    String base = key.substring(0, key.length() - 6);
    SafetensorsHeaders.TensorInfo actual = inventory.get(raw);
    String scales = names.get(base + "scales");
    String biases = names.get(base + "biases");
    if (actual.dtype().equals("U32")) {
      int rows = shape.get(0);
      int columns = shape.get(1);
      if (!affine || columns % 64 != 0 || scales == null || biases == null) {
        throw invalid("incomplete/unsupported affine matrix '" + raw + "'");
      }
      require(actual, raw, List.of(rows, columns / 8), Set.of("U32"));
      require(inventory.get(scales), scales, List.of(rows, columns / 64), FLOATS);
      require(inventory.get(biases), biases, List.of(rows, columns / 64), FLOATS);
    } else {
      require(actual, raw, shape, FLOATS);
      if (scales != null || biases != null) {
        throw invalid("float matrix with packed sidecars '" + raw + "'");
      }
    }
  }

  private static Map<String, Expected> expected(Qwen35Descriptor d, boolean vision) {
    Map<String, Expected> e = new LinkedHashMap<>();
    var t = d.text();
    put(e, "text.embed_tokens.weight", true, false, t.vocabulary(), t.hidden());
    put(e, "lm_head.weight", true, false, t.vocabulary(), t.hidden());
    put(e, "text.norm.weight", false, true, t.hidden());
    for (int i = 0; i < t.layers(); i++) {
      String p = "text.layers." + i + ".";
      block(e, t, p, t.linearLayers().get(i));
    }
    if (vision) {
      vision(e, d.vision());
    }
    return e;
  }

  private static void block(
      Map<String, Expected> e, Qwen35Descriptor.Text t, String p, boolean linear) {
    put(e, p + "input_layernorm.weight", false, true, t.hidden());
    put(e, p + "post_attention_layernorm.weight", false, true, t.hidden());
    put(e, p + "mlp.gate_proj.weight", true, false, t.intermediate(), t.hidden());
    put(e, p + "mlp.up_proj.weight", true, false, t.intermediate(), t.hidden());
    put(e, p + "mlp.down_proj.weight", true, false, t.hidden(), t.intermediate());
    if (linear) {
      p += "linear_attn.";
      int value = Qwen35Descriptor.product(t.valueHeads(), t.valueDim());
      put(e, p + "in_proj_qkv.weight", true, false, t.qkvWidth(), t.hidden());
      put(e, p + "in_proj_z.weight", true, false, value, t.hidden());
      put(e, p + "in_proj_b.weight", true, false, t.valueHeads(), t.hidden());
      put(e, p + "in_proj_a.weight", true, false, t.valueHeads(), t.hidden());
      put(e, p + "out_proj.weight", true, false, t.hidden(), value);
      put(e, p + "conv1d.weight", false, false, t.qkvWidth(), t.convolution(), 1);
      put(e, p + "A_log", false, false, t.valueHeads());
      put(e, p + "dt_bias", false, false, t.valueHeads());
      put(e, p + "norm.weight", false, false, t.valueDim());
    } else {
      p += "self_attn.";
      put(
          e,
          p + "q_proj.weight",
          true,
          false,
          Qwen35Descriptor.product(2, t.heads(), t.headDim()),
          t.hidden());
      put(
          e,
          p + "k_proj.weight",
          true,
          false,
          Qwen35Descriptor.product(t.kvHeads(), t.headDim()),
          t.hidden());
      put(
          e,
          p + "v_proj.weight",
          true,
          false,
          Qwen35Descriptor.product(t.kvHeads(), t.headDim()),
          t.hidden());
      put(
          e,
          p + "o_proj.weight",
          true,
          false,
          t.hidden(),
          Qwen35Descriptor.product(t.heads(), t.headDim()));
      put(e, p + "q_norm.weight", false, true, t.headDim());
      put(e, p + "k_norm.weight", false, true, t.headDim());
    }
  }

  private static void vision(Map<String, Expected> e, Qwen35Descriptor.Vision v) {
    String p = "vision.";
    // Shape-selected even in HF layout; quantization does not identify patch axes.
    put(
        e,
        p + "patch_embed.proj.weight",
        false,
        false,
        v.hidden(),
        v.temporalPatch(),
        v.patch(),
        v.patch(),
        3);
    put(e, p + "patch_embed.proj.bias", false, false, v.hidden());
    put(e, p + "pos_embed.weight", false, false, v.positions(), v.hidden());
    for (int i = 0; i < v.depth(); i++) {
      p = "vision.blocks." + i + ".";
      for (String norm : List.of("norm1", "norm2")) {
        put(e, p + norm + ".weight", false, false, v.hidden());
        put(e, p + norm + ".bias", false, false, v.hidden());
      }
      put(
          e,
          p + "attn.qkv.weight",
          false,
          false,
          Qwen35Descriptor.product(3, v.hidden()),
          v.hidden());
      put(e, p + "attn.qkv.bias", false, false, Qwen35Descriptor.product(3, v.hidden()));
      put(e, p + "attn.proj.weight", false, false, v.hidden(), v.hidden());
      put(e, p + "attn.proj.bias", false, false, v.hidden());
      put(e, p + "mlp.linear_fc1.weight", false, false, v.intermediate(), v.hidden());
      put(e, p + "mlp.linear_fc1.bias", false, false, v.intermediate());
      put(e, p + "mlp.linear_fc2.weight", false, false, v.hidden(), v.intermediate());
      put(e, p + "mlp.linear_fc2.bias", false, false, v.hidden());
    }
    p = "vision.merger.";
    int merged = Qwen35Descriptor.product(v.hidden(), v.merge(), v.merge());
    put(e, p + "norm.weight", false, false, v.hidden());
    put(e, p + "norm.bias", false, false, v.hidden());
    put(e, p + "linear_fc1.weight", false, false, merged, merged);
    put(e, p + "linear_fc1.bias", false, false, merged);
    put(e, p + "linear_fc2.weight", false, false, v.output(), merged);
    put(e, p + "linear_fc2.bias", false, false, v.output());
  }

  private static void validateMtp(
      Qwen35Descriptor.Text t,
      Set<String> keys,
      Map<String, SafetensorsHeaders.TensorInfo> inventory) {
    if (keys.isEmpty()) {
      return;
    }
    Map<String, Expected> expected = new HashMap<>();
    put(
        expected,
        "mtp.fc.weight",
        false,
        false,
        t.hidden(),
        Qwen35Descriptor.product(2, t.hidden()));
    for (String norm : List.of("norm", "pre_fc_norm_embedding", "pre_fc_norm_hidden")) {
      put(expected, "mtp." + norm + ".weight", false, false, t.hidden());
    }
    for (int i = 0; i < t.mtpLayers(); i++) {
      block(expected, t, "mtp.layers." + i + ".", false);
    }
    new TensorPlan(expected.keySet(), Set.of(), Set.of(), Set.of()).validate(keys);
    for (String key : keys) {
      require(inventory.get(key), key, expected.get(key).shape(), FLOATS);
    }
  }

  private static void put(
      Map<String, Expected> e, String key, boolean packable, boolean shifted, int... shape) {
    List<Integer> dimensions = new ArrayList<>();
    for (int dimension : shape) {
      dimensions.add(dimension);
    }
    e.put(key, new Expected(dimensions, packable, shifted));
  }

  private static void require(
      SafetensorsHeaders.TensorInfo actual, String key, List<Integer> shape, Set<String> dtypes) {
    if (actual == null || !actual.shape().equals(shape) || !dtypes.contains(actual.dtype())) {
      throw invalid(
          "invalid shape/dtype for '"
              + key
              + "': expected "
              + shape
              + " "
              + dtypes
              + ", got "
              + actual);
    }
  }

  private static IllegalArgumentException invalid(String reason) {
    return new IllegalArgumentException("Qwen3.5 checkpoint: " + reason);
  }
}
