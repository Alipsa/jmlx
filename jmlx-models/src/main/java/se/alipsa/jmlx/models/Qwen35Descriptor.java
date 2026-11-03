package se.alipsa.jmlx.models;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import se.alipsa.jmlx.nn.MultiAxisRope;
import tools.jackson.databind.JsonNode;

/** Pure strict configuration for the dense Qwen3.5 hybrid decoder. */
record Qwen35Descriptor(Text text, Vision vision, Qwen3VlPromptTokens tokens, boolean affine) {

  /** Validated text dimensions; all projection and retained-state products fit signed integers. */
  record Text(
      int vocabulary,
      int hidden,
      int intermediate,
      int layers,
      int heads,
      int kvHeads,
      int headDim,
      int keyHeads,
      int valueHeads,
      int keyDim,
      int valueDim,
      int convolution,
      int context,
      float epsilon,
      float theta,
      int rotaryDim,
      List<Integer> sections,
      List<Boolean> linearLayers,
      boolean tied,
      int mtpLayers) {
    Text {
      sections = List.copyOf(sections);
      linearLayers = List.copyOf(linearLayers);
    }

    MultiAxisRope rope() {
      return new MultiAxisRope(theta, sections.stream().mapToInt(Integer::intValue).toArray());
    }

    int qkvWidth() {
      return sum(product(2, keyHeads, keyDim), product(valueHeads, valueDim));
    }
  }

  /** Vision geometry is optional for text-only converted artifacts. */
  record Vision(
      int depth,
      int hidden,
      int intermediate,
      int heads,
      int positions,
      int patch,
      int temporalPatch,
      int merge,
      int output) {}

  private static final Set<String> OUTER =
      Set.of(
          "model_type",
          "architectures",
          "text_config",
          "vision_config",
          "tie_word_embeddings",
          "image_token_id",
          "video_token_id",
          "vision_start_token_id",
          "vision_end_token_id",
          "quantization",
          "quantization_config",
          "generation_config",
          "language_model_only");
  private static final Set<String> TEXT =
      Set.of(
          "model_type",
          "hidden_size",
          "intermediate_size",
          "num_hidden_layers",
          "vocab_size",
          "num_attention_heads",
          "num_key_value_heads",
          "head_dim",
          "hidden_act",
          "attention_bias",
          "attention_dropout",
          "attn_output_gate",
          "output_gate_type",
          "max_position_embeddings",
          "rms_norm_eps",
          "rope_parameters",
          "partial_rotary_factor",
          "tie_word_embeddings",
          "linear_conv_kernel_dim",
          "linear_key_head_dim",
          "linear_value_head_dim",
          "linear_num_key_heads",
          "linear_num_value_heads",
          "layer_types",
          "full_attention_interval",
          "mlp_only_layers",
          "mtp_num_hidden_layers",
          "mtp_use_dedicated_embeddings",
          "mamba_ssm_dtype");
  private static final Set<String> VISION =
      Set.of(
          "model_type",
          "depth",
          "hidden_size",
          "intermediate_size",
          "num_heads",
          "num_position_embeddings",
          "patch_size",
          "temporal_patch_size",
          "spatial_merge_size",
          "in_channels",
          "out_hidden_size",
          "hidden_act",
          "deepstack_visual_indexes",
          "rope_parameters");

  /** Parses observed structural fields and typed inert serialization/generation metadata. */
  static Qwen35Descriptor parse(JsonNode root) {
    return parse(root, false);
  }

  /** Converted mlx-lm serialization uses an explicit {@code type} alias for {@code rope_type}. */
  static Qwen35Descriptor parse(JsonNode root, boolean converted) {
    known(root, OUTER, "config");
    equal(root, "model_type", "qwen3_5");
    JsonNode architectures = required(root, "architectures");
    if (!architectures.isArray()
        || architectures.size() != 1
        || !architectures.get(0).isString()
        || !architectures.get(0).asString().equals("Qwen3_5ForConditionalGeneration")) {
      throw invalid("architectures", "expected dense Qwen3_5ForConditionalGeneration");
    }
    optionalBoolean(root, "language_model_only", false);
    boolean tied = bool(root, "tie_word_embeddings");
    Text text = parseText(required(root, "text_config"), tied, converted);
    final Vision vision =
        root.has("vision_config") ? parseVision(root.get("vision_config"), text.hidden()) : null;
    final Qwen3VlPromptTokens tokens =
        new Qwen3VlPromptTokens(
            token(root, "image_token_id", text.vocabulary()),
            token(root, "video_token_id", text.vocabulary()),
            token(root, "vision_start_token_id", text.vocabulary()),
            token(root, "vision_end_token_id", text.vocabulary()));
    if (root.has("generation_config")) {
      known(root.get("generation_config"), Set.of(), "generation_config");
      metadata(root.get("generation_config"), text.vocabulary());
    }
    metadata(root, text.vocabulary());
    boolean affine = root.has("quantization") || root.has("quantization_config");
    for (String name : List.of("quantization", "quantization_config")) {
      if (root.has(name)) {
        JsonNode q = root.get(name);
        knownExact(q, Set.of("group_size", "bits", "mode"), name);
        if (positive(q, "group_size") != 64 || positive(q, "bits") != 4) {
          throw invalid(name, "only affine 4-bit/group-64 is supported");
        }
        equal(q, "mode", "affine");
      }
    }
    return new Qwen35Descriptor(text, vision, tokens, affine);
  }

  private static Text parseText(JsonNode t, boolean outerTied, boolean converted) {
    known(t, TEXT, "text_config");
    equal(t, "model_type", "qwen3_5_text");
    equal(t, "hidden_act", "silu");
    optionalBoolean(t, "attention_bias", false);
    optionalBoolean(t, "attn_output_gate", true);
    optionalBoolean(t, "use_cache", true);
    optionalBoolean(t, "mtp_use_dedicated_embeddings", false);
    optionalEqual(t, "output_gate_type", "swish");
    optionalEqual(t, "mamba_ssm_dtype", "float32");
    if (t.has("attention_dropout") && number(t, "attention_dropout") != 0) {
      throw invalid("attention_dropout", "must be zero");
    }
    if (t.has("mlp_only_layers")
        && (!t.get("mlp_only_layers").isArray() || !t.get("mlp_only_layers").isEmpty())) {
      throw invalid("mlp_only_layers", "must be an empty array");
    }
    boolean tied = bool(t, "tie_word_embeddings");
    if (tied != outerTied) {
      throw invalid("tie_word_embeddings", "outer and text values disagree");
    }
    final int vocabulary = positive(t, "vocab_size");
    final int hidden = positive(t, "hidden_size");
    final int intermediate = positive(t, "intermediate_size");
    final int layers = positive(t, "num_hidden_layers");
    int heads = positive(t, "num_attention_heads");
    int kvHeads = positive(t, "num_key_value_heads");
    final int headDim = positive(t, "head_dim");
    int keyHeads = positive(t, "linear_num_key_heads");
    int valueHeads = positive(t, "linear_num_value_heads");
    final int keyDim = positive(t, "linear_key_head_dim");
    final int valueDim = positive(t, "linear_value_head_dim");
    final int convolution = positive(t, "linear_conv_kernel_dim");
    final int context = positive(t, "max_position_embeddings");
    final float epsilon = positiveFloat(t, "rms_norm_eps");
    if (heads % kvHeads != 0 || valueHeads % keyHeads != 0) {
      throw invalid("num_attention_heads/linear_num_value_heads", "must divide by their key heads");
    }
    JsonNode rope = required(t, "rope_parameters");
    knownExact(
        rope,
        Set.of(
            "rope_type",
            "type",
            "rope_theta",
            "partial_rotary_factor",
            "mrope_section",
            "mrope_interleaved"),
        "rope_parameters");
    if (rope.has("type")) {
      if (!converted) {
        throw invalid("rope_parameters.type", "alias is supported only for converted checkpoints");
      }
      equal(rope, "type", "default");
    }
    if (rope.has("rope_type") || !converted || !rope.has("type")) {
      equal(rope, "rope_type", "default");
    }
    if (!bool(rope, "mrope_interleaved")) {
      throw invalid("mrope_interleaved", "must be true");
    }
    double factor = number(rope, "partial_rotary_factor");
    if (factor <= 0
        || factor > 1
        || (t.has("partial_rotary_factor") && number(t, "partial_rotary_factor") != factor)) {
      throw invalid(
          "partial_rotary_factor", "nested factor must be in (0,1] and agree with any alias");
    }
    double width = headDim * factor;
    if (width != Math.rint(width) || width < 2 || width > headDim || ((int) width) % 2 != 0) {
      throw invalid("partial_rotary_factor", "must give a positive even integral rotary width");
    }
    JsonNode sections = required(rope, "mrope_section");
    if (!sections.isArray() || sections.size() != 3) {
      throw invalid("mrope_section", "must contain three nonnegative integers");
    }
    List<Integer> sectionList = new ArrayList<>();
    long sectionSum = 0;
    for (JsonNode section : sections) {
      int value = nonnegative(section, "mrope_section");
      sectionList.add(value);
      sectionSum += value;
    }
    if (2 * sectionSum != (int) width) {
      throw invalid("mrope_section", "twice its sum must equal rotary width");
    }
    final List<Boolean> schedule = schedule(t, layers);
    final int mtp =
        t.has("mtp_num_hidden_layers")
            ? nonnegative(t.get("mtp_num_hidden_layers"), "mtp_num_hidden_layers")
            : 0;
    // Validate tensor/state products before any shape allocation or native upload.
    final int qkv = sum(product(2, keyHeads, keyDim), product(valueHeads, valueDim));
    product(vocabulary, hidden);
    product(hidden, intermediate);
    product(2, heads, headDim, hidden);
    product(kvHeads, headDim, hidden);
    product(qkv, hidden);
    product(qkv, convolution);
    product(valueHeads, keyDim, valueDim);
    metadata(t, vocabulary);
    return new Text(
        vocabulary,
        hidden,
        intermediate,
        layers,
        heads,
        kvHeads,
        headDim,
        keyHeads,
        valueHeads,
        keyDim,
        valueDim,
        convolution,
        context,
        epsilon,
        positiveFloat(rope, "rope_theta"),
        (int) width,
        sectionList,
        schedule,
        tied,
        mtp);
  }

  private static List<Boolean> schedule(JsonNode t, int layers) {
    int interval = t.has("full_attention_interval") ? positive(t, "full_attention_interval") : 0;
    JsonNode explicit = t.get("layer_types");
    if (explicit == null && interval == 0) {
      throw invalid("layer_types/full_attention_interval", "a layer schedule is required");
    }
    if (explicit != null && (!explicit.isArray() || explicit.size() != layers)) {
      throw invalid("layer_types", "length must equal num_hidden_layers");
    }
    List<Boolean> result = new ArrayList<>();
    for (int i = 0; i < layers; i++) {
      boolean linear = (i + 1) % Math.max(1, interval) != 0;
      if (explicit != null) {
        JsonNode type = explicit.get(i);
        if (!type.isString()
            || (!type.asString().equals("linear_attention")
                && !type.asString().equals("full_attention"))) {
          throw invalid("layer_types[" + i + "]", "unknown layer kind");
        }
        boolean selected = type.asString().equals("linear_attention");
        if (interval > 0 && selected != linear) {
          throw invalid("layer_types/full_attention_interval", "schedules disagree at layer " + i);
        }
        linear = selected;
      }
      result.add(linear);
    }
    return List.copyOf(result);
  }

  private static Vision parseVision(JsonNode v, int textHidden) {
    known(v, VISION, "vision_config");
    JsonNode type = required(v, "model_type");
    if (!type.isString() || !Set.of("qwen3_5_vision", "qwen3_5").contains(type.asString())) {
      throw invalid("vision_config.model_type", "unsupported vision family");
    }
    equal(v, "hidden_act", "gelu_pytorch_tanh");
    JsonNode levels = required(v, "deepstack_visual_indexes");
    if (!levels.isArray() || !levels.isEmpty()) {
      throw invalid("deepstack_visual_indexes", "Qwen3.5 requires no DeepStack levels");
    }
    final int depth = positive(v, "depth");
    final int hidden = positive(v, "hidden_size");
    final int intermediate = positive(v, "intermediate_size");
    int heads = positive(v, "num_heads");
    final int positions = positive(v, "num_position_embeddings");
    final int patch = positive(v, "patch_size");
    final int temporal = positive(v, "temporal_patch_size");
    final int merge = positive(v, "spatial_merge_size");
    final int output = positive(v, "out_hidden_size");
    int side = (int) Math.sqrt(positions);
    if (hidden % heads != 0
        || (hidden / heads) % 4 != 0
        || side * side != positions
        || output != textHidden
        || positive(v, "in_channels") != 3) {
      throw invalid("vision_config", "invalid heads/grid/output/channels geometry");
    }
    if (v.has("rope_parameters")) {
      JsonNode rope = v.get("rope_parameters");
      knownExact(rope, Set.of("rope_type", "rope_theta"), "vision_config.rope_parameters");
      equal(rope, "rope_type", "axial");
      if (number(rope, "rope_theta") != 10_000) {
        throw invalid("vision_config.rope_theta", "must equal 10000");
      }
    }
    final int merger = product(hidden, merge, merge);
    product(hidden, 3, temporal, patch, patch);
    product(hidden, intermediate);
    product(merger, merger);
    product(merger, output);
    return new Vision(
        depth, hidden, intermediate, heads, positions, patch, temporal, merge, output);
  }

  private static void known(JsonNode node, Set<String> structural, String path) {
    if (node == null || !node.isObject()) {
      throw invalid(path, "must be an object");
    }
    for (var entry : node.properties()) {
      if (!structural.contains(entry.getKey())) {
        inert(entry.getKey(), entry.getValue(), path);
      }
    }
  }

  private static void knownExact(JsonNode node, Set<String> keys, String path) {
    if (node == null || !node.isObject()) {
      throw invalid(path, "must be an object");
    }
    for (var entry : node.properties()) {
      if (!keys.contains(entry.getKey())) {
        throw invalid(path + "." + entry.getKey(), "unknown key");
      }
    }
  }

  private static void inert(String key, JsonNode value, String path) {
    boolean valid =
        switch (key) {
          case "dtype", "torch_dtype" ->
              value.isNull()
                  || (value.isString()
                      && Set.of("float32", "float16", "bfloat16").contains(value.asString()));
          case "transformers_version", "_name_or_path" -> value.isString();
          case "initializer_range",
              "temperature",
              "top_p",
              "typical_p",
              "min_p",
              "repetition_penalty",
              "length_penalty",
              "diversity_penalty" ->
              value.isNumber() && Double.isFinite(value.asDouble());
          case "do_sample",
              "use_cache",
              "_from_model_config",
              "output_attentions",
              "output_hidden_states",
              "return_dict",
              "return_dict_in_generate",
              "output_scores",
              "early_stopping",
              "is_encoder_decoder",
              "is_decoder",
              "add_cross_attention",
              "tie_encoder_decoder",
              "torchscript",
              "use_bfloat16",
              "remove_invalid_values",
              "tf_legacy_loss" ->
              value.isBoolean();
          case "top_k",
              "max_length",
              "min_length",
              "num_beams",
              "num_beam_groups",
              "num_return_sequences",
              "no_repeat_ngram_size",
              "encoder_no_repeat_ngram_size",
              "chunk_size_feed_forward" ->
              value.isIntegralNumber() && value.canConvertToInt();
          case "bos_token_id", "eos_token_id", "pad_token_id" -> tokenMetadata(value);
          case "id2label", "label2id", "pruned_heads" -> value.isObject();
          case "bad_words_ids",
              "begin_suppress_tokens",
              "cross_attention_hidden_size",
              "decoder_start_token_id",
              "exponential_decay_length_penalty",
              "finetuning_task",
              "forced_bos_token_id",
              "forced_eos_token_id",
              "prefix",
              "problem_type",
              "sep_token_id",
              "suppress_tokens",
              "task_specific_params",
              "tokenizer_class" ->
              value.isNull();
          default -> throw invalid(path + "." + key, "unknown key");
        };
    if (!valid) {
      throw invalid(path + "." + key, "invalid metadata type/value");
    }
  }

  private static boolean tokenMetadata(JsonNode value) {
    if (value.isNull()) {
      return true;
    }
    if (value.isArray()) {
      if (value.isEmpty()) {
        return false;
      }
      for (JsonNode id : value) {
        if (!id.isIntegralNumber() || !id.canConvertToInt() || id.asInt() < 0) {
          return false;
        }
      }
      return true;
    }
    return value.isIntegralNumber() && value.canConvertToInt() && value.asInt() >= 0;
  }

  private static void metadata(JsonNode node, int vocabulary) {
    for (String key : List.of("bos_token_id", "eos_token_id", "pad_token_id")) {
      if (node.has(key) && !node.get(key).isNull()) {
        JsonNode value = node.get(key);
        if (!tokenMetadata(value)) {
          throw invalid(key, "invalid token metadata");
        }
        if (value.isArray()) {
          for (JsonNode id : value) {
            if (id.asInt() >= vocabulary) {
              throw invalid(key, "outside vocabulary");
            }
          }
        } else if (value.asInt() >= vocabulary) {
          throw invalid(key, "outside vocabulary");
        }
      }
    }
  }

  private static JsonNode required(JsonNode node, String key) {
    JsonNode value = node.get(key);
    if (value == null || value.isNull()) {
      throw invalid(key, "missing required value");
    }
    return value;
  }

  private static int positive(JsonNode node, String key) {
    int result = nonnegative(required(node, key), key);
    if (result == 0) {
      throw invalid(key, "must be positive");
    }
    return result;
  }

  private static int nonnegative(JsonNode value, String key) {
    if (!value.isIntegralNumber() || !value.canConvertToInt() || value.asInt() < 0) {
      throw invalid(key, "must be a nonnegative signed integer");
    }
    return value.asInt();
  }

  private static int token(JsonNode node, String key, int vocabulary) {
    int value = nonnegative(required(node, key), key);
    if (value >= vocabulary) {
      throw invalid(key, "outside vocabulary");
    }
    return value;
  }

  private static double number(JsonNode node, String key) {
    JsonNode value = required(node, key);
    if (!value.isNumber() || !Double.isFinite(value.asDouble())) {
      throw invalid(key, "must be a finite number");
    }
    return value.asDouble();
  }

  private static float positiveFloat(JsonNode node, String key) {
    float value = (float) number(node, key);
    if (!Float.isFinite(value) || value <= 0) {
      throw invalid(key, "must be positive finite FLOAT32");
    }
    return value;
  }

  private static boolean bool(JsonNode node, String key) {
    JsonNode value = required(node, key);
    if (!value.isBoolean()) {
      throw invalid(key, "must be boolean");
    }
    return value.asBoolean();
  }

  private static void optionalBoolean(JsonNode node, String key, boolean expected) {
    if (node.has(key) && bool(node, key) != expected) {
      throw invalid(key, "must equal " + expected);
    }
  }

  private static void equal(JsonNode node, String key, String expected) {
    JsonNode value = required(node, key);
    if (!value.isString() || !value.asString().equals(expected)) {
      throw invalid(key, "must equal '" + expected + "'");
    }
  }

  private static void optionalEqual(JsonNode node, String key, String expected) {
    if (node.has(key)) {
      equal(node, key, expected);
    }
  }

  static int product(int... values) {
    try {
      int result = 1;
      for (int value : values) {
        result = Math.multiplyExact(result, value);
      }
      return result;
    } catch (ArithmeticException e) {
      throw invalid("dimensions", "product overflows a signed integer");
    }
  }

  private static int sum(int a, int b) {
    try {
      return Math.addExact(a, b);
    } catch (ArithmeticException e) {
      throw invalid("dimensions", "sum overflows a signed integer");
    }
  }

  private static IllegalArgumentException invalid(String key, String reason) {
    return new IllegalArgumentException("Qwen3.5 " + key + ": " + reason);
  }
}
