package se.alipsa.jmlx.models;

import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import se.alipsa.jmlx.nn.MultiAxisRope;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * Strict Qwen3-VL checkpoint configuration and explicit HF-to-module tensor provenance.
 *
 * <p>The outer config must carry exactly the pinned artifact's ten keys (findings §3.1), the nested
 * {@code text_config} its twenty-one (§3.2) and the nested {@code vision_config} its fourteen
 * (§3.3); any other key is rejected by name, and every required key must be present with the
 * expected type. All structural cross-checks ({@code 2 * sum(mrope_section) == head_dim}, vision
 * output width against the text hidden size, DeepStack index range, the learned-position grid being
 * a perfect square, sentinel ranges and the vision head division) run in this pure parse, before
 * any native work.
 *
 * <p>The text {@link ArchitectureDescriptor} is the qwen3 family descriptor of the nested config
 * re-wrapped under {@code model_type qwen3_vl_text}: the family math (QK-normalized, explicit
 * head-dim, silu MLP) is identical, and {@code rope_scaling} is reduced to {@code {rope_type:
 * default}} because the real rotation always comes from the request's own {@link MultiAxisRope}
 * (the {@code mrope_section} table carried here), so the family's plain base-RoPE spec is never
 * consulted.
 */
record Qwen3VlDescriptor(
    ArchitectureDescriptor text,
    Qwen3VlPromptTokens tokens,
    int videoTokenId,
    int contextLength,
    int depth,
    int numHeads,
    int visionHidden,
    int visionIntermediate,
    int patchSize,
    int temporalPatchSize,
    int mergeSize,
    int inChannels,
    int outHiddenSize,
    int numPositionEmbeddings,
    int[] deepstackIndexes,
    int[] mropeSection) {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** The pinned outer config's complete key set (findings §3.1). */
  private static final Set<String> OUTER_KEYS =
      Set.of(
          "model_type",
          "architectures",
          "tie_word_embeddings",
          "text_config",
          "vision_config",
          "image_token_id",
          "video_token_id",
          "vision_start_token_id",
          "vision_end_token_id",
          "transformers_version");

  /** The pinned text_config's complete key set (findings §3.2). */
  private static final Set<String> TEXT_KEYS =
      Set.of(
          "attention_bias",
          "attention_dropout",
          "bos_token_id",
          "dtype",
          "eos_token_id",
          "head_dim",
          "hidden_act",
          "hidden_size",
          "initializer_range",
          "intermediate_size",
          "max_position_embeddings",
          "model_type",
          "num_attention_heads",
          "num_hidden_layers",
          "num_key_value_heads",
          "rms_norm_eps",
          "rope_theta",
          "rope_scaling",
          "tie_word_embeddings",
          "use_cache",
          "vocab_size");

  /** The pinned vision_config's complete key set (findings §3.3). */
  private static final Set<String> VISION_KEYS =
      Set.of(
          "deepstack_visual_indexes",
          "depth",
          "hidden_act",
          "hidden_size",
          "in_channels",
          "initializer_range",
          "intermediate_size",
          "model_type",
          "num_heads",
          "num_position_embeddings",
          "out_hidden_size",
          "patch_size",
          "spatial_merge_size",
          "temporal_patch_size");

  public Qwen3VlDescriptor {
    Objects.requireNonNull(text, "text");
    Objects.requireNonNull(tokens, "tokens");
    deepstackIndexes = Objects.requireNonNull(deepstackIndexes, "deepstackIndexes").clone();
    mropeSection = Objects.requireNonNull(mropeSection, "mropeSection").clone();
    if (contextLength <= 0) {
      throw new IllegalArgumentException("contextLength must be positive: " + contextLength);
    }
  }

  /**
   * Strictly parses and cross-checks one Qwen3-VL {@code config.json} tree (the dispatch passes the
   * tree {@code TextGenerationModels} already read, so the file is parsed once).
   *
   * @throws IllegalArgumentException when any key is unknown or missing, any value has the wrong
   *     type or value, or a structural cross-check fails
   */
  static Qwen3VlDescriptor parse(JsonNode root) {
    Objects.requireNonNull(root, "root");
    if (!root.isObject()) {
      throw new IllegalArgumentException("config.json must be a JSON object");
    }
    checkKnown(root, OUTER_KEYS, "config.json");
    if (!"qwen3_vl".equals(root.path("model_type").asString())) {
      throw new IllegalArgumentException(
          "config.json model_type must be qwen3_vl, got " + root.path("model_type"));
    }
    requireArchitectures(root);
    if (!root.get("tie_word_embeddings").isBoolean()) {
      throw new IllegalArgumentException("config.json tie_word_embeddings must be a boolean");
    }
    boolean outerTie = root.get("tie_word_embeddings").asBoolean();
    if (root.has("transformers_version") && !root.get("transformers_version").isString()) {
      throw new IllegalArgumentException("config.json transformers_version must be a string");
    }

    TextFields textFields = parseText(root.get("text_config"), outerTie);
    int vocab = textFields.dimensions.vocabSize();
    // The sentinels are prompt vocabulary tokens; a declared one outside the vocabulary cannot be
    // carried into a prompt.
    int imageId = checkSentinel(root, "image_token_id", vocab);
    int startId = checkSentinel(root, "vision_start_token_id", vocab);
    int endId = checkSentinel(root, "vision_end_token_id", vocab);
    // The video sentinel is validated when the config declares it, but the carrier records -1
    // (video generation is out of scope, so the grammar scan never treats it as a marker): the
    // declared ID is kept on the descriptor itself, so the model rejects a prompt containing it.
    int videoId = root.has("video_token_id") ? checkSentinel(root, "video_token_id", vocab) : -1;

    VisionFields vision = parseVision(root.get("vision_config"));
    if (vision.outHiddenSize != textFields.dimensions.hiddenSize()) {
      throw new IllegalArgumentException(
          "vision_config out_hidden_size "
              + vision.outHiddenSize
              + " must equal text_config hidden_size "
              + textFields.dimensions.hiddenSize());
    }

    ArchitectureDescriptor text =
        new ArchitectureDescriptor(
            textFields.dimensions,
            textFields.parsed.headDim(),
            textFields.parsed.rotaryDims(),
            textFields.parsed.rope(),
            textFields.parsed.norm(),
            textFields.parsed.mlp(),
            textFields.parsed.attention(),
            textFields.parsed.head(),
            textFields.parsed.embedding(),
            null,
            null);
    return new Qwen3VlDescriptor(
        text,
        new Qwen3VlPromptTokens(imageId, -1, startId, endId),
        videoId,
        textFields.contextLength,
        vision.depth,
        vision.numHeads,
        vision.hidden,
        vision.intermediate,
        vision.patchSize,
        vision.temporalPatchSize,
        vision.mergeSize,
        vision.inChannels,
        vision.outHiddenSize,
        vision.numPositionEmbeddings,
        vision.deepstackIndexes,
        textFields.mropeSection);
  }

  /**
   * The checkpoint tensor plan: the qwen3 text plan with every {@code model.} name rewritten to
   * {@code model.language_model.} ({@code lm_head.weight} kept as-is, since the checkpoint stores
   * it at the top level) plus the exact {@code model.visual.*} required set derived from the
   * configured depth and DeepStack indices (mirroring {@link
   * Qwen3VlVisionTower#visionKeyToTowerKey(String, int, int[])}); the text plan's {@code
   * .rotary_emb.inv_freq} ignore pattern carries over unchanged and matches the prefixed name.
   */
  TensorPlan tensorPlan() {
    TensorPlan text = ArchitectureMappings.tensorPlan(this.text);
    Set<String> required = new TreeSet<>();
    Set<String> optional = new TreeSet<>();
    Set<String> forbidden = new TreeSet<>();
    text.required().forEach(key -> required.add(textKey(key)));
    text.optional().forEach(key -> optional.add(textKey(key)));
    text.forbidden().forEach(key -> forbidden.add(textKey(key)));
    for (String key : visionKeys()) {
      required.add(Qwen3VlVisionTower.VISUAL_PREFIX + key);
    }
    return new TensorPlan(required, optional, forbidden, text.ignored());
  }

  /** The multi-axis rotary spec every request derives its rotations from. */
  MultiAxisRope multiAxisRope() {
    return new MultiAxisRope(text.dimensions().ropeTheta(), mropeSection);
  }

  /** The exact {@code model.visual.*} checkpoint key set, in sorted order. */
  private Set<String> visionKeys() {
    Set<String> keys = new TreeSet<>();
    for (String suffix : Set.of("weight", "bias")) {
      keys.add("patch_embed.proj." + suffix);
    }
    keys.add("pos_embed.weight");
    for (int i = 0; i < depth; i++) {
      String block = "blocks." + i + ".";
      for (String sub :
          Set.of(
              "norm1.weight",
              "norm1.bias",
              "norm2.weight",
              "norm2.bias",
              "attn.qkv.weight",
              "attn.qkv.bias",
              "attn.proj.weight",
              "attn.proj.bias",
              "mlp.linear_fc1.weight",
              "mlp.linear_fc1.bias",
              "mlp.linear_fc2.weight",
              "mlp.linear_fc2.bias")) {
        keys.add(block + sub);
      }
    }
    for (String merger :
        java.util.stream.Stream.concat(
                java.util.stream.Stream.of("merger"),
                java.util.stream.IntStream.range(0, deepstackIndexes.length)
                    .mapToObj(j -> "deepstack_merger_list." + j))
            .toList()) {
      for (String sub :
          Set.of(
              "norm.weight",
              "norm.bias",
              "linear_fc1.weight",
              "linear_fc1.bias",
              "linear_fc2.weight",
              "linear_fc2.bias")) {
        keys.add(merger + "." + sub);
      }
    }
    return keys;
  }

  /** Rewrites one text-plan name into the Qwen3-VL checkpoint's text prefix. */
  static String textKey(String name) {
    return name.startsWith("model.") ? "model.language_model." + name.substring(6) : name;
  }

  private static final class TextFields {
    private final ArchitectureDescriptor parsed;
    private final DecoderConfig dimensions;
    private final int contextLength;
    private final int[] mropeSection;

    private TextFields(
        ArchitectureDescriptor parsed,
        DecoderConfig dimensions,
        int contextLength,
        int[] mropeSection) {
      this.parsed = parsed;
      this.dimensions = dimensions;
      this.contextLength = contextLength;
      this.mropeSection = mropeSection;
    }
  }

  private static final class VisionFields {
    private final int depth;
    private final int numHeads;
    private final int hidden;
    private final int intermediate;
    private final int patchSize;
    private final int temporalPatchSize;
    private final int mergeSize;
    private final int inChannels;
    private final int outHiddenSize;
    private final int numPositionEmbeddings;
    private final int[] deepstackIndexes;

    private VisionFields(
        int depth,
        int numHeads,
        int hidden,
        int intermediate,
        int patchSize,
        int temporalPatchSize,
        int mergeSize,
        int inChannels,
        int outHiddenSize,
        int numPositionEmbeddings,
        int[] deepstackIndexes) {
      this.depth = depth;
      this.numHeads = numHeads;
      this.hidden = hidden;
      this.intermediate = intermediate;
      this.patchSize = patchSize;
      this.temporalPatchSize = temporalPatchSize;
      this.mergeSize = mergeSize;
      this.inChannels = inChannels;
      this.outHiddenSize = outHiddenSize;
      this.numPositionEmbeddings = numPositionEmbeddings;
      this.deepstackIndexes = deepstackIndexes;
    }
  }

  private static TextFields parseText(JsonNode node, boolean outerTie) {
    checkObject(node, "config.json text_config");
    checkKnown(node, TEXT_KEYS, "config.json text_config");
    if (!"qwen3_vl_text".equals(node.path("model_type").asString())) {
      throw new IllegalArgumentException(
          "text_config model_type must be qwen3_vl_text, got " + node.path("model_type"));
    }
    if (!node.get("attention_bias").isBoolean() || node.get("attention_bias").asBoolean()) {
      throw new IllegalArgumentException("text_config attention_bias must be false");
    }
    if (!"silu".equals(node.path("hidden_act").asString())) {
      throw new IllegalArgumentException(
          "text_config hidden_act must be silu, got " + node.path("hidden_act"));
    }
    int hiddenSize = positiveInt(node, "hidden_size", "text_config");
    int heads = positiveInt(node, "num_attention_heads", "text_config");
    int kvHeads = positiveInt(node, "num_key_value_heads", "text_config");
    if (heads % kvHeads != 0) {
      throw new IllegalArgumentException(
          "text_config num_attention_heads "
              + heads
              + " must be a multiple of num_key_value_heads "
              + kvHeads);
    }
    if (hiddenSize % heads != 0) {
      throw new IllegalArgumentException(
          "text_config hidden_size " + hiddenSize + " must be a multiple of num_attention_heads");
    }
    int headDim = positiveInt(node, "head_dim", "text_config");
    int layers = positiveInt(node, "num_hidden_layers", "text_config");
    int intermediate = positiveInt(node, "intermediate_size", "text_config");
    int vocab = positiveInt(node, "vocab_size", "text_config");
    float eps = positiveFiniteNumber(node, "rms_norm_eps", "text_config");
    float theta = positiveFiniteNumber(node, "rope_theta", "text_config");
    final DecoderConfig dimensions =
        new DecoderConfig(
            "qwen3_vl_text",
            vocab,
            hiddenSize,
            intermediate,
            layers,
            heads,
            kvHeads,
            eps,
            theta,
            outerTie,
            false,
            false);
    int[] mropeSection = parseRopeScaling(node);
    long rotated = 2L * sum(mropeSection);
    if (rotated != headDim) {
      throw new IllegalArgumentException(
          "text_config 2 * sum(mrope_section) "
              + rotated
              + " must equal head_dim "
              + headDim
              + ": the multi-axis rotation's partial-rotary tail would diverge from the reference");
    }
    if (node.has("tie_word_embeddings")) {
      if (!node.get("tie_word_embeddings").isBoolean()) {
        throw new IllegalArgumentException("text_config tie_word_embeddings must be a boolean");
      }
      if (node.get("tie_word_embeddings").asBoolean() != outerTie) {
        throw new IllegalArgumentException(
            "conflicting tie_word_embeddings between config.json and text_config");
      }
    }
    optionalNumber(node, "attention_dropout", "text_config");
    optionalInt(node, "bos_token_id", "text_config");
    optionalInt(node, "eos_token_id", "text_config");
    optionalString(node, "dtype", "text_config");
    optionalNumber(node, "initializer_range", "text_config");
    optionalBoolean(node, "use_cache", "text_config");

    // Re-parse through the qwen3 family with the family-strict rope_scaling (the mrope keys are
    // validated above; parseRope's allow-list rejects them) so the returned capability descriptor
    // is exactly what the ordinary qwen3 assembler builds.
    ObjectNode copy = (ObjectNode) node.deepCopy();
    copy.put("model_type", "qwen3");
    copy.set("rope_scaling", MAPPER.createObjectNode().put("rope_type", "default"));
    copy.put("tie_word_embeddings", outerTie);
    ArchitectureDescriptor parsed;
    try {
      parsed =
          ArchitectureMappings.parse(
              copy,
              key -> {
                throw new IllegalArgumentException("unrecognized text_config key '" + key + "'");
              });
    } catch (RuntimeException e) {
      // Unreachable for a config that passed the strict checks above; keep the failure attributed
      // to the nested block if it ever fires.
      throw new IllegalArgumentException("config.json text_config: " + e.getMessage(), e);
    }
    // Validated last: the family re-parse above never reads it for the default rope, so the
    // strict message is the first failure a broken value can produce.
    int contextLength = positiveInt(node, "max_position_embeddings", "text_config");
    return new TextFields(parsed, dimensions, contextLength, mropeSection);
  }

  /**
   * The text rope_scaling block, strictly: an object holding exactly {@code rope_type} (the string
   * {@code default}), {@code mrope_interleaved} (the boolean {@code true}) and {@code
   * mrope_section} (three non-negative integers).
   */
  private static int[] parseRopeScaling(JsonNode text) {
    JsonNode scaling = text.get("rope_scaling");
    if (scaling == null || !scaling.isObject()) {
      throw new IllegalArgumentException("text_config rope_scaling must be an object");
    }
    if (scaling.size() != 3
        || !scaling.has("rope_type")
        || !scaling.has("mrope_interleaved")
        || !scaling.has("mrope_section")) {
      throw new IllegalArgumentException(
          "text_config rope_scaling must hold exactly rope_type, mrope_interleaved and"
              + " mrope_section");
    }
    if (!"default".equals(scaling.path("rope_type").asString())) {
      throw new IllegalArgumentException(
          "text_config rope_scaling.rope_type must be default, got " + scaling.path("rope_type"));
    }
    if (!scaling.get("mrope_interleaved").isBoolean()
        || !scaling.get("mrope_interleaved").asBoolean()) {
      throw new IllegalArgumentException("text_config rope_scaling.mrope_interleaved must be true");
    }
    JsonNode section = scaling.get("mrope_section");
    if (!section.isArray() || section.size() != 3) {
      throw new IllegalArgumentException(
          "text_config rope_scaling.mrope_section must hold exactly 3 entries");
    }
    int[] mropeSection = new int[3];
    long total = 0;
    for (int i = 0; i < 3; i++) {
      JsonNode value = section.get(i);
      if (!value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() < 0) {
        throw new IllegalArgumentException(
            "text_config rope_scaling.mrope_section entries must be non-negative integers");
      }
      mropeSection[i] = value.intValue();
      total += mropeSection[i];
    }
    if (total < 1) {
      throw new IllegalArgumentException(
          "text_config rope_scaling.mrope_section must rotate at least one dimension pair");
    }
    return mropeSection;
  }

  private static VisionFields parseVision(JsonNode node) {
    checkObject(node, "config.json vision_config");
    checkKnown(node, VISION_KEYS, "config.json vision_config");
    if (!"qwen3_vl".equals(node.path("model_type").asString())) {
      throw new IllegalArgumentException(
          "vision_config model_type must be qwen3_vl, got " + node.path("model_type"));
    }
    if (!"gelu_pytorch_tanh".equals(node.path("hidden_act").asString())) {
      throw new IllegalArgumentException(
          "vision_config hidden_act must be gelu_pytorch_tanh, got " + node.path("hidden_act"));
    }
    int hidden = positiveInt(node, "hidden_size", "vision_config");
    int numHeads = positiveInt(node, "num_heads", "vision_config");
    if (hidden % numHeads != 0) {
      throw new IllegalArgumentException(
          "vision_config num_heads " + numHeads + " must divide hidden_size " + hidden);
    }
    if ((hidden / numHeads) % 4 != 0) {
      throw new IllegalArgumentException(
          "vision_config head dimension " + hidden / numHeads + " must be a multiple of 4");
    }
    if (!node.get("in_channels").isIntegralNumber() || node.get("in_channels").intValue() != 3) {
      throw new IllegalArgumentException("vision_config in_channels must be 3");
    }
    int patchSize = positiveInt(node, "patch_size", "vision_config");
    int temporalPatchSize = positiveInt(node, "temporal_patch_size", "vision_config");
    int mergeSize = positiveInt(node, "spatial_merge_size", "vision_config");
    int numPositionEmbeddings = positiveInt(node, "num_position_embeddings", "vision_config");
    int side = (int) Math.sqrt(numPositionEmbeddings);
    if (side * side != numPositionEmbeddings) {
      throw new IllegalArgumentException(
          "vision_config num_position_embeddings "
              + numPositionEmbeddings
              + " must be a perfect square grid");
    }
    optionalNumber(node, "initializer_range", "vision_config");
    int depth = positiveInt(node, "depth", "vision_config");
    int[] deepstackIndexes = parseDeepstackIndexes(node, depth);
    int intermediate = positiveInt(node, "intermediate_size", "vision_config");
    return new VisionFields(
        depth,
        numHeads,
        hidden,
        intermediate,
        patchSize,
        temporalPatchSize,
        mergeSize,
        3,
        positiveInt(node, "out_hidden_size", "vision_config"),
        numPositionEmbeddings,
        deepstackIndexes);
  }

  /**
   * The DeepStack block indices: non-empty, strictly increasing, each within {@code [0, depth)}.
   */
  private static int[] parseDeepstackIndexes(JsonNode vision, int depth) {
    JsonNode indexes = vision.get("deepstack_visual_indexes");
    if (indexes == null || !indexes.isArray() || indexes.size() == 0) {
      throw new IllegalArgumentException(
          "vision_config deepstack_visual_indexes must be a non-empty array");
    }
    int[] deepstackIndexes = new int[indexes.size()];
    int previous = -1;
    for (int i = 0; i < indexes.size(); i++) {
      JsonNode value = indexes.get(i);
      if (!value.isIntegralNumber() || !value.canConvertToInt()) {
        throw new IllegalArgumentException(
            "vision_config deepstack_visual_indexes entries must be integers");
      }
      int index = value.intValue();
      if (index <= previous) {
        throw new IllegalArgumentException(
            "vision_config deepstack_visual_indexes must be strictly increasing");
      }
      if (index >= depth) {
        throw new IllegalArgumentException(
            "vision_config deepstack_visual_indexes entry "
                + index
                + " is out of range [0, "
                + depth
                + ")");
      }
      deepstackIndexes[i] = index;
      previous = index;
    }
    return deepstackIndexes;
  }

  private static void requireArchitectures(JsonNode root) {
    JsonNode architectures = root.get("architectures");
    if (architectures == null
        || !architectures.isArray()
        || architectures.size() != 1
        || !"Qwen3VLForConditionalGeneration".equals(architectures.get(0).asString())) {
      throw new IllegalArgumentException(
          "config.json architectures must be exactly [\"Qwen3VLForConditionalGeneration\"]");
    }
  }

  private static int checkSentinel(JsonNode root, String key, int vocab) {
    JsonNode value = root.get(key);
    if (!value.isIntegralNumber() || !value.canConvertToInt()) {
      throw new IllegalArgumentException("config.json " + key + " must be an integer");
    }
    int id = value.intValue();
    if (id < 0 || id >= vocab) {
      throw new IllegalArgumentException(
          "config.json " + key + " " + id + " is outside the text vocabulary [0, " + vocab + ")");
    }
    return id;
  }

  private static void checkObject(JsonNode node, String name) {
    if (node == null || !node.isObject()) {
      throw new IllegalArgumentException(name + " must be a JSON object");
    }
  }

  private static void checkKnown(JsonNode node, Set<String> keys, String name) {
    for (var entry : node.properties()) {
      if (!keys.contains(entry.getKey())) {
        throw new IllegalArgumentException(name + " has unrecognized key '" + entry.getKey() + "'");
      }
    }
  }

  private static int positiveInt(JsonNode node, String key, String name) {
    JsonNode value = node.get(key);
    if (value == null
        || !value.isIntegralNumber()
        || !value.canConvertToInt()
        || value.intValue() <= 0) {
      throw new IllegalArgumentException(name + " " + key + " must be a positive integer");
    }
    return value.intValue();
  }

  private static float positiveFiniteNumber(JsonNode node, String key, String name) {
    JsonNode value = node.get(key);
    if (value == null || !value.isNumber()) {
      throw new IllegalArgumentException(name + " " + key + " must be a number");
    }
    float result = value.floatValue();
    if (!Float.isFinite(result) || result <= 0) {
      throw new IllegalArgumentException(name + " " + key + " must be positive and finite");
    }
    return result;
  }

  private static void optionalNumber(JsonNode node, String key, String name) {
    if (node.has(key) && !node.get(key).isNumber()) {
      throw new IllegalArgumentException(name + " " + key + " must be a number");
    }
  }

  private static void optionalInt(JsonNode node, String key, String name) {
    if (node.has(key) && (!node.get(key).isIntegralNumber() || !node.get(key).canConvertToInt())) {
      throw new IllegalArgumentException(name + " " + key + " must be an integer");
    }
  }

  private static void optionalString(JsonNode node, String key, String name) {
    if (node.has(key) && !node.get(key).isString()) {
      throw new IllegalArgumentException(name + " " + key + " must be a string");
    }
  }

  private static void optionalBoolean(JsonNode node, String key, String name) {
    if (node.has(key) && !node.get(key).isBoolean()) {
      throw new IllegalArgumentException(name + " " + key + " must be a boolean");
    }
  }

  private static long sum(int[] values) {
    long total = 0;
    for (int value : values) {
      total += value;
    }
    return total;
  }
}
