package se.alipsa.jmlx.models;

import java.util.Map;
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
 * expected type. Real transformers configs additionally carry a large set of inert generation and
 * beam-search defaults plus null model fields at every level; the recognized inert keys are typed,
 * type-checked and stripped from each level before the strict key check (so the pinned strict parse
 * is never loosened), and the MLX-affine quantization declaration is stripped as an outer inert
 * key, which is why provenance selection must read the raw tree first. All structural cross-checks
 * ({@code 2 * sum(mrope_section) == head_dim}, vision output width against the text hidden size,
 * DeepStack index range, the learned-position grid being a perfect square, sentinel ranges and the
 * vision head division) run in this pure parse, before any native work.
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

  /** The vision prefix the real MLX-converted artifact carries on every vision tensor key. */
  static final String VISION_TOWER_PREFIX = "vision_tower.";

  /**
   * The JSON types an inert config key may carry. The allow-lists below key each recognized inert
   * name to exactly one of these, so a present inert key of the wrong type is rejected by name
   * rather than silently dropped or leaking through into the strict key check.
   */
  private enum InertType {
    BOOLEAN {
      boolean matches(JsonNode value) {
        return value.isBoolean();
      }

      String label() {
        return "a boolean";
      }
    },
    INTEGER {
      boolean matches(JsonNode value) {
        return value.isIntegralNumber();
      }

      String label() {
        return "an integer";
      }
    },
    NUMBER {
      boolean matches(JsonNode value) {
        return value.isNumber();
      }

      String label() {
        return "a number";
      }
    },
    OBJECT {
      boolean matches(JsonNode value) {
        return value.isObject();
      }

      String label() {
        return "an object";
      }
    },
    NULL {
      boolean matches(JsonNode value) {
        return value.isNull();
      }

      String label() {
        return "null";
      }
    },
    STRING {
      boolean matches(JsonNode value) {
        return value.isString();
      }

      String label() {
        return "a string";
      }
    };

    abstract boolean matches(JsonNode value);

    abstract String label();
  }

  /**
   * The inert keys carried by every level of a real transformers checkpoint, each with the exact
   * JSON type the pinned 8B artifact uses: the generation/beam-search defaults plus the common
   * {@code null} model fields. They are stripped from a level's tree before the strict key check,
   * so a config that carries them parses as if they were absent. The per-level {@code *_INERT} maps
   * are this set plus a few level-specific additions, and none may overlap that level's pinned
   * keys.
   */
  private static final Map<String, InertType> COMMON_INERT =
      inert(
          InertType.BOOLEAN,
          "add_cross_attention",
          "do_sample",
          "early_stopping",
          "is_decoder",
          "is_encoder_decoder",
          "output_attentions",
          "output_hidden_states",
          "output_scores",
          "remove_invalid_values",
          "return_dict",
          "return_dict_in_generate",
          "tf_legacy_loss",
          "tie_encoder_decoder",
          "torchscript",
          "use_bfloat16",
          InertType.INTEGER,
          "chunk_size_feed_forward",
          "encoder_no_repeat_ngram_size",
          "max_length",
          "min_length",
          "no_repeat_ngram_size",
          "num_beam_groups",
          "num_beams",
          "num_return_sequences",
          "top_k",
          InertType.NUMBER,
          "diversity_penalty",
          "length_penalty",
          "repetition_penalty",
          "temperature",
          "top_p",
          "typical_p",
          InertType.OBJECT,
          "id2label",
          "label2id",
          "pruned_heads",
          InertType.NULL,
          "bad_words_ids",
          "begin_suppress_tokens",
          "cross_attention_hidden_size",
          "decoder_start_token_id",
          "exponential_decay_length_penalty",
          "finetuning_task",
          "forced_bos_token_id",
          "forced_eos_token_id",
          "pad_token_id",
          "prefix",
          "problem_type",
          "sep_token_id",
          "suppress_tokens",
          "task_specific_params",
          "tokenizer_class");

  /**
   * The outer-level inert additions, on top of the common set (the pinned outer set carries none of
   * these).
   */
  private static final Map<String, InertType> OUTER_INERT =
      inert(
          COMMON_INERT,
          InertType.NULL,
          "dtype",
          "bos_token_id",
          "eos_token_id",
          InertType.OBJECT,
          "quantization",
          "quantization_config");

  /**
   * The text-level inert additions. The text's own {@code dtype}/{@code bos_token_id}/{@code
   * eos_token_id} are pinned optionals at this level, so they are deliberately not inert here.
   */
  private static final Map<String, InertType> TEXT_INERT =
      inert(COMMON_INERT, InertType.STRING, "_name_or_path", InertType.NULL, "architectures");

  /** The vision-level inert additions, on top of the common set. */
  private static final Map<String, InertType> VISION_INERT =
      inert(
          COMMON_INERT,
          InertType.STRING,
          "_name_or_path",
          InertType.BOOLEAN,
          "tie_word_embeddings",
          InertType.NULL,
          "architectures",
          "dtype",
          "bos_token_id",
          "eos_token_id");

  /**
   * Builds one level's inert allow-list from alternating groups: each {@code InertType} governs the
   * {@code String} names that follow it until the next type. A name appearing under two types, or
   * before any type, is a programming error and fails loudly at class initialization.
   */
  private static Map<String, InertType> inert(Object... groups) {
    return inert(Map.of(), groups);
  }

  /**
   * Builds one level's inert allow-list from a base map plus alternating groups: each {@code
   * InertType} governs the {@code String} names that follow it until the next type. A name already
   * in the base under a different type, or a name before any type, is a programming error and fails
   * loudly at class initialization.
   */
  private static Map<String, InertType> inert(Map<String, InertType> base, Object... groups) {
    Map<String, InertType> result = new java.util.LinkedHashMap<>(base);
    InertType current = null;
    for (Object group : groups) {
      if (group instanceof InertType type) {
        current = type;
      } else if (group instanceof String name) {
        if (current == null) {
          throw new IllegalStateException("inert key before a type: " + name);
        }
        InertType previous = result.put(name, current);
        if (previous != null && previous != current) {
          throw new IllegalStateException("inert key under two types: " + name);
        }
      } else {
        throw new IllegalStateException("inert group must be a type or a key name: " + group);
      }
    }
    return Map.copyOf(result);
  }

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
    // Real transformers configs carry generation/beam-search defaults and null model fields at
    // every level; strip this level's recognized inert keys before the strict key check so a
    // config that carries them parses as if they were absent (the quantization declaration is one
    // of them, which is why the provenance selection must read the raw tree first).
    ObjectNode cleaned = stripInert((ObjectNode) root, OUTER_INERT, "config.json");
    checkKnown(cleaned, OUTER_KEYS, "config.json");
    if (!"qwen3_vl".equals(cleaned.path("model_type").asString())) {
      throw new IllegalArgumentException(
          "config.json model_type must be qwen3_vl, got " + cleaned.path("model_type"));
    }
    requireArchitectures(cleaned);
    if (!cleaned.get("tie_word_embeddings").isBoolean()) {
      throw new IllegalArgumentException("config.json tie_word_embeddings must be a boolean");
    }
    boolean outerTie = cleaned.get("tie_word_embeddings").asBoolean();
    if (cleaned.has("transformers_version") && !cleaned.get("transformers_version").isString()) {
      throw new IllegalArgumentException("config.json transformers_version must be a string");
    }

    TextFields textFields = parseText(cleaned.get("text_config"), outerTie);
    int vocab = textFields.dimensions.vocabSize();
    // The sentinels are prompt vocabulary tokens; a declared one outside the vocabulary cannot be
    // carried into a prompt.
    int imageId = checkSentinel(cleaned, "image_token_id", vocab);
    int startId = checkSentinel(cleaned, "vision_start_token_id", vocab);
    int endId = checkSentinel(cleaned, "vision_end_token_id", vocab);
    // The video sentinel is validated when the config declares it, but the carrier records -1
    // (video generation is out of scope, so the grammar scan never treats it as a marker): the
    // declared ID is kept on the descriptor itself, so the model rejects a prompt containing it.
    int videoId =
        cleaned.has("video_token_id") ? checkSentinel(cleaned, "video_token_id", vocab) : -1;

    VisionFields vision = parseVision(cleaned.get("vision_config"));
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
   * The affine-quantized parse: the identical strict parse of {@link #parse}, re-wrapped with the
   * config's MLX-affine {@code group_size}/{@code bits} on the text {@link ArchitectureDescriptor}
   * so {@link #affineTensorPlan(int, int)} and the packed decoder path consume it. The group size
   * and bits are already validated by {@link Qwen3VlProvenance}; this method trusts them.
   */
  static Qwen3VlDescriptor parseAffine(JsonNode root, int groupSize, int bits) {
    return parse(root).withAffineQuantization(groupSize, bits);
  }

  /** The same descriptor with a non-null text-side {@link ArchitectureDescriptor.Quantization}. */
  Qwen3VlDescriptor withAffineQuantization(int groupSize, int bits) {
    ArchitectureDescriptor quantized =
        new ArchitectureDescriptor(
            text.dimensions(),
            text.headDim(),
            text.rotaryDims(),
            text.rope(),
            text.norm(),
            text.mlp(),
            text.attention(),
            text.head(),
            text.embedding(),
            text.moe(),
            new ArchitectureDescriptor.Quantization(groupSize, bits));
    return new Qwen3VlDescriptor(
        quantized,
        tokens,
        videoTokenId,
        contextLength,
        depth,
        numHeads,
        visionHidden,
        visionIntermediate,
        patchSize,
        temporalPatchSize,
        mergeSize,
        inChannels,
        outHiddenSize,
        numPositionEmbeddings,
        deepstackIndexes,
        mropeSection);
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

  /**
   * The MLX-affine tensor plan: the affine-quantized text plan (with its {@code .scales}/{@code
   * .biases} optional companions) with every name rewritten under the artifact's {@code
   * language_model.} prefix ({@code model.embed_tokens.weight} becomes {@code
   * language_model.model.embed_tokens.weight}; {@code lm_head.weight} becomes {@code
   * language_model.lm_head.weight}) plus the exact {@code vision_tower.*} required set derived from
   * the configured depth and DeepStack indices. The plan is explicit, not derived from the HF plan:
   * the vision tower is all-float in the affine layout (its keys carry no {@code .scales}), and the
   * conv weight arrives channels-last (never transposed on load).
   */
  TensorPlan affineTensorPlan(int groupSize, int bits) {
    TensorPlan text =
        ArchitectureMappings.tensorPlan(withAffineQuantization(groupSize, bits).text());
    Set<String> required = new TreeSet<>();
    Set<String> optional = new TreeSet<>();
    Set<String> forbidden = new TreeSet<>();
    text.required().forEach(key -> required.add(affineTextKey(key)));
    text.optional().forEach(key -> optional.add(affineTextKey(key)));
    text.forbidden().forEach(key -> forbidden.add(affineTextKey(key)));
    for (String key : visionKeys()) {
      required.add(VISION_TOWER_PREFIX + key);
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

  /**
   * Rewrites one text-plan name into the MLX-affine artifact's text prefix: a pure {@code
   * language_model.} prefix (the converted artifact keeps the inner {@code model.} on every
   * non-head tensor and stores the head as {@code language_model.lm_head.*}).
   */
  static String affineTextKey(String name) {
    return "language_model." + name;
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

  private static TextFields parseText(JsonNode raw, boolean outerTie) {
    checkObject(raw, "config.json text_config");
    // The converted artifact's text_config carries a few inert keys (e.g. _name_or_path,
    // architectures); strip the recognized ones so the strict key check sees only the pinned set.
    ObjectNode node = stripInert((ObjectNode) raw, TEXT_INERT, "config.json text_config");
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

  private static VisionFields parseVision(JsonNode raw) {
    checkObject(raw, "config.json vision_config");
    // The converted artifact's vision_config carries inert keys (e.g. _name_or_path,
    // tie_word_embeddings); strip the recognized ones so the strict key check sees only the pinned
    // set (the text-level dtype/bos/eos stay pinned there and are not inert at either level).
    ObjectNode node = stripInert((ObjectNode) raw, VISION_INERT, "config.json vision_config");
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

  /**
   * Strips a level's recognized inert keys from the tree, returning a deep copy: every present
   * inert key is type-checked against its allow-list entry (mismatching types are rejected by name
   * rather than silently dropped) and removed, so the strict key check sees the config as if the
   * inert keys were never written. A key not on the allow-list is left in place for {@code
   * checkKnown} to reject, so an unknown key is never silently swallowed.
   */
  private static ObjectNode stripInert(ObjectNode node, Map<String, InertType> inert, String name) {
    ObjectNode copy = (ObjectNode) node.deepCopy();
    for (var entry : inert.entrySet()) {
      String key = entry.getKey();
      if (!copy.has(key)) {
        continue;
      }
      if (!entry.getValue().matches(copy.get(key))) {
        throw new IllegalArgumentException(
            name + " inert key '" + key + "' must be " + entry.getValue().label());
      }
      copy.remove(key);
    }
    return copy;
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
