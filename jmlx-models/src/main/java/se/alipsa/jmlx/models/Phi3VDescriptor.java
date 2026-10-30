package se.alipsa.jmlx.models;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import se.alipsa.jmlx.nn.RopeSpec;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Strict Phi-3.5-Vision configuration and explicit HF-to-module tensor provenance.
 *
 * <p>The pinned {@code phi3_v} config embeds its vision tower in two configuration blocks — {@code
 * img_processor} (the CLIP model identity and output dimension) and {@code embd_layer} (the
 * high-definition transform contract) — alongside plain Llama-shaped decoder fields. This parser
 * validates the two blocks against the pinned values the port implements, records their inert
 * fields, and strips the five keys that do not belong to the decoder ({@code img_processor}, {@code
 * embd_layer} plus the HF export metadata {@code _attn_implementation}, {@code _name_or_path} and
 * {@code auto_map}) before re-parsing the remainder with {@link ArchitectureMappings} — the only
 * family that accepts {@code su} rope scaling.
 */
record Phi3VDescriptor(ArchitectureDescriptor text, int numImgTokens, String embeddingCls) {

  /** The 31 top-level keys of the pinned 4.2B config; strict allow-list for the reparse. */
  private static final Set<String> TOP_LEVEL =
      Set.of(
          "_attn_implementation",
          "_name_or_path",
          "architectures",
          "attention_dropout",
          "auto_map",
          "bos_token_id",
          "embd_layer",
          "embd_pdrop",
          "eos_token_id",
          "hidden_act",
          "hidden_size",
          "img_processor",
          "initializer_range",
          "intermediate_size",
          "max_position_embeddings",
          "model_type",
          "num_attention_heads",
          "num_hidden_layers",
          "num_key_value_heads",
          "original_max_position_embeddings",
          "pad_token_id",
          "resid_pdrop",
          "rms_norm_eps",
          "rope_scaling",
          "rope_theta",
          "sliding_window",
          "tie_word_embeddings",
          "torch_dtype",
          "transformers_version",
          "use_cache",
          "vocab_size");

  private static final Set<String> IMG_PROCESSOR_KEYS =
      Set.of("name", "model_name", "image_dim_out", "num_img_tokens", "layer_idx", "type_feature");

  private static final Set<String> EMBD_LAYER_KEYS =
      Set.of(
          "projection_cls",
          "use_hd_transform",
          "hd_transform_order",
          "with_learnable_separator",
          "embedding_cls");

  private static final String CLIP_MODEL_NAME = "openai/clip-vit-large-patch14-336";
  private static final int CLIP_IMAGE_DIM_OUT = 1024;

  /**
   * The pinned ViT-L/14-336 encoder depth. The last discovered layer is plan-optional and never
   * executed, so the loader requires the discovered depth to match this exactly: a checkpoint that
   * dropped the final layer would run the tower one block too few and corrupt every image feature
   * with no error. The tiny synthetic checkpoints carry fewer layers and opt out explicitly via the
   * {@code jmlx.phi3v.expectedClipLayers} test property.
   */
  static final int PINNED_CLIP_LAYERS = 24;

  private static final String VISION_PREFIX = "model.vision_embed_tokens.";

  /**
   * Parses and validates the config; the five non-decoder keys are stripped from a copy, and the
   * remaining decoder fields must parse as an {@code su}-scaled {@code phi3_v} architecture.
   */
  static Phi3VDescriptor parse(JsonNode root) {
    known(root, TOP_LEVEL);
    if (!"phi3_v".equals(root.path("model_type").asString())) {
      throw new IllegalArgumentException("expected model_type=phi3_v");
    }
    JsonNode architectures = root.path("architectures");
    if (!architectures.isArray()
        || architectures.size() != 1
        || !"Phi3VForCausalLM".equals(architectures.get(0).asString())) {
      throw new IllegalArgumentException("architectures must be [Phi3VForCausalLM]");
    }
    ObjectNode copy = (ObjectNode) root.deepCopy();
    copy.remove("img_processor");
    copy.remove("embd_layer");
    copy.remove("_attn_implementation");
    copy.remove("_name_or_path");
    copy.remove("auto_map");
    ArchitectureDescriptor text =
        ArchitectureMappings.parse(
            copy,
            key -> {
              throw new IllegalArgumentException("unrecognized phi3_v config key '" + key + "'");
            });
    if (!(text.rope() instanceof RopeSpec.Su)) {
      throw new IllegalArgumentException("phi3_v requires su rope_scaling");
    }
    int numImgTokens = parseImgProcessor(root.path("img_processor"));
    String embeddingCls = parseEmbdLayer(root.path("embd_layer"));
    return new Phi3VDescriptor(text, numImgTokens, embeddingCls);
  }

  /**
   * The vision tower's checkpoint keys, relative to {@code model.vision_embed_tokens.} — the caller
   * prefixes them — mapped to the tower's module parameter paths. Executed CLIP layers only: layer
   * {@code clipLayers - 1} and the discarded pooled-output {@code post_layernorm} are plan-optional
   * but never extracted.
   */
  Map<String, String> visionKeys(int clipLayers) {
    Map<String, String> keys = new LinkedHashMap<>();
    keys.put("glb_GN", "glbGN");
    keys.put("sub_GN", "subGN");
    keys.put("img_processor.vision_model.embeddings.class_embedding", "classToken");
    keys.put(
        "img_processor.vision_model.embeddings.patch_embedding.weight", "patchEmbedding.weight");
    keys.put("img_processor.vision_model.embeddings.position_embedding.weight", "positionTable");
    keys.put("img_processor.vision_model.pre_layrnorm.weight", "preLayernorm.weight");
    keys.put("img_processor.vision_model.pre_layrnorm.bias", "preLayernorm.bias");
    for (int i = 0; i < clipLayers - 1; i++) {
      Map<String, String> names =
          Map.of(
              "self_attn.q_proj",
              "attention.query",
              "self_attn.k_proj",
              "attention.key",
              "self_attn.v_proj",
              "attention.value",
              "self_attn.out_proj",
              "attention.output",
              "layer_norm1",
              "attentionNorm",
              "layer_norm2",
              "outputNorm",
              "mlp.fc1",
              "feedForward.0",
              "mlp.fc2",
              "feedForward.2");
      for (var entry : names.entrySet()) {
        for (String suffix : Set.of(".weight", ".bias")) {
          String relative =
              "img_processor.vision_model.encoder.layers." + i + "." + entry.getKey() + suffix;
          keys.put(relative, "layer" + i + "." + entry.getValue() + suffix);
        }
      }
    }
    keys.put("img_projection.0.weight", "projector.0.weight");
    keys.put("img_projection.0.bias", "projector.0.bias");
    keys.put("img_projection.2.weight", "projector.2.weight");
    keys.put("img_projection.2.bias", "projector.2.bias");
    return Map.copyOf(keys);
  }

  /** The full checkpoint tensor plan for one CLIP layer count (executed layers required). */
  TensorPlan tensorPlan(int clipLayers) {
    if (clipLayers < 1) {
      throw new IllegalArgumentException("clipLayers must be positive: " + clipLayers);
    }
    TensorPlan decoder = ArchitectureMappings.tensorPlan(text);
    Set<String> required = new TreeSet<>(decoder.required());
    Set<String> optional = new TreeSet<>(decoder.optional());
    for (String key :
        Set.of(
            "glb_GN",
            "sub_GN",
            "img_processor.vision_model.embeddings.class_embedding",
            "img_processor.vision_model.embeddings.patch_embedding.weight",
            "img_processor.vision_model.embeddings.position_embedding.weight",
            "img_processor.vision_model.pre_layrnorm.weight",
            "img_processor.vision_model.pre_layrnorm.bias",
            "img_projection.0.weight",
            "img_projection.0.bias",
            "img_projection.2.weight",
            "img_projection.2.bias")) {
      required.add(visionKey(key));
    }
    for (int i = 0; i < clipLayers - 1; i++) {
      required.addAll(visionLayerKeys(i));
    }
    optional.addAll(visionLayerKeys(clipLayers - 1));
    optional.add(visionKey("img_processor.vision_model.post_layernorm.weight"));
    optional.add(visionKey("img_processor.vision_model.post_layernorm.bias"));
    Set<String> forbidden = new TreeSet<>(decoder.forbidden());
    forbidden.add(visionKey("img_processor.vision_model.embeddings.patch_embedding.bias"));
    return new TensorPlan(required, optional, forbidden, Set.of());
  }

  /** The vision tower weight for one HF checkpoint key. */
  static String visionKey(String relativeName) {
    return VISION_PREFIX + relativeName;
  }

  private static Set<String> visionLayerKeys(int layer) {
    Set<String> keys = new TreeSet<>();
    for (String projection :
        Set.of(
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "layer_norm1",
            "layer_norm2",
            "mlp.fc1",
            "mlp.fc2")) {
      for (String suffix : Set.of(".weight", ".bias")) {
        keys.add(layerKey(layer, projection) + suffix);
      }
    }
    return keys;
  }

  private static String layerKey(int layer, String projection) {
    return visionKey("img_processor.vision_model.encoder.layers." + layer + "." + projection);
  }

  private static int parseImgProcessor(JsonNode node) {
    knownBlock(node, "img_processor", IMG_PROCESSOR_KEYS);
    if (!"clip_vision_model".equals(node.path("name").asString())) {
      throw new IllegalArgumentException("img_processor.name must be clip_vision_model");
    }
    if (!CLIP_MODEL_NAME.equals(node.path("model_name").asString())) {
      throw new IllegalArgumentException("img_processor.model_name must be " + CLIP_MODEL_NAME);
    }
    if (positive(node, "image_dim_out") != CLIP_IMAGE_DIM_OUT) {
      throw new IllegalArgumentException(
          "img_processor.image_dim_out must be " + CLIP_IMAGE_DIM_OUT);
    }
    // Legacy constant (the actual row count is geometry-driven); recorded, never computed from.
    int numImgTokens = positive(node, "num_img_tokens");
    // Optional; accepted only at the bundled modeling defaults, which the port implements.
    if (node.has("layer_idx")
        && (!node.path("layer_idx").isIntegralNumber()
            || node.path("layer_idx").intValue() != -2)) {
      throw new IllegalArgumentException("img_processor.layer_idx must be -2");
    }
    if (node.has("type_feature") && !"patch".equals(node.path("type_feature").asString())) {
      throw new IllegalArgumentException("img_processor.type_feature must be patch");
    }
    return numImgTokens;
  }

  private static String parseEmbdLayer(JsonNode node) {
    knownBlock(node, "embd_layer", EMBD_LAYER_KEYS);
    // All four contract keys are required at the pinned values: the bundled modeling defaults
    // (projection linear, no HD transform, glb_sub order) differ, so a missing key is a reject.
    if (!"mlp".equals(node.path("projection_cls").asString())) {
      throw new IllegalArgumentException("embd_layer.projection_cls must be mlp");
    }
    if (!node.path("use_hd_transform").asBoolean(false)) {
      throw new IllegalArgumentException("embd_layer.use_hd_transform must be true");
    }
    if (!"sub_glb".equals(node.path("hd_transform_order").asString())) {
      throw new IllegalArgumentException("embd_layer.hd_transform_order must be sub_glb");
    }
    if (!node.path("with_learnable_separator").asBoolean(false)) {
      throw new IllegalArgumentException("embd_layer.with_learnable_separator must be true");
    }
    // The bundled code indexes this key directly, but never reads the value: required textual,
    // recorded and inert.
    String embeddingCls = node.path("embedding_cls").asString();
    if (embeddingCls.isEmpty()) {
      throw new IllegalArgumentException("embd_layer.embedding_cls must be a non-empty string");
    }
    return embeddingCls;
  }

  private static void known(JsonNode node, Set<String> keys) {
    if (!node.isObject()) {
      throw new IllegalArgumentException("configuration must be an object");
    }
    for (var entry : node.properties()) {
      if (!keys.contains(entry.getKey())) {
        throw new IllegalArgumentException(
            "unrecognized phi3_v config key '" + entry.getKey() + "'");
      }
    }
  }

  private static void knownBlock(JsonNode node, String block, Set<String> keys) {
    if (!node.isObject()) {
      throw new IllegalArgumentException(block + " must be an object");
    }
    for (var entry : node.properties()) {
      if (!keys.contains(entry.getKey())) {
        throw new IllegalArgumentException(
            "unrecognized " + block + " key '" + entry.getKey() + "'");
      }
    }
  }

  private static int positive(JsonNode node, String key) {
    JsonNode value = node.path(key);
    if (!value.isIntegralNumber() || !value.canConvertToInt() || value.intValue() <= 0) {
      throw new IllegalArgumentException(key + " must be a positive integer");
    }
    return value.intValue();
  }
}
