package se.alipsa.jmlx.models;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.node.ObjectNode;

/** Strict Idefics3 configuration and explicit HF-to-module tensor provenance. */
record SmolVlmDescriptor(
    ArchitectureDescriptor text,
    int imageTokenId,
    int scale,
    int contextLength,
    int imageSize,
    int patchSize,
    int visionHidden,
    int intermediate,
    int layers,
    int heads,
    float normEps,
    Map<String, String> visionKeys) {

  // Inert Hugging Face PretrainedConfig serialization and generation defaults. GenerationRequest
  // owns runtime generation policy; these are never computational decoder fields.
  private static final Set<String> EXPORT_METADATA =
      Set.of(
          "_attn_implementation_autoset",
          "_name_or_path",
          "architectures",
          "bad_words_ids",
          "begin_suppress_tokens",
          "bos_token_id",
          "decoder_start_token_id",
          "diversity_penalty",
          "do_sample",
          "early_stopping",
          "encoder_no_repeat_ngram_size",
          "eos_token_id",
          "exponential_decay_length_penalty",
          "finetuning_task",
          "forced_bos_token_id",
          "forced_eos_token_id",
          "id2label",
          "label2id",
          "is_decoder",
          "is_encoder_decoder",
          "length_penalty",
          "max_length",
          "min_length",
          "no_repeat_ngram_size",
          "num_beam_groups",
          "num_beams",
          "num_return_sequences",
          "output_attentions",
          "output_hidden_states",
          "output_scores",
          "pad_token_id",
          "prefix",
          "problem_type",
          "remove_invalid_values",
          "repetition_penalty",
          "return_dict",
          "return_dict_in_generate",
          "sep_token_id",
          "suppress_tokens",
          "task_specific_params",
          "temperature",
          "tf_legacy_loss",
          "tokenizer_class",
          "top_k",
          "top_p",
          "torchscript",
          "typical_p",
          "use_bfloat16",
          "neftune_noise_alpha");

  private static void removeInertExportFields(ObjectNode node) {
    EXPORT_METADATA.forEach(node::remove);
    for (String flag : Set.of("add_cross_attention", "tie_encoder_decoder", "rope_interleaved")) {
      if (node.has(flag)) {
        if (!node.path(flag).isBoolean() || node.path(flag).asBoolean()) {
          throw new IllegalArgumentException("unsupported " + flag);
        }
        node.remove(flag);
      }
    }
    if (node.hasNonNull("cross_attention_hidden_size")
        || node.path("chunk_size_feed_forward").asInt(0) != 0
        || (node.has("pruned_heads")
            && (!node.path("pruned_heads").isObject() || !node.path("pruned_heads").isEmpty()))) {
      throw new IllegalArgumentException(
          "cross-attention, chunking and pruned heads are unsupported");
    }
    node.remove("cross_attention_hidden_size");
    node.remove("chunk_size_feed_forward");
    node.remove("pruned_heads");
  }

  static SmolVlmDescriptor parse(JsonNode root) {
    known(
        root,
        Set.of(
            "model_type",
            "architectures",
            "text_config",
            "vision_config",
            "scale_factor",
            "image_token_id",
            "vocab_size",
            "hidden_size",
            "tie_word_embeddings",
            "torch_dtype",
            "dtype",
            "transformers_version",
            "use_cache",
            "pad_token_id",
            "transformers.js_config"));
    if (!"idefics3".equals(root.path("model_type").asString())) {
      throw new IllegalArgumentException("expected model_type=idefics3");
    }
    if (!root.path("text_config").isObject()) {
      throw new IllegalArgumentException("text_config must be an object");
    }
    ObjectNode nested = (ObjectNode) root.required("text_config").deepCopy();
    if (!"llama".equals(nested.path("model_type").asString())) {
      throw new IllegalArgumentException("Idefics3 requires a llama text_config");
    }
    // These historical export fields are explicitly inert in the pinned artifact.
    for (String key :
        Set.of(
            "perceiver_config",
            "pixel_shuffle_factor",
            "use_resampler",
            "qk_layer_norms",
            "is_llama_config",
            "transformers.js_config",
            "_flash_attn_2_enabled",
            "id2label",
            "label2id",
            "generation_config")) {
      nested.remove(key);
    }
    removeInertExportFields(nested);
    boolean tied = root.path("tie_word_embeddings").asBoolean(false);
    if (nested.has("tie_word_embeddings")
        && nested.path("tie_word_embeddings").asBoolean() != tied) {
      throw new IllegalArgumentException("conflicting tie_word_embeddings");
    }
    nested.put("tie_word_embeddings", tied);
    ArchitectureDescriptor text =
        ArchitectureMappings.parse(
            nested,
            key -> {
              throw new IllegalArgumentException("unrecognized text_config key '" + key + "'");
            });
    if (text.quantization() != null) {
      throw new IllegalArgumentException("quantized vision-language checkpoints are unsupported");
    }
    for (String key : Set.of("vocab_size", "hidden_size")) {
      int expected =
          key.equals("vocab_size") ? text.dimensions().vocabSize() : text.dimensions().hiddenSize();
      if (root.has(key) && positive(root, key) != expected) {
        throw new IllegalArgumentException("outer " + key + " disagrees with text_config");
      }
    }
    if (!root.path("vision_config").isObject()) {
      throw new IllegalArgumentException("vision_config must be an object");
    }
    ObjectNode v = (ObjectNode) root.required("vision_config").deepCopy();
    removeInertExportFields(v);
    known(
        v,
        Set.of(
            "model_type",
            "hidden_size",
            "intermediate_size",
            "image_size",
            "patch_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_channels",
            "hidden_act",
            "layer_norm_eps",
            "attention_dropout",
            "initializer_range",
            "use_base_siglip",
            "tie_word_embeddings",
            "size",
            "max_image_size",
            "projection_class",
            "ignore_index",
            "pad_token_id",
            "torch_dtype",
            "dtype",
            "transformers_version",
            "architectures"));
    if (!Set.of("idefics3_vision", "siglip_vision_model", "idefics3")
            .contains(v.path("model_type").asString())
        || !"gelu_pytorch_tanh".equals(v.path("hidden_act").asString())
        || (v.has("num_channels") && positive(v, "num_channels") != 3)
        || (v.has("use_base_siglip") && !v.path("use_base_siglip").asBoolean())) {
      throw new IllegalArgumentException("unsupported Idefics3 vision configuration");
    }
    int image = positive(v, "image_size");
    int patch = positive(v, "patch_size");
    int scale = positive(root, "scale_factor");
    int hidden = positive(v, "hidden_size");
    int heads = positive(v, "num_attention_heads");
    final int layers = positive(v, "num_hidden_layers");
    if (image % patch != 0 || (image / patch) % scale != 0 || hidden % heads != 0) {
      throw new IllegalArgumentException("incompatible patch/scale/head dimensions");
    }
    if (!v.path("layer_norm_eps").isNumber()) {
      throw new IllegalArgumentException("layer_norm_eps must be numeric");
    }
    float eps = v.required("layer_norm_eps").floatValue();
    if (!Float.isFinite(eps) || eps <= 0) {
      throw new IllegalArgumentException("layer_norm_eps must be positive and finite");
    }
    Map<String, String> keys = new LinkedHashMap<>();
    keys.put("embeddings.patch_embedding.weight", "patchEmbedding.weight");
    keys.put("embeddings.patch_embedding.bias", "patchEmbedding.bias");
    keys.put("embeddings.position_embedding.weight", "positionEmbedding.weight");
    keys.put("post_layernorm.weight", "postLayernorm.weight");
    keys.put("post_layernorm.bias", "postLayernorm.bias");
    for (int i = 0; i < layers; i++) {
      String hf = "encoder.layers." + i + ".";
      String local = "layer" + i + ".";
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
          keys.put(hf + entry.getKey() + suffix, local + entry.getValue() + suffix);
        }
      }
    }
    if (!root.path("image_token_id").isIntegralNumber()
        || !root.path("image_token_id").canConvertToInt()) {
      throw new IllegalArgumentException("image_token_id must be an integer");
    }
    int imageId = root.required("image_token_id").intValue();
    if (imageId < 0 || imageId >= text.dimensions().vocabSize()) {
      throw new IllegalArgumentException("image_token_id outside vocabulary");
    }
    return new SmolVlmDescriptor(
        text,
        imageId,
        scale,
        positive(nested, "max_position_embeddings"),
        image,
        patch,
        hidden,
        positive(v, "intermediate_size"),
        layers,
        heads,
        eps,
        Map.copyOf(keys));
  }

  int tokensPerTile() {
    int side = imageSize / patchSize / scale;
    return Math.multiplyExact(side, side);
  }

  TensorPlan tensorPlan() {
    TensorPlan decoder = ArchitectureMappings.tensorPlan(text);
    Set<String> required = new TreeSet<>();
    Set<String> optional = new TreeSet<>();
    decoder.required().forEach(k -> required.add(textKey(k)));
    decoder.optional().forEach(k -> optional.add(textKey(k)));
    visionKeys.keySet().forEach(k -> required.add("model.vision_model." + k));
    required.add("model.connector.modality_projection.proj.weight");
    return new TensorPlan(required, optional, Set.of(), Set.of());
  }

  static String textKey(String name) {
    return name.startsWith("model.") ? "model.text_model." + name.substring(6) : name;
  }

  private static void known(JsonNode node, Set<String> keys) {
    if (!node.isObject()) {
      throw new IllegalArgumentException("configuration must be an object");
    }
    for (var entry : node.properties()) {
      if (!keys.contains(entry.getKey())) {
        throw new IllegalArgumentException(
            "unrecognized Idefics3 config key '" + entry.getKey() + "'");
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
