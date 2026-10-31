package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.nn.RopeSpec;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

/** Pure-JSON strict parse of the phi3_v config: the allow-list, both vision blocks and the plan. */
class Phi3VDescriptorTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private static JsonNode config(String json) {
    try {
      return MAPPER.readTree(json);
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  private static String variant(String key, String value) {
    return TinyCheckpoints.phi3VisionConfigJson()
        .replace("\"vocab_size\": 21", "\"" + key + "\": " + value);
  }

  @Test
  void parsesTheTinyConfigWithSuRopeAndRecordedVisionContract() {
    Phi3VDescriptor descriptor =
        Phi3VDescriptor.parse(config(TinyCheckpoints.phi3VisionConfigJson()));
    assertEquals("phi3_v", descriptor.text().dimensions().modelType());
    assertEquals(21, descriptor.text().dimensions().vocabSize());
    assertEquals(128, descriptor.text().dimensions().hiddenSize());
    assertEquals(144, descriptor.numImgTokens());
    assertEquals("image", descriptor.embeddingCls());
    RopeSpec.Su su = assertInstanceOf(RopeSpec.Su.class, descriptor.text().rope());
    assertEquals(2048, su.maxPositions());
    assertEquals(470, su.originalMaxPositions());
    assertEquals(32, su.shortFactors().length);
    assertEquals(32, su.longFactors().length);
    assertTrue(descriptor.text().attention().fusedQkv());
    assertEquals(ArchitectureDescriptor.MlpLayout.FUSED_GATE_UP, descriptor.text().mlp().layout());
  }

  @Test
  void acceptsThePinnedArtifactsExtraMetadataKeys() {
    String json =
        TinyCheckpoints.phi3VisionConfigJson()
            .replace(
                "\"model_type\": \"phi3_v\",",
                "\"_attn_implementation\": \"flash_attention_2\","
                    + "\"_name_or_path\": \"Phi-3.5-vision-instruct\","
                    + "\"auto_map\": {\"AutoModelForCausalLM\": \"x\"},"
                    + "\"embd_pdrop\": 0.0,"
                    + "\"initializer_range\": 0.02,"
                    + "\"pad_token_id\": 32000,"
                    + "\"resid_pdrop\": 0.0,"
                    + "\"sliding_window\": 262144,"
                    + "\"torch_dtype\": \"bfloat16\","
                    + "\"transformers_version\": \"4.38.1\","
                    + "\"use_cache\": true,"
                    + "\"model_type\": \"phi3_v\",");
    Phi3VDescriptor descriptor = Phi3VDescriptor.parse(config(json));
    assertEquals(144, descriptor.numImgTokens());
  }

  @Test
  void rejectsUnknownTopLevelKeys() {
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class,
            () -> Phi3VDescriptor.parse(config(variant("bogus_key", "1"))));
    assertEquals("unrecognized phi3_v config key 'bogus_key'", failure.getMessage());
  }

  @Test
  void rejectsTheWrongModelTypeAndArchitectures() {
    IllegalArgumentException modelType =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Phi3VDescriptor.parse(
                    config(TinyCheckpoints.phi3VisionConfigJson().replace("phi3_v", "phi3"))));
    assertEquals("expected model_type=phi3_v", modelType.getMessage());
    IllegalArgumentException architectures =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Phi3VDescriptor.parse(
                    config(
                        TinyCheckpoints.phi3VisionConfigJson()
                            .replace(
                                "[\"Phi3VForCausalLM\"]", "[\"Phi3VForConditionalGeneration\"]"))));
    assertEquals("architectures must be [Phi3VForCausalLM]", architectures.getMessage());
    IllegalArgumentException multi =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Phi3VDescriptor.parse(
                    config(
                        TinyCheckpoints.phi3VisionConfigJson()
                            .replace(
                                "[\"Phi3VForCausalLM\"]", "[\"Phi3VForCausalLM\", \"Other\"]"))));
    assertEquals("architectures must be [Phi3VForCausalLM]", multi.getMessage());
  }

  @Test
  void rejectsImgProcessorDeviationsFromThePinnedValues() {
    String base = TinyCheckpoints.phi3VisionConfigJson();
    assertEquals(
        "img_processor.name must be clip_vision_model",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(base.replace("\"clip_vision_model\"", "\"other_processor\""))))
            .getMessage());
    assertEquals(
        "img_processor.model_name must be openai/clip-vit-large-patch14-336",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(
                            base.replace(
                                "openai/clip-vit-large-patch14-336", "openai/other-clip"))))
            .getMessage());
    assertEquals(
        "img_processor.image_dim_out must be 1024",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(base.replace("\"image_dim_out\": 1024", "\"image_dim_out\": 512"))))
            .getMessage());
    assertEquals(
        "image_dim_out must be a positive integer",
        assertThrows(
                IllegalArgumentException.class,
                () -> Phi3VDescriptor.parse(config(base.replace("\"image_dim_out\": 1024,", ""))))
            .getMessage());
    // Optional keys only accepted at the bundled modeling defaults.
    assertEquals(
        "img_processor.layer_idx must be -2",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(
                            base.replace(
                                "\"num_img_tokens\": 144",
                                "\"num_img_tokens\": 144, \"layer_idx\": -3"))))
            .getMessage());
    IllegalArgumentException typeFeature =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Phi3VDescriptor.parse(
                    config(
                        base.replace(
                            "\"num_img_tokens\": 144",
                            "\"num_img_tokens\": 144, \"type_feature\": \"pooled\""))));
    assertEquals("img_processor.type_feature must be patch", typeFeature.getMessage());
    // Unknown keys in the block are rejected by name.
    IllegalArgumentException unknown =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                Phi3VDescriptor.parse(
                    config(
                        base.replace(
                            "\"num_img_tokens\": 144", "\"num_img_tokens\": 144, \"bogus\": 1"))));
    assertEquals("unrecognized img_processor key 'bogus'", unknown.getMessage());
  }

  @Test
  void rejectsEmbdLayerDeviationsFromThePinnedValues() {
    String base = TinyCheckpoints.phi3VisionConfigJson();
    assertEquals(
        "embd_layer.projection_cls must be mlp",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(
                            base.replace(
                                "\"projection_cls\": \"mlp\"", "\"projection_cls\": \"linear\""))))
            .getMessage());
    assertEquals(
        "embd_layer.use_hd_transform must be true",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(
                            base.replace(
                                "\"use_hd_transform\": true", "\"use_hd_transform\": false"))))
            .getMessage());
    assertEquals(
        "embd_layer.hd_transform_order must be sub_glb",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(
                            base.replace(
                                "\"hd_transform_order\": \"sub_glb\"",
                                "\"hd_transform_order\": \"glb_sub\""))))
            .getMessage());
    // The bundled default ("false") differs from the pinned contract, so the key is required.
    assertEquals(
        "embd_layer.with_learnable_separator must be true",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(base.replace("\"with_learnable_separator\": true,", ""))))
            .getMessage());
    assertEquals(
        "embd_layer.embedding_cls must be a non-empty string",
        assertThrows(
                IllegalArgumentException.class,
                () ->
                    Phi3VDescriptor.parse(
                        config(
                            base.replace(
                                "\"embedding_cls\": \"image\"", "\"embedding_cls\": \"\""))))
            .getMessage());
  }

  @Test
  void rejectsRopeScalingThatIsNotSu() {
    String base = TinyCheckpoints.phi3VisionConfigJson();
    String yarn =
        base.replace(
            "{\"type\": \"su\", \"short_factor\":",
            "{\"type\": \"yarn\", \"rope_theta\": 10000, \"factor\": 1.0, \"short_factor\":");
    assertEquals(
        "config.json rope_scaling.rope_type 'yarn' is unsupported for phi3_v",
        assertThrows(IllegalArgumentException.class, () -> Phi3VDescriptor.parse(config(yarn)))
            .getMessage());
    String linear =
        base.replace(
            "{\"type\": \"su\", \"short_factor\":",
            "{\"type\": \"linear\", \"factor\": 2.0, \"short_factor\":");
    assertEquals(
        "config.json rope_scaling.rope_type 'linear' is unsupported for phi3_v",
        assertThrows(IllegalArgumentException.class, () -> Phi3VDescriptor.parse(config(linear)))
            .getMessage());
    String defaultType =
        base.replace(
            "\"rope_scaling\": {\"type\": \"su\", \"short_factor\": [",
            "\"rope_scaling\": {\"type\": \"default\", \"short_factor\": [");
    // The family only accepts su; the descriptor requires the spec outright.
    assertThrows(IllegalArgumentException.class, () -> Phi3VDescriptor.parse(config(defaultType)));
  }

  @Test
  void rejectsQuantizationKeysByTheTopLevelAllowList() {
    assertEquals(
        "unrecognized phi3_v config key 'quantization_config'",
        assertThrows(
                IllegalArgumentException.class,
                () -> Phi3VDescriptor.parse(config(variant("quantization_config", "{}"))))
            .getMessage());
  }

  @Test
  void tensorPlanNamesTheExecutedLayersRequiredAndTheTailOptional() {
    Phi3VDescriptor descriptor =
        Phi3VDescriptor.parse(config(TinyCheckpoints.phi3VisionConfigJson()));
    TensorPlan plan = descriptor.tensorPlan(3);
    String clip = "model.vision_embed_tokens.img_processor.vision_model.";
    assertTrue(plan.required().contains(clip + "encoder.layers.0.self_attn.q_proj.weight"));
    assertTrue(plan.required().contains(clip + "encoder.layers.1.mlp.fc2.bias"));
    assertFalse(plan.required().contains(clip + "encoder.layers.2.self_attn.q_proj.weight"));
    assertTrue(plan.optional().contains(clip + "encoder.layers.2.self_attn.out_proj.bias"));
    assertTrue(plan.optional().contains(clip + "post_layernorm.weight"));
    assertTrue(plan.optional().contains(clip + "post_layernorm.bias"));
    assertTrue(plan.forbidden().contains(clip + "embeddings.patch_embedding.bias"));
    assertTrue(plan.required().contains("model.vision_embed_tokens.img_projection.2.bias"));
    assertTrue(plan.required().contains("model.layers.0.self_attn.qkv_proj.weight"));
    assertTrue(plan.required().contains("model.layers.1.mlp.gate_up_proj.weight"));
    assertTrue(plan.required().contains("lm_head.weight"));
    // The two layers of the tiny text side plus the untied head and the fixed vision keys.
    long textKeys =
        plan.required().stream().filter(k -> !k.startsWith("model.vision_embed_tokens")).count();
    assertEquals(3 + 2 * 6, textKeys);
    long visionRequired =
        plan.required().stream().filter(k -> k.startsWith("model.vision_embed_tokens")).count();
    assertEquals(11 + 2 * 16, visionRequired);
  }
}
