package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.nn.RopeSpec;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ArrayNode;
import tools.jackson.databind.node.ObjectNode;

/**
 * Strict Qwen3-VL config parsing, pure: the pinned 2B artifact's config parses to the expected
 * components and every structural cross-check (findings §3) rejects by name. No native work.
 */
class Qwen3VlDescriptorTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  /** The pinned Qwen3-VL-2B-Instruct outer config (findings §3.1-3.3). */
  static ObjectNode pinnedConfig() {
    ObjectNode root = MAPPER.createObjectNode();
    root.put("model_type", "qwen3_vl");
    ArrayNode architectures = root.putArray("architectures");
    architectures.add("Qwen3VLForConditionalGeneration");
    root.put("tie_word_embeddings", true);

    ObjectNode text = root.putObject("text_config");
    text.put("attention_bias", false);
    text.put("attention_dropout", 0.0);
    text.put("bos_token_id", 151643);
    text.put("dtype", "bfloat16");
    text.put("eos_token_id", 151645);
    text.put("head_dim", 128);
    text.put("hidden_act", "silu");
    text.put("hidden_size", 2048);
    text.put("initializer_range", 0.02);
    text.put("intermediate_size", 6144);
    text.put("max_position_embeddings", 262144);
    text.put("model_type", "qwen3_vl_text");
    text.put("num_attention_heads", 16);
    text.put("num_hidden_layers", 28);
    text.put("num_key_value_heads", 8);
    text.put("rms_norm_eps", 0.000001);
    text.put("rope_theta", 5000000);
    ObjectNode scaling = text.putObject("rope_scaling");
    scaling.put("rope_type", "default");
    scaling.put("mrope_interleaved", true);
    ArrayNode section = scaling.putArray("mrope_section");
    section.add(24);
    section.add(20);
    section.add(20);
    text.put("tie_word_embeddings", true);
    text.put("use_cache", true);
    text.put("vocab_size", 151936);

    ObjectNode vision = root.putObject("vision_config");
    ArrayNode deepstack = vision.putArray("deepstack_visual_indexes");
    deepstack.add(5);
    deepstack.add(11);
    deepstack.add(17);
    vision.put("depth", 24);
    vision.put("hidden_act", "gelu_pytorch_tanh");
    vision.put("hidden_size", 1024);
    vision.put("in_channels", 3);
    vision.put("initializer_range", 0.02);
    vision.put("intermediate_size", 4096);
    vision.put("model_type", "qwen3_vl");
    vision.put("num_heads", 16);
    vision.put("num_position_embeddings", 2304);
    vision.put("out_hidden_size", 2048);
    vision.put("patch_size", 16);
    vision.put("spatial_merge_size", 2);
    vision.put("temporal_patch_size", 2);

    root.put("image_token_id", 151655);
    root.put("video_token_id", 151656);
    root.put("vision_start_token_id", 151652);
    root.put("vision_end_token_id", 151653);
    root.put("transformers_version", "4.57.0.dev0");
    return root;
  }

  private static Qwen3VlDescriptor parse() {
    return Qwen3VlDescriptor.parse(pinnedConfig());
  }

  private static void assertRejected(ObjectNode config, String expected) {
    IllegalArgumentException failure =
        assertThrows(IllegalArgumentException.class, () -> Qwen3VlDescriptor.parse(config));
    assertTrue(
        failure.getMessage().contains(expected),
        () -> "expected message containing " + expected + ", got " + failure.getMessage());
  }

  /** {@code text_config} as an object node: the base getter returns the read-only view type. */
  private static ObjectNode text(ObjectNode c) {
    return (ObjectNode) c.get("text_config");
  }

  /** {@code vision_config} as an object node. */
  private static ObjectNode vision(ObjectNode c) {
    return (ObjectNode) c.get("vision_config");
  }

  /** {@code text_config.rope_scaling} as an object node. */
  private static ObjectNode scaling(ObjectNode c) {
    return (ObjectNode) text(c).get("rope_scaling");
  }

  /** {@code text_config.rope_scaling.mrope_section} as an array node. */
  private static ArrayNode section(ObjectNode c) {
    return (ArrayNode) scaling(c).get("mrope_section");
  }

  @Test
  void pinnedTwoBConfigParses() {
    Qwen3VlDescriptor d = parse();
    DecoderConfig c = d.text().dimensions();
    assertEquals("qwen3_vl_text", c.modelType());
    assertEquals(151936, c.vocabSize());
    assertEquals(2048, c.hiddenSize());
    assertEquals(6144, c.intermediateSize());
    assertEquals(28, c.numHiddenLayers());
    assertEquals(16, c.numAttentionHeads());
    assertEquals(8, c.numKeyValueHeads());
    assertEquals(5000000.0f, c.ropeTheta());
    assertEquals(1e-6f, c.rmsNormEps());
    assertTrue(c.tieWordEmbeddings());
    assertEquals(128, d.text().headDim());
    assertEquals(128, d.text().rotaryDims());
    assertEquals(new RopeSpec.Base(5000000.0f), d.text().rope());
    assertEquals(
        new ArchitectureDescriptor.Norm(ArchitectureDescriptor.NormKind.RMS, 1e-6f, false),
        d.text().norm());
    assertEquals(
        new ArchitectureDescriptor.Mlp(
            ArchitectureDescriptor.MlpLayout.SEPARATE_GATE_UP,
            se.alipsa.jmlx.nn.Activation.SILU,
            false),
        d.text().mlp());
    assertEquals(
        new ArchitectureDescriptor.Attention(false, false, false, null, true),
        d.text().attention());
    assertEquals(new ArchitectureDescriptor.Head(true), d.text().head());
    assertEquals(new ArchitectureDescriptor.Embedding(false), d.text().embedding());
    assertEquals(null, d.text().moe());
    assertEquals(null, d.text().quantization());
    assertEquals(new Qwen3VlPromptTokens(151655, -1, 151652, 151653), d.tokens());
    assertEquals(151656, d.videoTokenId());
    // A config that declares no video token carries -1 on the descriptor.
    ObjectNode noVideo = pinnedConfig();
    noVideo.remove("video_token_id");
    assertEquals(-1, Qwen3VlDescriptor.parse(noVideo).videoTokenId());
    assertEquals(262144, d.contextLength());
    assertEquals(24, d.depth());
    assertEquals(16, d.numHeads());
    assertEquals(1024, d.visionHidden());
    assertEquals(4096, d.visionIntermediate());
    assertEquals(16, d.patchSize());
    assertEquals(2, d.temporalPatchSize());
    assertEquals(2, d.mergeSize());
    assertEquals(3, d.inChannels());
    assertEquals(2048, d.outHiddenSize());
    assertEquals(2304, d.numPositionEmbeddings());
    assertArrayEquals(new int[] {5, 11, 17}, d.deepstackIndexes());
    assertArrayEquals(new int[] {24, 20, 20}, d.mropeSection());
  }

  @Test
  void tensorPlanCarriesTheRewrittenTextPrefixAndExactVisualSet() {
    TensorPlan plan = parse().tensorPlan();
    assertTrue(plan.required().contains("model.language_model.embed_tokens.weight"));
    assertTrue(plan.required().contains("model.language_model.norm.weight"));
    assertTrue(plan.required().contains("model.language_model.layers.27.self_attn.q_norm.weight"));
    assertTrue(plan.required().contains("model.visual.blocks.0.attn.qkv.weight"));
    assertTrue(plan.required().contains("model.visual.blocks.23.mlp.linear_fc2.bias"));
    assertTrue(plan.required().contains("model.visual.pos_embed.weight"));
    assertTrue(plan.required().contains("model.visual.deepstack_merger_list.2.linear_fc2.bias"));
    // The bare unprefixed text names must not be expected, and the tied lm_head stays optional.
    assertFalse(plan.required().contains("model.layers.0.self_attn.q_proj.weight"));
    assertFalse(plan.required().contains("lm_head.weight"));
    assertTrue(plan.optional().contains("lm_head.weight"));
    assertTrue(
        plan.forbidden().contains("model.language_model.layers.0.self_attn.qkv_proj.weight"));
    // The text plan's inv_freq ignore pattern matches the rewritten name.
    assertTrue(
        plan.ignored().stream()
            .anyMatch(p -> p.matcher("model.language_model.rotary_emb.inv_freq").matches()));
    // Required text set: embed + norm + 28 layers * 11, no lm_head.
    long textRequired =
        plan.required().stream().filter(k -> k.startsWith("model.language_model.")).count();
    assertEquals(2 + 28L * 11, textRequired);
    // Required visual set: 3 prefix + 24 blocks * 12 + (1 main + 3 deepstack) mergers * 6.
    long visualRequired =
        plan.required().stream().filter(k -> k.startsWith("model.visual.")).count();
    assertEquals(3 + 24L * 12 + 4L * 6, visualRequired);
    assertDoesNotThrow(() -> plan.validate(new java.util.HashSet<>(plan.required())));
  }

  @Test
  void outerRejections() {
    ObjectNode c = pinnedConfig();
    c.put("unknown_outer_key", 1);
    assertRejected(c, "unrecognized key 'unknown_outer_key'");

    c = pinnedConfig();
    c.put("model_type", "qwen3_vl_moe");
    assertRejected(c, "model_type must be qwen3_vl");

    c = pinnedConfig();
    c.putArray("architectures").add("Qwen3VLForConditionalGeneration").add("Extra");
    assertRejected(c, "architectures must be exactly");

    c = pinnedConfig();
    c.putArray("architectures").add("Qwen3VLMoeForConditionalGeneration");
    assertRejected(c, "architectures must be exactly");

    c = pinnedConfig();
    c.put("tie_word_embeddings", "true");
    assertRejected(c, "tie_word_embeddings must be a boolean");

    c = pinnedConfig();
    c.put("transformers_version", 4);
    assertRejected(c, "transformers_version must be a string");

    c = pinnedConfig();
    c.put("image_token_id", 151936);
    assertRejected(c, "outside the text vocabulary");

    c = pinnedConfig();
    c.put("video_token_id", 151936);
    assertRejected(c, "outside the text vocabulary");

    c = pinnedConfig();
    c.put("vision_start_token_id", -1);
    assertRejected(c, "outside the text vocabulary");
  }

  @Test
  void textConfigRejections() {
    ObjectNode c = pinnedConfig();
    text(c).put("unknown_text_key", 1);
    assertRejected(c, "text_config has unrecognized key 'unknown_text_key'");

    c = pinnedConfig();
    text(c).put("model_type", "qwen3");
    assertRejected(c, "must be qwen3_vl_text");

    c = pinnedConfig();
    text(c).put("attention_bias", true);
    assertRejected(c, "attention_bias must be false");

    c = pinnedConfig();
    text(c).put("hidden_act", "gelu");
    assertRejected(c, "hidden_act must be silu");

    c = pinnedConfig();
    text(c).put("num_key_value_heads", 3);
    assertRejected(c, "must be a multiple of num_key_value_heads");

    c = pinnedConfig();
    text(c).put("hidden_size", 2040);
    assertRejected(c, "must be a multiple of num_attention_heads");

    c = pinnedConfig();
    section(c).set(2, MAPPER.getNodeFactory().numberNode(19));
    assertRejected(c, "must equal head_dim 128");

    c = pinnedConfig();
    scaling(c).put("extra", 1);
    assertRejected(c, "must hold exactly rope_type, mrope_interleaved and mrope_section");

    c = pinnedConfig();
    scaling(c).put("rope_type", "yarn");
    assertRejected(c, "rope_type must be default");

    c = pinnedConfig();
    scaling(c).put("mrope_interleaved", false);
    assertRejected(c, "mrope_interleaved must be true");

    c = pinnedConfig();
    scaling(c).set("mrope_section", MAPPER.createArrayNode().add(24).add(44));
    assertRejected(c, "must hold exactly 3 entries");

    c = pinnedConfig();
    section(c).set(0, MAPPER.getNodeFactory().numberNode(-1));
    assertRejected(c, "non-negative integers");

    c = pinnedConfig();
    scaling(c).set("mrope_section", MAPPER.createArrayNode().add(0).add(0).add(0));
    assertRejected(c, "rotate at least one dimension pair");

    c = pinnedConfig();
    text(c).put("rope_scaling", "default");
    assertRejected(c, "rope_scaling must be an object");

    c = pinnedConfig();
    text(c).put("tie_word_embeddings", false);
    assertRejected(c, "conflicting tie_word_embeddings");

    c = pinnedConfig();
    text(c).remove("max_position_embeddings");
    assertRejected(c, "max_position_embeddings must be a positive integer");

    c = pinnedConfig();
    text(c).put("head_dim", 0);
    assertRejected(c, "head_dim must be a positive integer");

    c = pinnedConfig();
    text(c).put("rms_norm_eps", -1);
    assertRejected(c, "rms_norm_eps must be positive and finite");

    c = pinnedConfig();
    text(c).put("rope_theta", 0);
    assertRejected(c, "rope_theta must be positive and finite");
  }

  @Test
  void visionConfigRejections() {
    ObjectNode c = pinnedConfig();
    vision(c).put("unknown_vision_key", 1);
    assertRejected(c, "vision_config has unrecognized key 'unknown_vision_key'");

    c = pinnedConfig();
    vision(c).put("model_type", "qwen2_vl");
    assertRejected(c, "must be qwen3_vl");

    c = pinnedConfig();
    vision(c).put("hidden_act", "silu");
    assertRejected(c, "must be gelu_pytorch_tanh");

    c = pinnedConfig();
    vision(c).put("in_channels", 4);
    assertRejected(c, "in_channels must be 3");

    c = pinnedConfig();
    vision(c).put("num_heads", 15);
    assertRejected(c, "must divide hidden_size");

    c = pinnedConfig();
    vision(c).put("hidden_size", 1008);
    assertRejected(c, "head dimension 63 must be a multiple of 4");

    c = pinnedConfig();
    vision(c).put("out_hidden_size", 512);
    assertRejected(c, "out_hidden_size 512 must equal text_config hidden_size 2048");

    c = pinnedConfig();
    vision(c).set("deepstack_visual_indexes", MAPPER.createArrayNode());
    assertRejected(c, "must be a non-empty array");

    c = pinnedConfig();
    vision(c).set("deepstack_visual_indexes", MAPPER.createArrayNode().add(5).add(5));
    assertRejected(c, "strictly increasing");

    c = pinnedConfig();
    vision(c).set("deepstack_visual_indexes", MAPPER.createArrayNode().add(11).add(5));
    assertRejected(c, "strictly increasing");

    c = pinnedConfig();
    vision(c).set("deepstack_visual_indexes", MAPPER.createArrayNode().add(24));
    assertRejected(c, "out of range [0, 24)");

    c = pinnedConfig();
    vision(c).put("num_position_embeddings", 2303);
    assertRejected(c, "perfect square");

    c = pinnedConfig();
    vision(c).put("depth", 0);
    assertRejected(c, "depth must be a positive integer");

    c = pinnedConfig();
    vision(c).put("patch_size", -16);
    assertRejected(c, "patch_size must be a positive integer");
  }
}
