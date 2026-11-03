package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/** Native-free validation against actual saved converter output and malformed checkpoints. */
class Qwen35CheckpointTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final Path FIXTURES = Path.of("..", "tools", "qwen35-reference", "fixtures");
  @TempDir Path temporary;

  @Test
  void actualSavedFloatAndAffineArtifactsClassifyByInventory() throws IOException {
    for (String kind : List.of("hf", "vlm-float", "vlm-affine", "lm-float", "lm-affine")) {
      Path directory = FIXTURES.resolve(kind);
      Qwen35Checkpoint.Checked checked = Qwen35Checkpoint.inspect(directory, config(kind));
      assertEquals(!kind.equals("hf"), checked.converted(), kind);
      assertEquals(!kind.startsWith("lm-"), checked.imageCapable(), kind);
      assertEquals(kind.endsWith("affine"), checked.descriptor().affine(), kind);
      assertEquals(4, checked.descriptor().text().layers());
      assertEquals(List.of(true, true, true, false), checked.descriptor().text().linearLayers());
      assertEquals(120, checked.descriptor().tokens().imageTokenId());
      assertEquals(kind.equals("hf") ? 11 : 0, checked.shiftedNorms().size());
      assertEquals(kind.equals("hf") ? 3 : 0, checked.transposedConvolutions().size());
      assertFalse(checked.names().isEmpty());
      assertTrue(checked.names().containsKey("lm_head.weight"));
    }
  }

  @Test
  void recordedRealArtifactHeadersPassWithoutAllocatingWeights() throws IOException {
    Path evidence = FIXTURES.getParent().resolve("evidence");
    for (String artifact : List.of("hf-08b", "mlx-27b")) {
      var config = MAPPER.readTree(evidence.resolve(artifact + "-config.json").toFile());
      var record = MAPPER.readTree(evidence.resolve(artifact + "-inventory.json").toFile());
      Map<String, SafetensorsHeaders.TensorInfo> inventory = new LinkedHashMap<>();
      for (var entry : record.get("tensors").properties()) {
        var value = entry.getValue();
        List<Integer> shape = new ArrayList<>();
        value.get("shape").forEach(d -> shape.add(d.asInt()));
        inventory.put(
            entry.getKey(),
            new SafetensorsHeaders.TensorInfo(
                Path.of(value.get("shard").asString()),
                value.get("dtype").asString(),
                shape,
                value.get("data_offsets").get(0).asLong(),
                value.get("data_offsets").get(1).asLong()));
      }
      Qwen35Checkpoint.Checked checked = Qwen35Checkpoint.classify(config, inventory);
      assertTrue(checked.imageCapable());
      assertEquals(artifact.equals("mlx-27b"), checked.converted());
      assertEquals(artifact.equals("mlx-27b"), checked.descriptor().affine());
      assertEquals(64, checked.descriptor().text().rotaryDim());
      assertEquals(2304, checked.descriptor().vision().positions());
      assertTrue(checked.names().size() < inventory.size() || checked.converted());
    }
  }

  @Test
  void strictConfigRejectsAmbiguousOrUnsupportedMathAndMetadata() throws IOException {
    rejectConfig(n -> text(n).put("partial_rotary_factor", .5), "partial_rotary_factor");
    rejectConfig(n -> rope(n).remove("partial_rotary_factor"), "partial_rotary_factor");
    rejectConfig(n -> rope(n).put("mrope_interleaved", false), "mrope_interleaved");
    rejectConfig(n -> rope(n).remove("mrope_interleaved"), "mrope_interleaved");
    rejectConfig(n -> rope(n).put("rope_type", "yarn"), "rope_type");
    rejectConfig(n -> rope(n).put("partial_rotary_factor", .24), "partial_rotary_factor");
    rejectConfig(n -> rope(n).putArray("mrope_section").add(3).add(3).add(3), "mrope_section");
    rejectConfig(n -> text(n).put("full_attention_interval", 2), "schedules disagree");
    rejectConfig(n -> text(n).put("linear_num_key_heads", 4), "key heads");
    rejectConfig(n -> text(n).put("num_key_value_heads", 3), "key heads");
    rejectConfig(n -> text(n).put("hidden_act", "gelu"), "hidden_act");
    rejectConfig(n -> text(n).put("attention_bias", true), "attention_bias");
    rejectConfig(n -> text(n).put("attn_output_gate", false), "attn_output_gate");
    rejectConfig(n -> text(n).put("output_gate_type", "sigmoid"), "output_gate_type");
    rejectConfig(n -> text(n).put("rms_norm_eps", 0), "rms_norm_eps");
    rejectConfig(n -> text(n).put("num_experts", 8), "num_experts");
    rejectConfig(n -> text(n).put("vocab_size", Integer.MAX_VALUE), "dimensions");
    rejectConfig(n -> n.put("language_model_only", true), "language_model_only");
    rejectConfig(n -> n.put("language_model_only", "false"), "language_model_only");
    rejectConfig(n -> n.put("image_token_id", 248056), "image_token_id");
    rejectConfig(n -> n.put("vision_end_token_id", 120), "distinct");
    rejectConfig(n -> n.put("unknown_future_field", false), "unknown_future_field");
    rejectConfig(n -> n.put("temperature", "warm"), "temperature");
    rejectConfig(n -> n.putObject("generation_config").put("eos_token_id", 1000), "eos_token_id");
    rejectConfig(
        n -> vision(n).putArray("deepstack_visual_indexes").add(0), "deepstack_visual_indexes");
    rejectConfig(n -> vision(n).put("out_hidden_size", 96), "geometry");
    rejectConfig(n -> vision(n).put("model_type", "qwen3_vl"), "model_type");
    ObjectNode config = config("hf");
    text(config).put("full_attention_interval", 4);
    text(config).put("partial_rotary_factor", .25);
    config.put("language_model_only", false);
    vision(config).put("model_type", "qwen3_5");
    assertEquals(16, Qwen35Descriptor.parse(config).text().rotaryDim());
    text(config).remove("layer_types");
    assertEquals(
        List.of(true, true, true, false), Qwen35Descriptor.parse(config).text().linearLayers());
    ObjectNode lm = config("lm-float");
    assertThrows(IllegalArgumentException.class, () -> Qwen35Descriptor.parse(lm));
    assertEquals(16, Qwen35Descriptor.parse(lm, true).text().rotaryDim());
    rope(lm).put("rope_type", "yarn");
    assertThrows(IllegalArgumentException.class, () -> Qwen35Descriptor.parse(lm, true));
  }

  @Test
  void inventoryRejectsMixedAxesPrefixesMtpPartialVisionAndUnknownWeights() throws IOException {
    rejectHeader(
        "vlm-float",
        h -> rename(h, "language_model.model.norm.weight", "model.language_model.norm.weight"),
        "mixed");
    rejectHeader(
        "vlm-float",
        h -> rename(h, "language_model.model.norm.weight", "language_model.model.mtp.norm.weight"),
        "MTP");
    rejectHeader(
        "vlm-float",
        h ->
            h.get("language_model.model.layers.0.linear_attn.conv1d.weight")
                .withArray("shape")
                .removeAll()
                .add(512)
                .add(1)
                .add(4),
        "conv1d");
    rejectHeader(
        "hf",
        h -> rename(h, "model.visual.pos_embed.weight", "model.visual.unknown.weight"),
        "missing tensor");
    rejectHeader(
        "hf",
        h -> rename(h, "model.language_model.norm.weight", "model.language_model.unknown.weight"),
        "missing tensor");
    rejectHeader(
        "hf",
        h -> rename(h, "model.language_model.norm.weight", "mtp.unknown.weight"),
        "missing tensor");
    rejectHeader(
        "vlm-affine",
        h ->
            rename(
                h,
                "language_model.model.layers.0.linear_attn.in_proj_a.scales",
                "language_model.model.layers.0.linear_attn.in_proj_a.wrong"),
        "unexpected tensor");
    rejectHeader(
        "vlm-affine",
        h ->
            ((ObjectNode) h.get("language_model.model.layers.0.linear_attn.norm.weight"))
                .put("dtype", "I32"),
        "dtype");
    rejectHeader(
        "hf",
        h ->
            h.get("model.language_model.norm.weight")
                .withArray("data_offsets")
                .removeAll()
                .add(0)
                .add(512),
        "ranges");
  }

  @Test
  void tiedValuesAreCheckedBeforeUploadAndIdenticalOptionalHeadIsNotRetained() throws Exception {
    ObjectNode config = config("hf");
    config.put("tie_word_embeddings", true);
    text(config).put("tie_word_embeddings", true);
    IllegalArgumentException mismatch =
        assertThrows(
            IllegalArgumentException.class,
            () -> Qwen35Checkpoint.inspect(FIXTURES.resolve("hf"), config));
    assertTrue(mismatch.getMessage().contains("tied lm_head"));
    // A temporary synthetic HF alias, not converter evidence or a replacement golden.
    Path original = FIXTURES.resolve("hf/model.safetensors");
    byte[] bytes = Files.readAllBytes(original);
    int headerSize =
        Math.toIntExact(ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getLong());
    ObjectNode header =
        (ObjectNode)
            MAPPER.readTree(
                new String(bytes, 8, headerSize, java.nio.charset.StandardCharsets.UTF_8));
    var embedding = header.get("model.language_model.embed_tokens.weight").get("data_offsets");
    var head = header.get("lm_head.weight").get("data_offsets");
    int count = head.get(1).asInt() - head.get(0).asInt();
    System.arraycopy(
        bytes,
        8 + headerSize + embedding.get(0).asInt(),
        bytes,
        8 + headerSize + head.get(0).asInt(),
        count);
    Path directory = Files.createTempDirectory(temporary, "tied-alias");
    Files.write(directory.resolve("model.safetensors"), bytes);
    Qwen35Checkpoint.Checked checked = Qwen35Checkpoint.inspect(directory, config);
    assertFalse(checked.names().containsKey("lm_head.weight"));
    assertTrue(checked.inventory().containsKey("lm_head.weight"));
  }

  @Test
  void metadataNeverOverridesVisionCapabilityAndAliasesStayTyped() throws IOException {
    ObjectNode lm = config("lm-float");
    lm.put("language_model_only", false);
    assertFalse(Qwen35Checkpoint.inspect(FIXTURES.resolve("lm-float"), lm).imageCapable());
    ObjectNode full = config("vlm-affine");
    full.put("language_model_only", false);
    assertTrue(Qwen35Checkpoint.inspect(FIXTURES.resolve("vlm-affine"), full).imageCapable());
    ((ObjectNode) full.get("quantization_config")).put("bits", 8);
    assertThrows(IllegalArgumentException.class, () -> Qwen35Descriptor.parse(full));
  }

  @Test
  void quantizationModeIsOptionalWithAffineImplied() throws IOException {
    ObjectNode implicit = config("vlm-affine");
    ((ObjectNode) implicit.get("quantization")).remove("mode");
    ((ObjectNode) implicit.get("quantization_config")).remove("mode");
    assertTrue(Qwen35Descriptor.parse(implicit).affine());
    ObjectNode explicit = config("vlm-affine");
    ((ObjectNode) explicit.get("quantization")).put("mode", "linear");
    IllegalArgumentException error =
        assertThrows(IllegalArgumentException.class, () -> Qwen35Descriptor.parse(explicit));
    assertTrue(error.getMessage().contains("mode"), error.getMessage());
  }

  private void rejectHeader(String kind, Consumer<ObjectNode> mutation, String message)
      throws IOException {
    Path source;
    try (var files = Files.list(FIXTURES.resolve(kind))) {
      source = files.filter(p -> p.toString().endsWith(".safetensors")).findFirst().orElseThrow();
    }
    byte[] content = Files.readAllBytes(source);
    int length = Math.toIntExact(ByteBuffer.wrap(content).order(ByteOrder.LITTLE_ENDIAN).getLong());
    ObjectNode header =
        (ObjectNode)
            MAPPER.readTree(
                new String(content, 8, length, java.nio.charset.StandardCharsets.UTF_8));
    mutation.accept(header);
    byte[] changed = MAPPER.writeValueAsBytes(header);
    Path directory = Files.createTempDirectory(temporary, "bad-checkpoint");
    ByteBuffer output =
        ByteBuffer.allocate(8 + changed.length + content.length - 8 - length)
            .order(ByteOrder.LITTLE_ENDIAN);
    output
        .putLong(changed.length)
        .put(changed)
        .put(content, 8 + length, content.length - 8 - length);
    Files.write(directory.resolve("model.safetensors"), output.array());
    IllegalArgumentException error =
        assertThrows(
            IllegalArgumentException.class,
            () -> Qwen35Checkpoint.inspect(directory, config(kind)));
    assertTrue(error.getMessage().contains(message), error.getMessage());
  }

  private static void rejectConfig(Consumer<ObjectNode> mutation, String message)
      throws IOException {
    ObjectNode node = config("hf");
    mutation.accept(node);
    IllegalArgumentException error =
        assertThrows(IllegalArgumentException.class, () -> Qwen35Descriptor.parse(node));
    assertTrue(error.getMessage().contains(message), error.getMessage());
  }

  private static ObjectNode config(String kind) throws IOException {
    return (ObjectNode) MAPPER.readTree(FIXTURES.resolve(kind).resolve("config.json").toFile());
  }

  private static ObjectNode text(ObjectNode node) {
    return (ObjectNode) node.get("text_config");
  }

  private static ObjectNode rope(ObjectNode node) {
    return (ObjectNode) text(node).get("rope_parameters");
  }

  private static ObjectNode vision(ObjectNode node) {
    return (ObjectNode) node.get("vision_config");
  }

  private static void rename(ObjectNode node, String before, String after) {
    node.set(after, node.remove(before));
  }
}
