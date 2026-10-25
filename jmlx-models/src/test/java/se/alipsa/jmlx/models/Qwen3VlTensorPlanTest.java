package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import tools.jackson.databind.ObjectMapper;
import tools.jackson.databind.node.ObjectNode;

/**
 * The Qwen3-VL checkpoint tensor plan against synthesized key sets, and the provenance selector
 * against hand-written shard headers: both run without native work.
 */
class Qwen3VlTensorPlanTest {

  private static final ObjectMapper MAPPER = new ObjectMapper();

  private static TensorPlan plan() {
    return Qwen3VlDescriptor.parse(Qwen3VlDescriptorTest.pinnedConfig()).tensorPlan();
  }

  private static void assertPlanRejected(TensorPlan plan, Set<String> available, String expected) {
    IllegalArgumentException failure =
        assertThrows(IllegalArgumentException.class, () -> plan.validate(available));
    assertTrue(
        failure.getMessage().contains(expected),
        () -> "expected message containing " + expected + ", got " + failure.getMessage());
  }

  @Test
  void requiredPlusOptionalPasses() {
    TensorPlan plan = plan();
    Set<String> available = new HashSet<>(plan.required());
    available.addAll(plan.optional());
    available.add("model.language_model.rotary_emb.inv_freq");
    assertDoesNotThrow(() -> plan.validate(available));
  }

  @Test
  void missingTensorsAreNamed() {
    TensorPlan plan = plan();
    Set<String> available = new HashSet<>(plan.required());
    available.remove("model.language_model.layers.0.self_attn.q_proj.weight");
    available.remove("model.visual.blocks.23.mlp.linear_fc2.bias");
    available.remove("model.visual.deepstack_merger_list.1.linear_fc1.weight");
    assertPlanRejected(
        plan,
        available,
        "missing tensor 'model.language_model.layers.0.self_attn.q_proj.weight'; missing tensor"
            + " 'model.visual.blocks.23.mlp.linear_fc2.bias'; missing tensor"
            + " 'model.visual.deepstack_merger_list.1.linear_fc1.weight'");
  }

  @Test
  void unexpectedAndOutOfSchemeTensorsAreNamed() {
    TensorPlan plan = plan();
    Set<String> available = new HashSet<>(plan.required());
    available.add("model.visual.rotary_emb.cos_cached");
    assertPlanRejected(plan, available, "unexpected tensor 'model.visual.rotary_emb.cos_cached'");

    // A block past the configured depth is not part of the exact visual set.
    available = new HashSet<>(plan.required());
    available.add("model.visual.blocks.24.norm1.weight");
    assertPlanRejected(plan, available, "unexpected tensor 'model.visual.blocks.24.norm1.weight'");

    // A DeepStack merger past the configured index list is rejected the same way.
    available = new HashSet<>(plan.required());
    available.add("model.visual.deepstack_merger_list.3.norm.weight");
    assertPlanRejected(
        plan, available, "unexpected tensor 'model.visual.deepstack_merger_list.3.norm.weight'");

    // The bare unprefixed text name is not expected: the plan rewrites it under the language
    // prefix.
    available = new HashSet<>(plan.required());
    available.add("model.layers.0.self_attn.q_proj.weight");
    assertPlanRejected(
        plan, available, "unexpected tensor 'model.layers.0.self_attn.q_proj.weight'");
  }

  @Test
  void forbiddenFusedQkvIsRejected() {
    TensorPlan plan = plan();
    Set<String> available = new HashSet<>(plan.required());
    available.add("model.language_model.layers.0.self_attn.qkv_proj.weight");
    assertPlanRejected(
        plan,
        available,
        "forbidden tensor 'model.language_model.layers.0.self_attn.qkv_proj.weight'");
  }

  // ---------------------------------------------------------------------------
  // Provenance selection over hand-written shard headers (8-byte little-endian length + JSON).
  // ---------------------------------------------------------------------------

  private static void writeShard(Path file, String headerJson) throws IOException {
    byte[] json = headerJson.getBytes(StandardCharsets.UTF_8);
    ByteBuffer buffer =
        ByteBuffer.allocate(Long.BYTES + json.length).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putLong(json.length);
    buffer.put(json);
    Files.write(file, buffer.array());
  }

  private static ObjectNode config() {
    return MAPPER.createObjectNode();
  }

  private static ObjectNode quant(ObjectNode config, String key) {
    ObjectNode block = config.putObject(key);
    block.put("bits", 4);
    block.put("group_size", 128);
    block.put("mode", "affine");
    return block;
  }

  @Test
  void hfFloatIsSelectedWithoutMetadataOrQuantization(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{}");
    assertEquals(
        new Qwen3VlProvenance.Resolved(Qwen3VlProvenance.Kind.HF_FLOAT, null, null),
        Qwen3VlProvenance.resolve(dir, config()));
    // A shard without a metadata block contributes nothing; agreement across shards still resolves.
    writeShard(dir.resolve("part-00001.safetensors"), "{}");
    assertEquals(
        new Qwen3VlProvenance.Resolved(Qwen3VlProvenance.Kind.HF_FLOAT, null, null),
        Qwen3VlProvenance.resolve(dir, config()));
  }

  @Test
  void pytorchMetadataSelectsFloatAndRejectsQuantization(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": {\"format\": \"pt\"}}");
    assertEquals(
        new Qwen3VlProvenance.Resolved(Qwen3VlProvenance.Kind.HF_FLOAT, null, null),
        Qwen3VlProvenance.resolve(dir, config()));
    ObjectNode config = config();
    quant(config, "quantization");
    assertRejectedContains(dir, config, "format=pt selects HF float");
  }

  @Test
  void mlxAffineIsRecognized(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": {\"format\": \"mlx\"}}");
    ObjectNode config = config();
    quant(config, "quantization");
    assertMlxResolved(dir, config);
  }

  @Test
  void mlxAffineUnderTheConfigAliasIsRecognized(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": {\"format\": \"mlx\"}}");
    ObjectNode config = config();
    quant(config, "quantization_config");
    assertMlxResolved(dir, config);
  }

  @Test
  void mlxAffineDeclaredUnderBothEqualKeysIsRecognized(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": {\"format\": \"mlx\"}}");
    ObjectNode config = config();
    quant(config, "quantization");
    config.set("quantization_config", config.get("quantization").deepCopy());
    assertMlxResolved(dir, config);
  }

  private static void assertMlxResolved(Path dir, ObjectNode config) throws IOException {
    assertEquals(
        new Qwen3VlProvenance.Resolved(Qwen3VlProvenance.Kind.MLX_AFFINE, 128, 4),
        Qwen3VlProvenance.resolve(dir, config));
  }

  @Test
  void disagreeingQuantizationKeysAreRejected(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": {\"format\": \"mlx\"}}");
    ObjectNode config = config();
    quant(config, "quantization");
    ObjectNode alias = quant(config, "quantization_config");
    alias.put("group_size", 256);
    assertRejectedContains(dir, config, "quantization and quantization_config disagree");
  }

  @Test
  void malformedAffineBlocksAreNamed(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": {\"format\": \"mlx\"}}");
    ObjectNode config = config();
    ObjectNode block = quant(config, "quantization");
    block.put("bits", 8);
    assertRejectedContains(dir, config, "quantization.bits must be 4");

    block = quant(config, "quantization");
    for (int invalid : new int[] {0, 16, 48, 256}) {
      block.put("group_size", invalid);
      assertRejectedContains(dir, config, "quantization.group_size must be 32, 64 or 128");
    }

    block = quant(config, "quantization");
    block.put("mode", "symmetric");
    assertRejectedContains(dir, config, "quantization.mode must be affine");

    block = quant(config, "quantization");
    block.put("scheme", "nf4");
    assertRejectedContains(dir, config, "must hold exactly bits, group_size and mode");

    // A missing declaration under format=mlx is its own named mismatch.
    config.remove("quantization");
    assertRejectedContains(dir, config, "requires a config quantization block");
  }

  @Test
  void quantizationWithoutFormatMetadataIsRejected(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{}");
    ObjectNode config = config();
    quant(config, "quantization");
    assertRejectedContains(
        dir, config, "declares a quantization block but the checkpoint shards carry no format");
  }

  @Test
  void unknownFormatIsNamed(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": {\"format\": \"gguf\"}}");
    assertRejectedContains(dir, config(), "unsupported Qwen3-VL checkpoint format 'gguf'");
  }

  @Test
  void invalidMetadataBlocksAreRejected(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": {\"format\": 5}}");
    assertRejectedContains(
        dir, config(), "'format' in " + dir.resolve("model.safetensors") + " must be a string");

    writeShard(dir.resolve("model.safetensors"), "{\"__metadata__\": [1]}");
    assertRejectedContains(dir, config(), "invalid __metadata__ block");
  }

  @Test
  void crossShardMetadataCollisionsAreNamed(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("a.safetensors"), "{\"__metadata__\": {\"format\": \"mlx\"}}");
    writeShard(dir.resolve("b.safetensors"), "{\"__metadata__\": {\"format\": \"hf\"}}");
    IllegalArgumentException failure =
        assertThrows(
            IllegalArgumentException.class, () -> Qwen3VlProvenance.resolve(dir, config()));
    assertTrue(
        failure.getMessage().contains("differs between"), () -> "got " + failure.getMessage());
    assertTrue(failure.getMessage().contains(dir.resolve("a.safetensors").toString()));
    assertTrue(failure.getMessage().contains(dir.resolve("b.safetensors").toString()));

    // Identical values across shards merge and resolve (to the MLX affine layout).
    writeShard(dir.resolve("b.safetensors"), "{\"__metadata__\": {\"format\": \"mlx\"}}");
    ObjectNode config = config();
    quant(config, "quantization");
    assertMlxResolved(dir, config);
  }

  @Test
  void metadataMergesAcrossShardsThatPartiallyDeclareIt(@TempDir Path dir) throws Exception {
    writeShard(dir.resolve("a.safetensors"), "{\"__metadata__\": {\"format\": \"mlx\"}}");
    writeShard(dir.resolve("b.safetensors"), "{\"__metadata__\": {\"other\": \"x\"}}");
    ObjectNode config = config();
    quant(config, "quantization");
    assertMlxResolved(dir, config);
  }

  private static void assertRejectedContains(Path dir, ObjectNode config, String expected) {
    IllegalArgumentException failure =
        assertThrows(IllegalArgumentException.class, () -> Qwen3VlProvenance.resolve(dir, config));
    assertTrue(
        failure.getMessage().contains(expected),
        () -> "expected message containing " + expected + ", got " + failure.getMessage());
  }
}
