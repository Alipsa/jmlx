package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.HashSet;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.tokenizer.HfTokenizer;
import tools.jackson.databind.node.ObjectNode;

/** Pure config, tokenizer-marker and complete tensor-plan validation. */
class SmolVlmDescriptorTest {
  private static ObjectNode config() throws Exception {
    return (ObjectNode) JsonFiles.read(SmolVlmGoldenTest.directory().resolve("config.json"));
  }

  @Test
  void pinnedRealArtifactConfigParsesWithoutIgnoringUnknownNumerics() throws Exception {
    byte[] bytes;
    try (var input = getClass().getResourceAsStream("/smolvlm/config.json")) {
      bytes = java.util.Objects.requireNonNull(input).readAllBytes();
    }
    assertEquals(
        "b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52",
        java.util.HexFormat.of()
            .formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(bytes)));
    SmolVlmDescriptor descriptor =
        SmolVlmDescriptor.parse(new tools.jackson.databind.ObjectMapper().readTree(bytes));
    assertEquals(64, descriptor.tokensPerTile());
    assertEquals(12, descriptor.layers());
    assertEquals(49190, descriptor.imageTokenId());
    assertEquals(49280, descriptor.text().dimensions().vocabSize());
  }

  @Test
  void tinyAndTargetGeometryHaveCorrectPositionAndFeatureCounts() throws Exception {
    ObjectNode config = config();
    SmolVlmDescriptor tiny = SmolVlmDescriptor.parse(config);
    assertEquals(4, tiny.tokensPerTile());
    var headers = CheckpointLoader.preflight(SmolVlmGoldenTest.directory(), tiny.tensorPlan());
    tiny.tensorPlan().validate(headers.tensorNames());
    HashSet<String> missing = new HashSet<>(headers.tensorNames());
    missing.remove("model.connector.modality_projection.proj.weight");
    assertThrows(IllegalArgumentException.class, () -> tiny.tensorPlan().validate(missing));
    HashSet<String> extra = new HashSet<>(headers.tensorNames());
    extra.add("model.vision_model.unrecognized.weight");
    assertThrows(IllegalArgumentException.class, () -> tiny.tensorPlan().validate(extra));
    ((ObjectNode) config.required("vision_config")).put("image_size", 512).put("patch_size", 16);
    config.put("scale_factor", 4);
    assertEquals(64, SmolVlmDescriptor.parse(config).tokensPerTile());
  }

  @Test
  void unknownAndUnsupportedComputationalOptionsFail() throws Exception {
    for (String key : new String[] {"unknown_computation", "quantization", "quantization_config"}) {
      ObjectNode config = config();
      config.put(key, 1);
      assertThrows(IllegalArgumentException.class, () -> SmolVlmDescriptor.parse(config));
    }
    ObjectNode config = config();
    ((ObjectNode) config.required("vision_config")).put("hidden_act", "relu");
    ObjectNode badActivation = config;
    assertThrows(IllegalArgumentException.class, () -> SmolVlmDescriptor.parse(badActivation));
    config = config();
    ((ObjectNode) config.required("vision_config")).put("unknown_numerics", true);
    ObjectNode unknownVision = config;
    assertThrows(IllegalArgumentException.class, () -> SmolVlmDescriptor.parse(unknownVision));
    config = config();
    ((ObjectNode) config.required("text_config")).put("unknown_numerics", true);
    ObjectNode unknownText = config;
    assertThrows(IllegalArgumentException.class, () -> SmolVlmDescriptor.parse(unknownText));
    config = config();
    config.put("vocab_size", 123);
    ObjectNode badVocabulary = config;
    assertThrows(IllegalArgumentException.class, () -> SmolVlmDescriptor.parse(badVocabulary));
  }

  @Test
  void productionResolutionChecksEveryRowColumnMarker() throws Exception {
    HfTokenizer tokenizer = HfTokenizer.fromDirectory(SmolVlmGoldenTest.directory());
    SmolVlmPromptTokens tokens = SmolVlmModel.resolveTokens(tokenizer);
    assertEquals(256, tokens.imageTokenId());
    assertEquals(-1, tokens.videoTokenId());
    for (int row = 1; row <= 6; row++) {
      for (int col = 1; col <= 6; col++) {
        assertEquals(
            tokens.rowColTokenIdBase() + (row - 1) * 6 + col,
            tokenizer.eosTokenId("<row_" + row + "_col_" + col + ">").orElseThrow());
      }
    }
  }
}
