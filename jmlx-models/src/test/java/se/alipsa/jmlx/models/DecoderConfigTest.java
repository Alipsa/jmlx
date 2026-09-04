package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DecoderConfigTest {
  @Test
  void readsQwenGroupedQueryConfiguration(@TempDir Path dir) throws Exception {
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        """
        {"model_type":"qwen2","vocab_size":151936,"hidden_size":896,
         "intermediate_size":4864,"num_hidden_layers":24,"num_attention_heads":14,
         "num_key_value_heads":2,"rms_norm_eps":0.000001,"rope_theta":1000000,
         "tie_word_embeddings":true}
        """);

    DecoderConfig parsed = DecoderConfig.fromFile(config);

    assertEquals("qwen2", parsed.modelType());
    assertEquals(2, parsed.numKeyValueHeads());
    assertEquals(1_000_000f, parsed.ropeTheta());
    assertTrue(parsed.tieWordEmbeddings());
  }

  @Test
  void rejectsHeadConfigurationsThatCannotUseGroupedQueryAttention() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new DecoderConfig("llama", 8, 10, 16, 1, 3, 2, 1e-6f, 10_000f, false, false, false));
  }

  @Test
  void validatesDirectModelTypeConstruction() {
    assertThrows(
        NullPointerException.class,
        () -> new DecoderConfig(null, 8, 4, 8, 1, 2, 1, 1e-6f, 10_000f, false, false, false));
    assertThrows(
        IllegalArgumentException.class,
        () -> new DecoderConfig(" ", 8, 4, 8, 1, 2, 1, 1e-6f, 10_000f, false, false, false));
  }

  @Test
  void acceptsNullHeadDimButRejectsUnsupportedRopeScaling(@TempDir Path dir) throws Exception {
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        """
        {"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"head_dim":null}
        """);
    assertEquals(4, DecoderConfig.fromFile(config).hiddenSize());
    Files.writeString(
        config,
        """
        {"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"rope_scaling":{"type":"linear"}}
        """);
    assertThrows(IllegalArgumentException.class, () -> DecoderConfig.fromFile(config));
  }

  @Test
  void rejectsQuantizedWeightConfigurations(@TempDir Path dir) throws Exception {
    Path config = dir.resolve("config.json");
    for (String field : java.util.List.of("quantization", "quantization_config")) {
      Files.writeString(
          config,
          """
          {"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":8,
           "num_hidden_layers":1,"num_attention_heads":2,"%s":{"group_size":64,"bits":4}}
          """
              .formatted(field));

      IllegalArgumentException error =
          assertThrows(IllegalArgumentException.class, () -> DecoderConfig.fromFile(config));
      assertTrue(error.getMessage().contains("quantized weights"), error.getMessage());
    }
  }

  @Test
  void rejectsSlidingWindowAttention(@TempDir Path dir) throws Exception {
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        """
        {"model_type":"qwen2","vocab_size":8,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"use_sliding_window":true}
        """);
    assertThrows(IllegalArgumentException.class, () -> DecoderConfig.fromFile(config));
  }

  @Test
  void rejectsUnsupportedHiddenActivation(@TempDir Path dir) throws Exception {
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        """
        {"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"hidden_act":"gelu"}
        """);
    assertThrows(IllegalArgumentException.class, () -> DecoderConfig.fromFile(config));
  }

  @Test
  void acceptsSwishAsAnAliasForSilu(@TempDir Path dir) throws Exception {
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        """
        {"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"hidden_act":"swish"}
        """);
    assertEquals(4, DecoderConfig.fromFile(config).hiddenSize());
  }

  @Test
  void readsAttentionAndMlpBiasFlags(@TempDir Path dir) throws Exception {
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        """
        {"model_type":"llama","vocab_size":8,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"attention_bias":true,"mlp_bias":true}
        """);
    DecoderConfig parsed = DecoderConfig.fromFile(config);
    assertTrue(parsed.attentionBias());
    assertTrue(parsed.mlpBias());
  }

  @Test
  void failsWithIoExceptionWhenConfigFileIsMissing(@TempDir Path dir) {
    assertThrows(IOException.class, () -> DecoderConfig.fromFile(dir.resolve("missing.json")));
  }
}
