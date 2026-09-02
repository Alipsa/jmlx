package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
    assertEquals(true, parsed.tieWordEmbeddings());
  }

  @Test
  void rejectsHeadConfigurationsThatCannotUseGroupedQueryAttention() {
    assertThrows(
        IllegalArgumentException.class,
        () -> new DecoderConfig("llama", 8, 10, 16, 1, 3, 2, 1e-6f, 10_000f, false));
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
}
