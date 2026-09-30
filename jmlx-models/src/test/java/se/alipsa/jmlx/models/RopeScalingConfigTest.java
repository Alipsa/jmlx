package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.nn.RopeSpec;
import tools.jackson.databind.ObjectMapper;

class RopeScalingConfigTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final String BASE =
      """
      {"model_type":"llama","vocab_size":16,"hidden_size":16,"intermediate_size":32,
       "num_hidden_layers":1,"num_attention_heads":2,"max_position_embeddings":64,
       "rope_scaling":%s}
      """;

  private static ArchitectureDescriptor parse(String scaling) {
    return ArchitectureMappings.parse(MAPPER.readTree(BASE.formatted(scaling)));
  }

  @Test
  void acceptsSupportedVariants() {
    assertInstanceOf(
        RopeSpec.Linear.class, parse("{\"rope_type\":\"linear\",\"factor\":4}").rope());
    assertInstanceOf(
        RopeSpec.DynamicNtk.class, parse("{\"rope_type\":\"dynamic\",\"factor\":4}").rope());
    assertInstanceOf(
        RopeSpec.Llama3.class,
        parse(
                """
                {"rope_type":"llama3","factor":8,"low_freq_factor":1,"high_freq_factor":4,
                 "original_max_position_embeddings":64}
                """)
            .rope());
    assertInstanceOf(
        RopeSpec.Yarn.class,
        parse(
                """
                {"rope_type":"yarn","factor":4,"original_max_position_embeddings":64}
                """)
            .rope());
  }

  @Test
  void partialRotaryFactorSetsEvenPrefix() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            MAPPER.readTree(
                """
                {"model_type":"llama","vocab_size":16,"hidden_size":16,"intermediate_size":32,
                 "num_hidden_layers":1,"num_attention_heads":2,"partial_rotary_factor":0.5}
                """));
    assertEquals(4, d.rotaryDims());
  }

  @Test
  void partialRotaryFactorTruncatesInDoublePrecisionLikeHuggingFace() {
    // Narrowing the factor to float turns 100 * 0.29 into 29.0 and 100 * 0.57 into 57.0, while
    // Hugging Face's double arithmetic truncates to 28 and 56.
    assertEquals(28, rotaryDimsFor(100, "0.29"));
    assertEquals(56, rotaryDimsFor(100, "0.57"));
  }

  private static int rotaryDimsFor(int headDim, String factor) {
    return ArchitectureMappings.parse(
            MAPPER.readTree(
                """
                {"model_type":"llama","vocab_size":16,"hidden_size":%d,"intermediate_size":32,
                 "num_hidden_layers":1,"num_attention_heads":1,"partial_rotary_factor":%s}
                """
                    .formatted(headDim, factor)))
        .rotaryDims();
  }

  @Test
  void yarnBetaZeroOrNullSelectsHuggingFaceDefaults() {
    for (String betas :
        new String[] {"\"beta_fast\":0,\"beta_slow\":0", "\"beta_fast\":null,\"beta_slow\":null"}) {
      RopeSpec.Yarn yarn =
          (RopeSpec.Yarn)
              parse(
                      "{\"rope_type\":\"yarn\",\"factor\":4,"
                          + "\"original_max_position_embeddings\":64,"
                          + betas
                          + "}")
                  .rope();
      assertEquals(32f, yarn.betaFast());
      assertEquals(1f, yarn.betaSlow());
    }
  }

  @Test
  void yarnNullFactorIsDerivedFromContextLengths() {
    RopeSpec.Yarn yarn =
        (RopeSpec.Yarn)
            parse(
                    """
                    {"rope_type":"yarn","factor":null,"original_max_position_embeddings":16}
                    """)
                .rope();
    assertEquals(4f, yarn.factor());
  }

  @Test
  void unsupportedTypeAndFieldNameTheirKeys() {
    for (String scaling :
        new String[] {
          "{\"rope_type\":\"longrope\"}", "{\"rope_type\":\"linear\",\"factor\":2,\"mystery\":3}"
        }) {
      var error = assertThrows(IllegalArgumentException.class, () -> parse(scaling));
      assertTrue(error.getMessage().contains("rope_scaling"), error.getMessage());
    }
  }

  @Test
  void missingFactorNamesField() {
    var error =
        assertThrows(IllegalArgumentException.class, () -> parse("{\"rope_type\":\"linear\"}"));
    assertTrue(error.getMessage().contains("factor"), error.getMessage());
  }

  @Test
  void dynamicNtkUsesMaxPositionEmbeddingsNotOriginalContext() {
    ArchitectureDescriptor d =
        parse(
            """
            {"rope_type":"dynamic","factor":4,"original_max_position_embeddings":16}
            """);
    assertEquals(64, ((RopeSpec.DynamicNtk) d.rope()).maxPositions());
  }
}
