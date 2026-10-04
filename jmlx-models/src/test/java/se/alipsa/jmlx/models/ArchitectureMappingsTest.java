package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.models.ArchitectureDescriptor.NormKind;
import se.alipsa.jmlx.nn.RopeSpec;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

class ArchitectureMappingsTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  private static JsonNode json(String text) {
    return MAPPER.readTree(text);
  }

  @Test
  void llamaMapsToTodaysBehavior() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,
                 "tie_word_embeddings":true}\
                """));
    assertEquals(2, d.headDim());
    assertEquals(NormKind.RMS, d.norm().kind());
    assertFalse(d.attention().qkvBias());
    assertTrue(d.head().tied());
    assertNull(d.attention().slidingWindow());
    assertNull(d.moe());
  }

  @Test
  void qwen2HardcodesQkvBiasAndNoOutBias() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                 "attention_bias":false}\
                """));
    assertTrue(d.attention().qkvBias());
    assertFalse(d.attention().outBias());
  }

  @Test
  void qwen3ParsesExplicitHeadDimAndQkNorm() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"qwen3","vocab_size":151936,"hidden_size":1024,
                 "intermediate_size":3072,"num_hidden_layers":28,"num_attention_heads":16,
                 "num_key_value_heads":8,"max_position_embeddings":40960,"rms_norm_eps":1e-06,
                 "rope_theta":1000000,"head_dim":128,"tie_word_embeddings":true,
                 "attention_bias":false,"hidden_act":"silu"}\
                """));
    // head_dim differs from hidden_size / num_attention_heads on purpose: the Qwen3 product check
    // must not reject the family.
    assertEquals(128, d.headDim());
    assertEquals(128, d.rotaryDims());
    assertTrue(d.attention().qkNorm());
    assertFalse(d.attention().qkvBias());
    assertFalse(d.attention().outBias());
    assertFalse(d.attention().fusedQkv());
    assertNull(d.attention().slidingWindow());
    assertTrue(d.head().tied());
    assertEquals(1_000_000f, ((RopeSpec.Base) d.rope()).theta());
  }

  @Test
  void qwen3LiveConfigKeysAreAllConsumedWithoutWarnings() {
    List<String> unknown = new ArrayList<>();
    ArchitectureMappings.parse(
        json(
            """
            {"architectures":["Qwen3ForCausalLM"],"attention_bias":false,
             "attention_dropout":0.0,"bos_token_id":151643,"eos_token_id":151645,
             "head_dim":128,"hidden_act":"silu","hidden_size":1024,
             "initializer_range":0.02,"intermediate_size":3072,
             "max_position_embeddings":40960,"max_window_layers":28,"model_type":"qwen3",
             "num_attention_heads":16,"num_hidden_layers":28,"num_key_value_heads":8,
             "rms_norm_eps":1e-06,"rope_scaling":null,"rope_theta":1000000,
             "sliding_window":null,"tie_word_embeddings":true,"torch_dtype":"bfloat16",
             "transformers_version":"4.51.0","use_cache":true,"use_sliding_window":false,
             "vocab_size":151936}\
            """),
        unknown::add);
    assertEquals(List.of(), unknown);
  }

  @Test
  void qwen3HonorsAttentionBiasWhenSet() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"qwen3","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                 "attention_bias":true}\
                """));
    assertTrue(d.attention().qkvBias());
    assertTrue(d.attention().outBias());
  }

  @Test
  void qwen3WithoutExplicitHeadDimFallsBackToHiddenPerHead() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"qwen3","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1}\
                """));
    assertEquals(4, d.headDim());
    assertEquals(4, d.rotaryDims());
    assertTrue(d.attention().qkNorm());
  }

  @Test
  void qwen3TensorPlanRequiresAndForbidsQkNormWeights() {
    TensorPlan qwen3 = ArchitectureMappings.tensorPlan(TestDescriptors.qwen3(2));
    assertTrue(qwen3.required().contains("model.layers.0.self_attn.q_norm.weight"));
    assertTrue(qwen3.required().contains("model.layers.1.self_attn.k_norm.weight"));
    assertFalse(qwen3.forbidden().contains("model.layers.0.self_attn.q_norm.weight"));
    TensorPlan qwen2 = ArchitectureMappings.tensorPlan(TestDescriptors.qwen2(2));
    assertTrue(qwen2.forbidden().contains("model.layers.0.self_attn.q_norm.weight"));
    assertTrue(qwen2.forbidden().contains("model.layers.0.self_attn.k_norm.weight"));
    // Norm weights never enter the affine companion allow-list, even when quantization is set.
    TensorPlan quantized =
        ArchitectureMappings.tensorPlan(
            TestDescriptors.llamaWithQuantization(32, 4));
    assertFalse(quantized.optional().contains("model.layers.0.self_attn.q_norm.scales"));
  }

  @Test
  void qwen3MoEIsRejectedByNamedDeferral() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ArchitectureMappings.parse(
                    json(
                        """
                        {"model_type":"qwen3_moe","vocab_size":16,"hidden_size":8,
                         "intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2}\
                        """)));
    assertTrue(error.getMessage().contains("qwen3_moe"));
    assertTrue(error.getMessage().contains("deferred"));
  }

  @Test
  void qwen3RejectsEnabledSlidingWindow() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ArchitectureMappings.parse(
                    json(
                        """
                        {"model_type":"qwen3","vocab_size":16,"hidden_size":8,
                         "intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                         "use_sliding_window":true}\
                        """)));
    assertTrue(error.getMessage().contains("use_sliding_window"));
  }

  @Test
  void freshlySavedConfigWithHarmlessKeysLoads() {
    List<String> unknown = new ArrayList<>();
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
                 "num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
                 "layer_types":["full_attention","full_attention"],"torch_dtype":"bfloat16",
                 "transformers_version":"5.0.0","use_cache":true,"bos_token_id":1,"eos_token_id":2,
                 "rope_parameters":{"rope_type":"default","rope_theta":1000000.0},
                 "brand_new_key":7}\
                """),
            unknown::add);
    assertEquals(1_000_000f, ((RopeSpec.Base) d.rope()).theta());
    assertEquals(List.of("brand_new_key"), unknown);
  }

  @Test
  void invalidLayerTypesAreRejectedByName() {
    for (String types :
        List.of("[\"full_attention\"]", "[\"full_attention\",\"sliding_attention\"]")) {
      var error =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  ArchitectureMappings.parse(
                      json(
                          """
                          {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
                           "num_hidden_layers":2,"num_attention_heads":2,"layer_types":%s}
                          """
                              .formatted(types))));
      assertTrue(error.getMessage().contains("layer_types"));
    }
  }

  @Test
  void disabledQwen2SlidingWindowSettingsAreAccepted() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                 "use_sliding_window":false,"sliding_window":32768,"max_window_layers":28}\
                """));
    assertNull(d.attention().slidingWindow());
  }

  @Test
  void conflictingRopeThetaLocationsAreRejected() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ArchitectureMappings.parse(
                    json(
                        """
                        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
                         "num_hidden_layers":1,"num_attention_heads":2,"rope_theta":10000,
                         "rope_parameters":{"rope_type":"default","rope_theta":500000}}\
                        """)));
    assertTrue(error.getMessage().contains("rope_theta"));
  }

  @Test
  void unknownModelTypeNamesTheKey() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () -> ArchitectureMappings.parse(json("{\"model_type\":\"falcon\"}")));
    assertTrue(error.getMessage().contains("model_type"));
    assertTrue(error.getMessage().contains("falcon"));
  }

  @Test
  void mistralAndPhi3ExposeTheirAttentionLayouts() {
    ArchitectureDescriptor mistral =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"mistral","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                 "sliding_window":4}
                """));
    assertEquals(4, mistral.attention().slidingWindow());
    ArchitectureDescriptor phi =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"phi3","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1}
                """));
    assertTrue(phi.attention().fusedQkv());
    assertEquals(ArchitectureDescriptor.MlpLayout.FUSED_GATE_UP, phi.mlp().layout());
  }

  @Test
  void biasFlagsHuggingFaceHardcodesOffAreIgnored() {
    for (String type : List.of("qwen2", "mistral", "phi3", "gemma", "mixtral")) {
      String extra =
          switch (type) {
            case "gemma" -> ",\"head_dim\":4";
            case "mixtral" -> ",\"num_local_experts\":2,\"num_experts_per_tok\":1";
            default -> "";
          };
      ArchitectureDescriptor d =
          ArchitectureMappings.parse(
              json(
                  """
                  {"model_type":"%s","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                   "num_hidden_layers":1,"num_attention_heads":2,"mlp_bias":true%s}
                  """
                      .formatted(type, extra)));
      assertFalse(d.mlp().bias(), type);
      assertFalse(d.dimensions().mlpBias(), type);
    }
  }

  @Test
  void phi3IgnoresAttentionBiasLikeHuggingFace() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"phi3","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"attention_bias":true}
                """));
    assertFalse(d.attention().qkvBias());
    assertFalse(d.attention().outBias());
  }

  @Test
  void llamaHonorsMlpBias() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"llama","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"mlp_bias":true}
                """));
    assertTrue(d.mlp().bias());
  }

  private static String familyConfig(String type, String extra) {
    String specific =
        switch (type) {
          case "gemma" -> ",\"head_dim\":4";
          case "mixtral" -> ",\"num_local_experts\":2,\"num_experts_per_tok\":1";
          default -> "";
        };
    return "{\"model_type\":\""
        + type
        + "\",\"vocab_size\":16,\"hidden_size\":8,\"intermediate_size\":16,"
        + "\"num_hidden_layers\":1,\"num_attention_heads\":2"
        + specific
        + extra
        + "}";
  }

  /** Pins one capability for every family, so a new or edited FAMILIES entry must update it. */
  private static void assertPerFamily(
      String extra, Function<ArchitectureDescriptor, String> describe, Map<String, String> want) {
    assertEquals(ArchitectureMappings.supportedModelTypes(), want.keySet());
    for (Map.Entry<String, String> entry : want.entrySet()) {
      String type = entry.getKey();
      String got;
      try {
        got = describe.apply(ArchitectureMappings.parse(json(familyConfig(type, extra)), k -> {}));
      } catch (IllegalArgumentException e) {
        got = "rejected";
      }
      assertEquals(entry.getValue(), got, type + " with " + extra);
    }
  }

  @Test
  void layerTypesScheduleIsAcceptedPerFamily() {
    assertPerFamily(
        ",\"layer_types\":[\"full_attention\"]",
        d -> "accepted",
        Map.of(
            "llama", "accepted",
            "qwen2", "accepted",
            "qwen3", "accepted",
            "mistral", "rejected",
            "phi3", "rejected",
            "gemma", "accepted",
            "mixtral", "accepted"));
  }

  @Test
  void maxWindowLayersIsAcceptedPerFamily() {
    assertPerFamily(
        ",\"max_window_layers\":1",
        d -> "accepted",
        Map.of(
            "llama", "rejected",
            "qwen2", "accepted",
            "qwen3", "accepted",
            "mistral", "rejected",
            "phi3", "rejected",
            "gemma", "rejected",
            "mixtral", "rejected"));
  }

  @Test
  void slidingWindowIsHandledPerFamily() {
    assertPerFamily(
        ",\"sliding_window\":4",
        d -> String.valueOf(d.attention().slidingWindow()),
        Map.of(
            "llama", "rejected",
            "qwen2", "null",
            "qwen3", "null",
            "mistral", "4",
            "phi3", "4",
            "gemma", "rejected",
            "mixtral", "4"));
  }

  @Test
  void attentionBiasFlagIsHandledPerFamily() {
    assertPerFamily(
        ",\"attention_bias\":true",
        d -> "qkv=" + d.attention().qkvBias() + " out=" + d.attention().outBias(),
        Map.of(
            "llama", "qkv=true out=true",
            "qwen2", "qkv=true out=false",
            "qwen3", "qkv=true out=true",
            "mistral", "qkv=false out=false",
            "phi3", "qkv=false out=false",
            "gemma", "rejected",
            "mixtral", "qkv=false out=false"));
  }

  @Test
  void mlpBiasFlagIsHandledPerFamily() {
    assertPerFamily(
        ",\"mlp_bias\":true",
        d -> String.valueOf(d.mlp().bias()),
        Map.of(
            "llama", "true",
            "qwen2", "false",
            "qwen3", "false",
            "mistral", "false",
            "phi3", "false",
            "gemma", "false",
            "mixtral", "false"));
  }

  @Test
  void structuralCapabilitiesArePinnedPerFamily() {
    assertPerFamily(
        "",
        d ->
            "qkv="
                + d.attention().qkvBias()
                + " fused="
                + d.attention().fusedQkv()
                + " moe="
                + (d.moe() != null)
                + " qk="
                + d.attention().qkNorm()
                + " offset="
                + d.norm().weightOffset()
                + " scaled="
                + d.embedding().scaleBySqrtHidden()
                + " tied="
                + d.head().tied(),
        Map.of(
            "llama",
            "qkv=false fused=false moe=false qk=false offset=false scaled=false tied=false",
            "qwen2", "qkv=true fused=false moe=false qk=false offset=false scaled=false tied=false",
            "qwen3", "qkv=false fused=false moe=false qk=true offset=false scaled=false tied=false",
            "mistral",
            "qkv=false fused=false moe=false qk=false offset=false scaled=false tied=false",
            "phi3", "qkv=false fused=true moe=false qk=false offset=false scaled=false tied=false",
            "gemma", "qkv=false fused=false moe=false qk=false offset=true scaled=true tied=true",
            "mixtral",
            "qkv=false fused=false moe=true qk=false offset=false scaled=false tied=false"));
  }

  @Test
  void mistralRejectsPerLayerSchedules() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ArchitectureMappings.parse(
                    json(
                        """
                        {"model_type":"mistral","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                         "num_hidden_layers":1,"num_attention_heads":2,"layer_types":["full_attention"]}
                        """)));
    assertTrue(error.getMessage().contains("layer_types"));
  }

  @Test
  void mistralNullWindowUsesFullAttention() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"mistral","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"sliding_window":null}
                """));
    assertNull(d.attention().slidingWindow());
  }

  @Test
  void phi3LongropeNamesUnsupportedVariant() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ArchitectureMappings.parse(
                    json(
                        """
                        {"model_type":"phi3","vocab_size":16,"hidden_size":8,
                         "intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                         "rope_scaling":{"rope_type":"longrope","factor":2}}
                        """)));
    assertTrue(error.getMessage().contains("rope_scaling.rope_type"));
    assertTrue(error.getMessage().contains("longrope"));
  }

  @Test
  void gemmaFallsBackToHiddenActWhenHiddenActivationIsNull() {
    ArchitectureDescriptor gemma =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"gemma","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                 "head_dim":6,"hidden_act":"gelu","hidden_activation":null}
                """));
    assertEquals(6, gemma.headDim());
    assertEquals(se.alipsa.jmlx.nn.Activation.GELU, gemma.mlp().activation());
    assertTrue(gemma.norm().weightOffset());
    assertTrue(gemma.embedding().scaleBySqrtHidden());
    assertTrue(gemma.head().tied());
  }

  @Test
  void gemmaDefaultsToTanhGeluWhenNeitherActivationKeyIsPresent() {
    ArchitectureDescriptor gemma =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"gemma","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"head_dim":6}
                """));
    assertEquals(se.alipsa.jmlx.nn.Activation.GELU_TANH, gemma.mlp().activation());
  }

  @Test
  void gemmaActivationErrorNamesGemmaActivationsNotSilu() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ArchitectureMappings.parse(
                    json(
                        """
                        {"model_type":"gemma","vocab_size":16,"hidden_size":8,
                         "intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                         "head_dim":6,"hidden_act":"relu"}
                        """)));
    assertTrue(error.getMessage().contains("gelu_pytorch_tanh"), error.getMessage());
    assertFalse(error.getMessage().contains("only implements silu"), error.getMessage());
  }

  @Test
  void mistralAndMixtralIgnoreAttentionBiasLikeHuggingFace() {
    for (String type : List.of("mistral", "mixtral")) {
      ArchitectureDescriptor d =
          ArchitectureMappings.parse(
              json(
                  """
                  {"model_type":"%s","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                   "num_hidden_layers":1,"num_attention_heads":2,"attention_bias":true%s}
                  """
                      .formatted(
                          type,
                          "mixtral".equals(type)
                              ? ",\"num_local_experts\":2,\"num_experts_per_tok\":1"
                              : "")));
      assertFalse(d.attention().qkvBias(), type);
      assertFalse(d.attention().outBias(), type);
      assertFalse(d.dimensions().attentionBias(), type);
    }
  }

  @Test
  void gemmaValidatesHiddenActEvenWhenHiddenActivationIsPresent() {
    for (String hiddenAct : List.of("relu", "silu")) {
      var error =
          assertThrows(
              IllegalArgumentException.class,
              () ->
                  ArchitectureMappings.parse(
                      json(
                          """
                          {"model_type":"gemma","vocab_size":16,"hidden_size":8,
                           "intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,
                           "head_dim":6,"hidden_act":"%s","hidden_activation":"gelu"}
                          """
                              .formatted(hiddenAct))));
      assertTrue(error.getMessage().contains("hidden_act '" + hiddenAct + "'"), hiddenAct);
    }
  }

  @Test
  void gemmaRequiresExplicitHeadDimension() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ArchitectureMappings.parse(
                    json(
                        """
                        {"model_type":"gemma","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                         "num_hidden_layers":1,"num_attention_heads":2}
                        """)));
    assertTrue(error.getMessage().contains("head_dim"));
  }

  @Test
  void mixtralMapsExpertsAndSlidingWindow() {
    ArchitectureDescriptor d =
        ArchitectureMappings.parse(
            json(
                """
                {"model_type":"mixtral","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                 "num_local_experts":4,"num_experts_per_tok":2,"sliding_window":4,
                 "output_router_logits":false}
                """));
    assertEquals(4, d.moe().experts());
    assertEquals(2, d.moe().topK());
    assertEquals(4, d.attention().slidingWindow());
    TensorPlan plan = ArchitectureMappings.tensorPlan(d);
    assertTrue(plan.required().contains("model.layers.0.block_sparse_moe.experts.3.w3.weight"));
  }

  @Test
  void denseFamilyRejectsExpertConfiguration() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ArchitectureMappings.parse(
                    json(
                        """
                        {"model_type":"llama","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                         "num_hidden_layers":1,"num_attention_heads":2,"num_local_experts":4}
                        """)));
    assertTrue(error.getMessage().contains("num_local_experts"));
  }

  @Test
  void unsupportedNumericFieldsNameTheirKeys() {
    String base =
        """
        {"model_type":"llama","vocab_size":16,"hidden_size":8,"intermediate_size":16,
         "num_hidden_layers":1,"num_attention_heads":2,%s}
        """;
    for (String field : List.of("sliding_window", "max_window_layers", "hidden_activation")) {
      String value = "hidden_activation".equals(field) ? "\"gelu\"" : "4";
      String config = base.formatted("\"" + field + "\":" + value);
      var error =
          assertThrows(
              IllegalArgumentException.class, () -> ArchitectureMappings.parse(json(config)));
      assertTrue(error.getMessage().contains(field), error.getMessage());
    }
  }
}
