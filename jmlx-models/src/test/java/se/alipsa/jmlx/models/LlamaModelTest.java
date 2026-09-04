package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class LlamaModelTest {
  @Test
  void loadsCheckpointAndGenerates(@TempDir Path dir) throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true}
        """);
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = tinyCheckpoint(saveScope);
      tensors.remove("lm_head.weight");
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
    }
    try (MLXScope modelScope = new MLXScope()) {
      LlamaModel model = LlamaModel.load(modelScope, dir);
      assertEquals(List.of(1, 0, 0), model.generate(new int[] {1}, 2, Set.of()));
      assertThrows(IllegalArgumentException.class, () -> model.generate(null, 2, Set.of()));
      List<GenerationEvent> events = new ArrayList<>();
      GenerationResult result =
          TextGenerationModels.load(modelScope, dir)
              .generate(
                  new GenerationRequest(
                      new int[] {1},
                      GenerationConfig.greedyDefaults(2, Set.of()),
                      CancellationToken.NONE),
                  events::add);
      assertEquals(List.of(1), result.promptTokenIds());
      assertEquals(List.of(0, 0), result.generatedTokenIds());
      assertEquals(FinishReason.MAX_TOKENS, result.finishReason());
      assertEquals(
          List.of(0, 0), events.subList(0, 2).stream().map(GenerationEvent::tokenId).toList());
      assertEquals(null, events.getLast().tokenId());
      assertEquals(FinishReason.MAX_TOKENS, events.getLast().finishReason());

      GenerationResult eos =
          model.generate(
              new GenerationRequest(
                  new int[] {1},
                  GenerationConfig.greedyDefaults(2, Set.of(0)),
                  CancellationToken.NONE),
              ignored -> {});
      assertEquals(FinishReason.EOS, eos.finishReason());
      assertEquals(List.of(0), eos.generatedTokenIds());

      GenerationConfig stopPolicy =
          new GenerationConfig(
              2, java.util.OptionalLong.empty(), 0, 0, 1, 0, 1, 0, 0, Set.of(), Set.of(0), false);
      assertEquals(
          FinishReason.STOP_TOKEN,
          model
              .generate(
                  new GenerationRequest(new int[] {1}, stopPolicy, CancellationToken.NONE),
                  ignored -> {})
              .finishReason());

      AtomicBoolean cancelled = new AtomicBoolean();
      GenerationResult cancelledResult =
          model.generate(
              new GenerationRequest(
                  new int[] {1}, GenerationConfig.greedyDefaults(2, Set.of()), cancelled::get),
              event -> cancelled.set(event.tokenId() != null));
      assertEquals(FinishReason.CANCELLED, cancelledResult.finishReason());
      assertEquals(List.of(0), cancelledResult.generatedTokenIds());

      assertThrows(
          UnsupportedOperationException.class,
          () ->
              model.generate(
                  new GenerationRequest(
                      new int[] {1},
                      new GenerationConfig(
                          1,
                          java.util.OptionalLong.empty(),
                          1,
                          0,
                          1,
                          0,
                          1,
                          0,
                          0,
                          Set.of(),
                          Set.of(),
                          false),
                      CancellationToken.NONE),
                  ignored -> {}));
      List<GenerationEvent> failedListenerEvents = new ArrayList<>();
      assertThrows(
          IllegalStateException.class,
          () ->
              model.generate(
                  new GenerationRequest(
                      new int[] {1},
                      GenerationConfig.greedyDefaults(2, Set.of()),
                      CancellationToken.NONE),
                  event -> {
                    failedListenerEvents.add(event);
                    throw new IllegalStateException("listener failed");
                  }));
      assertEquals(1, failedListenerEvents.size());
      assertEquals(0, failedListenerEvents.getFirst().tokenId());
      assertEquals(List.of(1, 0), model.generate(new int[] {1}, 1, Set.of()));
    }
  }

  @Test
  void rejectsQwenBeforeAttemptingCheckpointLoad(@TempDir Path dir) throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1}
        """);

    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.load(null, dir));
    assertTrue(e.getMessage().contains("expected model_type llama"), e.getMessage());
  }

  @Test
  void throwsWhenConfigRequiresAttentionBiasButCheckpointHasNone(@TempDir Path dir)
      throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true,
         "attention_bias":true}
        """);
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = tinyCheckpoint(saveScope);
      tensors.remove("lm_head.weight");
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
    }
    try (MLXScope modelScope = new MLXScope()) {
      IllegalArgumentException e =
          assertThrows(IllegalArgumentException.class, () -> LlamaModel.load(modelScope, dir));
      assertTrue(e.getMessage().contains("self_attn.q_proj.bias"), e.getMessage());
    }
  }

  private static Map<String, MLXArray> tinyCheckpoint(MLXScope scope) {
    Map<String, MLXArray> tensors = new LinkedHashMap<>();
    tensors.put("model.embed_tokens.weight", zeros(scope, 4, 4));
    tensors.put("model.norm.weight", zeros(scope, 4));
    tensors.put("lm_head.weight", zeros(scope, 4, 4));
    String p = "model.layers.0.";
    tensors.put(p + "input_layernorm.weight", zeros(scope, 4));
    tensors.put(p + "post_attention_layernorm.weight", zeros(scope, 4));
    // No bias tensors: attention_bias:true in config.json requires all four, so loading must
    // fail before any of these weights matter.
    tensors.put(p + "self_attn.q_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "self_attn.k_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "self_attn.v_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "self_attn.o_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "mlp.gate_proj.weight", zeros(scope, 8, 4));
    tensors.put(p + "mlp.up_proj.weight", zeros(scope, 8, 4));
    tensors.put(p + "mlp.down_proj.weight", zeros(scope, 4, 8));
    return tensors;
  }

  private static MLXArray zeros(MLXScope scope, int... shape) {
    return MLX.zeros(scope, shape, DType.FLOAT32);
  }
}
