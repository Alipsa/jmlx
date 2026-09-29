package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.tokenizer.HfTokenizer;

@EnabledIfNativeAvailable
class QwenModelTest {
  @Test
  void loadsCheckpointAndGeneratesWithGroupedQueryCache(@TempDir Path dir) throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true}
        """);
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = tinyCheckpoint(saveScope);
      tensors.remove("lm_head.weight");
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
    }
    try (MLXScope modelScope = new MLXScope()) {
      QwenModel model = QwenModel.load(modelScope, dir);
      assertEquals(List.of(1, 0, 0), model.generate(new int[] {1}, 2, Set.of()));
      TextGenerationModel common = TextGenerationModels.load(modelScope, dir);
      assertEquals(
          List.of(1, 0, 0),
          common
              .generate(
                  new GenerationRequest(
                      new int[] {1},
                      GenerationConfig.greedyDefaults(2, Set.of()),
                      CancellationToken.NONE),
                  ignored -> {})
              .tokenIds());

      GenerationConfig sampled =
          new GenerationConfig(
              2, OptionalLong.of(42), 1, 0, 1, 0, 1, 0, 0, Set.of(), Set.of(), true);
      GenerationResult first =
          model.generate(
              new GenerationRequest(new int[] {1}, sampled, CancellationToken.NONE), ignored -> {});
      GenerationResult repeat =
          model.generate(
              new GenerationRequest(new int[] {1}, sampled, CancellationToken.NONE), ignored -> {});
      assertEquals(first.generatedTokenIds(), repeat.generatedTokenIds());
      assertEquals(first.logProbabilities(), repeat.logProbabilities());

      Path tokenizerPath = writeTinyTokenizer(dir);
      HfTokenizer tokenizer = HfTokenizer.fromFile(tokenizerPath);
      GenerationResult text =
          model.generate(
              GenerationRequest.text(
                  tokenizer,
                  "p",
                  PromptSpecialTokens.OMIT,
                  GenerationConfig.greedyDefaults(2, Set.of()),
                  CancellationToken.NONE),
              ignored -> {});
      assertEquals(List.of(0, 0), text.generatedTokenIds());
      assertEquals("aa", text.generatedText());
    }
  }

  @Test
  void qwen2IgnoresAttentionBiasConfigFlagForOutProj(@TempDir Path dir) throws Exception {
    // Regression test for outBiasRequired: a Qwen2 config setting attention_bias (a field Qwen2
    // never actually defines) must not make o_proj.bias required -- Qwen2 hardcodes bias=False
    // for o_proj regardless. tinyCheckpoint() already has no o_proj.bias; that's the point.
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true,
         "attention_bias":true}
        """);
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = tinyCheckpoint(saveScope);
      tensors.remove("lm_head.weight");
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
    }
    try (MLXScope modelScope = new MLXScope()) {
      QwenModel model = QwenModel.load(modelScope, dir);
      assertEquals(List.of(1, 0, 0), model.generate(new int[] {1}, 2, Set.of()));
    }
  }

  @Test
  void throwsWhenRequiredQwenBiasIsMissing(@TempDir Path dir) throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":true}
        """);
    try (MLXScope saveScope = new MLXScope()) {
      Map<String, MLXArray> tensors = tinyCheckpoint(saveScope);
      tensors.remove("lm_head.weight");
      tensors.remove("model.layers.0.self_attn.q_proj.bias");
      MLXIO.saveSafetensors(dir.resolve("model.safetensors").toString(), tensors, Map.of());
    }
    try (MLXScope modelScope = new MLXScope()) {
      IllegalArgumentException e =
          assertThrows(IllegalArgumentException.class, () -> QwenModel.load(modelScope, dir));
      assertTrue(e.getMessage().contains("self_attn.q_proj.bias"), e.getMessage());
    }
  }

  @Test
  void rejectsLlamaBeforeAttemptingCheckpointLoad(@TempDir Path dir) throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"llama","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":2}
        """);

    try (MLXScope scope = new MLXScope()) {
      IllegalArgumentException e =
          assertThrows(IllegalArgumentException.class, () -> QwenModel.load(scope, dir));
      assertTrue(e.getMessage().contains("expected model_type qwen2"), e.getMessage());
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
    tensors.put(p + "self_attn.q_proj.weight", zeros(scope, 4, 4));
    // Qwen2's q/k/v projections always carry a bias; DecoderModel now requires it.
    tensors.put(p + "self_attn.q_proj.bias", zeros(scope, 4));
    tensors.put(p + "self_attn.k_proj.weight", zeros(scope, 2, 4));
    tensors.put(p + "self_attn.k_proj.bias", zeros(scope, 2));
    tensors.put(p + "self_attn.v_proj.weight", zeros(scope, 2, 4));
    tensors.put(p + "self_attn.v_proj.bias", zeros(scope, 2));
    tensors.put(p + "self_attn.o_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "mlp.gate_proj.weight", zeros(scope, 8, 4));
    tensors.put(p + "mlp.up_proj.weight", zeros(scope, 8, 4));
    tensors.put(p + "mlp.down_proj.weight", zeros(scope, 4, 8));
    return tensors;
  }

  private static Path writeTinyTokenizer(Path directory) throws Exception {
    Path path = directory.resolve("tokenizer.json");
    Files.writeString(
        path,
        """
        {
          "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
          "normalizer":null,
          "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,
                           "trim_offsets":true,"use_regex":false},
          "post_processor":null,"decoder":{"type":"ByteLevel"},
          "model":{"type":"BPE","dropout":null,"unk_token":null,
                   "continuing_subword_prefix":"","end_of_word_suffix":"",
                   "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                   "vocab":{"a":0,"p":1,"x":2,"y":3},"merges":[]}
        }
        """);
    return path;
  }

  private static MLXArray zeros(MLXScope scope, int... shape) {
    return MLX.zeros(scope, shape, DType.FLOAT32);
  }
}
