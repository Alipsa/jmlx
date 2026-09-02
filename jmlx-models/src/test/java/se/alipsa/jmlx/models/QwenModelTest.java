package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.DType;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;

@EnabledIfNativeAvailable
class QwenModelTest {
  @Test
  void loadsCheckpointAndGeneratesWithGroupedQueryCache(@TempDir Path dir) throws Exception {
    Files.writeString(
        dir.resolve("config.json"),
        """
        {"model_type":"qwen2","vocab_size":4,"hidden_size":4,"intermediate_size":8,
         "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
         "rms_norm_eps":0.000001,"rope_theta":10000,"tie_word_embeddings":false}
        """);
    try (MLXScope saveScope = new MLXScope()) {
      MLXIO.saveSafetensors(
          dir.resolve("model.safetensors").toString(), tinyCheckpoint(saveScope), Map.of());
    }
    try (MLXScope modelScope = new MLXScope()) {
      QwenModel model = QwenModel.load(modelScope, dir);
      assertEquals(List.of(1, 0, 0), model.generate(new int[] {1}, 2, Set.of()));
    }
  }

  private static Map<String, se.alipsa.jmlx.core.MLXArray> tinyCheckpoint(MLXScope scope) {
    Map<String, se.alipsa.jmlx.core.MLXArray> tensors = new LinkedHashMap<>();
    tensors.put("model.embed_tokens.weight", zeros(scope, 4, 4));
    tensors.put("model.norm.weight", zeros(scope, 4));
    tensors.put("lm_head.weight", zeros(scope, 4, 4));
    String p = "model.layers.0.";
    tensors.put(p + "input_layernorm.weight", zeros(scope, 4));
    tensors.put(p + "post_attention_layernorm.weight", zeros(scope, 4));
    tensors.put(p + "self_attn.q_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "self_attn.k_proj.weight", zeros(scope, 2, 4));
    tensors.put(p + "self_attn.v_proj.weight", zeros(scope, 2, 4));
    tensors.put(p + "self_attn.o_proj.weight", zeros(scope, 4, 4));
    tensors.put(p + "mlp.gate_proj.weight", zeros(scope, 8, 4));
    tensors.put(p + "mlp.up_proj.weight", zeros(scope, 8, 4));
    tensors.put(p + "mlp.down_proj.weight", zeros(scope, 4, 8));
    return tensors;
  }

  private static se.alipsa.jmlx.core.MLXArray zeros(MLXScope scope, int... shape) {
    return MLX.zeros(scope, shape, DType.FLOAT32);
  }
}
