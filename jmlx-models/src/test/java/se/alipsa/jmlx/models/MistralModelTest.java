package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXArray;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import se.alipsa.jmlx.memory.MLXScope;
import se.alipsa.jmlx.nn.KVCache;
import tools.jackson.databind.ObjectMapper;

@Tag("full-float32")
class MistralModelTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void rejectsFusedQkvTensor() {
    ArchitectureDescriptor descriptor =
        ArchitectureMappings.parse(
            MAPPER.readTree(
                """
                {"model_type":"mistral","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
                 "sliding_window":4}
                """));
    TensorPlan plan = ArchitectureMappings.tensorPlan(descriptor);
    Set<String> names = new HashSet<>(plan.required());
    names.add("model.layers.0.self_attn.qkv_proj.weight");
    String message =
        assertThrows(IllegalArgumentException.class, () -> plan.validate(names)).getMessage();
    assertTrue(message.contains("self_attn.qkv_proj.weight"), message);
  }

  @Test
  @EnabledIfNativeAvailable
  void nullCacheReportsWhichCacheIsMissingOnSlidingWindowModels(@TempDir Path dir)
      throws Exception {
    TinyCheckpoints.randomLlama(dir, 1, 2, false, false);
    Path config = dir.resolve("config.json");
    Files.writeString(
        config,
        Files.readString(config)
            .replace(
                "\"model_type\":\"llama\"", "\"model_type\":\"mistral\",\"sliding_window\":4"));
    try (MLXScope scope = new MLXScope()) {
      MistralModel model = MistralModel.load(scope, dir);
      MLXArray ids = MLX.array(scope, new int[] {1, 2}, new int[] {1, 2});
      List<KVCache> caches = new ArrayList<>();
      caches.add(null);
      caches.add(new KVCache(scope));
      String message =
          assertThrows(NullPointerException.class, () -> model.forward(ids, caches)).getMessage();
      assertEquals("cache 0", message);
    }
  }

  @Test
  @EnabledIfNativeAvailable
  void prefillAndDecodeMatchHuggingFace() throws Exception {
    FamilyReferenceLogits.assertMatches("mistral", MistralModel.class);
  }
}
