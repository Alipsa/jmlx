package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import tools.jackson.databind.ObjectMapper;

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
  void prefillAndDecodeMatchHuggingFace() throws Exception {
    FamilyReferenceLogits.assertMatches("mistral", MistralModel.class);
  }
}
