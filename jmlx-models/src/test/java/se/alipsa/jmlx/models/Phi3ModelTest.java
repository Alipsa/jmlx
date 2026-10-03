package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;
import tools.jackson.databind.ObjectMapper;

@Tag("full-float32")
class Phi3ModelTest {
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @Test
  void rejectsSeparateQkvTensor() {
    ArchitectureDescriptor descriptor =
        ArchitectureMappings.parse(
            MAPPER.readTree(
                """
                {"model_type":"phi3","vocab_size":16,"hidden_size":8,"intermediate_size":16,
                 "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1}
                """));
    TensorPlan plan = ArchitectureMappings.tensorPlan(descriptor);
    Set<String> names = new HashSet<>(plan.required());
    names.add("model.layers.0.self_attn.q_proj.weight");
    String message =
        assertThrows(IllegalArgumentException.class, () -> plan.validate(names)).getMessage();
    assertTrue(message.contains("self_attn.q_proj.weight"), message);
  }

  @Test
  @EnabledIfNativeAvailable
  void prefillAndDecodeMatchHuggingFace() throws Exception {
    FamilyReferenceLogits.assertMatches("phi3", Phi3Model.class);
  }
}
