package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;

class TensorPlanTest {
  private static TensorPlan llamaPlan(boolean bias) {
    return ArchitectureMappings.tensorPlan(TestDescriptors.llama(1, bias));
  }

  @Test
  void reportsEveryProblem() {
    TensorPlan plan = llamaPlan(false);
    Set<String> have = new HashSet<>(plan.required());
    have.remove("model.norm.weight");
    have.add("model.layers.0.self_attn.q_proj.bias");
    have.add("model.layers.0.mystery.weight");
    String message =
        assertThrows(IllegalArgumentException.class, () -> plan.validate(have)).getMessage();
    assertTrue(message.contains("missing tensor 'model.norm.weight'"));
    assertTrue(message.contains("forbidden tensor 'model.layers.0.self_attn.q_proj.bias'"));
    assertTrue(message.contains("unexpected tensor 'model.layers.0.mystery.weight'"));
  }

  @Test
  void allowsLegacyRotaryBufferAndOptionalTiedHead() {
    TensorPlan plan = ArchitectureMappings.tensorPlan(TestDescriptors.llama(1, false, true));
    Set<String> have = new HashSet<>(plan.required());
    have.add("model.layers.0.self_attn.rotary_emb.inv_freq");
    have.add("lm_head.weight");
    assertDoesNotThrow(() -> plan.validate(have));
  }

  @Test
  void untiedHeadAndQwenOutputBiasAreChecked() {
    TensorPlan llama = llamaPlan(false);
    Set<String> missingHead = new HashSet<>(llama.required());
    missingHead.remove("lm_head.weight");
    assertTrue(
        assertThrows(IllegalArgumentException.class, () -> llama.validate(missingHead))
            .getMessage()
            .contains("lm_head.weight"));
    TensorPlan qwen = ArchitectureMappings.tensorPlan(TestDescriptors.qwen2(1));
    Set<String> have = new HashSet<>(qwen.required());
    have.add("model.layers.0.self_attn.o_proj.bias");
    assertTrue(
        assertThrows(IllegalArgumentException.class, () -> qwen.validate(have))
            .getMessage()
            .contains("o_proj.bias"));
  }
}
