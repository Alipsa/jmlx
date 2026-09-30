package se.alipsa.jmlx.models;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/** Llama 3.1 RoPE scaling against the committed Hugging Face checkpoint and logits. */
class Llama31ModelTest {
  @Test
  @EnabledIfNativeAvailable
  void prefillAndDecodeMatchHuggingFace() throws Exception {
    FamilyReferenceLogits.assertMatches("llama31", LlamaModel.class);
  }
}
