package se.alipsa.jmlx.models;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/** Gemma v1 checkpoint and Hugging Face reference coverage. */
class GemmaModelTest {
  @Test
  @EnabledIfNativeAvailable
  void prefillAndDecodeMatchHuggingFace() throws Exception {
    FamilyReferenceLogits.assertMatches("gemma", GemmaModel.class);
  }
}
