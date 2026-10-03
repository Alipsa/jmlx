package se.alipsa.jmlx.models;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/** Gemma v1 checkpoint and Hugging Face reference coverage. */
@Tag("full-float32")
class GemmaModelTest {
  @Test
  @EnabledIfNativeAvailable
  void prefillAndDecodeMatchHuggingFace() throws Exception {
    FamilyReferenceLogits.assertMatches("gemma", GemmaModel.class);
  }
}
