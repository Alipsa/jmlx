package se.alipsa.jmlx.models;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/** Mixtral checkpoint coverage over a committed Hugging Face reference. */
class MixtralModelTest {
  @Test
  @EnabledIfNativeAvailable
  void prefillAndDecodeMatchHuggingFace() throws Exception {
    FamilyReferenceLogits.assertMatches("mixtral", MixtralModel.class);
  }
}
