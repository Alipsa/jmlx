package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ModelMetadataTest {
  @Test
  void validatesDirectMetadataConstruction() {
    assertThrows(NullPointerException.class, () -> new DecoderMetadata(null, 4, 1));
    assertThrows(IllegalArgumentException.class, () -> new DecoderMetadata(" ", 4, 1));
    assertThrows(IllegalArgumentException.class, () -> new DecoderMetadata("llama", 0, 1));
    assertThrows(IllegalArgumentException.class, () -> new DecoderMetadata("llama", 4, 0));
  }

  @Test
  void exposesStableMetadataFields() {
    ModelMetadata metadata = new DecoderMetadata("llama", 32_000, 32);

    assertEquals("llama", metadata.modelType());
    assertEquals(32_000, metadata.vocabSize());
    assertEquals(32, metadata.numHiddenLayers());
  }
}
