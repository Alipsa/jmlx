package se.alipsa.jmlx.jinja.internal;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TemplateFormatterTest {
  @Test
  void rejectsOnlyRepeatedIndentStringsBeyondThePinnedNodeLimit() {
    assertDoesNotThrow(() -> TemplateFormatter.validateRepeatedIndentLength(2_097_151, 256));
    assertThrows(
        IllegalArgumentException.class,
        () -> TemplateFormatter.validateRepeatedIndentLength(2_097_152, 256));
  }
}
