package se.alipsa.jmlx.buildsrc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class JsonParserTest {
  @Test
  void parsesManifestValues() {
    String source =
        """
        {"cap":3000000000,"ids":[1,-2],"vector":[1.5,2e-3],
         "enabled":true,"disabled":false,"missing":null,"chat":"hello\\nworld"}
        """;
    Map<?, ?> value = (Map<?, ?>) new JsonParser(source).parse();
    assertEquals(3000000000L, value.get("cap"));
    assertEquals(List.of(1L, -2L), value.get("ids"));
    assertEquals(List.of(new BigDecimal("1.5"), new BigDecimal("2e-3")), value.get("vector"));
    assertEquals(true, value.get("enabled"));
    assertEquals(false, value.get("disabled"));
    assertEquals(null, value.get("missing"));
    assertEquals("hello\nworld", value.get("chat"));
  }

  @Test
  void parsesEscapesAndLongBoundaries() {
    assertEquals(
        "\"/\\\b\f\n\r\tA😀",
        new JsonParser("\"\\\"\\/\\\\\\b\\f\\n\\r\\t\\u0041\\uD83D\\uDE00\"").parse());
    assertEquals(Long.MIN_VALUE, new JsonParser("-9223372036854775808").parse());
    assertEquals(Long.MAX_VALUE, new JsonParser("9223372036854775807").parse());
  }

  @Test
  void rejectsInvalidSyntaxAndOverflow() {
    for (String source :
        List.of(
            "9223372036854775808",
            "-9223372036854775809",
            "01",
            "1.",
            "1e",
            "+1",
            "NaN",
            "[1,]",
            "{\"x\":null,\"x\":1}",
            "\"\\uD800\"",
            "\"\\uDC00\"",
            "\"\\uZZZZ\"",
            "\"\\x\"",
            "\"\n\"",
            "true false")) {
      assertThrows(IllegalArgumentException.class, () -> new JsonParser(source).parse(), source);
    }
  }

  @Test
  void preservesInventoryStrictMode() {
    assertEquals(
        Map.of("x", List.of("value")), new JsonParser("{\"x\":[\"value\"]}", true).parse());
    for (String source : List.of("1", "1.5", "true", "null", "\"\\n\"", "\"\\u0041\"")) {
      assertThrows(IllegalArgumentException.class, () -> new JsonParser(source, true).parse());
    }
  }
}
