package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

/** Byte-exact resize comparison against the committed Pillow-oracle fixtures. */
class ImageOracleResizeTest {

  private static Resampling kernel(int resample, String name) {
    return switch (resample) {
      case 1 -> Resampling.LANCZOS;
      case 2 -> Resampling.BILINEAR;
      case 3 -> Resampling.BICUBIC;
      default ->
          throw new IllegalStateException("unsupported oracle resample " + resample + ": " + name);
    };
  }

  @Test
  void resizeMatchesThePinnedOracleForEveryCase() throws Exception {
    JsonNode cases = OracleFixtures.read("resize.input.json").required("cases");
    JsonNode expectedCases = OracleFixtures.read("resize.expected.json").required("cases");
    assertEquals(cases.size(), expectedCases.size());
    assertTrue(cases.size() > 0);
    for (int i = 0; i < cases.size(); i++) {
      JsonNode testCase = cases.get(i);
      String name = testCase.required("name").asString();
      JsonNode expectedCase = OracleFixtures.findCase(expectedCases, name);
      RgbImage source =
          ImageDecoder.decode(
              OracleFixtures.fixtures().resolve(testCase.required("image").asString()));
      assertArrayEquals(
          OracleFixtures.ints(expectedCase.required("inputShape")),
          new int[] {source.height(), source.width()},
          name);
      RgbImage resized =
          ImageTransforms.resize(
              source,
              expectedCase.required("width").intValue(),
              expectedCase.required("height").intValue(),
              kernel(testCase.required("resample").intValue(), name));
      assertEquals(expectedCase.required("width").intValue(), resized.width(), name);
      assertEquals(expectedCase.required("height").intValue(), resized.height(), name);
      assertArrayEquals(
          OracleFixtures.base64(expectedCase.required("pixelsBase64").asString()),
          resized.pixels(),
          name);
    }
  }
}
