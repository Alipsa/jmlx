package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

/**
 * Byte-exact EXIF-orientation parity against the committed Pillow-oracle fixtures: {@code IGNORE}
 * (the default) must reproduce the recorded stored, unrotated decode, and {@code APPLY} must
 * reproduce the recorded {@code ImageOps.exif_transpose} result, covering the LONG-type orientation
 * tag, a truncated IFD walk, the JPEG XMP fallback (attribute and element form), PNG eXIf last-wins
 * / post-IDAT / prefixed semantics, and the plain no-metadata twins.
 */
class ImageOracleExifOrientationTest {

  private static final String[] EXPECTED_NAMES = {
    "jpeg-orient-long",
    "jpeg-ifd-truncated",
    "jpeg-xmp-orient-attr",
    "jpeg-xmp-orient-el",
    "png-exif-twice",
    "png-exif-after-idat",
    "png-exif-prefixed",
    "jpeg-plain",
    "png-plain"
  };

  @Test
  void decodeMatchesThePinnedOracleForEveryCase() throws Exception {
    JsonNode cases = OracleFixtures.read("exif-orientation.input.json").required("cases");
    JsonNode expectedCases =
        OracleFixtures.read("exif-orientation.expected.json").required("cases");
    assertEquals(EXPECTED_NAMES.length, cases.size());
    assertEquals(cases.size(), expectedCases.size());
    assertTrue(cases.size() > 0);
    for (int i = 0; i < cases.size(); i++) {
      JsonNode testCase = cases.get(i);
      String name = testCase.required("name").asString();
      assertEquals(EXPECTED_NAMES[i], name, "input case order");
      JsonNode expectedCase = OracleFixtures.findCase(expectedCases, name);
      JsonNode stored = expectedCase.required("stored");
      Path image = OracleFixtures.fixtures().resolve(testCase.required("image").asString());

      // The default overload must stay on the stored, unrotated pixels.
      RgbImage ignored = ImageDecoder.decode(image);
      assertEquals(stored.required("width").intValue(), ignored.width(), name);
      assertEquals(stored.required("height").intValue(), ignored.height(), name);
      assertArrayEquals(
          OracleFixtures.base64(stored.required("pixelsBase64").asString()),
          ignored.pixels(),
          name);

      // The recorded transposed must be exactly the Pillow table applied to the stored pixels,
      // independent of the decoder (the test's own table is the cross-check).
      int orientation = expectedCase.required("orientation").intValue();
      JsonNode transposed = expectedCase.required("transposed");
      RgbImage derived = permuted(ignored, orientation);
      assertEquals(transposed.required("width").intValue(), derived.width(), name);
      assertEquals(transposed.required("height").intValue(), derived.height(), name);
      assertArrayEquals(
          OracleFixtures.base64(transposed.required("pixelsBase64").asString()),
          derived.pixels(),
          name);

      RgbImage applied =
          ImageDecoder.decode(image, AlphaPolicy.COMPOSITE_OVER_WHITE, ExifOrientationPolicy.APPLY);
      assertEquals(transposed.required("width").intValue(), applied.width(), name);
      assertEquals(transposed.required("height").intValue(), applied.height(), name);
      assertArrayEquals(
          OracleFixtures.base64(transposed.required("pixelsBase64").asString()),
          applied.pixels(),
          name);
    }
  }

  /**
   * Applies the Pillow {@code ImageOps.exif_transpose} table ({@code x} = column, {@code y} = row,
   * a {@code src} pair reading (column, row)) to {@code image}: 2: {@code src[W-1-x, y]}; 3: {@code
   * src[W-1-x, H-1-y]}; 4: {@code src[x, H-1-y]}; 5: {@code src[y, x]}; 6: {@code src[y, H-1-x]};
   * 7: {@code src[W-1-y, H-1-x]}; 8: {@code src[W-1-y, x]}; the result dimensions swap for 5-8; any
   * other orientation is returned untouched.
   */
  private static RgbImage permuted(RgbImage image, int orientation) {
    if (orientation < 2 || orientation > 8) {
      return image;
    }
    int w = image.width();
    int h = image.height();
    int w2 = orientation >= 5 ? h : w;
    int h2 = orientation >= 5 ? w : h;
    byte[] src = image.pixels();
    byte[] dst = new byte[src.length];
    for (int y = 0; y < h2; y++) {
      for (int x = 0; x < w2; x++) {
        int sx;
        int sy;
        switch (orientation) {
          case 2 -> {
            sx = w - 1 - x;
            sy = y;
          }
          case 3 -> {
            sx = w - 1 - x;
            sy = h - 1 - y;
          }
          case 4 -> {
            sx = x;
            sy = h - 1 - y;
          }
          case 5 -> {
            sx = y;
            sy = x;
          }
          case 6 -> {
            sx = y;
            sy = h - 1 - x;
          }
          case 7 -> {
            sx = w - 1 - y;
            sy = h - 1 - x;
          }
          default -> {
            sx = w - 1 - y;
            sy = x;
          }
        }
        System.arraycopy(src, (sy * w + sx) * 3, dst, (y * w2 + x) * 3, 3);
      }
    }
    return new RgbImage(w2, h2, dst);
  }
}
