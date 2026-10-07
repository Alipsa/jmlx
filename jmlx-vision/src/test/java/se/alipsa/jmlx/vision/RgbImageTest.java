package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class RgbImageTest {

  @Test
  void constructorRejectsInvalidArguments() {
    assertThrows(IllegalArgumentException.class, () -> new RgbImage(0, 4, new byte[0]));
    assertThrows(IllegalArgumentException.class, () -> new RgbImage(-2, 4, new byte[0]));
    assertThrows(IllegalArgumentException.class, () -> new RgbImage(4, 0, new byte[0]));
    assertThrows(IllegalArgumentException.class, () -> new RgbImage(4, -3, new byte[0]));
    assertThrows(IllegalArgumentException.class, () -> new RgbImage(4, 4, null));
    assertThrows(IllegalArgumentException.class, () -> new RgbImage(2, 2, new byte[11]));
    assertThrows(IllegalArgumentException.class, () -> new RgbImage(2, 2, new byte[13]));
    // 65536 x 65536 x 3 overflows an int; the rejection must not attempt the allocation.
    assertThrows(IllegalArgumentException.class, () -> new RgbImage(65536, 65536, new byte[12]));
  }

  @Test
  void constructorCopiesDefensively() {
    byte[] pixels = new byte[2 * 1 * 3];
    pixels[0] = 1;
    RgbImage image = new RgbImage(2, 1, pixels);
    pixels[0] = 99;
    assertEquals(1, image.pixels()[0]);
    byte[] view = image.pixels();
    view[0] = 77;
    assertEquals(1, image.pixels()[0]);
  }

  @Test
  void equalityComparesDimensionsAndAllPixelBytes() {
    byte[] a = new byte[] {1, 2, 3, 4, 5, 6};
    byte[] b = a.clone();
    RgbImage one = new RgbImage(2, 1, a);
    RgbImage two = new RgbImage(2, 1, b);
    assertEquals(one, two);
    assertEquals(one.hashCode(), two.hashCode());
    b[0] = 9;
    RgbImage three = new RgbImage(2, 1, b);
    assertNotEquals(one, three);
    // Same byte count, different shape: unequal.
    assertNotEquals(one, new RgbImage(1, 2, a));
    assertNotEquals(one, "not an image");
    assertFalse(one.equals(null));
  }

  @Test
  void copyPixelsToBulkCopies() {
    byte[] pixels = new byte[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    RgbImage image = new RgbImage(2, 2, pixels);
    byte[] destination = new byte[20];
    image.copyPixelsTo(destination, 4);
    assertArrayEquals(pixels, java.util.Arrays.copyOfRange(destination, 4, 16));
    assertThrows(IllegalArgumentException.class, () -> image.copyPixelsTo(null, 0));
    assertThrows(IndexOutOfBoundsException.class, () -> image.copyPixelsTo(destination, -1));
    assertThrows(IndexOutOfBoundsException.class, () -> image.copyPixelsTo(destination, 16));
    assertThrows(IndexOutOfBoundsException.class, () -> image.copyPixelsTo(new byte[11], 0));
  }

  @Test
  void toStringReportsDimensionsOnly() {
    assertEquals("RgbImage{2x3}", new RgbImage(2, 3, new byte[18]).toString());
  }
}
