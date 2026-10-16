package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import org.junit.jupiter.api.Test;

/**
 * The EXIF-orientation policy of {@link ImageDecoder}: the default (and every existing overload)
 * decodes to the stored, unrotated pixels, and {@link ExifOrientationPolicy#APPLY} applies the
 * stored orientation exactly as Pillow's {@code ImageOps.exif_transpose} does. The orientation
 * tables below are pinned against the reference (Pillow 12.3.0) in the Phase 7.3b WP2 cross-check;
 * the committed {@code decode/exif-orient6-100x60.jpg} fixture (a 100x60 stored image whose
 * big-endian EXIF block says rotate 90° clockwise) is the end-to-end anchor.
 */
class ImageDecoderExifOrientationTest {

  @Test
  void committedOrient6FixtureIsUnrotatedByDefaultAndRotatedUnderApply() throws Exception {
    Path path = OracleFixtures.fixtures().resolve("decode/exif-orient6-100x60.jpg");
    RgbImage stored = ImageDecoder.decode(path);
    assertEquals(100, stored.width());
    assertEquals(60, stored.height());
    // The explicit IGNORE overload agrees with the default: the stored pixels, untouched.
    assertEquals(
        stored,
        ImageDecoder.decode(
            path,
            ImageDecodeLimits.DEFAULTS,
            AlphaPolicy.COMPOSITE_OVER_WHITE,
            ExifOrientationPolicy.IGNORE));
    // Orientation 6 (PIL ROTATE_270) is a 90° clockwise turn: dst(x,y) = src(col y, row H-1-x),
    // with the dimensions swapped.
    RgbImage applied =
        ImageDecoder.decode(
            path,
            ImageDecodeLimits.DEFAULTS,
            AlphaPolicy.COMPOSITE_OVER_WHITE,
            ExifOrientationPolicy.APPLY);
    assertEquals(60, applied.width());
    assertEquals(100, applied.height());
    assertEquals(permuted(stored, 6), applied);
    // The convenience overloads substitute the default limits.
    assertEquals(
        applied,
        ImageDecoder.decode(path, AlphaPolicy.COMPOSITE_OVER_WHITE, ExifOrientationPolicy.APPLY));
    try (InputStream in = Files.newInputStream(path)) {
      assertEquals(
          applied,
          ImageDecoder.decode(
              in,
              ImageDecodeLimits.DEFAULTS,
              AlphaPolicy.COMPOSITE_OVER_WHITE,
              ExifOrientationPolicy.APPLY));
    }
  }

  @Test
  void allEightOrientationsApplyThePillowPermutationInBothEndians() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    assertEquals(96, stored.width());
    assertEquals(64, stored.height());
    for (int orientation = 2; orientation <= 8; orientation++) {
      for (boolean little : new boolean[] {true, false}) {
        byte[] spliced = spliceJpegApp1(jpeg, exifTiffBlock(little, orientation));
        // Splicing the APP1 segment is pixel-neutral: the default decode is unchanged.
        assertEquals(
            stored,
            ImageDecoder.decode(new ByteArrayInputStream(spliced)),
            label(little, orientation));
        RgbImage applied =
            ImageDecoder.decode(
                new ByteArrayInputStream(spliced),
                ImageDecodeLimits.DEFAULTS,
                AlphaPolicy.COMPOSITE_OVER_WHITE,
                ExifOrientationPolicy.APPLY);
        if (orientation >= 5) {
          assertEquals(64, applied.width(), label(little, orientation));
          assertEquals(96, applied.height(), label(little, orientation));
        } else {
          assertEquals(96, applied.width(), label(little, orientation));
          assertEquals(64, applied.height(), label(little, orientation));
        }
        assertEquals(permuted(stored, orientation), applied, label(little, orientation));
      }
    }
  }

  @Test
  void theTwoByOneRowAnchorsMatchThePillowTransforms() throws Exception {
    // The spec anchors for the stored row a b (2x1): orientations 5 and 6 give a on top of b,
    // 7 and 8 give b on top of a (a 1x2 column). The remaining orientations are checked against
    // the same Pillow table via permuted.
    byte[] a = {10, 20, 30};
    byte[] b = {(byte) 200, (byte) 150, 100};
    byte[] row = new byte[] {a[0], a[1], a[2], b[0], b[1], b[2]};
    RgbImage src = new RgbImage(2, 1, row);
    for (int orientation = 2; orientation <= 8; orientation++) {
      byte[] png = minimalTruecolorPng(2, 1, row, exifTiffBlock(true, orientation));
      RgbImage applied =
          ImageDecoder.decode(
              new ByteArrayInputStream(png),
              ImageDecodeLimits.DEFAULTS,
              AlphaPolicy.COMPOSITE_OVER_WHITE,
              ExifOrientationPolicy.APPLY);
      switch (orientation) {
        case 5, 6 ->
            assertEquals(new RgbImage(1, 2, concat(a, b)), applied, "orientation " + orientation);
        case 7, 8 ->
            assertEquals(new RgbImage(1, 2, concat(b, a)), applied, "orientation " + orientation);
        default -> assertEquals(permuted(src, orientation), applied, "orientation " + orientation);
      }
    }
  }

  @Test
  void noOrientationChangeForOtherTagsValueOneAndAbsentExif() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    // No EXIF at all: APPLY returns the stored pixels byte-identically.
    assertEquals(
        stored,
        ImageDecoder.decode(
            new ByteArrayInputStream(jpeg),
            ImageDecodeLimits.DEFAULTS,
            AlphaPolicy.COMPOSITE_OVER_WHITE,
            ExifOrientationPolicy.APPLY));
    // A valid EXIF block whose only IFD0 tag is 0x010E (Description, ASCII), not Orientation.
    for (boolean little : new boolean[] {true, false}) {
      byte[] spliced = spliceJpegApp1(jpeg, tiffBlockWithOtherTag(little));
      RgbImage ignored = ImageDecoder.decode(new ByteArrayInputStream(spliced));
      assertEquals(stored, ignored, label(little, -1));
      assertEquals(
          ignored,
          ImageDecoder.decode(
              new ByteArrayInputStream(spliced),
              ImageDecodeLimits.DEFAULTS,
              AlphaPolicy.COMPOSITE_OVER_WHITE,
              ExifOrientationPolicy.APPLY),
          label(little, -1));
      // Orientation value 1 is "top-left": no change, in either endianness.
      byte[] orient1 = spliceJpegApp1(jpeg, exifTiffBlock(little, 1));
      assertEquals(
          stored,
          ImageDecoder.decode(
              new ByteArrayInputStream(orient1),
              ImageDecodeLimits.DEFAULTS,
              AlphaPolicy.COMPOSITE_OVER_WHITE,
              ExifOrientationPolicy.APPLY),
          label(little, 1));
    }
  }

  @Test
  void malformedExifNeverThrowsAndNeverTransforms() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    for (boolean little : new boolean[] {true, false}) {
      List<byte[]> blocks =
          List.of(
              // A truncated TIFF block: only the byte order.
              Arrays.copyOf(exifTiffBlock(little, 6), 4),
              // The wrong TIFF magic (43 instead of 42).
              tiffBlockWithMagic(little, 43, 8),
              // A bad byte order (neither II nor MM).
              badByteOrderBlock(little),
              // A first-IFD offset past the end of the block.
              tiffBlockWithIfdOffset(little, 100),
              // A first-IFD offset that overflows a signed 32-bit addition when the bounds
              // check is computed in int arithmetic (0x7FFFFFFF + 2 wraps negative).
              tiffBlockWithIfdOffset(little, 0x7FFFFFFF),
              // An IFD entry count truncated mid-entry: count 2, but only one and a half
              // 12-byte entries present.
              tiffBlockWithTruncatedEntries(little));
      for (int i = 0; i < blocks.size(); i++) {
        byte[] spliced = spliceJpegApp1(jpeg, blocks.get(i));
        RgbImage applied =
            ImageDecoder.decode(
                new ByteArrayInputStream(spliced),
                ImageDecodeLimits.DEFAULTS,
                AlphaPolicy.COMPOSITE_OVER_WHITE,
                ExifOrientationPolicy.APPLY);
        assertEquals(stored, applied, label(little, -1) + " malformed block " + i);
      }
    }
  }

  @Test
  void pngExifChunkIsHonoredWithOrientationSix() throws Exception {
    // A minimal 8-bit truecolor PNG (4x2) with an eXIf chunk (little-endian block, orientation
    // 6): APPLY decodes to the 90° clockwise transform, IGNORE to the unrotated stored pixels.
    int width = 4;
    int height = 2;
    byte[] pixels = new byte[width * height * 3];
    int p = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        pixels[p++] = (byte) ((x * 40 + y * 50) % 256);
        pixels[p++] = (byte) ((x * 10 + y * 80) % 256);
        pixels[p++] = (byte) ((y * 120 + x * 15) % 256);
      }
    }
    byte[] png = minimalTruecolorPng(width, height, pixels, exifTiffBlock(true, 6));
    RgbImage ignored = ImageDecoder.decode(new ByteArrayInputStream(png));
    assertEquals(4, ignored.width());
    assertEquals(2, ignored.height());
    assertArrayEquals(pixels, ignored.pixels());
    RgbImage applied =
        ImageDecoder.decode(
            new ByteArrayInputStream(png),
            ImageDecodeLimits.DEFAULTS,
            AlphaPolicy.COMPOSITE_OVER_WHITE,
            ExifOrientationPolicy.APPLY);
    assertEquals(2, applied.width());
    assertEquals(4, applied.height());
    assertEquals(permuted(ignored, 6), applied);
  }

  @Test
  void applyThroughTheInputStreamOverloads() throws Exception {
    Path source = OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg");
    byte[] spliced = spliceJpegApp1(Files.readAllBytes(source), exifTiffBlock(false, 6));
    Path file = Files.createTempFile("jmlx-vision-exif-test", ".jpg");
    try {
      Files.write(file, spliced);
      RgbImage viaPath =
          ImageDecoder.decode(file, AlphaPolicy.COMPOSITE_OVER_WHITE, ExifOrientationPolicy.APPLY);
      assertEquals(64, viaPath.width());
      assertEquals(96, viaPath.height());
      // The 3-arg stream convenience overload substitutes the default limits, like the path one.
      try (InputStream in = new ByteArrayInputStream(spliced)) {
        assertEquals(
            viaPath,
            ImageDecoder.decode(in, AlphaPolicy.COMPOSITE_OVER_WHITE, ExifOrientationPolicy.APPLY));
      }
      // The explicit 4-arg stream overload agrees as well.
      try (InputStream in = new ByteArrayInputStream(spliced)) {
        assertEquals(
            viaPath,
            ImageDecoder.decode(
                in,
                ImageDecodeLimits.DEFAULTS,
                AlphaPolicy.COMPOSITE_OVER_WHITE,
                ExifOrientationPolicy.APPLY));
      }
    } finally {
      Files.deleteIfExists(file);
    }
  }

  @Test
  void nullOrientationIsRejectedLikeTheOtherNullArguments() throws Exception {
    Path path = OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg");
    ImageDecodeLimits limits = new ImageDecodeLimits(1_000_000L, 1_024);
    IllegalArgumentException orientation =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageDecoder.decode(
                    path, limits, AlphaPolicy.COMPOSITE_OVER_WHITE, (ExifOrientationPolicy) null));
    assertEquals("orientation must not be null", orientation.getMessage());
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ImageDecoder.decode(
                path, AlphaPolicy.COMPOSITE_OVER_WHITE, (ExifOrientationPolicy) null));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ImageDecoder.decode(
                (InputStream) null,
                limits,
                AlphaPolicy.COMPOSITE_OVER_WHITE,
                ExifOrientationPolicy.APPLY));
    assertThrows(
        IllegalArgumentException.class,
        () ->
            ImageDecoder.decode(
                (Path) null,
                limits,
                AlphaPolicy.COMPOSITE_OVER_WHITE,
                ExifOrientationPolicy.APPLY));
    // The pre-existing null checks keep their messages and their order.
    IllegalArgumentException policy =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageDecoder.decode(path, limits, (AlphaPolicy) null, ExifOrientationPolicy.APPLY));
    assertEquals("policy must not be null", policy.getMessage());
    IllegalArgumentException existing =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                ImageDecoder.decode(
                    path, (ImageDecodeLimits) null, AlphaPolicy.COMPOSITE_OVER_WHITE));
    assertEquals("limits must not be null", existing.getMessage());
  }

  // ----------------------------------------------------------------------------- helpers

  private static String label(boolean little, int orientation) {
    return "orientation " + orientation + " " + (little ? "II" : "MM");
  }

  private static byte[] concat(byte[] a, byte[] b) {
    byte[] out = new byte[a.length + b.length];
    System.arraycopy(a, 0, out, 0, a.length);
    System.arraycopy(b, 0, out, a.length, b.length);
    return out;
  }

  /**
   * The reference exif_transpose permutation of a stored {@code w x h} RGB buffer (the table on
   * {@link ExifOrientationPolicy#APPLY}), used as the expected value throughout these tests.
   */
  private static RgbImage permuted(RgbImage src, int orientation) {
    int w = src.width();
    int h = src.height();
    boolean dimsSwap = orientation >= 5;
    int w2 = dimsSwap ? h : w;
    int h2 = dimsSwap ? w : h;
    byte[] out = new byte[w2 * h2 * 3];
    byte[] pixels = src.pixels();
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
        System.arraycopy(pixels, (sy * w + sx) * 3, out, (y * w2 + x) * 3, 3);
      }
    }
    return new RgbImage(w2, h2, out);
  }

  /**
   * Splices an APP1 EXIF segment (the 6-byte {@code Exif\x00\x00} prefix plus a TIFF block)
   * immediately after the SOI.
   */
  private static byte[] spliceJpegApp1(byte[] jpeg, byte[] tiffBlock) {
    if ((jpeg[0] & 0xFF) != 0xFF || (jpeg[1] & 0xFF) != 0xD8) {
      throw new IllegalArgumentException("not a JPEG: expected the SOI marker");
    }
    byte[] payload = new byte[6 + tiffBlock.length];
    payload[0] = 'E';
    payload[1] = 'x';
    payload[2] = 'i';
    payload[3] = 'f';
    payload[4] = 0;
    payload[5] = 0;
    System.arraycopy(tiffBlock, 0, payload, 6, tiffBlock.length);
    byte[] out = new byte[jpeg.length + 4 + payload.length];
    out[0] = (byte) 0xFF;
    out[1] = (byte) 0xD8;
    out[2] = (byte) 0xFF;
    out[3] = (byte) 0xE1;
    int segmentLength = payload.length + 2; // includes the two length bytes
    out[4] = (byte) (segmentLength >>> 8);
    out[5] = (byte) segmentLength;
    System.arraycopy(payload, 0, out, 6, payload.length);
    System.arraycopy(jpeg, 2, out, 6 + payload.length, jpeg.length - 2);
    return out;
  }

  /**
   * The 26-byte TIFF/IFD block of a single-entry Orientation tag, the same recipe the reference
   * probe ({@code tools/hf-reference/qwen3vl_probe.py}) uses.
   */
  private static byte[] exifTiffBlock(boolean little, int orientation) {
    Tiff tiff = new Tiff(little);
    tiff.header(42, 8);
    tiff.u16(1); // one IFD entry
    tiff.u16(0x0112); // Orientation
    tiff.u16(3); // SHORT
    tiff.u32(1);
    tiff.u16(orientation);
    tiff.u16(0); // pad the second half of the value field
    tiff.u32(0); // no next IFD
    return tiff.bytes();
  }

  /**
   * The 8-byte TIFF header ({@code II}/{@code MM} byte order, magic, first-IFD offset), in the
   * block's endianness.
   */
  private static byte[] tiffHeader(boolean little, int magic, int ifdOffset) {
    byte[] h = new byte[8];
    h[0] = (byte) (little ? 'I' : 'M');
    h[1] = (byte) (little ? 'I' : 'M');
    writeU16(h, 2, magic, little);
    writeU32(h, 4, ifdOffset, little);
    return h;
  }

  /** A 12-byte IFD entry; a SHORT value goes in the 2-byte half of the value field. */
  private static byte[] ifdEntry(int tag, int type, int count, int value, boolean little) {
    byte[] e = new byte[12];
    writeU16(e, 0, tag, little);
    writeU16(e, 2, type, little);
    writeU32(e, 4, count, little);
    writeU16(e, 8, value, little);
    return e;
  }

  private static void writeU16(byte[] b, int off, int v, boolean little) {
    if (little) {
      b[off] = (byte) v;
      b[off + 1] = (byte) (v >>> 8);
    } else {
      b[off] = (byte) (v >>> 8);
      b[off + 1] = (byte) v;
    }
  }

  private static void writeU32(byte[] b, int off, int v, boolean little) {
    if (little) {
      b[off] = (byte) v;
      b[off + 1] = (byte) (v >>> 8);
      b[off + 2] = (byte) (v >>> 16);
      b[off + 3] = (byte) (v >>> 24);
    } else {
      b[off] = (byte) (v >>> 24);
      b[off + 1] = (byte) (v >>> 16);
      b[off + 2] = (byte) (v >>> 8);
      b[off + 3] = (byte) v;
    }
  }

  /** A valid EXIF block in which the only IFD0 tag is 0x010E (Description, ASCII "abc\0"). */
  private static byte[] tiffBlockWithOtherTag(boolean little) {
    byte[] entry = new byte[12];
    writeU16(entry, 0, 0x010E, little);
    writeU16(entry, 2, 2, little); // ASCII
    writeU32(entry, 4, 4, little);
    entry[8] = 'a';
    entry[9] = 'b';
    entry[10] = 'c';
    entry[11] = 0;
    byte[] header = tiffHeader(little, 42, 8);
    return concat(concat(header, entry), new byte[4]);
  }

  /** A full one-entry Orientation block with the given TIFF magic. */
  private static byte[] tiffBlockWithMagic(boolean little, int magic, int ifdOffset) {
    byte[] header = tiffHeader(little, magic, ifdOffset);
    return concat(concat(header, ifdEntry(0x0112, 3, 1, 6, little)), new byte[4]);
  }

  /**
   * A full one-entry Orientation block whose first-IFD offset is the given (out-of-range) value.
   */
  private static byte[] tiffBlockWithIfdOffset(boolean little, int ifdOffset) {
    byte[] header = tiffHeader(little, 42, ifdOffset);
    return concat(concat(header, ifdEntry(0x0112, 3, 1, 6, little)), new byte[4]);
  }

  /** A full one-entry Orientation block whose byte order is neither II nor MM. */
  private static byte[] badByteOrderBlock(boolean little) {
    byte[] header = tiffHeader(little, 42, 8);
    header[0] = 'N';
    header[1] = 'N';
    return concat(concat(header, ifdEntry(0x0112, 3, 1, 6, little)), new byte[4]);
  }

  /**
   * A block whose IFD claims two entries but carries one and a half: the entry walk is truncated
   * mid-entry.
   */
  private static byte[] tiffBlockWithTruncatedEntries(boolean little) {
    byte[] header = tiffHeader(little, 42, 8);
    byte[] count = new byte[2];
    writeU16(count, 0, 2, little);
    byte[] halfEntry = Arrays.copyOf(ifdEntry(0x0112, 3, 1, 6, little), 6);
    return concat(concat(concat(header, count), ifdEntry(0x0112, 3, 1, 6, little)), halfEntry);
  }

  /** A little/big-endian byte writer for hand-built TIFF blocks. */
  private static final class Tiff {
    private final boolean little;
    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    Tiff(boolean little) {
      this.little = little;
    }

    void header(int magic, int ifdOffset) {
      out.write(little ? 'I' : 'M');
      out.write(little ? 'I' : 'M');
      u16(magic);
      u32(ifdOffset);
    }

    void u16(int v) {
      if (little) {
        out.write(v & 0xFF);
        out.write((v >>> 8) & 0xFF);
      } else {
        out.write((v >>> 8) & 0xFF);
        out.write(v & 0xFF);
      }
    }

    void u32(int v) {
      if (little) {
        u16(v & 0xFFFF);
        u16((v >>> 16) & 0xFFFF);
      } else {
        u16((v >>> 16) & 0xFFFF);
        u16(v & 0xFFFF);
      }
    }

    byte[] bytes() {
      return out.toByteArray();
    }
  }

  /**
   * A minimal 8-bit truecolor PNG: signature, IHDR, an optional {@code eXIf} chunk (the raw TIFF
   * block, no prefix), one IDAT (raw scanlines, filter type 0) and IEND, with correct CRC32s.
   */
  private static byte[] minimalTruecolorPng(int width, int height, byte[] pixels, byte[] exif)
      throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    out.writeBytes(new byte[] {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, (byte) 0x1A, 0x0A});
    byte[] ihdr = new byte[13];
    writeU32(ihdr, 0, width, false);
    writeU32(ihdr, 4, height, false);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type: truecolor
    writeChunk(out, "IHDR", ihdr);
    if (exif != null) {
      writeChunk(out, "eXIf", exif);
    }
    ByteArrayOutputStream raw = new ByteArrayOutputStream();
    for (int y = 0; y < height; y++) {
      raw.write(0); // filter type: none
      raw.write(pixels, y * width * 3, width * 3);
    }
    Deflater deflater = new Deflater();
    deflater.setInput(raw.toByteArray());
    deflater.finish();
    byte[] compressed = new byte[raw.size() * 2 + 64];
    int n = deflater.deflate(compressed);
    deflater.end();
    writeChunk(out, "IDAT", Arrays.copyOf(compressed, n));
    writeChunk(out, "IEND", new byte[0]);
    return out.toByteArray();
  }

  private static void writeChunk(OutputStream out, String type, byte[] data) throws IOException {
    byte[] t = type.getBytes(StandardCharsets.US_ASCII);
    byte[] length = new byte[4];
    writeU32(length, 0, data.length, false);
    out.write(length);
    out.write(t);
    out.write(data);
    CRC32 crc = new CRC32();
    crc.update(t);
    crc.update(data);
    byte[] crcBytes = new byte[4];
    writeU32(crcBytes, 0, (int) crc.getValue(), false);
    out.write(crcBytes);
  }
}
