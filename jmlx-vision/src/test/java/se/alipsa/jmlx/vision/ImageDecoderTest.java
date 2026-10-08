package se.alipsa.jmlx.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.Test;
import tools.jackson.databind.JsonNode;

class ImageDecoderTest {

  /** Exact rejection messages, probed against the pinned fixtures and recorded here by name. */
  private static final Map<String, String> REJECTION_MESSAGES =
      Map.ofEntries(
          Map.entry(
              "reject-bmp-32x32",
              "unsupported image input: expected a PNG or JPEG file, "
                  + "found input starting with 0x42 0x4d 0x36 0x0c"),
          Map.entry("reject-png-1bit-gray", "1-bit grayscale PNG is not supported (only 8-bit)"),
          Map.entry("reject-png-4bit-gray", "4-bit grayscale PNG is not supported (only 8-bit)"),
          Map.entry("reject-png-16bit-gray", "16-bit PNG is not supported"),
          Map.entry("reject-png-16bit-rgb", "16-bit PNG is not supported"),
          Map.entry("reject-png-truncated", "corrupt PNG: truncated chunk at offset 33"),
          Map.entry(
              "reject-jpeg-cmyk-4comp", "CMYK/YCCK JPEG is not supported (four-component frame)"),
          Map.entry(
              "reject-jpeg-adobe-transform2",
              "CMYK/YCCK JPEG is not supported (Adobe APP14 transform code 2)"),
          Map.entry("reject-jpeg-truncated", "corrupt JPEG: truncated scan data"));

  @Test
  void decodeMatchesThePinnedOracleForEveryFixture() throws Exception {
    JsonNode cases = OracleFixtures.read("decode.input.json").required("cases");
    JsonNode expectedCases = OracleFixtures.read("decode.expected.json").required("cases");
    assertEquals(cases.size(), expectedCases.size());
    assertTrue(cases.size() > 0);
    for (int i = 0; i < cases.size(); i++) {
      JsonNode testCase = cases.get(i);
      String name = testCase.required("name").asString();
      JsonNode expectedCase = OracleFixtures.findCase(expectedCases, name);
      Path image = OracleFixtures.fixtures().resolve(testCase.required("image").asString());
      if (expectedCase.path("expectedRejected").asBoolean()) {
        IOException e = assertThrows(IOException.class, () -> ImageDecoder.decode(image), name);
        assertEquals(REJECTION_MESSAGES.get(name), e.getMessage(), name);
        continue;
      }
      // The limits fixtures exceed the default limits and are the subject of
      // defaultLimitsRejectTheLimitImagesExplicitLimitsAcceptThem; here they are decoded with
      // explicit limits so their pixels can be checked too.
      RgbImage decoded =
          testCase.path("defaultLimitsRejected").asBoolean()
              ? ImageDecoder.decode(image, new ImageDecodeLimits(60_000_000L, 25_000))
              : ImageDecoder.decode(image);
      assertEquals(expectedCase.required("width").intValue(), decoded.width(), name);
      assertEquals(expectedCase.required("height").intValue(), decoded.height(), name);
      if (expectedCase.has("pixelSha256")) {
        assertEquals(
            expectedCase.required("pixelSha256").asString(),
            OracleFixtures.sha256Hex(decoded.pixels()),
            name);
        JsonNode samples = expectedCase.required("samples");
        for (int s = 0; s < samples.size(); s++) {
          JsonNode sample = samples.get(s);
          byte[] patch =
              OracleFixtures.crop(
                  decoded,
                  sample.required("x0").intValue(),
                  sample.required("y0").intValue(),
                  sample.required("x1").intValue(),
                  sample.required("y1").intValue());
          assertArrayEquals(
              OracleFixtures.base64(sample.required("pixelsBase64").asString()), patch, name);
        }
      } else {
        assertArrayEquals(
            OracleFixtures.base64(expectedCase.required("pixelsBase64").asString()),
            decoded.pixels(),
            name);
      }
    }
  }

  @Test
  void defaultLimitsRejectTheLimitImagesExplicitLimitsAcceptThem() throws Exception {
    JsonNode cases = OracleFixtures.read("decode.input.json").required("cases");
    JsonNode expectedCases = OracleFixtures.read("decode.expected.json").required("cases");
    int checked = 0;
    for (int i = 0; i < cases.size(); i++) {
      JsonNode testCase = cases.get(i);
      String name = testCase.required("name").asString();
      JsonNode expectedCase = OracleFixtures.findCase(expectedCases, name);
      if (!testCase.path("defaultLimitsRejected").asBoolean()) {
        continue;
      }
      checked++;
      Path image = OracleFixtures.fixtures().resolve(testCase.required("image").asString());
      int width = expectedCase.required("width").intValue();
      int height = expectedCase.required("height").intValue();
      String expectedMessage;
      if (width > 16_384 || height > 16_384) {
        expectedMessage =
            "image dimension " + width + "x" + height + " exceeds the maximum dimension 16384";
      } else {
        expectedMessage =
            "image size "
                + width
                + "x"
                + height
                + " ("
                + (long) width * height
                + " pixels) exceeds the maximum of 16777216 pixels";
      }
      IOException e = assertThrows(IOException.class, () -> ImageDecoder.decode(image), name);
      assertEquals(expectedMessage, e.getMessage(), name);
      RgbImage decoded = ImageDecoder.decode(image, new ImageDecodeLimits(60_000_000L, 25_000));
      assertEquals(width, decoded.width(), name);
      assertEquals(height, decoded.height(), name);
      assertEquals(
          expectedCase.required("pixelSha256").asString(),
          OracleFixtures.sha256Hex(decoded.pixels()),
          name);
    }
    assertEquals(3, checked);
  }

  @Test
  void iccProfileAndExifDoNotChangeDecodedPixels() throws Exception {
    // The oracle pins the P3 twin to byte-identical pixels (no ICC management), and the
    // orientation-6 EXIF JPEG decodes to its stored, unrotated 100x60.
    RgbImage p3 = ImageDecoder.decode(OracleFixtures.fixtures().resolve("decode/icc-p3-32x32.png"));
    RgbImage twin =
        ImageDecoder.decode(OracleFixtures.fixtures().resolve("decode/icc-p3-twin-32x32.png"));
    assertEquals(p3, twin);
    RgbImage exif =
        ImageDecoder.decode(OracleFixtures.fixtures().resolve("decode/exif-orient6-100x60.jpg"));
    assertEquals(100, exif.width());
    assertEquals(60, exif.height());
  }

  @Test
  void tRnsOnTruecolorAndGrayscaleDecodesAsOpaque() throws Exception {
    // A tRNS chunk on truecolor/grayscale is ignored (the reference keeps mode RGB/L and its
    // RGB conversion drops the transparency): each twin decodes byte-identical to its opaque
    // counterpart.
    RgbImage rgb = ImageDecoder.decode(OracleFixtures.fixtures().resolve("decode/rgb-64x48.png"));
    RgbImage rgbTrns =
        ImageDecoder.decode(OracleFixtures.fixtures().resolve("decode/rgb-trns-64x48.png"));
    assertEquals(rgb, rgbTrns);
    RgbImage gray = ImageDecoder.decode(OracleFixtures.fixtures().resolve("decode/gray-60x40.png"));
    RgbImage grayTrns =
        ImageDecoder.decode(OracleFixtures.fixtures().resolve("decode/gray-trns-60x40.png"));
    assertEquals(gray, grayTrns);
  }

  @Test
  void tRnsOnPalettedPngDecodesAsOpaqueTwin() throws Exception {
    // Palette tRNS is dropped to match the pinned processor (its palette is rebuilt from
    // getpalette(), RGB only): each tRNS twin decodes byte-identical to its opaque counterpart.
    Path fixtures = OracleFixtures.fixtures();
    RgbImage palette = ImageDecoder.decode(fixtures.resolve("decode/palette-32x32.png"));
    assertEquals(
        palette, ImageDecoder.decode(fixtures.resolve("decode/palette-trns-single-32x32.png")));
    assertEquals(
        palette, ImageDecoder.decode(fixtures.resolve("decode/palette-trns-perentry-32x32.png")));
    assertEquals(
        palette, ImageDecoder.decode(fixtures.resolve("decode/palette-trns-short-32x32.png")));
    RgbImage palette8 = ImageDecoder.decode(fixtures.resolve("decode/palette8-32x32.png"));
    assertEquals(
        palette8, ImageDecoder.decode(fixtures.resolve("decode/palette8-trns-single-32x32.png")));
    assertEquals(
        palette8, ImageDecoder.decode(fixtures.resolve("decode/palette8-trns-perentry-32x32.png")));
  }

  @Test
  void truecolorPngWithSuggestedPlteDecodesAsOpaqueTwin() throws Exception {
    // A PLTE chunk on a truecolor image is only a *suggested* palette (RFC 2083 s11.2); the
    // decoder must ignore it, exactly as Pillow does.
    Path rgb = OracleFixtures.fixtures().resolve("decode/rgb-64x48.png");
    byte[] bytes = Files.readAllBytes(rgb);
    byte[] plte = new byte[12]; // four entries; the content is irrelevant
    for (int i = 0; i < plte.length; i++) {
      plte[i] = (byte) (i * 21);
    }
    RgbImage expected = ImageDecoder.decode(rgb);
    RgbImage decoded =
        ImageDecoder.decode(new ByteArrayInputStream(insertPngChunkAfterIhdr(bytes, "PLTE", plte)));
    assertEquals(expected, decoded);
  }

  @Test
  void duplicateTrnsChunksAreAllStripped() throws Exception {
    // Malformed input may carry more than one tRNS chunk, and the built-in reader would honor
    // each of them (a surviving truecolor tRNS adds an alpha band, which the band-count check
    // rejects). Every one must be stripped.
    Path rgb = OracleFixtures.fixtures().resolve("decode/rgb-64x48.png");
    byte[] bytes = Files.readAllBytes(rgb);
    byte[] twoTrns =
        insertPngChunkAfterIhdr(
            insertPngChunkAfterIhdr(bytes, "tRNS", new byte[] {10, 20, 30}),
            "tRNS",
            new byte[] {40, 50, 60});
    RgbImage expected = ImageDecoder.decode(rgb);
    assertEquals(expected, ImageDecoder.decode(new ByteArrayInputStream(twoTrns)));
    // The same for a palette image: a surviving tRNS would mark palette entries transparent.
    Path palette = OracleFixtures.fixtures().resolve("decode/palette8-32x32.png");
    byte[] paletteBytes = Files.readAllBytes(palette);
    byte[] paletteTwoTrns =
        insertPngChunkAfterIhdr(
            insertPngChunkAfterIhdr(paletteBytes, "tRNS", new byte[] {1, (byte) 0xFF}),
            "tRNS",
            new byte[] {2, (byte) 0xFF});
    RgbImage paletteExpected = ImageDecoder.decode(palette);
    assertEquals(paletteExpected, ImageDecoder.decode(new ByteArrayInputStream(paletteTwoTrns)));
  }

  @Test
  void fillByteBeforeTheEoiMarkerIsAccepted() throws Exception {
    // A fill byte directly before a marker must advance one byte, not two: FF FF D9 (a legal
    // fill byte before the EOI) used to be consumed as scan data and the file rejected.
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    assertEquals(0xFF, jpeg[jpeg.length - 2] & 0xFF, "fixture must end with the EOI marker");
    assertEquals(0xD9, jpeg[jpeg.length - 1] & 0xFF, "fixture must end with the EOI marker");
    RgbImage expected = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    for (int fills = 1; fills <= 2; fills++) {
      byte[] variant = new byte[jpeg.length + fills];
      System.arraycopy(jpeg, 0, variant, 0, jpeg.length - 2);
      Arrays.fill(variant, jpeg.length - 2, variant.length - 2, (byte) 0xFF);
      variant[variant.length - 2] = (byte) 0xFF;
      variant[variant.length - 1] = (byte) 0xD9;
      assertEquals(
          expected, ImageDecoder.decode(new ByteArrayInputStream(variant)), "fill count " + fills);
    }
  }

  @Test
  void shortSofSegmentIsRejectedWithIOException() {
    // The component count sits at segment offset 9, so a SOF declared with a length of 7
    // cannot be read; before the fix, a length-7 segment at end of file threw
    // ArrayIndexOutOfBoundsException instead of this IOException.
    IOException truncated =
        assertThrows(
            IOException.class,
            () -> ImageDecoder.decode(new ByteArrayInputStream(hex("FFD8FFC000070800100010"))));
    assertEquals("corrupt JPEG: truncated SOF segment", truncated.getMessage());
    // A declared length of 8 covers the count byte, so the walk continues past the segment and
    // the rest of the file (none of it present) is rejected as a truncated marker.
    IOException shortComponents =
        assertThrows(
            IOException.class,
            () -> ImageDecoder.decode(new ByteArrayInputStream(hex("FFD8FFC00008080010001003"))));
    assertEquals("corrupt JPEG: truncated marker at offset 12", shortComponents.getMessage());
  }

  /** Inserts a chunk with a valid CRC after the IHDR (offset 33) of a PNG byte sequence. */
  private static byte[] insertPngChunkAfterIhdr(byte[] png, String type, byte[] data) {
    byte[] chunk = new byte[12 + data.length];
    chunk[0] = (byte) (data.length >>> 24);
    chunk[1] = (byte) (data.length >>> 16);
    chunk[2] = (byte) (data.length >>> 8);
    chunk[3] = (byte) data.length;
    byte[] t = type.getBytes(java.nio.charset.StandardCharsets.US_ASCII);
    System.arraycopy(t, 0, chunk, 4, 4);
    System.arraycopy(data, 0, chunk, 8, data.length);
    java.util.zip.CRC32 crc = new java.util.zip.CRC32();
    crc.update(chunk, 4, 4 + data.length);
    long c = crc.getValue();
    chunk[8 + data.length] = (byte) (c >>> 24);
    chunk[9 + data.length] = (byte) (c >>> 16);
    chunk[10 + data.length] = (byte) (c >>> 8);
    chunk[11 + data.length] = (byte) c;
    byte[] out = new byte[png.length + chunk.length];
    System.arraycopy(png, 0, out, 0, 33);
    System.arraycopy(chunk, 0, out, 33, chunk.length);
    System.arraycopy(png, 33, out, 33 + chunk.length, png.length - 33);
    return out;
  }

  private static byte[] hex(String s) {
    byte[] out = new byte[s.length() / 2];
    for (int i = 0; i < out.length; i++) {
      out[i] = (byte) Integer.parseInt(s.substring(i * 2, i * 2 + 2), 16);
    }
    return out;
  }

  @Test
  void jpegRstFixtureCarriesRealDriAndRestartMarkers() throws Exception {
    // DRI coverage comes from the oracle fixture jpeg-rst-96x64, whose decode is byte-compared
    // against Pillow in decodeMatchesThePinnedOracleForEveryFixture - but only while the
    // fixture really carries the structures under test. Guard that here: a DRI segment
    // (FF DD with a length of 4; the marker PIL's writer emits together with the RST markers)
    // before the SOS scan, and RST0-RST7 markers inside the scan. (No injected-segment test
    // is possible: a DRI spliced into a scan that has no RST markers is not pixel-neutral,
    // because the decoder resets the DC predictor at every restart interval.)
    byte[] b = Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-rst-96x64.jpg"));
    int dri = -1;
    for (int i = 0; i + 5 < b.length; i++) {
      if ((b[i] & 0xFF) == 0xFF
          && (b[i + 1] & 0xFF) == 0xDD
          && (b[i + 2] & 0xFF) == 0x00
          && (b[i + 3] & 0xFF) == 0x04) { // DRI segment, length 4
        dri = i;
        break;
      }
    }
    assertTrue(dri >= 0, "DRI segment (FF DD, length 4) not found in the fixture");
    int sos = -1;
    for (int i = 0; i + 1 < b.length; i++) {
      if ((b[i] & 0xFF) == 0xFF && (b[i + 1] & 0xFF) == 0xDA) {
        sos = i;
        break;
      }
    }
    assertTrue(sos > dri, "the SOS scan must follow the DRI segment");
    // Walk the entropy-coded scan, with the same semantics as the decoder's own scan: an FF
    // followed by another FF is a fill byte and advances one byte (the second FF is re-examined
    // as a marker candidate, so a fill directly before the EOI is not swallowed); FF 00 a stuffed
    // data byte and FF D0-D7 a restart marker each advance two; any other FF xx ends the walk (a
    // real marker, e.g. the terminating EOI).
    int pos = sos + 2;
    int length = ((b[pos] & 0xFF) << 8) | (b[pos + 1] & 0xFF);
    pos += 2 + length;
    int rst = 0;
    while (pos + 1 < b.length) {
      if ((b[pos] & 0xFF) != 0xFF) {
        pos++;
        continue;
      }
      int m = b[pos + 1] & 0xFF;
      if (m == 0xFF) {
        pos++;
        continue;
      }
      if (m == 0x00) {
        pos += 2;
        continue;
      }
      if (m >= 0xD0 && m <= 0xD7) {
        rst++;
        pos += 2;
        continue;
      }
      break;
    }
    assertTrue(rst > 0, "no RST markers found in the scan");
  }

  @Test
  void decodeLeavesTheCallerStreamOpen() throws Exception {
    byte[] bytes = Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/rgb-64x48.png"));
    ClosingStream stream = new ClosingStream(bytes);
    RgbImage image = ImageDecoder.decode(stream);
    assertEquals(64, image.width());
    assertFalse(stream.closed);
  }

  private static final class ClosingStream extends ByteArrayInputStream {
    boolean closed;

    ClosingStream(byte[] bytes) {
      super(bytes);
    }

    @Override
    public void close() {
      closed = true;
    }
  }

  @Test
  void decodeRejectsNullArguments() {
    assertThrows(IllegalArgumentException.class, () -> ImageDecoder.decode((Path) null));
    assertThrows(IllegalArgumentException.class, () -> ImageDecoder.decode((InputStream) null));
    assertThrows(
        IllegalArgumentException.class,
        () -> ImageDecoder.decode((Path) null, new ImageDecodeLimits(1, 1)));
    assertThrows(
        IllegalArgumentException.class,
        () -> ImageDecoder.decode((InputStream) null, new ImageDecodeLimits(1, 1)));
    assertThrows(
        IllegalArgumentException.class,
        () -> ImageDecoder.decode(OracleFixtures.fixtures().resolve("decode/rgb-64x48.png"), null));
  }

  @Test
  void limitsValidatePositivity() {
    assertThrows(IllegalArgumentException.class, () -> new ImageDecodeLimits(0, 16384));
    assertThrows(IllegalArgumentException.class, () -> new ImageDecodeLimits(-1, 16384));
    assertThrows(IllegalArgumentException.class, () -> new ImageDecodeLimits(16_777_216L, 0));
  }

  @Test
  void nonImageInputIsRejectedFromTheMagicBeforeBuffering() {
    // A byte-by-byte zero stream reporting 512 MiB + 1 bytes: the bounded reader must refuse
    // from the format check on the first read, not after buffering to the cap.
    final long total = 512L * 1024 * 1024 + 1;
    InputStream stream =
        new InputStream() {
          long remaining = total;

          @Override
          public int read() {
            if (remaining <= 0) {
              return -1;
            }
            remaining--;
            return 0;
          }

          @Override
          public int read(byte[] b, int off, int len) {
            if (remaining <= 0) {
              return -1;
            }
            int n = (int) Math.min(len, remaining);
            Arrays.fill(b, off, off + n, (byte) 0);
            remaining -= n;
            return n;
          }
        };
    IOException e = assertThrows(IOException.class, () -> ImageDecoder.decode(stream));
    assertEquals(
        "unsupported image input: expected a PNG or JPEG file, "
            + "found input starting with 0x00 0x00 0x00 0x00",
        e.getMessage());
  }

  @Test
  void oversizedStreamInputIsRejectedByTheDecodeBufferLimit() {
    // The cap follows the limits (8 * maxPixels + 1 MiB); with the smallest limits it is
    // 1048584 bytes, so a stream carrying valid PNG magic but one byte more than the cap is
    // refused.
    IOException e =
        assertThrows(
            IOException.class,
            () ->
                ImageDecoder.decode(
                    new ByteArrayInputStream(pngMagicWithZeros(1048585)),
                    new ImageDecodeLimits(1, 1)));
    assertEquals("image input exceeds the 1048584-byte decode buffer limit", e.getMessage());
  }

  @Test
  void oversizedFileInputIsRejectedBeforeAnyByteIsRead() throws Exception {
    // The Path overload sizes the file first: a file over the cap is refused from its size,
    // before a single byte is read.
    Path file = Files.createTempFile("jmlx-vision-decode-test", ".bin");
    try {
      Files.write(file, pngMagicWithZeros(1048585));
      IOException e =
          assertThrows(
              IOException.class, () -> ImageDecoder.decode(file, new ImageDecodeLimits(1, 1)));
      assertEquals("image input exceeds the 1048584-byte decode buffer limit", e.getMessage());
    } finally {
      Files.deleteIfExists(file);
    }
  }

  /** A PNG signature followed by zero bytes, to the given total length. */
  private static byte[] pngMagicWithZeros(int totalLength) {
    byte[] out = new byte[totalLength];
    out[0] = (byte) 0x89;
    out[1] = 'P';
    out[2] = 'N';
    out[3] = 'G';
    out[4] = 0x0D;
    out[5] = 0x0A;
    out[6] = 0x1A;
    out[7] = 0x0A;
    return out;
  }
}
