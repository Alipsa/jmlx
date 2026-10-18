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
import java.util.function.Function;
import java.util.zip.CRC32;
import java.util.zip.Deflater;
import org.junit.jupiter.api.Test;

/**
 * The EXIF-orientation policy of {@link ImageDecoder}: the default (and every existing overload)
 * decodes to the stored, unrotated pixels, and {@link ExifOrientationPolicy#APPLY} applies the
 * stored orientation exactly as Pillow's {@code ImageOps.exif_transpose} does. The committed {@code
 * decode/exif-orient6-100x60.jpg} fixture (a 100x60 stored image whose big-endian EXIF block says
 * rotate 90° clockwise) is the end-to-end anchor; every other expectation below — the TIFF
 * type/count matrix, the truncation and last-wins rules, and the XMP fallback in both formats — was
 * verified against the reference in its pinned venv ({@code tools/image-oracle/.venv}, Pillow
 * 12.3.0) during the Phase 7.3b WP2 review before hardcoding.
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
              // The little-endian pass uses the BigTIFF prefix II 2b 00 (magic 43 instead of
              // 42): the reference's constructor takes the BigTIFF path and reads past the
              // head, so the head is invalid in either body endianness. The big-endian pass
              // uses MM 2b 00: a magic byte pair the reference's TIFF constructor does not
              // accept at all.
              little ? tiffBlockWithMagic(true, 43, 8) : tiffBlockWithMagic(false, 0x2b00, 8),
              // A bad byte order (neither II nor MM).
              badByteOrderBlock(little),
              // A first-IFD offset past the end of the block.
              tiffBlockWithIfdOffset(little, 100),
              // A first-IFD offset that overflows a signed 32-bit addition when the bounds
              // check is computed in int arithmetic (0x7FFFFFFF + 2 wraps negative).
              tiffBlockWithIfdOffset(little, 0x7FFFFFFF),
              // A single IFD entry truncated mid-entry: no entry is stored, so the tag stays
              // absent.
              tiffBlockWithTruncatedFirstEntry(little),
              // An IFD with a zero entry count: no tags at all.
              tiffBlockWithNoEntries(little));
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
  void truncatedEntriesKeepTheIntactPrefixAndRotate() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    for (boolean little : new boolean[] {true, false}) {
      // The IFD claims two entries but carries one and a half: the reference (Pillow 12.3.0,
      // verified in its pinned venv) keeps the intact first entry — Orientation 6 — and stops at
      // the truncated second one, so the image rotates in both endians.
      byte[] spliced = spliceJpegApp1(jpeg, tiffBlockWithTruncatedEntries(little));
      assertEquals(
          stored, ImageDecoder.decode(new ByteArrayInputStream(spliced)), label(little, 6));
      RgbImage applied = applyDecode(spliced);
      assertEquals(64, applied.width(), label(little, 6));
      assertEquals(96, applied.height(), label(little, 6));
      assertEquals(permuted(stored, 6), applied, label(little, 6));
    }
  }

  @Test
  void anEntryOffsetPastTheEndStopsTheWalkKeepingEarlierEntries() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    for (boolean little : new boolean[] {true, false}) {
      // The reference (Pillow 12.3.0, verified in its pinned venv) reads an offset value with a
      // _safe_read that raises on a short read: the whole IFD walk stops there, keeping the
      // entries stored before it. An intact Orientation 6 before the bad offset still rotates.
      byte[] spliced = spliceJpegApp1(jpeg, badOffsetAfterOrientationBlock(little));
      RgbImage applied = applyDecode(spliced);
      assertEquals(64, applied.width(), label(little, 6));
      assertEquals(96, applied.height(), label(little, 6));
      assertEquals(permuted(stored, 6), applied, label(little, 6));
      // …and a valid entry after the bad offset is never reached, so the orientation stays
      // absent under both the default policy and APPLY.
      byte[] splicedAbsent = spliceJpegApp1(jpeg, badOffsetBeforeOrientationBlock(little));
      assertEquals(
          stored, ImageDecoder.decode(new ByteArrayInputStream(splicedAbsent)), label(little, -1));
      assertEquals(stored, applyDecode(splicedAbsent), label(little, -1));
    }
  }

  @Test
  void anEarlierNonOrientationEntryBadOffsetStopsTheWalk() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    for (boolean little : new boolean[] {true, false}) {
      // The reference (Pillow 12.3.0, verified in its pinned venv) reads the data of every
      // entry with a registered type before it filters by tag: the short read of the 0x8769
      // DOUBLE entry's offset value aborts the walk, so the intact Orientation 6 that
      // follows is never stored and APPLY leaves the pixels as stored.
      byte[] spliced = spliceJpegApp1(jpeg, badOffsetOnPriorEntryBlock(little));
      assertEquals(stored, applyDecode(spliced), label(little, -1));
      // …and because the tag was never stored, an XMP packet supplies orientation 6.
      assertEquals(
          permuted(stored, 6),
          applyDecode(
              spliceJpegPayloads(
                  jpeg, exifPayload(badOffsetOnPriorEntryBlock(little)), xmpPayload(XMP_ATTR_6))),
          label(little, 6));
    }
  }

  @Test
  void orientationTypeMatrixMatchesTheReference() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    byte[] xmp = xmpPayload(XMP_ATTR_6);
    for (TypeCase typeCase : TYPE_CASES) {
      for (boolean little : new boolean[] {true, false}) {
        byte[] block =
            orientationBlock(
                little, typeCase.type(), typeCase.count(), typeCase.value().apply(little));
        // Without XMP: the tag's own (possibly never-rotating) value decides; an absent tag
        // leaves the pixels as stored.
        int plain = typeCase.present() ? typeCase.effective() : 0;
        assertEquals(
            effective(stored, plain),
            applyDecode(spliceJpegApp1(jpeg, block)),
            caseLabel(little, typeCase.label()));
        // With an XMP 6 packet: a present tag (even a never-rotating one) suppresses the
        // fallback; only an absent tag consults it.
        int withXmp = typeCase.present() ? typeCase.effective() : 6;
        assertEquals(
            effective(stored, withXmp),
            applyDecode(spliceJpegPayloads(jpeg, exifPayload(block), xmp)),
            caseLabel(little, typeCase.label()) + " + xmp");
      }
    }
    // Two same-tag entries: the later one overwrites the earlier, like the reference's
    // _tagdata dictionary.
    for (boolean little : new boolean[] {true, false}) {
      assertEquals(
          permuted(stored, 6),
          applyDecode(spliceJpegApp1(jpeg, duplicateOrientationBlock(little))),
          caseLabel(little, "DUP SHORT3 then LONG6"));
    }
  }

  @Test
  void jpegXmpFallbackSuppliesTheOrientationWhenTheTagIsAbsent() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    // No EXIF at all: the XMP packet alone decides.
    assertEquals(
        permuted(stored, 6), applyDecode(spliceJpegPayloads(jpeg, xmpPayload(XMP_ATTR_6))));
    // A packet without the tag leaves the pixels as stored.
    assertEquals(stored, applyDecode(spliceJpegPayloads(jpeg, xmpPayload(XMP_NO_TAG))));
    // The element form (>6) matches too.
    assertEquals(
        permuted(stored, 6), applyDecode(spliceJpegPayloads(jpeg, xmpPayload(XMP_ELEMENT_6))));
    // Exactly one digit is captured: "12" resolves to 1 (no rotation).
    assertEquals(stored, applyDecode(spliceJpegPayloads(jpeg, xmpPayload(XMP_ATTR_12))));
    // The first left-to-right match wins.
    assertEquals(
        permuted(stored, 3), applyDecode(spliceJpegPayloads(jpeg, xmpPayload(XMP_FIRST_3_THEN_6))));
    // Two XMP APP1 segments: the last one before the scan wins.
    assertEquals(
        permuted(stored, 6),
        applyDecode(spliceJpegPayloads(jpeg, xmpPayload(XMP_ATTR_3), xmpPayload(XMP_ATTR_6))));
    // An XMP segment after the scan is ignored, like the reference's _open, which breaks at the
    // first SOS.
    assertEquals(stored, applyDecode(spliceJpegAfterScan(jpeg, xmpPayload(XMP_ATTR_6))));
  }

  @Test
  void jpegXmpFallbackIsSuppressedByAPresentOrientationTag() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    byte[] xmp = xmpPayload(XMP_ATTR_6);
    // A present tag that rotates: the EXIF value (3) wins over the XMP's 6, in both endians.
    for (boolean little : new boolean[] {true, false}) {
      assertEquals(
          permuted(stored, 3),
          applyDecode(spliceJpegPayloads(jpeg, exifPayload(exifTiffBlock(little, 3)), xmp)),
          label(little, 3));
    }
    // A present tag whose value never rotates (0) suppresses the fallback.
    assertEquals(
        stored,
        applyDecode(
            spliceJpegPayloads(
                jpeg, exifPayload(orientationBlock(true, 3, 1, shortValue(true, 0))), xmp)));
    // An absent tag — a zero-entry IFD, a block without the tag, a bare package prefix — still
    // falls back to the XMP packet.
    assertEquals(
        permuted(stored, 6),
        applyDecode(spliceJpegPayloads(jpeg, exifPayload(tiffBlockWithNoEntries(true)), xmp)));
    assertEquals(
        permuted(stored, 6),
        applyDecode(spliceJpegPayloads(jpeg, exifPayload(tiffBlockWithOtherTag(true)), xmp)));
    assertEquals(permuted(stored, 6), applyDecode(spliceJpegPayloads(jpeg, EXIF_PREFIX, xmp)));
  }

  @Test
  void jpegInvalidExifHeadSuppressesTheXmpFallback() throws Exception {
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    // XMP says 3: when the tag rotates, its value must win over the packet, so the expected
    // result pins the EXIF-derived orientation, not the XMP one.
    byte[] xmp = xmpPayload(XMP_ATTR_3);
    // Invalid TIFF heads (Pillow 12.3.0, verified in its pinned venv): the reference's
    // open-time getexif() swallowed the ImageFileDirectory_v2 constructor error on each of
    // them and cached an empty exif object, so its XMP fallback never ran — orientation 0,
    // pixels as stored, despite the XMP packet's 3.
    List<byte[]> invalidHeads =
        List.of(
            // The exact fuzz case: 3 bytes left after the doubled Exif\0\0 package prefixes.
            concat(EXIF_PREFIX, concat(EXIF_PREFIX, new byte[] {'M', 'M', 0})),
            // Any other truncation under 8 bytes is equally invalid.
            exifPayload(new byte[] {'I', 'I', 0}),
            // A bad byte order (NN) with a full 8-byte head.
            exifPayload(badByteOrderBlock(true)),
            exifPayload(badByteOrderBlock(false)),
            // A valid byte order but the wrong TIFF magic (44) — pins that the rule is the
            // prefix check, not just the length.
            exifPayload(tiffBlockWithMagic(true, 44, 8)),
            exifPayload(tiffBlockWithMagic(false, 44, 8)),
            // The little-endian BigTIFF prefix II 2b 00: the reference reads an 8-byte
            // BigTIFF offset past its 8-byte head and always raises.
            exifPayload(tiffBlockWithMagic(true, 43, 8)),
            // The other two magic byte pairs the reference does not accept at all: II 00 2b
            // and MM 2b 00.
            exifPayload(tiffBlockWithMagic(true, 0x2b00, 8)),
            exifPayload(tiffBlockWithMagic(false, 0x2b00, 8)));
    for (int i = 0; i < invalidHeads.size(); i++) {
      byte[] spliced = spliceJpegPayloads(jpeg, invalidHeads.get(i), xmp);
      assertEquals(
          stored, ImageDecoder.decode(new ByteArrayInputStream(spliced)), "invalid head " + i);
      assertEquals(stored, applyDecode(spliced), "invalid head " + i);
    }
    // Parseable heads the reference's constructor accepts — the big-endian BigTIFF prefix
    // MM 00 2b and the cross-endian prefixes II 00 2a / MM 2a 00 — are read in the prefix's
    // byte order, so the Orientation 6 embedded in each of these blocks beats the XMP
    // packet's 3 (EXIF wins whenever the tag is present).
    List<byte[]> parseableHeads =
        List.of(
            exifPayload(tiffBlockWithMagic(false, 43, 8)),
            exifPayload(tiffBlockWithMagic(true, 0x2a00, 8)),
            exifPayload(tiffBlockWithMagic(false, 0x2a00, 8)));
    for (int i = 0; i < parseableHeads.size(); i++) {
      byte[] spliced = spliceJpegPayloads(jpeg, parseableHeads.get(i), xmp);
      assertEquals(permuted(stored, 6), applyDecode(spliced), "parseable head " + i);
    }
    // …and a parseable head without the tag (the plain valid-head zero-entry contrast)
    // still falls back to the XMP packet's 3.
    assertEquals(
        permuted(stored, 3),
        applyDecode(spliceJpegPayloads(jpeg, exifPayload(tiffBlockWithNoEntries(true)), xmp)));
  }

  @Test
  void crossEndianHeadsRotateInThePrefixByteOrderInBothFormats() throws Exception {
    // The three non-42 heads the reference's TIFF constructor accepts (Pillow 12.3.0,
    // verified in its pinned venv for the JPEG and the PNG paths alike): the constructor
    // takes the classic path for all of them and chooses the byte order from the II/MM
    // prefix, so the Orientation 6 embedded in each block rotates under APPLY — a 96x64
    // stored JPEG to 64x96, a 4x2 stored PNG to 2x4 — and the default policy leaves the
    // stored pixels untouched.
    List<byte[]> crossEndianBlocks =
        List.of(
            tiffBlockWithMagic(true, 0x2a00, 8), // II 00 2a, little-endian body
            tiffBlockWithMagic(false, 0x2a00, 8), // MM 2a 00, big-endian body
            tiffBlockWithMagic(false, 43, 8)); // MM 00 2b, big-endian body, BigTIFF magic
    byte[] jpeg =
        Files.readAllBytes(OracleFixtures.fixtures().resolve("decode/jpeg-color-96x64.jpg"));
    RgbImage stored = ImageDecoder.decode(new ByteArrayInputStream(jpeg));
    for (int i = 0; i < crossEndianBlocks.size(); i++) {
      byte[] spliced = spliceJpegPayloads(jpeg, exifPayload(crossEndianBlocks.get(i)));
      assertEquals(stored, ImageDecoder.decode(new ByteArrayInputStream(spliced)), "head " + i);
      RgbImage applied = applyDecode(spliced);
      assertEquals(64, applied.width(), "head " + i);
      assertEquals(96, applied.height(), "head " + i);
      assertEquals(permuted(stored, 6), applied, "head " + i);
    }
    // The same three heads in a PNG eXIf chunk (no XMP) rotate identically.
    byte[] pixels = gradientPixels(4, 2);
    RgbImage pngStored = new RgbImage(4, 2, pixels);
    for (int i = 0; i < crossEndianBlocks.size(); i++) {
      byte[] png = minimalTruecolorPng(4, 2, pixels, crossEndianBlocks.get(i));
      assertEquals(pngStored, ImageDecoder.decode(new ByteArrayInputStream(png)), "head " + i);
      RgbImage applied = applyDecode(png);
      assertEquals(2, applied.width(), "head " + i);
      assertEquals(4, applied.height(), "head " + i);
      assertEquals(permuted(pngStored, 6), applied, "head " + i);
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
  void pngExifAndXmpChunkSemanticsMatchTheReference() throws Exception {
    // The 4x2 pixel recipe of the Phase 7.3b WP2 venv cross-check (probe4/probe5): every
    // expectation below was verified against the pinned Pillow 12.3.0 venv before hardcoding.
    int width = 4;
    int height = 2;
    byte[] pixels = gradientPixels(width, height);
    RgbImage stored = new RgbImage(width, height, pixels);
    byte[] block6 = exifTiffBlock(true, 6);
    // Two eXIf chunks: the last one in file order wins.
    byte[] twoExif =
        multiChunkPng(
            width,
            height,
            pixels,
            List.of(new PngChunk("eXIf", exifTiffBlock(true, 3)), new PngChunk("eXIf", block6)),
            List.of());
    // An eXIf chunk after IDAT is still read, like the reference's load().
    assertEquals(stored, ImageDecoder.decode(new ByteArrayInputStream(twoExif)));
    assertEquals(permuted(stored, 6), applyDecode(twoExif));
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width, height, pixels, List.of(), List.of(new PngChunk("eXIf", block6)))));
    // A leading Exif\0\0 package prefix on the eXIf data is stripped before the TIFF parse.
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(new PngChunk("eXIf", concat(EXIF_PREFIX, block6))),
                List.of())));
    // A tEXt "exif" keyword carries the raw TIFF block, parseable just like an eXIf chunk.
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width, height, pixels, List.of(textChunk("tEXt", "exif", block6)), List.of())));
    // The XMP packet from a tEXt, a (decompressed) zTXt or an iTXt chunk all fire the fallback.
    PngChunk xmp = textChunk("tEXt", "XML:com.adobe.xmp", XMP_ATTR_6);
    assertEquals(
        permuted(stored, 6),
        applyDecode(multiChunkPng(width, height, pixels, List.of(xmp), List.of())));
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(textChunk("zTXt", "XML:com.adobe.xmp", XMP_ATTR_6)),
                List.of())));
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(textChunk("iTXt", "XML:com.adobe.xmp", XMP_ATTR_6)),
                List.of())));
    // A present eXIf orientation (3) wins over the XMP's 6.
    assertEquals(
        permuted(stored, 3),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(new PngChunk("eXIf", exifTiffBlock(true, 3)), xmp),
                List.of())));
    // An empty tEXt XMP value is falsy; the iTXt byte-valued slot supplies the packet.
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(
                    textChunk("tEXt", "XML:com.adobe.xmp", new byte[0]),
                    textChunk("iTXt", "XML:com.adobe.xmp", XMP_ATTR_6)),
                List.of())));
    // A non-TIFF "exif" tEXt value is treated as an absent tag here (the reference's getexif()
    // raises on it — a documented defensive divergence), so without XMP the pixels stay as stored.
    assertEquals(
        stored,
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(textChunk("tEXt", "exif", "garbage".getBytes(StandardCharsets.US_ASCII))),
                List.of())));
    // ... and with an XMP packet the fallback fires instead of raising.
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(
                    textChunk("tEXt", "exif", "garbage".getBytes(StandardCharsets.US_ASCII)), xmp),
                List.of())));
    // A post-IDAT XMP chunk is still read.
    assertEquals(
        permuted(stored, 6),
        applyDecode(multiChunkPng(width, height, pixels, List.of(), List.of(xmp))));
    // An eXIf before IDAT is overwritten by a later eXIf after IDAT.
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(new PngChunk("eXIf", exifTiffBlock(true, 3))),
                List.of(new PngChunk("eXIf", block6)))));
    // A bare Exif\0\0 prefix (eXIf data or tEXt value) leaves the tag absent: no rotation
    // without XMP, the fallback with it.
    assertEquals(
        stored,
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(new PngChunk("eXIf", EXIF_PREFIX.clone())),
                List.of())));
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(new PngChunk("eXIf", EXIF_PREFIX.clone()), xmp),
                List.of())));
    assertEquals(
        permuted(stored, 6),
        applyDecode(
            multiChunkPng(
                width,
                height,
                pixels,
                List.of(textChunk("tEXt", "exif", EXIF_PREFIX), xmp),
                List.of())));
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

  /** The 6-byte EXIF package prefix. */
  private static final byte[] EXIF_PREFIX = {'E', 'x', 'i', 'f', 0, 0};

  /** The 29-byte (NUL-terminated) Adobe XMP package prefix of a JPEG XMP {@code APP1} payload. */
  private static final byte[] XMP_PREFIX =
      "http://ns.adobe.com/xap/1.0/\u0000".getBytes(StandardCharsets.US_ASCII);

  private static final String XMP_ATTR_TEMPLATE =
      "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:Description"
          + " xmlns:tiff=\"http://ns.adobe.com/tiff/1.0/\" tiff:Orientation=\"%s\"/>"
          + "</rdf:RDF></x:xmpmeta>";

  /** The same packets the venv cross-check (probe4/probe5) was verified with. */
  private static final byte[] XMP_ATTR_6 =
      String.format(XMP_ATTR_TEMPLATE, "6").getBytes(StandardCharsets.US_ASCII);

  private static final byte[] XMP_ATTR_3 =
      String.format(XMP_ATTR_TEMPLATE, "3").getBytes(StandardCharsets.US_ASCII);
  private static final byte[] XMP_ATTR_12 =
      String.format(XMP_ATTR_TEMPLATE, "12").getBytes(StandardCharsets.US_ASCII);
  private static final byte[] XMP_ELEMENT_6 =
      ("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:Description"
           + " xmlns:tiff=\"http://ns.adobe.com/tiff/1.0/\"><tiff:Orientation>6</tiff:Orientation>"
           + "</rdf:Description></x:xmpmeta>")
          .getBytes(StandardCharsets.US_ASCII);
  private static final byte[] XMP_NO_TAG =
      ("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:Description"
              + " xmlns:tiff=\"http://ns.adobe.com/tiff/1.0/\"><tiff:Software>probe</tiff:Software>"
              + "</rdf:Description></x:xmpmeta>")
          .getBytes(StandardCharsets.US_ASCII);
  private static final byte[] XMP_FIRST_3_THEN_6 =
      concat(XMP_ATTR_3, "<second tiff:Orientation=\"6\"/>".getBytes(StandardCharsets.US_ASCII));

  /** One row of the reference's IFD Orientation type/count matrix. */
  private record TypeCase(
      String label,
      boolean present,
      int effective,
      int type,
      int count,
      Function<Boolean, byte[]> value) {}

  /**
   * The type/count matrix pinned against the reference (Pillow 12.3.0, its pinned venv, both
   * endians — probe3 and probe6 of the Phase 7.3b WP2 cross-check): {@code present} is whether the
   * tag is stored at all (unregistered types and count 0 keep it absent), and {@code effective} the
   * orientation APPLY resolves it to (2–8 when it rotates, 0 when a present value never matches the
   * transpose table).
   */
  private static final List<TypeCase> TYPE_CASES =
      List.of(
          new TypeCase("SHORT c1 6", true, 6, 3, 1, l -> shortValue(l, 6)),
          new TypeCase(
              "SHORT c2 (2,6)", true, 2, 3, 2, l -> concat(shortValue(l, 2), shortValue(l, 6))),
          new TypeCase(
              "SHORT c2 (1,6)", true, 1, 3, 2, l -> concat(shortValue(l, 1), shortValue(l, 6))),
          new TypeCase("LONG c1 6", true, 6, 4, 1, l -> longValue(l, 6)),
          new TypeCase("LONG c1 3", true, 3, 4, 1, l -> longValue(l, 3)),
          new TypeCase(
              "LONG c2 (1,6)", true, 1, 4, 2, l -> concat(longValue(l, 1), longValue(l, 6))),
          new TypeCase("SIGNED_BYTE c1 6", true, 6, 6, 1, l -> new byte[] {6}),
          new TypeCase("SIGNED_SHORT c1 6", true, 6, 8, 1, l -> shortValue(l, 6)),
          new TypeCase("SIGNED_LONG c1 6", true, 6, 9, 1, l -> longValue(l, 6)),
          new TypeCase("IFD c1 6", true, 6, 13, 1, l -> longValue(l, 6)),
          new TypeCase("LONG8 c1 6", true, 6, 16, 1, l -> long8Value(l, 6)),
          new TypeCase("RATIONAL (6,1)", true, 6, 5, 1, l -> rationalValue(l, 6, 1)),
          new TypeCase("RATIONAL (12,2)", true, 6, 5, 1, l -> rationalValue(l, 12, 2)),
          new TypeCase("RATIONAL (1,6)", true, 0, 5, 1, l -> rationalValue(l, 1, 6)),
          new TypeCase("RATIONAL (0,0)", true, 0, 5, 1, l -> rationalValue(l, 0, 0)),
          new TypeCase("SRATIONAL (6,1)", true, 6, 10, 1, l -> rationalValue(l, 6, 1)),
          new TypeCase("FLOAT 6.0", true, 6, 11, 1, l -> floatValue(l, 6.0f)),
          new TypeCase("FLOAT 2.5", true, 0, 11, 1, l -> floatValue(l, 2.5f)),
          new TypeCase("DOUBLE 6.0", true, 6, 12, 1, l -> doubleValue(l, 6.0)),
          new TypeCase("UNICODE(14) c1 6", false, 0, 14, 1, l -> longValue(l, 6)),
          new TypeCase("IFD8(17) c1 6", false, 0, 17, 1, l -> longValue(l, 6)),
          new TypeCase("SHORT c0", false, 0, 3, 0, l -> new byte[0]),
          new TypeCase("BYTE c1 6", true, 0, 1, 1, l -> new byte[] {6}),
          new TypeCase("ASCII c1 '6'", true, 0, 2, 1, l -> new byte[] {'6'}),
          new TypeCase("UNDEFINED c1 6", true, 0, 7, 1, l -> new byte[] {6}));

  private static String label(boolean little, int orientation) {
    return "orientation " + orientation + " " + (little ? "II" : "MM");
  }

  private static String caseLabel(boolean little, String label) {
    return label + " " + (little ? "II" : "MM");
  }

  /** The image APPLY should produce for an effective orientation (no rotation outside 2–8). */
  private static RgbImage effective(RgbImage stored, int orientation) {
    return orientation >= 2 && orientation <= 8 ? permuted(stored, orientation) : stored;
  }

  private static RgbImage applyDecode(byte[] bytes) throws IOException {
    return ImageDecoder.decode(
        new ByteArrayInputStream(bytes),
        ImageDecodeLimits.DEFAULTS,
        AlphaPolicy.COMPOSITE_OVER_WHITE,
        ExifOrientationPolicy.APPLY);
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
    return spliceJpegPayloads(jpeg, exifPayload(tiffBlock));
  }

  /**
   * Splices the given APP1 payloads (EXIF or XMP segments) immediately after the SOI, in the order
   * given.
   */
  private static byte[] spliceJpegPayloads(byte[] jpeg, byte[]... payloads) {
    if ((jpeg[0] & 0xFF) != 0xFF || (jpeg[1] & 0xFF) != 0xD8) {
      throw new IllegalArgumentException("not a JPEG: expected the SOI marker");
    }
    int total = 0;
    for (byte[] payload : payloads) {
      total += payload.length;
    }
    byte[] out = new byte[jpeg.length + 4 * payloads.length + total];
    out[0] = (byte) 0xFF;
    out[1] = (byte) 0xD8;
    int pos = 2;
    for (byte[] payload : payloads) {
      out[pos] = (byte) 0xFF;
      out[pos + 1] = (byte) 0xE1;
      int segmentLength = payload.length + 2; // includes the two length bytes
      out[pos + 2] = (byte) (segmentLength >>> 8);
      out[pos + 3] = (byte) segmentLength;
      System.arraycopy(payload, 0, out, pos + 4, payload.length);
      pos += 4 + payload.length;
    }
    System.arraycopy(jpeg, 2, out, pos, jpeg.length - 2);
    return out;
  }

  /** The EXIF APP1 payload: the 6-byte package prefix plus a TIFF block. */
  private static byte[] exifPayload(byte[] tiffBlock) {
    return concat(EXIF_PREFIX, tiffBlock);
  }

  /** The XMP APP1 payload: the 29-byte (NUL-terminated) Adobe package prefix plus the packet. */
  private static byte[] xmpPayload(byte[] packet) {
    return concat(XMP_PREFIX, packet);
  }

  /**
   * Splices one APP1 payload between the scan data and the trailing EOI marker — a segment the
   * reference's marker walk never sees, because it breaks out at the first SOS.
   */
  private static byte[] spliceJpegAfterScan(byte[] jpeg, byte[] payload) {
    int n = jpeg.length;
    if ((jpeg[n - 2] & 0xFF) != 0xFF || (jpeg[n - 1] & 0xFF) != 0xD9) {
      throw new IllegalArgumentException("not a JPEG: expected a trailing EOI marker");
    }
    byte[] out = new byte[n + 4 + payload.length];
    System.arraycopy(jpeg, 0, out, 0, n - 2);
    int pos = n - 2;
    out[pos] = (byte) 0xFF;
    out[pos + 1] = (byte) 0xE1;
    int segmentLength = payload.length + 2;
    out[pos + 2] = (byte) (segmentLength >>> 8);
    out[pos + 3] = (byte) segmentLength;
    System.arraycopy(payload, 0, out, pos + 4, payload.length);
    out[pos + 4 + payload.length] = (byte) 0xFF;
    out[pos + 4 + payload.length + 1] = (byte) 0xD9;
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

  /**
   * A full one-entry Orientation block with the given TIFF magic (a readable IFD at the offset).
   */
  private static byte[] tiffBlockWithMagic(boolean little, int magic, int ifdOffset) {
    byte[] header = tiffHeader(little, magic, ifdOffset);
    byte[] count = new byte[2];
    writeU16(count, 0, 1, little);
    return concat(concat(concat(header, count), ifdEntry(0x0112, 3, 1, 6, little)), new byte[4]);
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

  /** A block whose single IFD entry is truncated mid-entry (6 of its 12 bytes present). */
  private static byte[] tiffBlockWithTruncatedFirstEntry(boolean little) {
    byte[] header = tiffHeader(little, 42, 8);
    byte[] count = new byte[2];
    writeU16(count, 0, 1, little);
    byte[] halfEntry = Arrays.copyOf(ifdEntry(0x0112, 3, 1, 6, little), 6);
    return concat(concat(header, count), halfEntry);
  }

  /** A block whose IFD claims zero entries. */
  private static byte[] tiffBlockWithNoEntries(boolean little) {
    byte[] header = tiffHeader(little, 42, 8);
    byte[] count = new byte[2];
    writeU16(count, 0, 0, little);
    return concat(concat(header, count), new byte[4]);
  }

  /** An Orientation SHORT value (2 bytes, the block's endianness). */
  private static byte[] shortValue(boolean little, int v) {
    byte[] b = new byte[2];
    writeU16(b, 0, v, little);
    return b;
  }

  /** An Orientation LONG/SIGNED_LONG/IFD value (4 bytes, the block's endianness). */
  private static byte[] longValue(boolean little, long v) {
    byte[] b = new byte[4];
    writeU32(b, 0, (int) v, little);
    return b;
  }

  /**
   * An Orientation LONG8 value (8 bytes, the block's endianness; the low word first when little).
   */
  private static byte[] long8Value(boolean little, long v) {
    byte[] b = new byte[8];
    if (little) {
      writeU32(b, 0, (int) v, little);
      writeU32(b, 4, (int) (v >>> 32), little);
    } else {
      writeU32(b, 0, (int) (v >>> 32), little);
      writeU32(b, 4, (int) v, little);
    }
    return b;
  }

  /** An Orientation RATIONAL/SIGNED_RATIONAL value (an 8-byte numerator/denominator pair). */
  private static byte[] rationalValue(boolean little, long numerator, long denominator) {
    return concat(longValue(little, numerator), longValue(little, denominator));
  }

  /** An Orientation FLOAT value (4 raw IEEE-754 bytes, the block's endianness). */
  private static byte[] floatValue(boolean little, float v) {
    return longValue(little, Float.floatToRawIntBits(v));
  }

  /** An Orientation DOUBLE value (8 raw IEEE-754 bytes, the block's endianness). */
  private static byte[] doubleValue(boolean little, double v) {
    return long8Value(little, Double.doubleToRawLongBits(v));
  }

  /**
   * A TIFF block whose single Orientation entry has the given type and count and the given encoded
   * value bytes (already in the block's endianness). Values of four bytes or less sit inline in the
   * entry's value field; longer values are stored at block offset 26 (header(8) + count(2) +
   * entry(12) + next-IFD(4)) through the entry's offset field, the same layout the venv probes use.
   */
  private static byte[] orientationBlock(boolean little, int type, int count, byte[] value) {
    byte[] block = new byte[value.length > 4 ? 26 + value.length : 26];
    block[0] = (byte) (little ? 'I' : 'M');
    block[1] = (byte) (little ? 'I' : 'M');
    writeU16(block, 2, 42, little);
    writeU32(block, 4, 8, little); // first IFD at offset 8
    writeU16(block, 8, count, little);
    writeU16(block, 10, 0x0112, little); // Orientation
    writeU16(block, 12, type, little);
    writeU32(block, 14, count, little);
    if (value.length <= 4) {
      System.arraycopy(value, 0, block, 18, value.length);
    } else {
      writeU32(block, 18, 26, little);
      System.arraycopy(value, 0, block, 26, value.length);
    }
    return block;
  }

  /** A two-entry block: Orientation SHORT 3, then Orientation LONG 6 (the later one wins). */
  private static byte[] duplicateOrientationBlock(boolean little) {
    byte[] block = new byte[38];
    block[0] = (byte) (little ? 'I' : 'M');
    block[1] = (byte) (little ? 'I' : 'M');
    writeU16(block, 2, 42, little);
    writeU32(block, 4, 8, little);
    writeU16(block, 8, 2, little);
    writeU16(block, 10, 0x0112, little);
    writeU16(block, 12, 3, little); // SHORT
    writeU32(block, 14, 1, little);
    writeU16(block, 18, 3, little);
    writeU16(block, 22, 0x0112, little);
    writeU16(block, 24, 4, little); // LONG
    writeU32(block, 26, 1, little);
    writeU32(block, 30, 6, little);
    return block;
  }

  /**
   * A two-entry block: Orientation SHORT 6, then an Orientation RATIONAL whose 8-byte value offset
   * points past the end of the block (the reference's short offset read stops the walk there).
   */
  private static byte[] badOffsetAfterOrientationBlock(boolean little) {
    return badOffsetOrientationBlock(little, true);
  }

  /** The reverse of {@link #badOffsetAfterOrientationBlock}: the bad-offset entry comes first. */
  private static byte[] badOffsetBeforeOrientationBlock(boolean little) {
    return badOffsetOrientationBlock(little, false);
  }

  private static byte[] badOffsetOrientationBlock(boolean little, boolean orientationFirst) {
    byte[] block = new byte[38];
    block[0] = (byte) (little ? 'I' : 'M');
    block[1] = (byte) (little ? 'I' : 'M');
    writeU16(block, 2, 42, little);
    writeU32(block, 4, 8, little);
    writeU16(block, 8, 2, little);
    int good = orientationFirst ? 10 : 22;
    writeShortOrientation6(block, good, little);
    writeBadOffsetRational(block, good == 10 ? 22 : 10, little);
    return block;
  }

  /** An Orientation SHORT 1 6 entry at the given 12-byte entry offset. */
  private static void writeShortOrientation6(byte[] block, int entry, boolean little) {
    writeU16(block, entry, 0x0112, little);
    writeU16(block, entry + 2, 3, little); // SHORT
    writeU32(block, entry + 4, 1, little);
    writeU16(block, entry + 8, 6, little);
  }

  /** An Orientation RATIONAL entry whose 8-byte value offset points past the end of the block. */
  private static void writeBadOffsetRational(byte[] block, int entry, boolean little) {
    writeU16(block, entry, 0x0112, little);
    writeU16(block, entry + 2, 5, little); // RATIONAL
    writeU32(block, entry + 4, 1, little);
    writeU32(block, entry + 8, 0x10000000, little); // past the end of the block
  }

  /**
   * A two-entry block: a non-Orientation entry (0x8769, DOUBLE, count 1) whose 8-byte data offset
   * points past the end of the block, then an intact Orientation SHORT 6 entry. The reference reads
   * the data of every registered entry before it filters by tag, so its short read of the 0x8769
   * value aborts the walk and the Orientation entry is never stored.
   */
  private static byte[] badOffsetOnPriorEntryBlock(boolean little) {
    Tiff tiff = new Tiff(little);
    tiff.header(42, 8);
    tiff.u16(2); // two IFD entries
    tiff.u16(0x8769); // the Exif IFD pointer tag, not Orientation
    tiff.u16(12); // DOUBLE
    tiff.u32(1);
    tiff.u32(0x10000000); // past the end of the block
    tiff.u16(0x0112); // Orientation
    tiff.u16(3); // SHORT
    tiff.u32(1);
    tiff.u16(6);
    tiff.u16(0); // pad the second half of the value field
    tiff.u32(0); // no next IFD
    return tiff.bytes();
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
    List<PngChunk> preIdat = exif != null ? List.of(new PngChunk("eXIf", exif)) : List.of();
    return multiChunkPng(width, height, pixels, preIdat, List.of());
  }

  /**
   * An 8-bit truecolor PNG with arbitrary chunks spliced in before the first IDAT and between the
   * IDAT and the IEND — the reference reads eXIf and the text chunks in both positions during
   * {@code load()} — with correct CRC32s.
   */
  private static byte[] multiChunkPng(
      int width, int height, byte[] pixels, List<PngChunk> preIdat, List<PngChunk> postIdat)
      throws IOException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    out.writeBytes(new byte[] {(byte) 0x89, 'P', 'N', 'G', 0x0D, 0x0A, (byte) 0x1A, 0x0A});
    byte[] ihdr = new byte[13];
    writeU32(ihdr, 0, width, false);
    writeU32(ihdr, 4, height, false);
    ihdr[8] = 8; // bit depth
    ihdr[9] = 2; // color type: truecolor
    writeChunk(out, "IHDR", ihdr);
    for (PngChunk chunk : preIdat) {
      writeChunk(out, chunk.type(), chunk.data());
    }
    ByteArrayOutputStream raw = new ByteArrayOutputStream();
    for (int y = 0; y < height; y++) {
      raw.write(0); // filter type: none
      raw.write(pixels, y * width * 3, width * 3);
    }
    writeChunk(out, "IDAT", zlibCompress(raw.toByteArray()));
    for (PngChunk chunk : postIdat) {
      writeChunk(out, chunk.type(), chunk.data());
    }
    writeChunk(out, "IEND", new byte[0]);
    return out.toByteArray();
  }

  /** One named PNG chunk for the multi-chunk builder. */
  private record PngChunk(String type, byte[] data) {}

  /** A text chunk ({@code tEXt}/{@code zTXt}/{@code iTXt}) with the given keyword and value. */
  private static PngChunk textChunk(String kind, String keyword, byte[] value) {
    byte[] k = keyword.getBytes(StandardCharsets.US_ASCII);
    return switch (kind) {
      case "tEXt" -> new PngChunk(kind, concat(k, concat(new byte[] {0}, value)));
      case "zTXt" -> new PngChunk(kind, concat(k, concat(new byte[] {0, 0}, zlibCompress(value))));
      // cf=0 (uncompressed), cm=0, an empty language tag and translated keyword
      default -> new PngChunk(kind, concat(k, concat(new byte[] {0, 0, 0, 0, 0}, value)));
    };
  }

  /** A complete zlib (RFC 1950) stream over {@code data}. */
  private static byte[] zlibCompress(byte[] data) {
    Deflater deflater = new Deflater();
    deflater.setInput(data);
    deflater.finish();
    byte[] compressed = new byte[data.length * 2 + 64];
    int n = deflater.deflate(compressed);
    deflater.end();
    return Arrays.copyOf(compressed, n);
  }

  /** The 4x2 pixel recipe of the venv cross-check, sized for any width and height. */
  private static byte[] gradientPixels(int width, int height) {
    byte[] pixels = new byte[width * height * 3];
    int p = 0;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        pixels[p++] = (byte) (x * 60 % 256);
        pixels[p++] = (byte) (y * 120 % 256);
        pixels[p++] = (byte) ((255 - x * 20) % 256);
      }
    }
    return pixels;
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
