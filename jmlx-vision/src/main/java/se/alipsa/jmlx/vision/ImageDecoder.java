package se.alipsa.jmlx.vision;

import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.MultiPixelPackedSampleModel;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Iterator;
import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.spi.IIORegistry;
import javax.imageio.spi.ImageReaderSpi;
import javax.imageio.stream.MemoryCacheImageInputStream;

/**
 * Decodes PNG and JPEG input to an {@link RgbImage} using the JDK's built-in {@code javax.imageio}
 * PNG/JPEG readers.
 *
 * <p><b>Supported:</b> 8-bit RGB, grayscale, RGBA and grayscale-with-alpha PNG, 1-, 2-, 4- and
 * 8-bit palette PNG (the built-in reader expands packed palette indices to 8 bits, and Pillow's own
 * default for a small palette is 4-bit) with optional {@code tRNS}, and 8-bit RGB and grayscale
 * JPEG, including scans with restart markers ({@code RST0}–{@code RST7}). A {@code tRNS} chunk on a
 * truecolor or grayscale PNG is ignored — Pillow keeps those images in mode RGB/L and the
 * reference's RGB conversion drops the transparency — and is stripped from the bytes before the
 * built-in reader would turn it into an alpha band. Decoded samples are extracted directly from the
 * {@link Raster}'s byte buffer without per-pixel {@code ColorModel}/{@code getRGB} conversion; the
 * expected band count is validated against a byte-level pre-scan of the file header, and the
 * band-to-channel mapping is probed once through the decoded {@link ColorModel} — the built-in
 * readers produce {@code B,G,R} and {@code A,B,G,R} raster layouts, never {@code R,G,B}, so the
 * band order is never assumed.
 *
 * <p><b>Rejected before conversion, with format-specific {@link IOException}s:</b> 16-bit PNG,
 * sub-8-bit and 16-bit grayscale/truecolor PNG, CMYK/YCCK JPEG (four-component frames and Adobe
 * {@code APP14} transform codes other than RGB/unspecified), inputs above the configured {@link
 * ImageDecodeLimits}, and non-PNG/non-JPEG or corrupt input.
 *
 * <p><b>Alpha policy:</b> alpha (RGBA, LA and palette {@code tRNS}) is composited over a white
 * background using Pillow's exact integer compositing formula (ported from the pinned Pillow 12.3.0
 * {@code src/libImaging/AlphaComposite.c}), matching the reference processor's {@code
 * convert_to_rgb} step. Opaque input passes through unchanged.
 *
 * <p><b>EXIF orientation is ignored:</b> an EXIF-rotated JPEG decodes to its stored, unrotated
 * pixels, matching the pinned reference path ({@code PIL.Image.open} → {@code load()} with no
 * {@code exif_transpose}). Callers that need the rotated view — in particular for phone photos —
 * must rotate the image themselves before decoding. No orientation accessor is exposed in this
 * milestone.
 *
 * <p><b>ICC and gAMA color management are ignored:</b> embedded color profiles and gamma chunks are
 * not applied to samples, matching the reference path, which performs no conversion. Callers doing
 * color management must do it themselves.
 *
 * <p>The no-options overloads use {@code ImageDecodeLimits(16_777_216L, 16_384)}: a conservative
 * default that rejects typical 24 MP and 48 MP phone photos. Use the explicit limits overloads to
 * accept such inputs when your memory budget allows. Model-specific aggregate pixel/tile/token
 * budgets are separate.
 *
 * <p>Reader providers are selected through {@link IIORegistry} with a java.desktop-module check; a
 * missing built-in provider is rejected rather than falling back to installed plugins. Global
 * ImageIO cache settings are never changed.
 */
public final class ImageDecoder {
  private static final ImageDecodeLimits DEFAULT_LIMITS =
      new ImageDecodeLimits(16_777_216L, 16_384);

  // Safety valve against unbounded buffering: the dimension limits above already bound the
  // decoded pixel buffer to ~50 MB, and any input within them buffers far below this cap.
  private static final long MAX_INPUT_BYTES = 512L * 1024 * 1024;

  private ImageDecoder() {}

  /**
   * Decodes the PNG or JPEG file at {@code path} using the default limits ({@code
   * ImageDecodeLimits(16_777_216L, 16_384)}, which reject typical 24 MP/48 MP phone photos).
   *
   * @param path existing PNG or JPEG file
   * @return the decoded image, alpha composited over white
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds the
   *     limits
   */
  public static RgbImage decode(Path path) throws IOException {
    return decode(path, DEFAULT_LIMITS);
  }

  /**
   * Decodes the PNG or JPEG content of {@code input} using the default limits. The stream is
   * consumed in full; it is not closed by this method.
   *
   * @param input non-null stream with the whole image
   * @return the decoded image, alpha composited over white
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds the
   *     limits
   */
  public static RgbImage decode(InputStream input) throws IOException {
    return decode(input, DEFAULT_LIMITS);
  }

  /**
   * Decodes the PNG or JPEG file at {@code path} with explicit limits permitting larger inputs.
   *
   * @param path existing PNG or JPEG file
   * @param limits positive decode bounds checked from header metadata before any pixel allocation
   * @return the decoded image, alpha composited over white
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds {@code
   *     limits}
   */
  public static RgbImage decode(Path path, ImageDecodeLimits limits) throws IOException {
    if (path == null) {
      throw new IllegalArgumentException("path must not be null");
    }
    if (limits == null) {
      throw new IllegalArgumentException("limits must not be null");
    }
    byte[] bytes;
    try (InputStream in = Files.newInputStream(path)) {
      bytes = readAllBounded(in);
    }
    return decode(bytes, limits);
  }

  /**
   * Decodes the PNG or JPEG content of {@code input} with explicit limits. The stream is consumed
   * in full; it is not closed by this method (internal wrappers are closed).
   *
   * @param input non-null stream with the whole image
   * @param limits positive decode bounds checked from header metadata before any pixel allocation
   * @return the decoded image, alpha composited over white
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds {@code
   *     limits}
   */
  public static RgbImage decode(InputStream input, ImageDecodeLimits limits) throws IOException {
    if (input == null) {
      throw new IllegalArgumentException("input must not be null");
    }
    if (limits == null) {
      throw new IllegalArgumentException("limits must not be null");
    }
    byte[] bytes = readAllBounded(input);
    return decode(bytes, limits);
  }

  private static RgbImage decode(byte[] bytes, ImageDecodeLimits limits) throws IOException {
    String format;
    int width;
    int height;
    DecodePlan plan;
    if (isPng(bytes)) {
      PngInfo info = parsePngHeader(bytes);
      format = "png";
      width = info.width;
      height = info.height;
      plan = new DecodePlan(info);
      if (info.trns != null && (info.colorType == 0 || info.colorType == 2)) {
        // The reference keeps truecolor/grayscale PNGs in mode RGB/L and its RGB conversion
        // drops the tRNS, so the chunk is ignored; the built-in reader would turn it into an
        // alpha band instead, so strip it before decode.
        int end = info.trnsOffset + 12 + info.trns.length;
        bytes =
            concat(
                Arrays.copyOfRange(bytes, 0, info.trnsOffset),
                Arrays.copyOfRange(bytes, end, bytes.length));
      }
    } else if (isJpeg(bytes)) {
      JpegInfo info = parseJpegHeader(bytes);
      format = "jpeg";
      width = info.width;
      height = info.height;
      plan = new DecodePlan(info);
    } else {
      throw new IOException(
          "unsupported image input: expected a PNG or JPEG file, found " + describeMagic(bytes));
    }
    checkLimits(width, height, limits);

    MemoryCacheImageInputStream stream =
        new MemoryCacheImageInputStream(new ByteArrayInputStream(bytes));
    try {
      ImageReader reader = newReader(format);
      try {
        reader.setInput(stream);
        BufferedImage bi = reader.read(0, null);
        return toRgbImage(bi, plan);
      } finally {
        reader.dispose();
      }
    } finally {
      stream.close();
    }
  }

  private static void checkLimits(int width, int height, ImageDecodeLimits limits)
      throws IOException {
    long pixels = (long) width * height;
    if (width > limits.maxDimension() || height > limits.maxDimension()) {
      throw new IOException(
          "image dimension "
              + width
              + "x"
              + height
              + " exceeds the maximum dimension "
              + limits.maxDimension());
    }
    if (pixels > limits.maxPixels()) {
      throw new IOException(
          "image size "
              + width
              + "x"
              + height
              + " ("
              + pixels
              + " pixels) exceeds the maximum of "
              + limits.maxPixels()
              + " pixels");
    }
  }

  private static ImageReader newReader(String formatName) throws IOException {
    Iterator<ImageReaderSpi> it =
        IIORegistry.getDefaultInstance().getServiceProviders(ImageReaderSpi.class, false);
    while (it.hasNext()) {
      ImageReaderSpi spi = it.next();
      if (spi == null) {
        continue;
      }
      boolean matches = false;
      for (String name : spi.getFormatNames()) {
        if (name.regionMatches(true, 0, formatName, 0, formatName.length())) {
          matches = true;
          break;
        }
      }
      if (!matches) {
        continue;
      }
      // Only the JDK's built-in readers (java.desktop) are acceptable; installed plugins are
      // never a fallback, so decode behavior cannot drift with the local plugin set.
      if (spi.getClass().getModule() != ImageIO.class.getModule()) {
        continue;
      }
      ImageReader reader = spi.createReaderInstance();
      if (reader != null) {
        return reader;
      }
    }
    throw new IOException("no built-in " + formatName + " image reader is available in this JDK");
  }

  private static byte[] readAllBounded(InputStream in) throws IOException {
    byte[] chunk = new byte[8192];
    byte[] out = new byte[64 * 1024];
    int len = 0;
    long total = 0;
    int r;
    while ((r = in.read(chunk)) != -1) {
      total += r;
      if (total > MAX_INPUT_BYTES) {
        throw new IOException(
            "image input exceeds the " + MAX_INPUT_BYTES + "-byte decode buffer limit");
      }
      if (len + r > out.length) {
        int newLength = (int) Math.min(MAX_INPUT_BYTES, Math.max((long) out.length * 2, total));
        out = Arrays.copyOf(out, newLength);
      }
      System.arraycopy(chunk, 0, out, len, r);
      len += r;
    }
    return len == out.length ? out : Arrays.copyOf(out, len);
  }

  private static byte[] concat(byte[] a, byte[] b) {
    byte[] out = new byte[a.length + b.length];
    System.arraycopy(a, 0, out, 0, a.length);
    System.arraycopy(b, 0, out, a.length, b.length);
    return out;
  }

  // ----------------------------------------------------------------------------- header scan

  private static boolean isPng(byte[] b) {
    return b.length >= 8
        && (b[0] & 0xFF) == 0x89
        && b[1] == 'P'
        && b[2] == 'N'
        && b[3] == 'G'
        && b[4] == 0x0D
        && b[5] == 0x0A
        && b[6] == 0x1A
        && b[7] == 0x0A;
  }

  private static boolean isJpeg(byte[] b) {
    return b.length >= 2 && (b[0] & 0xFF) == 0xFF && (b[1] & 0xFF) == 0xD8;
  }

  private static String describeMagic(byte[] b) {
    if (b.length < 4) {
      return "input of " + b.length + " byte(s)";
    }
    StringBuilder sb = new StringBuilder("input starting with ");
    for (int i = 0; i < 4; i++) {
      if (i > 0) {
        sb.append(' ');
      }
      sb.append(String.format("0x%02x", b[i] & 0xFF));
    }
    return sb.toString();
  }

  /** PNG metadata extracted from the header before any pixel decoding. */
  private static final class PngInfo {
    int width;
    int height;
    int bitDepth;
    int colorType;
    byte[] palette; // RGB triples, paletteCount * 3 bytes
    int paletteCount;
    byte[] trns; // raw tRNS bytes, or null
    int trnsOffset = -1; // offset of the tRNS chunk's length field, or -1 when absent
  }

  private static PngInfo parsePngHeader(byte[] b) throws IOException {
    if (b.length < 33) {
      throw new IOException("corrupt PNG: truncated signature or IHDR chunk");
    }
    // First chunk must be IHDR: length(4) type(4) data(13) crc(4) = signature(8) + 33 - 8.
    if (u32(b, 8) != 13 || u32(b, 12) != 0x49484452) { // "IHDR"
      throw new IOException("corrupt PNG: first chunk is not IHDR");
    }
    PngInfo info = new PngInfo();
    info.width = (int) u32(b, 16);
    info.height = (int) u32(b, 20);
    info.bitDepth = b[24] & 0xFF;
    info.colorType = b[25] & 0xFF;
    if (info.width <= 0 || info.height <= 0) {
      throw new IOException(
          "corrupt PNG: non-positive dimensions " + info.width + "x" + info.height);
    }
    switch (info.colorType) {
      case 0, 2, 4, 6 -> {
        if (info.bitDepth != 8) {
          String kind =
              switch (info.colorType) {
                case 0 -> "grayscale";
                case 2 -> "truecolor";
                case 4 -> "grayscale-with-alpha";
                default -> "RGBA";
              };
          if (info.bitDepth == 16) {
            throw new IOException("16-bit PNG is not supported");
          }
          throw new IOException(
              info.bitDepth + "-bit " + kind + " PNG is not supported (only 8-bit)");
        }
      }
      case 3 -> {
        // Palette PNGs may be 1-, 2-, 4- or 8-bit (Pillow's own default for a small palette is
        // 4-bit); the built-in reader expands packed indices to 8 bits, so the band count is 1
        // either way and no other depth is valid for this color type.
        if (info.bitDepth != 1 && info.bitDepth != 2 && info.bitDepth != 4 && info.bitDepth != 8) {
          throw new IOException(
              info.bitDepth + "-bit palette PNG is not supported (only 1-, 2-, 4- and 8-bit)");
        }
      }
      default ->
          throw new IOException(
              "unsupported PNG color type " + info.colorType + " is not supported");
    }
    if (info.colorType == 3) {
      info.paletteCount = 0;
    }
    // Walk chunks (PLTE and tRNS must precede IDAT).
    int pos = 8;
    boolean haveIdat = false;
    while (pos + 12 <= b.length && !haveIdat) {
      long length = u32(b, pos);
      if (pos + 8 + length + 4 > b.length) {
        throw new IOException("corrupt PNG: truncated chunk at offset " + pos);
      }
      long cid = u32(b, pos + 4);
      if (cid == 0x504C5445) { // "PLTE"
        if (length < 3 || length % 3 != 0) {
          throw new IOException("corrupt PNG: malformed PLTE chunk");
        }
        info.palette = Arrays.copyOfRange(b, pos + 8, pos + 8 + (int) length);
        info.paletteCount = (int) (length / 3);
      } else if (cid == 0x74524E53) { // "tRNS"
        info.trns = Arrays.copyOfRange(b, pos + 8, pos + 8 + (int) length);
        info.trnsOffset = pos;
      } else if (cid == 0x49444154) { // "IDAT"
        haveIdat = true;
      }
      pos += 12 + (int) length;
    }
    if (info.colorType == 3 && info.palette == null) {
      throw new IOException("corrupt PNG: palette image has no PLTE chunk");
    }
    return info;
  }

  /** JPEG metadata extracted by walking the full marker structure. */
  private static final class JpegInfo {
    int width;
    int height;
    int components;
    int adobeTransform = -1;
  }

  /**
   * Walks the complete marker structure from SOI to EOI. The JDK's built-in JPEG decoder is lenient
   * about truncated scan data (it silently produces an image padded with wrong pixels), so
   * truncation can only be detected by requiring the well-formed EOI the decoder never checks for.
   * Every {@code FF FF} pair outside the scan is a fill byte — the optional DRI (restart interval)
   * segment is a plain {@code FF DD} length-prefixed segment, handled by the general marker path
   * below. The scan itself is walked as raw bytes: an {@code FF} followed by a byte that is neither
   * another {@code FF} (fill), {@code 00} (treated leniently as data), nor one of the {@code
   * RST0}–{@code RST7} restart markers (valid inside a scan written with a restart interval)
   * terminates it.
   */
  private static JpegInfo parseJpegHeader(byte[] b) throws IOException {
    JpegInfo info = new JpegInfo();
    int pos = 2;
    int n = b.length;
    boolean eoiSeen = false;
    while (!eoiSeen) {
      while (pos + 1 < n && (b[pos] & 0xFF) == 0xFF && (b[pos + 1] & 0xFF) == 0xFF) {
        pos++; // fill byte
      }
      if (pos + 1 >= n) {
        throw new IOException("corrupt JPEG: truncated marker at offset " + pos);
      }
      if ((b[pos] & 0xFF) != 0xFF) {
        throw new IOException("corrupt JPEG: expected a marker at offset " + pos);
      }
      int marker = b[pos + 1] & 0xFF;
      if (marker == 0xFF
          || (marker >= 0xD0 && marker <= 0xD7)
          || marker == 0x01
          || marker == 0xD8) {
        pos += 2; // standalone marker
        continue;
      }
      if (marker == 0xD9) {
        eoiSeen = true;
        break;
      }
      if (pos + 4 > n) {
        throw new IOException("corrupt JPEG: truncated marker at offset " + pos);
      }
      int length = u16(b, pos + 2);
      if (length < 2 || pos + 2 + length > n) {
        throw new IOException("corrupt JPEG: bad segment length " + length + " at offset " + pos);
      }
      if (marker == 0xDA) { // SOS: the compressed scan follows; find its terminating marker
        if (length < 6) {
          throw new IOException("corrupt JPEG: bad SOS segment length " + length);
        }
        pos += 2 + length;
        boolean terminated = false;
        while (pos < n) {
          if ((b[pos] & 0xFF) != 0xFF) {
            pos++;
            continue;
          }
          if (pos + 1 >= n) {
            throw new IOException("corrupt JPEG: truncated scan data");
          }
          int m = b[pos + 1] & 0xFF;
          if (m == 0xFF || m == 0x00 || (m >= 0xD0 && m <= 0xD7)) {
            pos += 2; // fill byte, a stuffed 00, or a restart marker inside the scan
            continue;
          }
          // A real marker: leave pos on it for the outer loop to classify.
          terminated = true;
          break;
        }
        if (!terminated) {
          throw new IOException("corrupt JPEG: truncated scan data");
        }
        continue;
      }
      if (marker == 0xEE
          && length >= 14) { // APP14: "Adobe" + version(2) + flags0(2) + flags1(2) + transform(1)
        if ((b[pos + 4] & 0xFF) == 0x41
            && (b[pos + 5] & 0xFF) == 0x64
            && (b[pos + 6] & 0xFF) == 0x6F
            && (b[pos + 7] & 0xFF) == 0x62
            && (b[pos + 8] & 0xFF) == 0x65) { // "Adobe"
          info.adobeTransform = b[pos + 15] & 0xFF;
        }
      } else if ((marker >= 0xC0 && marker <= 0xC3)
          || (marker >= 0xC5 && marker <= 0xC7)
          || (marker >= 0xC9 && marker <= 0xCB)
          || (marker >= 0xCD && marker <= 0xCF)) {
        if (length < 7) {
          throw new IOException("corrupt JPEG: truncated SOF segment");
        }
        int precision = b[pos + 4] & 0xFF;
        if (precision != 8) {
          throw new IOException(
              "JPEG with " + precision + "-bit sample precision is not supported (only 8-bit)");
        }
        info.height = u16(b, pos + 5);
        info.width = u16(b, pos + 7);
        info.components = b[pos + 9] & 0xFF;
        if (info.width <= 0 || info.height <= 0) {
          throw new IOException("corrupt JPEG: non-positive dimensions");
        }
      }
      pos += 2 + length;
    }
    if (info.width == 0) {
      throw new IOException("corrupt JPEG: no frame header (SOF) found");
    }
    if (info.components == 4) {
      throw new IOException("CMYK/YCCK JPEG is not supported (four-component frame)");
    }
    if (info.adobeTransform != -1 && info.adobeTransform != 0 && info.adobeTransform != 1) {
      throw new IOException(
          "CMYK/YCCK JPEG is not supported (Adobe APP14 transform code "
              + info.adobeTransform
              + ")");
    }
    return info;
  }

  private static long u32(byte[] b, int off) {
    return ((b[off] & 0xFFL) << 24)
        | ((b[off + 1] & 0xFFL) << 16)
        | ((b[off + 2] & 0xFFL) << 8)
        | (b[off + 3] & 0xFFL);
  }

  private static int u16(byte[] b, int off) {
    return ((b[off] & 0xFF) << 8) | (b[off + 1] & 0xFF);
  }

  // --------------------------------------------------------------- raster to RgbImage

  /** Everything the raster extraction needs to know about the decoded image. */
  private static final class DecodePlan {
    final int expectedBands;
    final byte[] palette;
    final int paletteCount;
    final int[] paletteAlpha;

    DecodePlan(Object header) {
      if (header instanceof PngInfo png) {
        switch (png.colorType) {
          case 0 -> expectedBands = 1;
          case 2 -> expectedBands = 3;
          case 3 -> expectedBands = 1;
          case 4 -> expectedBands = 2;
          default -> expectedBands = 4;
        }
        palette = png.palette;
        paletteCount = png.paletteCount;
        paletteAlpha = paletteAlpha(png.paletteCount, png.trns);
      } else {
        JpegInfo jpeg = (JpegInfo) header;
        expectedBands = jpeg.components == 1 ? 1 : 3;
        palette = null;
        paletteCount = 0;
        paletteAlpha = null;
      }
    }

    // PIL semantics for palette + tRNS (PngImagePlugin.chunk_tRNS): a tRNS body of 0xFFs with
    // exactly one 0x00 marks a single transparent palette index (an index past the palette is out
    // of range and leaves every entry opaque); otherwise each byte is the alpha of the palette
    // entry at that index. An oversized chunk (more samples than palette entries, technically
    // malformed per RFC 2083) is benign for PIL and libpng: the samples past the palette are
    // ignored, never an error.
    private static int[] paletteAlpha(int count, byte[] trns) {
      if (count == 0 || trns == null) {
        return null;
      }
      int[] alpha = new int[count];
      Arrays.fill(alpha, 255);
      if (isSimpleTrns(trns)) {
        int idx = trnsIndex(trns);
        if (idx < count) {
          alpha[idx] = 0;
        }
      } else {
        for (int i = 0; i < Math.min(trns.length, count); i++) {
          alpha[i] = trns[i] & 0xFF;
        }
      }
      return alpha;
    }

    private static boolean isSimpleTrns(byte[] t) {
      int zeroPos = -1;
      for (int i = 0; i < t.length; i++) {
        if ((t[i] & 0xFF) == 0x00) {
          if (zeroPos >= 0) {
            return false; // more than one transparent entry
          }
          zeroPos = i;
        } else if ((t[i] & 0xFF) != 0xFF) {
          return false; // partial alpha value: general per-entry table
        }
      }
      return zeroPos >= 0;
    }

    private static int trnsIndex(byte[] t) {
      for (int i = 0; i < t.length; i++) {
        if ((t[i] & 0xFF) == 0x00) {
          return i;
        }
      }
      return -1;
    }
  }

  private static RgbImage toRgbImage(BufferedImage bi, DecodePlan plan) throws IOException {
    int w = bi.getWidth();
    int h = bi.getHeight();
    Raster raster = bi.getRaster();
    int bands = raster.getNumBands();
    if (bands != plan.expectedBands) {
      throw new IOException(
          "unexpected decoded raster: " + bands + " bands, expected " + plan.expectedBands);
    }
    DataBuffer db = raster.getDataBuffer();
    if (!(db instanceof DataBufferByte)) {
      throw new IOException(
          "unexpected raster sample precision for an 8-bit image (data buffer "
              + db.getClass().getSimpleName()
              + ")");
    }
    long n = (long) w * h;
    // The per-band byte offsets come from the sample model, which reflects the readers' storage
    // order (B,G,R for 3-band, A,B,G,R for 4-band) — the raw buffer order, not channel order.
    // The one non-interleaved layout the readers produce is the sub-8-bit palette: several
    // indices packed per byte in a MultiPixelPackedSampleModel (Pillow's own default for a
    // small palette is 4-bit); the palette branch below handles that model directly.
    SampleModel sampleModel = raster.getSampleModel();
    int[] bandOffsets = null;
    int pixelStride = 0;
    int scanlineStride = 0;
    if (plan.palette == null) {
      if (!(sampleModel instanceof PixelInterleavedSampleModel interleaved)) {
        throw new IOException(
            "unexpected raster sample model: " + sampleModel.getClass().getSimpleName());
      }
      bandOffsets = interleaved.getBandOffsets();
      pixelStride = interleaved.getPixelStride();
      scanlineStride = interleaved.getScanlineStride();
      validateBandOffsets(bands, pixelStride, bandOffsets);
    }
    int bankOffset = db.getOffset();

    // The band-to-channel mapping is probed from the color model, never assumed: the built-in
    // readers store B,G,R (3-band) and A,B,G,R (4-band) bytes, and the 2-band gray+alpha layout
    // may order its bands either way.
    int redBand = 0;
    int greenBand = 0;
    int blueBand = 0;
    int alphaBand = -1;
    if (plan.palette == null && bands > 1) {
      ColorModel cm = bi.getColorModel();
      if (!(cm instanceof ComponentColorModel)) {
        throw new IOException(
            "unexpected color model for an "
                + bands
                + "-band image: "
                + cm.getClass().getSimpleName());
      }
      int[] mapping = probeBandMapping(cm, bands, scanlineStride, pixelStride, bandOffsets);
      redBand = mapping[0];
      greenBand = mapping[1];
      blueBand = mapping[2];
      alphaBand = mapping[3];
    }

    if (n > Integer.MAX_VALUE / 3) {
      throw new IOException("image of " + n + " pixels exceeds the maximum decodable buffer size");
    }
    int pixels = (int) n;
    byte[] out = new byte[pixels * 3];
    byte[] data = ((DataBufferByte) db).getData();
    if (plan.expectedBands == 1 && plan.palette == null) {
      // Grayscale: replicate the single band.
      int p = 0;
      for (int y = 0; y < h; y++) {
        int rowBase = bankOffset + y * scanlineStride;
        for (int x = 0; x < w; x++) {
          int l = data[rowBase + x * pixelStride + bandOffsets[0]] & 0xFF;
          out[p] = (byte) l;
          out[p + 1] = (byte) l;
          out[p + 2] = (byte) l;
          p += 3;
        }
      }
    } else if (plan.expectedBands == 1) {
      // Palette: expand indices, composite tRNS entries over white. Sub-8-bit palettes arrive
      // packed (several indices per byte); 8-bit palettes are interleaved, one byte per index.
      int p = 0;
      if (sampleModel instanceof MultiPixelPackedSampleModel) {
        // Several indices are packed per byte, MSB-first per the PNG spec; the raster's own
        // sample path (the same one BufferedImage.getRGB uses) handles the packing, so the
        // raw byte buffer and the sample model's bit-offset getters are not used directly.
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            int idx = raster.getSample(x, y, 0);
            p = compositePaletteIndex(idx, plan, out, p);
          }
        }
      } else if (sampleModel instanceof PixelInterleavedSampleModel interleaved) {
        int[] offsets = interleaved.getBandOffsets();
        int pixStride = interleaved.getPixelStride();
        int lineStride = interleaved.getScanlineStride();
        validateBandOffsets(bands, pixStride, offsets);
        for (int y = 0; y < h; y++) {
          int rowBase = bankOffset + y * lineStride;
          for (int x = 0; x < w; x++) {
            int idx = data[rowBase + x * pixStride + offsets[0]] & 0xFF;
            p = compositePaletteIndex(idx, plan, out, p);
          }
        }
      } else {
        throw new IOException(
            "unexpected raster sample model: " + sampleModel.getClass().getSimpleName());
      }
    } else if (plan.expectedBands == 2) {
      // Grayscale + alpha: composite L over white, replicate. redBand carries the gray band
      // (red == green == blue, validated by the probe).
      int p = 0;
      for (int y = 0; y < h; y++) {
        int rowBase = bankOffset + y * scanlineStride;
        for (int x = 0; x < w; x++) {
          int base = rowBase + x * pixelStride;
          int l = data[base + bandOffsets[redBand]] & 0xFF;
          int a = data[base + bandOffsets[alphaBand]] & 0xFF;
          int c = compositeOverWhite(l, a);
          out[p] = (byte) c;
          out[p + 1] = (byte) c;
          out[p + 2] = (byte) c;
          p += 3;
        }
      }
    } else if (plan.expectedBands == 3) {
      int p = 0;
      for (int y = 0; y < h; y++) {
        int rowBase = bankOffset + y * scanlineStride;
        for (int x = 0; x < w; x++) {
          int base = rowBase + x * pixelStride;
          out[p] = (byte) (data[base + bandOffsets[redBand]] & 0xFF);
          out[p + 1] = (byte) (data[base + bandOffsets[greenBand]] & 0xFF);
          out[p + 2] = (byte) (data[base + bandOffsets[blueBand]] & 0xFF);
          p += 3;
        }
      }
    } else {
      // RGBA: composite over white.
      int p = 0;
      for (int y = 0; y < h; y++) {
        int rowBase = bankOffset + y * scanlineStride;
        for (int x = 0; x < w; x++) {
          int base = rowBase + x * pixelStride;
          int r = data[base + bandOffsets[redBand]] & 0xFF;
          int g = data[base + bandOffsets[greenBand]] & 0xFF;
          int b = data[base + bandOffsets[blueBand]] & 0xFF;
          int a = data[base + bandOffsets[alphaBand]] & 0xFF;
          out[p] = (byte) compositeOverWhite(r, a);
          out[p + 1] = (byte) compositeOverWhite(g, a);
          out[p + 2] = (byte) compositeOverWhite(b, a);
          p += 3;
        }
      }
    }
    return new RgbImage(w, h, out);
  }

  /** Rejects band offsets that fall outside the pixel stride or collide with each other. */
  private static void validateBandOffsets(int bands, int pixelStride, int[] bandOffsets)
      throws IOException {
    for (int band = 0; band < bands; band++) {
      int offset = bandOffsets[band];
      boolean bad = offset < 0 || offset >= pixelStride;
      for (int other = 0; other < band && !bad; other++) {
        bad = bandOffsets[other] == offset;
      }
      if (bad) {
        throw new IOException(
            "unexpected raster band offsets "
                + Arrays.toString(bandOffsets)
                + " for pixel stride "
                + pixelStride);
      }
    }
  }

  /**
   * Probes which raster band drives each output channel. For each band, a 1x1 interleaved raster
   * carrying the real {@code bandOffsets}/{@code pixelStride} with that band set to 255 and every
   * other band 0 is read back with {@code getDataElements} and converted through the color model's
   * {@code getRGB(byte[])}, the exact path {@code BufferedImage.getRGB(x, y)} uses for real pixels
   * ({@code cm.getRGB(raster.getDataElements(x, y, null))}) — {@code getRGB}'s array index is the
   * <em>component</em> index (R, G, B, A), so a probe built from the real sample model is what
   * carries the band permutation into the result. The channels that respond identify the band.
   * Returns {@code [redBand, greenBand, blueBand, alphaBand]}; {@code alphaBand} is -1 when the
   * model has no alpha. Rejects a mapping in which any channel is driven by zero or more than one
   * band, a 2-band model whose color channels come from different bands or whose alpha is not the
   * other band, or a 3/4-band model whose channels are not distinct (or whose alpha band collides
   * with a color band).
   */
  private static int[] probeBandMapping(
      ColorModel cm, int bands, int scanlineStride, int pixelStride, int[] bandOffsets)
      throws IOException {
    int redBand = -1;
    int greenBand = -1;
    int blueBand = -1;
    int alphaBand = -1;
    int redCount = 0;
    int greenCount = 0;
    int blueCount = 0;
    int alphaCount = 0;
    for (int band = 0; band < bands; band++) {
      byte[] probeData = new byte[pixelStride];
      WritableRaster probe =
          Raster.createInterleavedRaster(
              new DataBufferByte(probeData, pixelStride),
              1,
              1,
              scanlineStride,
              pixelStride,
              bandOffsets,
              null);
      probe.setSample(0, 0, band, 255);
      byte[] components = (byte[]) probe.getDataElements(0, 0, null);
      int argb = cm.getRGB(components);
      if (((argb >>> 16) & 0xFF) > 0) {
        redCount++;
        redBand = band;
      }
      if (((argb >>> 8) & 0xFF) > 0) {
        greenCount++;
        greenBand = band;
      }
      if ((argb & 0xFF) > 0) {
        blueCount++;
        blueBand = band;
      }
      if (cm.hasAlpha() && ((argb >>> 24) & 0xFF) > 0) {
        alphaCount++;
        alphaBand = band;
      }
    }
    boolean ok = redCount == 1 && greenCount == 1 && blueCount == 1;
    if (cm.hasAlpha()) {
      ok = ok && alphaCount == 1;
    }
    if (bands == 2) {
      ok = ok && redBand == greenBand && greenBand == blueBand && alphaBand != redBand;
    } else {
      ok =
          ok
              && redBand != greenBand
              && redBand != blueBand
              && greenBand != blueBand
              && (alphaBand == -1
                  || (alphaBand != redBand && alphaBand != greenBand && alphaBand != blueBand));
    }
    if (!ok) {
      throw new IOException(
          "unexpected raster band mapping: red="
              + redBand
              + " green="
              + greenBand
              + " blue="
              + blueBand
              + " alpha="
              + alphaBand);
    }
    return new int[] {redBand, greenBand, blueBand, alphaBand};
  }

  /**
   * Expands one palette index to three output bytes, compositing the entry's {@code tRNS} alpha
   * over white. Writes at {@code out[p..p+2]} and returns the next free output position.
   */
  private static int compositePaletteIndex(int idx, DecodePlan plan, byte[] out, int p)
      throws IOException {
    if (idx >= plan.paletteCount) {
      throw new IOException(
          "corrupt PNG: palette index "
              + idx
              + " is out of range (palette has "
              + plan.paletteCount
              + " entries)");
    }
    int base = idx * 3;
    int a = plan.paletteAlpha == null ? 255 : plan.paletteAlpha[idx];
    out[p] = (byte) compositeOverWhite(plan.palette[base] & 0xFF, a);
    out[p + 1] = (byte) compositeOverWhite(plan.palette[base + 1] & 0xFF, a);
    out[p + 2] = (byte) compositeOverWhite(plan.palette[base + 2] & 0xFF, a);
    return p + 3;
  }

  // Pillow's exact integer alpha compositing over an opaque white background, ported from
  // src/libImaging/AlphaComposite.c, pinned at:
  //   https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/AlphaComposite.c
  //   SHA-256 0063c136b8bb2be51e2fb76fac222a7ae96b438f3c09706834862d9c1ac62c93
  // (blend/outa255/coef1/coef2 in 32-bit integer math, with the SHIFTFORDIV255(x) =
  // (((x >> 8) + x) >> 8) macro from the same tree's src/libImaging/ImagingUtils.h). For
  // a == 255 the formula reduces to the identity (verified for all 256 channel values), so
  // the opaque case is a copy.
  private static int compositeOverWhite(int c, int a) {
    if (a == 0) {
      return 255;
    }
    if (a == 255) {
      return c;
    }
    long coef1 = ((long) a * 255 * 255 * 128) / 65025;
    long coef2 = 32640L - coef1;
    int tmp = (int) (c * coef1 + 255L * coef2 + 16384L);
    return (((tmp >> 8) + tmp) >> 8) >> 7;
  }
}
