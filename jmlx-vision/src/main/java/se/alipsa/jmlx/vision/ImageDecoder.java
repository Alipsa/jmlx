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
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.zip.DataFormatException;
import java.util.zip.Inflater;
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
 * JPEG, including scans with restart markers ({@code RST0}–{@code RST7}). A {@code tRNS} chunk is
 * stripped from the bytes for every color type before the built-in reader would turn it into an
 * alpha band or a transparent palette entry — the pinned processor never applies it (see <b>Alpha
 * policy</b>). A {@code PLTE} chunk on a truecolor or RGBA image is only a <em>suggested</em>
 * palette (RFC 2083 s11.2) and is ignored, as it is by the built-in reader and by the reference
 * pipeline: only palette images decode through the palette. Decoded samples are extracted directly
 * from the {@link Raster}'s byte buffer without per-pixel {@code ColorModel}/{@code getRGB}
 * conversion; the expected band count is validated against a byte-level pre-scan of the file
 * header, and the band-to-channel mapping is probed once through the decoded {@link ColorModel} —
 * the built-in readers produce {@code B,G,R} and {@code A,B,G,R} raster layouts, never {@code
 * R,G,B}, so the band order is never assumed.
 *
 * <p><b>Rejected before conversion, with format-specific {@link IOException}s:</b> 16-bit PNG,
 * sub-8-bit and 16-bit grayscale/truecolor PNG, CMYK/YCCK JPEG (four-component frames and Adobe
 * {@code APP14} transform codes other than RGB/unspecified), inputs above the configured {@link
 * ImageDecodeLimits}, and non-PNG/non-JPEG or corrupt input.
 *
 * <p><b>Alpha policy:</b> by default alpha (RGBA and LA) is composited over a white background
 * using Pillow's exact integer compositing formula (ported from the pinned Pillow 12.3.0 {@code
 * src/libImaging/AlphaComposite.c}). The overloads that take an {@link AlphaPolicy} select instead
 * between that compositing ({@link AlphaPolicy#COMPOSITE_OVER_WHITE}) and dropping the alpha band
 * while keeping the source color channels ({@link AlphaPolicy#DROP_ALPHA}, the behavior of Pillow's
 * own {@code convert("RGB")} — the reference Qwen3-VL image path drops alpha rather than
 * compositing; see the Phase 7.3b WP1 reference findings). Palette {@code tRNS} is <em>not</em>
 * applied under either policy: the pinned processor rebuilds a palette image from {@code
 * getpalette()} (RGB only) before the RGB conversion, so its transparency never survives and
 * transparent entries decode to their opaque palette color — the same outcome as a truecolor or
 * grayscale {@code tRNS}, which the RGB conversion drops. Opaque input passes through unchanged
 * under either policy.
 *
 * <p><b>EXIF orientation:</b> by default — every overload that does not take an explicit {@link
 * ExifOrientationPolicy} — an EXIF-rotated JPEG decodes to its stored, unrotated pixels. That
 * matches the pinned SmolVLM/Idefics3 image-oracle reference path ({@code PIL.Image.open} → {@code
 * load()} with no {@code exif_transpose}), and the committed SmolVLM decode fixtures stay pinned
 * under it. The overloads that take an {@link ExifOrientationPolicy} select instead between that
 * ignoring ({@link ExifOrientationPolicy#IGNORE}) and applying the stored orientation ({@link
 * ExifOrientationPolicy#APPLY}, the behavior of Pillow's {@code ImageOps.exif_transpose} followed
 * by {@code convert("RGB")} — the reference Qwen3-VL/Qwen2VL slow-processor image path; see {@code
 * req/plans/phase7-3b-reference-findings.md} §4.4, golden case {@code exif_orientation6}).
 *
 * <p>Under {@link ExifOrientationPolicy#APPLY} the effective orientation is resolved exactly the
 * way the pinned reference's {@code getexif()} resolves it. The IFD0 {@code Orientation} tag
 * (0x0112) wins whenever it is <em>present</em>: in a JPEG from the first {@code APP1} segment
 * whose payload starts with {@code Exif\x00\x00} (later EXIF {@code APP1}s are concatenated to its
 * bytes, but the first segment's IFD0 dominates the parse, and only {@code APP1}s before the first
 * scan are considered); in a PNG from the last {@code eXIf} chunk or text-chunk {@code exif}
 * keyword in file order, including chunks after {@code IDAT}. A later same-tag entry overwrites an
 * earlier one; entries stored before a truncated IFD or a short offset read are kept; only the
 * reference's registered TIFF types (BYTE, ASCII, SHORT, LONG, RATIONAL, SIGNED_BYTE, UNDEFINED,
 * SIGNED_SHORT, SIGNED_LONG, FLOAT, DOUBLE, IFD, LONG8) with a non-zero count make the tag present
 * — unregistered types and count 0 keep it absent. A present tag rotates only when its first value
 * is an integer in 2–8, or an integer-valued float or rational (a 0/0 rational is nan): BYTE, ASCII
 * and UNDEFINED values, non-integral values and out-of-range values never rotate, but they still
 * suppress the fallback. Only an absent tag consults the XMP fallback: the first single-digit match
 * of {@code tiff:Orientation(="|>)} in the XMP packet — in a JPEG the last {@code APP1} before the
 * scan whose payload starts with the 29-byte (NUL-terminated) Adobe XMP package prefix; in a PNG
 * the last text-chunk keyword {@code XML:com.adobe.xmp} ({@code tEXt} or decompressed {@code zTXt}
 * value first, the {@code iTXt} value otherwise) — supplies the orientation, or none.
 *
 * <p>Like the reference, a malformed EXIF or XMP block is data, not corruption: it is treated as
 * absent and never breaks the decode (the header walkers still reject structural corruption of the
 * container itself). Where the reference's {@code getexif()} would instead raise on pathological
 * metadata — a bad TIFF byte order or head prefix, a truncated post-{@code IDAT} chunk, an {@code
 * iTXt} XMP value stored as an object its regex cannot scan — the decode here treats the
 * orientation as absent, and a {@code zTXt} or valid-UTF-8 {@code iTXt} {@code exif} value, which
 * the reference stores as a string it cannot load, is parsed as the underlying bytes (and can
 * rotate): a defensive leniency on input the reference rejects outright.
 *
 * <p><b>ICC and gAMA color management are ignored:</b> embedded color profiles and gamma chunks are
 * not applied to samples, matching the reference path, which performs no conversion. Callers doing
 * color management must do it themselves.
 *
 * <p>The no-limits overloads ({@code decode(Path)}, {@code decode(InputStream)}, {@code
 * decode(Path, AlphaPolicy)}, {@code decode(InputStream, AlphaPolicy)}, {@code decode(Path,
 * AlphaPolicy, ExifOrientationPolicy)} and {@code decode(InputStream, AlphaPolicy,
 * ExifOrientationPolicy)}) use {@link ImageDecodeLimits#DEFAULTS} ({@code
 * ImageDecodeLimits(16_777_216L, 16_384)}): a conservative default that rejects typical 24 MP and
 * 48 MP phone photos. Use the explicit limits overloads to accept such inputs when your memory
 * budget allows. Model-specific aggregate pixel/tile/token budgets are separate.
 *
 * <p>Reader providers are selected through {@link IIORegistry} with a java.desktop-module check; a
 * missing built-in provider is rejected rather than falling back to installed plugins. Global
 * ImageIO cache settings are never changed.
 */
public final class ImageDecoder {
  private ImageDecoder() {}

  /** The IFD0 Orientation tag. */
  private static final int ORIENTATION_TAG = 0x0112;

  /** The EXIF package prefix, stripped (repeatedly) before the TIFF block is parsed. */
  private static final byte[] EXIF_PACKAGE_PREFIX = {'E', 'x', 'i', 'f', 0, 0};

  /** The 29-byte (NUL-terminated) Adobe XMP package prefix of a JPEG XMP {@code APP1} segment. */
  private static final byte[] XMP_PACKAGE_PREFIX =
      "http://ns.adobe.com/xap/1.0/\u0000".getBytes(StandardCharsets.US_ASCII);

  /** The reference's XMP fallback search tag. */
  private static final byte[] XMP_ORIENTATION_TAG =
      "tiff:Orientation".getBytes(StandardCharsets.US_ASCII);

  /** The PNG text-chunk keyword whose value carries EXIF bytes. */
  private static final byte[] EXIF_TEXT_KEYWORD = "exif".getBytes(StandardCharsets.US_ASCII);

  /** The PNG text-chunk keyword whose value carries the XMP packet. */
  private static final byte[] XMP_TEXT_KEYWORD =
      "XML:com.adobe.xmp".getBytes(StandardCharsets.US_ASCII);

  /**
   * Decodes the PNG or JPEG file at {@code path} using {@link ImageDecodeLimits#DEFAULTS} ({@code
   * ImageDecodeLimits(16_777_216L, 16_384)}, which reject typical 24 MP/48 MP phone photos).
   *
   * @param path existing PNG or JPEG file
   * @return the decoded image, alpha composited over white
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds the
   *     default limits
   */
  public static RgbImage decode(Path path) throws IOException {
    return decode(path, ImageDecodeLimits.DEFAULTS);
  }

  /**
   * Decodes the PNG or JPEG file at {@code path} using {@link ImageDecodeLimits#DEFAULTS} and an
   * explicit {@link AlphaPolicy} for RGBA / grayscale-with-alpha input. To accept inputs beyond the
   * default limits, use {@link #decode(Path, ImageDecodeLimits, AlphaPolicy)} instead.
   *
   * @param path existing PNG or JPEG file
   * @param policy non-null policy for the alpha band; palette {@code tRNS} stripping is unaffected
   *     by the choice
   * @return the decoded image, with alpha handled per {@code policy}
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds the
   *     default limits
   */
  public static RgbImage decode(Path path, AlphaPolicy policy) throws IOException {
    return decode(path, ImageDecodeLimits.DEFAULTS, policy);
  }

  /**
   * Decodes the PNG or JPEG file at {@code path} with explicit limits and an explicit {@link
   * AlphaPolicy} for RGBA / grayscale-with-alpha input. EXIF orientation is ignored. {@link
   * #decode(Path, AlphaPolicy)} is the same call with {@link ImageDecodeLimits#DEFAULTS}
   * substituted for {@code limits}.
   *
   * @param path existing PNG or JPEG file
   * @param limits positive decode bounds checked from header metadata before any pixel allocation
   * @param policy non-null policy for the alpha band; palette {@code tRNS} stripping is unaffected
   *     by the choice
   * @return the decoded image, with alpha handled per {@code policy}
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds {@code
   *     limits}
   */
  public static RgbImage decode(Path path, ImageDecodeLimits limits, AlphaPolicy policy)
      throws IOException {
    return decode(path, limits, policy, ExifOrientationPolicy.IGNORE);
  }

  /**
   * Decodes the PNG or JPEG file at {@code path} with explicit limits, an explicit {@link
   * AlphaPolicy} for RGBA / grayscale-with-alpha input, and an explicit {@link
   * ExifOrientationPolicy} for EXIF-rotated input. {@link #decode(Path, AlphaPolicy,
   * ExifOrientationPolicy)} is the same call with {@link ImageDecodeLimits#DEFAULTS} substituted
   * for {@code limits}.
   *
   * @param path existing PNG or JPEG file
   * @param limits positive decode bounds checked from header metadata before any pixel allocation
   * @param policy non-null policy for the alpha band; palette {@code tRNS} stripping is unaffected
   *     by the choice
   * @param orientation non-null policy for the EXIF orientation tag; {@link
   *     ExifOrientationPolicy#IGNORE} returns the stored, unrotated pixels
   * @return the decoded image, with alpha and orientation handled per the policies
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds {@code
   *     limits}
   */
  public static RgbImage decode(
      Path path, ImageDecodeLimits limits, AlphaPolicy policy, ExifOrientationPolicy orientation)
      throws IOException {
    if (policy == null) {
      throw new IllegalArgumentException("policy must not be null");
    }
    if (path == null) {
      throw new IllegalArgumentException("path must not be null");
    }
    if (limits == null) {
      throw new IllegalArgumentException("limits must not be null");
    }
    if (orientation == null) {
      throw new IllegalArgumentException("orientation must not be null");
    }
    long cap = maxInputBytes(limits);
    if (Files.size(path) > cap) {
      throw new IOException("image input exceeds the " + cap + "-byte decode buffer limit");
    }
    byte[] bytes;
    try (InputStream in = Files.newInputStream(path)) {
      bytes = readAllBounded(in, cap);
    }
    return decode(bytes, limits, policy, orientation);
  }

  /**
   * Decodes the PNG or JPEG file at {@code path} using {@link ImageDecodeLimits#DEFAULTS} with an
   * explicit {@link AlphaPolicy} for RGBA / grayscale-with-alpha input and an explicit {@link
   * ExifOrientationPolicy} for EXIF-rotated input. To accept inputs beyond the default limits, use
   * {@link #decode(Path, ImageDecodeLimits, AlphaPolicy, ExifOrientationPolicy)} instead.
   *
   * @param path existing PNG or JPEG file
   * @param policy non-null policy for the alpha band; palette {@code tRNS} stripping is unaffected
   *     by the choice
   * @param orientation non-null policy for the EXIF orientation tag; {@link
   *     ExifOrientationPolicy#IGNORE} returns the stored, unrotated pixels
   * @return the decoded image, with alpha and orientation handled per the policies
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds the
   *     default limits
   */
  public static RgbImage decode(Path path, AlphaPolicy policy, ExifOrientationPolicy orientation)
      throws IOException {
    return decode(path, ImageDecodeLimits.DEFAULTS, policy, orientation);
  }

  /**
   * Decodes the PNG or JPEG content of {@code input} using {@link ImageDecodeLimits#DEFAULTS}. The
   * stream is consumed in full; it is not closed by this method.
   *
   * @param input non-null stream with the whole image
   * @return the decoded image, alpha composited over white
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds the
   *     default limits
   */
  public static RgbImage decode(InputStream input) throws IOException {
    return decode(input, ImageDecodeLimits.DEFAULTS);
  }

  /**
   * Decodes the PNG or JPEG content of {@code input} using {@link ImageDecodeLimits#DEFAULTS} and
   * an explicit {@link AlphaPolicy} for RGBA / grayscale-with-alpha input. The stream is consumed
   * in full; it is not closed by this method (internal wrappers are closed). To accept inputs
   * beyond the default limits, use {@link #decode(InputStream, ImageDecodeLimits, AlphaPolicy)}
   * instead.
   *
   * @param input non-null stream with the whole image
   * @param policy non-null policy for the alpha band; palette {@code tRNS} stripping is unaffected
   *     by the choice
   * @return the decoded image, with alpha handled per {@code policy}
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds the
   *     default limits
   */
  public static RgbImage decode(InputStream input, AlphaPolicy policy) throws IOException {
    return decode(input, ImageDecodeLimits.DEFAULTS, policy);
  }

  /**
   * Decodes the PNG or JPEG content of {@code input} with explicit limits and an explicit {@link
   * AlphaPolicy}. EXIF orientation is ignored. The stream is consumed in full; it is not closed by
   * this method (internal wrappers are closed). {@link #decode(InputStream, AlphaPolicy)} is the
   * same call with {@link ImageDecodeLimits#DEFAULTS} substituted for {@code limits}.
   *
   * @param input non-null stream with the whole image
   * @param limits positive decode bounds checked from header metadata before any pixel allocation
   * @param policy non-null policy for the alpha band; palette {@code tRNS} stripping is unaffected
   *     by the choice
   * @return the decoded image, with alpha handled per {@code policy}
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds {@code
   *     limits}
   */
  public static RgbImage decode(InputStream input, ImageDecodeLimits limits, AlphaPolicy policy)
      throws IOException {
    return decode(input, limits, policy, ExifOrientationPolicy.IGNORE);
  }

  /**
   * Decodes the PNG or JPEG content of {@code input} with explicit limits, an explicit {@link
   * AlphaPolicy} for RGBA / grayscale-with-alpha input, and an explicit {@link
   * ExifOrientationPolicy} for EXIF-rotated input. The stream is consumed in full; it is not closed
   * by this method (internal wrappers are closed). {@link #decode(InputStream, AlphaPolicy,
   * ExifOrientationPolicy)} is the same call with {@link ImageDecodeLimits#DEFAULTS} substituted
   * for {@code limits}.
   *
   * @param input non-null stream with the whole image
   * @param limits positive decode bounds checked from header metadata before any pixel allocation
   * @param policy non-null policy for the alpha band; palette {@code tRNS} stripping is unaffected
   *     by the choice
   * @param orientation non-null policy for the EXIF orientation tag; {@link
   *     ExifOrientationPolicy#IGNORE} returns the stored, unrotated pixels
   * @return the decoded image, with alpha and orientation handled per the policies
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds {@code
   *     limits}
   */
  public static RgbImage decode(
      InputStream input,
      ImageDecodeLimits limits,
      AlphaPolicy policy,
      ExifOrientationPolicy orientation)
      throws IOException {
    if (policy == null) {
      throw new IllegalArgumentException("policy must not be null");
    }
    if (input == null) {
      throw new IllegalArgumentException("input must not be null");
    }
    if (limits == null) {
      throw new IllegalArgumentException("limits must not be null");
    }
    if (orientation == null) {
      throw new IllegalArgumentException("orientation must not be null");
    }
    byte[] bytes = readAllBounded(input, maxInputBytes(limits));
    return decode(bytes, limits, policy, orientation);
  }

  /**
   * Decodes the PNG or JPEG content of {@code input} using {@link ImageDecodeLimits#DEFAULTS} with
   * an explicit {@link AlphaPolicy} for RGBA / grayscale-with-alpha input and an explicit {@link
   * ExifOrientationPolicy} for EXIF-rotated input. The stream is consumed in full; it is not closed
   * by this method (internal wrappers are closed). To accept inputs beyond the default limits, use
   * {@link #decode(InputStream, ImageDecodeLimits, AlphaPolicy, ExifOrientationPolicy)} instead.
   *
   * @param input non-null stream with the whole image
   * @param policy non-null policy for the alpha band; palette {@code tRNS} stripping is unaffected
   *     by the choice
   * @param orientation non-null policy for the EXIF orientation tag; {@link
   *     ExifOrientationPolicy#IGNORE} returns the stored, unrotated pixels
   * @return the decoded image, with alpha and orientation handled per the policies
   * @throws IOException if the format is unsupported, the input is corrupt, or it exceeds the
   *     default limits
   */
  public static RgbImage decode(
      InputStream input, AlphaPolicy policy, ExifOrientationPolicy orientation) throws IOException {
    return decode(input, ImageDecodeLimits.DEFAULTS, policy, orientation);
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
    return decode(path, limits, AlphaPolicy.COMPOSITE_OVER_WHITE);
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
    return decode(input, limits, AlphaPolicy.COMPOSITE_OVER_WHITE);
  }

  private static RgbImage decode(
      byte[] bytes, ImageDecodeLimits limits, AlphaPolicy policy, ExifOrientationPolicy orientation)
      throws IOException {
    String format;
    int width;
    int height;
    DecodePlan plan;
    int orientationTag = 0;
    if (isPng(bytes)) {
      PngInfo info = parsePngHeader(bytes);
      format = "png";
      width = info.width;
      height = info.height;
      plan = new DecodePlan(info);
      orientationTag = resolveOrientation(info.exif, info.xmpText, info.xmpBytes);
      if (info.trns != null) {
        // The pinned processor never applies tRNS: truecolor/grayscale PNGs stay in mode RGB/L
        // and the RGB conversion drops the chunk, and a palette image is rebuilt from
        // getpalette() (RGB only), so its tRNS never survives. The built-in reader would turn
        // the chunk into an alpha band (RGB/L) or a transparent palette entry (P), so strip it
        // before decode for every color type. Malformed input may carry more than one tRNS
        // chunk, and every one of them would take effect in the built-in reader, so every one
        // is stripped — in a single pass, copying the input exactly once.
        bytes = stripPngChunks(bytes, info.trns);
      }
    } else if (isJpeg(bytes)) {
      JpegInfo info = parseJpegHeader(bytes);
      format = "jpeg";
      width = info.width;
      height = info.height;
      plan = new DecodePlan(info);
      if (info.exif.xmpSuppressed()) {
        // The reference's open-time getexif() swallowed the EXIF parse error (SyntaxError /
        // struct.error from ImageFileDirectory_v2 on an invalid TIFF head) and cached an empty
        // object, so its XMP fallback never ran: no rotation regardless of any XMP packet. This
        // matches the reference only when the JFIF header carries no DPI (the case where
        // _read_dpi_from_exif calls getexif() at open); with JFIF DPI the reference raises instead,
        // which we tolerate.
        orientationTag = 0;
      } else {
        orientationTag = resolveOrientation(info.exif, null, info.xmp);
      }
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
        RgbImage image = toRgbImage(bi, plan, policy);
        // The transform is a pure pixel permutation and commutes with the color/alpha mapping
        // (a pixel is a pixel), so permuting the final RGB buffer is exactly the reference
        // order: transpose in the original mode, then convert to RGB.
        return applyOrientation(image, orientation, orientationTag);
      } finally {
        reader.dispose();
      }
    } finally {
      stream.close();
    }
  }

  /**
   * Permutes the decoded pixels to the stored EXIF orientation when {@code policy} is {@link
   * ExifOrientationPolicy#APPLY} and {@code orientationTag} is one of the values 2–8; returns the
   * image untouched — no allocation — otherwise. The transform is Pillow's {@code
   * ImageOps.exif_transpose} table (see {@link ExifOrientationPolicy#APPLY}), applied to the final
   * RGB buffer with {@code x} = column, {@code y} = row, and a {@code src} pair reading (column,
   * row): 2: {@code src[W-1-x, y]}; 3: {@code src[W-1-x, H-1-y]}; 4: {@code src[x, H-1-y]}; 5:
   * {@code src[y, x]}; 6: {@code src[y, H-1-x]}; 7: {@code src[W-1-y, H-1-x]}; 8: {@code src[W-1-y,
   * x]}; the result dimensions swap for 5–8.
   */
  private static RgbImage applyOrientation(
      RgbImage image, ExifOrientationPolicy policy, int orientationTag) {
    if (policy != ExifOrientationPolicy.APPLY || orientationTag < 2 || orientationTag > 8) {
      return image;
    }
    int w = image.width();
    int h = image.height();
    boolean dimsSwap = orientationTag >= 5;
    int w2 = dimsSwap ? h : w;
    int h2 = dimsSwap ? w : h;
    byte[] src = image.pixelsRaw();
    byte[] dst = new byte[src.length];
    for (int y = 0; y < h2; y++) {
      for (int x = 0; x < w2; x++) {
        int sx;
        int sy;
        switch (orientationTag) {
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
    // dst was allocated and filled here and is never shared, so it is adopted without the
    // public constructor's defensive copy.
    return RgbImage.ofUnchecked(w2, h2, dst);
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

  /**
   * The cap on buffered input bytes for the given limits. An image within the limits carries at
   * most about four bytes of raw 8-bit sample data per pixel (four bands plus one PNG filter byte
   * per scanline; JPEG entropy data is comparable) plus per-segment overhead, so eight times the
   * pixel budget with a fixed margin is a generous ceiling for any valid input within the limits
   * while still bounding the transient buffer of an oversized untrusted input. The computation
   * saturates instead of overflowing: a huge {@code maxPixels} — including {@code Long.MAX_VALUE},
   * a natural way to say "no limit" — would otherwise wrap the 8x product to a small or negative
   * cap that rejects valid input; it clamps to the largest addressable {@code byte[]} instead.
   */
  private static long maxInputBytes(ImageDecodeLimits limits) {
    long maxPixels = limits.maxPixels();
    long unsaturated =
        maxPixels > (Long.MAX_VALUE - (1L << 20)) / 8
            ? Long.MAX_VALUE
            : 8L * maxPixels + (1L << 20);
    return Math.min(unsaturated, (long) Integer.MAX_VALUE - 8);
  }

  /**
   * Reads the whole stream, refusing to buffer more than {@code cap} bytes. The PNG/JPEG magic is
   * checked after accumulating up to eight bytes — {@link InputStream#read(byte[], int, int)} may
   * return fewer bytes than are available (network, {@code SequenceInputStream} and decompressing
   * streams commonly do), so one short read checked against a zero-filled buffer would reject valid
   * input — and the magic is compared only against the bytes actually read, so input in another
   * format is rejected after a few bytes instead of being buffered to the cap first.
   */
  private static byte[] readAllBounded(InputStream in, long cap) throws IOException {
    byte[] chunk = new byte[8192];
    int first = in.readNBytes(chunk, 0, 8);
    if (first < 2) {
      throw new IOException(
          "unsupported image input: expected a PNG or JPEG file, found input of "
              + first
              + " byte(s)");
    }
    byte[] magic = Arrays.copyOf(chunk, first);
    if (!isPng(magic) && !isJpeg(magic)) {
      throw new IOException(
          "unsupported image input: expected a PNG or JPEG file, found " + describeMagic(magic));
    }
    byte[] out = new byte[(int) Math.min(cap, 64L * 1024)];
    int len = 0;
    long total = 0;
    int r = first;
    while (r != -1) {
      total += r;
      if (total > cap) {
        throw new IOException("image input exceeds the " + cap + "-byte decode buffer limit");
      }
      if (len + r > out.length) {
        int newLength = (int) Math.min(cap, Math.max((long) out.length * 2, total));
        out = Arrays.copyOf(out, newLength);
      }
      System.arraycopy(chunk, 0, out, len, r);
      len += r;
      r = in.read(chunk);
    }
    return len == out.length ? out : Arrays.copyOf(out, len);
  }

  /**
   * Removes the chunks at {@code {offset of length field, data length}} (in file order) from the
   * PNG bytes in a single pass: one output allocation and two {@code System.arraycopy} calls per
   * kept gap, no intermediate copies.
   */
  private static byte[] stripPngChunks(byte[] bytes, List<int[]> chunks) {
    int stripped = 0;
    for (int[] chunk : chunks) {
      stripped += 12 + chunk[1];
    }
    byte[] out = new byte[bytes.length - stripped];
    int src = 0;
    int dst = 0;
    for (int[] chunk : chunks) {
      int start = chunk[0];
      int end = start + 12 + chunk[1];
      System.arraycopy(bytes, src, out, dst, start - src);
      dst += start - src;
      src = end;
    }
    System.arraycopy(bytes, src, out, dst, bytes.length - src);
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
    // {offset of the length field, data length} for every pre-IDAT tRNS chunk in file order, or
    // null when the file carries none. Malformed input may carry more than one; the built-in
    // reader would honor each of them, so all must be stripped.
    List<int[]> trns;
    // Raw value of the last "exif" entry in file order — an eXIf chunk's data (the TIFF block
    // directly, no prefix) or a tEXt/zTXt/iTXt "exif" keyword's value — or null when the file
    // carries none. Leading "Exif\x00\x00" prefixes are stripped when it is parsed.
    byte[] exifPayload;
    // Value of the last "XML:com.adobe.xmp" text-chunk entry (tEXt or decompressed zTXt value),
    // or null; an empty value is falsy, like the reference's str-valued slot.
    byte[] xmpText;
    // (Decompressed) value of the last iTXt "XML:com.adobe.xmp" entry — the reference's
    // byte-valued slot — or null.
    byte[] xmpBytes;
    // The orientation resolved from exifPayload, filled in after the walk.
    TiffOrientation exif = TiffOrientation.ABSENT;
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
    // Walk chunks: PLTE and tRNS must precede IDAT; eXIf and the text chunks may legally follow
    // it (the reference reads them during load()), so the walk continues past the first IDAT to
    // IEND. A truncation before IDAT is container corruption and is rejected; one after IDAT
    // never breaks the metadata walk — the available bytes are processed, then the walk stops.
    int pos = 8;
    boolean haveIdat = false;
    while (pos + 12 <= b.length) {
      long length = u32(b, pos);
      boolean complete = pos + 8 + length + 4 <= b.length;
      if (!complete && !haveIdat) {
        throw new IOException("corrupt PNG: truncated chunk at offset " + pos);
      }
      long cid = u32(b, pos + 4);
      pngChunk(b, pos, complete ? (int) length : b.length - pos - 8, !haveIdat, info);
      if (cid == 0x49454E44) { // "IEND"
        break;
      }
      if (cid == 0x49444154) { // "IDAT"
        haveIdat = true;
      }
      if (!complete) {
        break;
      }
      pos += 12 + (int) length;
    }
    if (info.colorType == 3 && info.palette == null) {
      throw new IOException("corrupt PNG: palette image has no PLTE chunk");
    }
    if (info.exifPayload != null) {
      info.exif = parseExifPayload(info.exifPayload);
    }
    return info;
  }

  /**
   * Records the orientation-relevant metadata of one PNG chunk at {@code pos} with {@code length}
   * data bytes (the full declared length, or the truncated remainder after IDAT). Later chunks of
   * the same kind overwrite earlier ones, like the reference's {@code im_info} dictionary.
   */
  private static void pngChunk(byte[] b, int pos, int length, boolean preIdat, PngInfo info)
      throws IOException {
    long cid = u32(b, pos + 4);
    int data = pos + 8;
    if (cid == 0x504C5445) { // "PLTE"
      if (!preIdat) {
        return;
      }
      if (length < 3 || length % 3 != 0) {
        throw new IOException("corrupt PNG: malformed PLTE chunk");
      }
      info.palette = Arrays.copyOfRange(b, data, data + length);
      info.paletteCount = length / 3;
    } else if (cid == 0x74524E53) { // "tRNS"
      if (!preIdat) {
        return;
      }
      if (info.trns == null) {
        info.trns = new ArrayList<>();
      }
      info.trns.add(new int[] {pos, length});
    } else if (cid == 0x65584966) { // "eXIf"
      // The chunk data is the TIFF block directly (no "Exif" prefix); the last one in file
      // order wins, like the reference.
      info.exifPayload = Arrays.copyOfRange(b, data, data + length);
    } else if (cid == 0x74455874) { // "tEXt"
      pngTextChunk(b, data, length, info);
    } else if (cid == 0x7A545874) { // "zTXt"
      pngZtxtChunk(b, data, length, info);
    } else if (cid == 0x69545874) { // "iTXt"
      pngItxtChunk(b, data, length, info);
    }
  }

  /**
   * A {@code tEXt} chunk's data: {@code keyword \0 value}. The reference stores the raw value bytes
   * in its {@code exif} entry for the exact keyword {@code exif} (parseable EXIF!) and the value in
   * its str-valued XMP entry for {@code XML:com.adobe.xmp}; any other keyword is irrelevant to
   * orientation. A missing NUL keeps the whole data as the keyword (the reference's broken-tEXt
   * fallback) and leaves the value empty.
   */
  private static void pngTextChunk(byte[] b, int data, int length, PngInfo info) {
    int end = data + length;
    int nul = indexOfByte(b, data, end);
    int keyEnd = nul >= 0 ? nul : end;
    int valueStart = nul >= 0 ? nul + 1 : end;
    if (keyEnd == data) {
      return;
    }
    if (equalsRegion(b, data, keyEnd - data, EXIF_TEXT_KEYWORD)) {
      info.exifPayload = Arrays.copyOfRange(b, valueStart, end);
    } else if (equalsRegion(b, data, keyEnd - data, XMP_TEXT_KEYWORD)) {
      info.xmpText = Arrays.copyOfRange(b, valueStart, end);
    }
  }

  /**
   * A {@code zTXt} chunk's data: {@code keyword \0 comp_method zlib(value)}. Only comp method 0 is
   * known — the reference raises on any other, so it is simply not captured here. A corrupt or
   * incomplete zlib payload yields the reference's empty value, which is stored (the slot is set,
   * to a falsy value), like the reference's {@code v = b""} on {@code zlib.error}.
   */
  private static void pngZtxtChunk(byte[] b, int data, int length, PngInfo info) {
    int end = data + length;
    int nul = indexOfByte(b, data, end);
    int keyEnd = nul >= 0 ? nul : end;
    int restStart = nul >= 0 ? nul + 1 : end;
    if (keyEnd == data) {
      return;
    }
    if (restStart < end && (b[restStart] & 0xFF) != 0) {
      return;
    }
    byte[] value = inflateZlib(Arrays.copyOfRange(b, restStart + 1, end));
    if (value == null) {
      value = new byte[0];
    }
    if (equalsRegion(b, data, keyEnd - data, EXIF_TEXT_KEYWORD)) {
      info.exifPayload = value;
    } else if (equalsRegion(b, data, keyEnd - data, XMP_TEXT_KEYWORD)) {
      info.xmpText = value;
    }
  }

  /**
   * An {@code iTXt} chunk's data: {@code keyword \0 cf(1) cm(1) lang \0 translated \0 value}. The
   * reference stores the (decompressed) value in its byte-valued {@code xmp} entry for the exact
   * keyword {@code XML:com.adobe.xmp} — before the strict UTF-8 check — and, only when the keyword,
   * language, translation and value all decode as strict UTF-8, in its str-valued entry; the exact
   * keyword {@code exif} ends up string-valued there, which makes the reference's {@code getexif()}
   * raise — here it is parsed as bytes instead (and is almost always absent). A compressed value
   * ({@code cf} 1) must use comp method 0, and any inflate failure leaves the slots untouched, like
   * the reference's early returns.
   */
  private static void pngItxtChunk(byte[] b, int data, int length, PngInfo info) {
    int end = data + length;
    int nul1 = indexOfByte(b, data, end);
    if (nul1 < 0) {
      return;
    }
    int p = nul1 + 1;
    if (p + 2 > end) {
      return;
    }
    int cf = b[p] & 0xFF;
    int cm = b[p + 1] & 0xFF;
    int langStart = p + 2;
    int langEnd = indexOfByte(b, langStart, end);
    if (langEnd < 0) {
      return;
    }
    int tkEnd = indexOfByte(b, langEnd + 1, end);
    if (tkEnd < 0) {
      return;
    }
    int valueStart = tkEnd + 1;
    byte[] value = Arrays.copyOfRange(b, valueStart, end);
    if (cf != 0) {
      if (cm != 0) {
        return;
      }
      byte[] inflated = inflateZlib(value);
      if (inflated == null) {
        return;
      }
      value = inflated;
    }
    if (equalsRegion(b, data, nul1 - data, XMP_TEXT_KEYWORD)) {
      info.xmpBytes = value;
    }
    if (validUtf8(b, data, nul1)
        && validUtf8(b, langStart, langEnd)
        && validUtf8(b, langEnd + 1, tkEnd)
        && validUtf8(value, 0, value.length)) {
      if (equalsRegion(b, data, nul1 - data, EXIF_TEXT_KEYWORD)) {
        info.exifPayload = value;
      } else if (equalsRegion(b, data, nul1 - data, XMP_TEXT_KEYWORD)) {
        info.xmpText = value;
      }
    }
  }

  /** JPEG metadata extracted by walking the full marker structure. */
  private static final class JpegInfo {
    int width;
    int height;
    int components;
    int adobeTransform = -1;
    // The reference's info["exif"] value: the first EXIF-prefixed APP1 segment's payload (prefix
    // included) with every later EXIF-prefixed segment's payload (minus its prefix) concatenated,
    // or null when the file carries none before the first scan.
    ByteArrayOutputStream exifBuffer;
    // True once the first EXIF-prefixed APP1 segment has been seen.
    boolean exifSeen;
    // Payload after the 29-byte (NUL-terminated) Adobe XMP package prefix of the last XMP APP1
    // segment before the first scan, or null.
    byte[] xmp;
    // The orientation resolved from exifBuffer, filled in after the walk.
    TiffOrientation exif = TiffOrientation.ABSENT;
  }

  /**
   * Walks the complete marker structure from SOI to EOI. The JDK's built-in JPEG decoder is lenient
   * about truncated scan data (it silently produces an image padded with wrong pixels), so
   * truncation can only be detected by requiring the well-formed EOI the decoder never checks for.
   * Every {@code FF FF} pair outside the scan is a fill byte — the optional DRI (restart interval)
   * segment is a plain {@code FF DD} length-prefixed segment, handled by the general marker path
   * below. The scan itself is walked as raw bytes: an {@code FF} followed by another {@code FF} is
   * a fill byte and advances <em>one</em> byte, so the second {@code FF} is re-examined as a marker
   * candidate (a fill byte directly before the EOI must not swallow it); an {@code FF} followed by
   * {@code 00} (a stuffed data byte) or one of the {@code RST0}–{@code RST7} restart markers (valid
   * inside a scan written with a restart interval) advances two; any other following byte
   * terminates the scan and is the marker the outer loop classifies.
   */
  private static JpegInfo parseJpegHeader(byte[] b) throws IOException {
    JpegInfo info = new JpegInfo();
    int pos = 2;
    int n = b.length;
    boolean eoiSeen = false;
    boolean sosSeen = false;
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
        sosSeen = true;
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
          if (m == 0xFF) {
            pos++; // fill byte: the following FF itself starts the marker to be classified
            continue;
          }
          if (m == 0x00 || (m >= 0xD0 && m <= 0xD7)) {
            pos += 2; // a stuffed 00 data byte, or a restart marker inside the scan
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
      if (marker == 0xE1 && !sosSeen) { // APP1: only EXIF and XMP segments carry orientation
        // Like the reference's _open, only APP1 segments before the first scan are considered:
        // it breaks out of the marker walk at the first SOS. The EXIF payload is the 6-byte
        // prefix "Exif\x00\x00" followed by a TIFF block; the first EXIF-prefixed segment sets
        // the value (prefix included), every later one is concatenated without its prefix — but
        // the first segment's IFD0 still dominates the parse, since it is read first and the
        // only tag honored (Orientation) lives in IFD0. A malformed block is EXIF data, not
        // container corruption: the walk must continue and the decode must never fail because
        // of it (the reference's getexif() simply returns an empty object for such input), so
        // parseTiffOrientation reports absence instead of throwing.
        int payloadStart = pos + 4;
        int payloadLength = length - 2; // the segment length includes the two length bytes
        if (startsWith(b, payloadStart, payloadLength, EXIF_PACKAGE_PREFIX)) {
          if (!info.exifSeen) {
            info.exifSeen = true;
            info.exifBuffer = new ByteArrayOutputStream();
            info.exifBuffer.write(b, payloadStart, payloadLength);
          } else {
            info.exifBuffer.write(b, payloadStart + 6, payloadLength - 6);
          }
        } else if (startsWith(b, payloadStart, payloadLength, XMP_PACKAGE_PREFIX)) {
          // The XMP payload is everything after the 29-byte (NUL-terminated) package prefix;
          // the last XMP segment before the scan wins.
          int xmpStart = payloadStart + XMP_PACKAGE_PREFIX.length;
          info.xmp = Arrays.copyOfRange(b, xmpStart, payloadStart + payloadLength);
        }
      } else if (marker == 0xEE
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
        // The component count sits at pos + 9; the segment-length check above guarantees the
        // declared segment is fully present, so length >= 8 is the only guard the read below
        // needs. The declared length is deliberately not cross-checked against 8 + 3 *
        // components: a file whose length covers fewer specifiers than the count claims (the
        // pinned reject-jpeg-cmyk-4comp fixture is exactly that — a 3-component frame with its
        // count byte set to 4) must still be classified by its count, and anything this walk
        // lets through reaches the JDK reader, the final gate.
        if (length < 8) {
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
    if (info.exifBuffer != null) {
      info.exif = parseExifPayload(info.exifBuffer.toByteArray());
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

  private static int leU16(byte[] b, int off) {
    return (b[off] & 0xFF) | (b[off + 1] & 0xFF) << 8;
  }

  private static int leU32(byte[] b, int off) {
    return (b[off] & 0xFF)
        | (b[off + 1] & 0xFF) << 8
        | (b[off + 2] & 0xFF) << 16
        | (b[off + 3] & 0xFF) << 24;
  }

  private static long leU64(byte[] b, int off) {
    long v = 0;
    for (int i = 0; i < 8; i++) {
      v |= (b[off + i] & 0xFFL) << (i * 8);
    }
    return v;
  }

  private static long u64(byte[] b, int off) {
    long v = 0;
    for (int i = 0; i < 8; i++) {
      v = (v << 8) | (b[off + i] & 0xFFL);
    }
    return v;
  }

  /**
   * The IFD0 {@code Orientation} tag (0x0112) of one EXIF payload, resolved to what the reference's
   * {@code exif_transpose} would do with it: {@code present} is false when the block carries no
   * Orientation tag at all (malformed block, missing tag, unregistered type or count 0) — in which
   * case the XMP fallback may fire — and {@code value} is the effective orientation: a usable 2–8,
   * or 0 when the tag is present but its first value (a BYTE/ASCII/ UNDEFINED value, a non-integral
   * or out-of-range value, a 0/0 rational) never matches the reference's int-keyed transpose table.
   * {@code xmpSuppressed} is true exactly when the TIFF head itself is invalid (see {@link
   * #parseableTiffHead}): the reference's open-time {@code getexif()} then swallowed the parse
   * error and cached an empty object, so its XMP fallback never ran and no rotation happens
   * regardless of any XMP packet. It is consulted only by the JPEG path (for PNG an invalid head
   * makes the reference raise, which is tolerated with the XMP fallback intact).
   */
  private record TiffOrientation(int value, boolean present, boolean xmpSuppressed) {
    static final TiffOrientation ABSENT = new TiffOrientation(0, false, false);
    static final TiffOrientation INVALID = new TiffOrientation(0, false, true);
  }

  /**
   * Resolves the effective orientation exactly the way the reference's {@code getexif()} does: the
   * EXIF Orientation tag wins whenever it is <em>present</em> (even when it never rotates); only an
   * absent tag consults the XMP fallback, where the str-valued slot (the PNG text chunks) wins over
   * the byte-valued one and an empty value is falsy.
   */
  private static int resolveOrientation(TiffOrientation exif, byte[] xmpText, byte[] xmpBytes) {
    if (exif.present()) {
      return exif.value();
    }
    byte[] xmp = xmpText != null && xmpText.length > 0 ? xmpText : xmpBytes;
    return xmp != null && xmp.length > 0 ? xmpOrientationScan(xmp) : 0;
  }

  /**
   * Strips the repeated leading {@code Exif\x00\x00} package prefixes the reference strips in a
   * loop, then parses the remaining TIFF block; an empty remainder (a bare prefix) is absent.
   */
  private static TiffOrientation parseExifPayload(byte[] payload) {
    int off = 0;
    while (off + EXIF_PACKAGE_PREFIX.length <= payload.length
        && bytesEqualAt(payload, off, EXIF_PACKAGE_PREFIX)) {
      off += EXIF_PACKAGE_PREFIX.length;
    }
    if (off >= payload.length) {
      return TiffOrientation.ABSENT;
    }
    return parseTiffOrientation(payload, off, payload.length - off);
  }

  /**
   * Whether the reference's TIFF constructor ({@code ImageFileDirectory_v2}) would accept the head
   * at {@code b[off..off+len)}: it reads the first {@code min(8, len)} bytes and raises — a
   * bad-prefix {@code SyntaxError} or a truncated-head {@code struct.error} — unless {@code len >=
   * 8} and the first four bytes are one of the five parseable prefixes {@code II\x2a\x00}, {@code
   * MM\x00\x2a}, {@code II\x00\x2a}, {@code MM\x2a\x00} or {@code MM\x00\x2b}. The two cross-endian
   * prefixes ({@code II\x00\x2a}, {@code MM\x2a\x00}) are accepted even though they pair a byte
   * order with the other endian's magic, and so is the big-endian BigTIFF prefix {@code
   * MM\x00\x2b}: its third head byte is not 43, so the constructor takes the classic path and only
   * needs {@code len >= 8}. The remaining three of the four magic byte pairs are rejected: the
   * little-endian BigTIFF prefix {@code II\x2b\x00} (third byte 43) takes the BigTIFF path, which
   * reads an 8-byte offset from the always-empty bytes 8–15 of the 8-byte head and always raises;
   * {@code II\x00\x2b} and {@code MM\x2b\x00} are not among the reference's accepted prefixes at
   * all. This method only decides which unparseable heads the JPEG path must treat as invalid
   * (suppressing the XMP fallback); the parseable heads — all five, including the three whose magic
   * field is not 42 — are read normally by the parse below, which chooses the byte order from the
   * II/MM prefix and ignores the magic.
   */
  private static boolean parseableTiffHead(byte[] b, int off, int len) {
    if (len < 8) {
      return false;
    }
    int p0 = b[off] & 0xFF;
    int p1 = b[off + 1] & 0xFF;
    int p2 = b[off + 2] & 0xFF;
    int p3 = b[off + 3] & 0xFF;
    boolean ii = p0 == 0x49 && p1 == 0x49; // "II"
    boolean mm = p0 == 0x4d && p1 == 0x4d; // "MM"
    boolean classic = (p2 == 0x2a && p3 == 0x00) || (p2 == 0x00 && p3 == 0x2a);
    if (classic) {
      return ii || mm;
    }
    return mm && p2 == 0x00 && p3 == 0x2b;
  }

  /**
   * Reads the IFD0 {@code Orientation} tag (0x0112) from the TIFF block at {@code b[off..off+len)}
   * (the part of an EXIF payload left behind by {@link #parseExifPayload}), mirroring the
   * reference's IFD walk ({@code ImageFileDirectory_v2.load}): entries are processed in file order
   * and a later same-tag entry overwrites an earlier one; a truncated entry count, or a short read
   * of an offset value on any entry — not only the Orientation entry, since the reference loads
   * every entry's data before it filters by tag — stops the walk, keeping the entries stored before
   * it; entries whose type has no registered loader, or whose count is 0, are skipped without being
   * stored. A head the reference's TIFF constructor rejects — a head shorter than 8 bytes, a byte
   * order that is neither {@code II} (little-endian) nor {@code MM} (big-endian), or an unparseable
   * prefix (see {@link #parseableTiffHead}) — carries the invalid marker, which suppresses the JPEG
   * path's XMP fallback; the constructor reads only the 8-byte head, so the marker holds no matter
   * what follows it. The two-byte magic field is never checked — the reference's classic path reads
   * every parseable head, magic 42 or not, choosing the byte order from the prefix — so a non-42
   * magic on a parseable head is read normally. An out-of-range first-IFD offset on a parseable
   * head is reported as an absent orientation, never thrown, so a bad EXIF block can never break a
   * decode. Every access is bounds-checked against {@code len}.
   */
  private static TiffOrientation parseTiffOrientation(byte[] b, int off, int len) {
    // An invalid TIFF head makes the reference's open-time getexif() (JPEG without JFIF DPI)
    // swallow its parse error and cache an empty object: its XMP fallback never ran, so the JPEG
    // path must not consult one either. For PNG the same head makes the reference raise, which is
    // tolerated with the fallback intact — the marker is consulted only on the JPEG path. The
    // constructor reads only the 8-byte head, so the marker holds no matter what follows it: it is
    // returned before any structural read.
    if (!parseableTiffHead(b, off, len)) {
      return TiffOrientation.INVALID;
    }
    // The parseable prefixes guarantee len >= 8 and an II/MM byte-order prefix — the 8-byte head
    // is byte order (2) + magic (2) + first-IFD offset (4); the two-byte magic field is never
    // checked.
    int byteOrder = u16(b, off);
    boolean little = byteOrder == 0x4949; // "II"
    int ifdOffset = little ? leU32(b, off + 4) : (int) u32(b, off + 4);
    if (ifdOffset < 0 || (long) ifdOffset + 2 > len) {
      // A parseable head whose IFD offset is out of range is NOT invalid: the reference's
      // constructor succeeds and the IFD walk just aborts internally, XMP fallback allowed.
      return TiffOrientation.ABSENT; // the 2-byte entry count must be inside the block
    }
    int count = little ? leU16(b, off + ifdOffset) : u16(b, off + ifdOffset);
    TiffOrientation orientation = TiffOrientation.ABSENT;
    // Each IFD entry is 12 bytes: tag (2), type (2), count (4), value-or-offset (4).
    for (int i = 0; i < count; i++) {
      int entry = off + ifdOffset + 2 + i * 12;
      if (entry + 12 > off + len) {
        break; // a short entry read stops the walk; the earlier entries are kept
      }
      int type = little ? leU16(b, entry + 2) : u16(b, entry + 2);
      int unit = unitSize(type);
      if (unit == 0) {
        continue; // an unregistered type is skipped for every tag, not stored, and the walk
        // continues
      }
      long entryCount = little ? leU32(b, entry + 4) : u32(b, entry + 4);
      long size = entryCount * unit;
      int dataOff = entry + 8;
      if (size > 4) {
        long offset = little ? leU32(b, entry + 8) : u32(b, entry + 8);
        if (offset < 0 || (long) off + offset + size > (long) off + len) {
          break; // the reference's short read raises: any entry's bad offset stops the walk
        }
        dataOff = off + (int) offset;
      }
      int tag = little ? leU16(b, entry) : u16(b, entry);
      if (tag != ORIENTATION_TAG || entryCount == 0) {
        continue; // the reference's "if not data: continue"
      }
      orientation = orientationValue(type, b, dataOff, little);
    }
    return orientation;
  }

  /** The loader's unit size for a registered TIFF type, 0 for an unregistered one. */
  private static int unitSize(int type) {
    return switch (type) {
      case 1, 2, 6, 7 -> 1; // BYTE, ASCII, SIGNED_BYTE, UNDEFINED
      case 3, 8 -> 2; // SHORT, SIGNED_SHORT
      case 4, 9, 11, 13 -> 4; // LONG, SIGNED_LONG, FLOAT, IFD
      case 5, 10, 12, 16 -> 8; // RATIONAL, SIGNED_RATIONAL, DOUBLE, LONG8
      default -> 0;
    };
  }

  /**
   * Decodes the first element of a stored Orientation value to the rotation the reference's
   * transpose table would apply: the integer types rotate when their value is one of 2–8, the float
   * types when it is an integer-valued value in 2–8, the rational types when the quotient is
   * integral and in 2–8 (a 0/0 quotient is nan); BYTE, ASCII and UNDEFINED are stored and present
   * but never match the table. The value is always present.
   */
  private static TiffOrientation orientationValue(int type, byte[] b, int off, boolean little) {
    int rotation;
    switch (type) {
      case 3 -> rotation = usable(little ? leU16(b, off) : u16(b, off));
      case 4, 13 -> rotation = usable(little ? leU32(b, off) : u32(b, off));
      case 6 -> rotation = usable(b[off]);
      case 8 -> rotation = usable((short) (little ? leU16(b, off) : u16(b, off)));
      case 9 -> rotation = usable(little ? leU32(b, off) : (int) u32(b, off));
      case 11 ->
          rotation =
              usableFloat(
                  little
                      ? Float.intBitsToFloat(leU32(b, off))
                      : Float.intBitsToFloat((int) u32(b, off)));
      case 12 ->
          rotation =
              usableFloat(
                  little
                      ? Double.longBitsToDouble(leU64(b, off))
                      : Double.longBitsToDouble(u64(b, off)));
      case 16 -> rotation = usable(little ? leU64(b, off) : u64(b, off));
      case 5 ->
          rotation =
              usableRational(
                  little ? leU32(b, off) : u32(b, off),
                  little ? leU32(b, off + 4) : u32(b, off + 4));
      case 10 ->
          rotation =
              usableRational(
                  little ? leU32(b, off) : (int) u32(b, off),
                  little ? leU32(b, off + 4) : (int) u32(b, off + 4));
      default -> rotation = 0; // BYTE, ASCII, UNDEFINED
    }
    return new TiffOrientation(rotation, true, false);
  }

  /** An integer value rotates when it is one of the transpose table's keys 2–8. */
  private static int usable(long v) {
    return v >= 2 && v <= 8 ? (int) v : 0;
  }

  /** A float value rotates when it is integral and in 2–8 (a Python float equals its int twin). */
  private static int usableFloat(double d) {
    return d >= 2 && d <= 8 && d == Math.rint(d) ? (int) d : 0;
  }

  /**
   * A rational value rotates when the quotient is integral and in 2–8 (the reference compares it
   * against the int-keyed table via its Fraction value, so 6/1 and 12/2 both match key 6); a 0/0
   * quotient is nan and a non-integral quotient never matches.
   */
  private static int usableRational(long numerator, long denominator) {
    if (denominator == 0 || numerator % denominator != 0) {
      return 0;
    }
    return usable(numerator / denominator);
  }

  /**
   * The XMP fallback: the first left-to-right match of the reference's own regex {@code
   * tiff:Orientation(="|>)([0-9])} — exactly one digit, 0–9 — or 0 when the packet carries no
   * match. A {@code tiff:Orientation} occurrence whose following characters do not form {@code ="d}
   * or {@code >d} does not end the scan, like the regex engine.
   */
  private static int xmpOrientationScan(byte[] xmp) {
    int at = indexOf(xmp, 0, XMP_ORIENTATION_TAG);
    while (at >= 0) {
      int j = at + XMP_ORIENTATION_TAG.length;
      int digit = -1;
      if (j + 3 <= xmp.length && (xmp[j] & 0xFF) == '=' && (xmp[j + 1] & 0xFF) == '"') {
        digit = digitAt(xmp, j + 2);
      } else if (j + 2 <= xmp.length && (xmp[j] & 0xFF) == '>') {
        digit = digitAt(xmp, j + 1);
      }
      if (digit >= 0) {
        return digit;
      }
      at = indexOf(xmp, at + 1, XMP_ORIENTATION_TAG);
    }
    return 0;
  }

  /** The digit at {@code b[i]}, or -1 when the byte is not 0–9. */
  private static int digitAt(byte[] b, int i) {
    int c = b[i] & 0xFF;
    return c >= '0' && c <= '9' ? c - '0' : -1;
  }

  /** The first index at or after {@code from} where {@code needle} occurs in {@code haystack}. */
  private static int indexOf(byte[] haystack, int from, byte[] needle) {
    for (int i = from; i + needle.length <= haystack.length; i++) {
      if (bytesEqualAt(haystack, i, needle)) {
        return i;
      }
    }
    return -1;
  }

  /** True when the region {@code b[off..)} of the right length starts with {@code prefix}. */
  private static boolean startsWith(byte[] b, int off, int len, byte[] prefix) {
    return len >= prefix.length && bytesEqualAt(b, off, prefix);
  }

  /** True when {@code b[off..off+len)} equals {@code keyword} exactly. */
  private static boolean equalsRegion(byte[] b, int off, int len, byte[] keyword) {
    return len == keyword.length && bytesEqualAt(b, off, keyword);
  }

  /** True when {@code b[off..off+needle.length)} equals {@code needle}. */
  private static boolean bytesEqualAt(byte[] b, int off, byte[] needle) {
    if (off < 0 || off + needle.length > b.length) {
      return false;
    }
    for (int i = 0; i < needle.length; i++) {
      if (b[off + i] != needle[i]) {
        return false;
      }
    }
    return true;
  }

  /** The index of the first NUL byte in {@code b[from..to)}, or -1. */
  private static int indexOfByte(byte[] b, int from, int to) {
    for (int i = from; i < to; i++) {
      if (b[i] == 0) {
        return i;
      }
    }
    return -1;
  }

  /** True when {@code b[off..end)} is valid strict UTF-8, the reference's strict decode. */
  private static boolean validUtf8(byte[] b, int off, int end) {
    if (off >= end) {
      return true;
    }
    try {
      StandardCharsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(b, off, end - off));
      return true;
    } catch (CharacterCodingException e) {
      return false;
    }
  }

  /**
   * Inflates a complete zlib (RFC 1950) stream, the reference's {@code _safe_zlib_decompress}: the
   * inflated bytes when the data ends at a stream end, null when it is corrupt or incomplete.
   */
  private static byte[] inflateZlib(byte[] data) {
    Inflater inflater = new Inflater();
    try {
      inflater.setInput(data);
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      byte[] buffer = new byte[8192];
      while (!inflater.finished()) {
        int n = inflater.inflate(buffer);
        if (n == 0 && (inflater.needsInput() || inflater.needsDictionary())) {
          return null; // an incomplete stream
        }
        out.write(buffer, 0, n);
      }
      return out.toByteArray();
    } catch (DataFormatException e) {
      return null;
    } finally {
      inflater.end();
    }
  }

  // --------------------------------------------------------------- raster to RgbImage

  /** Everything the raster extraction needs to know about the decoded image. */
  private static final class DecodePlan {
    final int expectedBands;
    final byte[] palette;
    final int paletteCount;

    DecodePlan(Object header) {
      if (header instanceof PngInfo png) {
        switch (png.colorType) {
          case 0 -> expectedBands = 1;
          case 2 -> expectedBands = 3;
          case 3 -> expectedBands = 1;
          case 4 -> expectedBands = 2;
          default -> expectedBands = 4;
        }
        // A PLTE chunk on a truecolor or RGBA image is only a *suggested* palette (RFC 2083
        // s11.2); the built-in reader ignores it, so the palette is carried only for palette
        // images. Carrying it for other color types would send toRgbImage down the palette path
        // with bandOffsets still null (an NPE), since no sample-model setup was performed.
        palette = png.colorType == 3 ? png.palette : null;
        paletteCount = png.colorType == 3 ? png.paletteCount : 0;
      } else {
        JpegInfo jpeg = (JpegInfo) header;
        expectedBands = jpeg.components == 1 ? 1 : 3;
        palette = null;
        paletteCount = 0;
      }
    }
  }

  private static RgbImage toRgbImage(BufferedImage bi, DecodePlan plan, AlphaPolicy policy)
      throws IOException {
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
      // Palette: expand indices (palette tRNS was stripped before decode — see the class
      // javadoc). Sub-8-bit palettes arrive packed (several indices per byte); 8-bit palettes are
      // interleaved, one byte per index.
      int p = 0;
      if (sampleModel instanceof MultiPixelPackedSampleModel) {
        // Several indices are packed per byte, MSB-first per the PNG spec; the raster's own
        // sample path (the same one BufferedImage.getRGB uses) handles the packing, so the
        // raw byte buffer and the sample model's bit-offset getters are not used directly.
        for (int y = 0; y < h; y++) {
          for (int x = 0; x < w; x++) {
            int idx = raster.getSample(x, y, 0);
            p = expandPaletteIndex(idx, plan, out, p);
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
            p = expandPaletteIndex(idx, plan, out, p);
          }
        }
      } else {
        throw new IOException(
            "unexpected raster sample model: " + sampleModel.getClass().getSimpleName());
      }
    } else if (plan.expectedBands == 2) {
      // Grayscale + alpha: composite L over white, replicate — or keep L as-is when the policy
      // drops alpha (Pillow's convert("RGB") on an LA image replicates the gray band and
      // discards the alpha). redBand carries the gray band (red == green == blue, validated by
      // the probe).
      int p = 0;
      for (int y = 0; y < h; y++) {
        int rowBase = bankOffset + y * scanlineStride;
        for (int x = 0; x < w; x++) {
          int base = rowBase + x * pixelStride;
          int l = data[base + bandOffsets[redBand]] & 0xFF;
          int a = data[base + bandOffsets[alphaBand]] & 0xFF;
          int c = policy == AlphaPolicy.DROP_ALPHA ? l : compositeOverWhite(l, a);
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
      // RGBA: composite over white, or keep the source color channels and drop the alpha band
      // (Pillow's convert("RGB") preserves R/G/B and discards the alpha).
      int p = 0;
      for (int y = 0; y < h; y++) {
        int rowBase = bankOffset + y * scanlineStride;
        for (int x = 0; x < w; x++) {
          int base = rowBase + x * pixelStride;
          int r = data[base + bandOffsets[redBand]] & 0xFF;
          int g = data[base + bandOffsets[greenBand]] & 0xFF;
          int b = data[base + bandOffsets[blueBand]] & 0xFF;
          int a = data[base + bandOffsets[alphaBand]] & 0xFF;
          if (policy == AlphaPolicy.DROP_ALPHA) {
            out[p] = (byte) r;
            out[p + 1] = (byte) g;
            out[p + 2] = (byte) b;
          } else {
            out[p] = (byte) compositeOverWhite(r, a);
            out[p + 1] = (byte) compositeOverWhite(g, a);
            out[p + 2] = (byte) compositeOverWhite(b, a);
          }
          p += 3;
        }
      }
    }
    // out was allocated and filled here and is never shared, so it is adopted without the
    // public constructor's defensive copy.
    return RgbImage.ofUnchecked(w, h, out);
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
   * Expands one palette index to three output bytes. Palette {@code tRNS} was stripped before
   * decode (the pinned processor never applies it), so the palette color is written as-is. Writes
   * at {@code out[p..p+2]} and returns the next free output position.
   */
  private static int expandPaletteIndex(int idx, DecodePlan plan, byte[] out, int p)
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
    out[p] = plan.palette[base];
    out[p + 1] = plan.palette[base + 1];
    out[p + 2] = plan.palette[base + 2];
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
