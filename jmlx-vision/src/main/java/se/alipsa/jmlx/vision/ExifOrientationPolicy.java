package se.alipsa.jmlx.vision;

/**
 * How {@link ImageDecoder} handles the EXIF {@code Orientation} tag of JPEG and PNG input.
 *
 * <p>Only the IFD0 {@code Orientation} tag (0x0112) is honored, resolved exactly the way the pinned
 * reference's {@code getexif()} resolves it. Its source is: in a JPEG, the {@code APP1} segment
 * before the first scan whose payload starts with the 6-byte prefix {@code Exif\x00\x00} (later
 * EXIF {@code APP1}s are concatenated to its bytes, but the first segment's IFD0 dominates the
 * parse); in a PNG, the last {@code eXIf} chunk or text-chunk {@code exif} keyword in file order —
 * including after {@code IDAT} — whose value is the TIFF block (an {@code eXIf} chunk's data
 * directly, with any leading {@code Exif\x00\x00} prefixes stripped). A later same-tag IFD entry
 * overwrites an earlier one, and the entries stored before a truncated IFD or a short offset read
 * are kept.
 *
 * <p>Whenever the tag is <em>present</em> — stored by one of the reference's registered TIFF types
 * (BYTE, ASCII, SHORT, LONG, RATIONAL, SIGNED_BYTE, UNDEFINED, SIGNED_SHORT, SIGNED_LONG, FLOAT,
 * DOUBLE, IFD, LONG8) with a non-zero count — it decides the outcome on its own and the XMP
 * fallback below never fires. It rotates only when its first value is an integer in 2–8, or an
 * integer-valued float or rational (so 6/1 and 12/2 both match 6); a BYTE, ASCII or UNDEFINED
 * value, a non-integral or out-of-range value, and a 0/0 rational never rotate. A block without the
 * tag — a malformed or truncated block (bad byte order, wrong TIFF magic, an out-of-range IFD
 * offset, a truncated IFD, a bare {@code Exif\x00\x00} prefix), an unregistered entry type, or a
 * zero entry count — keeps the tag absent.
 *
 * <p>Only an absent tag consults the XMP fallback: the first single-digit match of {@code
 * tiff:Orientation(="|>)} in the XMP packet supplies the orientation, or none when the packet
 * carries no match. The packet is: in a JPEG, the payload after the 29-byte (NUL-terminated) Adobe
 * XMP package prefix of the last XMP {@code APP1} segment before the first scan; in a PNG, the
 * value of the last text-chunk keyword {@code XML:com.adobe.xmp} (a {@code tEXt} or decompressed
 * {@code zTXt} value first, the {@code iTXt} value otherwise).
 *
 * <p>A malformed or truncated EXIF or XMP block is data, not corruption: under either policy the
 * pixels are returned as stored, and the decode never fails because of it. Where the reference's
 * {@code getexif()} would itself raise on pathological metadata (a bad TIFF byte order or magic, a
 * truncated post-{@code IDAT} chunk, an {@code iTXt} XMP value it stores as an object its regex
 * cannot scan), {@link ImageDecoder} treats the orientation as absent instead, and a {@code zTXt}
 * or valid-UTF-8 {@code iTXt} {@code exif} value — a string the reference cannot load — is parsed
 * as the underlying bytes (and can rotate): defensive divergences documented on {@link
 * ImageDecoder}.
 *
 * <p><b>Scope (Phase 7.3b WP9):</b> the Qwen3-VL file path decodes with {@link #APPLY} (Pillow
 * {@code ImageOps.exif_transpose} followed by {@code convert("RGB")}); the SmolVLM/Idefics3 path
 * keeps {@link #IGNORE}, the existing decode contract pinned byte-for-byte by the committed
 * image-oracle fixtures.
 */
public enum ExifOrientationPolicy {

  /**
   * EXIF orientation metadata is read for nothing: the decoded pixels are returned exactly as
   * stored. This is the behavior of every {@link ImageDecoder} overload that does not take an
   * orientation policy — the pinned SmolVLM/Idefics3 reference path performs no {@code
   * exif_transpose}.
   */
  IGNORE,

  /**
   * Applies the stored EXIF orientation as Pillow's {@code ImageOps.exif_transpose} does (Pillow
   * 12.3.0, the same pin whose alpha compositing {@link ImageDecoder} ports), before the RGB
   * conversion. The transform is an exact pixel permutation — no resampling — of the stored {@code
   * W}x{@code H} pixels into the displayed orientation:
   *
   * <ul>
   *   <li>2 — flip horizontally (PIL {@code FLIP_LEFT_RIGHT})
   *   <li>3 — rotate 180° (PIL {@code ROTATE_180})
   *   <li>4 — flip vertically (PIL {@code FLIP_TOP_BOTTOM})
   *   <li>5 — transpose, main-diagonal flip (PIL {@code TRANSPOSE})
   *   <li>6 — rotate 90° clockwise (PIL {@code ROTATE_270})
   *   <li>7 — transverse, anti-diagonal flip (PIL {@code TRANSVERSE})
   *   <li>8 — rotate 90° counter-clockwise (PIL {@code ROTATE_90})
   * </ul>
   *
   * <p>For 5–8 the result dimensions swap ({@code W'} = {@code H}, {@code H'} = {@code W}).
   */
  APPLY
}
