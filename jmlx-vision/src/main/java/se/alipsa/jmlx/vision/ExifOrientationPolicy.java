package se.alipsa.jmlx.vision;

/**
 * How {@link ImageDecoder} handles the EXIF {@code Orientation} tag of JPEG and PNG input.
 *
 * <p>Only the {@code Orientation} tag (0x0112) of IFD0 of the first valid EXIF block is honored: in
 * a JPEG, the first {@code APP1} segment whose payload starts with the 6-byte prefix {@code
 * Exif\x00\x00}; in a PNG, the first {@code eXIf} chunk, whose chunk data is the TIFF block
 * directly (no prefix). A malformed or truncated EXIF block — bad byte order, wrong TIFF magic, an
 * out-of-range IFD offset, a truncated IFD — is treated as absent under either policy: the pixels
 * are returned as stored, and the decode never fails because of it, matching the reference, whose
 * {@code getexif()} simply returns an empty object for such input. A valid block without the tag, a
 * value of 0 or 1, or a value outside 2–8 likewise leaves the pixels unchanged.
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
