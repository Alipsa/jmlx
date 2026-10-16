package se.alipsa.jmlx.vision;

/**
 * How {@link ImageDecoder} handles the alpha band of RGBA (and grayscale-with-alpha) input.
 *
 * <p>{@code tRNS} handling is independent of this choice: a {@code tRNS} chunk is stripped before
 * decode for every color type under every policy, because the pinned reference processor never
 * applies it (its palette is rebuilt from RGB-only {@code getpalette()} data, and its RGB
 * conversion drops a truecolor/grayscale {@code tRNS}). Opaque input decodes identically under both
 * policies.
 */
public enum AlphaPolicy {
  /**
   * Composites the color channels over an opaque white background using Pillow's exact integer
   * compositing formula (the behavior of every {@link ImageDecoder} overload that does not take
   * this parameter).
   */
  COMPOSITE_OVER_WHITE,
  /**
   * Keeps the source color channels and discards the alpha band, matching Pillow's {@code
   * convert("RGB")} — the reference Qwen3-VL image path drops alpha instead of compositing (see the
   * Phase 7.3b WP1 reference findings).
   */
  DROP_ALPHA
}
