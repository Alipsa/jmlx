package se.alipsa.jmlx.vision;

/**
 * Resampling kernels for {@link ImageTransforms#resize}.
 *
 * <p>Each kernel is a port of the pinned Pillow 12.3.0 C resampler; see {@code NOTICE} for the
 * source file hash and license.
 */
public enum Resampling {
  /** Linear interpolation (Pillow {@code BILINEAR}, support 1). */
  BILINEAR,
  /** Cubic convolution with {@code a = -0.5} (Pillow {@code BICUBIC}, support 2). */
  BICUBIC,
  /** Three-lobe windowed sinc (Pillow {@code LANCZOS}, support 3). */
  LANCZOS
}
