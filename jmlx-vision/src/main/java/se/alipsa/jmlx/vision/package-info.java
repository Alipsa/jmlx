/**
 * Pure-Java image preprocessing for {@code jmlx-models}: PNG/JPEG decoding, RGB conversion,
 * Pillow-exact resampling, rescale/normalize, and the SmolVLM/Idefics3 image processor (two-stage
 * resize chain, tile splitting, global thumbnail, padding masks, grid/order metadata).
 *
 * <p>This module has no native dependency and requires no bootstrap; it is safe on any Java 21+
 * JVM. Resampling (bilinear, bicubic, three-lobe LANCZOS) is a bit-exact port of the pinned Pillow
 * 12.3.0 {@code src/libImaging/Resample.c}; the full MIT-CMU license text, source and license
 * hashes, and the per-file notice are recorded in the module's {@code NOTICE} file and in the
 * header of the ported source file.
 *
 * <p><b>Decode policy.</b> {@link ImageDecoder} decodes 8-bit RGB, grayscale, RGBA,
 * grayscale-with-alpha and palette (with {@code tRNS}) PNG and 8-bit RGB/grayscale JPEG through the
 * JDK's built-in {@code javax.imageio} readers, extracting raw {@code Raster} samples without
 * {@code ColorModel}/{@code getRGB} color conversion. It <b>ignores EXIF orientation</b> (an
 * EXIF-rotated JPEG yields its stored, unrotated pixels, matching the pinned reference path;
 * callers must rotate phone photos themselves) and <b>ignores ICC and gAMA color management</b>
 * (embedded profiles and gamma chunks are not applied to samples). Alpha is composited over a white
 * background using Pillow's exact integer compositing formula, matching the reference processor's
 * {@code convert_to_rgb} step. 16-bit PNG and CMYK/YCCK JPEG input is rejected with format-specific
 * {@link java.io.IOException}s before conversion.
 */
package se.alipsa.jmlx.vision;
