// This file is a Java port of Pillow's C resampling source, pinned at:
//
//   Source file: https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
//   Source SHA-256: 808c212d03289b9ab70fad67168da1b14ed8dd74f1607c7a7278c50cbc1ae0ab
//   License file:  https://github.com/python-pillow/Pillow/blob/12.3.0/LICENSE
//   License SHA-256: 15181e7363dca9aed78b79bebebc7fde7f1814b8bd311ea3b87ae8ccadfc185b
//
// The Python Imaging Library (PIL) is
//
//     Copyright (c) 1997-2011 by Secret Labs AB
//     Copyright (c) 1995-2011 by Fredrik Lundh and contributors
//
// Pillow is the friendly PIL fork. It is
//
//     Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors
//
// Like PIL, Pillow is licensed under the open source MIT-CMU License:
//
// By obtaining, using, and/or copying this software and/or its associated
// documentation, you agree that you have read, understood, and will comply
// with the following terms and conditions:
//
// Permission to use, copy, modify and distribute this software and its
// documentation for any purpose and without fee is hereby granted,
// provided that the above copyright notice appears in all copies, and that
// both that copyright notice and this permission notice appear in supporting
// documentation, and that the name of Secret Labs AB or the author not be
// used in advertising or publicity pertaining to distribution of the software
// without specific, written prior permission.
//
// SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
// SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
// IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
// INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
// LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
// OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
// PERFORMANCE OF THIS SOFTWARE.
//
// The port preserves the C algorithm line for line: the per-direction
// coefficient precomputation (support, center/scale, normalization), the
// fixed-point coefficient rounding (22 precision bits, round-half-away-from-zero),
// the two-pass horizontal-then-vertical composition with the 8-bit intermediate
// clip, and the clip8 lookup behavior. Double arithmetic follows IEEE 754, the
// same semantics as the C source on this host's toolchain.

package se.alipsa.jmlx.vision;

final class PillowResample {
  // 8 bits for result. Filter can have negative areas; in one case the sum of the
  // coefficients will be negative, in the other it will be more than 1.0. That is why
  // we need two extra bits for overflow and int type.
  private static final int PRECISION_BITS = 32 - 8 - 2;

  // Handles values from -640 to 639 (index 640 maps to 0). Built with the same
  // construction as the C table: 640 zeros, the 0..255 identity, then 255s.
  private static final int[] CLIP8_LOOKUPS = buildClip8Lookups();

  private PillowResample() {}

  /**
   * Resizes the interleaved RGB source to {@code outWidth x outHeight} with the pinned Pillow
   * kernel, returning a fresh row-major byte array that never aliases {@code in}.
   *
   * @throws IllegalArgumentException if {@code outWidth * outHeight * 3}, or the two-pass
   *     intermediate buffer ({@code outWidth} times the spanned source rows, times 3), overflows an
   *     int
   */
  static byte[] resize(
      byte[] in, int inWidth, int inHeight, int outWidth, int outHeight, Resampling method) {
    if ((long) outWidth * outHeight > Integer.MAX_VALUE / 3) {
      throw new IllegalArgumentException(
          "resize target "
              + outWidth
              + "x"
              + outHeight
              + " overflows an int pixel buffer: width*height is "
              + (long) outWidth * outHeight
              + " but must be at most "
              + (Integer.MAX_VALUE / 3));
    }
    if (outWidth == inWidth && outHeight == inHeight) {
      // ImagingResampleInner with no horizontal/vertical pass: ImagingCopy.
      return in.clone();
    }

    // Vertical coefficients are computed first, exactly as in ImagingResampleInner.
    Coeffs vert = precompute(inHeight, 0, inHeight, outHeight, method);
    // First used row in the source image.
    int yboxFirst = vert.bounds[0];
    // Last used row in the source image.
    int yboxLast = vert.bounds[outHeight * 2 - 2] + vert.bounds[outHeight * 2 - 1];

    byte[] out;
    if (outWidth != inWidth) {
      // Two-pass resize, horizontal pass.
      Coeffs horiz = precompute(inWidth, 0, inWidth, outWidth, method);
      // Shift bounds for the vertical pass.
      for (int i = 0; i < outHeight; i++) {
        vert.bounds[i * 2] -= yboxFirst;
      }
      int intermediateHeight = yboxLast - yboxFirst;
      if ((long) outWidth * intermediateHeight > Integer.MAX_VALUE / 3) {
        // The output check above covers outWidth * outHeight only; the horizontal pass allocates
        // outWidth * intermediateHeight * 3, and intermediateHeight spans source rows, so a tall
        // narrow source resized to a wide short target overflows it even when the target fits.
        throw new IllegalArgumentException(
            "resize target "
                + outWidth
                + "x"
                + outHeight
                + " overflows an int intermediate pixel buffer: outWidth*intermediateHeight is "
                + (long) outWidth * intermediateHeight
                + " but must be at most "
                + (Integer.MAX_VALUE / 3));
      }
      byte[] intermediate =
          horizontalPass(in, inWidth, yboxFirst, intermediateHeight, outWidth, horiz);
      out =
          outHeight != inHeight
              ? verticalPass(intermediate, outWidth, intermediateHeight, 0, outHeight, vert)
              : intermediate;
    } else {
      // imIn can be the original image (no horizontal pass was performed).
      out = verticalPass(in, inWidth, inHeight, 0, outHeight, vert);
    }
    return out;
  }

  private static final class Coeffs {
    final int ksize;
    final int[] bounds;
    final double[] kk;

    Coeffs(int ksize, int[] bounds, double[] kk) {
      this.ksize = ksize;
      this.bounds = bounds;
      this.kk = kk;
    }
  }

  private static Coeffs precompute(int inSize, int in0, int in1, int outSize, Resampling method) {
    double scale = (double) (in1 - in0) / outSize;
    // Prepare for horizontal stretch.
    double filterscale = scale;
    if (filterscale < 1.0) {
      filterscale = 1.0;
    }

    // Determine support size (length of resampling filter).
    double support = support(method) * filterscale;

    // Maximum number of coeffs.
    int ksize = (int) Math.ceil(support) * 2 + 1;

    // Check for overflow (the C source checks against malloc capacity).
    if (outSize > Integer.MAX_VALUE / (ksize * 8)) {
      throw new IllegalArgumentException(
          "coefficient buffer for output size " + outSize + " overflows an int");
    }

    double[] kk = new double[outSize * ksize];
    int[] bounds = new int[outSize * 2];

    double invFilterscale = 1.0 / filterscale; // invariant over the loop
    for (int xx = 0; xx < outSize; xx++) {
      double center = in0 + (xx + 0.5) * scale;
      // Round the value.
      int xmin = (int) (center - support + 0.5);
      if (xmin < 0) {
        xmin = 0;
      }
      // Round the value.
      int xmax = (int) (center + support + 0.5);
      if (xmax > inSize) {
        xmax = inSize;
      }
      xmax -= xmin;
      double ww = 0.0;
      for (int x = 0; x < xmax; x++) {
        double w = filter(method, (x + xmin - center + 0.5) * invFilterscale);
        kk[xx * ksize + x] = w;
        ww += w;
      }
      if (ww != 0.0) {
        for (int x = 0; x < xmax; x++) {
          kk[xx * ksize + x] /= ww;
        }
      }
      // Remaining values should stay empty if they are used despite of xmax.
      for (int x = xmax; x < ksize; x++) {
        kk[xx * ksize + x] = 0;
      }
      bounds[xx * 2 + 0] = xmin;
      bounds[xx * 2 + 1] = xmax;
    }
    return new Coeffs(ksize, bounds, kk);
  }

  private static int[] normalize8bpc(double[] prekk) {
    int[] kk = new int[prekk.length];
    for (int x = 0; x < prekk.length; x++) {
      if (prekk[x] < 0) {
        kk[x] = (int) (-0.5 + prekk[x] * (1L << PRECISION_BITS));
      } else {
        kk[x] = (int) (0.5 + prekk[x] * (1L << PRECISION_BITS));
      }
    }
    return kk;
  }

  /**
   * Horizontal 8-bit pass over three bands: reads rows {@code [offset, offset + outHeight)} of the
   * source and writes {@code outWidth x outHeight}.
   */
  private static byte[] horizontalPass(
      byte[] in, int inWidth, int offset, int outHeight, int outWidth, Coeffs coeffs) {
    int[] k = normalize8bpc(coeffs.kk);
    int ksize = coeffs.ksize;
    byte[] out = new byte[outWidth * outHeight * 3];
    for (int yy = 0; yy < outHeight; yy++) {
      int rowInBase = (yy + offset) * inWidth * 3;
      for (int xx = 0; xx < outWidth; xx++) {
        int xmin = coeffs.bounds[xx * 2 + 0];
        int xmax = coeffs.bounds[xx * 2 + 1];
        int kbase = xx * ksize;
        int ss0 = 1 << (PRECISION_BITS - 1);
        int ss1 = 1 << (PRECISION_BITS - 1);
        int ss2 = 1 << (PRECISION_BITS - 1);
        for (int x = 0; x < xmax; x++) {
          int base = rowInBase + (x + xmin) * 3;
          ss0 += (in[base] & 0xFF) * k[kbase + x];
          ss1 += (in[base + 1] & 0xFF) * k[kbase + x];
          ss2 += (in[base + 2] & 0xFF) * k[kbase + x];
        }
        int outBase = (yy * outWidth + xx) * 3;
        out[outBase] = (byte) clip8(ss0);
        out[outBase + 1] = (byte) clip8(ss1);
        out[outBase + 2] = (byte) clip8(ss2);
      }
    }
    return out;
  }

  /**
   * Vertical 8-bit pass over three bands: reads rows {@code [offset, offset + sourceHeight)} of the
   * source and writes {@code inWidth x outHeight}.
   */
  private static byte[] verticalPass(
      byte[] in, int inWidth, int sourceHeight, int offset, int outHeight, Coeffs coeffs) {
    int[] k = normalize8bpc(coeffs.kk);
    int ksize = coeffs.ksize;
    byte[] out = new byte[inWidth * outHeight * 3];
    for (int yy = 0; yy < outHeight; yy++) {
      int ymin = coeffs.bounds[yy * 2 + 0];
      int ymax = coeffs.bounds[yy * 2 + 1];
      int kbase = yy * ksize;
      for (int xx = 0; xx < inWidth; xx++) {
        int ss0 = 1 << (PRECISION_BITS - 1);
        int ss1 = 1 << (PRECISION_BITS - 1);
        int ss2 = 1 << (PRECISION_BITS - 1);
        for (int y = 0; y < ymax; y++) {
          int base = ((y + ymin + offset) * inWidth + xx) * 3;
          ss0 += (in[base] & 0xFF) * k[kbase + y];
          ss1 += (in[base + 1] & 0xFF) * k[kbase + y];
          ss2 += (in[base + 2] & 0xFF) * k[kbase + y];
        }
        int outBase = (yy * inWidth + xx) * 3;
        out[outBase] = (byte) clip8(ss0);
        out[outBase + 1] = (byte) clip8(ss1);
        out[outBase + 2] = (byte) clip8(ss2);
      }
    }
    return out;
  }

  private static int clip8(int in) {
    return CLIP8_LOOKUPS[640 + (in >> PRECISION_BITS)];
  }

  private static int[] buildClip8Lookups() {
    int[] table = new int[1280];
    for (int i = 0; i < 1280; i++) {
      int v = i - 640;
      table[i] = v <= 0 ? 0 : (v >= 255 ? 255 : v);
    }
    return table;
  }

  private static double support(Resampling method) {
    return switch (method) {
      case BILINEAR -> 1.0;
      case BICUBIC -> 2.0;
      case LANCZOS -> 3.0;
    };
  }

  private static double filter(Resampling method, double x) {
    return switch (method) {
      case BILINEAR -> bilinearFilter(x);
      case BICUBIC -> bicubicFilter(x);
      case LANCZOS -> lanczosFilter(x);
    };
  }

  private static double bilinearFilter(double x) {
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return 1.0 - x;
    }
    return 0.0;
  }

  // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
  private static double bicubicFilter(double x) {
    // a = -0.5
    if (x < 0.0) {
      x = -x;
    }
    if (x < 1.0) {
      return ((-0.5 + 2.0) * x - (-0.5 + 3.0)) * x * x + 1;
    }
    if (x < 2.0) {
      return (((x - 5) * x + 8) * x - 4) * -0.5;
    }
    return 0.0;
  }

  private static double sincFilter(double x) {
    if (x == 0.0) {
      return 1.0;
    }
    x = x * Math.PI;
    return Math.sin(x) / x;
  }

  private static double lanczosFilter(double x) {
    // Truncated sinc.
    if (-3.0 <= x && x < 3.0) {
      return sincFilter(x) * sincFilter(x / 3);
    }
    return 0.0;
  }
}
