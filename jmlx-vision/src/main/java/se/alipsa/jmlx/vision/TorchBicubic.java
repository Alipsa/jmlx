// This file is a Java port of PyTorch's CPU upsampling (interpolate) source, pinned at tag v2.6.0:
//
//   Source files:
//     https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cpu/UpSampleKernel.cpp
//     Source SHA-256: e3b7bf13fecd1af33e1c6d7dfb441d97590b71075194d040c5ad325814fdac1f
//     https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/UpSample.h
//     Source SHA-256: d49801938b31a544d2f55d616eea5db4cbd611ba691dca128ccd6aa870d008ba
//
//   License file:  https://github.com/pytorch/pytorch/blob/v2.6.0/LICENSE
//   License SHA-256: 47a26beb94e3f6b333a3677fc85d546f1fdfd2f0b3686c26d2fb5b10e0134165
//
// PyTorch is licensed under the BSD 3-Clause "New" or "Revised" License; the full license text
// is carried in the module's NOTICE file.
//
// What is ported, and exactly which reference code path it mirrors:
//
//   torch.nn.functional.interpolate(x, size=(H2, W2), mode='bicubic')
//   with the default align_corners=False and antialias=False, on a contiguous float32 NCHW
//   tensor with batch size 1 (a 3-channel image), executes the CPU float path
//   upsample_bicubic2d_kernel_impl -> upsample_generic_Nd_kernel_impl<2, scale_t,
//   HelperInterpCubic>, which (UpSampleKernel.cpp lines 1290-1329 and 1419-1494, UpSample.h
//   lines 270-314, 397-448):
//
//     1. computes the per-dimension scale in float32: scale = (float) input_size / output_size
//        (area_pixel_compute_scale -> compute_scales_value with no user scale factor);
//     2. for each output index i computes the float32 source position
//        real = scale * (i + 0.5f) - 0.5f (area_pixel_compute_source_index with
//        align_corners=false and cubic=true, i.e. negative real positions are NOT clamped),
//        input_index = min(floorf(real), input_size - 1),
//        lambda = clamp(real - input_index, 0, 1);
//     3. the four cubic weights through get_cubic_upsample_coefficients with A = -0.75f
//        (cubic_convolution1 / cubic_convolution2 as defined at UpSample.h lines 397-407);
//     4. the four input indices are clamped to [0, input_size - 1];
//     5. each output pixel is the height-outer, width-inner separable accumulation
//        (Interpolate<2>/Interpolate<1> at UpSampleKernel.cpp lines 76-108), in float32,
//        with the first term a bare product and each subsequent term output += t * wts.
//
// The pinned reference wheels are built by clang for AArch64 with floating-point contraction
// enabled (the C++ default), so every C subexpression of the form `a * b + c` in steps 2, 3
// and 5 is executed by the hardware fused-multiply-add instruction: the product a * b is
// formed exactly and only the final sum is rounded to float32. This port reproduces that
// compiled arithmetic exactly: each such subexpression is evaluated with Math.fma, the bit-exact
// IEEE 754 binary32 fused multiply-add (single rounding, round-to-nearest-even), while the
// bare products, subtractions and the scale division remain plain single-rounding float32
// operations in the C source's order. A plain * and + transcription deviates from the wheels
// by up to ~881 float ulps on the shared fixtures; the fused form is bit-identical (verified
// by the phi3v image-oracle family). The module's own PillowResample BICUBIC kernel is a
// different algorithm (Pillow's 8-bit fixed-point a = -0.5 pipeline) and is NOT used here.

package se.alipsa.jmlx.vision;

final class TorchBicubic {

  private TorchBicubic() {}

  /**
   * Resizes one contiguous CHW float32 image (three channels, {@code inHeight x inWidth} per
   * channel, row-major within each channel) to {@code outHeight x outWidth} with the pinned torch
   * bicubic semantics: {@code align_corners = false}, no antialias, {@code A = -0.75}, in float32
   * with the exact fused-multiply-add contraction of the pinned wheels (see the file header).
   *
   * <p>The {@code input} buffer is read and never mutated; a fresh {@code 3 * outHeight * outWidth}
   * buffer is returned.
   *
   * @throws IllegalArgumentException if {@code input} is null or has the wrong length, or a
   *     dimension is not positive
   */
  static float[] resizeChw(float[] input, int inHeight, int inWidth, int outHeight, int outWidth) {
    if (input == null) {
      throw new IllegalArgumentException("input must not be null");
    }
    if (inHeight <= 0 || inWidth <= 0) {
      throw new IllegalArgumentException(
          "input dimensions must be positive: " + inHeight + "x" + inWidth);
    }
    if (outHeight <= 0 || outWidth <= 0) {
      throw new IllegalArgumentException(
          "output dimensions must be positive: " + outHeight + "x" + outWidth);
    }
    long required = (long) inHeight * inWidth * 3;
    if (input.length != required) {
      throw new IllegalArgumentException(
          "input length must be 3*height*width=" + required + " but was " + input.length);
    }
    long outCount = (long) outHeight * outWidth * 3;
    if (outCount > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "output " + outHeight + "x" + outWidth + " overflows an int float buffer: " + outCount);
    }

    // Per-dimension tables of the four clamped input indices and the four float32 weights,
    // exactly the HelperInterpCubic::compute_indices_weights output tensors (one per output
    // index, stride already folded in: the height indices are measured in rows of inWidth
    // elements, the width indices in elements).
    int[] idxH = new int[outHeight * 4];
    float[] wtsH = new float[outHeight * 4];
    computeDimTable(inHeight, outHeight, inWidth, idxH, wtsH);
    int[] idxW = new int[outWidth * 4];
    float[] wtsW = new float[outWidth * 4];
    computeDimTable(inWidth, outWidth, 1, idxW, wtsW);

    int inPlane = inHeight * inWidth;
    int outPlane = outHeight * outWidth;
    float[] out = new float[3 * outPlane];
    for (int c = 0; c < 3; c++) {
      float[] inPlaneBuf = input;
      int channelOffset = c * inPlane;
      int outOffset = c * outPlane;
      for (int oh = 0; oh < outHeight; oh++) {
        int heightBase = oh * 4;
        for (int ow = 0; ow < outWidth; ow++) {
          int widthBase = ow * 4;
          // Inner (width) interpolation of the first height tap, then the separable
          // height-outer accumulation: first term a bare product, the rest fused adds.
          float row = innerRow(inPlaneBuf, channelOffset + idxH[heightBase], idxW, widthBase, wtsW);
          float acc = row * wtsH[heightBase];
          for (int jh = 1; jh < 4; jh++) {
            row =
                innerRow(inPlaneBuf, channelOffset + idxH[heightBase + jh], idxW, widthBase, wtsW);
            acc = fma32(row, wtsH[heightBase + jh], acc);
          }
          out[outOffset + oh * outWidth + ow] = acc;
        }
      }
    }
    return out;
  }

  /**
   * The width tap inner-product of one height row: {@code sum_j in[row + idxW[j]] * wtsW[j]}, with
   * the contracted accumulation of the compiled kernel: a bare first product, then one fused
   * multiply-add per further term.
   */
  private static float innerRow(
      float[] in, int rowOffset, int[] idxW, int widthBase, float[] wtsW) {
    float acc = in[rowOffset + idxW[widthBase]] * wtsW[widthBase];
    acc = fma32(in[rowOffset + idxW[widthBase + 1]], wtsW[widthBase + 1], acc);
    acc = fma32(in[rowOffset + idxW[widthBase + 2]], wtsW[widthBase + 2], acc);
    return fma32(in[rowOffset + idxW[widthBase + 3]], wtsW[widthBase + 3], acc);
  }

  /**
   * Fills the four-index / four-weight table of one spatial dimension (UpSample.h {@code
   * area_pixel_compute_scale}, {@code area_pixel_compute_source_index}, {@code
   * guard_index_and_lambda}, {@code get_cubic_upsample_coefficients} with {@code A = -0.75}, and
   * the clamped index computation at UpSampleKernel.cpp line 1321), in float32 with the compiled
   * kernel's fused-multiply-add contraction.
   *
   * @param inSize source dimension size
   * @param outSize target dimension size
   * @param stride the input stride of this dimension, in elements (rows for height, 1 for width)
   */
  private static void computeDimTable(
      int inSize, int outSize, int stride, int[] idxOut, float[] wtsOut) {
    // compute_scales_value with no user scale: float32 division, both operands converted to
    // float32 (exact: the sizes fit in 24 mantissa bits).
    float scale = ((float) inSize) / outSize;
    for (int i = 0; i < outSize; i++) {
      // area_pixel_compute_source_index(align_corners=false, cubic=true): the C expression
      // scale * (i + 0.5f) - 0.5f is one fused multiply-add in the compiled kernel.
      float real = fma32(scale, (float) (i + 0.5f), -0.5f);
      // guard_index_and_lambda: floorf, then clamp; the int64 is exact in float32 here.
      int inputIndex = (int) Math.floor(real);
      if (inputIndex > inSize - 1) {
        inputIndex = inSize - 1;
      }
      float lambda = real - (float) inputIndex;
      if (lambda < 0.0f) {
        lambda = 0.0f;
      } else if (lambda > 1.0f) {
        lambda = 1.0f;
      }
      // get_cubic_upsample_coefficients(coeffs, lambda) with A = -0.75.
      float x1 = lambda;
      float x2 = 1.0f - x1;
      wtsOut[i * 4 + 0] = cubic2(x1 + 1.0f);
      wtsOut[i * 4 + 1] = cubic1(x1);
      wtsOut[i * 4 + 2] = cubic1(x2);
      wtsOut[i * 4 + 3] = cubic2(x2 + 1.0f);
      // UpSampleKernel.cpp:1321: max(min(input_index + j - 1, input_size - 1), 0) * stride.
      for (int j = 0; j < 4; j++) {
        int tap = inputIndex + j - 1;
        if (tap > inSize - 1) {
          tap = inSize - 1;
        } else if (tap < 0) {
          tap = 0;
        }
        idxOut[i * 4 + j] = tap * stride;
      }
    }
  }

  // cubic_convolution1 (UpSample.h:400) with A = -0.75: ((A + 2) * x - (A + 3)) * x * x + 1,
  // as contracted by the compiler: ((A + 2) * x + (-(A + 3))) fuses, then x * x, then
  // (that * x + 1) fuses. A + 2 = 1.25 and A + 3 = 2.25 are exact in float32.
  private static float cubic1(float x) {
    float t = fma32(1.25f, x, -2.25f);
    t *= x;
    return fma32(t, x, 1.0f);
  }

  // cubic_convolution2 (UpSample.h:405) with A = -0.75: ((A * x - 5 * A) * x + 8 * A) * x
  // - 4 * A, fully fused: -5 * A = 3.75, 8 * A = -6.0 and -4 * A = 3.0, all exact in float32.
  private static float cubic2(float x) {
    float t = fma32(-0.75f, x, 3.75f);
    t = fma32(t, x, -6.0f);
    return fma32(t, x, 3.0f);
  }

  /**
   * Bit-exact IEEE 754 binary32 fused multiply-add: the single rounding of the exact value {@code a
   * * b + c} to {@code float}, round-to-nearest-even, as the pinned wheels' AArch64 {@code fmla}
   * instruction computes it. {@link Math#fma(float, float, float)} is specified to be exactly that
   * (single rounding, same NaN/infinity semantics), so it is used directly.
   */
  private static float fma32(float a, float b, float c) {
    return Math.fma(a, b, c);
  }
}
