package se.alipsa.jmlx.models;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/**
 * Default-precision (TF32) evidence for the WP8 acceptance golden: the same six cases, run through
 * the Java preprocessor, tower and cached decode, are checked against the strict CPU-HF float32
 * reference within the measured TF32 bound. The greedy-ID assertion is skipped here because the
 * reduced-precision logit margin is not pinned; the strict {@code full-float32} suite keeps it. The
 * measured maximum is recorded in {@code req/phase6-golden-precision.md}.
 */
@EnabledIfNativeAvailable
class Qwen3VlDefaultPrecisionTest {
  // Measured at 0.0019590855 on the pinned WP8 checkpoint under the default (TF32) precision
  // mode (bit-reproducible across runs on the pinned native stack); the bound keeps about 28%
  // margin. Recorded in req/phase6-golden-precision.md; kept separate from the strict 1e-4
  // float32 gate.
  private static final float TF32_LOGIT_BOUND = 0.0025f;

  @Test
  void baseAndForcedCasesStayWithinMeasuredTf32Bound() throws Exception {
    float maximum = Qwen3VlGoldenTest.runCases(TF32_LOGIT_BOUND, false);
    System.out.println("Qwen3-VL WP8 default-precision maximum logit/feature error: " + maximum);
  }
}
