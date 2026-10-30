package se.alipsa.jmlx.models;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/**
 * Default-precision (TF32) evidence for the WP6 acceptance golden: the same six cases, run through
 * the Java preprocessor, tower and cached decode, are checked against the strict CPU-HF float32
 * reference within the measured TF32 bound. The greedy-token assertion is skipped here because the
 * reduced-precision logit margin is not pinned; the strict {@code full-float32} suite keeps it. The
 * measured maximum is recorded in {@code req/phase6-golden-precision.md}.
 */
@EnabledIfNativeAvailable
class Phi3VDefaultPrecisionTest {
  // Measured at 9.266138E-4 on the local M5 Max host against the pinned WP6 golden under the
  // default (TF32) precision mode (bit-reproducible across runs on the pinned native stack); the
  // bound keeps about 30% margin. Recorded in req/phase6-golden-precision.md; kept separate from
  // the strict 1e-4 float32 gate.
  private static final float TF32_LOGIT_BOUND = 0.0012f;

  @Test
  void baseAndForcedCasesStayWithinMeasuredTf32Bound() throws Exception {
    float maximum = Phi3VGoldenTest.runCases(TF32_LOGIT_BOUND, false);
    System.out.println("Phi-3.5-Vision WP6 default-precision maximum logit error: " + maximum);
  }
}
