package se.alipsa.jmlx.nn;

import org.junit.jupiter.api.Test;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/** Independent default-precision coverage against the fixed CPU references. */
@EnabledIfNativeAvailable
class Phase71DefaultModeTest {
  @Test
  void committedCasesAlsoMatchDefaultPrecision() {
    new Phase71OracleTest().everyCommittedCaseMatches();
  }
}
