package se.alipsa.jmlx.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import se.alipsa.jmlx.core.MLX;
import se.alipsa.jmlx.core.MLXIO;
import se.alipsa.jmlx.ffi.EnabledIfNativeAvailable;

/**
 * PR #79 review finding: the selective {@code MLXIO.loadSafetensors} overload registered a
 * target-scope handle for every matched entry before checking that every requested name exists, so
 * a public caller with a long-lived target scope kept those lazy file-backed handles until the
 * scope closed. The matched handles must be released from the target scope before the missing-name
 * throw.
 */
@EnabledIfNativeAvailable
class MLXIOSelectionScopeTest {

  @Test
  void missingSelectedNameReleasesMatchedHandlesFromTheTargetScope(@TempDir Path dir)
      throws Exception {
    String file = dir.resolve("selected.safetensors").toString();
    try (MLXScope save = new MLXScope()) {
      MLXIO.saveSafetensors(
          file, Map.of("keep", MLX.array(save, new float[] {2, 3}, new int[] {2})), Map.of());
    }
    try (MLXScope target = new MLXScope()) {
      int handles = target.handleCount();
      assertThrows(
          IllegalArgumentException.class,
          () -> MLXIO.loadSafetensors(target, file, Set.of("keep", "missing")));
      // Pre-fix the matched "keep" handle stayed registered in target until the scope closed;
      // this is the regression pin for the finding.
      assertEquals(
          handles, target.handleCount(), "matched handles must be released on a missing name");
    }
  }
}
